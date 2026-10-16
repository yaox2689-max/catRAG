"""
公众号批量采集 — BFS 深爬 + JS 交互 + 清洗 + 去重验证

用法:
  uv run batch_crawl_wechat.py --profile_url "https://mp.weixin.qq.com/mp/profile_ext?action=home&__biz=xxx" --max 15
  uv run batch_crawl_wechat.py --profile_url "..." --max 20 --headless False
"""
from __future__ import annotations

import argparse
import asyncio
import io
import re
import sys
from datetime import datetime
from typing import Any

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from crawl4ai import *

# 假设 backend 在项目根下
sys.path.insert(0, __file__.rsplit("/", 1)[0] if "/" in __file__ else ".")
from backend.crawler_utils import (
    clean_wechat_markdown,
    validate_crawl_result,
)

ARTICLE_URL_RE = re.compile(r"https://mp\.weixin\.qq\.com/s/[a-zA-Z0-9_-]+")

EXPAND_JS = """
// 点击"查看全部消息"展开文章列表
var btn = document.querySelector('.js_profile_article_area');
if (!btn) btn = document.querySelector('.weui_btn_plain_default');
if (!btn) btn = document.querySelector('[data-action="profile_article_area"]');
if (btn) btn.click();
// 等待列表渲染
await new Promise(r => setTimeout(r, 2000));
// 多次滚动到底部加载更多
for (var i = 0; i < 6; i++) {
    window.scrollTo(0, document.body.scrollHeight);
    await new Promise(r => setTimeout(r, 1200));
}
"""


async def extract_article_urls(
    crawler: AsyncWebCrawler,
    profile_url: str,
    headless: bool = True,
    max_urls: int = 20,
) -> list[str]:
    """第一步：访问公众号列表页，展开文章列表，提取所有文章 URL。"""
    result = await crawler.arun(
        url=profile_url,
        config=CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            magic=True,
            override_navigator=True,
            simulate_user=True,
            scan_full_page=True,
            scroll_delay=1.0,
            js_code=[EXPAND_JS],
            page_timeout=60000,
            verbose=True,
        ),
    )

    if not result.success:
        print(f"❌ 列表页访问失败: {result.error_message}")
        return []

    # 从 markdown/HTML 中提取所有文章 URL
    content = (result.markdown or "") + (result.html or "")
    urls = ARTICLE_URL_RE.findall(content)
    # 去重 + 去空
    unique_urls = list(dict.fromkeys(urls))
    print(f"  从列表页提取到 {len(unique_urls)} 个不重复链接（取前 {max_urls} 篇）")
    return unique_urls[:max_urls]


async def crawl_single_article(
    crawler: AsyncWebCrawler,
    url: str,
) -> dict | None:
    """第二步：爬取单篇公众号文章正文，返回清洗后的结构化数据。"""
    result = await crawler.arun(
        url=url,
        config=CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            magic=True,
            override_navigator=True,
            simulate_user=True,
            wait_for="body",
            word_count_threshold=5,
            page_timeout=30000,
            verbose=False,
        ),
    )

    if not result.success:
        print(f"  ❌ 采集失败: {url[:60]} → {result.error_message}")
        return None

    title = (result.metadata.get("title") or "").strip() if result.metadata else ""
    raw_markdown = result.markdown or ""
    images = result.media.get("images", []) if result.media else []

    # 清洗
    cleaned_text = clean_wechat_markdown(raw_markdown)

    # 验证
    validation = validate_crawl_result(
        title=title,
        content_length=len(cleaned_text),
        images_count=len(images),
    )

    return {
        "url": url,
        "title": title,
        "content": cleaned_text,
        "raw_length": len(raw_markdown),
        "clean_length": len(cleaned_text),
        "images": images,
        "validation": validation,
        "crawled_at": datetime.now().isoformat(),
    }


async def batch_crawl_wechat(
    profile_url: str,
    max_articles: int = 15,
    headless: bool = True,
) -> list[dict]:
    """批量采集公众号文章：展开列表 → 提取链接 → 逐篇爬取 → 清洗验证。

    参数:
        profile_url: 公众号 profile 页 URL
        max_articles: 最多采集篇数
        headless: 是否无头模式（首次需 False 扫码登录）

    返回:
        成功采集的文章列表（已清洗、已验证）
    """
    print("=" * 60)
    print(f"公众号批量采集")
    print(f"  Profile URL: {profile_url}")
    print(f"  目标篇数: {max_articles}")
    print(f"  Headless: {headless}")
    print("=" * 60)

    async with AsyncWebCrawler(
        config=BrowserConfig(
            headless=headless,
            verbose=True,
            browser_type="chromium",
            enable_stealth=True,
            user_agent_mode="random",
            viewport_width=1920,
            viewport_height=1080,
        )
    ) as crawler:
        # 第一步：提取链接
        print("\n📋 第一步：展开列表，提取文章链接...")
        urls = await extract_article_urls(
            crawler, profile_url, headless=headless, max_urls=max_articles,
        )
        if not urls:
            print("❌ 未提取到任何文章链接")
            return []

        # 第二步：逐篇爬取
        print(f"\n📄 第二步：逐篇采集 {len(urls)} 篇文章...")
        articles: list[dict] = []
        success_count = 0
        fail_count = 0

        for i, url in enumerate(urls, 1):
            print(f"  [{i}/{len(urls)}] 正在采集... ", end="")
            article = await crawl_single_article(crawler, url)
            if article:
                valid = article["validation"]
                status = "✅" if valid["passed"] else "⚠️"
                print(f"{status} {article['title'][:40] or '无标题'}  "
                      f"({article['clean_length']} chars)")
                if not valid["passed"]:
                    print(f"      原因: {', '.join(valid['reasons'])}")
                articles.append(article)
                success_count += 1
            else:
                print("❌")
                fail_count += 1

        print(f"\n{'=' * 60}")
        print(f"采集完成：成功 {success_count} 篇，失败 {fail_count} 篇")
        valid_count = sum(1 for a in articles if a["validation"]["passed"])
        print(f"验证通过：{valid_count}/{success_count} 篇")
        print(f"{'=' * 60}")

        return articles


async def main() -> None:
    parser = argparse.ArgumentParser(description="公众号批量采集")
    parser.add_argument("--profile_url", required=True, help="公众号 profile 页 URL")
    parser.add_argument("--max", type=int, default=15, help="最多采集篇数")
    parser.add_argument("--headless", type=str, default="True",
                        choices=["True", "False"], help="是否无头模式")
    args = parser.parse_args()

    articles = await batch_crawl_wechat(
        profile_url=args.profile_url,
        max_articles=args.max,
        headless=args.headless == "True",
    )

    # 打印摘要
    print("\n📊 采集摘要：")
    for a in articles:
        flag = "✅" if a["validation"]["passed"] else "⚠️"
        title_short = a["title"][:50] if a["title"] else "(无标题)"
        print(f"  {flag} {title_short} [{a['clean_length']} chars]")


if __name__ == "__main__":
    asyncio.run(main())
