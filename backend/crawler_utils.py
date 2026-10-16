"""
爬虫工具集：清洗、去重、验证

用法：
  from crawler_utils import clean_wechat_markdown, is_duplicate_entry, validate_crawl_result
"""
from __future__ import annotations

import re
from typing import Any


# ====== 1. 公众号 markdown 清洗 ======

NOISE_PATTERNS = [
    r"阅读\s*\d+",
    r"在看\d*",
    r"赞\d*",
    r"喜欢此内容的人还喜欢",
    r"不喜欢",
    r"写下你的留言",
    r"免责声明\s*[:：].*",
    r"^\s*广告\s*$",
    r"素材来源.*",
]


def clean_wechat_markdown(text: str, min_line_len: int = 4) -> str:
    """清洗公众号 markdown：去掉页脚噪声、过短行。"""
    if not text or not text.strip():
        return ""

    lines = text.split("\n")
    cleaned: list[str] = []
    for line in lines:
        stripped = line.strip()
        if any(re.search(p, stripped) for p in NOISE_PATTERNS):
            continue
        if len(stripped) < min_line_len:
            continue
        cleaned.append(stripped)
    return "\n".join(cleaned)


# ====== 2. 小红书笔记清洗 ======

XHS_NOISE_COMMENTS = re.compile(
    r"^(已收藏|收藏了|写得好|真棒|不错|学习了|谢谢分享|赞|点赞|转发了|关注了|沙发|板凳|前排)$"
)


def clean_xiaohongshu_note(note: dict, max_comments: int = 10) -> dict | None:
    """清洗单篇小红书笔记。正文过短则返回 None。"""
    title = (note.get("title") or "").strip()
    content = (note.get("content") or "").strip()
    comments = note.get("comments") or []

    if len(content) < 20:
        return None

    meaningful: list[str] = []
    for c in comments:
        text = str(c).strip()
        if not text:
            continue
        if XHS_NOISE_COMMENTS.match(text):
            continue
        if len(text) < 2:
            continue
        meaningful.append(text)

    if len(title) > 120:
        title = title[:120]

    return {
        "title": title,
        "content": content,
        "comments": meaningful[:max_comments],
    }


# ====== 3. 入库前去重 ======


def is_duplicate_entry(title: str, milvus_manager: Any) -> bool:
    """按标题精确匹配检查是否已入库。"""
    if not title:
        return False
    escaped = title.replace("\\", "\\\\").replace('"', '\\"')
    expr = f'title == "{escaped}"'
    try:
        results = milvus_manager.query(
            filter_expr=expr,
            output_fields=["filename"],
            limit=1,
        )
        return len(results) > 0
    except Exception:
        return False


# ====== 4. 爬取结果验证标准 ======


def validate_crawl_result(
    title: str,
    content_length: int,
    images_count: int = 0,
) -> dict:
    """验证单次爬取结果是否达到入库标准。"""
    reasons: list[str] = []

    if not title:
        reasons.append("标题提取失败")
    if content_length < 500:
        reasons.append(f"正文过短: {content_length} < 500")

    return {
        "passed": len(reasons) == 0,
        "reasons": reasons,
        "meta": {
            "title": title,
            "content_length": content_length,
            "images_count": images_count,
        },
    }
