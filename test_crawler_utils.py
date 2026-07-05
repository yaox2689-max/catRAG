"""
Test crawler utilities.
Usage:
  uv run python test_crawler_utils.py
"""
from __future__ import annotations

import sys
sys.path.insert(0, "backend")

from crawler_utils import (
    clean_wechat_markdown,
    clean_xiaohongshu_note,
    validate_crawl_result,
)

PASS = 0
FAIL = 0


def check(name: str, condition: bool) -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}")


def test_clean_wechat_markdown() -> None:
    print("\n=== WeChat markdown cleaning ===")

    dirty = "正文内容\n阅读 10086\n喜欢此内容的人还喜欢\n继续正文"
    clean = clean_wechat_markdown(dirty)
    check("remove view count", "阅读" not in clean)
    check("remove recommendation", "喜欢此内容" not in clean)
    check("keep main text", "正文内容" in clean and "继续正文" in clean)

    dirty2 = "正文内容\n\n赞\n好\n继续"
    clean2 = clean_wechat_markdown(dirty2)
    check("remove short lines", "赞" not in clean2 and "好" not in clean2)

    check("empty text returns empty", clean_wechat_markdown("") == "")


def test_clean_xiaohongshu() -> None:
    print("\n=== Xiaohongshu note cleaning ===")

    note = {
        "title": "Audi A6L review",
        "content": "Three months in, sharing my real experience...",
        "comments": ["写得好", "How's the fuel economy?", "赞", "Great car"],
    }
    clean = clean_xiaohongshu_note(note)
    check("valid note kept", clean is not None)
    check("filter noise comment", clean is not None and "写得好" not in clean["comments"])
    check("keep meaningful comment", clean is not None and "How's the fuel economy?" in clean["comments"])

    short_note = {"title": "test", "content": "short", "comments": []}
    check("short content dropped", clean_xiaohongshu_note(short_note) is None)


def test_validate_crawl() -> None:
    print("\n=== Crawl validation ===")

    v1 = validate_crawl_result(title="Title", content_length=1000)
    check("valid passes", v1["passed"])

    v2 = validate_crawl_result(title="", content_length=1000)
    check("no title fails", not v2["passed"])

    v3 = validate_crawl_result(title="Title", content_length=50)
    check("short content fails", not v3["passed"])

    v4 = validate_crawl_result(title="", content_length=10)
    check("double failure, 2 reasons", len(v4["reasons"]) == 2)


def main() -> None:
    global PASS, FAIL
    PASS, FAIL = 0, 0

    test_clean_wechat_markdown()
    test_clean_xiaohongshu()
    test_validate_crawl()

    print(f"\n{'=' * 30}")
    print(f"Result: {PASS} passed, {FAIL} failed")
    sys.exit(0 if FAIL == 0 else 1)


if __name__ == "__main__":
    main()
