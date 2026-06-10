"""语义三层分块：页 → 段落 → 句子（超长句子按 token 固定长度切分）。"""
from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import List

LEVEL_3_MAX_TOKENS = int(os.getenv("CHUNK_LEVEL3_MAX_TOKENS", "300"))
LEVEL_3_TOKEN_OVERLAP = int(os.getenv("CHUNK_LEVEL3_TOKEN_OVERLAP", "30"))

# 中英文句子切分（保留标点在前一句末尾）
# 注意：分号 ； 不作为主分割点，因为法律条文的列举项（（一）；（二）；（三））应保留为同一句
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？!?])\s*")
# 段落：空行、缩进换行
_PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n+")


def split_paragraphs(page_text: str) -> List[str]:
    """第二层：按段落语义边界切分。"""
    text = (page_text or "").strip()
    if not text:
        return []

    parts = [p.strip() for p in _PARAGRAPH_SPLIT_RE.split(text) if p.strip()]
    if len(parts) > 1:
        return _merge_short_paragraphs(parts)

    if "\n" in text:
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        if len(lines) > 1:
            return _merge_short_paragraphs(lines)

    return [text]


def split_sentences(paragraph_text: str) -> List[str]:
    """第三层前置：按句子语义边界切分。"""
    text = (paragraph_text or "").strip()
    if not text:
        return []

    parts = [p.strip() for p in _SENTENCE_SPLIT_RE.split(text) if p.strip()]
    if len(parts) <= 1 and len(text) > 400:
        # 无标点长段落：按逗号次级切分（不按分号，保留法律列举完整性）
        parts = [p.strip() for p in re.split(r"(?<=[，,])\s*", text) if p.strip()]
    return parts if parts else [text]


def split_sentence_to_token_chunks(
    sentence: str,
    max_tokens: int = LEVEL_3_MAX_TOKENS,
    overlap: int = LEVEL_3_TOKEN_OVERLAP,
) -> List[str]:
    """句子过长时按固定 token 窗口切分（仅用于 Milvus 叶子层）。"""
    text = (sentence or "").strip()
    if not text:
        return []

    token_ids = _encode(text)
    if len(token_ids) <= max_tokens:
        return [text]

    overlap = max(0, min(overlap, max_tokens // 4))
    chunks: List[str] = []
    start = 0
    while start < len(token_ids):
        end = min(start + max_tokens, len(token_ids))
        piece_ids = token_ids[start:end]
        piece = _decode(piece_ids).strip()
        if piece:
            chunks.append(piece)
        if end >= len(token_ids):
            break
        start = end - overlap if overlap else end
    return chunks if chunks else [text]


def _merge_short_paragraphs(parts: List[str], min_len: int = 40) -> List[str]:
    """合并过短段落，避免碎片过多。"""
    if not parts:
        return []
    merged: List[str] = []
    buffer = ""
    for part in parts:
        if not buffer:
            buffer = part
            continue
        if len(buffer) < min_len:
            buffer = f"{buffer}\n{part}"
        else:
            merged.append(buffer)
            buffer = part
    if buffer:
        merged.append(buffer)
    return merged


@lru_cache(maxsize=1)
def _get_tokenizer():
    model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    except Exception as exc:
        print(f"[semantic_chunker] 无法加载 {model_name} tokenizer，使用字符估算: {exc}", flush=True)
        return None


def _encode(text: str) -> List[int]:
    tokenizer = _get_tokenizer()
    if tokenizer is not None:
        return tokenizer.encode(text, add_special_tokens=False)
    # 无 tokenizer 时按字符估算 token（中文约 1 字 1 token）
    return [ord(c) for c in text]


def _decode(token_ids: List[int]) -> str:
    tokenizer = _get_tokenizer()
    if tokenizer is not None:
        return tokenizer.decode(token_ids, skip_special_tokens=True)
    return "".join(chr(i) for i in token_ids if 0 <= i <= 0x10FFFF)


def count_tokens(text: str) -> int:
    return len(_encode(text or ""))
