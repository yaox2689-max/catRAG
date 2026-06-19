"""语义三层分块：页 → 段落 → 句子（超长句子按 token 固定长度切分）。
改进：L3 层支持基于 embedding 相似度的真正语义切割。
"""
from __future__ import annotations

import math
import os
import re
from functools import lru_cache
from typing import Callable, List, Optional

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


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """计算两个向量的余弦相似度。"""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return dot / (norm_a * norm_b)


def _percentile(values: list[float], p: int) -> float:
    """计算百分位数。"""
    if not values:
        return 0.0
    sorted_v = sorted(values)
    k = (len(sorted_v) - 1) * p / 100
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_v[int(k)]
    return sorted_v[f] * (c - k) + sorted_v[c] * (k - f)


def semantic_split_sentences(
    paragraph_text: str,
    embed_fn: Optional[Callable[[list[str]], list[list[float]]]] = None,
    threshold_percentile: int = 25,
    min_chunk_tokens: int = 30,
) -> List[str]:
    """
    在段落内部用 embedding 相似度做语义切割。

    流程：
    1. 先按句号粗切得到句子列表
    2. 计算相邻句子的余弦相似度
    3. 相似度低于动态阈值（下四分位数）处切一刀
    4. 过短的块向前合并
    5. 超长块仍走 split_sentence_to_token_chunks 兜底

    如果 embed_fn 为 None 或调用失败，退化为 split_sentences。
    """
    if not paragraph_text or not paragraph_text.strip():
        return []

    # 第一步：按句号粗切
    raw_sentences = split_sentences(paragraph_text)
    if len(raw_sentences) <= 1:
        return raw_sentences

    # 无 embed_fn 时退化为结构切割
    if embed_fn is None:
        return raw_sentences

    # 第二步：获取句子 embedding
    try:
        embeddings = embed_fn(raw_sentences)
    except Exception:
        return raw_sentences

    if not embeddings or len(embeddings) != len(raw_sentences):
        return raw_sentences

    # 第三步：计算相邻句子相似度
    similarities = [
        _cosine_similarity(embeddings[i], embeddings[i + 1])
        for i in range(len(embeddings) - 1)
    ]

    # 第四步：动态阈值（下四分位数）
    threshold = _percentile(similarities, threshold_percentile)

    # 第五步：在相似度低于阈值处切一刀
    chunks: List[str] = []
    current_parts: List[str] = [raw_sentences[0]]
    for i, sim in enumerate(similarities):
        if sim < threshold:
            chunks.append("".join(current_parts))
            current_parts = [raw_sentences[i + 1]]
        else:
            current_parts.append(raw_sentences[i + 1])
    if current_parts:
        chunks.append("".join(current_parts))

    # 第六步：合并过短块（向前合并）
    merged: List[str] = []
    for chunk in chunks:
        if merged and count_tokens(chunk) < min_chunk_tokens:
            merged[-1] += chunk
        else:
            merged.append(chunk)
    # 处理最后一个块过短的情况
    if len(merged) > 1 and count_tokens(merged[-1]) < min_chunk_tokens:
        merged[-2] += merged[-1]
        merged.pop()

    # 第七步：超长块兜底（token 切割）
    result: List[str] = []
    for chunk in merged:
        if count_tokens(chunk) > LEVEL_3_MAX_TOKENS:
            result.extend(split_sentence_to_token_chunks(chunk))
        else:
            result.append(chunk)

    return result if result else raw_sentences


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
