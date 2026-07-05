"""
消融实验：Dense vs Sparse vs Hybrid + RRF k 调优 + Rerank 延迟分析

用法：
  uv run test_ablation.py                          # 全部实验
  uv run test_ablation.py --mode retrieval         # 仅检索消融 (Dense/Sparse/Hybrid)
  uv run test_ablation.py --mode k_tuning          # 仅 RRF k 调优
  uv run test_ablation.py --mode rerank_latency    # 仅 Rerank 延迟
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from datetime import datetime
from typing import Any

from dotenv import load_dotenv
from datasets import Dataset

load_dotenv()
backend_path = os.path.join(os.path.dirname(__file__), "backend")
sys.path.append(backend_path)

from rag_utils import retrieve_documents, _rerank_documents, _auto_merge_documents
from embedding import embedding_service
from milvus_client import MilvusManager
from rag_utils import LEAF_RETRIEVE_LEVEL

_milvus_manager = MilvusManager()

DEFAULT_GOLD_CSV = os.path.join(os.path.dirname(__file__), "data", "ragas_eval_gold.csv")


def _load_questions(path: str) -> list[dict]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"找不到测试集 CSV：{path}")
    rows: list[dict] = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            q = (raw.get("question") or "").strip()
            if not q or q.startswith("#"):
                continue
            rows.append({
                "id": (raw.get("id") or "").strip(),
                "question": q,
                "ground_truth": (raw.get("ground_truth") or "").strip(),
            })
    return rows


def _retrieve_dense_only(query: str, top_k: int = 5) -> dict:
    """仅 Dense 检索"""
    candidate_k = max(top_k * 3, top_k)
    filter_expr = f"chunk_level == {LEAF_RETRIEVE_LEVEL}"
    try:
        dense_embedding = embedding_service.get_embeddings([query])[0]
        retrieved = _milvus_manager.dense_retrieve(
            dense_embedding=dense_embedding,
            top_k=candidate_k,
            filter_expr=filter_expr,
        )
        reranked, rerank_meta = _rerank_documents(query=query, docs=retrieved, top_k=top_k)
        merged_docs, merge_meta = _auto_merge_documents(docs=reranked, top_k=top_k)
        return {"docs": merged_docs, "meta": {**rerank_meta, **merge_meta}}
    except Exception as e:
        return {"docs": [], "meta": {"error": str(e)}}


def _retrieve_sparse_only(query: str, top_k: int = 5) -> dict:
    """仅 Sparse 检索"""
    candidate_k = max(top_k * 3, top_k)
    filter_expr = f"chunk_level == {LEAF_RETRIEVE_LEVEL}"
    try:
        sparse_embedding = embedding_service.get_sparse_embedding(query)
        retrieved = _milvus_manager.sparse_retrieve(
            sparse_embedding=sparse_embedding,
            top_k=candidate_k,
            filter_expr=filter_expr,
        )
        reranked, rerank_meta = _rerank_documents(query=query, docs=retrieved, top_k=top_k)
        merged_docs, merge_meta = _auto_merge_documents(docs=reranked, top_k=top_k)
        return {"docs": merged_docs, "meta": {**rerank_meta, **merge_meta}}
    except Exception as e:
        return {"docs": [], "meta": {"error": str(e)}}


def _retrieve_hybrid(query: str, top_k: int = 5) -> dict:
    """双路 Hybrid 检索（线上默认）"""
    return retrieve_documents(query, top_k=top_k)


def _count_retrieved(meta: dict, key: str = "") -> int:
    return meta.get("candidate_count", 0)


def run_retrieval_ablation(rows: list[dict], limit: int = 0) -> None:
    """对比 Dense vs Sparse vs Hybrid 的检索召回数量"""
    if limit > 0:
        rows = rows[:limit]

    print("=" * 60)
    print("检索消融实验：Dense vs Sparse vs Hybrid")
    print("=" * 60)

    results: dict[str, list] = {"dense": [], "sparse": [], "hybrid": []}

    for row in rows:
        q = row["question"]
        rid = row["id"]

        dense_result = _retrieve_dense_only(q)
        sparse_result = _retrieve_sparse_only(q)
        hybrid_result = _retrieve_hybrid(q)

        dense_docs = len(dense_result.get("docs", []))
        sparse_docs = len(sparse_result.get("docs", []))
        hybrid_docs = len(hybrid_result.get("docs", []))

        results["dense"].append(dense_docs)
        results["sparse"].append(sparse_docs)
        results["hybrid"].append(hybrid_docs)

        print(f"  id={rid}  dense={dense_docs}  sparse={sparse_docs}  hybrid={hybrid_docs}")

    print("\n" + "-" * 40)
    print("汇总（平均 retrieval count / top-5 命中条数）：")
    for mode in ["dense", "sparse", "hybrid"]:
        vals = results[mode]
        avg = sum(vals) / len(vals) if vals else 0
        full_hits = sum(1 for v in vals if v >= 5)
        print(f"  {mode:8s}  avg={avg:.2f}  top-5 满额={full_hits}/{len(vals)}")
    print("-" * 40)

    # 重叠分析：docs 的 chunk_id 交集
    print("\n重叠分析（前 20 条）：")
    overlap_counts = {"dense_only": 0, "sparse_only": 0, "both": 0}
    for row in rows[:20]:
        q = row["question"]
        d = _retrieve_dense_only(q).get("docs", [])
        s = _retrieve_sparse_only(q).get("docs", [])
        d_ids = {doc.get("chunk_id") for doc in d if doc.get("chunk_id")}
        s_ids = {doc.get("chunk_id") for doc in s if doc.get("chunk_id")}
        both = d_ids & s_ids
        if both:
            overlap_counts["both"] += 1
        elif len(d_ids) > len(s_ids):
            overlap_counts["dense_only"] += 1
        else:
            overlap_counts["sparse_only"] += 1
    print(f"  两路有交集: {overlap_counts['both']}/20")
    print(f"  Dense 单路命中: {overlap_counts['dense_only']}/20")
    print(f"  Sparse 单路命中: {overlap_counts['sparse_only']}/20")


def _retrieve_with_rrf_k(query: str, k: int, top_k: int = 5) -> list:
    """使用指定 k 值做 RRF 融合检索"""
    from pymilvus import AnnSearchRequest, RRFRanker

    candidate_k = max(top_k * 3, top_k)
    filter_expr = f"chunk_level == {LEAF_RETRIEVE_LEVEL}"
    try:
        dense_embedding = embedding_service.get_embeddings([query])[0]
        sparse_embedding = embedding_service.get_sparse_embedding(query)
        dense_search = AnnSearchRequest(
            data=[dense_embedding],
            anns_field="dense_embedding",
            param={"metric_type": "IP", "params": {"ef": 64}},
            limit=candidate_k,
            expr=filter_expr,
        )
        sparse_search = AnnSearchRequest(
            data=[sparse_embedding],
            anns_field="sparse_embedding",
            param={"metric_type": "IP", "params": {"drop_ratio_search": 0.2}},
            limit=candidate_k,
            expr=filter_expr,
        )
        results = _milvus_manager._run_with_reconnect(
            lambda client: client.hybrid_search(
                collection_name=_milvus_manager.collection_name,
                reqs=[dense_search, sparse_search],
                ranker=RRFRanker(k=k),
                limit=candidate_k,
                output_fields=["text", "filename", "chunk_id"],
            )
        )
        formatted = []
        for hits in results:
            for hit in hits:
                formatted.append({
                    "chunk_id": hit.get("chunk_id", ""),
                    "score": hit.get("distance", 0.0),
                })
        return formatted
    except Exception:
        return []


def run_rrf_k_tuning(rows: list[dict], limit: int = 30) -> None:
    """测试 k=30/60/100 哪个融合效果最好"""
    if limit > 0:
        rows = rows[:limit]

    print("\n" + "=" * 60)
    print("RRF k 值调优: k=30 vs k=60 vs k=100")
    print("=" * 60)

    k_values = [30, 60, 100]
    results: dict[int, list[int]] = {k: [] for k in k_values}

    for row in rows:
        q = row["question"]
        rid = row["id"]
        for k in k_values:
            docs = _retrieve_with_rrf_k(q, k=k)
            results[k].append(len(docs))
        print(f"  id={rid}  k=30 -> {results[30][-1]}  | k=60 -> {results[60][-1]}  | k=100 -> {results[100][-1]}")

    print("\n汇总（平均召回条数）：")
    for k in k_values:
        avg = sum(results[k]) / len(results[k]) if results[k] else 0
        print(f"  k={k:3d}  avg recall={avg:.1f}")


def run_rerank_latency(rows: list[dict], limit: int = 20) -> None:
    """对比开/关 Rerank 的响应时间"""
    if limit > 0:
        rows = rows[:limit]

    print("\n" + "=" * 60)
    print("Rerank 延迟分析：Rerank ON vs Rerank OFF")
    print("=" * 60)

    import rag_utils as ru

    # 临时关闭 Rerank
    original_rerank_model = os.environ.get("RERANK_MODEL", "")
    original_rerank_key = os.environ.get("RERANK_API_KEY", "")
    original_rerank_host = os.environ.get("RERANK_BINDING_HOST", "")

    # 热加载：Rerank ON
    latencies_on = []
    latencies_off = []

    for row in rows:
        q = row["question"]
        rid = row["id"]

        # Rerank ON
        start = time.time()
        ru.retrieve_documents(q, top_k=5)
        elapsed_on = time.time() - start
        latencies_on.append(elapsed_on)

        # Rerank OFF：临时清空环境变量，重试
        if original_rerank_model:
            os.environ["RERANK_MODEL"] = ""
            os.environ["RERANK_API_KEY"] = ""
            os.environ["RERANK_BINDING_HOST"] = ""
            # 重新加载模块中的常量（hack: 因为模块级变量在 import 时已缓存）
            ru.RERANK_MODEL = ""
            ru.RERANK_API_KEY = ""
            ru.RERANK_BINDING_HOST = ""

        start = time.time()
        ru.retrieve_documents(q, top_k=5)
        elapsed_off = time.time() - start
        latencies_off.append(elapsed_off)

        # 恢复环境变量
        if original_rerank_model:
            os.environ["RERANK_MODEL"] = original_rerank_model
            os.environ["RERANK_API_KEY"] = original_rerank_key
            os.environ["RERANK_BINDING_HOST"] = original_rerank_host
            ru.RERANK_MODEL = original_rerank_model
            ru.RERANK_API_KEY = original_rerank_key
            ru.RERANK_BINDING_HOST = original_rerank_host

        print(f"  id={rid}  ON={elapsed_on:.3f}s  OFF={elapsed_off:.3f}s  diff={elapsed_on - elapsed_off:+.3f}s")

    avg_on = sum(latencies_on) / len(latencies_on) if latencies_on else 0
    avg_off = sum(latencies_off) / len(latencies_off) if latencies_off else 0
    print(f"\n  平均 Rerank ON:  {avg_on:.3f}s")
    print(f"  平均 Rerank OFF: {avg_off:.3f}s")
    print(f"  平均增量:        {avg_on - avg_off:+.3f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG 消融实验套件")
    parser.add_argument("--mode", default="all", choices=["all", "retrieval", "k_tuning", "rerank_latency"],
                        help="实验模式")
    parser.add_argument("--csv", default=DEFAULT_GOLD_CSV, help="测试集 CSV")
    parser.add_argument("--limit", type=int, default=0, help="限制测试条数")
    args = parser.parse_args()

    rows = _load_questions(args.csv)
    if args.limit > 0:
        rows = rows[:args.limit]
    print(f"已加载 {len(rows)} 条测试 query")

    if args.mode in ("all", "retrieval"):
        run_retrieval_ablation(rows, limit=args.limit if args.mode != "all" else 0)

    if args.mode in ("all", "k_tuning"):
        run_rrf_k_tuning(rows, limit=min(args.limit or 30, 30))

    if args.mode in ("all", "rerank_latency"):
        run_rerank_latency(rows, limit=min(args.limit or 20, 20))


if __name__ == "__main__":
    main()
