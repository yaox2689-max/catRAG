"""
RAGAS 端到端评估脚本。

流程：CSV 黄金集 → run_rag_graph 检索 → LLM 基于 context 生成答案 → RAGAS 多指标打分。

指标说明（均为 0~1，越高越好）：
  - faithfulness：答案陈述是否可由检索上下文支撑（抗幻觉）
  - answer_relevancy：答案与问题的语义相关度（需 embedding）
  - context_precision：检索片段与参考答案的精确度（需 ground_truth）
  - context_recall：参考答案要点是否被检索上下文覆盖（需 ground_truth）
  - answer_correctness：答案与参考答案的一致性（需 ground_truth + embedding + LLM，complete 预设）
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from datetime import datetime
from typing import Any

from dotenv import load_dotenv
from datasets import Dataset
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import evaluate
from ragas.metrics._answer_relevance import answer_relevancy
from ragas.metrics._context_precision import context_precision
from ragas.metrics._context_recall import context_recall
from ragas.metrics._faithfulness import faithfulness

try:
    from ragas.metrics._answer_correctness import answer_correctness as _answer_correctness
except ImportError:
    _answer_correctness = None

load_dotenv()
backend_path = os.path.join(os.path.dirname(__file__), "backend")
sys.path.append(backend_path)

from rag_pipeline import run_rag_graph

DEFAULT_GOLD_CSV = os.path.join(os.path.dirname(__file__), "data", "ragas_eval_gold.csv")
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data")

_ANSWER_SYSTEM = (
    "你是法律咨询助手。请仅根据「参考资料」作答；若资料不足以得出结论，请明确说明。"
    "不要编造参考资料中未出现的条文编号或事实；回答使用简洁中文。"
)

# RAGAS 结果列名 → 中文说明（用于终端汇总）
METRIC_DESCRIPTIONS: dict[str, str] = {
    "faithfulness": "忠实度：答案是否被检索上下文支撑",
    "answer_relevancy": "答案相关度：回答与问题的语义匹配",
    "context_precision": "上下文精确度：检索片段是否精准相关",
    "context_recall": "上下文召回：标准答案要点是否被检索覆盖",
    "answer_correctness": "答案正确性：与参考答案的一致性（含语义+事实）",
}

METRIC_PRESETS: dict[str, str] = {
    "faithfulness": "仅忠实度（最快，不需 ground_truth 判分逻辑外的约束）",
    "retrieval": "仅检索：context_precision + context_recall",
    "standard": "faithfulness + context_precision + context_recall（推荐，不需 answer_relevancy 向量）",
    "full": "faithfulness + answer_relevancy + context_precision + context_recall（默认全面评测）",
    "complete": "full + answer_correctness（最全面，需 ground_truth 与 embedding）",
}


def _load_gold_rows(path: str) -> list[dict[str, str]]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"找不到测试集 CSV：{path}")
    rows: list[dict[str, str]] = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "question" not in reader.fieldnames:
            raise ValueError("CSV 必须包含表头，且至少要有 question 列。")
        for raw in reader:
            q = (raw.get("question") or "").strip()
            if not q or q.startswith("#"):
                continue
            row: dict[str, str] = {
                "id": (raw.get("id") or "").strip(),
                "question": q,
                "ground_truth": (raw.get("ground_truth") or "").strip(),
            }
            rows.append(row)
    if not rows:
        raise ValueError(f"CSV 中没有有效数据行：{path}")
    return rows


def _format_docs_fallback(docs: list[dict]) -> str:
    parts: list[str] = []
    for i, doc in enumerate(docs, 1):
        source = doc.get("filename", "Unknown")
        page = doc.get("page_number", "N/A")
        text = doc.get("text", "")
        parts.append(f"[{i}] {source} (Page {page}):\n{text}")
    return "\n\n---\n\n".join(parts) if parts else ""


def _retrieval_context(rag_result: dict) -> str:
    ctx = (rag_result.get("context") or "").strip()
    if ctx:
        return ctx
    docs = rag_result.get("docs") or []
    return _format_docs_fallback(docs) if isinstance(docs, list) else ""


def generate_rag_answer(llm: Any, question: str, context: str) -> str:
    if not context.strip():
        return "（未检索到可用参考资料，无法基于知识库作答。）"
    user = f"用户问题：\n{question}\n\n参考资料：\n{context}\n\n请作答。"
    msg = llm.invoke(
        [SystemMessage(content=_ANSWER_SYSTEM), HumanMessage(content=user)]
    )
    return (getattr(msg, "content", None) or "").strip()


def _resolve_metrics(preset: str) -> tuple[list, bool, bool]:
    """
    返回 (metrics 列表, 是否需要 HuggingFaceEmbeddings, 是否必须有 ground_truth)。
    """
    p = (preset or "full").strip().lower()
    aliases = {
        "f": "faithfulness",
        "no-relevancy": "standard",
        "faithfulness+ctx": "standard",
        "f+ctx": "standard",
        "all": "full",
    }
    p = aliases.get(p, p)

    if p == "faithfulness":
        return [faithfulness], False, False
    if p == "retrieval":
        return [context_precision, context_recall], False, True
    if p == "standard":
        return [faithfulness, context_precision, context_recall], False, True
    if p == "full":
        return [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ], True, True
    if p == "complete":
        metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ]
        if _answer_correctness is not None:
            metrics.append(_answer_correctness)
        else:
            print("⚠️ 当前 ragas 版本无 answer_correctness，complete 将等同 full。")
        return metrics, True, True

    valid = ", ".join(sorted(METRIC_PRESETS))
    raise ValueError(f"未知 --metrics={preset!r}，可选: {valid}")


def _metric_requires_ground_truth(metric: Any) -> bool:
    return metric in (
        context_precision,
        context_recall,
        _answer_correctness,
    )


def _print_preset_help() -> None:
    print("\n📖 指标预设 (--metrics):")
    for name, desc in METRIC_PRESETS.items():
        print(f"  {name:14} {desc}")
    print("\n📖 各指标含义:")
    for key, desc in METRIC_DESCRIPTIONS.items():
        print(f"  {key:22} {desc}")


def _summarize_metrics(df: Any, metric_columns: list[str]) -> None:
    print("\n" + "=" * 60)
    print("📈 RAGAS 指标汇总（均值 / 标准差 / 有效样本数）")
    print("=" * 60)
    for col in metric_columns:
        if col not in df.columns:
            continue
        series = df[col].dropna()
        if len(series) == 0:
            print(f"  {col}: 无有效分数")
            continue
        mean_v = float(series.mean())
        std_v = float(series.std()) if len(series) > 1 else 0.0
        desc = METRIC_DESCRIPTIONS.get(col, "")
        print(f"\n  ⭐ {col}")
        if desc:
            print(f"     {desc}")
        print(f"     均值: {mean_v:.4f}  标准差: {std_v:.4f}  样本数: {len(series)}/{len(df)}")


def _save_results(
    df: Any,
    output_path: str,
    sample_ids: list[str],
    *,
    verbose: bool = True,
) -> str:
    out_df = df.copy()
    if sample_ids and len(sample_ids) == len(out_df):
        out_df.insert(0, "id", sample_ids)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    out_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    abs_path = os.path.abspath(output_path)
    if verbose:
        print(f"\n💾 明细结果已保存: {abs_path}")
    return abs_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAGAS 评估：从 CSV 读题，run_rag_graph 检索 + LLM 生成答案后多指标打分。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  uv run test_ragas_eval.py
  uv run test_ragas_eval.py --metrics full
  uv run test_ragas_eval.py --metrics standard --limit 5
  uv run test_ragas_eval.py --metrics complete --output data/ragas_eval_results.csv

环境变量: ARK_API_KEY, BASE_URL, MODEL 或 RAGAS_EVAL_MODEL；
          RAGAS_ANSWER_MODEL / FAST_MODEL；EMBEDDING_MODEL / EMBEDDING_DEVICE；
          RAGAS_METRICS（默认 full）、RAGAS_EVAL_CSV、RAGAS_EVAL_OUTPUT。
""",
    )
    parser.add_argument(
        "--csv",
        default=os.getenv("RAGAS_EVAL_CSV", DEFAULT_GOLD_CSV),
        help="黄金集 CSV（需 question；context/正确性类指标需 ground_truth）",
    )
    parser.add_argument(
        "--metrics",
        default=os.getenv("RAGAS_METRICS", "full"),
        metavar="PRESET",
        help="预设: faithfulness | retrieval | standard | full | complete（默认 full）",
    )
    parser.add_argument(
        "--output",
        default=os.getenv("RAGAS_EVAL_OUTPUT", ""),
        help="结果 CSV 路径（默认 data/ragas_eval_results_<时间戳>.csv）",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="仅评测前 N 条（0 表示全部，用于快速试跑）",
    )
    parser.add_argument(
        "--list-metrics",
        action="store_true",
        help="打印预设与指标说明后退出",
    )
    args = parser.parse_args()

    if args.list_metrics:
        _print_preset_help()
        return

    try:
        metrics_list, need_embeddings, preset_needs_gt = _resolve_metrics(args.metrics)
    except ValueError as e:
        print(f"❌ {e}")
        _print_preset_help()
        sys.exit(2)

    metric_names = [m.name for m in metrics_list]
    needs_ground_truth = preset_needs_gt or any(
        _metric_requires_ground_truth(m) for m in metrics_list
    )

    api_key = os.getenv("ARK_API_KEY")
    base_url = os.getenv("BASE_URL")
    eval_model = os.getenv("RAGAS_EVAL_MODEL") or os.getenv("MODEL")
    answer_model = (
        os.getenv("RAGAS_ANSWER_MODEL")
        or os.getenv("FAST_MODEL")
        or eval_model
    )
    if not api_key or not base_url or not eval_model:
        print(
            "❌ 请在 .env 中配置 ARK_API_KEY、BASE_URL，并设置 MODEL 或 RAGAS_EVAL_MODEL（RAGAS 判分用 LLM）。"
        )
        sys.exit(1)
    if not answer_model:
        print("❌ 无法确定生成答案用模型：请配置 MODEL / RAGAS_ANSWER_MODEL / FAST_MODEL。")
        sys.exit(1)

    try:
        test_rows = _load_gold_rows(args.csv)
    except (OSError, ValueError) as e:
        print(f"❌ 读取测试集失败: {e}")
        sys.exit(1)

    if args.limit > 0:
        test_rows = test_rows[: args.limit]
        print(f"⚠️ --limit={args.limit}，仅评测前 {len(test_rows)} 条")

    missing_gt = sum(1 for r in test_rows if not r["ground_truth"])
    if needs_ground_truth and missing_gt:
        print(
            f"⚠️ 有 {missing_gt}/{len(test_rows)} 条缺少 ground_truth，"
            "context_precision / context_recall / answer_correctness 将跳过这些行。"
        )

    print(f"📂 已加载 {len(test_rows)} 条黄金集：{os.path.abspath(args.csv)}")
    print(f"📊 评测预设: {args.metrics} → 指标: {metric_names}")
    if need_embeddings:
        print(f"📊 将加载本地 embedding: {os.getenv('EMBEDDING_MODEL', 'BAAI/bge-m3')}")
    print("🚀 运行检索并生成答案（与线上一致：run_rag_graph + LLM 基于 context 作答）...")

    answer_llm = init_chat_model(
        model=answer_model,
        model_provider="openai",
        api_key=api_key,
        base_url=base_url,
        temperature=0,
    )

    data: list[dict[str, Any]] = []
    sample_ids: list[str] = []
    skipped_no_gt = 0

    for row in test_rows:
        question = row["question"]
        ground_truth = row["ground_truth"]
        rid = row["id"] or str(len(sample_ids) + 1)

        if needs_ground_truth and not ground_truth:
            skipped_no_gt += 1
            print(f"  ⊘ id={rid} 跳过（无 ground_truth）: {question[:40]}…")
            continue

        rag_result = run_rag_graph(question)
        if not isinstance(rag_result, dict):
            print(f"⚠️ 结果不是字典，跳过 id={rid!r} {question[:40]}...")
            continue

        docs = rag_result.get("docs", [])
        rag_trace = rag_result.get("rag_trace", {})
        if not rag_trace or not isinstance(rag_trace, dict):
            print(f"⚠️ 无效 rag_trace，跳过 id={rid!r} {question[:40]}...")
            continue

        chunks: list[str] = []
        if isinstance(docs, list):
            for doc in docs:
                if isinstance(doc, dict):
                    t = doc.get("text", "")
                    if t:
                        chunks.append(t)

        context = _retrieval_context(rag_result)
        answer = generate_rag_answer(answer_llm, question, context)

        data.append(
            {
                "question": question,
                "answer": answer,
                "contexts": chunks if chunks else ["No context retrieved"],
                "ground_truth": ground_truth,
            }
        )
        sample_ids.append(rid)
        preview = answer[:80] + ("…" if len(answer) > 80 else "")
        n_ctx = len(chunks)
        print(f"  ✓ id={rid} chunks={n_ctx} 答案预览: {preview}")

    if skipped_no_gt:
        print(f"ℹ️ 因缺少 ground_truth 跳过 {skipped_no_gt} 条")

    if not data:
        print("❌ 未能收集到任何有效数据，请检查 run_rag_graph、CSV 与 ground_truth。")
        sys.exit(1)

    dataset = Dataset.from_list(data)
    print(f"\n📦 已构建 HuggingFace Dataset，共 {len(dataset)} 条")
    print("📊 正在计算 RAGAS 指标（LLM 判分 + 可选 embedding）...")

    ragas_llm = init_chat_model(
        model=eval_model,
        model_provider="openai",
        api_key=api_key,
        base_url=base_url,
        temperature=0,
    )

    ragas_embeddings = None
    if need_embeddings:
        embed_model = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
        embed_device = os.getenv("EMBEDDING_DEVICE", "cpu")
        hf_endpoint = os.getenv("HF_ENDPOINT")
        if hf_endpoint:
            import huggingface_hub

            huggingface_hub.constants.HF_HUB_ENDPOINT = hf_endpoint
        ragas_embeddings = HuggingFaceEmbeddings(
            model_name=embed_model,
            model_kwargs={"device": embed_device},
            encode_kwargs={"normalize_embeddings": True},
        )

    try:
        eval_kw: dict[str, Any] = {
            "dataset": dataset,
            "metrics": metrics_list,
            "llm": ragas_llm,
        }
        if ragas_embeddings is not None:
            eval_kw["embeddings"] = ragas_embeddings
        score = evaluate(**eval_kw)
        df = score.to_pandas()

        display_cols = ["user_input"] if "user_input" in df.columns else []
        if "user_input" not in display_cols and "question" in df.columns:
            display_cols = ["question"]
        display_cols.extend(metric_names)
        existing_display = [c for c in display_cols if c in df.columns]

        print("\n✅ 评估完成！逐条分数（节选）:")
        print(df[existing_display].to_string())

        _summarize_metrics(df, metric_names)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = (args.output or "").strip()
        if not output_path:
            output_path = os.path.join(
                DEFAULT_OUTPUT_DIR,
                f"ragas_eval_results_{timestamp}.csv",
            )
        saved_path = _save_results(df, output_path, sample_ids)
        latest_path = os.path.join(DEFAULT_OUTPUT_DIR, "ragas_eval_results_latest.csv")
        shutil.copy2(saved_path, latest_path)
        print(f"💾 同步最新副本: {os.path.abspath(latest_path)}")

    except Exception as e:
        print(f"❌ 评估失败: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
