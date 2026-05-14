"""
RAGAS 评估脚本。

Faithfulness（忠实度）：衡量「模型回答」中的陈述能否从「检索到的 contexts」中推断出来，
与 CSV 里的 ground_truth 无直接关系；判分依赖 RAGAS 使用的 LLM（RAGAS_EVAL_MODEL / MODEL）。

其它指标：answer_relevancy 需要本地/云端 embeddings；context_precision / context_recall 需要 ground_truth。
"""
import argparse
import csv
import os
import sys
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

load_dotenv()
backend_path = os.path.join(os.path.dirname(__file__), "backend")
sys.path.append(backend_path)

from rag_pipeline import run_rag_graph

DEFAULT_GOLD_CSV = os.path.join(os.path.dirname(__file__), "data", "ragas_eval_gold.csv")

_ANSWER_SYSTEM = (
    "你是法律咨询助手。请仅根据「参考资料」作答；若资料不足以得出结论，请明确说明。"
    "不要编造参考资料中未出现的条文编号或事实；回答使用简洁中文。"
)


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


def _resolve_metrics(preset: str) -> tuple[list, bool]:
    """
    返回 (metrics 列表, 是否需要 HuggingFaceEmbeddings)。
    legacy evaluate() 仅接受 ragas.metrics.base.Metric 实例。
    """
    p = (preset or "full").strip().lower()
    if p in ("faithfulness", "f"):
        return [faithfulness], False
    if p in ("no-relevancy", "faithfulness+ctx", "f+ctx"):
        return [faithfulness, context_precision, context_recall], False
    if p in ("full", "all"):
        return [faithfulness, answer_relevancy, context_precision, context_recall], True
    raise ValueError(
        f"未知 --metrics={preset!r}，请使用: faithfulness | no-relevancy | full"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAGAS 评估：从 CSV 读题，run_rag_graph 检索 + LLM 生成答案后打分。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
指标预设 (--metrics):
  faithfulness   仅忠实度（答案是否被 contexts 支撑）；只需判分 LLM，无需本地向量模型。
  no-relevancy   faithfulness + context_precision + context_recall；不需 answer_relevancy 的向量。
  full           上述全部 + answer_relevancy（需 EMBEDDING_MODEL 本地加载）。

环境变量: ARK_API_KEY, BASE_URL, MODEL 或 RAGAS_EVAL_MODEL；可选 RAGAS_ANSWER_MODEL / FAST_MODEL。
""",
    )
    parser.add_argument(
        "--csv",
        default=os.getenv("RAGAS_EVAL_CSV", DEFAULT_GOLD_CSV),
        help="黄金集 CSV（需 question；ground_truth 在 full/no-relevancy 下用于 context 指标）",
    )
    parser.add_argument(
        "--metrics",
        default=os.getenv("RAGAS_METRICS", "faithfulness"),
        metavar="PRESET",
        help="faithfulness | no-relevancy | full（默认 faithfulness；可用环境变量 RAGAS_METRICS）",
    )
    args = parser.parse_args()

    try:
        metrics_list, need_embeddings = _resolve_metrics(args.metrics)
    except ValueError as e:
        print(f"❌ {e}")
        sys.exit(2)

    needs_ground_truth = any(
        m is context_precision or m is context_recall for m in metrics_list
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

    print(f"📂 已加载 {len(test_rows)} 条黄金集：{os.path.abspath(args.csv)}")
    print("🚀 运行检索并生成答案（与线上一致：run_rag_graph + LLM 基于 context 作答）...")

    answer_llm = init_chat_model(
        model=answer_model,
        model_provider="openai",
        api_key=api_key,
        base_url=base_url,
        temperature=0,
    )

    data: list[dict[str, Any]] = []
    for row in test_rows:
        question = row["question"]
        ground_truth = row["ground_truth"]
        rid = row["id"]

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

        if needs_ground_truth and not ground_truth:
            print(f"⚠️ 当前指标需要 ground_truth，已跳过：id={rid!r}")
            continue

        data.append(
            {
                "question": question,
                "answer": answer,
                "contexts": chunks if chunks else ["No context retrieved"],
                # context_precision/recall 映射为 reference；faithfulness-only 时可为空字符串
                "ground_truth": ground_truth if ground_truth else "",
            }
        )
        preview = answer[:80] + ("…" if len(answer) > 80 else "")
        print(f"  ✓ id={rid or '-'} 答案预览: {preview}")

    if not data:
        print("❌ 未能收集到任何有效数据，请检查 run_rag_graph 与 CSV。")
        sys.exit(1)

    dataset = Dataset.from_list(data)
    print(f"📦 已构建 HuggingFace Dataset，共 {len(dataset)} 条")
    print(f"📊 RAGAS 指标: {[m.name for m in metrics_list]}")

    print("📊 正在计算 RAGAS 指标...")
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
        print("\n✅ 评估完成！结果如下:")
        df = score.to_pandas()
        print(df)
        if "faithfulness" in df.columns:
            s = df["faithfulness"]
            mean_f = float(s.mean()) if len(s) else float("nan")
            print(f"\n⭐ Faithfulness 均值: {mean_f:.4f}（1 表示回答陈述均可被上下文支持；越低表示幻觉/脱离上下文越多）")
    except Exception as e:
        print(f"❌ 评估失败: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
