"""文本向量化服务 - 支持密集向量和稀疏向量（BM25），词表与 df 持久化 + 增量更新"""
import json
import math
import os
import re
import threading
from collections import Counter
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

_DEFAULT_STATE_PATH = Path(__file__).resolve().parent.parent / "data" / "bm25_state.json"


def _create_dense_embedder():
    """仅在首次构造 EmbeddingService 时导入 sentence-transformers / torch，避免 import embedding 即崩溃。"""
    from langchain_huggingface import HuggingFaceEmbeddings

    model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    device = os.getenv("EMBEDDING_DEVICE", "cpu")
    
    # 设置 Hugging Face 镜像站点（如果环境变量中设置了）
    hf_endpoint = os.getenv("HF_ENDPOINT")
    if hf_endpoint:
        import huggingface_hub
        huggingface_hub.constants.HF_HUB_ENDPOINT = hf_endpoint
    
    try:
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )
    except Exception as e:
        print(f"警告: 无法从 Hugging Face 加载模型 {model_name}: {e}")
        print("尝试使用备用模型或检查网络连接...")
        raise


class EmbeddingService:
    """文本向量化服务 - 密集向量本地模型 + BM25 稀疏向量（持久化统计）"""

    def __init__(self, state_path: Path | str | None = None):
        #调用上面的函数，构建密集向量嵌入模型
        self._embedder = _create_dense_embedder()
        #默认路径：D:\pycharm\SuperMew-main\data\bm25_state.json
        #持久化bm25统计数据，防止每次加载重新统计
        self._state_path = Path(state_path or os.getenv("BM25_STATE_PATH", _DEFAULT_STATE_PATH))
        #保护多线程并发访问时的数据安全
        self._lock = threading.Lock()

        # BM25 参数
        self.k1 = 1.5 #调节文档中关键词出现次数对相关性的影响大小
        self.b = 0.75 #文档长度归一化参数

        self._vocab: dict[str, int] = {}            # 词表：词 -> ID
        self._vocab_counter = 0                     # 词表计数器
        self._doc_freq: Counter[str] = Counter()    # 文档频率：词 -> 出现文档数
        self._total_docs = 0
        self._sum_token_len = 0
        self._avg_doc_len = 1.0
        #从 bm25_state.json 文件恢复之前的统计信息
        #实现增量更新（不会每次重启都清空）
        self._load_state()

    #重新计算文件平均长度
    def _recompute_avg_len(self) -> None:
        self._avg_doc_len = (
            self._sum_token_len / self._total_docs if self._total_docs > 0 else 1.0
        )

    #从bm25_state.json文件中恢复信息
    def _load_state(self) -> None:
        """从持久化文件中恢复统计信息"""
        path = self._state_path
        if not path.is_file():
            return
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        if raw.get("version") != 1:
            return
        #键（词）转为字符串 str(k)
        #值（ID）转为整数 int(v)
        self._vocab = {str(k): int(v) for k, v in raw.get("vocab", {}).items()}
        #从 JSON 加载文档频率统计（词 → 出现文档数）
        self._doc_freq = Counter({str(k): int(v) for k, v in raw.get("doc_freq", {}).items()})
        #加载已处理的文档总数
        self._total_docs = int(raw.get("total_docs", 0))
        self._sum_token_len = int(raw.get("sum_token_len", 0))
        #如果词表不为空：找到词表中最大的 ID：max(self._vocab.values())，计数器设为最大值 +1：+ 1
        #如果词表为空：计数器从 0 开始
        if self._vocab:
            self._vocab_counter = max(self._vocab.values()) + 1
        else:
            self._vocab_counter = 0
        self._recompute_avg_len()

    def _persist_unlocked(self) -> None:
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "total_docs": self._total_docs,
            "sum_token_len": self._sum_token_len,
            "vocab": self._vocab,
            "doc_freq": dict(self._doc_freq),
        }
        #"上传 Word 后，先把更新的统计数据写入临时文件 bm25_state.json.tmp，
        # 确认写入成功后，用临时文件替换原文件 bm25_state.json，确保数据安全。"

        #生成临时文件路径，用于原子写入
        tmp = self._state_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        #用临时文件替换原文件（原子操作)
        tmp.replace(self._state_path)
    #用户 A 上传文档 → 线程 A 获取锁 ✅
    #用户 B 上传文档 → 线程 B 等待... ⏳ (锁被 A 占用)
    def _persist(self) -> None:
        with self._lock:
            self._persist_unlocked()

    def increment_add_documents(self, texts: list[str]) -> None:
        """
        将每个 text 视为 BM25 中的一篇文档（与当前 chunk 写入粒度一致），增量更新 N / df / 长度和。
        """
        if not texts:
            return
        with self._lock:
            for text in texts:
                tokens = self.tokenize(text)
                doc_len = len(tokens)
                self._sum_token_len += doc_len
                self._total_docs += 1
                for token in set(tokens):
                    if token not in self._vocab:
                        self._vocab[token] = self._vocab_counter
                        self._vocab_counter += 1
                    self._doc_freq[token] += 1
            self._recompute_avg_len()
            self._persist_unlocked()

    def increment_remove_documents(self, texts: list[str]) -> None:
        """
        从语料统计中移除与 increment_add_documents 对称的文档集合（如删除某文件的全部 chunk 文本）。
        词表索引不回收，避免与 Milvus 中仍可能存在的旧稀疏向量维度冲突。
        """
        if not texts:
            return
        with self._lock:
            for text in texts:
                tokens = self.tokenize(text)
                doc_len = len(tokens)
                self._sum_token_len = max(0, self._sum_token_len - doc_len)
                self._total_docs = max(0, self._total_docs - 1)
                for token in set(tokens):
                    if token not in self._doc_freq:
                        continue
                    self._doc_freq[token] -= 1
                    if self._doc_freq[token] <= 0:
                        del self._doc_freq[token]
            self._recompute_avg_len()
            self._persist_unlocked()


    #把中文、英文字符计算为token数
    def tokenize(self, text: str) -> list[str]:
        text = text.lower()
        tokens = []
        chinese_pattern = re.compile(r"[\u4e00-\u9fff]")    # 匹配单个中文字符
        english_pattern = re.compile(r"[a-zA-Z]+")          # 匹配连续英文字母
        i = 0
        while i < len(text):
            #如果是中文则直接token+1
            char = text[i]
            if chinese_pattern.match(char):
                tokens.append(char)
                i += 1
            #如果是英文则匹配连续英文如study则token+1
            elif english_pattern.match(char):
                match = english_pattern.match(text[i:])
                if match:
                    tokens.append(match.group())
                    i += len(match.group())
            else:
                i += 1
        return tokens
    #把text转化为稀疏向量
    def _sparse_vector_for_text_unlocked(self, text: str) -> tuple[dict, bool]:
        tokens = self.tokenize(text)
        doc_len = len(tokens)
        tf = Counter(tokens)
        sparse_vector: dict[int, float] = {}
        vocab_changed = False
        n = max(self._total_docs, 0)
        avg = max(self._avg_doc_len, 1.0)

        for token, freq in tf.items():
            #在生成稀疏向量时，如果发现新词，就把它加入词表并分配一个新的唯一 ID。
            if token not in self._vocab:
                #如果_vocab的长度为2，有两个字典在里面分别为0、1，则新token的id为2
                self._vocab[token] = self._vocab_counter
                self._vocab_counter += 1
                vocab_changed = True
            #逆文档频率计算公式，计算token的重要程度
            idx = self._vocab[token]
            #df:很多文档出现则值大，在少数文档出现则值小
            df = self._doc_freq.get(token, 0)
            if df == 0:
                idf = math.log((n + 1) / 1)
            else:
                idf = math.log((n - df + 0.5) / (df + 0.5) + 1)

            numerator = freq * (self.k1 + 1)
            denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / avg)
            score = idf * numerator / denominator
            if score > 0:
                sparse_vector[idx] = float(score)

        return sparse_vector, vocab_changed

    def get_sparse_embedding(self, text: str) -> dict:
        with self._lock:
            sparse_vector, vocab_changed = self._sparse_vector_for_text_unlocked(text)
            if vocab_changed:
                self._persist_unlocked()
        return sparse_vector

    def get_sparse_embeddings(self, texts: list[str]) -> list[dict]:
        if not texts:
            return []
        with self._lock:
            out: list[dict] = []
            any_new_vocab = False
            for text in texts:
                sparse_vector, vocab_changed = self._sparse_vector_for_text_unlocked(text)
                out.append(sparse_vector)
                any_new_vocab = any_new_vocab or vocab_changed
            if any_new_vocab:
                self._persist_unlocked()
        return out

    #生成密集向量
    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            return self._embedder.embed_documents(texts)
        except Exception as e:
            raise Exception(f"本地嵌入模型调用失败: {str(e)}") from e

    def get_all_embeddings(self, texts: list[str]) -> tuple[list[list[float]], list[dict]]:
        dense_embeddings = self.get_embeddings(texts)
        sparse_embeddings = self.get_sparse_embeddings(texts)
        return dense_embeddings, sparse_embeddings


_embedding_singleton: "EmbeddingService | None" = None


def _get_embedding_service() -> "EmbeddingService":
    """首次调用检索/向量相关 API 时再加载 SentenceTransformer，避免导入阶段加载 torch 导致进程静默崩溃。"""
    global _embedding_singleton
    if _embedding_singleton is None:
        print(
            "[喵呜助手] 正在加载本地嵌入模型（首次较慢，请稍候）…",
            flush=True,
        )
        _embedding_singleton = EmbeddingService()
    return _embedding_singleton


class _LazyEmbeddingServiceProxy:
    __slots__ = ()

    def __getattr__(self, name: str) -> Any:
        return getattr(_get_embedding_service(), name)

    def __repr__(self) -> str:
        return "<LazyEmbeddingServiceProxy>"


# 与原先单例语义一致，但推迟到首次属性访问时才构造 EmbeddingService
embedding_service: EmbeddingService = _LazyEmbeddingServiceProxy()  # type: ignore[assignment]
