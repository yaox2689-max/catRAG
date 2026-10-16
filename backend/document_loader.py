"""文档加载和语义三层分块服务。

层级说明：
- L1 整页内容 → PostgreSQL（parent_chunks）
- L2 段落     → PostgreSQL
- L3 句子（超长按 300 token 固定切分）→ Milvus
"""
import os
from typing import Dict, List

from semantic_chunker import (
    LEVEL_3_MAX_TOKENS,
    semantic_split_sentences,
    split_paragraphs,
    split_sentence_to_token_chunks,
    split_sentences,
)


class DocumentLoader:
    """文档加载与语义三层分块。"""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50, embed_fn=None):
        # 保留参数以兼容旧调用；L3 长度由 CHUNK_LEVEL3_MAX_TOKENS 控制
        self._legacy_chunk_size = chunk_size
        self._legacy_chunk_overlap = chunk_overlap
        # 语义切割用的 embedding 函数，由调用方注入（如 embedding_service.get_embeddings）
        self._embed_fn = embed_fn

    @staticmethod
    def _build_chunk_id(filename: str, page_number: int, level: int, index: int) -> str:
        return f"{filename}::p{page_number}::l{level}::{index}"

    def _split_page_to_three_levels(
        self,
        text: str,
        base_doc: Dict,
        page_global_chunk_idx: int,
    ) -> List[Dict]:
        page_text = (text or "").strip()
        if not page_text:
            return []

        page_number = int(base_doc.get("page_number", 0))
        filename = base_doc["filename"]
        root_chunks: List[Dict] = []

        # L1：整页（语义单元 = 文档加载器给出的单页正文）
        level_1_id = self._build_chunk_id(filename, page_number, 1, 0)
        level_1_chunk = {
            **base_doc,
            "text": page_text,
            "chunk_id": level_1_id,
            "parent_chunk_id": "",
            "root_chunk_id": level_1_id,
            "chunk_level": 1,
            "chunk_idx": page_global_chunk_idx,
        }
        root_chunks.append(level_1_chunk)
        page_global_chunk_idx += 1

        paragraphs = split_paragraphs(page_text)
        if not paragraphs:
            paragraphs = [page_text]

        level_3_counter = 0
        for para_idx, para_text in enumerate(paragraphs):
            para_text = para_text.strip()
            if not para_text:
                continue

            level_2_id = self._build_chunk_id(filename, page_number, 2, para_idx)
            root_chunks.append({
                **base_doc,
                "text": para_text,
                "chunk_id": level_2_id,
                "parent_chunk_id": level_1_id,
                "root_chunk_id": level_1_id,
                "chunk_level": 2,
                "chunk_idx": page_global_chunk_idx,
            })
            page_global_chunk_idx += 1

            sentences = semantic_split_sentences(para_text, embed_fn=self._embed_fn)
            for sent_text in sentences:
                leaf_parts = split_sentence_to_token_chunks(sent_text, max_tokens=LEVEL_3_MAX_TOKENS)
                for leaf_text in leaf_parts:
                    leaf_text = leaf_text.strip()
                    if not leaf_text:
                        continue
                    level_3_id = self._build_chunk_id(filename, page_number, 3, level_3_counter)
                    level_3_counter += 1
                    root_chunks.append({
                        **base_doc,
                        "text": leaf_text,
                        "chunk_id": level_3_id,
                        "parent_chunk_id": level_2_id,
                        "root_chunk_id": level_1_id,
                        "chunk_level": 3,
                        "chunk_idx": page_global_chunk_idx,
                    })
                    page_global_chunk_idx += 1

        return root_chunks

    def load_document(self, file_path: str, filename: str) -> list[dict]:
        """
        加载单个文档并分片
        :param file_path: 文件路径
        :param filename: 文件名
        :return: 分片后的文档列表（含 L1/L2/L3）
        """
        file_lower = filename.lower()

        from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, UnstructuredExcelLoader

        if file_lower.endswith(".pdf"):
            doc_type = "PDF"
            loader = PyPDFLoader(file_path)
        elif file_lower.endswith((".docx", ".doc")):
            doc_type = "Word"
            loader = Docx2txtLoader(file_path)
        elif file_lower.endswith((".xlsx", ".xls")):
            doc_type = "Excel"
            loader = UnstructuredExcelLoader(file_path)
        else:
            raise ValueError(f"不支持的文件类型: {filename}")

        try:
            raw_docs = loader.load()
            documents = []
            page_global_chunk_idx = 0
            for doc in raw_docs:
                base_doc = {
                    "filename": filename,
                    "file_path": file_path,
                    "file_type": doc_type,
                    "page_number": doc.metadata.get("page", 0),
                }
                page_chunks = self._split_page_to_three_levels(
                    text=(doc.page_content or "").strip(),
                    base_doc=base_doc,
                    page_global_chunk_idx=page_global_chunk_idx,
                )
                page_global_chunk_idx += len(page_chunks)
                documents.extend(page_chunks)
            return documents
        except Exception as e:
            raise Exception(f"处理文档失败: {str(e)}")

    def load_documents_from_folder(self, folder_path: str) -> list[dict]:
        """从文件夹加载所有文档并分片。"""
        all_documents = []

        for filename in os.listdir(folder_path):
            file_lower = filename.lower()
            if not (
                file_lower.endswith(".pdf")
                or file_lower.endswith((".docx", ".doc"))
                or file_lower.endswith((".xlsx", ".xls"))
            ):
                continue

            file_path = os.path.join(folder_path, filename)
            try:
                documents = self.load_document(file_path, filename)
                all_documents.extend(documents)
            except Exception:
                continue

        return all_documents
