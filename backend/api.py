import json
import os
import re
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

print("[api] 加载 crud …", flush=True)
from crud import chat_with_agent, chat_with_agent_stream, storage

print("[api] 加载 auth …", flush=True)
from auth import authenticate_user, create_access_token, get_current_user, get_db, get_password_hash, require_admin, resolve_role

print("[api] 加载 document_loader …", flush=True)
from document_loader import DocumentLoader

print("[api] 加载 embedding …", flush=True)
from embedding import embedding_service

print("[api] 加载 milvus …", flush=True)
from milvus_client import MilvusManager
from milvus_writer import MilvusWriter

print("[api] 加载 sqlbase / parent_chunk_store …", flush=True)
from sqlbase import User
from parent_chunk_store import ParentChunkStore

print("[api] 加载 schemas …", flush=True)
from schemas import (
    AuthResponse,
    ChatRequest,
    ChatResponse,
    CurrentUserResponse,
    DocumentDeleteJobResponse,
    DocumentDeleteResponse,
    DocumentDeleteStartResponse,
    DocumentInfo,
    DocumentListResponse,
    DocumentUploadJobResponse,
    DocumentUploadResponse,
    DocumentUploadStartResponse,
    LoginRequest,
    OCRUploadResponse,
    MessageInfo,
    RegisterRequest,
    SessionDeleteResponse,
    SessionInfo,
    SessionListResponse,
    SessionMessagesResponse,
)
print("[api] 加载 upload_jobs …", flush=True)
from upload_jobs import DELETE_STEPS, delete_job_manager, upload_job_manager

print("[api] 加载 OCR 服务 …", flush=True)
from ocr_service import is_supported_image, recognize_image_bytes

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"
UPLOAD_DIR = DATA_DIR / "documents"

print("[api] 初始化 Milvus / 上传任务依赖 …", flush=True)
parent_chunk_store = ParentChunkStore()
milvus_manager = MilvusManager()
milvus_writer = MilvusWriter(embedding_service=embedding_service, milvus_manager=milvus_manager)

_loader: DocumentLoader | None = None


def _get_document_loader() -> DocumentLoader:
    """首次上传/解析文档时再构造 DocumentLoader（会加载 text_splitters）。"""
    global _loader
    if _loader is None:
        print("[api] 首次文档处理，正在初始化 DocumentLoader …", flush=True)
        _loader = DocumentLoader()
    return _loader

print("[api] 模块加载完成。\n", flush=True)
router = APIRouter()


def _remove_bm25_stats_for_filename(filename: str) -> None:
    """删除 Milvus 中该文件对应 chunk 前，先从持久化 BM25 统计中扣减。"""
    rows = milvus_manager.query_all(
        filter_expr=f'filename == "{filename}"',
        output_fields=["text"],
    )
    texts = [r.get("text") or "" for r in rows]
    embedding_service.increment_remove_documents(texts)


def _delete_local_uploaded_file(filename: str) -> None:
    """删除 data/documents 下对应的本地文件，包含 OCR 子目录。"""
    candidates = [UPLOAD_DIR / filename, UPLOAD_DIR / "ocr" / filename]
    for path in candidates:
        try:
            if path.exists() and path.is_file():
                path.unlink()
        except Exception:
            pass


def _delete_mysql_document_data(filename: str) -> int:
    """删除 MySQL/PostgreSQL 中与文档相关的父级分块数据。"""
    return parent_chunk_store.delete_by_filename(filename)


@router.post("/auth/register", response_model=AuthResponse)
async def register(request: RegisterRequest, db: Session = Depends(get_db)):
    username = (request.username or "").strip()
    password = (request.password or "").strip()
    if not username or not password:
        raise HTTPException(status_code=400, detail="用户名和密码不能为空")

    exists = db.query(User).filter(User.username == username).first()
    if exists:
        raise HTTPException(status_code=409, detail="用户名已存在")

    role = resolve_role(request.role, request.admin_code)
    user = User(username=username, password_hash=get_password_hash(password), role=role)
    db.add(user)
    db.commit()

    token = create_access_token(username=username, role=role)
    return AuthResponse(access_token=token, username=username, role=role)


@router.post("/auth/login", response_model=AuthResponse)
async def login(request: LoginRequest, db: Session = Depends(get_db)):
    user = authenticate_user(db, request.username, request.password)
    if not user:
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    token = create_access_token(username=user.username, role=user.role)
    return AuthResponse(access_token=token, username=user.username, role=user.role)


@router.get("/auth/me", response_model=CurrentUserResponse)
async def me(current_user: User = Depends(get_current_user)):
    return CurrentUserResponse(username=current_user.username, role=current_user.role)


@router.get("/sessions/{session_id}", response_model=SessionMessagesResponse)
async def get_session_messages(session_id: str, current_user: User = Depends(get_current_user)):
    """获取指定会话的所有消息"""
    try:
        messages = [
            MessageInfo(
                type=msg["type"],
                content=msg["content"],
                timestamp=msg["timestamp"],
                rag_trace=msg.get("rag_trace"),
            )
            for msg in storage.get_session_messages(current_user.username, session_id)
        ]
        return SessionMessagesResponse(messages=messages)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions", response_model=SessionListResponse)
async def list_sessions(current_user: User = Depends(get_current_user)):
    """获取当前用户的所有会话列表"""
    try:
        sessions = [SessionInfo(**item) for item in storage.list_session_infos(current_user.username)]
        sessions.sort(key=lambda x: x.updated_at, reverse=True)
        return SessionListResponse(sessions=sessions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/sessions/{session_id}", response_model=SessionDeleteResponse)
async def delete_session(session_id: str, current_user: User = Depends(get_current_user)):
    """删除当前用户的指定会话"""
    try:
        deleted = storage.delete_session(current_user.username, session_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="会话不存在")
        return SessionDeleteResponse(session_id=session_id, message="成功删除会话")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, current_user: User = Depends(get_current_user)):
    try:
        session_id = request.session_id or "default_session"
        resp = chat_with_agent(
            request.message,
            current_user.username,
            session_id,
            image_context=request.image_context or "",
            file_name=request.file_name or "",
            file_content_b64=request.file_content_b64 or "",
        )
        if isinstance(resp, dict):
            return ChatResponse(**resp)
        return ChatResponse(response=resp)
    except Exception as e:
        message = str(e)
        match = re.search(r"Error code:\s*(\d{3})", message)
        if match:
            code = int(match.group(1))
            if code == 429:
                raise HTTPException(
                    status_code=429,
                    detail=(
                        "上游模型服务触发限流/额度限制（429）。请检查账号额度/模型状态。\n"
                        f"原始错误：{message}"
                    ),
                )
            if code in (401, 403):
                raise HTTPException(status_code=code, detail=message)
            raise HTTPException(status_code=code, detail=message)
        raise HTTPException(status_code=500, detail=message)


@router.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest, current_user: User = Depends(get_current_user)):
    """跟 Agent 对话 (流式)"""

    async def event_generator():
        try:
            session_id = request.session_id or "default_session"
            async for chunk in chat_with_agent_stream(
                request.message,
                current_user.username,
                session_id,
                image_context=request.image_context or "",
                file_name=request.file_name or "",
                file_content_b64=request.file_content_b64 or "",
            ):
                yield chunk
        except Exception as e:
            error_data = {"type": "error", "content": str(e)}
            yield f"data: {json.dumps(error_data)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _is_supported_document(filename: str) -> bool:
    file_lower = filename.lower()
    return (
        file_lower.endswith(".pdf")
        or file_lower.endswith((".docx", ".doc"))
        or file_lower.endswith((".xlsx", ".xls"))
    )


def _is_supported_image(filename: str) -> bool:
    file_lower = filename.lower()
    return file_lower.endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"))


async def _save_upload_file(file: UploadFile, file_path: Path) -> None:
    """按块写入上传文件，避免大文件一次性读入内存。"""
    with open(file_path, "wb") as f:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)


def _extract_image_ocr_text(file_path: str, filename: str) -> str:
    """OCR 占位接口：在这里接入你的 OCR 引擎。"""
    # TODO: 在这里接入 OCR（例如 PaddleOCR / EasyOCR / Tesseract / 云 OCR 服务）
    # 返回识别到的纯文本；识别失败时返回空字符串或抛出异常均可。
    raise NotImplementedError("OCR 尚未接入，请在 _extract_image_ocr_text 中实现")


def _process_upload_job(job_id: str, file_path: str, filename: str) -> None:
    """后台执行耗时的解析、分块、向量化入库，并持续更新任务进度。"""
    failed_step = "cleanup"
    try:
        upload_job_manager.complete_step(job_id, "upload", "文件已保存到服务器")

        failed_step = "cleanup"
        upload_job_manager.update_step(job_id, "cleanup", 10, "running", "正在清理同名旧文档")
        milvus_manager.init_collection()
        delete_expr = f'filename == "{filename}"'
        try:
            _remove_bm25_stats_for_filename(filename)
        except Exception:
            pass
        try:
            milvus_manager.delete(delete_expr)
        except Exception:
            pass
        try:
            _delete_mysql_document_data(filename)
        except Exception:
            pass
        upload_job_manager.complete_step(job_id, "cleanup", "旧版本清理完成")

        failed_step = "parse"
        upload_job_manager.update_step(job_id, "parse", 5, "running", "正在解析文档并执行语义三层分块（页/段/句）")
        new_docs = _get_document_loader().load_document(file_path, filename)
        if not new_docs:
            raise ValueError("文档处理失败，未能提取内容")

        parent_docs = [doc for doc in new_docs if int(doc.get("chunk_level", 0) or 0) in (1, 2)]
        leaf_docs = [doc for doc in new_docs if int(doc.get("chunk_level", 0) or 0) == 3]
        if not leaf_docs:
            raise ValueError("文档处理失败，未生成可检索叶子分块")
        upload_job_manager.complete_step(
            job_id,
            "parse",
            f"解析完成：父级分块 {len(parent_docs)} 个，叶子分块 {len(leaf_docs)} 个",
        )

        failed_step = "parent_store"
        upload_job_manager.update_step(job_id, "parent_store", 20, "running", "正在写入父级分块")
        parent_chunk_store.upsert_documents(parent_docs)
        upload_job_manager.complete_step(job_id, "parent_store", f"父级分块已入库：{len(parent_docs)} 个")

        failed_step = "vector_store"
        total_leaf = len(leaf_docs)
        upload_job_manager.update_step(
            job_id,
            "vector_store",
            0,
            "running",
            f"正在向量化入库：0 / {total_leaf}",
            total_chunks=total_leaf,
            processed_chunks=0,
        )

        def _on_vector_progress(processed: int, total: int) -> None:
            percent = round(processed * 100 / total) if total else 100
            upload_job_manager.update_step(
                job_id,
                "vector_store",
                percent,
                "running",
                f"正在向量化入库：{processed} / {total}",
                total_chunks=total,
                processed_chunks=processed,
            )

        milvus_writer.write_documents(leaf_docs, progress_callback=_on_vector_progress)
        upload_job_manager.complete_step(job_id, "vector_store", f"向量化入库完成：{total_leaf} 个叶子分块")
        upload_job_manager.complete_job(job_id, f"成功上传并处理 {filename}")
    except Exception as e:
        upload_job_manager.fail_job(job_id, failed_step, str(e))


def _process_delete_job(job_id: str, filename: str) -> None:
    """后台执行文档删除，并把每个删除阶段同步给前端行内进度卡片。"""
    failed_step = "prepare"
    try:
        failed_step = "prepare"
        delete_job_manager.update_step(job_id, "prepare", 20, "running", "正在初始化 Milvus 集合")
        milvus_manager.init_collection()
        delete_expr = f'filename == "{filename}"'
        delete_job_manager.complete_step(job_id, "prepare", "删除任务已创建")

        failed_step = "bm25"
        delete_job_manager.update_step(job_id, "bm25", 20, "running", "正在同步 BM25 统计")
        _remove_bm25_stats_for_filename(filename)
        delete_job_manager.complete_step(job_id, "bm25", "BM25 统计已同步")

        failed_step = "milvus"
        delete_job_manager.update_step(job_id, "milvus", 30, "running", "正在删除 Milvus 向量数据")
        result = milvus_manager.delete(delete_expr)
        deleted_count = result.get("delete_count", 0) if isinstance(result, dict) else 0
        delete_job_manager.complete_step(job_id, "milvus", f"向量数据已删除：{deleted_count} 条")

        failed_step = "parent_store"
        delete_job_manager.update_step(job_id, "parent_store", 30, "running", "正在删除 PostgreSQL 父级分块")
        _delete_mysql_document_data(filename)
        delete_job_manager.complete_step(job_id, "parent_store", "父级分块已删除")

        # 删除本地物理文件
        failed_step = "file_cleanup"
        delete_job_manager.update_step(job_id, "file_cleanup", 20, "running", "正在删除本地文件")
        _delete_local_uploaded_file(filename)
        delete_job_manager.complete_step(job_id, "file_cleanup", "本地文件已删除")

        # 完成摘要会由前端保留 3 秒，再自动从文档列表移除。
        delete_job_manager.complete_job(job_id, f"已删除 {filename}，向量数据 {deleted_count} 条")
    except Exception as e:
        delete_job_manager.fail_job(job_id, failed_step, str(e))


def _persist_image_ocr_document(filename: str, text: str, file_path: str) -> int:
    """将图片 OCR 文本按文档分块并写入 PostgreSQL / Milvus。"""
    cleaned_text = (text or "").strip() or f"[图片OCR未识别到文本] 文件名：{filename}"

    milvus_manager.init_collection()
    try:
        _remove_bm25_stats_for_filename(filename)
    except Exception:
        pass
    try:
        milvus_manager.delete(f'filename == "{filename}"')
    except Exception:
        pass
    try:
        parent_chunk_store.delete_by_filename(filename)
    except Exception:
        pass

    base_doc = {
        "filename": filename,
        "file_path": file_path,
        "file_type": "Image",
        "page_number": 0,
    }
    chunks = _get_document_loader()._split_page_to_three_levels(cleaned_text, base_doc, 0)
    parent_docs = [doc for doc in chunks if int(doc.get("chunk_level", 0) or 0) in (1, 2)]
    leaf_docs = [doc for doc in chunks if int(doc.get("chunk_level", 0) or 0) == 3]
    if not leaf_docs:
        raise ValueError("OCR 文本过短，未生成可检索分块")

    parent_chunk_store.upsert_documents(parent_docs)
    milvus_writer.write_documents(leaf_docs)
    return len(leaf_docs)


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(_: User = Depends(require_admin)):
    """获取 data/documents 下的文件清单，并显示每个文件的三级子块数量。"""
    try:
        # 1) 以 data/documents 实际文件为准（包含 ocr 子目录）
        supported_suffixes = {
            ".pdf": "PDF",
            ".doc": "Word",
            ".docx": "Word",
            ".xls": "Excel",
            ".xlsx": "Excel",
            ".png": "Image",
            ".jpg": "Image",
            ".jpeg": "Image",
            ".bmp": "Image",
            ".gif": "Image",
            ".webp": "Image",
            ".tif": "Image",
            ".tiff": "Image",
        }

        filenames: dict[str, str] = {}
        if UPLOAD_DIR.exists():
            for p in UPLOAD_DIR.rglob("*"):
                if not p.is_file():
                    continue
                ext = p.suffix.lower()
                if ext not in supported_suffixes:
                    continue
                filenames[p.name] = supported_suffixes[ext]

        # 2) 从 Milvus 统计每个文件的三级子块数量（chunk_level == 3）
        leaf_counts: dict[str, int] = {}
        milvus_types: dict[str, str] = {}
        try:
            milvus_manager.init_collection()
            leaf_rows = milvus_manager.query_all(
                filter_expr="chunk_level == 3",
                output_fields=["filename", "file_type"],
            )
            for row in leaf_rows:
                name = row.get("filename") or ""
                if not name:
                    continue
                leaf_counts[name] = leaf_counts.get(name, 0) + 1
                if name not in milvus_types:
                    milvus_types[name] = row.get("file_type") or ""
        except Exception as milvus_err:
            print(f"[list_documents] Milvus 不可用，仅展示本地文件: {milvus_err}", flush=True)

        # 3) 合并：展示 data/documents 中的文件；若 Milvus 有孤儿数据也补充展示
        all_names = set(filenames.keys()) | set(leaf_counts.keys())
        documents: list[DocumentInfo] = []
        for name in sorted(all_names):
            file_type = filenames.get(name) or milvus_types.get(name) or "Unknown"
            documents.append(
                DocumentInfo(
                    filename=name,
                    file_type=file_type,
                    chunk_count=leaf_counts.get(name, 0),
                )
            )

        return DocumentListResponse(documents=documents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取文档列表失败: {str(e)}")

@router.post("/documents/upload/async", response_model=DocumentUploadStartResponse)
async def upload_document_async(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    _: User = Depends(require_admin),
):
    """轻量版异步上传：文件落盘后立即返回 job_id，后台继续解析和向量化。"""
    filename = file.filename or ""
    if not filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")
    if not _is_supported_document(filename):
        raise HTTPException(status_code=400, detail="仅支持 PDF、Word 和 Excel 文档")

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    job = upload_job_manager.create_job(filename)
    file_path = UPLOAD_DIR / filename

    try:
        upload_job_manager.update_step(job["job_id"], "upload", 1, "running", "正在保存文件到服务器")
        await _save_upload_file(file, file_path)
        upload_job_manager.complete_step(job["job_id"], "upload", "文件已上传，等待后台处理")
    except Exception as e:
        upload_job_manager.fail_job(job["job_id"], "upload", f"文件保存失败: {e}")
        raise HTTPException(status_code=500, detail=f"文件保存失败: {e}")

    background_tasks.add_task(_process_upload_job, job["job_id"], str(file_path), filename)
    return DocumentUploadStartResponse(
        job_id=job["job_id"],
        filename=filename,
        message="文件已上传，正在后台解析和向量化入库",
    )


@router.get("/documents/upload/jobs/{job_id}", response_model=DocumentUploadJobResponse)
async def get_upload_job(job_id: str, _: User = Depends(require_admin)):
    job = upload_job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="上传任务不存在或已过期")
    return DocumentUploadJobResponse(**job)


@router.get("/documents/upload/jobs", response_model=list[DocumentUploadJobResponse])
async def list_upload_jobs(_: User = Depends(require_admin)):
    jobs = upload_job_manager.list_jobs()
    jobs.sort(key=lambda item: item.get("created_at", ""), reverse=True)
    return [DocumentUploadJobResponse(**job) for job in jobs]


@router.delete("/documents/delete/async/{filename}", response_model=DocumentDeleteStartResponse)
async def delete_document_async(
    filename: str,
    background_tasks: BackgroundTasks,
    _: User = Depends(require_admin),
):
    """轻量版异步删除：立即返回 job_id，实际删除在后台执行。"""
    job = delete_job_manager.create_job(
        filename,
        steps=DELETE_STEPS,
        current_step="prepare",
        message="等待删除",
        completion_step="file_cleanup",
    )
    delete_job_manager.update_step(job["job_id"], "prepare", 1, "running", "删除任务已提交")
    background_tasks.add_task(_process_delete_job, job["job_id"], filename)
    return DocumentDeleteStartResponse(
        job_id=job["job_id"],
        filename=filename,
        message=f"正在删除 {filename}",
    )


@router.get("/documents/delete/jobs/{job_id}", response_model=DocumentDeleteJobResponse)
async def get_delete_job(job_id: str, _: User = Depends(require_admin)):
    job = delete_job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="删除任务不存在或已过期")
    return DocumentDeleteJobResponse(**job)


@router.post("/ocr/upload", response_model=OCRUploadResponse)
async def ocr_upload(file: UploadFile = File(...), _: User = Depends(get_current_user)):
    filename = file.filename or ""
    if not filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")
    if not is_supported_image(filename):
        raise HTTPException(status_code=400, detail="仅支持图片文件：png/jpg/jpeg/bmp/gif/webp")

    try:
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="图片内容不能为空")
        result = recognize_image_bytes(image_bytes, filename=filename)
        text = (result.get("text") or "").strip()
        provider = result.get("provider") or "ocr"
        message = "OCR 识别成功" if text else f"OCR 已上传，当前未识别到文本（{provider}）"
        return OCRUploadResponse(
            filename=filename,
            text=text,
            provider=provider,
            message=message,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR 识别失败: {str(e)}")


@router.post("/ocr/upload/admin", response_model=OCRUploadResponse)
async def ocr_upload_admin(file: UploadFile = File(...), _: User = Depends(require_admin)):
    filename = file.filename or ""
    if not filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")
    if not is_supported_image(filename):
        raise HTTPException(status_code=400, detail="仅支持图片文件：png/jpg/jpeg/bmp/gif/webp")

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    image_dir = UPLOAD_DIR / "ocr"
    os.makedirs(image_dir, exist_ok=True)
    file_path = image_dir / filename

    try:
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="图片内容不能为空")
        with open(file_path, "wb") as f:
            f.write(image_bytes)
        result = recognize_image_bytes(image_bytes, filename=filename)
        text = (result.get("text") or "").strip()
        provider = result.get("provider") or "ocr"
        if text:
            try:
                chunks_processed = _persist_image_ocr_document(filename, text, str(file_path))
                message = f"OCR 识别成功，已切割并入库 {chunks_processed} 个叶子分块"
            except Exception as persist_err:
                raise HTTPException(status_code=500, detail=f"OCR 文本入库失败: {persist_err}")
        else:
            message = f"OCR 已上传，当前未识别到文本（{provider}）"
        return OCRUploadResponse(
            filename=filename,
            text=text,
            provider=provider,
            message=message,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR 识别失败: {str(e)}")


@router.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...), _: User = Depends(require_admin)):
    """上传文档并进行 embedding（管理员）"""
    try:
        filename = file.filename or ""
        file_lower = filename.lower()
        if not filename:
            raise HTTPException(status_code=400, detail="文件名不能为空")
        if not (
            file_lower.endswith(".pdf")
            or file_lower.endswith((".docx", ".doc"))
            or file_lower.endswith((".xlsx", ".xls"))
        ):
            raise HTTPException(status_code=400, detail="仅支持 PDF、Word 和 Excel 文档")

        os.makedirs(UPLOAD_DIR, exist_ok=True)
        milvus_manager.init_collection()

        delete_expr = f'filename == "{filename}"'
        try:
            _remove_bm25_stats_for_filename(filename)
        except Exception:
            pass
        try:
            milvus_manager.delete(delete_expr)
        except Exception:
            pass
        try:
            parent_chunk_store.delete_by_filename(filename)
        except Exception:
            pass

        file_path = UPLOAD_DIR / filename
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        try:
            new_docs = _get_document_loader().load_document(str(file_path), filename)
        except Exception as doc_err:
            raise HTTPException(status_code=500, detail=f"文档处理失败: {doc_err}")

        if not new_docs:
            raise HTTPException(status_code=500, detail="文档处理失败，未能提取内容")

        parent_docs = [doc for doc in new_docs if int(doc.get("chunk_level", 0) or 0) in (1, 2)]
        leaf_docs = [doc for doc in new_docs if int(doc.get("chunk_level", 0) or 0) == 3]
        if not leaf_docs:
            raise HTTPException(status_code=500, detail="文档处理失败，未生成可检索叶子分块")

        parent_chunk_store.upsert_documents(parent_docs)
        milvus_writer.write_documents(leaf_docs)

        return DocumentUploadResponse(
            filename=filename,
            chunks_processed=len(leaf_docs),
            message=(
                f"成功上传并处理 {filename}，叶子分块 {len(leaf_docs)} 个，"
                f"父级分块 {len(parent_docs)} 个（存入 PostgreSQL）"
            ),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文档上传失败: {str(e)}")


@router.delete("/documents/{filename}", response_model=DocumentDeleteResponse)
async def delete_document(filename: str, _: User = Depends(require_admin)):
    """删除文档在 Milvus 中的向量，并删除 data/documents 下的本地文件。"""
    try:
        milvus_manager.init_collection()

        delete_expr = f'filename == "{filename}"'
        _remove_bm25_stats_for_filename(filename)
        result = milvus_manager.delete(delete_expr)
        _delete_mysql_document_data(filename)
        _delete_local_uploaded_file(filename)

        return DocumentDeleteResponse(
            filename=filename,
            chunks_deleted=result.get("delete_count", 0) if isinstance(result, dict) else 0,
            message=f"成功删除文档 {filename} 的向量数据和本地文件",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除文档失败: {str(e)}")
