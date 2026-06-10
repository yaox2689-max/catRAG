from dotenv import load_dotenv
import os
import json
import asyncio
from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk, SystemMessage
from tools import get_current_weather, search_knowledge_base, get_last_rag_context, reset_tool_call_guards, set_rag_step_queue
from datetime import datetime
from cache import cache
from database import SessionLocal
from sqlbase import User, ChatSession, ChatMessage

load_dotenv()

# Token 预算管理常量
WATERMARK_TOKENS = int(os.getenv("WATERMARK_TOKENS", "22000"))  # 触发摘要的水位线
MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "25000"))  # 总上下文硬上限
KEEP_RECENT_MESSAGES = int(os.getenv("KEEP_RECENT_MESSAGES", "10"))  # 摘要时保留最近消息数

_tiktoken_encoder = None

def _get_encoder():
    global _tiktoken_encoder
    if _tiktoken_encoder is None:
        import tiktoken
        _tiktoken_encoder = tiktoken.get_encoding("cl100k_base")
    return _tiktoken_encoder

def _count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(_get_encoder().encode(text))

def _count_message_tokens(messages: list) -> int:
    total = 0
    for msg in messages:
        content = msg.content if hasattr(msg, "content") else str(msg)
        total += _count_tokens(content) + 4  # role formatting overhead
    return total

def _check_token_watermark(messages: list, model) -> list:
    """检查 token 水位线，超过阈值时触发摘要压缩。返回处理后的消息列表。"""
    total = _count_message_tokens(messages)
    if total <= WATERMARK_TOKENS:
        return messages

    if len(messages) <= KEEP_RECENT_MESSAGES:
        return messages

    old_messages = messages[:-KEEP_RECENT_MESSAGES]
    recent_messages = messages[-KEEP_RECENT_MESSAGES:]
    old_tokens = _count_message_tokens(old_messages)

    if old_tokens < 500:
        return messages

    summary = summarize_old_messages(model, old_messages)
    summary_msg = SystemMessage(content=f"之前的对话摘要：\n{summary}")
    return [summary_msg] + recent_messages

API_KEY = os.getenv("ARK_API_KEY")
MODEL = os.getenv("MODEL")
BASE_URL = os.getenv("BASE_URL")

class ConversationStorage:
    """对话存储（PostgreSQL + Redis）。"""

    @staticmethod
    def _messages_cache_key(user_id: str, session_id: str) -> str:
        return f"chat_messages:{user_id}:{session_id}"

    @staticmethod
    def _sessions_cache_key(user_id: str) -> str:
        return f"chat_sessions:{user_id}"

    @staticmethod
    def _to_langchain_messages(records: list[dict]) -> list:
        messages = []
        for msg_data in records:
            msg_type = msg_data.get("type")
            content = msg_data.get("content", "")
            if msg_type == "human":
                messages.append(HumanMessage(content=content))
            elif msg_type == "ai":
                messages.append(AIMessage(content=content))
            elif msg_type == "system":
                messages.append(SystemMessage(content=content))
        return messages

    def save(self, user_id: str, session_id: str, messages: list, metadata: dict = None, extra_message_data: list = None):
        """保存对话"""
        db = SessionLocal()
        try:
            user = db.query(User).filter(User.username == user_id).first()
            if not user:
                return

            session = (
                db.query(ChatSession)
                .filter(ChatSession.user_id == user.id, ChatSession.session_id == session_id)
                .first()
            )
            if not session:
                session = ChatSession(user_id=user.id, session_id=session_id, metadata_json=metadata or {})
                db.add(session)
                db.flush()
            #如果存在更新元数据
            else:
                session.metadata_json = metadata or {}
            #删除改会话的所有旧消息
            db.query(ChatMessage).filter(ChatMessage.session_ref_id == session.id).delete(synchronize_session=False)

            serialized = []
            now = datetime.utcnow()
            for idx, msg in enumerate(messages):
                rag_trace = None
                if extra_message_data and idx < len(extra_message_data):
                    #统一处理：无论有没有数据，extra 都是字典
                    extra = extra_message_data[idx] or {}
                    #提取 RAG 追踪信息：从字典中安全地获取 "rag_trace" 键的值
                    rag_trace = extra.get("rag_trace")

                db.add(
                    ChatMessage(
                        session_ref_id=session.id,
                        message_type=msg.type,
                        content=str(msg.content),
                        timestamp=now,
                        rag_trace=rag_trace,
                    )
                )
                serialized.append(
                    {
                        "type": msg.type,
                        "content": str(msg.content),
                        "timestamp": now.isoformat(),
                        "rag_trace": rag_trace,
                    }
                )

            session.updated_at = now
            db.commit()

            cache.set_json(self._messages_cache_key(user_id, session_id), serialized)
            #删除所有会话缓存，当前窗口有数据才保留
            cache.delete(self._sessions_cache_key(user_id))
        finally:
            db.close()

    def load(self, user_id: str, session_id: str) -> list:
        """加载对话"""
        cached = cache.get_json(self._messages_cache_key(user_id, session_id))
        if cached is not None:
            return self._to_langchain_messages(cached)

        records = self.get_session_messages(user_id, session_id)
        cache.set_json(self._messages_cache_key(user_id, session_id), records)
        return self._to_langchain_messages(records)

    def list_sessions(self, user_id: str) -> list:
        """列出用户的所有会话"""
        return [item["session_id"] for item in self.list_session_infos(user_id)]

    def list_session_infos(self, user_id: str) -> list[dict]:
        cached = cache.get_json(self._sessions_cache_key(user_id))
        if cached is not None:
            return cached

        db = SessionLocal()
        try:
            user = db.query(User).filter(User.username == user_id).first()
            if not user:
                return []

            sessions = (
                db.query(ChatSession)
                .filter(ChatSession.user_id == user.id)
                .order_by(ChatSession.updated_at.desc())
                .all()
            )
            result = []
            for s in sessions:
                count = db.query(ChatMessage).filter(ChatMessage.session_ref_id == s.id).count()
                result.append(
                    {
                        "session_id": s.session_id,
                        "updated_at": s.updated_at.isoformat(),
                        "message_count": count,
                    }
                )
            cache.set_json(self._sessions_cache_key(user_id), result)
            return result
        finally:
            db.close()

    def get_session_messages(self, user_id: str, session_id: str) -> list[dict]:
        cached = cache.get_json(self._messages_cache_key(user_id, session_id))
        if cached is not None:
            return cached

        db = SessionLocal()
        try:
            user = db.query(User).filter(User.username == user_id).first()
            if not user:
                return []
            session = (
                db.query(ChatSession)
                .filter(ChatSession.user_id == user.id, ChatSession.session_id == session_id)
                .first()
            )
            if not session:
                return []

            rows = (
                db.query(ChatMessage)
                .filter(ChatMessage.session_ref_id == session.id)
                .order_by(ChatMessage.id.asc())
                .all()
            )
            result = [
                {
                    "type": row.message_type,
                    "content": row.content,
                    "timestamp": row.timestamp.isoformat(),
                    "rag_trace": row.rag_trace,
                }
                for row in rows
            ]
            cache.set_json(self._messages_cache_key(user_id, session_id), result)
            return result
        finally:
            db.close()

    def delete_session(self, user_id: str, session_id: str) -> bool:
        """删除指定用户的会话，返回是否删除成功"""
        db = SessionLocal()
        try:
            user = db.query(User).filter(User.username == user_id).first()
            if not user:
                return False
            session = (
                db.query(ChatSession)
                .filter(ChatSession.user_id == user.id, ChatSession.session_id == session_id)
                .first()
            )
            if not session:
                return False

            db.delete(session)
            db.commit()
            cache.delete(self._messages_cache_key(user_id, session_id))
            cache.delete(self._sessions_cache_key(user_id))
            return True
        finally:
            db.close()

def create_agent_instance():
    # 推迟导入：避免 import crud 时加载 agents 子系统导致进程退出或长时间无输出
    from langchain.agents import create_agent
    from langchain.chat_models import init_chat_model

    model = init_chat_model(
        model=MODEL,
        model_provider="openai",
        api_key=API_KEY,
        base_url=BASE_URL,
        temperature=0.3,
        stream_usage=True,
    )

    agent = create_agent(
        model=model,
        tools=[get_current_weather, search_knowledge_base],
        system_prompt=(
            "You are a cute cat bot that loves to help users. "
            "When responding, you may use tools to assist. "
            "Use search_knowledge_base when users ask document/knowledge questions. "
            "Do not call the same tool repeatedly in one turn. At most one knowledge tool call per turn. "
            "Once you call search_knowledge_base and receive its result, you MUST immediately produce the Final Answer based on that result. "
            "After receiving search_knowledge_base result, you MUST NOT call any tool again (including get_current_weather or search_knowledge_base). "
            "If the retrieved context is insufficient, answer honestly that you don't know instead of making up facts. "
            "If tool results include a Step-back Question/Answer, use that general principle to reason and answer, "
            "but do not reveal chain-of-thought. "
            "If you don't know the answer, admit it honestly."
        ),
    )
    return agent, model


_agent_singleton = None
_model_singleton = None


def _get_agent_model():
    """首次对话时再初始化 LangChain Agent，避免 import 阶段长时间阻塞或网络卡住。"""
    global _agent_singleton, _model_singleton
    if _agent_singleton is None:
        print("[喵呜助手] 正在初始化大模型与 Agent（首次可能较慢）…", flush=True)
        _agent_singleton, _model_singleton = create_agent_instance()
    return _agent_singleton, _model_singleton


storage = ConversationStorage()

def summarize_old_messages(model, messages: list) -> str:
    """将旧消息总结为结构化摘要"""
    old_conversation = "\n".join([
        f"{'用户' if msg.type == 'human' else 'AI'}: {msg.content}"
        for msg in messages
    ])

    summary_prompt = f"""请对以下对话生成结构化摘要，包含以下维度：
1. 用户身份与背景（如有提及）
2. 已讨论的法律领域和关键问题
3. 已解决的问题和关键结论
4. 未解决的问题或待跟进事项
5. 用户的偏好和特殊要求

对话内容：
{old_conversation}

结构化摘要："""

    summary = model.invoke(summary_prompt).content
    return summary


def _build_user_payload(
    user_text: str,
    image_context: str = "",
    file_name: str = "",
    file_content_b64: str = "",
) -> str:
    extra_parts = []
    if image_context:
        extra_parts.append(f"图片OCR内容：\n{image_context}")
    if file_name and file_content_b64:
        from chat_file_parser import extract_text_from_upload

        extracted = extract_text_from_upload(file_name, file_content_b64)
        if extracted:
            extra_parts.append(f"文件名：{file_name}\n文件内容：\n{extracted}")
        else:
            extra_parts.append(f"文件名：{file_name}\n（未能解析出可读内容，请确认文件格式是否正确）")
    if extra_parts:
        return f"{user_text}\n\n" + "\n\n".join(extra_parts) if user_text else "\n\n".join(extra_parts)
    return user_text


def chat_with_agent(user_text: str, user_id: str = "default_user", session_id: str = "default_session", image_context: str = "", file_name: str = "", file_content_b64: str = ""):
    """使用 Agent 处理用户消息并返回响应"""
    agent, model = _get_agent_model()
    messages = storage.load(user_id, session_id)

    # 清理可能残留的 RAG 上下文，避免跨请求污染
    get_last_rag_context(clear=True)
    reset_tool_call_guards()

    # Token 水位线管理：超过阈值时自动摘要压缩
    messages = _check_token_watermark(messages, model)

    user_payload = _build_user_payload(user_text, image_context, file_name, file_content_b64)
    messages.append(HumanMessage(content=user_payload))
    result = agent.invoke(
        {"messages": messages},
        config={"recursion_limit": 8},
    )

    response_content = ""
    if isinstance(result, dict):
        #output是大模型的输出
        if "output" in result:
            response_content = result["output"]
        elif "messages" in result and result["messages"]:
            msg = result["messages"][-1]
            response_content = getattr(msg, "content", str(msg))
        else:
            response_content = str(result)
    elif hasattr(result, "content"):
        response_content = result.content
    else:
        response_content = str(result)
    
    messages.append(AIMessage(content=response_content))

    rag_context = get_last_rag_context(clear=True)
    rag_trace = rag_context.get("rag_trace") if rag_context else None

    extra_message_data = [None] * (len(messages) - 1) + [{"rag_trace": rag_trace}]
    storage.save(user_id, session_id, messages, extra_message_data=extra_message_data)

    return {
        "response": response_content,
        "rag_trace": rag_trace,
    }

async def chat_with_agent_stream(user_text: str, user_id: str = "default_user", session_id: str = "default_session", image_context: str = "", file_name: str = "", file_content_b64: str = ""):
    """使用 Agent 处理用户消息并流式返回响应。
    
    架构：使用统一输出队列 + 后台任务，确保 RAG 检索步骤在工具执行期间实时推送，
    而非等待工具完成后才显示。
    """
    messages = storage.load(user_id, session_id)

    # 清理可能残留的 RAG 上下文
    get_last_rag_context(clear=True)
    reset_tool_call_guards()

    agent, model = _get_agent_model()

    # 统一输出队列：所有事件（content / rag_step）都汇入这里
    output_queue = asyncio.Queue()

    class _RagStepProxy:
        """代理对象：将 emit_rag_step 的原始 step dict 包装后放入统一输出队列。"""
        def put_nowait(self, step):
            output_queue.put_nowait({"type": "rag_step", "step": step})

    set_rag_step_queue(_RagStepProxy())

    # Token 水位线管理：超过阈值时自动摘要压缩
    messages = _check_token_watermark(messages, model)

    user_payload = _build_user_payload(user_text, image_context, file_name, file_content_b64)
    messages.append(HumanMessage(content=user_payload))

    full_response = ""

    async def _agent_worker():
        """后台任务：运行 agent 并将内容 chunk 推入输出队列。"""
        nonlocal full_response
        try:
            async for msg, metadata in agent.astream(
                {"messages": messages},
                stream_mode="messages",
                config={"recursion_limit": 8},
            ):
                if not isinstance(msg, AIMessageChunk):
                    continue
                if getattr(msg, "tool_call_chunks", None):
                    continue

                content = ""
                if isinstance(msg.content, str):
                    content = msg.content
                elif isinstance(msg.content, list):
                    for block in msg.content:
                        if isinstance(block, str):
                            content += block
                        elif isinstance(block, dict) and block.get("type") == "text":
                            content += block.get("text", "")

                if content:
                    full_response += content
                    await output_queue.put({"type": "content", "content": content})
        except Exception as e:
            await output_queue.put({"type": "error", "content": str(e)})
        finally:
            # 哨兵：通知主循环 agent 已完成
            await output_queue.put(None)

    # 启动后台任务
    agent_task = asyncio.create_task(_agent_worker())

    try:
        # 主循环：持续从统一队列取事件并 yield SSE
        # RAG 步骤在工具执行期间通过 call_soon_threadsafe 实时入队，不需要等 agent 产出 chunk
        while True:
            event = await output_queue.get()
            if event is None:
                break
            yield f"data: {json.dumps(event)}\n\n"
    except GeneratorExit:
        # 客户端断开连接（AbortController）时，FastAPI 会向此生成器抛出 GeneratorExit
        # 我们必须在此处取消后台任务
        agent_task.cancel()
        try:
            await agent_task
        except asyncio.CancelledError:
            pass  # 任务已成功取消
        raise  # 重新抛出 GeneratorExit 以便 FastAPI 正确处理关闭
    finally:
        # 正常结束或异常退出时清理
        set_rag_step_queue(None)
        if not agent_task.done():
             agent_task.cancel()

    # 获取 RAG trace
    rag_context = get_last_rag_context(clear=True)
    rag_trace = rag_context.get("rag_trace") if rag_context else None

    # 发送 trace 信息
    if rag_trace:
        yield f"data: {json.dumps({'type': 'trace', 'rag_trace': rag_trace})}\n\n"

    # 发送结束信号
    yield "data: [DONE]\n\n"

    # 保存对话
    messages.append(AIMessage(content=full_response))
    extra_message_data = [None] * (len(messages) - 1) + [{"rag_trace": rag_trace}]
    storage.save(user_id, session_id, messages, extra_message_data=extra_message_data)
