# 喵呜助手 · catRAG

![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)
![Vue.js](https://img.shields.io/badge/Vue.js-3.x-brightgreen.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)

面向法律知识场景的 **RAG 智能问答系统**，支持 PDF / Word / Excel 异步入库、图片 OCR、混合向量检索、LangGraph 可编排检索链路、SSE 流式对话与全链路 `rag_trace` 追溯。

**仓库地址：** [https://github.com/yaox2689-max/catRAG](https://github.com/yaox2689-max/catRAG)

---

## 功能概览

| 模块 | 说明 |
|------|------|
| 知识库管理 | 管理员上传文档，后台解析 → 三级分块 → 父块入库 PostgreSQL、句子层向量化入 Milvus |
| 混合检索 | BGE-M3 稠密向量 + BM25 稀疏向量，Milvus Hybrid Search + RRF 融合，可选 Jina Rerank |
| Auto-merging | 句子层命中后向上聚合段/页父块，缓解上下文碎片化 |
| LangGraph | 初检 → 相关性评分 → 低相关时 Step-Back / HyDE 扩展检索 |
| 多轮对话 | 会话与消息持久化 PostgreSQL，Redis 缓存热数据 |
| 可观测性 | `rag_trace` 记录策略、召回片段、Rerank 分数、页码；SSE 推送检索步骤 |
| 聊天附件 | 上传 docx/pdf 等解析进当轮上下文（不入库）；图片 OCR 识别 |
| RAGAS 评测 | `test_ragas_eval.py` + 黄金测试集，量化 Faithfulness / Context Precision 等 |

---

## 技术架构

```
┌─────────────┐     SSE / REST      ┌──────────────────────────────────────┐
│  Vue3 前端   │ ◄────────────────► │  FastAPI (backend/app.py)            │
└─────────────┘                     │  · Agent + 工具调用 search_knowledge │
                                    │  · 异步入库/删除 Job                  │
                                    └───────────┬──────────────────────────┘
                                                │
          ┌─────────────────────────────────────┼─────────────────────────┐
          ▼                     ▼                 ▼                         ▼
   ┌─────────────┐      ┌─────────────┐   ┌─────────────┐          ┌─────────────┐
   │ PostgreSQL  │      │   Redis     │   │   Milvus    │          │  LLM API    │
   │ 用户/会话   │      │ 会话缓存    │   │ L3 向量检索 │          │ Ark/OpenAI  │
   │ L1/L2 父块  │      │             │   │ Dense+BM25  │          │ 兼容接口    │
   └─────────────┘      └─────────────┘   └─────────────┘          └─────────────┘
```

### 三级分块与存储

| 层级 | 语义单元 | 存储 |
|------|----------|------|
| L1 | 整页正文 | PostgreSQL `parent_chunks` |
| L2 | 段落 | PostgreSQL |
| L3 | 句子（超长句按 ~300 token 切分） | Milvus（检索 + 向量） |

检索仅在 **L3** 执行；命中后通过 **Auto-merging** 从 PostgreSQL 拉取 L2/L1 父块。

### LangGraph 检索流程

```mermaid
flowchart LR
  A[retrieve_initial] --> B[grade_documents]
  B -->|相关| C[END → Agent 生成]
  B -->|不相关| D[rewrite_question]
  D --> E[retrieve_expanded]
  E --> C
```

扩展阶段支持 **Step-Back**、**HyDE** 及组合策略（由 LLM 路由选择）。

---

## 环境要求

- Python **≥ 3.12**
- [uv](https://github.com/astral-sh/uv) 或 pip
- Node.js **≥ 18**（前端开发）
- Docker & Docker Compose（推荐，用于 PostgreSQL / Redis / Milvus）

---

## 快速开始

### 1. 启动基础设施

```bash
docker compose up -d
```

将启动：

| 服务 | 端口 | 说明 |
|------|------|------|
| PostgreSQL | 5432 | 库名 `langchain_app`，用户/密码 `postgres` |
| Redis | 6379 | 会话缓存 |
| Milvus | 19530 | 向量库 |
| Attu | 8080 | Milvus 可视化管理（可选） |

### 2. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env`，至少配置：

- `ARK_API_KEY`、`MODEL`、`BASE_URL` — 对话与检索编排所用大模型
- `DATABASE_URL` — 使用 Docker 中 Postgres 时示例：  
  `postgresql+psycopg2://postgres:postgres@127.0.0.1:5432/langchain_app`
- `REDIS_URL` — 默认 `redis://localhost:6379/0`
- `MILVUS_HOST` / `MILVUS_PORT` — 默认 `127.0.0.1:19530`

可选：`RERANK_*`（Jina Rerank）、`BAIDU_OCR_*`（图片 OCR）、`ADMIN_INVITE_CODE`（注册管理员）。

### 3. 安装依赖并启动后端

```bash
# 推荐使用 uv
uv sync
cd backend
uv run python app.py
```

API 默认：`http://127.0.0.1:8000`  
Swagger 文档：`http://127.0.0.1:8000/docs`

### 4. 启动前端（开发模式）

```bash
cd frontend
npm install
npm run dev
```

浏览器访问：`http://127.0.0.1:5173`（Vite 已将 `/auth`、`/chat`、`/documents` 等代理到后端）。

### 5. 生产构建（可选）

```bash
cd frontend && npm run build
cd ../backend && uv run python app.py
```

若存在 `frontend/dist`，后端会自动托管静态资源。

---

## 使用说明

### 注册与登录

1. 打开前端，注册普通用户或管理员（管理员需填写 `.env` 中的 `ADMIN_INVITE_CODE`）。
2. 登录后即可使用对话；管理员可进入 **设置** 管理知识库。

### 管理员：文档入库

1. **设置 → 上传文档**，选择 PDF / Word / Excel。
2. 系统后台执行：清理旧版本 → **语义三层分块** → 父块写入 PostgreSQL → 叶子块向量化写入 Milvus。
3. 页面展示分阶段进度；完成后在「已上传文档」列表中查看。

### 用户：智能问答

1. **新建会话** 后输入法律相关问题。
2. 可附加 **图片**（OCR 识别后作为上下文）或 **文档**（仅当轮解析，不写入知识库）。
3. 流式回答过程中可查看 **检索步骤**；展开 **检索过程** 查看 `rag_trace` 与引用来源。

---

## 项目结构

```
catRAG/
├── backend/
│   ├── app.py              # FastAPI 入口
│   ├── api.py              # REST / SSE 路由
│   ├── crud.py             # Agent 对话、流式输出
│   ├── document_loader.py  # 文档加载 + 三级分块编排
│   ├── semantic_chunker.py # 页/段/句语义切分、token 切分
│   ├── rag_pipeline.py     # LangGraph 检索图
│   ├── rag_utils.py        # 混合检索、Auto-merging、Rerank
│   ├── milvus_client.py    # Milvus 混合检索
│   ├── parent_chunk_store.py
│   ├── embedding.py        # BGE-M3 + BM25
│   └── ocr_service.py
├── frontend/               # Vue 3 + Vite
├── data/                   # 上传文档、BM25 状态、评测数据
├── docker-compose.yml
├── test_ragas_eval.py      # RAGAS 评测脚本
└── pyproject.toml
```

---

## 主要 API

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/auth/register` `/auth/login` | 注册 / 登录 |
| POST | `/chat/stream` | SSE 流式对话（含 `rag_step`、`trace`） |
| GET | `/documents` | 文档列表（管理员） |
| POST | `/documents/upload/async` | 异步入库 |
| DELETE | `/documents/delete/async/{filename}` | 异步删除 |
| POST | `/ocr/upload` | 图片 OCR（用户） |
| POST | `/ocr/upload/admin` | 图片 OCR 并入库（管理员） |

完整接口见 Swagger：`/docs`。

---

## RAGAS 评测

准备黄金集 CSV（至少含 `question` 列，推荐含 `ground_truth`），默认路径 `data/ragas_eval_gold.csv`：

```bash
uv run python test_ragas_eval.py --csv data/ragas_eval_gold.csv --preset standard
```

常用预设：`faithfulness` | `retrieval` | `standard` | `full` | `complete`  
结果输出至 `data/` 目录。

---

## 环境变量参考

| 变量 | 说明 | 默认 |
|------|------|------|
| `DATABASE_URL` | SQLAlchemy 连接串 | 见 `database.py` |
| `REDIS_URL` | Redis | `redis://localhost:6379/0` |
| `MILVUS_HOST` / `MILVUS_PORT` | Milvus | `localhost` / `19530` |
| `EMBEDDING_MODEL` | 稠密向量模型 | `BAAI/bge-m3` |
| `CHUNK_LEVEL3_MAX_TOKENS` | L3 超长句切分 token 上限 | `300` |
| `AUTO_MERGE_THRESHOLD` | Auto-merging 子块数阈值 | `2` |
| `LEAF_RETRIEVE_LEVEL` | Milvus 检索层级 | `3` |
| `JWT_SECRET_KEY` | JWT 密钥 | 务必在生产环境修改 |

更多见 [.env.example](.env.example)。

---

## 服务器部署

生产环境部署步骤（Docker 中间件 + systemd + Nginx + HTTPS）见 **[deploy/DEPLOY.md](deploy/DEPLOY.md)**。

---

## 开发指南

### 分支策略

本项目采用 **Git Flow** 分支策略：

```
main          # 生产分支，只接受 PR 合并
├── develop   # 开发分支，功能合并到这里
├── feature/* # 功能分支，从 develop 创建
├── fix/*     # 修复分支
└── release/* # 发布分支
```

**工作流程：**
1. 从 `develop` 创建功能分支：`git checkout -b feature/your-feature develop`
2. 开发完成后推送：`git push origin feature/your-feature`
3. 在 GitHub 创建 Pull Request 到 `develop`
4. 代码审查后合并

### 提交规范

使用 **Conventional Commits** 格式：

```
<type>(<scope>): <description>

[optional body]
[optional footer]
```

**类型：**
- `feat`: 新功能
- `fix`: 修复 bug
- `docs`: 文档更新
- `style`: 代码格式（不影响功能）
- `refactor`: 重构
- `test`: 测试
- `chore`: 构建/工具

**示例：**
```
feat(rag): add semantic chunking with auto-merging
fix(auth): resolve login compatibility issue
docs(readme): add installation guide
```

### Pull Request 规范

- PR 标题清晰描述变更内容
- 包含变更说明和测试结果
- 关联相关 Issue（如有）
- 至少一位审查者批准

---

## 常见问题

**Q: 文档列表加载失败，提示 Milvus `closed channel`？**  
A: 确认 `docker compose` 中 Milvus 已健康启动；后端会对 Milvus 断连尝试重连，列表接口在 Milvus 不可用时会降级为仅展示本地 `data/documents` 文件。

**Q: 聊天里上传的 Word 和设置里入库有何区别？**  
A: 聊天附件由 `chat_file_parser` 解析后注入**当轮消息**，不写入 Milvus；设置中的上传才会完整分块并向量化。

**Q: 首次启动 embedding 很慢？**  
A: 需下载 `BAAI/bge-m3`；可配置 `HF_ENDPOINT` 使用镜像加速（见 [HUGGINGFACE_FIX.md](HUGGINGFACE_FIX.md)）。

---

## 路线图

- [ ] 支持更多文档格式（PPT、Markdown）
- [ ] 实现文档版本管理
- [ ] 添加用户权限细粒度控制
- [ ] 支持多模型切换
- [ ] 添加单元测试和集成测试
- [ ] 优化向量检索性能
- [ ] 支持私有化部署文档

## 贡献指南

欢迎贡献！请遵循以下步骤：

1. Fork 本仓库
2. 创建功能分支：`git checkout -b feature/amazing-feature`
3. 提交更改：`git commit -m 'feat: add amazing feature'`
4. 推送分支：`git push origin feature/amazing-feature`
5. 创建 Pull Request

请确保：
- 代码符合项目规范
- 添加必要的测试
- 更新相关文档
- 提交信息符合 Conventional Commits 规范

---

## 许可证

本项目采用 [MIT License](LICENSE) 开源许可证。

Copyright (c) 2026 yaox2689-max
