import os
import sys
from pathlib import Path

# 从项目根或 backend 目录启动时，都能解析 `import api`、`from crud import ...`
_BACKEND_ROOT = Path(__file__).resolve().parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

print("[喵呜助手] 正在加载后端…", flush=True)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

print("[喵呜助手] 正在加载路由与业务模块（不连接大模型，请稍候）…", flush=True)
import api as api_module

print("[喵呜助手] 正在加载数据库配置…", flush=True)
from database import init_db

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIST = BASE_DIR / "frontend" / "dist"


def create_app() -> FastAPI:
    app = FastAPI(title="喵呜助手 API")

    @app.on_event("startup")
    async def _startup_init_db():
        init_db()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # No-cache middleware for development
    @app.middleware("http")
    async def _no_cache(request, call_next):
        response = await call_next(request)
        path = request.url.path or ""
        if path == "/" or path.endswith((".html", ".js", ".css")):
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response

    app.include_router(api_module.router)

    # 前后端分离：生产环境在 frontend 目录执行 npm run build 后，由后端托管 dist（可选）
    if FRONTEND_DIST.is_dir() and any(FRONTEND_DIST.iterdir()):
        app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="spa")

    return app


print("[喵呜助手] 正在创建 FastAPI 应用实例…", flush=True)
app = create_app()
print("[喵呜助手] 应用已就绪，即将启动 Web 服务。\n", flush=True)


def main() -> None:
    """启动 Uvicorn。请用 ``uv run python app.py`` 或 ``python app.py``，勿单独使用 ``uv run app.py``（部分 uv 版本不会执行 __main__）。"""
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    print(
        f"\n喵呜助手 API 监听 {host}:{port}\n"
        f"  本机文档: http://127.0.0.1:{port}/docs\n"
        f"  前端(Vite开发): http://127.0.0.1:5173 （需另开 npm run dev）\n",
        flush=True,
    )
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
