"""
FastAPI 应用入口。

启动方式：
    python main.py
或：
    uvicorn main:app --host 0.0.0.0 --port 8000

API 文档：http://localhost:8000/docs
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router as api_router
from app.config import settings
from app.repositories.task_store import get_task_store


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理：启动时初始化数据库连接和建表。"""
    # 启动时初始化 task_store（建表）
    get_task_store()
    yield
    # 关闭时不需要额外清理（SQLite 连接随进程结束自动关闭）


app = FastAPI(
    title="MC 服务端事故复盘系统",
    description="基于 LangGraph 的 MC 服务端崩溃日志分析系统，输入崩溃日志，输出根因分析和修复建议。",
    version="0.1.0",
    lifespan=lifespan,
)

# 注册 API 路由
app.include_router(api_router)


@app.get("/health", tags=["health"])
async def health_check() -> dict:
    """健康检查接口。

    Returns:
        服务状态信息。
    """
    return {
        "status": "ok",
        "version": app.version,
        "model": settings.LLM_MODEL,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=False,
    )
