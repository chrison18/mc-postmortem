"""
FastAPI 路由定义。

提供三个接口：
- POST /api/analyze — 上传崩溃日志（文件或文本），触发异步分析，返回 task_id
- GET  /api/tasks/{task_id} — 查询任务状态和分析结果
- GET  /api/tasks — 列出最近的分析任务

分析任务在后台线程中执行（graph.invoke 是同步阻塞操作），
状态通过 SQLite 任务存储持久化，客户端轮询查询结果。
"""

import os
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from app.config import settings
from app.core.graph import create_initial_state, graph
from app.repositories.task_store import (
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_RUNNING,
    get_task_store,
)

router = APIRouter(prefix="/api", tags=["analysis"])

# 后台分析线程池，限制最大并发数，防止 LLM 配额或系统资源被打满
# 线程池无界队列，任务会排队等待不会丢失；max_workers=10 是 MVP 配置，后续可根据 LLM 配额调整
_executor = ThreadPoolExecutor(max_workers=10, thread_name_prefix="analysis")


# ---------------------------------------------------------------------------
# 请求/响应模型
# ---------------------------------------------------------------------------

class AnalyzeTextRequest(BaseModel):
    """JSON 方式提交日志文本的请求体。"""
    content: str


class TaskResponse(BaseModel):
    """任务状态和结果的响应模型。"""
    id: str
    status: str
    raw_log_path: str | None = None
    fault_category: str | None = None
    classify_reason: str | None = None
    root_cause: str | None = None
    fix_suggestion: str | None = None
    retrieved_cases: list = []
    loop_count: int = 0
    error: str | None = None
    created_at: str | None = None
    completed_at: str | None = None


# ---------------------------------------------------------------------------
# 后台分析执行
# ---------------------------------------------------------------------------

def _run_analysis(task_id: str, raw_log_path: str) -> None:
    """在后台线程中执行完整的 graph 分析流程。

    状态流转：pending → running → completed/failed。
    任何异常都被捕获并写入任务的 error 字段，不会导致进程崩溃。

    Args:
        task_id: 任务 ID。
        raw_log_path: 日志文件路径。
    """
    store = get_task_store()
    try:
        store.update_status(task_id, STATUS_RUNNING)
        result = graph.invoke(create_initial_state(raw_log_path))
        store.complete_task(task_id, result)
    except Exception as e:
        store.fail_task(task_id, f"{type(e).__name__}: {e}")


def _ensure_raw_log_dir() -> None:
    """确保原始日志存放目录存在。"""
    os.makedirs(settings.RAW_LOG_DIR, exist_ok=True)


def _save_log_and_start(task_id: str, content: str | bytes, is_binary: bool) -> str:
    """保存日志内容到文件并启动后台分析。

    Args:
        task_id: 任务 ID。
        content: 日志内容（字符串或字节）。
        is_binary: 是否以二进制方式写入。

    Returns:
        保存后的文件路径。
    """
    _ensure_raw_log_dir()
    file_path = os.path.join(settings.RAW_LOG_DIR, f"{task_id}.txt")
    if is_binary:
        with open(file_path, "wb") as f:
            f.write(content)
    else:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

    # 提交到线程池执行，超过 max_workers 的任务排队等待
    _executor.submit(_run_analysis, task_id, file_path)
    return file_path


# ---------------------------------------------------------------------------
# 路由
# ---------------------------------------------------------------------------

@router.post("/analyze", response_model=TaskResponse)
async def analyze_log(
    file: UploadFile | None = File(default=None, description="崩溃日志文件"),
    content: str | None = Form(default=None, description="崩溃日志文本（与 file 二选一）"),
) -> TaskResponse:
    """上传崩溃日志并触发异步分析。

    支持两种方式（二选一）：
    - multipart/form-data 上传文件（file 参数）
    - form-data 提交文本（content 参数）

    分析在后台执行，立即返回 task_id，客户端通过 GET /api/tasks/{task_id} 轮询结果。

    Args:
        file: 上传的日志文件。
        content: 日志文本内容。

    Returns:
        任务信息（含 task_id 和初始状态 pending）。

    Raises:
        HTTPException: file 和 content 都未提供时返回 400。
    """
    if file is None and not content:
        raise HTTPException(status_code=400, detail="必须提供 file 或 content 其中之一")

    store = get_task_store()
    task_id = store.create_task(raw_log_path="")

    if file is not None:
        file_content = await file.read()
        file_path = _save_log_and_start(task_id, file_content, is_binary=True)
    else:
        file_path = _save_log_and_start(task_id, content, is_binary=False)

    store.set_raw_log_path(task_id, file_path)
    task = store.get_task(task_id)
    return TaskResponse(**task)


@router.post("/analyze/text", response_model=TaskResponse)
async def analyze_log_text(request: AnalyzeTextRequest) -> TaskResponse:
    """JSON 方式提交日志文本并触发异步分析。

    适用于客户端直接发送 JSON 的场景。

    Args:
        request: 包含 content 字段的请求体。

    Returns:
        任务信息（含 task_id 和初始状态 pending）。
    """
    if not request.content.strip():
        raise HTTPException(status_code=400, detail="content 不能为空")

    store = get_task_store()
    task_id = store.create_task(raw_log_path="")
    file_path = _save_log_and_start(task_id, request.content, is_binary=False)

    store.set_raw_log_path(task_id, file_path)
    task = store.get_task(task_id)
    return TaskResponse(**task)


@router.get("/tasks/{task_id}", response_model=TaskResponse)
async def get_task(task_id: str) -> TaskResponse:
    """查询单个任务的状态和分析结果。

    Args:
        task_id: 任务 ID。

    Returns:
        任务完整信息。status 为 completed 时包含 root_cause / fix_suggestion 等结果。

    Raises:
        HTTPException: 任务不存在时返回 404。
    """
    store = get_task_store()
    task = store.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"任务 {task_id} 不存在")
    return TaskResponse(**task)


@router.get("/tasks", response_model=list[TaskResponse])
async def list_tasks(limit: int = 20) -> list[TaskResponse]:
    """列出最近的分析任务。

    Args:
        limit: 返回数量上限，默认 20。

    Returns:
        任务列表，按创建时间倒序。
    """
    store = get_task_store()
    tasks = store.list_tasks(limit=limit)
    return [TaskResponse(**t) for t in tasks]
