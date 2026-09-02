"""
FastAPI 路由定义。

提供三个接口：
- POST /api/analyze — 上传崩溃日志（文件或文本），触发异步分析，返回 task_id
- GET  /api/tasks/{task_id} — 查询任务状态和分析结果
- GET  /api/tasks — 列出最近的分析任务

分析任务在后台线程中执行（graph.invoke 是同步阻塞操作），
状态通过 SQLite 任务存储持久化，客户端轮询查询结果。
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from app.config import settings
from app.core.graph import create_initial_state, graph
from app.core.parsers import parse_crash_report
from app.repositories.case_store import build_embedding_text, get_case_store
from app.repositories.feedback_store import get_feedback_store
from app.repositories.memory_store import get_memory_store
from app.repositories.task_store import (
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_RUNNING,
    get_task_store,
)

router = APIRouter(prefix="/api", tags=["analysis"])

# 上传文件大小限制（10MB），流式读取避免大文件占内存
_MAX_UPLOAD_SIZE = 10 * 1024 * 1024

# 后台分析线程池，限制最大并发数，防止 LLM 配额或系统资源被打满
# 线程池无界队列，任务会排队等待不会丢失；max_workers=10 是 MVP 配置，后续可根据 LLM 配额调整
_executor = ThreadPoolExecutor(max_workers=10, thread_name_prefix="analysis")


# ---------------------------------------------------------------------------
# 请求/响应模型
# ---------------------------------------------------------------------------

class AnalyzeTextRequest(BaseModel):
    """JSON 方式提交日志文本的请求体。"""
    content: str


class FeedbackRequest(BaseModel):
    """用户纠错反馈请求体。"""
    correct_root_cause: str
    correct_fix_suggestion: str | None = None
    comment: str | None = None


class TaskResponse(BaseModel):
    """任务状态和结果的响应模型。"""
    id: str
    status: str
    raw_log_path: str | None = None
    parent_task_id: str | None = None
    fault_category: str | None = None
    classify_reason: str | None = None
    root_cause: str | None = None
    fix_suggestion: str | None = None
    summary: str | None = None
    confidence: str | None = None
    review_count: int = 0
    review_opinion: str | None = None
    verified: bool = False
    retrieved_cases: list = []
    loop_count: int = 0
    error: str | None = None
    created_at: str | None = None
    completed_at: str | None = None


# ---------------------------------------------------------------------------
# 后台分析执行
# ---------------------------------------------------------------------------

def _run_analysis(task_id: str, raw_log_path: str, initial_messages: list | None = None) -> None:
    """在后台线程中执行完整的 graph 分析流程。

    状态流转：pending → running → completed/failed。
    任何异常都被捕获并写入任务的 error 字段，不会导致进程崩溃。

    Args:
        task_id: 任务 ID。
        raw_log_path: 日志文件路径。
        initial_messages: 初始消息列表（重跑时带入原任务中期记忆提示）。
    """
    store = get_task_store()
    try:
        # 设置中期记忆的当前任务 ID，用于 save_memory 溯源
        from app.core.tools.memory_tool import set_current_task_id
        set_current_task_id(task_id)
        store.update_status(task_id, STATUS_RUNNING)
        result = graph.invoke(create_initial_state(raw_log_path, initial_messages))
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
        # 只读限制+1字节，超过说明文件超限，避免全量读入内存
        file_content = await file.read(_MAX_UPLOAD_SIZE + 1)
        if len(file_content) > _MAX_UPLOAD_SIZE:
            raise HTTPException(status_code=400, detail="文件大小超过 10MB 限制")
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
    if len(request.content) > _MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=400, detail="文本内容超过 10MB 限制")

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


# ---------------------------------------------------------------------------
# 反馈闭环：正向确认 / 负向纠错 / 任务重跑
# ---------------------------------------------------------------------------

def _safe_parse_log(raw_log_path: str | None) -> dict | None:
    """安全解析崩溃日志，解析失败返回 None。

    Args:
        raw_log_path: 日志文件路径。

    Returns:
        ParsedLog 字典，解析失败时返回 None。
    """
    if not raw_log_path:
        return None
    try:
        return parse_crash_report(raw_log_path)
    except Exception:
        return None


def _read_raw_log(raw_log_path: str | None) -> str:
    """读取原始日志全文，文件不存在或读取失败时返回空字符串。

    Args:
        raw_log_path: 日志文件路径。

    Returns:
        日志全文，失败时返回空字符串。
    """
    if not raw_log_path:
        return ""
    try:
        with open(raw_log_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except Exception:
        return ""


@router.post("/tasks/{task_id}/confirm")
async def confirm_task(task_id: str) -> dict:
    """正向确认：将已完成任务的分析结果沉淀为 RAG 案例。

    重复确认幂等（同 id 覆盖）。

    Args:
        task_id: 任务 ID。

    Returns:
        {"case_id": ..., "message": ...}

    Raises:
        HTTPException: 任务不存在 404；任务未完成 400。
    """
    store = get_task_store()
    task = store.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"任务 {task_id} 不存在")
    if task.get("status") != STATUS_COMPLETED:
        raise HTTPException(status_code=400, detail="只能确认已完成的任务")

    parsed_log = _safe_parse_log(task.get("raw_log_path"))

    # source_title 取值链：summary → exception_type → 兜底
    if parsed_log:
        source_title = task.get("summary") or parsed_log.get("exception_type") or "用户确认案例"
        server_type = parsed_log.get("server_type", "unknown")
        mc_version = parsed_log.get("server_version", "")
        java_version = parsed_log.get("java_version", "")
        exception_type = parsed_log.get("exception_type", "")
        plugins_json = json.dumps(parsed_log.get("plugins", []), ensure_ascii=False)
    else:
        source_title = task.get("summary") or "用户确认案例"
        server_type = "unknown"
        mc_version = ""
        java_version = ""
        exception_type = ""
        plugins_json = "[]"

    # fix_solution：fix_suggestion 为 None 时只写根因
    root_cause = task.get("root_cause") or ""
    fix_suggestion = task.get("fix_suggestion")
    if fix_suggestion:
        fix_solution = f"【用户确认】根因：{root_cause}\n修复：{fix_suggestion}"
    else:
        fix_solution = f"【用户确认】根因：{root_cause}"

    # embedding_text：解析成功用结构化字段，失败用 root_cause 兜底
    if parsed_log:
        plugin_names = [
            f"{p.get('name', '')} {p.get('version', '')}".strip()
            for p in parsed_log.get("plugins", [])
        ]
        embedding_text = build_embedding_text(
            exception_type,
            parsed_log.get("exception_message", ""),
            parsed_log.get("key_stack_frames", []),
            plugin_names,
            parsed_log.get("caused_by_chain", []),
        )
    else:
        embedding_text = root_cause

    case_id = f"user_confirmed_{task_id}"
    case = {
        "id": case_id,
        "source_type": "user_confirmed",
        "source_url": "",
        "source_title": source_title,
        "server_type": server_type,
        "mc_version": mc_version,
        "java_version": java_version,
        "exception_type": exception_type,
        "plugins": plugins_json,
        "raw_log": _read_raw_log(task.get("raw_log_path")),
        "fix_solution": fix_solution,
        "quality": "high",
        "collected_at": datetime.now().strftime("%Y-%m-%d"),
        "embedding_text": embedding_text,
    }
    get_case_store().add_cases([case])

    return {"case_id": case_id, "message": "案例已沉淀到知识库"}


@router.post("/tasks/{task_id}/feedback")
async def feedback_task(task_id: str, request: FeedbackRequest) -> dict:
    """负向纠错：记录用户反馈，并将错误经验沉淀为中期记忆 pitfall。

    Args:
        task_id: 任务 ID。
        request: 纠错内容，correct_root_cause 必填。

    Returns:
        {"feedback_id": ..., "message": ...}

    Raises:
        HTTPException: 任务不存在 404；状态不符或无分析结果 400。
    """
    store = get_task_store()
    task = store.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"任务 {task_id} 不存在")
    if task.get("status") not in (STATUS_COMPLETED, STATUS_FAILED):
        raise HTTPException(status_code=400, detail="只能对已完成或失败的任务提交反馈")
    original_root_cause = task.get("root_cause")
    if not original_root_cause:
        raise HTTPException(status_code=400, detail="任务尚无分析结果，无法纠错")

    # 存 feedback 表
    feedback_id = get_feedback_store().add(
        task_id=task_id,
        original_root_cause=original_root_cause,
        correct_root_cause=request.correct_root_cause,
        correct_fix_suggestion=request.correct_fix_suggestion,
        comment=request.comment,
    )

    # 存 pitfall 中期记忆：exception_type 从日志重新解析，失败用 fault_category
    parsed_log = _safe_parse_log(task.get("raw_log_path"))
    if parsed_log:
        exc_type = parsed_log.get("exception_type", "")
    else:
        exc_type = ""
    pitfall_key = f"pitfall:{exc_type}" if exc_type else f"pitfall:{task.get('fault_category') or 'unknown'}"
    pitfall_value = (
        f"分析此问题时曾错误判断为：{original_root_cause}。"
        f"正确答案：{request.correct_root_cause}。"
        f"{request.correct_fix_suggestion or ''}"
    )
    get_memory_store().add(task_id, pitfall_key, pitfall_value)

    return {"feedback_id": feedback_id, "message": "反馈已记录，经验已沉淀到中期记忆"}


@router.post("/tasks/{task_id}/rerun", response_model=TaskResponse)
async def rerun_task(task_id: str) -> TaskResponse:
    """任务重跑：复用原日志，创建新任务并把原任务中期记忆作为初始上下文带入。

    Args:
        task_id: 原任务 ID。

    Returns:
        新任务信息（status=pending，parent_task_id 指向原任务）。

    Raises:
        HTTPException: 原任务不存在 404；原任务未完成 400。
    """
    store = get_task_store()
    original = store.get_task(task_id)
    if original is None:
        raise HTTPException(status_code=404, detail=f"任务 {task_id} 不存在")
    if original.get("status") not in (STATUS_COMPLETED, STATUS_FAILED):
        raise HTTPException(status_code=400, detail="任务尚未完成，无法重跑")

    raw_log_path = original.get("raw_log_path") or ""
    new_task_id = store.create_task(raw_log_path=raw_log_path, parent_task_id=task_id)

    # 把原任务的中期记忆格式化为初始 HumanMessage
    memories = get_memory_store().list_by_task(task_id)
    if memories:
        lines = ["【重跑提示：以下是上次分析时记录的经验，请参考避免重复踩坑】"]
        for m in memories:
            lines.append(f"[{m['key']}] {m['value']}")
        initial_messages = [HumanMessage(content="\n".join(lines))]
    else:
        initial_messages = None

    # 启动后台分析（_run_analysis 内部会设置当前任务 ID）
    _executor.submit(_run_analysis, new_task_id, raw_log_path, initial_messages)

    new_task = store.get_task(new_task_id)
    return TaskResponse(**new_task)
