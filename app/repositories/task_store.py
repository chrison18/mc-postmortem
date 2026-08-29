"""
SQLite 任务存储。

管理异步分析任务的生命周期：pending → running → completed/failed。
使用标准库 sqlite3，不引入额外依赖。
任务结果（root_cause / fix_suggestion / retrieved_cases 等）直接存在任务表中，
查询时一次性返回，不需要关联表。
"""

import json
import sqlite3
import uuid
from datetime import datetime, timezone

from app.config import settings

# 任务状态枚举
STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"

# 建表 SQL
_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    status TEXT NOT NULL DEFAULT 'pending',
    raw_log_path TEXT,
    fault_category TEXT,
    classify_reason TEXT,
    root_cause TEXT,
    fix_suggestion TEXT,
    retrieved_cases TEXT,
    loop_count INTEGER DEFAULT 0,
    error TEXT,
    created_at TEXT NOT NULL,
    completed_at TEXT
);
"""

# 单例
_store: "TaskStore | None" = None


def _now_iso() -> str:
    """返回当前 UTC 时间的 ISO 格式字符串。"""
    return datetime.now(timezone.utc).isoformat()


class TaskStore:
    """SQLite 任务存储，封装任务的创建、更新和查询。"""

    def __init__(self) -> None:
        """初始化 SQLite 连接并建表。

        数据库文件路径由 settings.SQLITE_PATH 配置。
        check_same_thread=False 允许在后台线程中使用同一连接。
        """
        self._conn = sqlite3.connect(settings.SQLITE_PATH, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(_CREATE_TABLE_SQL)
        self._conn.commit()

    def create_task(self, raw_log_path: str) -> str:
        """创建新任务，返回任务 ID。

        Args:
            raw_log_path: 上传的日志文件路径。

        Returns:
            新创建的任务 ID（UUID 字符串）。
        """
        task_id = str(uuid.uuid4())
        self._conn.execute(
            "INSERT INTO tasks (id, status, raw_log_path, created_at) VALUES (?, ?, ?, ?)",
            (task_id, STATUS_PENDING, raw_log_path, _now_iso()),
        )
        self._conn.commit()
        return task_id

    def update_status(self, task_id: str, status: str) -> None:
        """更新任务状态。

        Args:
            task_id: 任务 ID。
            status: 新状态（pending/running/completed/failed）。
        """
        self._conn.execute(
            "UPDATE tasks SET status = ? WHERE id = ?",
            (status, task_id),
        )
        self._conn.commit()

    def set_raw_log_path(self, task_id: str, raw_log_path: str) -> None:
        """更新任务关联的日志文件路径。

        Args:
            task_id: 任务 ID。
            raw_log_path: 日志文件路径。
        """
        self._conn.execute(
            "UPDATE tasks SET raw_log_path = ? WHERE id = ?",
            (raw_log_path, task_id),
        )
        self._conn.commit()

    def complete_task(self, task_id: str, result: dict) -> None:
        """标记任务完成并写入分析结果。

        Args:
            task_id: 任务 ID。
            result: graph.invoke() 返回的完整状态字典，从中提取各字段。
        """
        retrieved_cases = result.get("retrieved_cases", [])
        retrieved_json = json.dumps(retrieved_cases, ensure_ascii=False) if retrieved_cases else None

        self._conn.execute(
            """UPDATE tasks SET
                status = ?,
                fault_category = ?,
                classify_reason = ?,
                root_cause = ?,
                fix_suggestion = ?,
                retrieved_cases = ?,
                loop_count = ?,
                completed_at = ?
            WHERE id = ?""",
            (
                STATUS_COMPLETED,
                result.get("fault_category"),
                result.get("classify_reason"),
                result.get("root_cause"),
                result.get("fix_suggestion"),
                retrieved_json,
                result.get("loop_count", 0),
                _now_iso(),
                task_id,
            ),
        )
        self._conn.commit()

    def fail_task(self, task_id: str, error: str) -> None:
        """标记任务失败并写入错误信息。

        Args:
            task_id: 任务 ID。
            error: 错误描述。
        """
        self._conn.execute(
            "UPDATE tasks SET status = ?, error = ?, completed_at = ? WHERE id = ?",
            (STATUS_FAILED, error, _now_iso(), task_id),
        )
        self._conn.commit()

    def get_task(self, task_id: str) -> dict | None:
        """查询单个任务。

        Args:
            task_id: 任务 ID。

        Returns:
            任务字典（retrieved_cases 已从 JSON 字符串解析为列表），不存在时返回 None。
        """
        row = self._conn.execute(
            "SELECT * FROM tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
        if row is None:
            return None

        task = dict(row)
        # retrieved_cases 从 JSON 字符串解析回列表
        if task.get("retrieved_cases"):
            try:
                task["retrieved_cases"] = json.loads(task["retrieved_cases"])
            except json.JSONDecodeError:
                task["retrieved_cases"] = []
        else:
            task["retrieved_cases"] = []
        return task

    def list_tasks(self, limit: int = 20) -> list[dict]:
        """列出最近的任务。

        Args:
            limit: 返回数量上限。

        Returns:
            任务列表，按创建时间倒序。
        """
        rows = self._conn.execute(
            "SELECT * FROM tasks ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        tasks = []
        for row in rows:
            task = dict(row)
            if task.get("retrieved_cases"):
                try:
                    task["retrieved_cases"] = json.loads(task["retrieved_cases"])
                except json.JSONDecodeError:
                    task["retrieved_cases"] = []
            else:
                task["retrieved_cases"] = []
            tasks.append(task)
        return tasks

    def close(self) -> None:
        """关闭数据库连接。"""
        self._conn.close()


def get_task_store() -> TaskStore:
    """获取 TaskStore 单例。

    Returns:
        TaskStore 实例。
    """
    global _store
    if _store is None:
        _store = TaskStore()
    return _store
