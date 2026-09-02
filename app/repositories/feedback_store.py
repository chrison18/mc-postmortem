"""
用户反馈存储（feedback）。

记录用户对分析结果的纠错：原始根因 vs 正确根因。
与 TaskStore / MemoryStore 平级，独立连接，单例模式。
WAL 已由 TaskStore 先开启，不重复开。
"""

import sqlite3
from datetime import datetime, timezone

from app.config import settings

# 建表 SQL
_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    original_root_cause TEXT,
    correct_root_cause TEXT NOT NULL,
    correct_fix_suggestion TEXT,
    comment TEXT,
    created_at TEXT NOT NULL
);
"""

# 单例
_store: "FeedbackStore | None" = None


def _now_iso() -> str:
    """返回当前 UTC 时间的 ISO 格式字符串。"""
    return datetime.now(timezone.utc).isoformat()


class FeedbackStore:
    """用户反馈存储，封装纠错记录的写入和按任务查询。"""

    def __init__(self) -> None:
        """初始化 SQLite 连接并建表。

        使用 settings.SQLITE_PATH，与 tasks / working_memory 同一个数据库文件但独立连接。
        """
        self._conn = sqlite3.connect(settings.SQLITE_PATH, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        # WAL 已由 TaskStore 先开启，不重复开
        self._conn.execute(_CREATE_TABLE_SQL)
        self._conn.commit()

    def add(
        self,
        task_id: str,
        original_root_cause: str | None,
        correct_root_cause: str,
        correct_fix_suggestion: str | None = None,
        comment: str | None = None,
    ) -> int:
        """插入一条反馈记录。

        Args:
            task_id: 被纠错的任务 ID。
            original_root_cause: 任务原分析的根因。
            correct_root_cause: 用户给出的正确根因。
            correct_fix_suggestion: 用户给出的正确修复建议（可选）。
            comment: 补充说明（可选）。

        Returns:
            新插入记录的自增 id。
        """
        cursor = self._conn.execute(
            """INSERT INTO feedback
               (task_id, original_root_cause, correct_root_cause, correct_fix_suggestion, comment, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (task_id, original_root_cause, correct_root_cause, correct_fix_suggestion, comment, _now_iso()),
        )
        self._conn.commit()
        return int(cursor.lastrowid)

    def list_by_task(self, task_id: str) -> list[dict]:
        """按 task_id 查询所有反馈记录。

        Args:
            task_id: 任务 ID。

        Returns:
            反馈记录列表，按 created_at 倒序。
        """
        rows = self._conn.execute(
            "SELECT * FROM feedback WHERE task_id = ? ORDER BY created_at DESC",
            (task_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def close(self) -> None:
        """关闭数据库连接。"""
        self._conn.close()


def get_feedback_store() -> FeedbackStore:
    """获取 FeedbackStore 单例。

    Returns:
        FeedbackStore 实例。
    """
    global _store
    if _store is None:
        _store = FeedbackStore()
    return _store
