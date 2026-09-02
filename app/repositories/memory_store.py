"""
中期记忆存储（working_memory）。

保存分析过程中的经验：有效检索词、排除方向、中间结论、踩坑记录。
与 TaskStore 平级，独立连接，单例模式。
全局共享，跨任务可用，task_id 仅作溯源记录。

不建索引：LIKE '%keyword%' 用不上索引，MVP 数据少全表扫描没问题。
不加锁：写操作少，SQLite WAL 单写够用。
WAL 模式已由 TaskStore 先开启，这里不重复开。
"""

import sqlite3
from datetime import datetime, timezone

from app.config import settings

# 建表 SQL
_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS working_memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""

# 单例
_store: "MemoryStore | None" = None


def _now_iso() -> str:
    """返回当前 UTC 时间的 ISO 格式字符串。"""
    return datetime.now(timezone.utc).isoformat()


class MemoryStore:
    """中期记忆存储，封装经验的写入和关键词检索。

    独立连接，不复用 TaskStore 的连接；key 格式不校验，LLM 传什么存什么。
    """

    def __init__(self) -> None:
        """初始化 SQLite 连接并建表。

        使用 settings.SQLITE_PATH，与 tasks 表同一个数据库文件但独立连接。
        check_same_thread=False 允许在后台分析线程中使用。
        """
        self._conn = sqlite3.connect(settings.SQLITE_PATH, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        # WAL 已由 TaskStore 先开启，不重复开
        self._conn.execute(_CREATE_TABLE_SQL)
        self._conn.commit()

    def add(self, task_id: str, key: str, value: str) -> int:
        """插入一条中期记忆。

        Args:
            task_id: 来源任务 ID，仅作溯源记录。
            key: 记忆键，建议格式 {类型}:{对象}，类型为 retrieval_tip /
                excluded_direction / intermediate_conclusion / pitfall。
            value: 记忆内容。

        Returns:
            新插入记录的自增 id。
        """
        cursor = self._conn.execute(
            "INSERT INTO working_memory (task_id, key, value, created_at) VALUES (?, ?, ?, ?)",
            (task_id, key, value, _now_iso()),
        )
        self._conn.commit()
        return int(cursor.lastrowid)

    def search(
        self, keyword: str, memory_type: str | None = None, limit: int = 5
    ) -> list[dict]:
        """在 key 和 value 中做 LIKE 模糊匹配检索。

        Args:
            keyword: 检索关键词。
            memory_type: 不为空时按 key 前缀过滤，如 "retrieval_tip"。
            limit: 返回条数上限。

        Returns:
            记忆列表，按 created_at 倒序，每条含 id/task_id/key/value/created_at。
        """
        sql = "SELECT * FROM working_memory WHERE (key LIKE ? OR value LIKE ?)"
        params = [f"%{keyword}%", f"%{keyword}%"]
        if memory_type:
            sql += " AND key LIKE ?"
            params.append(f"{memory_type}:%")
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(sql, params).fetchall()
        return [dict(row) for row in rows]

    def close(self) -> None:
        """关闭数据库连接。"""
        self._conn.close()


def get_memory_store() -> MemoryStore:
    """获取 MemoryStore 单例。

    Returns:
        MemoryStore 实例。
    """
    global _store
    if _store is None:
        _store = MemoryStore()
    return _store
