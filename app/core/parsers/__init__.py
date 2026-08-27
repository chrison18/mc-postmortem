"""日志解析器模块，支持不同格式的崩溃日志解析。"""

from .crash_report import parse_crash_report

__all__ = ["parse_crash_report"]
