"""
Prompt 模块入口。

统一导出各 prompt 构建函数，graph 节点从此处导入，不直接依赖子模块。
"""

from .classify import build_classify_prompt

__all__ = ["build_classify_prompt"]
