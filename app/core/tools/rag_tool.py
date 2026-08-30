"""
RAG 追加检索工具。

react_agent 循环中 LLM 可调用的工具，用于按需追加检索相似案例。
使用 OpenAI function calling schema 定义，DeepSeek v4 flash 已确认支持。

MVP 只做这一个工具。工具内部调 case_store.search_similar，
不泄露 Chroma API，与预检索共用同一套 embedding 空间。
"""

from langchain_core.tools import tool

from app.repositories.case_store import get_case_store

# 检索返回的最大案例数
_TOP_K = 5


@tool
def search_similar_cases(query: str) -> str:
    """检索与查询描述相似的历史崩溃案例。

    当你需要更多参考案例来辅助根因分析时调用此工具。
    可以多次调用，每次用不同的查询角度。

    向量库中案例的 embedding 文本由以下部分组成（组织查询词时可参考）：
    - 异常类型 + 异常消息（如 java.lang.NullPointerException: ...）
    - 插件列表
    - Caused by 异常链
    - 前 10 条堆栈帧

    查询词建议用自然语言描述你想找的相似故障场景，
    例如 "NullPointerException during plugin enable" 或 "WorldEdit version conflict"。

    Args:
        query: 自然语言查询描述，说明你想找什么样的相似案例。

    Returns:
        格式化后的相似案例列表，每条包含案例 ID、相似度、异常类型、修复方案和来源。
    """
    store = get_case_store()
    cases = store.search_similar(query, top_k=_TOP_K)

    if not cases:
        return "未找到相似案例。"

    lines = []
    for i, case in enumerate(cases, 1):
        lines.append(f"--- 案例 {i} ---")
        lines.append(f"ID: {case.get('id', '')}")
        lines.append(f"相似度距离: {case.get('distance', '')}（越小越相似）")
        lines.append(f"质量: {case.get('quality', '')}")
        lines.append(f"异常类型: {case.get('exception_type', '')}")
        lines.append(f"来源: {case.get('source_title', '')}")
        lines.append(f"来源链接: {case.get('source_url', '')}")
        fix = case.get("fix_solution", "")
        if fix:
            lines.append(f"修复方案: {fix}")
        lines.append("")

    return "\n".join(lines)
