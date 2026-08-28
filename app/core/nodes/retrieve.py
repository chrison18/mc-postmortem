"""
相似案例检索节点。

基于 parsed_log 构建查询文本，从 Chroma 向量库检索相似历史案例。
作为强制预检索节点，在 react 循环之前执行，保证 LLM 至少获得一次 RAG 上下文。
"""

from app.core.state import AgentState
from app.repositories.case_store import build_embedding_text, get_case_store

# 检索返回的最大案例数
_TOP_K = 5


def retrieve_cases(state: AgentState) -> dict:
    """检索与当前崩溃日志相似的历史案例。

    从 state 读取 parsed_log，用异常类型+消息+插件+Caused by+前10堆栈构建查询文本，
    调用向量库检索 top-k 相似案例。

    Args:
        state: 当前图状态，需包含 parsed_log。

    Returns:
        包含 retrieved_cases 的部分状态更新。
    """
    parsed_log = state.get("parsed_log")
    if not parsed_log:
        return {"retrieved_cases": []}

    # plugins 从 dict 列表转成字符串列表，与入库格式一致
    plugins = []
    for p in parsed_log.get("plugins", []):
        if isinstance(p, dict):
            name = p.get("name", "")
            version = p.get("version", "")
            plugins.append(f"{name} {version}".strip())
        elif p:
            plugins.append(str(p))

    query_text = build_embedding_text(
        exception_type=parsed_log.get("exception_type", ""),
        exception_message=parsed_log.get("exception_message", ""),
        stack_frames=parsed_log.get("key_stack_frames", []),
        plugins=plugins,
        caused_by_chain=parsed_log.get("caused_by_chain", []),
    )

    if not query_text.strip():
        return {"retrieved_cases": []}

    store = get_case_store()
    cases = store.search_similar(query_text, top_k=_TOP_K)
    return {"retrieved_cases": cases}
