"""
中期记忆工具。

react_agent 循环中 LLM 可调用的工具，用于读写分析过程经验。
- save_memory：保存有效检索词、排除方向、中间结论、踩坑记录
- search_memory：按关键词检索历史分析经验

与 search_similar_cases 的区别：
- search_similar_cases 找外部案例的修复方案（别人怎么修的），向量相似度，只读
- search_memory 找本系统历史分析经验（踩过什么坑、什么检索词有效），关键词匹配，可读可写

工具内部捕获异常，返回错误字符串，不抛异常——工具失败不能中断 graph。
"""

from langchain_core.tools import tool

from app.repositories.memory_store import get_memory_store

# 当前分析任务的 task_id，由 routes 层 _run_analysis 开始时设置
# 用于 save_memory 时溯源，中期记忆全局共享，task_id 仅作记录
_current_task_id: str = ""


def set_current_task_id(task_id: str) -> None:
    """设置当前任务 ID，在 _run_analysis 开始时调用。

    Args:
        task_id: 当前分析任务 ID。
    """
    global _current_task_id
    _current_task_id = task_id


@tool
def save_memory(key: str, value: str) -> str:
    """保存一条中期记忆（分析过程经验），全局共享跨任务可用。

    key 建议格式：{类型}:{对象}，类型固定四种：
    - retrieval_tip：有效检索词（如 retrieval_tip:NullPointerException）
    - excluded_direction：已验证排除的方向（如 excluded_direction:WorldEdit）
    - intermediate_conclusion：待验证的中间结论（如 intermediate_conclusion:插件冲突）
    - pitfall：踩坑记录（如 pitfall:旧版Java）

    保存场景：
    - 发现某个检索词效果特别好
    - 验证某个方向无关
    - 得出待验证的中间结论
    - 踩了坑

    Args:
        key: 记忆键，建议 {类型}:{对象} 格式。
        value: 记忆内容，描述具体经验。

    Returns:
        保存成功提示，含记录 id 和 key；失败时返回错误描述。
    """
    try:
        store = get_memory_store()
        mem_id = store.add(_current_task_id, key, value)
        return f"已保存记忆(id={mem_id}): {key}"
    except Exception as e:
        return f"保存记忆失败: {type(e).__name__}: {e}"


@tool
def search_memory(keyword: str, memory_type: str = "") -> str:
    """检索本系统历史分析经验（中期记忆），关键词匹配。

    与 search_similar_cases 的区别：
    - search_similar_cases：找外部案例的修复方案（别人怎么修的），向量相似度检索，只读
    - search_memory：找本系统历史分析经验（之前分析类似问题时踩过什么坑、什么检索词有效），关键词匹配，可读可写

    调用时机：
    1. 分析开始时：按当前异常类型或插件名搜，看有没有历史经验或已知排除方向
    2. RAG 检索结果差时：搜 retrieval_tip 类型，看之前有没有更好的检索词
    3. 准备排除某个方向时：搜 excluded_direction 类型，避免重复排除已验证无关的方向

    Args:
        keyword: 检索关键词，在 key 和 value 中模糊匹配。
        memory_type: 按类型过滤，可选 retrieval_tip / excluded_direction /
            intermediate_conclusion / pitfall；为空时不过滤。

    Returns:
        格式化后的记忆列表；空结果返回"未找到相关记忆"；失败时返回错误描述。
    """
    try:
        store = get_memory_store()
        results = store.search(keyword, memory_type if memory_type else None)
        if not results:
            return "未找到相关记忆"
        lines = [f"[{m['key']}] {m['value']}" for m in results]
        return "\n".join(lines)
    except Exception as e:
        return f"检索记忆失败: {type(e).__name__}: {e}"
