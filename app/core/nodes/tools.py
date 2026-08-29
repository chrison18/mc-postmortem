"""
react_agent 的 tools 节点。

执行 agent_node 返回的 tool_calls，将工具结果包装为 ToolMessage 追加到 messages。
执行后回到 agent_node，形成 ReAct 回环。

MVP 只有一个工具 search_similar_cases，但结构上支持多工具并行执行。
"""

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool

from app.core.state import AgentState
from app.core.tools.rag_tool import TOOLS

# 工具名 -> 工具实例的映射，用于按名查找执行
_TOOL_MAP: dict[str, BaseTool] = {tool.name: tool for tool in TOOLS}


def tools_node(state: AgentState) -> dict:
    """tools 节点：执行 AI 消息中的 tool_calls，返回工具结果。

    从 messages 最后一条 AI 消息提取 tool_calls，逐个执行对应工具，
    结果包装为 ToolMessage 追加到 messages。单个工具执行失败时返回错误信息，
    不中断整个节点。

    Args:
        state: 当前图状态，messages 最后一条应为含 tool_calls 的 AIMessage。

    Returns:
        包含 messages（追加 ToolMessage 列表）的部分状态更新。
    """
    messages = state["messages"]
    if not messages:
        return {"messages": []}

    last_msg = messages[-1]
    tool_calls = getattr(last_msg, "tool_calls", None) or []
    if not tool_calls:
        return {"messages": []}

    tool_messages = []
    for tool_call in tool_calls:
        name = tool_call.get("name", "")
        args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id", "")

        result = _execute_tool(name, args)
        tool_messages.append(
            ToolMessage(content=result, tool_call_id=tool_call_id, name=name)
        )

    return {"messages": tool_messages}


def _execute_tool(name: str, args: dict) -> str:
    """按名称执行工具，捕获异常返回错误信息。

    Args:
        name: 工具名称。
        args: 工具参数字典。

    Returns:
        工具执行结果文本，失败时返回错误描述。
    """
    tool = _TOOL_MAP.get(name)
    if tool is None:
        return f"错误：未知工具 '{name}'，可用工具: {list(_TOOL_MAP.keys())}"

    try:
        result = tool.invoke(args)
        # @tool 装饰器的返回值可能是字符串，也可能是其他类型，统一转字符串
        return str(result)
    except Exception as e:
        return f"工具 '{name}' 执行失败: {type(e).__name__}: {e}"
