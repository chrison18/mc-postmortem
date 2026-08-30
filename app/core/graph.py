"""
LangGraph 主流程定义。

设计为混合 RAG 模式：
- retrieve_cases 节点在 react 循环前强制预检索一次，保证 LLM 至少获得一次历史案例上下文，
  避免 LLM 自认为不需要检索就直接下结论。
- react_agent 循环内通过 RAG tool 支持追加检索，LLM 按需调用。

react_agent 拆分为标准 LangGraph 双节点：
- agent_node：调用 LLM（带 function calling），返回 tool_calls 或最终答案
- tools_node：执行工具调用，结果追加到 messages
- 条件边：agent 返回 tool_calls → tools_node；返回最终答案 → END
- tools_node → agent_node（回环）
- loop_count 达到 MAX_REACT_LOOPS 时，agent_node 强制不传 tools，迫使 LLM 直接输出最终答案

流程：parse_log → classify → retrieve_cases(强制预检索) → agent ⇄ tools → END
"""

from langgraph.graph import END, StateGraph

from app.config import settings
from app.core.nodes.agent import agent_node
from app.core.nodes.classify import classify
from app.core.nodes.retrieve import retrieve_cases as retrieve_cases_impl
from app.core.nodes.tools import tools_node
from app.core.parsers import parse_crash_report
from app.core.state import AgentState


# ---------------------------------------------------------------------------
# 初始状态构建
# ---------------------------------------------------------------------------

def create_initial_state(raw_log_path: str) -> AgentState:
    """构建图的初始状态，所有字段填默认值。

    Args:
        raw_log_path: 原始崩溃日志文件路径。

    Returns:
        完整的 AgentState 字典，可直接传入 graph.invoke()。
    """
    return {
        "raw_log_path": raw_log_path,
        "parsed_log": None,
        "fault_category": None,
        "classify_reason": None,
        "messages": [],
        "retrieved_cases": [],
        "root_cause": None,
        "fix_suggestion": None,
        "loop_count": 0,
        "should_stop": False,
    }


# ---------------------------------------------------------------------------
# 节点定义（parse_log 直接实现；classify / retrieve_cases / agent / tools 为薄包装）
# ---------------------------------------------------------------------------

def parse_log(state: AgentState) -> dict:
    """读取原始崩溃日志并解析为 ParsedLog 结构。

    当前支持 Paper/Spigot/Purpur 的 crash report 格式。
    从 state["raw_log_path"] 读取文件，调用解析器提取结构化字段。

    Args:
        state: 当前图状态。

    Returns:
        包含 parsed_log 的部分状态更新。
    """
    parsed = parse_crash_report(state["raw_log_path"])
    return {"parsed_log": parsed}


def classify_node(state: AgentState) -> dict:
    """故障分类节点：调用 LLM 判断崩溃日志所属类别。

    真实实现见 app.core.nodes.classify.classify，此处为薄包装。

    Args:
        state: 当前图状态。

    Returns:
        包含 fault_category 和 classify_reason 的部分状态更新。
    """
    return classify(state)


def retrieve_cases_node(state: AgentState) -> dict:
    """相似案例检索节点：从向量库检索相似历史案例。

    真实实现见 app.core.nodes.retrieve.retrieve_cases，此处为薄包装。

    Args:
        state: 当前图状态。

    Returns:
        包含 retrieved_cases 的部分状态更新。
    """
    return retrieve_cases_impl(state)


# ---------------------------------------------------------------------------
# 条件边
# ---------------------------------------------------------------------------

def should_use_tools(state: AgentState) -> str:
    """判断 agent_node 返回后是否需要执行工具调用。

    检查 messages 最后一条 AI 消息是否含有 tool_calls。
    有 tool_calls → 进入 tools_node 执行工具；
    无 tool_calls → 视为最终答案，结束流程（root_cause / fix_suggestion 已在 agent_node 中解析写入）。

    Args:
        state: 当前图状态。

    Returns:
        下一个节点名 "tools_node"，或 END 表示终止。
    """
    # 即使 LLM 异常返回 tool_calls，达到最大轮次也强制终止，防止死循环
    if state["loop_count"] >= settings.MAX_REACT_LOOPS:
        return END

    messages = state["messages"]
    if not messages:
        return END

    last_msg = messages[-1]
    tool_calls = getattr(last_msg, "tool_calls", None) or []
    if tool_calls:
        return "tools_node"
    return END


# ---------------------------------------------------------------------------
# 图构建
# ---------------------------------------------------------------------------

builder = StateGraph(AgentState)

# 注册节点
builder.add_node("parse_log", parse_log)
builder.add_node("classify", classify_node)
builder.add_node("retrieve_cases", retrieve_cases_node)
builder.add_node("agent", agent_node)
builder.add_node("tools", tools_node)

# 入口与线性边
builder.set_entry_point("parse_log")
builder.add_edge("parse_log", "classify")
builder.add_edge("classify", "retrieve_cases")
builder.add_edge("retrieve_cases", "agent")

# ReAct 循环条件边：agent → 有 tool_calls 去 tools，无则 END
builder.add_conditional_edges(
    "agent",
    should_use_tools,
    {
        "tools_node": "tools",
        END: END,
    },
)

# tools 执行完回到 agent，形成回环
builder.add_edge("tools", "agent")

# 编译为可执行图
graph = builder.compile()
