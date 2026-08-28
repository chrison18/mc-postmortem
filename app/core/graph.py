"""
LangGraph 主流程定义。

设计为混合 RAG 模式：
- retrieve_cases 节点在 react 循环前强制预检索一次，保证 LLM 至少获得一次历史案例上下文，
  避免 LLM 自认为不需要检索就直接下结论。
- react_agent 循环内后续会接入 RAG tool（及其他工具），支持 LLM 按需追加检索。

当前状态：parse_log / classify / retrieve_cases 均已实现真实逻辑，仅 react_agent 仍为 stub。
react_agent 后续将拆分为 agent_node + tools_node 双节点（见 docs/TECHNICAL_DESIGN.md）。

流程：parse_log → classify → retrieve_cases(强制预检索) → react_agent(循环, 内含 RAG tool) → END
"""

from langgraph.graph import END, StateGraph

from app.config import settings
from app.core.nodes.classify import classify
from app.core.nodes.retrieve import retrieve_cases as retrieve_cases_impl
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
        "loop_count": 0,
        "should_stop": False,
    }


# ---------------------------------------------------------------------------
# 节点定义（parse_log 直接实现；classify_node / retrieve_cases_node 为薄包装；react_agent 为 stub）
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


def react_agent(state: AgentState) -> dict:
    """【stub】ReAct 循环节点，分析根因并决定是否继续。

    后续替换为真实 LLM ReAct 逻辑：
    - 接入 tools（含 RAG tool，支持追加检索历史案例）
    - 通过 messages 字段维护对话历史
    - LLM 自主决定调用工具或输出最终根因

    当前每次调用 loop_count + 1，达到上限后由条件边终止。

    Args:
        state: 当前图状态。

    Returns:
        包含 root_cause 和 loop_count 的部分状态更新。
    """
    return {
        "root_cause": "待实现：ReAct 根因分析",
        "loop_count": state["loop_count"] + 1,
    }


# ---------------------------------------------------------------------------
# 条件边
# ---------------------------------------------------------------------------

def should_continue(state: AgentState) -> str:
    """判断 ReAct 循环是否继续。

    loop_count 未达上限则回到 react_agent，否则结束。

    Args:
        state: 当前图状态。

    Returns:
        下一个节点名 "react_agent"，或 END 表示终止。
    """
    if state["loop_count"] < settings.MAX_REACT_LOOPS:
        return "react_agent"
    return END


# ---------------------------------------------------------------------------
# 图构建
# ---------------------------------------------------------------------------

builder = StateGraph(AgentState)

# 注册节点
builder.add_node("parse_log", parse_log)
builder.add_node("classify", classify_node)
builder.add_node("retrieve_cases", retrieve_cases_node)
builder.add_node("react_agent", react_agent)

# 入口与线性边
builder.set_entry_point("parse_log")
builder.add_edge("parse_log", "classify")
builder.add_edge("classify", "retrieve_cases")
builder.add_edge("retrieve_cases", "react_agent")

# ReAct 循环条件边
builder.add_conditional_edges("react_agent", should_continue)

# 编译为可执行图
graph = builder.compile()
