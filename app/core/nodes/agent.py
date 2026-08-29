"""
react_agent 的 agent 节点。

调用 LLM（带 function calling），根据当前上下文决定：
- 返回 tool_calls（需要调用工具追加检索）
- 返回最终答案（JSON 格式的 root_cause + fix_suggestion）

与 tools_node 形成回环：agent_node → tools_node → agent_node → ...
loop_count 达到 MAX_REACT_LOOPS 时，本轮强制不传 tools，迫使 LLM 直接输出最终答案。
"""

import time

from langchain_core.messages import AIMessage, SystemMessage

from app.config import settings
from app.core.state import AgentState
from app.core.tools.rag_tool import TOOLS
from app.prompts.react_agent import build_react_system_prompt
from app.services.llm import get_llm

# LLM 调用重试配置（与 classify 节点一致）
_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 1  # 指数退避基数（秒），实际为 1s / 2s


def agent_node(state: AgentState) -> dict:
    """agent 节点：调用 LLM（带 tools），返回工具调用或最终答案。

    从 state 读取上下文构建 system prompt，拼接历史 messages 后调用 LLM。
    最后一轮（loop_count 达到上限）时不传 tools，强制 LLM 直接输出结论。

    Args:
        state: 当前图状态。

    Returns:
        包含 messages（追加 AI 消息）和 loop_count（+1）的部分状态更新。
    """
    loop_count = state["loop_count"]
    is_last_round = loop_count >= settings.MAX_REACT_LOOPS - 1

    # 构建 system prompt
    system_prompt = build_react_system_prompt(state)
    if is_last_round:
        system_prompt += (
            "\n\n⚠️ 注意：这是最后一轮分析机会，"
            "请基于已有信息直接输出最终结论（JSON 格式），不要再调用任何工具。"
        )

    # 拼接消息：system prompt + 历史消息（AI / Tool 消息由 add_messages 累积）
    messages = [SystemMessage(content=system_prompt)] + list(state["messages"])

    # 创建 LLM 实例，最后一轮不绑定 tools
    llm = get_llm(temperature=0.1)
    if is_last_round:
        llm_with_tools = llm
    else:
        llm_with_tools = llm.bind_tools(TOOLS)

    # 带重试调用 LLM
    ai_msg = _call_llm_with_retry(llm_with_tools, messages)
    if ai_msg is None:
        # LLM 调用全部失败，降级为一条无 tool_calls 的 AI 消息，使流程走向 END
        ai_msg = AIMessage(
            content="LLM 调用失败（已重试3次），无法完成根因分析。",
        )

    return {"messages": [ai_msg], "loop_count": loop_count + 1}


def _call_llm_with_retry(llm, messages: list) -> AIMessage | None:
    """带指数退避重试的 LLM 调用。

    与 classify 节点的重试逻辑一致，但返回完整的 AIMessage（含 tool_calls），
    而不是仅返回 content 文本。

    Args:
        llm: 已绑定 tools（或未绑定）的 ChatOpenAI 实例。
        messages: 发送给 LLM 的消息列表。

    Returns:
        LLM 返回的 AIMessage，全部重试失败时返回 None。
    """
    for attempt in range(_MAX_RETRIES):
        try:
            response = llm.invoke(messages)
            return response
        except Exception:
            if attempt == _MAX_RETRIES - 1:
                return None
            time.sleep(_RETRY_BASE_DELAY * (2**attempt))
    return None
