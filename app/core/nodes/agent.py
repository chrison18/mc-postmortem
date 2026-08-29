"""
react_agent 的 agent 节点。

调用 LLM（带 function calling），根据当前上下文决定：
- 返回 tool_calls（需要调用工具追加检索）
- 返回最终答案（JSON 格式的 root_cause + fix_suggestion）

与 tools_node 形成回环：agent_node → tools_node → agent_node → ...
loop_count 达到 MAX_REACT_LOOPS 时，本轮强制不传 tools，迫使 LLM 直接输出最终答案。
当 AI 消息不含 tool_calls 时，视为最终答案，解析 JSON 写入 root_cause 和 fix_suggestion。
"""

import json
import re
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
    当返回的 AI 消息不含 tool_calls 时，解析最终答案 JSON，
    将 root_cause 和 fix_suggestion 写入状态。

    Args:
        state: 当前图状态。

    Returns:
        包含 messages（追加 AI 消息）、loop_count（+1）的部分状态更新。
        若为最终答案，还包含 root_cause 和 fix_suggestion。
    """
    loop_count = state["loop_count"]
    is_last_round = loop_count >= settings.MAX_REACT_LOOPS - 1

    # 构建 system prompt
    system_prompt = build_react_system_prompt(state)
    if is_last_round:
        system_prompt += (
            "\n\n注意：这是最后一轮分析机会，"
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

    result: dict = {"messages": [ai_msg], "loop_count": loop_count + 1}

    # 没有 tool_calls 说明是最终答案，解析 JSON 提取 root_cause 和 fix_suggestion
    tool_calls = getattr(ai_msg, "tool_calls", None) or []
    if not tool_calls:
        root_cause, fix_suggestion = _parse_final_answer(ai_msg.content)
        result["root_cause"] = root_cause
        result["fix_suggestion"] = fix_suggestion

    return result


def _parse_final_answer(content: str) -> tuple[str, str | None]:
    """解析 LLM 最终答案中的 JSON，提取 root_cause 和 fix_suggestion。

    支持三种格式：
    1. 纯 JSON 文本
    2. markdown 代码块包裹的 JSON（```json ... ```）
    3. 混合文本（前面有分析文字，后面包含 JSON 对象）——用正则提取 JSON

    解析失败时降级：全文当 root_cause，fix_suggestion 为 None。

    Args:
        content: LLM 返回的文本内容。

    Returns:
        (root_cause, fix_suggestion) 元组。
    """
    if not content or not content.strip():
        return "LLM 返回为空", None

    cleaned = content.strip()

    # 格式2：去除 markdown 代码块标记
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    # 尝试直接解析（格式1：纯 JSON）
    result = _try_extract_json(cleaned)
    if result is not None:
        return result

    # 格式3：从混合文本中用正则提取 JSON 对象（第一个 { 到最后一个 }）
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if match:
        result = _try_extract_json(match.group())
        if result is not None:
            return result

    # 降级：全文当 root_cause
    return content.strip(), None


def _try_extract_json(text: str) -> tuple[str, str | None] | None:
    """尝试解析 JSON 文本，提取 root_cause 和 fix_suggestion。

    Args:
        text: 待解析的文本。

    Returns:
        (root_cause, fix_suggestion) 元组，解析失败或无 root_cause 时返回 None。
    """
    try:
        data = json.loads(text)
        root_cause = str(data.get("root_cause", "")).strip()
        fix_suggestion = data.get("fix_suggestion")
        if fix_suggestion is not None:
            fix_suggestion = str(fix_suggestion).strip() or None
        if root_cause:
            return root_cause, fix_suggestion
    except json.JSONDecodeError:
        pass
    return None


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
