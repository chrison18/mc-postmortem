"""
审查节点：独立审查主 Agent 的分析结论。

调用 LLM（temperature=0.3，不带 tools），输出 {passed, issues, suggestion}。
不读主 Agent 的 messages 历史，只看结构化日志 + 结论 + 检索案例，保证独立性。

审查通过：verified=True，review_opinion="[]"
审查不通过：verified=False，review_opinion=json.dumps(issues)，审查意见作为 Human 消息加入 messages，
            loop_count 重置为 MAX_REACT_LOOPS-2（最多 2 轮修正：可选择 1 轮工具 + 1 轮输出，或直接 1 轮输出）
审查 LLM 失败：fail-open，视为通过，verified=True，review_opinion="审查服务不可用，跳过"
"""

import json
import re
import time

from langchain_core.messages import HumanMessage, SystemMessage

from app.config import settings
from app.core.state import AgentState
from app.prompts.review import build_review_system_prompt
from app.services.llm import get_llm

_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 1


def review_node(state: AgentState) -> dict:
    """审查节点：独立审查主 Agent 结论，决定通过或打回修正。

    Args:
        state: 当前图状态。

    Returns:
        包含 review_count(+1)、review_opinion、verified 的状态更新。
        若不通过，还包含 messages（审查意见 HumanMessage）和 loop_count（重置）。
    """
    review_count = state.get("review_count", 0)

    # 调审查 LLM（温度 0.3，不带 tools）
    system_prompt = build_review_system_prompt(state)
    llm = get_llm(temperature=0.3)
    ai_msg = _call_llm_with_retry(llm, [SystemMessage(content=system_prompt)])

    result: dict = {"review_count": review_count + 1}

    if ai_msg is None:
        # fail-open：审查服务不可用，视为通过
        result["verified"] = True
        result["review_opinion"] = "审查服务不可用，跳过"
        return result

    # 解析审查结果
    passed, issues, suggestion = _parse_review_result(ai_msg.content)

    if passed:
        result["verified"] = True
        result["review_opinion"] = "[]"
    else:
        # 审查不通过：打回修正
        result["verified"] = False
        result["review_opinion"] = json.dumps(issues, ensure_ascii=False)
        # 审查意见作为 Human 消息加入 messages，主 Agent 会看到并修正
        feedback = (
            f"【独立审查反馈】你的分析存在以下问题：\n"
            + "\n".join(f"- {issue}" for issue in issues)
            + f"\n\n审查建议：{suggestion}\n"
            + "请基于原始日志和检索案例修正你的结论，输出完整的 JSON（summary/root_cause/fix_suggestion/confidence）。"
        )
        result["messages"] = [HumanMessage(content=feedback)]
        # 重置 loop_count，最多 2 轮修正（1 轮工具 + 1 轮输出，或直接 1 轮输出）
        result["loop_count"] = settings.MAX_REACT_LOOPS - 2

    return result


def _strip_double_braces(text: str) -> str:
    """如果文本被双层大括号 {{...}} 包裹，剥离一层变为 {...}。

    与 agent.py 的 _strip_double_braces 逻辑一致，审查 LLM 温度更高，输出更不可控。
    """
    text = text.strip()
    if text.startswith("{{") and text.endswith("}}"):
        return text[1:-1]
    return text


def _parse_review_result(content: str) -> tuple[bool, list[str], str]:
    """解析审查 LLM 的 JSON 输出。

    支持纯 JSON、markdown 代码块、双层大括号、混合文本。
    解析失败时视为通过（fail-open，不阻塞主流程）。

    Args:
        content: LLM 返回的文本。

    Returns:
        (passed, issues, suggestion) 元组。
    """
    if not content or not content.strip():
        return True, [], ""

    cleaned = content.strip()

    # 去 markdown 代码块
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    # 尝试直接解析（先剥离双层大括号）
    result = _try_parse_json(_strip_double_braces(cleaned))
    if result is not None:
        return result

    # 从混合文本中提取 JSON 块
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if match:
        result = _try_parse_json(_strip_double_braces(match.group()))
        if result is not None:
            return result

    # 解析失败视为通过（fail-open）
    return True, [], ""


def _try_parse_json(text: str) -> tuple[bool, list[str], str] | None:
    """尝试解析 JSON 文本，提取 passed/issues/suggestion。

    Args:
        text: 待解析文本。

    Returns:
        (passed, issues, suggestion) 元组，解析失败返回 None。
    """
    try:
        data = json.loads(text)
        passed = bool(data.get("passed", False))
        issues = data.get("issues", [])
        if not isinstance(issues, list):
            issues = []
        issues = [str(i).strip() for i in issues if str(i).strip()]
        suggestion = str(data.get("suggestion", "")).strip()
        return passed, issues, suggestion
    except json.JSONDecodeError:
        return None


def _call_llm_with_retry(llm, messages: list):
    """带指数退避重试的 LLM 调用。"""
    for attempt in range(_MAX_RETRIES):
        try:
            return llm.invoke(messages)
        except Exception:
            if attempt == _MAX_RETRIES - 1:
                return None
            time.sleep(_RETRY_BASE_DELAY * (2**attempt))
    return None
