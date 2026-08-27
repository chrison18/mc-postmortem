"""
故障分类节点。

调用 LLM 对结构化崩溃日志进行故障分类，输出 7 类之一。
包含指数退避重试、JSON 解析、枚举校验，失败时降级为 unknown。
"""

import json
import re
import time

from langchain_core.messages import HumanMessage, SystemMessage

from app.core.state import AgentState
from app.prompts import build_classify_prompt
from app.services.llm import get_llm

# 合法分类枚举（与 prompt 中的 7 类保持一致）
VALID_CATEGORIES: set[str] = {
    "plugin_conflict",
    "version_mismatch",
    "plugin_bug",
    "config_error",
    "resource_issue",
    "core_issue",
    "unknown",
}

# LLM 调用重试配置
_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 1  # 指数退避基数（秒），实际为 1s / 2s


def classify(state: AgentState) -> dict:
    """故障分类节点：调用 LLM 判断崩溃日志所属类别。

    从 state 读取 parsed_log，拼接分类 prompt 调用 LLM，
    解析返回的 JSON 并做枚举校验，失败时降级为 unknown。

    Args:
        state: 当前图状态，需包含 parsed_log。

    Returns:
        包含 fault_category 和 classify_reason 的部分状态更新。
    """
    parsed_log = state.get("parsed_log")
    if not parsed_log:
        return {
            "fault_category": "unknown",
            "classify_reason": "parsed_log 为空，跳过分类",
        }

    # 构建消息：系统提示词 + 结构化日志作为用户输入
    messages = [
        SystemMessage(content=build_classify_prompt()),
        HumanMessage(
            content="请对以下崩溃日志进行分类：\n\n"
            + json.dumps(parsed_log, ensure_ascii=False, indent=2)
        ),
    ]

    llm = get_llm(temperature=0.1)

    # 指数退避重试调用 LLM
    response_text = _call_llm_with_retry(llm, messages)
    if response_text is None:
        return {
            "fault_category": "unknown",
            "classify_reason": "LLM 调用失败（已重试3次），降级为 unknown",
        }

    # 解析 JSON 响应
    category, reason = _parse_classify_response(response_text)

    # 枚举校验：非法类别降级 unknown
    if category not in VALID_CATEGORIES:
        reason = (
            f"LLM 返回非法类别 '{category}'，降级为 unknown。"
            f"原始 reason: {reason}"
        )
        category = "unknown"

    return {"fault_category": category, "classify_reason": reason}


def _call_llm_with_retry(llm, messages: list) -> str | None:
    """带指数退避重试的 LLM 调用。

    Args:
        llm: ChatOpenAI 实例。
        messages: 发送给 LLM 的消息列表。

    Returns:
        LLM 返回的文本内容，全部重试失败时返回 None。
    """
    for attempt in range(_MAX_RETRIES):
        try:
            response = llm.invoke(messages)
            return response.content
        except Exception as e:
            if attempt == _MAX_RETRIES - 1:
                return None
            time.sleep(_RETRY_BASE_DELAY * (2**attempt))
    return None


def _parse_classify_response(text: str) -> tuple[str, str]:
    """解析 LLM 返回的 JSON，提取 category 和 reason。

    自动处理 markdown 代码块包裹的情况。

    Args:
        text: LLM 返回的原始文本。

    Returns:
        (category, reason) 元组，解析失败时 category 为 "unknown"。
    """
    if not text or not text.strip():
        return "unknown", "LLM 返回为空"

    # 去除 markdown 代码块标记（```json ... ```）
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        data = json.loads(cleaned)
        category = str(data.get("category", "unknown"))
        reason = str(data.get("reason", ""))
        return category, reason
    except json.JSONDecodeError:
        return "unknown", f"JSON 解析失败，原始返回前200字: {text[:200]}"
