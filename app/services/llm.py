"""
LLM 客户端统一封装。

基于 langchain-openai 的 ChatOpenAI，对接 DeepSeek（OpenAI 兼容格式）。
所有需要调用 LLM 的节点通过 get_llm() 获取客户端实例，
temperature 由调用方按需指定，不同阶段（分类/分析/审查）使用不同温度。
"""

from langchain_openai import ChatOpenAI

from app.config import settings


def get_llm(temperature: float = 0.1) -> ChatOpenAI:
    """创建并返回 LLM 客户端实例。

    ChatOpenAI 为轻量配置对象，创建成本可忽略，每次调用返回新实例，
    以便不同节点使用不同的 temperature。

    Args:
        temperature: 采样温度，越低越确定，越高越发散。
            分类/提取类任务建议 0.1，分析/审查类任务可适当提高。

    Returns:
        配置好的 ChatOpenAI 实例。
    """
    return ChatOpenAI(
        base_url=settings.LLM_BASE_URL,
        api_key=settings.LLM_API_KEY,
        model=settings.LLM_MODEL,
        temperature=temperature,
    )
