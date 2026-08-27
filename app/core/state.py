"""
LangGraph 状态定义模块。

使用 TypedDict 定义图中传递的状态结构，保持简单轻量。
messages 字段使用 Annotated + add_messages reducer，是 ReAct 循环的核心。
"""

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class ParsedLog(TypedDict):
    """结构化后的崩溃日志，由日志解析节点填充。"""

    # 服务端类型，如 paper / spigot / forge / fabric / vanilla
    server_type: str
    # 服务端版本号，如 1.20.1
    server_version: str
    # Java 版本，如 17.0.1
    java_version: str
    # 异常类型，如 java.lang.NullPointerException
    exception_type: str
    # 异常消息
    exception_message: str
    # 崩溃发生的线程名
    crash_thread: str
    # caused by 链，从外到内逐层记录
    caused_by_chain: list[str]
    # 全量堆栈帧（保留所有 at 行，不过滤 JDK 内部帧，为后续分析留全量信息）
    key_stack_frames: list[str]
    # 崩溃时加载的插件列表，每项含 name / version / author
    plugins: list[dict]
    # 疑似导致崩溃的插件名，无法判断时为 None
    suspected_plugin: str | None
    # 崩溃发生时间
    crash_time: str
    # 原始日志文件路径（只存路径，不存全文）
    raw_log_path: str


class AgentState(TypedDict):
    """LangGraph 中各节点共享的状态。"""

    # 原始崩溃日志文件路径（入口参数，只存路径不存全文）
    raw_log_path: str
    # 结构化解析结果，初始为 None，由解析节点填充
    parsed_log: ParsedLog | None
    # 故障分类结果，如 plugin_conflict / version_mismatch 等
    fault_category: str | None
    # ReAct 消息列表，使用 add_messages reducer 自动合并  即上下文窗口 context window
    messages: Annotated[list, add_messages]
    # 从向量库检索到的历史相似案例
    retrieved_cases: list[dict]
    # 最终根因分析结论
    root_cause: str | None
    # 当前 ReAct 循环次数，用于判断是否达到 MAX_REACT_LOOPS
    loop_count: int
    # 是否终止循环的标志位
    should_stop: bool
