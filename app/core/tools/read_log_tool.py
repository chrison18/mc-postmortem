"""
原始日志片段读取工具。

react_agent 循环中 LLM 可调用的工具，用于按需读取原始崩溃日志的指定行范围。
原始日志全文不进 system prompt（可能几千行，易爆上下文窗口），
LLM 通过此工具按需读取片段，每次建议 50-100 行。

与 rag_tool.py 风格一致：@tool 装饰器，失败返回错误描述字符串，不抛异常。
"""

from langchain_core.tools import tool


@tool
def read_log_snippet(path: str, start_line: int, end_line: int) -> str:
    """读取原始日志文件的指定行范围，返回带行号的日志片段。

    当结构化日志信息不足、需要查看原始日志细节时调用此工具。
    每次建议读取 50-100 行，可多次调用覆盖不同区间。

    Args:
        path: 日志文件路径（使用 system prompt 中给出的原始日志路径）。
        start_line: 起始行号，从 1 开始；小于 1 时按 1 处理。
        end_line: 结束行号；超过文件总行数时读到末尾。

    Returns:
        带行号的日志片段，每行格式为 "行号 | 日志内容"。
        范围无效或读取失败时返回错误描述字符串。
    """
    # 边界处理：start_line 最小为 1
    if start_line < 1:
        start_line = 1

    # start_line > end_line 直接返回提示
    if start_line > end_line:
        return f"错误：start_line({start_line}) 不能大于 end_line({end_line})"

    try:
        # UTF-8 读取，errors="replace" 兜底乱码字符
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()
    except FileNotFoundError:
        return f"错误：日志文件不存在: {path}"
    except Exception as e:
        return f"错误：读取日志文件失败: {type(e).__name__}: {e}"

    total = len(all_lines)

    # start_line 超过总行数，返回提示
    if start_line > total:
        return f"错误：起始行 {start_line} 超过文件总行数 {total}"

    # end_line 超过总行数则截断到末尾
    if end_line > total:
        end_line = total

    # 拼接带行号的片段，行号从 1 开始
    snippet = []
    for i in range(start_line - 1, end_line):
        line_num = i + 1
        content = all_lines[i].rstrip("\n").rstrip("\r")
        snippet.append(f"{line_num} | {content}")

    return "\n".join(snippet)
