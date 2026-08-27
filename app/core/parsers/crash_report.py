"""
Paper / Spigot / Purpur 崩溃报告解析器。

解析 crash-reports/crash-*.txt 格式的崩溃报告，提取结构化字段填充 ParsedLog。
MVP 阶段仅支持 Bukkit 系服务端（Paper/Spigot/Purpur），不支持 Forge/Fabric。
"""

import re
from pathlib import Path

from app.core.state import ParsedLog

# crash report 文件头标记
_CRASH_REPORT_HEADER = "---- Minecraft Crash Report ----"

# 分隔线：详细堆栈从此开始，之前的是主异常信息
_DETAILED_WALKTHROUGH = "A detailed walkthrough of the error"


def parse_crash_report(file_path: str) -> ParsedLog:
    """解析崩溃报告文件为 ParsedLog 结构。

    Args:
        file_path: 崩溃报告文件的绝对或相对路径。

    Returns:
        结构化的 ParsedLog 字典。

    Raises:
        FileNotFoundError: 文件不存在。
        ValueError: 文件不是有效的崩溃报告格式。
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"崩溃报告不存在: {file_path}")

    content = _read_file(path)
    lines = content.splitlines()

    if not _is_crash_report(lines):
        raise ValueError(f"文件不是有效的崩溃报告格式（缺少文件头标记）: {file_path}")

    return {
        "server_type": _extract_server_type(lines),
        "server_version": _extract_minecraft_version(lines),
        "java_version": _extract_java_version(lines),
        "exception_type": _extract_exception_type(lines),
        "exception_message": _extract_exception_message(lines),
        "crash_thread": _extract_crash_thread(lines),
        "caused_by_chain": _extract_caused_by_chain(lines),
        "key_stack_frames": _extract_stack_frames(lines),
        "plugins": _extract_plugins(lines),
        # suspected_plugin 由后续分类/分析节点推断，解析器不做推断
        "suspected_plugin": None,
        "crash_time": _extract_crash_time(lines),
        "raw_log_path": str(path.resolve()),
    }


# ---------------------------------------------------------------------------
# 文件读取与格式判断
# ---------------------------------------------------------------------------

def _read_file(path: Path) -> str:
    """读取文件内容，自动尝试多种编码。

    crash report 在不同系统上编码可能不同，优先 utf-8，回退 gbk / latin-1。

    Args:
        path: 文件路径。

    Returns:
        文件文本内容。
    """
    for encoding in ("utf-8", "gbk", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except (UnicodeDecodeError, LookupError):
            continue
    # latin-1 不会失败，理论上到不了这里
    return path.read_text(encoding="latin-1")


def _is_crash_report(lines: list[str]) -> bool:
    """判断文件内容是否为崩溃报告格式。

    Args:
        lines: 文件行列表。

    Returns:
        前 5 行内包含文件头标记则返回 True。
    """
    return any(_CRASH_REPORT_HEADER in line for line in lines[:5])


# ---------------------------------------------------------------------------
# 主异常信息提取（Description 段，分隔线之前）
# ---------------------------------------------------------------------------

def _get_main_section(lines: list[str]) -> list[str]:
    """获取主异常段（从 Description 到详细堆栈分隔线之间的行）。

    Args:
        lines: 全文行列表。

    Returns:
        主异常段的行列表，找不到时返回空列表。
    """
    start = None
    for i, line in enumerate(lines):
        if line.strip().startswith("Description:"):
            start = i
            break
    if start is None:
        return []

    end = len(lines)
    for i in range(start, len(lines)):
        if _DETAILED_WALKTHROUGH in lines[i]:
            end = i
            break
    return lines[start:end]


def _extract_exception_type(lines: list[str]) -> str:
    """提取异常类型（如 java.lang.NullPointerException）。

    Args:
        lines: 全文行列表。

    Returns:
        异常类型全限定名，提取失败返回空字符串。
    """
    section = _get_main_section(lines)
    for line in section:
        stripped = line.strip()
        # 匹配 "java.lang.XxxException: message" 或 "java.lang.XxxError: message"
        m = re.match(r"^(\S+(?:Exception|Error|Throwable))\s*:", stripped)
        if m:
            return m.group(1)
    return ""


def _extract_exception_message(lines: list[str]) -> str:
    """提取异常消息（冒号后的部分）。

    Args:
        lines: 全文行列表。

    Returns:
        异常消息文本，提取失败返回空字符串。
    """
    section = _get_main_section(lines)
    for line in section:
        stripped = line.strip()
        m = re.match(r"^\S+(?:Exception|Error|Throwable)\s*:\s*(.*)$", stripped)
        if m:
            return m.group(1).strip()
    return ""


def _extract_stack_frames(lines: list[str]) -> list[str]:
    """提取全量堆栈帧（主异常段内所有 at 行）。

    保留所有 at 行，不过滤 JDK 内部帧，为后续分析保留完整信息。

    Args:
        lines: 全文行列表。

    Returns:
        堆栈帧字符串列表，每项为 "at 类名.方法(源文件:行号)" 格式。
    """
    section = _get_main_section(lines)
    frames = []
    for line in section:
        stripped = line.strip()
        if stripped.startswith("at "):
            frames.append(stripped)
    return frames


def _extract_caused_by_chain(lines: list[str]) -> list[str]:
    """提取 Caused by 链（从外到内）。

    Args:
        lines: 全文行列表。

    Returns:
        Caused by 行列表，每项为 "Caused by: 异常类型: 消息" 格式。
    """
    section = _get_main_section(lines)
    chain = []
    for line in section:
        stripped = line.strip()
        if stripped.startswith("Caused by:"):
            chain.append(stripped)
    return chain


# ---------------------------------------------------------------------------
# System Details 段提取
# ---------------------------------------------------------------------------

def _get_system_details(lines: list[str]) -> list[str]:
    """获取 -- System Details -- 段的内容行。

    Args:
        lines: 全文行列表。

    Returns:
        System Details 段的行列表，找不到返回空列表。
    """
    start = None
    for i, line in enumerate(lines):
        if line.strip().startswith("-- System Details --"):
            start = i + 1
            break
    if start is None:
        return []

    # System Details 通常是最后一段，取到文件末尾
    return lines[start:]


def _extract_minecraft_version(lines: list[str]) -> str:
    """提取 Minecraft 版本号。

    Args:
        lines: 全文行列表。

    Returns:
        版本号如 "1.20.4"，提取失败返回空字符串。
    """
    details = _get_system_details(lines)
    for line in details:
        m = re.match(r"\s*Minecraft Version:\s*(.+)", line)
        if m:
            return m.group(1).strip()
    return ""


def _extract_java_version(lines: list[str]) -> str:
    """提取 Java 版本。

    Args:
        lines: 全文行列表。

    Returns:
        Java 版本如 "17.0.1"，提取失败返回空字符串。
    """
    details = _get_system_details(lines)
    for line in details:
        m = re.match(r"\s*Java Version:\s*([^,]+)", line)
        if m:
            return m.group(1).strip()
    return ""


def _extract_server_type(lines: list[str]) -> str:
    """提取服务端类型（Paper / Spigot / Purpur）。

    优先从 "Server Running:" 行提取（Paper/Purpur 特有），
    缺失时默认 "spigot"（Spigot 无此字段）。

    Args:
        lines: 全文行列表。

    Returns:
        服务端类型小写字符串，如 "paper" / "spigot" / "purpur"。
    """
    details = _get_system_details(lines)
    for line in details:
        m = re.match(r"\s*Server Running:\s*(\w+)", line)
        if m:
            return m.group(1).lower()
    return "spigot"


def _extract_crash_time(lines: list[str]) -> str:
    """提取崩溃时间。

    Args:
        lines: 全文行列表。

    Returns:
        时间字符串如 "2024-01-15 10:30:45"，提取失败返回空字符串。
    """
    for line in lines[:20]:
        m = re.match(r"\s*Time:\s*(.+)", line)
        if m:
            return m.group(1).strip()
    return ""


def _extract_crash_thread(lines: list[str]) -> str:
    """提取崩溃发生的线程名。

    优先从 -- Head -- 段的 "Thread:" 行提取。

    Args:
        lines: 全文行列表。

    Returns:
        线程名如 "Server thread"，提取失败返回空字符串。
    """
    for line in lines:
        m = re.match(r"\s*Thread:\s*(.+)", line)
        if m:
            return m.group(1).strip()
    return ""


def _extract_plugins(lines: list[str]) -> list[dict]:
    """提取插件列表，解析为 name / version / url 字典。

    解析 "Plugins: {EssentialsX v2.20.1 (https://...), Vault v1.7.3, ...}" 格式。

    Args:
        lines: 全文行列表。

    Returns:
        插件字典列表，每项含 name / version / url 字段。
    """
    details = _get_system_details(lines)
    plugins_line = None
    for line in details:
        if line.strip().startswith("Plugins:"):
            plugins_line = line
            break
    if not plugins_line:
        return []

    # 去掉 "Plugins:" 前缀和外层大括号
    inner = plugins_line.split(":", 1)[1].strip().strip("{}")
    if not inner:
        return []

    plugins = []
    for part in inner.split(","):
        part = part.strip()
        if not part:
            continue
        # 匹配 "Name vVersion (url)" 或 "Name vVersion" 或 "Name version"
        m = re.match(r"^(\S+)\s+v?(\S+?)(?:\s+\(([^)]+)\))?$", part)
        if m:
            plugins.append({
                "name": m.group(1),
                "version": m.group(2),
                "url": m.group(3) or "",
            })
        else:
            # 解析失败时整段作为 name，保证不丢信息
            plugins.append({"name": part, "version": "", "url": ""})
    return plugins
