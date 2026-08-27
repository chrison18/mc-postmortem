"""
Paper / Spigot / Purpur 崩溃日志解析器。

支持多种输入格式：
- 完整 crash report（crash-reports/crash-*.txt）
- crash report 片段（issue 中粘贴的部分日志）
- latest.log 格式片段（带 [时间 线程/级别] 前缀）

解析策略：全文特征搜索，能提取多少提取多少，缺失字段留空，绝不瞎填默认值。
无法识别为日志的文本也返回结构，原始全文存入 raw_content 供后续 LLM 提取。
"""

import re
from pathlib import Path

from app.core.state import ParsedLog

# latest.log 行前缀：[HH:MM:SS] [ThreadName/LEVEL]:
_LOG_PREFIX_RE = re.compile(r"^\[\d{2}:\d{2}:\d{2}\]\s+\[[^\]]+\]:\s*")

# crash report 分隔线
_SYSTEM_DETAILS_MARKER = "-- System Details --"


def parse_crash_report(file_path: str) -> ParsedLog:
    """解析崩溃日志文件为 ParsedLog 结构。

    不严格要求文件格式，尽力提取各字段。缺失字段留空字符串/空列表，
    原始全文存入 raw_content，确保后续 LLM 不会因字段缺失而瞎编。

    Args:
        file_path: 崩溃日志文件的绝对或相对路径。

    Returns:
        结构化的 ParsedLog 字典。

    Raises:
        FileNotFoundError: 文件不存在。
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"崩溃日志不存在: {file_path}")

    content = _read_file(path)
    raw_lines = content.splitlines()
    # 逐行剥离 latest.log 前缀，得到统一格式的行
    lines = [_strip_log_prefix(line) for line in raw_lines]

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
        "raw_content": content,
    }


# ---------------------------------------------------------------------------
# 文件读取与行预处理
# ---------------------------------------------------------------------------

def _read_file(path: Path) -> str:
    """读取文件内容，自动尝试多种编码。

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
    return path.read_text(encoding="latin-1")


def _strip_log_prefix(line: str) -> str:
    """剥离 latest.log 格式的行前缀 [时间 线程/级别]: 。

    非 latest.log 格式的行原样返回。

    Args:
        line: 原始行。

    Returns:
        剥离前缀后的行。
    """
    return _LOG_PREFIX_RE.sub("", line)


# ---------------------------------------------------------------------------
# 异常信息提取（全文特征搜索）
# ---------------------------------------------------------------------------

def _extract_exception_type(lines: list[str]) -> str:
    """提取异常类型（如 java.lang.NullPointerException）。

    优先从非 Caused by 行提取（支持行首和行内匹配），
    找不到时退化到从第一条 Caused by 行提取。

    Args:
        lines: 预处理后的行列表。

    Returns:
        异常类型全限定名，提取失败返回空字符串。
    """
    # 第一轮：非 Caused by 行，行内搜索异常类型
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("Caused by"):
            continue
        m = re.search(r"(\S+(?:Exception|Error|Throwable))\s*:", stripped)
        if m:
            return m.group(1)
    # 第二轮：退化到 Caused by 行
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("Caused by:"):
            m = re.search(r"Caused by:\s*(\S+(?:Exception|Error|Throwable))\s*:", stripped)
            if m:
                return m.group(1)
    return ""


def _extract_exception_message(lines: list[str]) -> str:
    """提取异常消息（异常类型冒号后的部分）。

    与 exception_type 采用相同的搜索策略。

    Args:
        lines: 预处理后的行列表。

    Returns:
        异常消息文本，提取失败返回空字符串。
    """
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("Caused by"):
            continue
        m = re.search(r"\S+(?:Exception|Error|Throwable)\s*:\s*(.+)$", stripped)
        if m:
            return m.group(1).strip()
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("Caused by:"):
            m = re.search(
                r"Caused by:\s*\S+(?:Exception|Error|Throwable)\s*:\s*(.+)$",
                stripped,
            )
            if m:
                return m.group(1).strip()
    return ""


def _extract_stack_frames(lines: list[str]) -> list[str]:
    """提取全量堆栈帧（全文所有 at 开头的行）。

    保留所有 at 行，不过滤 JDK 内部帧，为后续分析保留完整信息。

    Args:
        lines: 预处理后的行列表。

    Returns:
        堆栈帧字符串列表。
    """
    frames = []
    for line in lines:
        stripped = line.strip()
        # 堆栈帧特征：以 at 开头，且包含括号（源文件:行号）
        if stripped.startswith("at ") and "(" in stripped and ")" in stripped:
            frames.append(stripped)
    return frames


def _extract_caused_by_chain(lines: list[str]) -> list[str]:
    """提取 Caused by 链（全文所有 Caused by: 行）。

    Args:
        lines: 预处理后的行列表。

    Returns:
        Caused by 行列表，从外到内。
    """
    chain = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("Caused by:"):
            chain.append(stripped)
    return chain


# ---------------------------------------------------------------------------
# System Details 段提取（有则提取，无则留空）
# ---------------------------------------------------------------------------

def _get_system_details(lines: list[str]) -> list[str]:
    """获取 -- System Details -- 段的内容行。

    Args:
        lines: 预处理后的行列表。

    Returns:
        System Details 段的行列表，找不到返回空列表。
    """
    start = None
    for i, line in enumerate(lines):
        if line.strip().startswith(_SYSTEM_DETAILS_MARKER):
            start = i + 1
            break
    if start is None:
        return []
    return lines[start:]


def _extract_minecraft_version(lines: list[str]) -> str:
    """提取 Minecraft 版本号。

    Args:
        lines: 预处理后的行列表。

    Returns:
        版本号，提取失败返回空字符串。
    """
    for line in _get_system_details(lines):
        m = re.match(r"\s*Minecraft Version:\s*(.+)", line)
        if m:
            return m.group(1).strip()
    return ""


def _extract_java_version(lines: list[str]) -> str:
    """提取 Java 版本。

    Args:
        lines: 预处理后的行列表。

    Returns:
        Java 版本，提取失败返回空字符串。
    """
    for line in _get_system_details(lines):
        m = re.match(r"\s*Java Version:\s*([^,]+)", line)
        if m:
            return m.group(1).strip()
    return ""


def _extract_server_type(lines: list[str]) -> str:
    """提取服务端类型（Paper / Spigot / Purpur）。

    从 "Server Running:" 行提取，缺失时返回 "unknown"（不瞎猜）。

    Args:
        lines: 预处理后的行列表。

    Returns:
        服务端类型小写字符串，或 "unknown"。
    """
    for line in _get_system_details(lines):
        m = re.match(r"\s*Server Running:\s*(\w+)", line)
        if m:
            return m.group(1).lower()
    return "unknown"


def _extract_crash_time(lines: list[str]) -> str:
    """提取崩溃时间。

    Args:
        lines: 预处理后的行列表。

    Returns:
        时间字符串，提取失败返回空字符串。
    """
    for line in lines[:30]:
        m = re.match(r"\s*Time:\s*(.+)", line)
        if m:
            return m.group(1).strip()
    return ""


def _extract_crash_thread(lines: list[str]) -> str:
    """提取崩溃发生的线程名。

    Args:
        lines: 预处理后的行列表。

    Returns:
        线程名，提取失败返回空字符串。
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
        lines: 预处理后的行列表。

    Returns:
        插件字典列表，每项含 name / version / url 字段。
    """
    plugins_line = None
    for line in _get_system_details(lines):
        if line.strip().startswith("Plugins:"):
            plugins_line = line
            break
    if not plugins_line:
        return []

    inner = plugins_line.split(":", 1)[1].strip().strip("{}")
    if not inner:
        return []

    plugins = []
    for part in inner.split(","):
        part = part.strip()
        if not part:
            continue
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
