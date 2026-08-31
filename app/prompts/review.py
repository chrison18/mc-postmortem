"""
审查 Agent 的 system prompt 模块。

独立审查主 Agent 的分析结论，检查证据支撑、修复可操作性、线索遗漏、逻辑自洽、置信度合理性。
不读主 Agent 的 messages 历史，只看结构化日志 + 最终结论 + 检索案例，保证独立性。
"""


def build_review_system_prompt(state: dict) -> str:
    """构建审查 Agent 的 system prompt。

    从 state 读取 parsed_log / root_cause / fix_suggestion / summary / confidence / retrieved_cases。

    Args:
        state: 当前 AgentState 字典。

    Returns:
        完整的审查 system prompt。
    """
    parsed = state.get("parsed_log") or {}
    retrieved = state.get("retrieved_cases", [])

    # 格式化结构化日志（只取关键字段，不拼全文）
    log_lines = []
    log_lines.append(f"异常类型: {parsed.get('exception_type', 'unknown')}")
    log_lines.append(f"异常消息: {parsed.get('exception_message', '')}")
    caused_by = parsed.get("caused_by_chain", [])
    if caused_by:
        log_lines.append("Caused by 链:")
        for c in caused_by:
            log_lines.append(f"  {c}")
    plugins = parsed.get("plugins", [])
    if plugins:
        log_lines.append("插件列表:")
        for p in plugins:
            log_lines.append(f"  - {p.get('name', '')} {p.get('version', '')}".strip())
    frames = parsed.get("key_stack_frames", [])
    if frames:
        log_lines.append("堆栈帧（前10条）:")
        for f in frames[:10]:
            log_lines.append(f"  {f}")
    log_text = "\n".join(log_lines)

    # 格式化检索案例（最多3条）
    case_lines = []
    for i, case in enumerate(retrieved[:3], 1):
        fix = case.get("fix_solution", "")[:100]
        case_lines.append(f"案例{i}: {case.get('exception_type', '')} - {fix}")
    case_text = "\n".join(case_lines) if case_lines else "（无检索案例）"

    prompt = f"""你是一名资深 Minecraft 服务端事故审查专家。
你的任务是独立审查下方的崩溃分析报告，挑出问题，而不是复述结论。

## 审查维度（逐项检查）
1. 证据支撑：根因是否有日志堆栈或检索案例支撑，还是纯猜测？
2. 修复可操作性：修复建议是具体步骤（含版本号/文件名），还是泛泛而谈？
3. 线索遗漏：Caused by 链、关键堆栈帧、插件列表中有没有被忽略的线索？
4. 逻辑自洽：根因和修复建议是否对应？分类和结论是否矛盾？
5. 置信度合理性：证据不足但标了 high，或证据充分但标了 low？

## 崩溃日志（结构化摘要）
{log_text}

## 检索到的参考案例
{case_text}

## 待审查的分析报告
- 一句话总结：{state.get('summary', '（无）')}
- 根因分析：{state.get('root_cause', '（无）')}
- 修复建议：{state.get('fix_suggestion', '（无）')}
- 置信度：{state.get('confidence', '（无）')}

## 输出格式
严格输出 JSON（不要 markdown 代码块，不要分析文字）：
{{
  "passed": true 或 false,
  "issues": ["问题1", "问题2"],
  "suggestion": "整体修正建议（一句话）"
}}

如果结论质量合格、没有明显问题，passed=true，issues=[]。
如果存在问题，passed=false，issues 列出具体问题（每条一句话，指向具体证据缺失或逻辑错误）。
"""
    return prompt
