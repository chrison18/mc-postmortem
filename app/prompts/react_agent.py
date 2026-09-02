"""
react_agent 的 system prompt 模块。

Prompt 独立成模块，绝对不写在节点或 graph 里。
build_react_system_prompt() 从 state 读取上下文，拼接成完整 system prompt。

注入内容：
- parsed_log：结构化崩溃日志
- fault_category + classify_reason：前置分类结果（不主动传会被静默丢弃）
- retrieved_cases：预检索相似案例（强制预检索的结果）
- RAG tool 使用说明：告知向量库的 embedding 文本格式，指导 LLM 有方向地自由查询
- read_log_snippet 工具说明：原始日志全文不进 prompt，LLM 按需读取指定行范围
- 边界约束：只能基于日志原文和检索案例下结论，不编造，不确定就标注
"""

import json


def _format_parsed_log(parsed_log: dict | None, raw_log_path: str = "") -> str:
    """将结构化日志格式化为可读文本。

    raw_content 全文不进 prompt（可能几千行，易爆上下文窗口），
    只保留原始日志路径，由 LLM 按需调用 read_log_snippet 工具读取片段。

    Args:
        parsed_log: ParsedLog 字典，为 None 时返回提示。
        raw_log_path: 原始日志文件路径，供 LLM 按需读取。

    Returns:
        格式化后的日志文本。
    """
    if not parsed_log:
        return "（日志解析失败，无结构化字段，可调用 read_log_snippet 工具读取原始日志）"

    lines = []
    lines.append(f"服务端类型: {parsed_log.get('server_type', 'unknown')}")
    lines.append(f"服务端版本: {parsed_log.get('server_version', '')}")
    lines.append(f"Java 版本: {parsed_log.get('java_version', '')}")
    lines.append(f"异常类型: {parsed_log.get('exception_type', '')}")
    lines.append(f"异常消息: {parsed_log.get('exception_message', '')}")
    lines.append(f"崩溃线程: {parsed_log.get('crash_thread', '')}")

    caused_by = parsed_log.get("caused_by_chain", [])
    if caused_by:
        lines.append("Caused by 链:")
        for c in caused_by:
            lines.append(f"  {c}")

    plugins = parsed_log.get("plugins", [])
    if plugins:
        lines.append("已加载插件:")
        for p in plugins:
            name = p.get("name", "")
            version = p.get("version", "")
            lines.append(f"  - {name} {version}".strip())

    frames = parsed_log.get("key_stack_frames", [])
    if frames:
        lines.append(f"堆栈帧（共 {len(frames)} 条，展示前 20 条）:")
        for f in frames[:20]:
            lines.append(f"  {f}")

    # raw_content 全文不进 prompt，只保留路径，由 read_log_snippet 工具按需读取
    if raw_log_path:
        lines.append(
            f"原始日志路径: {raw_log_path}，需要查看日志细节时调用 read_log_snippet 工具读取指定行范围"
        )

    return "\n".join(lines)


def _format_retrieved_cases(cases: list[dict]) -> str:
    """将预检索案例格式化为可读文本。

    Args:
        cases: 检索结果列表，每条含 id / distance / fix_solution 等字段。

    Returns:
        格式化后的案例文本。
    """
    if not cases:
        return "（预检索未找到相似案例）"

    lines = []
    for i, case in enumerate(cases, 1):
        lines.append(f"--- 案例 {i} ---")
        lines.append(f"ID: {case.get('id', '')}")
        lines.append(f"相似度距离: {case.get('distance', '')}（越小越相似）")
        lines.append(f"质量: {case.get('quality', '')}")
        lines.append(f"异常类型: {case.get('exception_type', '')}")
        lines.append(f"来源: {case.get('source_title', '')} ({case.get('source_url', '')})")
        fix = case.get("fix_solution", "")
        if fix:
            lines.append(f"修复方案: {fix}")
        lines.append("")
    return "\n".join(lines)


def build_react_system_prompt(state: dict) -> str:
    """构建 react_agent 的 system prompt。

    从 state 读取 parsed_log / fault_category / classify_reason / retrieved_cases / raw_log_path，
    拼接成完整的 system prompt，包含角色定义、上下文、工具使用说明和输出规范。

    Args:
        state: 当前 AgentState 字典。

    Returns:
        完整的 system prompt 文本。
    """
    parsed_log = state.get("parsed_log")
    fault_category = state.get("fault_category", "unknown")
    classify_reason = state.get("classify_reason", "")
    retrieved_cases = state.get("retrieved_cases", [])
    raw_log_path = state.get("raw_log_path", "")

    prompt = f"""你是一名资深 Minecraft 服务端（Paper/Spigot/Purpur）事故复盘工程师。
你的任务是分析崩溃日志，给出根因分析和修复建议。

## 工作方式
1. 先阅读下方的结构化日志、分类结果和预检索案例。
2. 如果现有信息足够，直接输出最终结论。
3. 如果需要更多相似案例，可以调用 search_similar_cases 工具追加检索。
4. 如果想参考历史分析经验（有效检索词、排除方向、踩坑记录），可以调用 search_memory 工具。
5. 检索工具可以多次调用，每次可以换不同的查询角度。
6. 如果结构化日志信息不足，可以调用 read_log_snippet 工具读取原始日志的指定行范围。

## 故障分类结果（前置节点给出，供参考）
- 分类: {fault_category}
- 分类理由: {classify_reason or '（无）'}

## 结构化崩溃日志
{_format_parsed_log(parsed_log, raw_log_path)}

## 预检索相似案例（已强制检索一次，以下是结果）
{_format_retrieved_cases(retrieved_cases)}

## 检索工具使用说明
向量库中每条案例的 embedding 文本由以下部分组成（你组织查询词时可以参考这个结构，但不必严格照搬）：
- 异常类型 + 异常消息
- 插件列表
- Caused by 异常链
- 前 10 条堆栈帧

查询词建议：用自然语言描述你想找的相似故障场景，例如 "NullPointerException at plugin load" 或 "WorldEdit version mismatch"。工具会返回最相似的案例及其修复方案。

## 中期记忆工具使用说明
你有两个记忆检索工具，用途不同，不要混淆：

- search_similar_cases：找外部案例的修复方案（别人怎么修的），向量相似度检索，只读
- search_memory：找本系统历史分析经验（之前分析类似问题时踩过什么坑、什么检索词有效），关键词匹配，可读可写

### 什么时候调用 search_memory（三个节点主动搜）
1. 分析开始时：按当前异常类型或插件名搜一下，看有没有历史分析经验或已知排除方向
2. RAG 检索结果差时：搜 retrieval_tip 类型，看之前有没有总结过更好的检索词
3. 准备排除某个方向时：搜 excluded_direction 类型，避免重复排除已经验证过无关的方向

### 什么时候调用 save_memory
分析过程中有价值的经验随时存：
- 发现某个检索词效果特别好 → save_memory("retrieval_tip:{{异常类型}}", "检索词XXX效果好")
- 验证某个方向无关 → save_memory("excluded_direction:{{对象}}", "已验证XXX无关，因为...")
- 得出待验证的中间结论 → save_memory("intermediate_conclusion:{{类别}}", "初步判断...")
- 踩了坑 → save_memory("pitfall:{{对象}}", "不要XXX，会导致...")

保存和检索是对等的，只存不搜记忆库会变成坟墓，只搜不存经验无法积累。

## 日志读取工具使用说明
read_log_snippet 工具用于按需读取原始崩溃日志的指定行范围，避免将全文塞入上下文。
参数：
- path：日志文件路径（使用上方"原始日志路径"给出的路径）
- start_line：起始行号，从 1 开始
- end_line：结束行号
返回带行号的日志片段，格式如 "123 | 日志内容"。
每次建议读 50-100 行，可多次调用覆盖不同区间。

## 边界约束（严格遵守，防止幻觉）
1. 只能基于上方的日志原文和检索到的案例下结论，不得编造不存在的案例、插件或修复方案。
2. 如果日志信息不足或检索案例不相关，明确说明"信息不足"，不要强行猜测。
3. 修复建议必须有依据：要么来自检索案例的 fix_solution，要么来自日志中明确的错误信息。
4. 区分"已证实"和"推断"：来自案例的标注【已证实】，自己推理的标注【推断】。
5. 不要给出与 Bukkit 插件生态无关的建议（如 Forge/Fabric 模组方案）。

## 最终输出格式
当你认为分析完成时，输出严格的 JSON 格式（不要包裹在 markdown 代码块中，直接输出 JSON 文本，不要输出任何分析文字）：
{{
  "summary": "一句话总结崩溃原因（不超过50字）",
  "root_cause": "详细根因分析，说明是什么导致了崩溃，为什么",
  "fix_suggestion": "修复建议，分点列出具体可操作的步骤，标注【已证实】或【推断】",
  "confidence": "high/medium/low，high表示有明确案例或日志证据支撑，medium表示有较强推断，low表示信息不足"
}}

如果信息不足无法给出结论，root_cause 填写"信息不足，无法确定根因"，fix_suggestion 填写"建议补充完整日志或提供更多上下文"。
"""
    return prompt
