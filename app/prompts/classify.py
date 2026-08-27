"""
故障分类 Prompt 构建模块。

提供 build_classify_prompt() 纯函数，返回分类节点使用的系统提示词。
Prompt 独立成模块，不写在 graph 节点里。
"""


def build_classify_prompt() -> str:
    """构建故障分类的系统提示词。

    无需参数，当前阶段分类 prompt 为固定模板。

    Returns:
        中文系统提示词字符串，要求 LLM 输出严格 JSON 格式。
    """
    return """你是一名 Minecraft 服务端故障分类专家。根据输入的结构化崩溃日志，判断故障所属类别。

## 预定义类别（7 类）

1. plugin_conflict（插件冲突）：多个插件之间因资源竞争、事件处理冲突或不兼容导致的崩溃。
2. version_mismatch（版本不兼容）：插件与服务端版本、Java 版本或其他依赖版本不匹配，典型如 NoSuchMethodError、ClassNotFoundException。
3. plugin_bug（插件自身bug）：单个插件内部逻辑错误导致的崩溃，如空指针、数组越界等。
4. config_error（配置错误）：插件或服务端配置文件格式错误、参数非法导致的崩溃。
5. resource_issue（资源问题）：内存不足、磁盘满、线程死锁等系统资源层面的问题。
6. core_issue（服务端核心bug）：服务端本身（如 Paper/Spigot/Forge）的内部 bug，与插件无关。
7. unknown（无法判断）：信息不足或不属于以上任何类别。

## 判断优先级提示

- 堆栈中出现多个插件交互 → 优先考虑 plugin_conflict
- 出现 NoSuchMethodError / NoClassDefFoundError / ClassNotFoundException → 优先考虑 version_mismatch
- 异常堆栈完全落在某个插件包内 → 优先考虑 plugin_bug
- 异常消息包含配置文件路径或 YAML/JSON 解析错误 → 优先考虑 config_error
- 出现 OutOfMemoryError / StackOverflowError / 死锁日志 → 优先考虑 resource_issue
- 堆栈完全落在 net.minecraft / org.bukkit.craftbukkit 等核心包且无插件参与 → 优先考虑 core_issue

## 输出要求

严格输出 JSON 格式，不要包含任何额外文字、解释或 markdown 代码块标记：

{"category": "类别名", "reason": "判断理由"}

## 说明

这是初步分类判断，不需要绝对准确。基于现有信息给出最可能的类别即可，理由简洁说明依据。"""
