# MC 服务端崩溃案例库

- 采集时间：2026-08-27（第一批 case_001~030）、2026-08-28（第二批 case_031~060、第三批 case_061~090）
- 案例数：90
- 场景分布：npe 15 / version 17 / conflict 7 / config 5 / loadfail 5 / oom 3 / watchdog 3 / runtime 19 / other 16
- 来源分布：GitHub Issue 81 / 托管商知识库 9
- 服务端分布：Paper 25 / Purpur 8 / Spigot 2 / unknown 55（含 Folia 系分支及未标注）
- 质量分布：high 23 / medium 60 / low 7
- fix_solution 分布：【已证实】55 / 【推断】11 / 待补充 15 / KB原文 9
- 每条 case 一个 JSON，字段见 `../schema.json`；原始证据存于 `../_raw/`

## 场景说明

| 场景 | 含义 |
|---|---|
| npe | NullPointerException，插件空指针 |
| version | 版本不兼容（NoSuchMethod/NoClassDefFound/UnsupportedClassVersion 等） |
| conflict | 插件冲突、循环依赖、类加载器冲突 |
| config | 配置文件错误（YAML、编码、snakeyaml 限制等） |
| loadfail | 插件加载/启用失败（缺依赖、InvalidPlugin、库解析受限等） |
| oom | OutOfMemoryError |
| watchdog | 主线程卡死/Watchdog 崩溃 |
| runtime | 命令/事件运行时异常（CME、越界、IllegalArgument 等） |
| other | 其他罕见异常（StackOverflow、ZipException、netty、Folia 线程违规等） |

## 采集原则

1. 全部来自真实公开页面（GitHub Issue / 托管商知识库），每条保留可访问 `source_url`。
2. `raw_log` 逐字复制原文，不改写堆栈；节选时只删无关行，保留异常头+完整堆栈+上下文。
3. 场景按根因标注（写在每条 notes 开头），异常类名仅作参考。
4. 低质量样本（截断日志/仅签名）如实标 low/medium，不补全。
5. 第二批优先 1.18+：30 条中已知版本全部 ≥1.18（28 条明确为 1.18+，2 条版本未知）。
6. 第二批重点补「other 罕见异常」与高质量样本：16 条 high 全部来自第二批，多为带 Java 版本+完整插件列表+完整堆栈的 mclo.gs 全日志。
7. 第三批（case_061~090）多样性纠偏：ItemsAdder 系仅 1 条，覆盖 LuckPerms/EssentialsX/WorldGuard/WorldEdit/ViaVersion/CoreProtect/PlaceholderAPI/Citizens/BlueMap/Geyser 等；闭环 issue 优先，12 条 high、4 条待补充；已知版本全部 ≥1.18。
8. 质量分级新口径（2026-08-28 起）：high = 完整堆栈 + 系统信息（插件列表或 System Details）+ 【已证实】修复方案；medium = 完整堆栈但 fix 为【推断】/待补充，或日志完整但缺系统信息；low = 日志截断/仅签名。
9. fix_solution 非空强制：所有案例 fix_solution 必须为【已证实】/【推断】/待补充/KB原文之一，禁止空串。

## 备注

- 2026 年 Minecraft Java 版启用新年份版本号（如 26.1.2、26.2），`mc_version` 按日志原文记录。
- Folia 系分支（Folia / Luminol / Leaf / Pufferfish / Sable 等）`server_type` 按 schema 枚举记为 `unknown`，具体分支写进 notes。
- 第二批 mclo.gs 案例的 `raw_log` 由「启动信息段（Java/服务端版本/插件列表）+ 异常堆栈段」拼成，两段均为证据原文的逐字子串；校验脚本对每行做原文回查。
- 崩溃报告（crash-report）类日志无 Bukkit 插件列表，plugins 为空数组。
- 知识库教学文没有真实服务器堆栈，raw_log 为文中报错描述原文，fix_solution 为修复步骤原文。
- 个别 issue 正文粘贴的日志行之间带空行（Markdown 粘贴所致），按原样保留。

## 校验

```
python ../_build/verify_all.py
```

检查：编号连续、字段/枚举合法、fix_solution 非空且格式合法、质量分级符合新口径、URL 与指纹全局唯一、指纹可复算（第二批及以后）、raw_log 每行都能在 `_raw/` 证据原文中找到（第二批及以后非 KB 类）、Java 异常类名出现在 raw_log 中。
