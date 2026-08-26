# mc-postmortem

MC 服务端事故复盘多 Agent 系统。输入崩溃日志，输出根因分析、代码级修复建议，支持反馈闭环与知识沉淀。

## 技术栈

- **语言**：Python 3.11+
- **后端**：FastAPI
- **Agent 编排**：LangGraph
- **LLM**：DeepSeek（OpenAI 兼容格式）
- **Embedding**：BGE-M3（本地）
- **存储**：SQLite + Chroma（MVP），预留迁移 PostgreSQL + Qdrant
- **目标生态**：Paper / Spigot 服务端

## 核心流程

日志解析 → 故障分类 → ReAct 工具调用循环 → 结论生成 → 对抗审查 → 报告输出

支持正向/负向反馈闭环，知识沉淀到三层记忆（短期/中期/长期 RAG）。

## 项目结构

```
app/
├── api/              # API 层（参数校验+响应封装）
├── core/
│   ├── nodes/        # LangGraph 节点
│   ├── classifiers/  # 故障分类器（预留规则+LLM混合）
│   ├── tools/        # Function Calling 工具集
│   ├── state.py      # State 定义
│   └── graph.py      # 图构建
├── prompts/          # Prompt 模块（独立管理）
├── repositories/     # 数据访问层
├── models/           # 数据模型
├── services/         # 业务服务
├── config.py         # 配置管理
└── main.py           # FastAPI 入口
data/
├── raw_logs/         # 原始日志文件
└── chroma/           # 向量数据库
tests/
```

## 开发

```bash
# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env 填入 LLM_API_KEY 等

# 启动服务
uvicorn app.main:app --reload
```

## 约束

- 遵循 KISS 原则：不做过度设计，模块解耦，可读性优先
- Prompt 独立成模块，不集成到 graph 节点
- 配置全部走环境变量，不硬编码
- git push 由开发者手动执行
