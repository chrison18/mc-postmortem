"""
全局配置管理模块。

所有配置从环境变量 / .env 文件读取，禁止在代码中硬编码。
使用 pydantic-settings 的 BaseSettings 实现类型安全的配置加载。
"""

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用全局配置，字段对应环境变量。"""

    # LLM 服务地址（OpenAI 兼容格式），默认指向 DeepSeek 官方 API
    LLM_BASE_URL: str = "https://api.deepseek.com"

    # LLM API 密钥，必填项，缺失时在导入阶段报错
    LLM_API_KEY: str = ""

    # 使用的 LLM 模型名称
    LLM_MODEL: str = "deepseek-v4-flash"

    # 本地 embedding 模型名称（sentence-transformers 加载）
    EMBEDDING_MODEL: str = "BAAI/bge-m3"

    # ReAct 循环最大迭代次数，防止无限循环
    MAX_REACT_LOOPS: int = 5

    # SQLite 数据库文件路径
    SQLITE_PATH: str = "./data/tasks.db"

    # Chroma 向量数据库持久化目录
    CHROMA_PATH: str = "./data/chroma"

    # 原始崩溃日志存放目录
    RAW_LOG_DIR: str = "./data/raw_logs"

    # FastAPI 服务监听地址
    HOST: str = "0.0.0.0"

    # FastAPI 服务监听端口
    PORT: int = 8000

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @model_validator(mode="after")
    def _check_api_key(self) -> "Settings":
        """校验 LLM_API_KEY 是否已配置，缺失时抛出友好的中文错误。"""
        if not self.LLM_API_KEY or self.LLM_API_KEY.strip() == "":
            raise RuntimeError(
                "未检测到 LLM_API_KEY，请在项目根目录创建 .env 文件并配置 LLM_API_KEY=你的密钥"
            )
        return self


# 全局单例，模块导入时即完成校验（LLM_API_KEY 缺失会在此处报错）
settings = Settings()
