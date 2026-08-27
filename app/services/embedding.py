"""
本地 embedding 封装。

基于 sentence-transformers 加载 BGE-M3 模型，生成向量。
模型首次使用时自动下载到本地缓存，后续复用。
国内网络默认使用 HF_ENDPOINT 镜像，可在 .env 中覆盖。
"""

import os

# 必须在导入 sentence_transformers 之前设置 HF_ENDPOINT，否则不生效
from app.config import settings

if settings.HF_ENDPOINT:
    os.environ["HF_ENDPOINT"] = settings.HF_ENDPOINT

from sentence_transformers import SentenceTransformer

# 全局单例，模型加载耗时，复用
_model: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    """获取 embedding 模型单例。

    首次调用时加载模型（可能需要下载），后续复用。

    Returns:
        SentenceTransformer 模型实例。
    """
    global _model
    if _model is None:
        _model = SentenceTransformer(settings.EMBEDDING_MODEL)
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """批量生成文本的 embedding 向量。

    Args:
        texts: 待编码的文本列表。

    Returns:
        向量列表，每个向量为 float 列表，已归一化。
    """
    if not texts:
        return []
    model = get_embedding_model()
    embeddings = model.encode(texts, normalize_embeddings=True)
    return embeddings.tolist()


def embed_text(text: str) -> list[float]:
    """生成单条文本的 embedding 向量。

    Args:
        text: 待编码的文本。

    Returns:
        归一化后的向量。
    """
    return embed_texts([text])[0]
