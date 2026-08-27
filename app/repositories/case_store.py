"""
案例向量库封装。

基于 Chroma 持久化存储，提供 add_cases / search_similar 语义化接口。
内部实现不对外暴露 Chroma 特有 API，后续可平滑迁移到 pgvector 等其他向量库。

入库和检索必须使用 build_embedding_text() 构建文本，保证 embedding 空间一致。
"""

import json

import chromadb

from app.config import settings
from app.services.embedding import embed_texts

# Chroma collection 名称
_COLLECTION_NAME = "cases"

# 单例
_store: "CaseStore | None" = None


def build_embedding_text(
    exception_type: str,
    exception_message: str,
    stack_frames: list[str],
) -> str:
    """构建用于 embedding 的文本。

    入库和检索必须调用此函数，保证文本组合一致：
    异常类型 + 异常消息 + 前 10 条堆栈帧。

    Args:
        exception_type: 异常全限定类名。
        exception_message: 异常消息。
        stack_frames: 堆栈帧列表。

    Returns:
        拼接后的文本，用于生成 embedding。
    """
    frames = "\n".join(stack_frames[:10])
    text = f"{exception_type}: {exception_message}\n{frames}"
    return text.strip()


class CaseStore:
    """案例向量库，封装 Chroma 的入库与检索。"""

    def __init__(self) -> None:
        """初始化 Chroma 持久化客户端和 collection。"""
        self._client = chromadb.PersistentClient(path=settings.CHROMA_PATH)
        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def add_cases(self, cases: list[dict]) -> None:
        """批量入库案例（幂等，同 id 覆盖）。

        Args:
            cases: 案例列表，每个 dict 需包含：
                - id: 案例唯一标识
                - embedding_text: 用于 embedding 的文本（由 build_embedding_text 生成）
                - 其他字段作为 metadata 存储（必须是 str/int/float/bool 标量）
        """
        if not cases:
            return

        ids = []
        texts = []
        metadatas = []
        for case in cases:
            ids.append(case["id"])
            texts.append(case["embedding_text"])
            # 排除 embedding_text，其余作为 metadata；非标的量序列化为 JSON 字符串
            meta = {}
            for k, v in case.items():
                if k == "embedding_text":
                    continue
                if isinstance(v, (str, int, float, bool)) or v is None:
                    meta[k] = v if v is not None else ""
                else:
                    meta[k] = json.dumps(v, ensure_ascii=False)
            metadatas.append(meta)

        embeddings = embed_texts(texts)
        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

    def search_similar(self, query_text: str, top_k: int = 5) -> list[dict]:
        """检索与查询文本最相似的案例。

        Args:
            query_text: 查询文本（由 build_embedding_text 生成，与入库格式一致）。
            top_k: 返回最相似的案例数量。

        Returns:
            相似案例列表，按相似度从高到低排序，每条含：
            - id: 案例 ID
            - distance: 余弦距离（越小越相似）
            - 其余为入库时的 metadata 字段
        """
        if not query_text.strip():
            return []

        query_embedding = embed_texts([query_text])
        results = self._collection.query(
            query_embeddings=query_embedding,
            n_results=min(top_k, self._collection.count() or 1),
        )

        cases = []
        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        for i, case_id in enumerate(ids):
            case = {"id": case_id, "distance": distances[i]}
            case.update(metadatas[i] if i < len(metadatas) else {})
            cases.append(case)
        return cases

    def count(self) -> int:
        """返回库中案例总数。

        Returns:
            案例数量。
        """
        return self._collection.count()


def get_case_store() -> CaseStore:
    """获取 CaseStore 单例。

    Returns:
        CaseStore 实例。
    """
    global _store
    if _store is None:
        _store = CaseStore()
    return _store
