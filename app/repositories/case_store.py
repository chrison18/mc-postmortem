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
    plugins: list[str] | None = None,
    caused_by_chain: list[str] | None = None,
) -> str:
    """构建用于 embedding 的文本。

    入库和检索必须调用此函数，保证文本组合一致。
    包含：异常类型+消息、插件列表、Caused by 链、前 10 条堆栈帧。

    Args:
        exception_type: 异常全限定类名。
        exception_message: 异常消息。
        stack_frames: 堆栈帧列表。
        plugins: 插件名列表（如 ["EssentialsX v2.20", "Vault v1.7"]）。
        caused_by_chain: Caused by 行列表。

    Returns:
        拼接后的文本，用于生成 embedding。
    """
    parts = []

    # 异常类型 + 消息
    if exception_type or exception_message:
        parts.append(f"{exception_type}: {exception_message}".strip())

    # 插件列表（对插件冲突/版本不匹配类故障的区分度关键）
    if plugins:
        parts.append("插件: " + ", ".join(plugins))

    # Caused by 异常链
    if caused_by_chain:
        parts.extend(caused_by_chain)

    # 堆栈帧（前 10 条）
    if stack_frames:
        parts.append("堆栈:")
        parts.extend(stack_frames[:10])

    return "\n".join(parts).strip()


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

    def _metadata_to_case(self, case_id: str, meta: dict, doc: str) -> dict:
        """将 Chroma 返回的 metadata + document 转换为统一的 case dict。

        list_all 和 get_by_id 都调用此方法，避免重复代码。

        Args:
            case_id: 案例 ID。
            meta: Chroma metadata 字典。
            doc: Chroma document（即 embedding_text）。

        Returns:
            结构化案例 dict，包含 id、反序列化后的 plugins 列表、embedding_text，
            以及其余 metadata 字段原样保留。
        """
        case = {"id": case_id}
        case.update(meta)
        # 反序列化 plugins 字段（add_cases 时 list 被 json.dumps 成字符串）
        plugins = case.get("plugins")
        if isinstance(plugins, str):
            try:
                case["plugins"] = json.loads(plugins)
            except (json.JSONDecodeError, TypeError):
                # 反序列化失败保持原字符串
                pass
        # 把 embedding_text 从 document 注入
        case["embedding_text"] = doc
        return case

    def list_all(self, limit: int = 20, offset: int = 0) -> list[dict]:
        """分页获取所有案例。

        注意：Chroma 没有直接的分页 API，这里全量 get 后在 Python 层面切片。
        全量 get 会拉取每条案例的 raw_log 大字段（user_confirmed 案例的 metadata
        里存了崩溃日志全文）。MVP 阶段案例数几百条，内存占用可接受；后续案例量大
        了需要改成分页查询或去掉 raw_log。

        Args:
            limit: 返回数量上限。
            offset: 偏移量。

        Returns:
            案例列表，每条为结构化 dict。空库返回空列表。
        """
        result = self._collection.get(include=["metadatas", "documents"])
        ids = result.get("ids", [])
        metadatas = result.get("metadatas", [])
        documents = result.get("documents", [])

        cases = []
        end = min(offset + limit, len(ids))
        for i in range(offset, end):
            cases.append(self._metadata_to_case(ids[i], metadatas[i] or {}, documents[i] or ""))
        return cases

    def get_by_id(self, case_id: str) -> dict | None:
        """按 ID 查询单个案例。

        Args:
            case_id: 案例 ID。

        Returns:
            案例 dict，不存在时返回 None。
        """
        result = self._collection.get(ids=[case_id], include=["metadatas", "documents"])
        ids = result.get("ids", [])
        if not ids:
            return None
        metadatas = result.get("metadatas", [])
        documents = result.get("documents", [])
        return self._metadata_to_case(ids[0], metadatas[0] or {}, documents[0] or "")

    def delete(self, case_id: str) -> bool:
        """按 ID 删除案例。

        注意：先查后删不是原子操作，并发删除可能有竞态。MVP 管理接口并发低，可接受。

        Args:
            case_id: 案例 ID。

        Returns:
            删除成功返回 True，案例不存在返回 False。
        """
        if self.get_by_id(case_id) is None:
            return False
        self._collection.delete(ids=[case_id])
        return True


def get_case_store() -> CaseStore:
    """获取 CaseStore 单例。

    Returns:
        CaseStore 实例。
    """
    global _store
    if _store is None:
        _store = CaseStore()
    return _store
