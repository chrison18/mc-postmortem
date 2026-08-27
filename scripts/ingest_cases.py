"""
案例入库脚本。

读取 data/cases/ 下所有 case_*.json，解析 raw_log 生成 embedding 文本，
批量存入 Chroma 向量库。幂等运行，重复执行会覆盖已有案例。

用法：
    python -m scripts.ingest_cases
"""

import json
import os
import tempfile

from app.core.parsers import parse_crash_report
from app.repositories.case_store import build_embedding_text, get_case_store

CASES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "cases")


def load_all_cases() -> list[dict]:
    """加载 data/cases/ 下所有案例 JSON。

    Returns:
        案例字典列表。
    """
    cases = []
    for filename in sorted(os.listdir(CASES_DIR)):
        if not filename.startswith("case_") or not filename.endswith(".json"):
            continue
        with open(os.path.join(CASES_DIR, filename), "r", encoding="utf-8") as f:
            cases.append(json.load(f))
    return cases


def prepare_case_for_ingest(case: dict) -> dict:
    """将原始案例 JSON 转换为入库格式。

    解析 raw_log 得到结构化字段，用 build_embedding_text 生成 embedding 文本。

    Args:
        case: 原始案例 JSON 字典。

    Returns:
        入库用的字典，含 id / embedding_text / metadata 字段。
    """
    # 写临时文件供解析器使用
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as tf:
        tf.write(case.get("raw_log", ""))
        tmp_path = tf.name

    try:
        parsed = parse_crash_report(tmp_path)
    finally:
        os.unlink(tmp_path)

    embedding_text = build_embedding_text(
        exception_type=parsed["exception_type"],
        exception_message=parsed["exception_message"],
        stack_frames=parsed["key_stack_frames"],
    )

    return {
        "id": case["id"],
        "embedding_text": embedding_text,
        "exception_type": case.get("exception_type", ""),
        "fix_solution": case.get("fix_solution", ""),
        "source_url": case.get("source_url", ""),
        "source_title": case.get("source_title", ""),
        "quality": case.get("quality", ""),
        "source_type": case.get("source_type", ""),
    }


def main() -> None:
    """入库主流程。"""
    print(f"读取案例目录: {CASES_DIR}")
    raw_cases = load_all_cases()
    print(f"找到 {len(raw_cases)} 条案例")

    print("解析 raw_log 并构建 embedding 文本...")
    ingest_cases = [prepare_case_for_ingest(c) for c in raw_cases]

    store = get_case_store()
    print(f"入库前案例数: {store.count()}")
    store.add_cases(ingest_cases)
    print(f"入库后案例数: {store.count()}")
    print("入库完成")


if __name__ == "__main__":
    main()
