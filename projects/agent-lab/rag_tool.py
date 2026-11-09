import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "rag-kb"))

import config
import query_lab
from index_store import ensure_index, open_index
from langchain_core.tools import tool


@tool
def search_kb(question: str) -> str:
    """在制度知识库里检索。输入一个自然语言问题，返回最相关的几段原文。"""
    # 空问题直接返回提示：既不触发下面的索引重建（会产生付费 embedding），
    # 也不让异常冒到 ToolNode 变成 error 状态的 ToolMessage。
    if not question.strip():
        return "问题为空，请提供具体要查的内容。"
    # 索引丢了（例如被清理）就现场重建，避免空库在 query 里抛出难懂的 TypeError。
    ensure_index(config.INDEX_PATH, config.DATA_DIR)
    collection = open_index(config.INDEX_PATH)
    hits = query_lab.query_index(collection, question, max_distance=config.MAX_DISTANCE)
    if not hits:
        return "无命中"
    parts = []
    for index, hit in enumerate(hits, start=1):
        parts.append(f"[{index}]{hit[0]}：{hit[1]}\n\n")
    return "".join(parts)
