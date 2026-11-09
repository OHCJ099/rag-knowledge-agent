import json
from pathlib import Path

import config
from config import TOP_K
from index_store import open_index
from retriever import embed


def query_index(
    collection, question, k=TOP_K, source=None, max_distance=None
) -> list[tuple[str, str]]:
    if not question or not str(question).strip():
        raise ValueError(f"问题[{question!r}]为空")

    q = str(question).strip()
    q_embed = embed(q)

    where = {"source": source} if source else None
    result = collection.query(
        query_embeddings=[q_embed],
        n_results=min(k, collection.count()),
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    docs = result["documents"][0]
    metas = result["metadatas"][0]
    dists = result["distances"][0]

    return [
        (m.get("source"), d)
        for m, d, dist in zip(metas, docs, dists)
        if max_distance is None or dist <= max_distance
    ]


def tokens_calculate(question: str, prompt_tokens: int, completion_tokens: int) -> None:
    """把一次问答的 token 用量与估算费用追加到 JSONL 日志。

    token 数是原值，费用单独算一个字段 —— 两者不能混在一个键里。
    """
    cost = (
        prompt_tokens * config.DEEPSEEK_FLASH_PROMPT
        + completion_tokens * config.DEEPSEEK_FLASH_COMPLETION
    )
    row: dict[str, str | int | float] = {
        "question": question,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cost_cny": cost,
    }
    log_path = Path(config.COST_LOG_PATH)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="UTF-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    collection = open_index(config.INDEX_PATH)

    for label, q, expect in [
        ("① 年假", "年假有几天", "应 3 条"),
        ("② 年终奖", "年终奖怎么发", "应 0 条"),
    ]:
        hits = query_index(collection, q, max_distance=config.MAX_DISTANCE)
        print(f"{label}（{expect}）→ 实际 {len(hits)} 条")
        for i, (src, doc) in enumerate(hits, 1):
            print(f"  {i}. 来源={src} | {doc[:30]}")

    # for label, bad in [("② 空串", ""), ("③ 纯空格", "   ")]:
    #     print(label)
    #     try:
    #         query_index(collection, bad)
    #     except ValueError as v:
    #         print(f"  {v}")

    # print("④ 不存在的来源")
    # name = "不存在.md"
    # res = query_index(collection, q, source=name)
    # if not res:
    #     print(f"  没找到来源为 {name} 的内容")
    # else:
    #     print(f"  返回 {len(res)} 条")

    # print("⑤ k=100")
    # print(f"  返回 {len(query_index(collection, q, k=100))} 条")
