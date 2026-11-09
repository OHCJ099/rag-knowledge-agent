import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from qa import build_context, retrieve_hits

DATA_DIR = Path(__file__).parent / "data"
ENV_FILE = Path(__file__).parent.parent.parent / ".env"
# 读取生成回答所需的 DeepSeek 密钥。
load_dotenv(ENV_FILE)
BASE_URL = os.getenv("DEEPSEEK_BASE_URL") or "http://127.0.0.1:8787/v1"
MODEL = os.getenv("DEEPSEEK_MODEL") or "deepseek-v4.1-flash-cn"
ds_key = os.getenv("DEEPSEEK_API_KEY")
if not ds_key:
    raise SystemExit("没找到 DEEPSEEK_API_KEY，请检查 .env")

client = OpenAI(base_url=BASE_URL, api_key=ds_key, timeout=60.0)


def rerank(
    collection, question: str, recall_k: int = 10, k: int = 3
) -> list[tuple[int, str, str]]:
    """先多召回（recall_k），再让 LLM 重排，返回前 k 条。"""
    hits = retrieve_hits(collection, question, k=recall_k)
    context = build_context(hits)

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    f"请将参考材料按与问题的相似度从高到低排序，"
                    f"返回 1 到 {len(hits)} 之间的整数下标，"
                    f"格式为 python 可解析的列表，例如 [3,1,7,2]，不返回其他任何内容。"
                ),
            },
            {"role": "user", "content": f"参考材料：\n{context}\n\n问题：{question}"},
        ],
        extra_body={"thinking": {"type": "disabled"}},
    )

    reply = resp.choices[0].message.content
    if not reply:
        return hits[:k]

    try:
        answer = json.loads(reply)
    except json.JSONDecodeError:
        return hits[:k]

    if not isinstance(answer, list):
        return hits[:k]

    n = len(hits)
    seen = set()
    rerank_list = []
    for index in answer:
        if not isinstance(index, int) or not (1 <= index <= n):
            return hits[:k]
        if index in seen:
            continue
        seen.add(index)
        rerank_list.append(hits[index - 1])

    if not rerank_list:
        return hits[:k]
    top = [
        (index, name, context)
        for index, (_, name, context) in enumerate(rerank_list, start=1)
    ]
    return top[:k]
