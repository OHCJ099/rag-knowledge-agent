# 手写语义检索：将问题和文本转成向量，按相似度返回结果。
import math
import os
from pathlib import Path

from config import CHUNK_SIZE, OVERLAP
from dotenv import load_dotenv
from loader import load_documents
from openai import OpenAI
from splitter import split_text

# 加载 .env 文件里的环境变量
load_dotenv()

# 读取 DashScope 的 API Key，如果没有就报错退出
key = os.getenv("DASHSCOPE_API_KEY")
if not key:
    raise SystemExit("没找到 DASHSCOPE_API_KEY，请检查 .env")


# 向量化：调用远程服务，把文本转成浮点数列表。
def embed(text: str) -> list[float]:
    client = OpenAI(
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=key,
    )
    resp = client.embeddings.create(model="text-embedding-v4", input=text)
    vec = resp.data[0].embedding  # 取出向量列表
    return vec


# 余弦相似度衡量两个向量方向是否接近，越大通常越相似。
def cos_sim(a, b):
    dot = sum(x * y for x, y in zip(a, b))  # 点积
    norm_a = math.sqrt(sum(x * x for x in a))  # a 的模长
    norm_b = math.sqrt(sum(x * x for x in b))  # b 的模长
    # 点积除以两边模长；这里假定输入向量的模长都不为 0。
    return dot / (norm_a * norm_b)


def retrieve(
    question: str,
    chunks: list[dict],
    k: int = 3,
    min_score: float | None = None,
    source_filter: str | None = None,
) -> list[tuple[dict, float]]:
    """
    检索函数：根据问题，从所有文本块中找出最相似的 k 个。

    参数：
        question      : 用户的问题
        chunks        : 列表，每个元素是 {"source": 来源, "text": 文本块}
        k             : 返回最相似的前 k 个
        min_score     : 最低相似度阈值，低于这个分数的不要
        source_filter : 如果指定，只在这个来源的文件里检索

    返回：
        列表，每个元素是 (chunk字典, 相似度分数)
    """

    # 先把问题也转成向量
    question_embed = embed(question)

    embed_chunks_list = []  # 存放每个参与计算的文本块的向量
    finally_cossin = []  # 存放每个文本块与问题的余弦相似度
    before_sorte = []  # 存放 (chunk, 相似度) 的配对，用于排序
    used_chunks = []  # 存放真正参与计算的 chunk，和 embed_chunks_list 一一对应

    # 如果指定了 source_filter，就只挑出该来源的 chunk
    if source_filter is not None:
        for chunk in chunks:
            if chunk["source"] == source_filter:
                used_chunks.append(chunk)  # 记录这个 chunk
                embed_chunks_list.append(embed(chunk["text"]))  # 计算它的向量
            else:
                continue
    # 如果没有指定 source_filter，就全量计算
    else:
        for chunk in chunks:
            used_chunks.append(chunk)
            embed_chunks_list.append(embed(chunk["text"]))

    # 逐个计算每个文本块向量与问题向量的余弦相似度
    for chunk in embed_chunks_list:
        Vec_value = cos_sim(question_embed, chunk)
        finally_cossin.append(Vec_value)

    # 把 chunk 和它的相似度重新配对
    # 注意：这里用的是 used_chunks，保证和 finally_cossin 一一对应
    for y in zip(used_chunks, finally_cossin):
        before_sorte.append((y[0], y[1]))

    # 按相似度从高到低排序
    after_sorted = sorted(before_sorte, key=lambda w: w[1], reverse=True)

    # 如果没有设置最低分，直接取前 k 个
    if min_score is None:
        return after_sorted[:k]
    else:
        # 否则先过滤掉低于 min_score 的，再取前 k 个
        filted = [i for i in after_sorted if i[1] >= min_score]
        return filted[:k]


# 直接运行时，演示带来源筛选的检索。
if __name__ == "__main__":
    # 加载 data 目录下的所有文档
    docs = load_documents(Path(__file__).parent / "data")

    # new_chunks 用来存放所有切分后的文本块，每个块带上来源
    new_chunks = []

    for doc in docs:
        # 把每篇文档切分成 CHUNK_SIZE 字左右的块，块之间重叠 OVERLAP 字
        for chunk in split_text(doc["text"], CHUNK_SIZE, OVERLAP):
            new_chunks.append(
                {
                    "source": doc["source"],  # 来源文件名
                    "text": chunk,  # 切分出来的文本
                }
            )

    # 要检索的问题列表
    question_list = [
        "员工请假的规则是什么？",
    ]

    for question in question_list:
        # 检索最相似的前 10 个块
        anwsers = retrieve(question, new_chunks, 10, source_filter="考勤制度.md")

        # 打印每个结果：来源、文本前 30 字、相似度
        for anwser in anwsers:
            print(
                f"来源: {anwser[0]['source']} 回复: {anwser[0]['text'][:30]} 相似值: {anwser[1]:.2f}"
            )
