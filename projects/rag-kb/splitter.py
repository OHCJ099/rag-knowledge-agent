# 文本切分：将长文分成小块，相邻块可保留重叠内容。
import re
from pathlib import Path

from loader import load_documents

DATA_DIR = Path(__file__).parent / "data"

# 此处在导入模块时也会加载文档，供下方演示使用。
docs = load_documents(DATA_DIR)  # 这样拿到的是清洗后的文本，正好串起整条流水线


# chunk_size 是每块字符数，overlap 是相邻块重叠的字符数。
def split_text(text, chunk_size, overlap) -> list[str]:
    chunk_list = []
    # 每次前进的字符数必须大于 0，重叠长度也不能为负。
    if chunk_size - overlap <= 0:
        raise ValueError("chunk_size - overlap 必须大于 0")
    if overlap < 0:
        raise ValueError("overlap 必须大于等于 0")
    # 每次前进 chunk_size - overlap 个字符，再截取一块文本。
    for start in range(0, len(text), chunk_size - overlap):
        # 切片超出文本末尾时会自动截断，因此最后一块可能较短。
        chunk_list.append(text[start : start + chunk_size])
    return chunk_list


def split_by_boundary(text: str, chunk_size: int) -> list[str]:
    """按边界切分 v1：段落优先 → 超长段拆句 → 超长句硬切（复用 split_text）。不重叠。"""
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")
    chunks: list[str] = []
    chunk = ""
    # 约定：空串 → []；每块 ≤ chunk_size；chunk_size <= 0 → ValueError
    paragraphs = text.split("\n\n")
    for paragraph in paragraphs:
        if not paragraph.strip():
            continue

        if len(paragraph) > chunk_size:
            if chunk:
                chunks.append(chunk)
                chunk = ""

            sentences = re.findall(r"[。！？]+|[^。！？]+[。！？]*", paragraph)
            for sentence in sentences:
                if len(sentence) > chunk_size:
                    # 先保存已有 chunk，再硬切这句
                    if chunk:
                        chunks.append(chunk)
                        chunk = ""
                    # 接住 split_text 返回的列表，处理满块和尾巴
                    tl = split_text(sentence, chunk_size, 0)
                    for t in tl:
                        if len(t) == chunk_size:
                            chunks.append(t)
                        else:
                            chunk += t
                else:
                    if len(chunk) + len(sentence) > chunk_size:
                        chunks.append(chunk)
                        chunk = sentence
                    else:
                        chunk += sentence
        else:
            if chunk:
                separator = "\n\n"
            else:
                separator = ""
            # 正常句子：能装就拼，装不下就封口并用这句开新块
            if len(chunk) + len(paragraph) + len(separator) > chunk_size:
                chunks.append(chunk)
                chunk = paragraph
            else:
                chunk += separator + paragraph
    if chunk:
        chunks.append(chunk)
    return chunks


# 直接运行时：小样例、参数检查、各文档切分对比。
if __name__ == "__main__":
    # A) 小样例：普通段落、超长段落、超长句。
    text = (
        "第一段：" + "甲" * 20 + "。\n\n"
        "第二段：" + "乙" * 30 + "。" + "丙" * 30 + "。\n\n"
        "第三段：" + "丁" * 130 + "。" + "尾句。"
    )

    chunks = split_by_boundary(text, 60)

    for i, chunk in enumerate(chunks, 1):
        print(f"第{i}块 | 长度：{len(chunk)} | 开头：{chunk[:20]}")

    # B) 空串和非法参数检查。
    print("空串结果：", split_by_boundary("", 500))

    try:
        split_by_boundary("abc", 0)
    except ValueError as e:
        print("捕获参数错误：", e)

    # C) 每份文档分别切分、统计接缝、打印结果。
    for doc in docs:
        o500_50 = split_text(doc["text"], 500, 50)
        o500_0 = split_text(doc["text"], 500, 0)
        n500 = split_by_boundary(doc["text"], 500)

        a = 0
        for i in range(len(o500_0) - 1):
            c1 = o500_0[i]
            c2 = o500_0[i + 1]

            if not (
                c1[-1] in "。！？：" or (c1[-15:] + "\n\n" + c2[:15]) in doc["text"]
            ):
                a += 1

        b = 0
        for i in range(len(n500) - 1):
            c1 = n500[i]
            c2 = n500[i + 1]

            if not (
                c1[-1] in "。！？：" or (c1[-15:] + "\n\n" + c2[:15]) in doc["text"]
            ):
                b += 1

        print(
            f"{doc['source']} | 老500/50: {len(o500_50)}块 | "
            f"老500/0: {len(o500_0)}块/接缝坏{a}处 | "
            f"新500: {len(n500)}块/接缝坏{b}处"
        )
