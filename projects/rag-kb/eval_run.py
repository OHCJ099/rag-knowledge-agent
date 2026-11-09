"""检索评估：对比基础召回、查询改写与重排序，统计人工标注结果。"""

from pathlib import Path

from loader import load_documents
from multi_retrieve import merge_hits
from qa import build_index, retrieve_hits
from query_rewrite import rewrite_query
from rerank import rerank

DATA_DIR = Path(__file__).parent / "data"
EVAL_FILE = Path(__file__).parent / "eval_questions.md"
K = 3
RUN_SETS = ("开发",)  # 留出集调参结束前不跑：跑过就等于把它当开发集用了


def load_eval(path: Path) -> list[dict[str, str]]:
    """读评估集 md 表格 → 每题一个 dict。

    一行长这样：| 1 | 开发 | 有答案 | 问题文本 | 请假制度.md | 要点 | 命中 |
    解析：去掉行首尾的 |，再按 | 切开 —— cells[0] 是编号、cells[1] 是集，依此类推。
    空行 / 表头 / 分隔行（|---|）/ 还没填问题的行，都会被跳过。
    """
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 4 or not cells[0] or cells[0].startswith("-"):
            continue
        if cells[0] == "编号" or not cells[3]:  # 表头 / 问题还没填
            continue
        while len(cells) < 9:  # 判定列留空时行尾少几格，补上
            cells.append("")
        rows.append(
            {
                "编号": cells[0],
                "集": cells[1],
                "题型": cells[2],
                "问题": cells[3],
                "期望来源": cells[4],
                "期望要点": cells[5],
                "判定": cells[6],
                "判定A": cells[7],
                "判定B": cells[8],
            }
        )
    return rows


def run_eval(questions: list[dict[str, str]], collection) -> None:
    """逐题检索 + 打印；然后按「判定」列统计。"""
    for q in questions:
        print(
            f"\n== [{q['编号']}] {q['问题']} ==  期望来源：{q['期望来源']}（{q['题型']}）"
        )
        hit = rerank(collection, q["问题"], k=K)
        for h in hit:
            print(f"[{h[0]}] 来源: {h[1]}\n{h[2]}")

    fen_mu = 0
    fen_zi = 0
    fen_zi_a = 0
    fen_zi_b = 0
    no_anwser = []
    not_checked = []
    # 统计并打印（基线读「判定」列、优化A 读「判定A」列）：
    for q in questions:
        #  - 开发集召回：分母 = 题型为「有答案」或「部分可答」的题数；分子 = 判定为「命中」的题数
        if q["题型"] == "有答案" or q["题型"] == "部分可答":
            fen_mu += 1
            if q["判定"] == "命中":
                fen_zi += 1
            elif q["判定"] == "未命中":
                pass
            else:
                not_checked.append(q["编号"])
            if q["判定A"] == "命中":
                fen_zi_a += 1
            if q["判定B"] == "命中":
                fen_zi_b += 1
        if q["题型"] == "无答案":
            no_anwser.append(q["编号"])
    #  - 输出形如：开发集召回 X/Y
    #  - 无答案题不进分母，单独列出它们的编号；没填判定的题也单独提醒
    print(
        f"开发集召回（基线）：{fen_zi}/{fen_mu}\n"
        f"开发集召回（优化A）：{fen_zi_a}/{fen_mu}\n"
        f"开发集召回（优化B）：{fen_zi_b}/{fen_mu}\n"
        f"无答案编号：{no_anwser}\n待判定编号：{not_checked}"
    )


def retrieve_hits_multi(
    collection, question: str, k: int = 3
) -> list[tuple[int, str, str]]:
    """多路检索（优化 A）：原问题 + 改写变体各自检索，合并后返回。"""
    hit_groups = []
    # ① 变体 —— 调 rewrite_query(question)；注意它失败时返回什么
    questions = rewrite_query(question)
    questions.append(question)
    # ② 每路检索 —— 对每一路调 retrieve_hits(collection, 路, k=k)，
    for q in questions:
        hit_groups.append(retrieve_hits(collection, q, k=K))
    #   收集成 hit_groups（列表的列表：[路1的hits, 路2的hits, ...]）
    # ③ 合并 —— 调 merge_hits；k 传什么想清楚（要的是"最终几条"）
    f_hit = merge_hits(hit_groups, k=K)
    return f_hit


def main() -> None:
    docs = load_documents(DATA_DIR)
    collection = build_index(docs)
    questions = load_eval(EVAL_FILE)
    questions = [q for q in questions if q["集"] in RUN_SETS]
    run_eval(questions, collection)


if __name__ == "__main__":
    main()
