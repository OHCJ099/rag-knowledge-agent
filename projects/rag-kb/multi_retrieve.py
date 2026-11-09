"""多路合并：按各路排名轮流取片段，去重后限制数量并重新编号。"""


def merge_hits(hit_groups, k: int = 3) -> list[tuple[int, str, str]]:
    final_list = []
    seen = []
    for j in range(k):
        for i in range(len(hit_groups)):
            if j >= len(hit_groups[i]):
                continue
            hit = hit_groups[i][j]
            if (hit[1], hit[2]) not in seen:
                seen.append((hit[1], hit[2]))
                final_list.append((len(final_list) + 1, hit[1], hit[2]))
            else:
                continue
            if len(final_list) == k:
                return final_list

    return final_list


if __name__ == "__main__":
    hit_groups = [
        [(1, "A.md", "正文甲"), (2, "A.md", "正文乙")],
        [(1, "A.md", "正文甲"), (2, "B.md", "正文丙")],
        [(1, "C.md", "正文丁"), (2, "A.md", "正文乙")],
    ]
    print(merge_hits(hit_groups, k=3))
