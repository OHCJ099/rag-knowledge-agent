# projects/rag-kb/loader.py
# 文档加载：读取制度文件，并清理多余空白。
import io
from pathlib import Path

import pypdf.errors
from pypdf import PdfReader

DATA_DIR = Path(__file__).parent / "data"  # 锚定，不管从哪个目录跑都对


def clean_text(text: str) -> str:
    """清洗：统一换行 / 去每行首尾空白 / 连续空行压缩"""
    # 统一 Windows 和其他系统的换行格式，便于按行处理。
    text_replace = text.replace("\r\n", "\n")
    # 拆成多行，去掉每行两端的空格，再重新合并。
    lines = text_replace.split("\n")
    lines_strip = [line.strip() for line in lines]
    lines_join = "\n".join(lines_strip)
    # 三个换行表示至少两个空行；反复替换，最多保留一个空行。
    while "\n\n\n" in lines_join:
        lines_join = lines_join.replace("\n\n\n", "\n\n")
    # 清除整段文本开头和结尾的空白。
    return lines_join.strip()


# 返回文档列表，每份文档都保留来源文件名和清洗后的正文。
def load_documents(data_dir: Path) -> list[dict[str, str]]:
    """读 data_dir 下所有 .md，返回 [{"source": 文件名, "text": 清洗后全文}, ...]。"""
    docs = []
    # 当前实现只读取本目录的 .md 文件，并按文件名排序。
    files = sorted(data_dir.glob("*.md"))
    for path in files:
        # 按 UTF-8 读取中文文件，再清洗正文。
        raw = path.read_text(encoding="utf-8")
        text = clean_text(raw)
        # source 保存来源，text 保存正文，方便后续切块和检索。
        doc = {"source": path.name, "text": text}
        docs.append(doc)
    return docs


# ============================================================================
# 今日新增：parse_bytes —— 让「解析规则」只存在一份
#
# 【要解决的问题】
# 同一个文件，现在有两条解析路径，各写各的，能力还不一样：
#   路径 A（网页上传）：api.py 载入逻辑 → 坏 PDF 能兜住
#   路径 B（命令行入库）：load_document  → 坏 PDF 直接炸（没有那三行）
# 结果：改一处漏一处，两边行为不一致。目标 = 解析只留一份，放在本文件。
#
# 【为什么参数收 content（字节）而不是 path（路径）】
#   因为「校验必须在落盘之前」—— 落盘前磁盘上还没有这个文件，只有内存里的字节。
#   两条路手里各有什么：
#     api.py 上传   ：await file.read() 拿到的 content        → 有字节，还没有路径
#     load_document ：path                                   → 有路径，没有字节
#   所以新入口收「名字 + 字节」，两种调用方都能用。
#
# 【content 到底是什么】
#   就是 api.py 里 `await file.read()` 那个变量，类型是 bytes。
#   bytes = 还没解码的原始数据；要变成文字得靠 decode()。
#   对照：`content.decode("utf-8")` 就是「字节 → 文字」（这步现在在本函数内部做）。
#        load_document 原来那行 path.read_text(encoding="utf-8") 内部也是这两步。
# ============================================================================
def parse_bytes(filename: str, content: bytes) -> str:
    """名字 + 字节 -> 清洗后的正文；任何失败都 raise ValueError。

    filename：只用来看扩展名，以及报错时说清是哪个文件。
    content ：文件原始字节。本函数不读磁盘、不写磁盘。
    """
    suffix = Path(filename).suffix.lower()  # '报销.PDF' -> '.pdf'

    if suffix == ".md":
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError as e:
            raise ValueError(f"文件编码无法识别: {filename}") from e

    elif suffix == ".pdf":
        try:
            reader = PdfReader(io.BytesIO(content))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        except pypdf.errors.PyPdfError as e:
            raise ValueError(f"文件无法解析: {filename}") from e

    else:
        raise ValueError(f"只支持 .md / .pdf: {filename}")

    cleaned = clean_text(text)
    if not cleaned:
        raise ValueError(f"文件内容为空: {filename}")

    return cleaned


def load_document(path: str | Path) -> dict[str, str]:
    """读单个文件（.md / .pdf），返回 {"source": 文件名, "text": 清洗后正文}。"""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")

    content = path.read_bytes()
    text = parse_bytes(path.name, content)

    # ④ 包成原来的返回形状（source 用文件名，不是完整路径）。
    return {"source": path.name, "text": text}


# 直接运行本文件才执行演示；被其他文件导入时不执行。
if __name__ == "__main__":
    data_dir = Path(__file__).parent / "data"
    # 保留未清洗的文档，用于比较清洗前后的字符数。
    dirty_docs = []
    dirty_files = sorted(data_dir.glob("*.md"))
    for path in dirty_files:
        raw = path.read_text(encoding="utf-8")
        doc = {"source": path.name, "text": raw}
        dirty_docs.append(doc)

    print("清洗前的字数：")
    for num in dirty_docs:
        number = len(num["text"])
        print(f"{num['source']}: {number}字")

    print("清洗后的字数：")
    clean = load_documents(DATA_DIR)
    for num in clean:
        number = len(num["text"])
        print(f"{num['source']}: {number}字")

    # 用含多余空白的样本检查清洗效果。
    dirty = "  第一行  \r\n\r\n\r\n\t第二行\n\n\n\n第三行"

    cleaned = clean_text(dirty)
    print(cleaned)
    # repr 会显示换行、制表符等不可见字符，方便对比。
    # 同时检查空字符串和纯空格这两种边界输入。
    print("dirty repr:", repr(dirty))
    print("clean repr:", repr(clean_text(dirty)))
    print("空串:", repr(clean_text("")))
    print("纯空格:", repr(clean_text("   ")))

    # —— 今日新增：load_document 六个测试用例（samples/ 目录）——
    print("\n=== load_document 测试 ===")
    sample_dir = Path(__file__).parent / "samples"
    test_list = [
        "小样本.md",
        "报销制度.pdf",
        "空白页.pdf",
        "空文件.md",
        "空md文件.md",
        "未知格式.txt",
    ]
    for file in test_list:
        try:
            path = sample_dir / file
            print(load_document(path))
        except ValueError as v:
            print(v)
