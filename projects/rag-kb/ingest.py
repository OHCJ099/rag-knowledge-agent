from pathlib import Path

from config import CHUNK_SIZE, OVERLAP
from index_store import add_chunks, delete_source, open_index
from loader import load_document
from retriever import embed
from splitter import split_text


def ingest_file(
    path: str | Path, index_path: str | Path, replace: bool = False, loaded_text=None
) -> int:

    if loaded_text is not None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {path}")
        file = {"source": path.name, "text": loaded_text}

    else:
        file = load_document(path)

    chunks = split_text(file["text"], CHUNK_SIZE, OVERLAP)
    if len(chunks) == 0:
        raise ValueError(f"文件{file['source']}: chunks块为0")

    embeddings = []
    for chunk in chunks:
        embeddings.append(embed(chunk))

    collection = open_index(index_path)

    # ★ 覆盖逻辑加在这里。想清楚两件事：
    if replace:
        delete_source(collection, file["source"])
        #   ② 这一步放在「算向量」之前还是之后？为什么？
    add_count = add_chunks(collection, file["source"], chunks, embeddings)
    return add_count
