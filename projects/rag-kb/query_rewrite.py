import json
import os
from pathlib import Path

from dotenv import load_dotenv
from qa import MODEL, client

# 文档和 .env 都根据本文件位置定位，不依赖启动目录。
DATA_DIR = Path(__file__).parent / "data"
ENV_FILE = Path(__file__).parent.parent.parent / ".env"
BASE_URL = os.getenv("DEEPSEEK_BASE_URL") or "http://127.0.0.1:8787/v1"

load_dotenv(ENV_FILE)
ds_key = os.getenv("DEEPSEEK_API_KEY")
if not ds_key:
    raise SystemExit("没找到 DEEPSEEK_API_KEY，请检查 .env")


def rewrite_query(question: str) -> list[str]:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "将你接收到的question换两种检索问法，例如用户问“出差吃饭补贴怎么算？”，模型可以换成“出差伙食补助标准是什么？”，输出的格式严格按照json数组返回，不要带解释、编号或 Markdown 代码围栏，只要数组本身",
            },
            {"role": "user", "content": f"{question}"},
        ],
        extra_body={"thinking": {"type": "disabled"}},  # 关掉思考模式，省 token
    )
    reply = resp.choices[0].message.content
    if reply is None:
        return [question]

    try:
        answer = json.loads(reply)
    except json.JSONDecodeError:
        return [question]
    else:
        if isinstance(answer, list) and len(answer) == 2:
            for item in answer:
                if isinstance(item, str):
                    if not item.strip():
                        return [question]
                else:
                    return [question]
            return answer
        else:
            return [question]


if __name__ == "__main__":
    question = "出差吃饭补贴怎么算？"
    print(rewrite_query(question))
