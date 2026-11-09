# RAG Knowledge Agent · 文档知识库问答与工具助手

面向制度文档查询的个人项目：上传 Markdown/PDF，切分并生成向量，使用 Chroma 持久化检索，在网页返回回答与引用；同一检索能力可作为 LangGraph Agent 工具使用。

技术栈：Python / FastAPI / Pydantic / JavaScript / Chroma / OpenAI-compatible API / LangChain / LangGraph / Docker Compose。

## 功能

- 文档处理：Markdown/PDF 解析、文本清洗、重叠切分、按来源写入与替换。
- 知识问答：DashScope 向量化、Chroma 检索、距离阈值过滤、基于材料生成回答和来源展示。
- 网页与接口：文档上传、提问、来源列表与删除；共享口令校验和按 IP 查询限流。
- 工具 Agent：模型选择知识库检索与乘法工具；进程内多轮记忆、循环步数上限、连续工具失败终止。
- 检索评估：保留20道问题及查询改写、多路合并、LLM重排代码。评估入口与Web基础检索分开，不宣称Web已接通所有优化策略。

## 本地运行

需要 Python >=3.13 和 uv。所有命令从本仓库根目录执行。

```powershell
uv sync --frozen
Copy-Item .env.example .env
```

编辑 `.env`：配置实际可用的模型平台地址、模型名、模型密钥、DashScope密钥及自定访问口令。示例占位符不能直接调用。密钥只在本地保存。

```powershell
uv run uvicorn api:app --app-dir projects/rag-kb --host 127.0.0.1 --port 9000
```

打开 http://127.0.0.1:9000 ，输入 `.env` 中的 `DEMO_TOKEN`。首次提问时空索引会从 `projects/rag-kb/data/*.md` 构建；也可通过网页上传文档。实际向量化和模型回答会调用所配置的外部API。

## Docker运行

先填好本地 `.env`，再运行：

```powershell
docker compose up -d --build
```

访问地址仍为 http://127.0.0.1:9000 。索引和上传文件保存在命名卷，重建容器不会自动删除；`.env` 不进入镜像。若模型服务运行在宿主机，容器中的地址应使用 `host.docker.internal`，不能使用容器自身的 `127.0.0.1`。

## Agent与检索评估

```powershell
uv run python projects/agent-lab/multi_tool_agent.py
uv run python projects/rag-kb/eval_run.py
```

Agent入口运行内置复合问答示例；检索工具与Web共享索引。评估入口使用独立内存索引，可能多次调用付费API；人工标注的样本统计不等于泛化准确率。

## 接口

下列接口均需要请求头 `X-Demo-Token`。未配置服务口令返回503，未带或带错口令返回401；鉴权在查询限流之前执行。

| 方法 | 路径 | 用途 |
|---|---|---|
| POST | `/upload` | 上传Markdown/PDF（multipart字段file） |
| POST | `/query` | 提问，JSON包含question和可选k（1—10） |
| GET | `/sources` | 列出来源及块数 |
| DELETE | `/source/{name}` | 删除来源对应索引记录 |

## 目录

- `projects/rag-kb/`：文档处理、索引、问答API、网页、评估数据及策略。
- `projects/agent-lab/`：LangGraph工具Agent及检索工具。
- `tests/`：使用模拟向量和模型响应的离线回归，不消耗API额度，不代表真实检索质量。
- `pyproject.toml` / `uv.lock`：依赖声明与锁定版本。
- `Dockerfile` / `compose.yaml` / `.env.example`：运行环境与配置示例。

```powershell
uv run pytest -q
```

## 实现边界

- Web目前为普通JSON问答，不是流式响应；Agent入口为命令行示例。
- 共享口令用于本地演示，不提供多用户、权限分级或生产级认证。
- 按IP限流仅在当前进程内；代理可能合并来源IP。
- Agent记忆只在进程内，进程退出后不保留。
- 非空索引不会自动核对完整性；删除来源只移除索引记录，保留原文件。
- 检索距离阈值依赖模型、索引及样本，可能拒绝有答案的问题。
- `data/*.md`是注明公开出处的示例语料，仅供检索演示，不构成法律或人事建议。

本仓库由个人原始项目代码整理导出，移除了教学手册、教练提示词、周练习、学习日报、求职资料、运行缓存和真实凭据。源快照：`00d37642ce3a81473279180cf56ae51f1dc5b681`。
