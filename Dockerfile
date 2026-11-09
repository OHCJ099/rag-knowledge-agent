FROM python:3.13-slim
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN pip install --no-cache-dir uv && uv sync --frozen --no-dev
COPY projects/rag-kb ./projects/rag-kb
ENV PATH="/app/.venv/bin:$PATH"
CMD ["uvicorn", "api:app", "--app-dir", "projects/rag-kb", "--host", "0.0.0.0", "--port", "8000"]
