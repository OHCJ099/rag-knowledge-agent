"""Exercise exported API wiring with local Chroma and fake model/vector responses."""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "projects" / "rag-kb"))
os.environ["DEEPSEEK_API_KEY"] = "offline-test-placeholder"
os.environ["DASHSCOPE_API_KEY"] = "offline-test-placeholder"
os.environ["DEEPSEEK_BASE_URL"] = "http://127.0.0.1:1/v1"
os.environ["DEMO_TOKEN"] = "offline-demo-token"
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import api
import config
import ingest
import loader
import query_lab
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "DEMO_TOKEN", "offline-demo-token")
    monkeypatch.setattr(api, "INDEX_PATH", tmp_path / "index")
    monkeypatch.setattr(api, "UPLOAD_PATH", tmp_path / "uploads")
    monkeypatch.setattr(config, "COST_LOG_PATH", str(tmp_path / "cost.jsonl"))
    monkeypatch.setattr(ingest, "embed", lambda text: [1.0, 0.0, 0.0])
    monkeypatch.setattr(query_lab, "embed", lambda text: [1.0, 0.0, 0.0])
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="依据材料回答。[1]"))],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
    )
    monkeypatch.setattr(
        api.qa.client.chat.completions, "create", lambda **kwargs: response
    )
    api._buckets.clear()
    with TestClient(api.app) as test_client:
        yield test_client


AUTH = {"X-Demo-Token": "offline-demo-token"}


def test_static_page(client):
    assert client.get("/").status_code == 200


@pytest.mark.parametrize("headers", [{}, {"X-Demo-Token": "wrong"}])
def test_sources_requires_token(client, headers):
    assert client.get("/sources", headers=headers).status_code == 401


def test_missing_server_token_fails_closed(client, monkeypatch):
    monkeypatch.setattr(api, "DEMO_TOKEN", "")
    assert client.get("/sources", headers=AUTH).status_code == 503


def test_invalid_token_does_not_consume_rate_limit(client):
    for _ in range(6):
        response = client.post("/query", json={"question": "年假"})
        assert response.status_code == 401
    assert api._buckets == {}
    response = client.post("/query", headers=AUTH, json={"question": " "})
    assert response.status_code == 400


def test_upload_query_and_delete_flow(client):
    response = client.post(
        "/upload", headers=AUTH,
        files={"file": ("demo.md", "年假为五天。".encode(), "text/markdown")},
    )
    assert response.status_code == 200
    assert response.json()["blocks"] > 0
    sources = client.get("/sources", headers=AUTH).json()["sources"]
    assert sources[0]["source"] == "demo.md"
    response = client.post("/query", headers=AUTH, json={"question": "年假几天"})
    assert response.status_code == 200
    assert response.json()["sources"][0][0] == "demo.md"
    assert "[1]" in response.json()["answer"]
    assert client.delete("/source/demo.md", headers=AUTH).status_code == 200
    assert client.get("/sources", headers=AUTH).json() == {"sources": []}
    assert client.delete("/source/demo.md", headers=AUTH).status_code == 404


def test_rejects_empty_document(client):
    response = client.post(
        "/upload", headers=AUTH,
        files={"file": ("empty.md", b"  ", "text/markdown")},
    )
    assert response.status_code == 400


def test_rejects_invalid_k(client):
    response = client.post("/query", headers=AUTH, json={"question": "年假", "k": 0})
    assert response.status_code == 422


def test_sample_documents_are_available():
    docs = loader.load_documents(ROOT / "projects" / "rag-kb" / "data")
    assert len(docs) == 5
    assert all(doc["text"].strip() for doc in docs)


def test_agent_tools_import_without_demo_client():
    sys.path.insert(0, str(ROOT / "projects" / "agent-lab"))
    import tools
    import rag_tool
    import multi_tool_agent

    assert tools.multiply.invoke({"a": 3, "b": 4}) == "12.0"
    assert "问题为空" in rag_tool.search_kb.invoke(" ")
    assert multi_tool_agent.app is not None
