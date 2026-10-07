"""HTTP API: OpenAI / llama.cpp-совместимость и свои ручки."""

import json

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    from brain.server import app, host

    with TestClient(app) as client:
        host.ready.wait(timeout=300)
        client.post("/api/read", json={"name": "факты", "text": "Столица Франции Париж. " * 3})
        yield client


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_models_lists_brains(client):
    data = client.get("/v1/models").json()
    assert data["object"] == "list"
    assert all(model["id"].endswith(".pt") for model in data["data"])


def test_chat_completion(client):
    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "Столица Франции"}]},
    ).json()

    assert response["object"] == "chat.completion"
    assert response["choices"][0]["message"]["role"] == "assistant"
    assert response["usage"]["prompt_tokens"] == 2
    assert "confidence" in response["brain"]


def test_chat_completion_stream(client):
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={"stream": True, "messages": [{"role": "user", "content": "Столица Франции"}]},
    ) as response:
        events = [line[6:] for line in response.iter_lines() if line.startswith("data: ")]

    assert events[-1] == "[DONE]"
    final = json.loads(events[-2])
    assert final["choices"][0]["finish_reason"] == "stop"
    assert "text" in final["brain"]


def test_completion_is_read_only(client):
    before = client.get("/api/state").json()["tokens"]
    response = client.post("/completion", json={"prompt": "столица франции"}).json()

    assert response["stop"] is True
    assert client.get("/api/state").json()["tokens"] == before


def test_command_and_errors(client):
    output = client.post("/api/command", json={"text": "/predict столица"}).json()["output"]
    assert output and "мозг ожидает" in output[0]["text"] or "нет ожидания" in output[0]["text"]

    assert client.post("/api/load", json={"name": "nope.pt"}).status_code == 404


def test_feedback_and_emotions(client):
    response = client.post("/api/feedback", json={"reward": 1}).json()
    assert response["emotion"]["name"] == "радость"

    client.post("/api/emotions", json={"enabled": False})
    assert client.get("/api/state").json()["emotionsEnabled"] is False
    client.post("/api/emotions", json={"enabled": True})
