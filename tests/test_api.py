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
    # Отвечающий мозг в списке, даже если ещё не сохранён.
    current = client.get("/api/brains").json()["current"]
    assert current in [model["id"] for model in data["data"]]


def test_stream_while_waking_is_503(client):
    from brain.server import host

    host.ready.clear()
    try:
        response = client.post(
            "/v1/chat/completions",
            json={"stream": True, "messages": [{"role": "user", "content": "Столица"}]},
        )
    finally:
        host.ready.set()

    assert response.status_code == 503


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
    assert client.post("/api/new", json={"name": "x", "size": "огромный"}).status_code == 400


def test_feedback_and_emotions(client):
    response = client.post("/api/feedback", json={"reward": 1}).json()
    assert response["emotion"]["name"] == "радость"

    client.post("/api/emotions", json={"enabled": False})
    assert client.get("/api/state").json()["emotionsEnabled"] is False
    client.post("/api/emotions", json={"enabled": True})


def test_background_reading_does_not_block_chat(client):
    import time

    text = "Столица Франции Париж.\nСтолица Германии Берлин.\n" * 60
    queued = client.post("/api/read", json={"name": "длинный", "text": text, "background": True}).json()
    assert queued["queued"] and queued["sentences"] > 50

    start = time.time()
    client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "Столица Франции"}]})
    # Ответ пришёл, пока чтение ещё идёт.
    assert client.get("/api/state").json()["reading"], "чтение закончилось раньше ответа — тест не проверил главное"
    assert time.time() - start < 30

    client.post("/api/stop", json={})
    for _ in range(100):
        if not client.get("/api/state").json()["reading"]:
            break
        time.sleep(0.2)
    assert not client.get("/api/state").json()["reading"]


def test_falls_asleep_only_when_tired(client):
    from brain.server import host

    cfg = host.mind.cfg
    brain = host.mind.brain
    adenosine = brain.chemistry_names.index("adenosine")
    level = float(brain.chemistry[adenosine])
    try:
        # Аденозина мало (утро) — не хочет спать, даже в долгой тишине.
        brain.chemistry[adenosine] = 0.0
        for _ in range(5):
            host.mind.language.pass_time(1.0)
        assert not host._should_fall_asleep(last_activity=0)

        # Вечер: рецепторы заняты — в тишине VLPO побеждает, засыпает.
        brain.chemistry[adenosine] = 10 * cfg.adenosine_half
        for _ in range(5):
            host.mind.language.pass_time(1.0)
        assert host._should_fall_asleep(last_activity=0)
        # Сразу после сообщения — не засыпает.
        assert not host._should_fall_asleep(last_activity=__import__("time").time())
    finally:
        brain.chemistry[adenosine] = level


def test_neuron_card(client):
    anatomy = client.get("/api/state").json()["anatomy"]
    meaning = next(a for a in anatomy["areas"] if a["name"] == "association")
    state = client.get("/api/state").json()["tokens"]

    # Нейрон из ансамбля «париж»: карточка должна знать это слово.
    cards = [
        client.get(f"/api/neuron/{meaning['E']['start'] + i}").json()
        for i in range(0, meaning["E"]["size"], max(1, meaning["E"]["size"] // 200))
    ]
    card = cards[0]
    assert card["area"] == "association" and card["type"] == "E"
    assert {"tuning", "predicts", "wokenBy", "incoming", "outgoing", "state"} <= card.keys()
    assert any(entry["word"] == "париж" for c in cards for entry in c["tuning"])

    assert client.get("/api/neuron/999999999").status_code == 404
    # Карточка — только чтение.
    assert client.get("/api/state").json()["tokens"] == state


def test_features_list_and_toggle(client):
    features = {f["name"]: f for f in client.get("/api/features").json()}
    assert "dendritic_context" in features and features["emotions_enabled"]["title"]
    assert all(feature["help"] for feature in features.values())

    after = client.post("/api/features", json={"name": "thinking_enabled", "enabled": True}).json()
    assert {f["name"]: f["enabled"] for f in after}["thinking_enabled"] is True
    client.post("/api/features", json={"name": "thinking_enabled", "enabled": False})

    assert client.post("/api/features", json={"name": "нет", "enabled": True}).status_code == 404
