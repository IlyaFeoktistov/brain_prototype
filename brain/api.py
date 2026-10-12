"""
HTTP API мозга — как у llama.cpp / OpenAI, чтобы подключить любой клиент.

OpenAI-совместимые:
    GET  /v1/models                 мозги (*.pt) как «модели»
    POST /v1/chat/completions       диалог: мозг учится на реплике и отвечает
                                    (stream: true — SSE по словам)
llama.cpp-совместимые:
    GET  /health
    POST /completion                продолжить текст БЕЗ обучения
                                    (learn: true — с обучением)
Свои:
    GET  /api/state                 состояние, эмоции, выученное
    POST /api/command   {text}      любая команда чата: /predict, /stats…
    POST /api/read      {name,text} прочитать текст (обучение)
    POST /api/feedback  {reward}    +1 похвала, -1 ругань
    GET  /api/brains                сохранённые мозги
    POST /api/load      {name}      переключиться на другой мозг
    POST /api/new       {name,size} новый мозг (size — tiny/small/… или число)
    POST /api/emotions  {enabled}   включить / выключить эмоции
    POST /api/thinking  {enabled}   размышлять перед ответом
    POST /api/live      {enabled}   живой режим: думает в тишине, сам заговаривает
    POST /api/initiative {enabled}  заговаривать первым (реплики — WebSocket
                                    событие "initiative")
    GET  /api/features              функции мозга: [{name, title, help, enabled}]
    POST /api/features {name,enabled} включить / выключить функцию
    POST /api/stop                  остановить чтение

Живая активность нейронов — WebSocket /ws (см. brain/server.py).

У ответов есть расширение "brain": уверенность каждого слова, эмоция,
dopamine, сколько слов мозг угадал, — этого нет у LLM.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from brain.config import FEATURES


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    stream: bool = False


class CompletionRequest(BaseModel):
    prompt: str
    stream: bool = False
    learn: bool = False


class TextRequest(BaseModel):
    text: str


class ReadRequest(BaseModel):
    name: str = "текст"
    text: str
    # True — в очередь, мозг читает порциями между сообщениями.
    background: bool = False


class FeedbackRequest(BaseModel):
    reward: float = Field(description="> 0 — похвала, < 0 — ругань")


class LoadRequest(BaseModel):
    name: str


class NewBrainRequest(BaseModel):
    name: str
    # Пусто — как у текущего мозга.
    size: str = ""


class EmotionsRequest(BaseModel):
    enabled: bool


class ThinkingRequest(BaseModel):
    enabled: bool


class InitiativeRequest(BaseModel):
    enabled: bool


class FeatureRequest(BaseModel):
    name: str
    enabled: bool


def _features(session) -> list[dict]:
    return [
        {"name": name, "title": FEATURES[name][0], "help": FEATURES[name][2], "enabled": enabled}
        for name, enabled in session.mind.features().items()
    ]


def _sse(data: dict | str) -> str:
    payload = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
    return f"data: {payload}\n\n"


def build_router(host) -> APIRouter:
    router = APIRouter()

    async def call(fn, sink=None):
        if not host.ready.is_set():
            raise HTTPException(503, "Мозг ещё просыпается")

        return await asyncio.wrap_future(host.submit(fn, sink))

    def brain_extra(session, reply=None) -> dict:
        language = session.language
        extra = {
            "brain": _current(session),
            "emotion": language.emotions.state() if session.mind.cfg.emotions_enabled else None,
        }

        if reply is not None:
            extra.update(
                confidence=reply.confidence,
                dopamine=reply.report.dopamine,
                guessed=reply.report.hits,
                predicted=len(reply.report.errors),
                prompt_seconds=reply.prompt_seconds,
                generate_seconds=reply.generate_seconds,
                thoughts=reply.thoughts,
                deliberation=reply.deliberation,
                unknown=reply.unknown,
                praised=reply.praised,
            )

        return extra

    def streamed(fn, wrap):
        """
        Запустить fn в потоке мозга и отдавать слова по мере выбора.
        wrap(kind, value) -> строка SSE: kind = "token" | "done".
        """

        # Проверить до начала ответа: после первого байта статус уже не
        # поменять, и 503 превращается в оборванное соединение.
        if not host.ready.is_set():
            raise HTTPException(503, "Мозг ещё просыпается")

        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def sink(text: str, kind: str) -> None:
            if kind == "token":
                token, _, sure = text.partition("\t")
                loop.call_soon_threadsafe(queue.put_nowait, ("token", (token, float(sure or 0))))

        async def events():
            future = asyncio.ensure_future(call(fn, sink))
            future.add_done_callback(lambda _: queue.put_nowait(("end", None)))

            while True:
                kind, value = await queue.get()

                if kind == "end":
                    break

                yield wrap("token", value)

            try:
                result = future.result()
            except Exception as error:
                # Ответ уже идёт — ошибку отдать событием, а не обрывом.
                yield _sse({"error": {"message": str(error)}})
                return

            yield wrap("done", result)

        return StreamingResponse(events(), media_type="text/event-stream")

    # ==============================================================
    # llama.cpp
    # ==============================================================

    @router.get("/health")
    async def health():
        return {"status": "ok" if host.ready.is_set() else "loading model"}

    @router.post("/completion")
    async def completion(request: CompletionRequest):
        def run(session):
            if request.learn:
                return session.language.converse(request.prompt)

            # Без обучения: продолжение в изоляции, мозг не меняется.
            return session.language.generate(request.prompt)

        def finish(session_result, session):
            if request.learn:
                reply = session_result
                return {
                    "content": reply.text,
                    "tokens_evaluated": reply.report.tokens,
                    "tokens_predicted": reply.generated,
                    "stop": True,
                    "brain": brain_extra(session, reply),
                }

            produced = session_result
            return {
                "content": " ".join(t for t, _ in produced),
                "tokens_evaluated": len(session.language.tokenize(request.prompt)),
                "tokens_predicted": len(produced),
                "stop": True,
                "brain": {**brain_extra(session), "confidence": [c for _, c in produced]},
            }

        if not request.stream:
            return await call(lambda session: finish(run(session), session))

        def run_streaming(session):
            session.language.on_speak = lambda token, sure: session.out(f"{token}\t{sure:.3f}", "token")
            try:
                return finish(run(session), session)
            finally:
                session.language.on_speak = None

        def wrap(kind, value):
            if kind == "token":
                token, sure = value
                return _sse({"content": token, "stop": False, "brain": {"confidence": sure}})
            return _sse({**value, "content": ""})

        return streamed(run_streaming, wrap)

    # ==============================================================
    # OpenAI
    # ==============================================================

    @router.get("/v1/models")
    async def models():
        def run(session):
            names = [path.name for path in session.brain_files()]
            # Новый мозг ещё не сохранён, но отвечает именно он.
            current = _current(session)
            return names if current in names else [current, *names]

        names = await call(run)
        return {
            "object": "list",
            "data": [{"id": name, "object": "model", "owned_by": "brain_prototype"} for name in names],
        }

    @router.post("/v1/chat/completions")
    async def chat_completions(request: ChatRequest):
        user = next((m.content for m in reversed(request.messages) if m.role == "user"), None)

        if not user:
            raise HTTPException(400, "Нужно хотя бы одно сообщение role=user")

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        def run(session):
            # Модель = файл мозга: если просят другой — переключиться.
            if request.model and request.model != _current(session):
                session.handle(f"/load {request.model}")

            reply = session.say(user)
            return {
                "reply": reply,
                "model": _current(session),
                "extra": brain_extra(session, reply),
            }

        def body(result):
            reply = result["reply"]
            return {
                "id": completion_id,
                "object": "chat.completion",
                "created": created,
                "model": result["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": reply.text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": reply.report.tokens,
                    "completion_tokens": reply.generated,
                    "total_tokens": reply.report.tokens + reply.generated,
                },
                "brain": result["extra"],
            }

        if not request.stream:
            return body(await call(run))

        def wrap(kind, value):
            if kind == "token":
                token, sure = value
                return _sse(
                    {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "choices": [{"index": 0, "delta": {"content": token + " "}, "finish_reason": None}],
                        "brain": {"confidence": sure},
                    }
                )

            full = body(value)
            final = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": full["model"],
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": full["usage"],
                "brain": {**full["brain"], "text": full["choices"][0]["message"]["content"]},
            }
            return _sse(final) + _sse("[DONE]")

        return streamed(run, wrap)

    # ==============================================================
    # Свои ручки
    # ==============================================================

    @router.get("/api/state")
    async def state():
        if not host.ready.is_set():
            return {"ready": False}

        return {"ready": True, **host.status(), "brains": host.brains(), "anatomy": host.anatomy()}

    @router.post("/api/command")
    async def command(request: TextRequest):
        output: list[dict] = []

        def sink(text: str, kind: str) -> None:
            if kind in ("info", "brain"):
                output.append({"kind": kind, "text": text})

        await call(lambda session: session.handle(request.text), sink)
        return {"output": output}

    @router.post("/api/read")
    async def read(request: ReadRequest):
        if request.background:
            sentences = await call(lambda session: host.enqueue_read(request.name, request.text))
            return {"queued": True, "sentences": sentences}

        def run(session):
            session.stop.clear()
            report = session.read_text(request.name, request.text)
            session.autosave(force=True)
            return {
                "tokens": report.tokens,
                "sentences": report.sentences,
                "guessed": report.hits,
                "predicted": len(report.errors),
                "prediction_error": report.mean_error(),
                "dopamine": report.dopamine,
            }

        return await call(run)

    @router.post("/api/feedback")
    async def feedback(request: FeedbackRequest):
        word = "+" if request.reward > 0 else "-"

        def run(session):
            reply = session.language.converse(word)
            return {"text": reply.text, **brain_extra(session)}

        return await call(run)

    @router.get("/api/brains")
    async def brains():
        return host.brains() if host.ready.is_set() else {"items": []}

    @router.get("/api/neuron/{index}")
    async def neuron(index: int):
        """Карточка нейрона (brain/neuron.py): что он знает. Только чтение."""

        from brain.neuron import describe

        def run(session):
            card = describe(session.mind, index)

            if card is None:
                raise HTTPException(404, f"Нет нейрона {index}")

            return card

        return await call(run)

    @router.post("/api/load")
    async def load(request: LoadRequest):
        def run(session):
            if session.resolve_brain(request.name) is None:
                raise HTTPException(404, f"Нет такого мозга: {request.name}")
            session.handle(f"/load {request.name}")
            return {"brain": _current(session), "neurons": session.mind.brain.n}

        return await call(run)

    @router.post("/api/new")
    async def new_brain(request: NewBrainRequest):
        def run(session):
            before = session.mind
            text = session.new_brain(f"{request.name} {request.size}".strip())

            if session.mind is before:
                raise HTTPException(400, text)

            return {"brain": _current(session), "neurons": session.mind.brain.n, "text": text}

        return await call(run)

    @router.post("/api/emotions")
    async def emotions(request: EmotionsRequest):
        def run(session):
            session.mind.cfg.emotions_enabled = request.enabled
            return brain_extra(session)

        return await call(run)

    @router.post("/api/thinking")
    async def thinking(request: ThinkingRequest):
        def run(session):
            session.mind.cfg.thinking_enabled = request.enabled
            return {"thinking": request.enabled}

        return await call(run)

    @router.get("/api/features")
    async def features():
        def run(session):
            return _features(session)

        return await call(run)

    @router.post("/api/features")
    async def set_feature(request: FeatureRequest):
        def run(session):
            if request.name not in FEATURES:
                raise HTTPException(404, f"Нет такой функции: {request.name}")
            session.mind.set_feature(request.name, request.enabled)
            return _features(session)

        return await call(run)

    @router.post("/api/initiative")
    async def initiative(request: InitiativeRequest):
        def run(session):
            session.mind.cfg.initiative_enabled = request.enabled
            return {"initiative": request.enabled}

        return await call(run)

    @router.post("/api/live")
    async def live(request: InitiativeRequest):
        def run(session):
            session.mind.cfg.live_mode = request.enabled
            return {"live": request.enabled}

        return await call(run)

    @router.post("/api/stop")
    async def stop():
        if host.session is not None:
            host.session.stop.set()
        return {"stopping": True}

    return router


def _current(session) -> str:
    from pathlib import Path

    return Path(session.mind.cfg.save_path).name
