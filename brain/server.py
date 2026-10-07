"""
Веб-интерфейс мозга: живые спайки всех зон + чат.

    python -m brain.server            # http://localhost:8000

Мозг живёт в одном рабочем потоке (GPU). Браузер получает
по WebSocket анатомию, спайки (раз в ~30 мс пачкой),
каждое услышанное слово с ожиданием и dopamine, и ответы.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
import os
import queue
import threading
import time

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from brain.commands import Session
from brain.config import BrainConfig
from brain.language import Reply, Step
from brain.mind import Mind


WEB_DIST = Path(__file__).resolve().parent.parent / "web" / "dist"
SPIKE_FLUSH_SECONDS = 0.03


class BrainHost:
    """Мозг в отдельном потоке + рассылка событий всем вкладкам."""

    def __init__(self):
        self.loop: asyncio.AbstractEventLoop | None = None
        self.clients: set[asyncio.Queue] = set()
        self.jobs: queue.Queue = queue.Queue()
        self.ready = threading.Event()
        self.busy = False

        self.mind: Mind | None = None
        self.session: Session | None = None

        self._chunks: list[list[int]] = []
        self._chunks_time = 0.0
        self._last_flush = 0.0
        self._lock = threading.Lock()

    # ==============================================================
    # РАССЫЛКА
    # ==============================================================

    def emit(self, message: dict) -> None:
        if self.loop is None:
            return

        def put():
            for client in list(self.clients):
                if client.qsize() < 2000:
                    client.put_nowait(message)

        self.loop.call_soon_threadsafe(put)

    def _on_chunk(self, fired, time_ms: float) -> None:
        with self._lock:
            self._chunks.append(fired.tolist())
            self._chunks_time = time_ms

        if time.perf_counter() - self._last_flush > SPIKE_FLUSH_SECONDS:
            self.flush_spikes()

    def flush_spikes(self) -> None:
        with self._lock:
            chunks, self._chunks = self._chunks, []

        self._last_flush = time.perf_counter()

        if chunks:
            self.emit({"type": "spikes", "t": self._chunks_time, "chunks": chunks})

    def _on_step(self, step: Step, learn: bool) -> None:
        self.flush_spikes()
        self.emit(
            {
                "type": "step",
                "token": step.token,
                "expected": step.expected,
                "error": step.error,
                "dopamine": step.dopamine,
                "learn": learn,
            }
        )

    def _out(self, text: str, kind: str) -> None:
        self.emit({"type": "out", "kind": kind, "text": text})

    def _on_reply(self, reply: Reply) -> None:
        report = reply.report
        self.emit(
            {
                "type": "reply",
                "text": reply.text,
                "promptTokens": report.tokens,
                "promptSeconds": reply.prompt_seconds,
                "generated": reply.generated,
                "generateSeconds": reply.generate_seconds,
                "hits": report.hits,
                "predicted": len(report.errors),
                "dopamine": report.dopamine,
            }
        )

    # ==============================================================
    # СОСТОЯНИЕ ДЛЯ ИНТЕРФЕЙСА
    # ==============================================================

    def anatomy(self) -> dict:
        brain = self.mind.brain

        def pop(p):
            return {"start": p.start, "size": p.size}

        return {
            "type": "anatomy",
            "device": str(brain.device),
            "neurons": brain.n,
            "synapses": brain.synapse_count(),
            "areas": [
                {
                    "name": a.name,
                    "title": a.title,
                    "kind": a.kind,
                    "E": pop(a.E),
                    "I": pop(a.I),
                }
                for a in brain.areas.values()
            ],
            "pathways": [
                {
                    "src": p.src,
                    "dst": p.dst,
                    "plastic": p.plastic,
                    "synapses": brain.projection(f"{p.src} -> {p.dst}").weight.numel(),
                }
                for p in self.mind.cfg.pathways
            ],
        }

    def status(self) -> dict:
        brain = self.mind.brain
        language = self.mind.language

        return {
            "type": "status",
            "busy": self.busy,
            "words": len(language.semantic),
            "tokens": language.experienced_tokens,
            "dialogTurns": language.dialog_turns,
            "timeMs": brain.time_ms,
            "grown": brain.synapses_grown,
            "plastic": [
                {
                    "name": p.name,
                    "synapses": p.weight.numel(),
                    "learned": int((p.weight > p.innate_max).sum()),
                    "mean": float(p.weight.mean()),
                }
                for p in brain.plastic
            ],
        }

    # ==============================================================
    # РАБОЧИЙ ПОТОК
    # ==============================================================

    def start(self) -> None:
        threading.Thread(target=self._run, name="brain", daemon=True).start()

    def _run(self) -> None:
        cfg = BrainConfig()
        self.mind, restored = Mind.open(cfg)
        self._attach()
        self.ready.set()
        self.emit(self.anatomy())
        self.emit(self.status())
        self._out(
            ("Загружен сохранённый мозг: " if restored else "Новый мозг: ")
            + cfg.save_path,
            "info",
        )

        while True:
            job = self.jobs.get()

            if job is None:
                break

            kind, payload = job
            self.busy = True
            self.emit(self.status())

            try:
                if kind == "line":
                    mind = self.session.mind
                    self.session.handle(payload)

                    if self.session.mind is not mind:
                        # /load — новый мозг, новая анатомия.
                        self.mind = self.session.mind
                        self._attach()
                        self.emit(self.anatomy())
                elif kind == "read_text":
                    name, text = payload
                    self.session.stop.clear()
                    self.session.read_text(name, text)

                    if self.mind.cfg.autosave:
                        self.mind.save()
            except Exception as error:  # noqa: BLE001 — показать в чате, не уронить мозг
                self._out(f"Ошибка: {error}", "info")
            finally:
                self.flush_spikes()
                self.busy = False
                self.emit(self.status())

    def _attach(self) -> None:
        self.mind.brain.on_chunk = self._on_chunk
        self.mind.language.on_step = self._on_step
        self.session = Session(self.mind, self._out, self._on_reply)

    def shutdown(self) -> None:
        self.jobs.put(None)

        if self.mind is not None and self.mind.cfg.autosave:
            self.mind.save()


host = BrainHost()


@asynccontextmanager
async def lifespan(app: FastAPI):
    host.loop = asyncio.get_running_loop()
    host.start()
    yield
    host.shutdown()


app = FastAPI(lifespan=lifespan)


@app.websocket("/ws")
async def socket(ws: WebSocket) -> None:
    await ws.accept()
    outbox: asyncio.Queue = asyncio.Queue()
    host.clients.add(outbox)

    if host.ready.is_set():
        await ws.send_json(host.anatomy())
        await ws.send_json(host.status())

    async def sender():
        while True:
            await ws.send_json(await outbox.get())

    task = asyncio.create_task(sender())

    try:
        while True:
            message = await ws.receive_json()
            kind = message.get("type")

            if kind == "line":
                host.jobs.put(("line", str(message.get("text", ""))))
            elif kind == "read_text":
                host.jobs.put(("read_text", (str(message.get("name", "файл")), str(message.get("text", "")))))
            elif kind == "stop" and host.session is not None:
                host.session.stop.set()
    except WebSocketDisconnect:
        pass
    finally:
        task.cancel()
        host.clients.discard(outbox)


if WEB_DIST.exists():
    app.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="web")
else:
    @app.get("/")
    def index() -> HTMLResponse:
        return HTMLResponse(
            "<p>Интерфейс не собран: <code>cd web && npm install && npm run build</code></p>"
        )


def main() -> None:
    import uvicorn

    uvicorn.run(
        app,
        host=os.environ.get("BRAIN_HOST", "127.0.0.1"),
        port=int(os.environ.get("BRAIN_PORT", "8000")),
    )


if __name__ == "__main__":
    main()
