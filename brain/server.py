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
from collections import deque
from concurrent.futures import Future
import queue
import threading
import time

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from brain.commands import Session, report_line
from brain.language import ReadReport
from brain.config import FEATURES, BrainConfig
from brain.language import Reply, Step
from brain.mind import Mind
from brain.api import build_router
from brain.live import Liveliness


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
        # Спит во время тишины (между порциями сна тоже).
        self.dozing = False
        # Сколько проигрываний осталось в текущем сне.
        self.sleep_left = 0
        # Очередь фонового чтения: мозг читает порциями между сообщениями.
        self.reading: deque = deque()
        # Когда мозг последний раз заговорил сам и думал в тишине.
        self.live = Liveliness()

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
                "dream": self.mind.language.sleeping,
                "musing": self.mind.language.musing,
            }
        )
        self._emit_emotion()

    def _emit_emotion(self) -> None:
        if self.mind.cfg.emotions_enabled:
            self.emit({"type": "emotion", **self.mind.language.emotions.state()})

    def _out(self, text: str, kind: str) -> None:
        if kind == "face":
            self._emit_emotion()
            return

        if kind == "thought":
            thought, verdict, score = text.split("\t")
            self.emit({"type": "thought", "thought": thought, "verdict": verdict, "score": float(score)})
            return

        if kind == "token":
            token, _, sure = text.partition("\t")
            self.flush_spikes()
            self.emit({"type": "speak", "token": token, "confidence": float(sure or 0)})
        else:
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
                "confidence": reply.confidence,
            }
        )

    # ==============================================================
    # СОСТОЯНИЕ ДЛЯ ИНТЕРФЕЙСА
    # ==============================================================

    def anatomy(self) -> dict:
        brain = self.mind.brain

        def pop(p):
            return {"start": p.start, "size": p.size}

        language = self.mind.language
        grids = {}

        if self.mind.cfg.concepts_enabled:
            grids["association"] = language.concepts.cols

        if self.mind.cfg.memory_places:
            grids["prefrontal"] = language.memory.cols

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
                    # Зона-карта (места слов): E лежат сеткой в столько столбцов.
                    "grid": grids.get(a.name),
                }
                for a in brain.areas.values()
            ],
            # Путь берётся из мозга: «слух -> смысл» при понятиях учится.
            "pathways": [
                {
                    "src": p.src,
                    "dst": p.dst,
                    "plastic": q.plastic if q else p.plastic,
                    "synapses": (getattr(q, "n_synapses", None) or q.weight.numel()) if q else 0,
                }
                for p in self.mind.cfg.pathways
                for q in [next((q for q in brain.projections if q.name == f"{p.src} -> {p.dst}"), None)]
            ],
        }

    def brains(self) -> dict:
        current = Path(self.mind.cfg.save_path).expanduser().resolve()

        return {
            "type": "brains",
            "current": current.name,
            "items": [
                {
                    "name": path.name,
                    "megabytes": round(path.stat().st_size / 2**20),
                    "modified": path.stat().st_mtime,
                }
                for path in self.session.brain_files()
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
            "emotionsEnabled": self.mind.cfg.emotions_enabled,
            "sleeping": language.sleeping or self.dozing,
            "thinking": self.mind.cfg.thinking_enabled,
            "initiative": self.mind.cfg.initiative_enabled,
            "live": self.mind.cfg.live_mode,
            "features": [
                {"name": name, "title": FEATURES[name][0], "help": FEATURES[name][2], "enabled": enabled}
                for name, enabled in self.mind.features().items()
            ],
            "reading": [
                {"name": item["name"], "done": item["done"], "total": len(item["sentences"])}
                for item in self.reading
            ],
            "sleepPressure": language.sleep_pressure,
            "episodes": len(language.hippocampus),
            "emotion": language.emotions.state(),
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
        self.thread = threading.Thread(target=self._run, name="brain", daemon=True)
        self.thread.start()

    def _run(self) -> None:
        cfg = BrainConfig()
        self.mind, restored = Mind.open(cfg)
        self._attach()
        self.ready.set()
        self.emit(self.anatomy())
        self.emit(self.status())
        self.emit(self.brains())
        self._out(
            ("Загружен сохранённый мозг: " if restored else "Новый мозг: ")
            + cfg.save_path,
            "info",
        )

        last_activity = time.time()

        while True:
            try:
                # Пока есть что читать — не ждать: читать порцию.
                job = self.jobs.get(timeout=0.0 if self.reading else 1.0)
            except queue.Empty:
                if self.reading and not self.dozing:
                    # Устал читать — уснуть (порциями, будится сообщением),
                    # выспаться и читать дальше. Внутри порции не спит,
                    # иначе чат ждал бы минуты сна.
                    language = self.mind.language
                    cfg = self.mind.cfg

                    if cfg.nap_while_reading and language.sleep_pressure >= cfg.nap_after_episodes:
                        self.sleep_left = language.sleep_pressure * cfg.replays_per_episode
                        self._out(
                            f"Устал читать: новых впечатлений {language.sleep_pressure}, "
                            f"посплю ({self.sleep_left} проигрываний) и продолжу",
                            "info",
                        )
                        self._doze(True)
                    else:
                        self._execute(lambda session: self._read_portion(session), Future(), None, quiet=True)
                        continue

                # В тишине эмоции успокаиваются.
                if self.mind.cfg.emotions_enabled:
                    self.mind.language.pass_time(1.0)

                # Живой режим: заговорить первым (повод — эмоция)
                # или подумать про себя.
                if self._should_speak_up(last_activity):
                    self._speak_up()
                elif self._should_wander(last_activity):
                    self._wander()

                # Устал и тихо — засыпает надолго: длительность сна задаётся
                # при засыпании (N проигрываний на каждое новое впечатление).
                if not self.dozing and self._should_fall_asleep(last_activity):
                    language = self.mind.language
                    self.sleep_left = language.sleep_pressure * self.mind.cfg.replays_per_episode
                    self._out(
                        f"Засыпаю: новых впечатлений {language.sleep_pressure}, "
                        f"сон на {self.sleep_left} проигрываний",
                        "info",
                    )
                    self._doze(True)

                # «Остановить» во сне: проснуться и бросить чтение.
                if self.dozing and self.session.stop.is_set():
                    self.session.stop.clear()
                    self.sleep_left = 0
                    self._doze(False)
                    if self.reading:
                        self.reading.clear()
                        self._out("Проснулся (разбудили), чтение остановлено", "info")
                    else:
                        self._out("Проснулся (разбудили)", "info")
                    continue

                # Спит порциями, чтобы сообщение будило за секунду-две.
                if self.dozing:
                    portion = min(self.mind.cfg.sleep_batch, self.sleep_left)
                    self._execute(
                        lambda session: session.language.sleep(portion),
                        Future(),
                        None,
                        quiet=True,
                    )
                    self.sleep_left -= portion
                    # Между порциями мозг остаётся спящим.
                    self.mind.language.emotions.sleeping = True

                    if self.sleep_left <= 0:
                        self._doze(False)
                        self.mind.language.emotions.sleepy = False
                        self._out("Выспался: новые впечатления закреплены", "info")
                        self.session.autosave(force=True)
                continue

            if job is None:
                break

            # Сообщение будит мозг; недоспанное останется давлением сна.
            if self.dozing:
                self._doze(False)
                self.sleep_left = 0
                self._out("Проснулся (разбудили)", "info")
            last_activity = time.time()
            self._execute(*job)

    # ==============================================================
    # ФОНОВОЕ ЧТЕНИЕ
    # ==============================================================

    def enqueue_read(self, name: str, text: str) -> int:
        """Поставить текст в очередь: мозг читает порциями между сообщениями."""

        sentences = [
            sentence for sentence in self.mind.language.sentences(text)
            if self.mind.language.tokenize(sentence)
        ]

        if sentences:
            self.reading.append(
                {"name": name, "sentences": sentences, "done": 0,
                 "report": ReadReport(), "start": time.time()}
            )
            self.emit(self.status())

        return len(sentences)

    def fall_asleep(self) -> str:
        """
        /sleep в вебе: заснуть так же, как сам в тишине, — порциями,
        чтобы сообщение или «Разбудить» будили за секунду-две.
        """

        if self.dozing:
            return "Уже сплю."

        language = self.mind.language
        cfg = self.mind.cfg
        self.sleep_left = max(cfg.sleep_batch, language.sleep_pressure * cfg.replays_per_episode)
        self._doze(True)

        return (
            f"Засыпаю: новых впечатлений {language.sleep_pressure}, "
            f"сон на {self.sleep_left} проигрываний (разбудить — сообщением или кнопкой)"
        )

    def _read_portion(self, session) -> None:
        if session.stop.is_set():
            self.reading.clear()
            session.stop.clear()
            self._out("Чтение остановлено. Прочитанное уже в синапсах.", "info")
            return

        item = self.reading[0]
        size = self.mind.cfg.read_portion_sentences
        portion = item["sentences"][item["done"]: item["done"] + size]
        item["report"].merge(session.language.read_text("\n".join(portion), nap=False))
        item["done"] += len(portion)

        elapsed = time.time() - item["start"]
        self._out(f"{item['name']}: {report_line(item['report'], elapsed)}", "progress")

        if item["done"] >= len(item["sentences"]):
            self.reading.popleft()
            self._out(f"Прочитал «{item['name']}»: {report_line(item['report'], elapsed)}", "info")
            session.autosave(force=not self.reading)

    def _doze(self, asleep: bool) -> None:
        if self.dozing == asleep:
            return

        self.dozing = asleep
        self.mind.language.emotions.sleeping = asleep
        self.emit(self.status())
        self._emit_emotion()

    def _should_wander(self, last_activity: float) -> bool:
        busy = bool(self.reading) or self.dozing
        return self.live.should_wander(self.mind, last_activity, busy)

    def _wander(self) -> None:
        self.live.wandered_at = time.time()
        future: Future = Future()

        def run(session):
            session.language.on_think = lambda step: self._out(
                f"{step['thought']}\t{step['verdict']}\t{step['score']}", "thought"
            )
            try:
                return session.language.wander()
            finally:
                session.language.on_think = None

        self._execute(run, future, None, quiet=True)
        result = future.result() if future.done() and not future.exception() else None

        if result:
            self.emit({"type": "musing", "text": result[0]})

    def _should_speak_up(self, last_activity: float) -> bool:
        busy = bool(self.reading) or self.dozing
        return self.live.should_speak_up(self.mind, last_activity, bool(self.clients), busy)

    def _speak_up(self) -> None:
        self.live.spoke_up_at = time.time()
        future: Future = Future()
        self._execute(lambda session: session.language.initiative(), future, None, quiet=True)
        said = future.result() if future.done() and not future.exception() else None

        if said:
            reason, text = said
            self.emit({"type": "initiative", "reason": reason, "text": text})
            self.mind.language.dialog.append({"who": "мозг", "text": text, "initiative": reason})
            self._emit_emotion()

    def _should_fall_asleep(self, last_activity: float) -> bool:
        return self.live.should_fall_asleep(self.mind, last_activity, bool(self.reading))

    def submit(self, fn, sink=None) -> Future:
        """
        Выполнить fn(session) в потоке мозга. sink(text, kind) получает
        всё, что сессия говорит (ответ, служебные строки, слова стрима).
        """

        future: Future = Future()
        self.jobs.put((fn, future, sink))

        return future

    def _execute(self, fn, future: Future, sink, quiet: bool = False) -> None:
        # Сон во время тишины не занимает мозг для интерфейса.
        self.busy = not quiet
        self.emit(self.status())

        if sink is not None:
            self.session.out = lambda text, kind: (sink(text, kind), self._out(text, kind))

        try:
            mind = self.session.mind
            future.set_result(fn(self.session))

            if self.session.mind is not mind:
                # /load или /new — другой мозг, новая анатомия.
                self.mind = self.session.mind
                self._attach()
                self.emit(self.anatomy())
        except Exception as error:  # noqa: BLE001 — вернуть вызывающему, мозг жив
            self._out(f"Ошибка: {error}", "info")
            future.set_exception(error)
        finally:
            self.session.out = self._out
            self.flush_spikes()
            self.busy = False
            self.emit(self.status())
            self.emit(self.brains())

    def _attach(self) -> None:
        self.mind.brain.on_chunk = self._on_chunk
        self.mind.language.on_step = self._on_step
        self.mind.language.on_grow = lambda links: self.emit(
            {"type": "grow", "links": [{"path": n, "pre": pre, "post": post} for n, pre, post in links]}
        )
        self.mind.language.on_choice = lambda token, chances: self.emit(
            {"type": "choice", "token": token, "candidates": [[w, round(p, 3)] for w, p in chances]}
        )
        self.session = Session(self.mind, self._out, self._on_reply)
        # /read в вебе не блокирует мозг: текст встаёт в очередь.
        self.session.background_read = self.enqueue_read
        self.session.fall_asleep = self.fall_asleep
        self.reading.clear()

    def shutdown(self) -> None:
        # Дождаться потока мозга: иначе он ещё спит или читает порцию на
        # видеокарте, пока здесь сохраняется мозг (и пока в том же процессе
        # запускается другой — тесты: запись CUDA Graph ломалась).
        self.session and self.session.stop.set()
        self.jobs.put(None)
        self.thread.join(timeout=60)

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
        await ws.send_json(host.brains())

    async def sender():
        while True:
            await ws.send_json(await outbox.get())

    task = asyncio.create_task(sender())

    try:
        while True:
            message = await ws.receive_json()
            kind = message.get("type")

            # Действия идут через HTTP API (brain/api.py); здесь
            # оставлено для совместимости и остановки чтения.
            if kind == "line":
                text = str(message.get("text", ""))
                host.submit(lambda session: session.handle(text))
            elif kind == "stop" and host.session is not None:
                host.session.stop.set()
    except WebSocketDisconnect:
        pass
    finally:
        task.cancel()
        host.clients.discard(outbox)


app.include_router(build_router(host))


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
