"""
Веб-интерфейс анатомического мозга — главный ввод: разговор, уроки и
наблюдение.

    python -m brain.server            # http://localhost:8000

Мозг не загружается сам: в интерфейсе сначала выбирают сохранённый или
создают новый (размер — число нейронов, клетки делятся между зонами сами,
anatomy.area_sizes).

Разговор — через мир, как в мессенджере (brain/session.py): реплика
человека появляется на экране мира, мозг читает её глазом и печатает ответ
рукой; «верно» / «неверно» — вкус-награда. Уроки — учитель мира
(brain/teacher.py). Мозг живёт в одном рабочем потоке (видеокарта).

Наблюдение не тормозит сеть: спайки забираются раз в ~50 мс реального
времени одним снимком (engine.on_chunk), карточки клеток и синапсов
читаются рядом с потоком мозга, только по запросу.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from concurrent.futures import Future
from pathlib import Path
import os
import queue
import threading
import time

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from brain import anatomy, neuron
from brain.config import BRAINS, BrainConfig, brain_path
from brain.engine import KINDS
from brain.mind import Mind, describe_saved
from brain.session import Session
from brain.teacher import Teacher
from brain.view import Sight
from brain.world import ROWS


WEB_DIST = Path(__file__).resolve().parent.parent / "web" / "dist"
STATUS_SECONDS = 0.5
# Глаз и рука (brain/view.py) — снимок раз в столько секунд (~5 Гц).
VIEW_SECONDS = 0.2
# Жизнь в тишине (флаг в интерфейсе): мозг живёт окнами по столько мс.
LIVE_WINDOW_MS = 100.0

# Системы мозга для показа (порядок срезов в микроскопе); зона без системы —
# «прочее».
SYSTEMS = (
    ("Кора", ("visual", "letters", "auditory", "association", "prefrontal", "goal", "category", "limbic",
              "motor", "premotor", "hand")),
    ("Таламус", ("lgn", "mgn", "md", "thalamus", "vl_cb", "trn", "trn_mgn", "trn_md", "trn_motor")),
    ("Слуховой ствол", ("cn", "vnll", "ic")),
    ("Взгляд", ("colliculus",)),
    ("Базальные ганглии", ("striatum_d1", "striatum_d2", "fsi", "gpe", "stn", "gpi")),
    ("Гиппокамп", ("entorhinal", "dentate", "ca3", "ca1")),
    ("Мозжечок", ("granule", "purkinje", "dcn", "io")),
    ("Миндалина и награда", ("amygdala", "cea", "vta", "lhb")),
    ("Модуляторы и сон", ("lc", "raphe", "nbm", "orexin", "vlpo", "scn", "dmh")),
)


class BrainHost:
    """Мозг в своём потоке и рассылка событий всем вкладкам."""

    def __init__(self):
        self.loop: asyncio.AbstractEventLoop | None = None
        self.clients: set[asyncio.Queue] = set()
        self.jobs: queue.Queue = queue.Queue()
        self.busy: str | None = None
        self.live = False
        self.mind: Mind | None = None
        self.session: Session | None = None
        self.teacher: Teacher | None = None
        self.sight: Sight | None = None
        self._last_status = 0.0
        self._last_view = 0.0
        # Пик дофамина с начала фиксации (лента фиксаций).
        self._dopamine = 0.0

    # ==============================================================
    # РАССЫЛКА
    # ==============================================================

    def emit(self, message: dict) -> None:
        if self.loop is None:
            return

        def put():
            for client in list(self.clients):
                if client.qsize() < 500:
                    client.put_nowait(message)

        self.loop.call_soon_threadsafe(put)

    def info(self, text: str) -> None:
        self.emit({"type": "info", "text": text})

    def _on_chunk(self, fired, time_ms: float) -> None:
        # Снимок «кто стрелял» уже прорежен движком (watch_seconds).
        self.emit({"type": "spikes", "t": time_ms, "chunks": [fired.tolist()]})
        self._dopamine = max(self._dopamine, self.mind.brain.occupancy("dopamine"))

        if self.sight is not None:
            self.sight.add(fired)

            if time.perf_counter() - self._last_view > VIEW_SECONDS:
                self._last_view = time.perf_counter()
                self.emit(self.sight.snapshot())

        if time.perf_counter() - self._last_status > STATUS_SECONDS:
            self.emit(self.status())

    def _fixation(self, gaze: float, typed: str | None = None) -> None:
        """
        Лента фиксаций: слово под взглядом (или напечатанная буква) и пик
        дофамина за фиксацию — насколько выше занятости в покое.
        """

        s = self.session
        brain = self.mind.brain
        word, letter = typed or "", 0

        if typed is None:
            # Слово, на которое пришлась точка взгляда (px от начала строки).
            start = 0

            for part in s.line.split(" "):
                end = start + len(part)

                if part and s.screen.width(s.line[:end]) >= gaze:
                    word = part
                    letter = next(i for i in range(len(part)) if s.screen.width(s.line[:start + i + 1]) >= gaze)
                    break
                start = end + 1

        self.emit({"type": "fixation", "t": brain.time_ms, "word": word, "letter": letter, "typed": typed is not None,
                   "dopamine": self._dopamine - (self._rest("dopamine") or 0.0)})
        self._dopamine = brain.occupancy("dopamine")

    def _typed(self, char: str) -> None:
        """Нажатие руки: символ, из чего выбирала (топ-4 клавиши по доле клеток) и в ответ ли."""

        s = self.session
        shares = s.shares.cpu()
        top = shares.topk(min(4, len(shares)))
        choices = [[s.keyboard.type(int(k)), round(float(v), 3)] for v, k in zip(top.values, top.indices)]
        self.emit({"type": "typed", "char": char})
        self.emit({"type": "press", "t": self.mind.brain.time_ms, "char": char, "reply": s.replying, "choices": choices})
        self._fixation(s.gaze, typed=char)
        self.screen()

    def _saccade(self, before: float, gaze: float) -> None:
        self._fixation(before)
        self.screen()

    def watch(self) -> None:
        """Спайки забираются только для открытых вкладок."""

        if self.mind is not None:
            self.mind.brain.on_chunk = self._on_chunk if self.clients else None

    def screen(self) -> None:
        if self.session is not None:
            s = self.session
            self.emit({"type": "screen", "line": s.line, "gaze": s.gaze, "pointer": s.pointer})

    # ==============================================================
    # СОСТОЯНИЕ ДЛЯ ИНТЕРФЕЙСА
    # ==============================================================

    def anatomy(self) -> dict:
        brain = self.mind.brain
        system = {name: title for title, names in SYSTEMS for name in names}
        order = {name: i for i, (_, names) in enumerate(SYSTEMS) for name in names}

        def area(a) -> dict:
            cells = a.cells
            return {
                "name": a.name,
                "title": a.title,
                "system": system.get(a.name, "Прочее"),
                "layered": a.layers is not None,
                "start": cells.start,
                "size": cells.stop - cells.start,
                # Популяции зоны (слои колонки или главные и тормозные ядра).
                "populations": [
                    {"label": label, "start": p.start, "size": p.size,
                     "kind": "PV" if label.split(".")[-1] == "I" and a.layers is not None else label.split(".")[-1]}
                    for label, p in a.populations
                ],
            }

        areas = sorted(brain.areas.values(), key=lambda a: (order.get(a.name, 99), a.name))
        sizes: dict[str, int] = {}

        for p in brain.projections:
            sizes[p.name] = sizes.get(p.name, 0) + (getattr(p, "n_synapses", None) or p.weight.numel())

        return {
            "type": "anatomy",
            "device": str(brain.device),
            "neurons": brain.n,
            "synapses": brain.synapse_count(),
            "systems": [title for title, _ in SYSTEMS] + ["Прочее"],
            # Клавиатура мира по рядам (порядок клавиш — как у руки).
            "keyboard": list(ROWS),
            "areas": [area(a) for a in areas],
            "pathways": [
                {"src": p.src, "dst": p.dst, "plastic": p.plastic, "modulator": p.modulator if p.plastic else None,
                 "synapses": sizes.get(p.name or f"{p.src} -> {p.dst}", 0)}
                for p in self.mind.cfg.pathways
            ],
        }

    def status(self) -> dict:
        self._last_status = time.perf_counter()

        if self.mind is None:
            return {"type": "status", "loaded": False, "busy": self.busy, "live": self.live}

        brain = self.mind.brain

        return {
            "type": "status",
            "loaded": True,
            "busy": self.busy,
            "live": self.live,
            "brain": Path(self.mind.cfg.save_path).name,
            "timeMs": brain.time_ms,
            "hour": brain.hour(),
            # Спит, когда выброс VLPO (галанин) больше норадреналина LC.
            "asleep": brain.level("galanin") > brain.level("noradrenaline"),
            "chemistry": [
                {"name": name, "level": brain.level(name), "occupancy": brain.occupancy(name), "rest": self._rest(name)}
                for name in brain.chemistry_names
            ],
        }

    def _rest(self, name: str) -> float | None:
        """
        Занятость рецепторов наяву в покое: ядро стреляет со своей частотой
        записи (anatomy.RATES_HZ), уровень — спайков на клетку за τ. С ней
        лицо сравнивает текущую: выше покоя — выброс, ниже — провал.
        Аденозина в покое нет — он копится весь день.
        """

        nucleus = self.mind.brain.MODULATORS.get(name)
        cfg = self.mind.cfg
        if nucleus is None or nucleus not in anatomy.RATES_HZ:
            return None
        level = anatomy.RATES_HZ[nucleus] * getattr(cfg, f"{name}_tau_ms") / 1000.0

        return level / (level + getattr(cfg, f"{name}_half"))

    def brains(self) -> dict:
        files = sorted(BRAINS.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True) if BRAINS.exists() else []
        current = Path(self.mind.cfg.save_path).name if self.mind is not None else None

        # Только мозги нового формата (старые этим кодом не открываются).
        items = [d for d in (describe_saved(p) for p in files) if not d["old"]]
        return {"type": "brains", "current": current, "items": items,
                "minNeurons": anatomy.min_size()}

    # ==============================================================
    # РАБОЧИЙ ПОТОК
    # ==============================================================

    def start(self) -> None:
        self.thread = threading.Thread(target=self._run, name="brain", daemon=True)
        self.thread.start()

    def _attach(self) -> None:
        self.session = Session(self.mind)
        self.session.on_typed = self._typed
        self.session.on_saccade = self._saccade
        self.sight = Sight(self.session)
        self.teacher = Teacher(self.session, report=self.info)
        self.watch()
        self.emit(self.anatomy())
        self.screen()

    def _run(self) -> None:
        while True:
            try:
                job = self.jobs.get(timeout=0.0 if self.live and self.mind else 1.0)
            except queue.Empty:
                if self.mind is not None and self.live:
                    # Жизнь в тишине: мозг смотрит на экран и живёт. Без неё
                    # мозг стоит — время мозга идёт, только пока считаются
                    # нейроны (ответ, урок, жизнь в тишине).
                    self.session.live(LIVE_WINDOW_MS)
                continue

            if job is None:
                break

            self._execute(*job)

    def submit(self, fn, label: str) -> Future:
        """Выполнить fn() в потоке мозга."""

        future: Future = Future()
        self.jobs.put((fn, future, label))
        return future

    def _execute(self, fn, future: Future, label: str) -> None:
        self.busy = label
        self.emit(self.status())

        try:
            mind = self.mind
            future.set_result(fn())

            if self.mind is not mind:
                self._attach()
        except Exception as error:  # noqa: BLE001 — вернуть вызывающему, мозг жив
            self.info(f"Ошибка: {error}")
            future.set_exception(error)
        finally:
            self.busy = None
            self.emit(self.status())
            self.emit(self.brains())

    def need(self) -> Mind:
        if self.mind is None:
            raise RuntimeError("мозг не выбран")
        return self.mind

    # ==============================================================
    # ДЕЙСТВИЯ (в потоке мозга)
    # ==============================================================

    def say(self, text: str) -> str:
        """Реплика человека — на экран мира; ответ — что напечатала рука."""

        self.need()
        s = self.session
        s.line, s.gaze = text + " ", 0.0
        self.screen()
        answer = s.reply(text)
        self.screen()
        return answer

    def lesson(self, kind: str, n: int, text: str, first: int, last: int) -> str:
        self.need()
        t, s = self.teacher, self.session
        s.halt = False

        if kind == "letters":
            t.letters(n)
        elif kind == "babble":
            t.babble(n)
        elif kind == "copy":
            t.copy(n)
        elif kind == "course":
            t.course(n, first, last)
        elif kind == "read":
            t.read(text)
        else:
            raise ValueError(f"нет урока {kind}")

        self.screen()
        return f"урок «{kind}» окончен"

    def health(self, ms: float) -> dict:
        """Частоты ядер и слоёв колонок в покое против записей у бодрствующих."""

        brain = self.need().brain

        with brain.frozen():
            rates = brain.observe(ms)

        targets = {layer: dict(zip(KINDS, v)) for layer, *v in self.mind.cfg.column_target_hz}
        nuclei, columns = [], []

        for name, area in brain.areas.items():
            if area.layers is None:
                nuclei.append({"name": name, "title": area.title, "hz": float(rates[area.E.slice].mean()),
                               "target": anatomy.RATES_HZ.get(name)})
            else:
                columns.append({"name": name, "title": area.title, "layers": [
                    {"layer": layer, "kind": kind, "hz": float(rates[pop.slice].mean()),
                     "target": targets.get(layer, {}).get(kind)}
                    for layer, group in area.layers.items() for kind, pop in group.items()
                ]})

        return {"ms": ms, "nuclei": nuclei, "columns": columns}

    def load(self, name: str) -> str:
        if self.mind is not None and self.mind.cfg.autosave:
            self.mind.save()

        self.mind = Mind.load(name)
        return self.mind.cfg.save_path

    def new(self, name: str, neurons: int) -> str:
        if self.mind is not None and self.mind.cfg.autosave:
            self.mind.save()

        cfg = BrainConfig()
        cfg.size = str(int(neurons))
        cfg.save_path = str(brain_path(name if name.endswith(".pt") else name + ".pt"))

        if Path(cfg.save_path).exists():
            raise ValueError(f"мозг {Path(cfg.save_path).name} уже есть")

        self.mind = Mind(cfg)
        self.mind.save()
        return cfg.save_path

    def save(self) -> str:
        return str(self.need().save())

    def shutdown(self) -> None:
        if self.session is not None:
            self.session.halt = True

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


class Say(BaseModel):
    text: str


class Praise(BaseModel):
    good: bool


class Flag(BaseModel):
    enabled: bool


class Load(BaseModel):
    name: str


class New(BaseModel):
    name: str
    neurons: int


class Lesson(BaseModel):
    kind: str
    n: int = 50
    text: str = ""
    first: int = 1
    last: int = 99


async def _run(fn, label: str):
    future = host.submit(fn, label)

    try:
        return await asyncio.wrap_future(future)
    except Exception as error:  # noqa: BLE001
        raise HTTPException(400, str(error)) from error


async def _read(fn):
    """Чтение мозга рядом с его потоком (карточки): сеть не ждёт."""

    if host.mind is None:
        raise HTTPException(400, "мозг не выбран")

    try:
        return await asyncio.to_thread(fn, host.mind.brain)
    except IndexError as error:
        raise HTTPException(404, str(error)) from error


@app.post("/api/say")
async def say(body: Say) -> dict:
    text = body.text.strip()

    if not text:
        raise HTTPException(400, "пустая реплика")

    return {"text": await _run(lambda: host.say(text), "отвечает")}


@app.post("/api/praise")
async def praise(body: Praise) -> dict:
    await _run(lambda: host.need() and host.session.praise(body.good), "вкус")
    return {"ok": True}


@app.post("/api/lesson")
async def lesson(body: Lesson) -> dict:
    text = await _run(lambda: host.lesson(body.kind, body.n, body.text, body.first, body.last), f"урок: {body.kind}")
    return {"text": text}


@app.post("/api/stop")
async def stop() -> dict:
    if host.session is not None:
        host.session.halt = True
    return {"ok": True}


@app.post("/api/live")
async def live(body: Flag) -> dict:
    host.live = body.enabled
    host.emit(host.status())
    return {"live": host.live}


@app.get("/api/health")
async def health(ms: float = 500.0) -> dict:
    return await _run(lambda: host.health(ms), "здоровье")


@app.get("/api/summary")
async def summary() -> dict:
    return await _read(neuron.summary)


@app.get("/api/neuron/{index}")
async def neuron_card(index: int) -> dict:
    return await _read(lambda brain: neuron.describe(brain, index))


@app.get("/api/synapse/{sid}")
async def synapse_card(sid: int) -> dict:
    return await _read(lambda brain: neuron.synapse(brain, sid))


@app.get("/api/sizes")
async def sizes(neurons: int) -> dict:
    total = anatomy.total_for(max(1, neurons))
    return {"neurons": neurons, "total": total, "min": anatomy.min_size()}


@app.post("/api/load")
async def load(body: Load) -> dict:
    return {"brain": await _run(lambda: host.load(body.name), "загрузка")}


@app.post("/api/new")
async def new(body: New) -> dict:
    name = body.name.strip()

    if not name:
        raise HTTPException(400, "нужно имя")

    return {"brain": await _run(lambda: host.new(name, body.neurons), "новый мозг")}


@app.post("/api/save")
async def save() -> dict:
    return {"brain": await _run(host.save, "сохранение")}


@app.websocket("/ws")
async def socket(ws: WebSocket) -> None:
    await ws.accept()
    outbox: asyncio.Queue = asyncio.Queue()
    host.clients.add(outbox)
    host.watch()
    await ws.send_json(host.brains())
    await ws.send_json(host.status())

    if host.mind is not None:
        await ws.send_json(host.anatomy())
        host.screen()

    async def sender():
        while True:
            await ws.send_json(await outbox.get())

    task = asyncio.create_task(sender())

    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        task.cancel()
        host.clients.discard(outbox)
        host.watch()


if WEB_DIST.exists():
    app.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="web")
else:
    @app.get("/")
    def index() -> HTMLResponse:
        return HTMLResponse("<p>Интерфейс не собран: <code>cd web && npm install && npm run build</code></p>")


def main() -> None:
    import uvicorn

    uvicorn.run(app, host=os.environ.get("BRAIN_HOST", "127.0.0.1"), port=int(os.environ.get("BRAIN_PORT", "8000")))


if __name__ == "__main__":
    main()
