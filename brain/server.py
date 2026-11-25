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

import torch
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from brain import anatomy, flags, neuron, places
from brain.config import BRAINS, BrainConfig, brain_path
from brain.presets import PRESETS
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
# Медленный гомеостаз живого мозга (галка в вебе, «обычный» режим): тот же регулятор,
# что при развитии (engine.homeostasis_update), но в SLOW_HOMEO раз медленнее.
# Развитие — ускоренный режим: 30 с модельного времени ~ сутки настоящей подстройки
# (масштабирование синапсов и собственная возбудимость — 24–48 ч, Turrigiano et al.
# 1998, https://doi.org/10.1038/36103) → 24 ч / 30 с = 2880 [своё: сопоставление
# «развитие ≈ сутки»]. За сессию в вебе сдвиг — доли мВ: правильность, а не «доделка».
SLOW_HOMEO = 2880.0
HOMEO_WINDOW_MS = 1000.0

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


class WordsLife:
    """
    Жизнь мозга без глаза (cfg.organs без "eye", словесный мозг): окно за
    окном мозг слушает тишину порта слов (brain/body.Words). Экрана мира нет —
    поля экрана пустые, чтобы веб показывал пустую строку.
    """

    def __init__(self, mind):
        from brain.body import Words

        self.mind = mind
        self.brain = mind.brain
        self.port = Words.of(self.brain) if "words" in mind.cfg.organs else None
        self.line, self.gaze, self.pointer = "", 0.0, None
        self.halt = False

    def live(self, ms: float) -> str:
        sensory = self.port.hear(None, ms) if self.port is not None else None
        self.brain.observe(ms, sensory)
        return ""

    def reply(self, text: str) -> str:
        return ("Чат словами в вебе ещё не подключён (порт слов есть — CHAT_PLAN шаги 4–6); "
                "мозг живёт, его клетки видны на карте и в срезах.")


class BrainHost:
    """Мозг в своём потоке и рассылка событий всем вкладкам."""

    def __init__(self):
        self.loop: asyncio.AbstractEventLoop | None = None
        self.clients: set[asyncio.Queue] = set()
        self.jobs: queue.Queue = queue.Queue()
        self.busy: str | None = None
        # «Стоп» для долгих работ потока мозга (развитие): проверяется после каждого окна.
        self.stopping = threading.Event()
        self.closing = False
        # Живой режим по умолчанию включён: мозг живёт сам (смотрит на экран,
        # время идёт), а не стоит до реплики; BRAIN_LIVE=0 — выключить
        # (например, чтобы не занимать видеокарту).
        self.live = os.environ.get("BRAIN_LIVE", "1") != "0"
        self.mind: Mind | None = None
        # Медленный (биологический) гомеостаз у готового мозга — по галке, по умолчанию выкл.
        self.slow_homeo = False
        self._homeo_counts: torch.Tensor | None = None
        self._homeo_ms = 0.0
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
            return {"type": "status", "loaded": False, "busy": self.busy, "live": self.live,
                    "homeostasis": self.slow_homeo}

        brain = self.mind.brain

        return {
            "type": "status",
            "loaded": True,
            "busy": self.busy,
            "live": self.live,
            "homeostasis": self.slow_homeo,
            "developDoneS": float(getattr(self.mind.cfg, "develop_done_s", 0.0)),
            "homeoShiftMv": float(brain.homeo_shift.abs().mean()) if getattr(brain, "homeo_shift", None) is not None else 0.0,
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
        presets = [{"name": p.name, "title": p.title, "ready": not p.missing(), "seeds": p.seeds()} for p in PRESETS.values()]
        small = BrainConfig()
        flags.configure(small, {"small_columns": True})
        return {"type": "brains", "current": current, "items": items,
                "minNeurons": anatomy.min_size(), "minNeuronsSmall": anatomy.min_size(small),
                "presets": presets, "flags": flags.describe()}

    def flag_state(self) -> dict:
        """С чем собран текущий мозг и что включено на ходу (из его конфига — сохраняется в файле)."""

        cfg = self.need().cfg
        return {"preset": cfg.preset, "seed": cfg.seed, "neurons": self.mind.brain.n, "flags": flags.describe(cfg)}

    def switch(self, key: str, on: bool) -> dict:
        flags.switch(self.need(), key, on)
        self.info(f"флаг «{flags.BY_KEY[key].title}»: {'вкл' if on else 'выкл'}")
        return self.flag_state()

    # ==============================================================
    # РАБОЧИЙ ПОТОК
    # ==============================================================

    def start(self) -> None:
        self.thread = threading.Thread(target=self._run, name="brain", daemon=True)
        self.thread.start()

    def _attach(self) -> None:
        overflow = os.environ.get("BRAIN_EVENT_OVERFLOW")
        if overflow:
            # Сохранённый мозг несёт свою настройку; переменная веба — сильнее
            # (BrainConfig.event_overflow).
            self.mind.cfg.event_overflow = overflow
            self.mind.brain.cfg.event_overflow = overflow
        if "eye" not in self.mind.cfg.organs:
            # Мозг без глаза (словесный): живёт, слушая тишину порта слов;
            # экрана мира, руки и уроков у него нет.
            self.session = WordsLife(self.mind)
            self.sight = None
            self.teacher = None
        else:
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
                    try:
                        t0 = self.mind.brain.time_ms
                        self.session.live(LIVE_WINDOW_MS)
                        if self.slow_homeo:
                            self._slow_homeostasis(self.mind.brain.time_ms - t0)
                    except Exception as error:  # noqa: BLE001 — поток мозга не должен умирать молча
                        # Мозг на паузу, причина — в разговор; реплики и уроки
                        # по-прежнему принимаются.
                        import traceback

                        traceback.print_exc()
                        self.live = False
                        self.info(f"Мозг остановлен: {error}")
                        self.emit(self.status())
                continue

            if job is None:
                break

            self._execute(*job)

    def _slow_homeostasis(self, ms: float) -> None:
        """
        Медленный гомеостаз: спайки окон жизни копятся до HOMEO_WINDOW_MS, затем
        engine.homeostasis_update с шагом homeostasis_step_mv / SLOW_HOMEO.
        """

        b = self.mind.brain
        if ms <= 0:
            return
        counts = b.counts[: b.n].float()
        self._homeo_counts = counts.clone() if self._homeo_counts is None else self._homeo_counts + counts
        self._homeo_ms += ms
        if self._homeo_ms < HOMEO_WINDOW_MS:
            return
        rates = self._homeo_counts * (1000.0 / self._homeo_ms)
        self._homeo_counts, self._homeo_ms = None, 0.0
        step = b.cfg.homeostasis_step_mv
        b.cfg.homeostasis_step_mv = step / SLOW_HOMEO
        try:
            b.homeostasis_update(rates)
        finally:
            b.cfg.homeostasis_step_mv = step

    def submit(self, fn, label: str) -> Future:
        """Выполнить fn() в потоке мозга."""

        future: Future = Future()
        self.jobs.put((fn, future, label))
        return future

    def _execute(self, fn, future: Future, label: str) -> None:
        # «Стоп», нажатый до начала работы (или по прошлой), к ней не относится.
        if not self.closing:
            self.stopping.clear()
        self.busy = label
        self.emit(self.status())

        try:
            mind = self.mind
            future.set_result(fn())

            if self.mind is not mind:
                self._attach()
        except Exception as error:  # noqa: BLE001 — вернуть вызывающему, мозг жив
            from brain.develop import Stopped

            self.info(f"{error}" if isinstance(error, Stopped) else f"Ошибка: {error}")
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
        if isinstance(s, WordsLife):
            return s.reply(text)
        s.line, s.gaze = text + " ", 0.0
        self.screen()
        answer = s.reply(text)
        self.screen()
        return answer

    def lesson(self, kind: str, n: int, text: str, first: int, last: int) -> str:
        self.need()
        if self.teacher is None:
            raise ValueError("уроки — у мозга с глазом и рукой")
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

    def _letter_groups(self, letter: str):
        """Выученные буквы и группы клеток зоны букв (L4 E, по 20 на букву у
        ямки, ранг по расстоянию до ямки % числа букв — разметка tools/letters_us.py)."""

        mind = self.need()
        chosen = PRESETS.get(mind.cfg.preset)
        letters = chosen.letters if chosen else ""
        if not letters:
            raise ValueError("зона букв этого мозга не учена — нужен мозг из пресета «working»")
        if letter not in letters:
            raise ValueError(f"буква «{letter}» не училась; выученные: {letters}")
        b = mind.brain
        L4 = b.areas["letters"].layers["L4"]["E"]
        near = (b.field_cols[L4.name] - b.cfg.fovea_col).abs().argsort()[: 20 * len(letters)].to(b.device)
        group = torch.arange(len(near), device=b.device) % len(letters)
        return letters, near + L4.start, group

    def _letter_show(self, letter: str, cells, teach: torch.Tensor | None = None) -> torch.Tensor:
        """Буква 350 мс по центру ямки (взгляд держит стенд, не двухолмие), потом
        пустой экран 150 мс; teach — клетки, которые учитель заставляет стрелять
        (cfg.teacher_hz), с правилом учёбы пути V1 → буквы. Возвращает спайки cells."""

        b, s = self.mind.brain, self.session
        if isinstance(s, WordsLife):
            raise ValueError("буквы — у мозга с глазом")
        show_ms, rest_ms = 350.0, 150.0
        s.line, s.pointer = letter, None
        s.gaze = s.screen.width(letter) / 2
        self.screen()
        if teach is not None:
            b.teach_pop.zero_()
            b.teach_pop[cells] = True
            b.teach_target.zero_()
            b.teach_target[teach] = True
            b.teach_on = b.us_on = True
        try:
            count = b.observe(show_ms, s._view(show_ms))[cells] * (show_ms / 1000.0)
        finally:
            b.teach_on = b.us_on = False
        s.line, s.gaze = "", 0.0
        b.observe(rest_ms, s._blank(rest_ms))
        self.screen()
        return count

    def show_letter(self, letter: str) -> dict:
        """
        Сценарий «Покажи букву»: ответ — группа клеток зоны букв с наибольшим
        числом спайков за показ; молчат все — «не вижу».
        """

        letters, cells, group = self._letter_groups(letter)
        count = self._letter_show(letter, cells)
        spikes = {c: int(round(float(count[group == k].sum()))) for k, c in enumerate(letters)}
        best = max(spikes, key=spikes.get)
        return {"letter": letter, "seen": best if spikes[best] > 0 else "", "spikes": spikes}

    def correct_letter(self, shown: str, right: str) -> dict:
        """
        «Нет, это буква right»: тот же показ буквы shown, учитель заставляет
        стрелять группу right, правило Urbanczik & Senn (cfg.us_*) меняет веса
        пути V1 → буквы — как одна учебная проба tools/letters_us.py. Учитель —
        зажим кодом (ISSUES 7): так в мозг приходит названная буква, пока нет
        пути «слово → вид буквы» сверху.
        """

        letters, cells, group = self._letter_groups(right)
        if shown not in letters:
            raise ValueError(f"буква «{shown}» не училась; выученные: {letters}")
        path = next(p for p in self.mind.brain.projections if p.name == "visual -> letters")
        w0 = path.weight.clone()
        self._letter_show(shown, cells, teach=cells[group == letters.index(right)])
        changed = float((path.weight - w0).abs().mean() / w0.mean())
        return {"shown": shown, "right": right, "changed": changed}

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

    def new(self, name: str, neurons: int, preset: str | None = None, seed: int = 1,
            chosen_flags: dict[str, bool] | None = None, organs: list[str] | None = None,
            develop_s: float = 0.0) -> str:
        """Новый мозг: голый заданного размера или по пресету (brain/presets.py, размер — пресета);
        зерно и флаги (brain/flags.py) — в конфиг, он сохраняется с мозгом. develop_s > 0 —
        развитие при создании (brain/develop.py): develop_s с гомеостаза на тишине сдвигают
        фон клеток под собранную анатомию (у пресета — вместо сохранённых сдвигов)."""

        chosen = PRESETS[preset] if preset else None
        chosen_flags = {k: v for k, v in (chosen_flags or {}).items() if v}
        if chosen and chosen_flags.get("small_columns"):
            raise ValueError("малые колонки — только у голого мозга: подгонки пресета посчитаны на колонках 20 тыс.")
        if chosen and chosen.missing(seed, develop=develop_s > 0):
            raise ValueError(f"у пресета «{chosen.name}» нет файлов для зерна {seed} (в репозитории — только зерно 1)")

        if self.mind is not None and self.mind.cfg.autosave:
            self.mind.save()

        if organs is not None:
            unknown = set(organs) - set(anatomy.ORGAN_AREAS)
            if unknown or not organs:
                raise ValueError(f"органы: {', '.join(sorted(unknown)) or 'ни одного не выбрано'}")
            if chosen and "eye" not in organs:
                raise ValueError(f"пресет «{chosen.name}» подогнан под зрительный путь — нужен глаз")

        cfg = chosen.config(seed) if chosen else BrainConfig()
        if organs is not None:
            cfg.organs = tuple(o for o in anatomy.ORGAN_AREAS if o in organs)
        cfg.seed = int(seed)
        cfg.preset = chosen.name if chosen else ""
        flags.configure(cfg, chosen_flags)
        if not chosen:
            cfg.size = str(int(neurons))
        cfg.save_path = str(brain_path(name if name.endswith(".pt") else name + ".pt"))

        if Path(cfg.save_path).exists():
            raise ValueError(f"мозг {Path(cfg.save_path).name} уже есть")

        def progress(w: int, n: int, rates) -> None:
            self.busy = f"развитие: {w + 1} из {n} с"
            self.emit(self.status())

        # Переменная веба BRAIN_EVENT_OVERFLOW — уже при сборке и развитии (иначе первый
        # залп развития роняет создание; _attach применяет её только после).
        overflow = os.environ.get("BRAIN_EVENT_OVERFLOW")
        if overflow:
            cfg.event_overflow = overflow
        # Новый мозг становится текущим только целиком: при ошибке сборки или
        # развития остаётся прежний (иначе поток жизни видит мозг без сессии).
        from brain.develop import BUDGET_S

        if develop_s > BUDGET_S:
            raise ValueError(f"развитие не больше {BUDGET_S:.0f} с (ускоренный гомеостаз ≈ сутки подстройки)")
        if chosen:
            mind = chosen.build(seed, cfg=cfg, develop_s=develop_s, on_window=progress, stop=self.stopping.is_set)
        else:
            mind = Mind(cfg)
            if develop_s > 0:
                from brain.develop import develop

                develop(mind, develop_s, on_window=progress, stop=self.stopping.is_set)
        mind.cfg.develop_done_s = float(develop_s)
        flags.after_build(mind)
        mind.save()
        self.mind = mind
        return cfg.save_path

    def develop_more(self, seconds: float) -> dict:
        """
        Доразвить готовый мозг: ускоренный гомеостаз ещё seconds с, но всего не больше
        develop.BUDGET_S на мозг (при создании + доразвитие). Без прогрева — мозг уже живёт.
        """

        from brain.develop import BUDGET_S, WINDOW_MS, develop

        mind = self.need()
        left = BUDGET_S - mind.cfg.develop_done_s
        s = min(float(seconds), left)
        if s < 1.0:
            raise ValueError(f"бюджет развития исчерпан: прожито {mind.cfg.develop_done_s:.0f} из {BUDGET_S:.0f} с — "
                             "дальше только медленный гомеостаз")

        done = [0]

        def progress(w: int, n: int, rates) -> None:
            done[0] = w + 1
            self.busy = f"доразвитие: {w + 1} из {n} с"
            self.emit(self.status())

        before = mind.cfg.develop_done_s
        try:
            develop(mind, s, warm_ms=0.0, on_window=progress, stop=self.stopping.is_set)
        finally:
            # Прожитые окна — в бюджет, и при остановке посреди (сдвиги фона уже в мозге).
            mind.cfg.develop_done_s = before + done[0] * WINDOW_MS / 1000.0
        return {"developed": mind.cfg.develop_done_s - before, "done": mind.cfg.develop_done_s, "budget": BUDGET_S}

    def save(self) -> str:
        return str(self.need().save())

    def shutdown(self) -> None:
        if self.session is not None:
            self.session.halt = True
        self.closing = True
        self.stopping.set()

        self.jobs.put(None)
        self.thread.join(timeout=60)

        if self.thread.is_alive():
            # Поток мозга ещё работает (запись гигабайтов, развитие): своя запись сейчас
            # снимет мозг посреди шага — прежний файл остаётся как был.
            print("сервер: поток мозга не закончил за 60 с — автосохранение пропущено", flush=True)
            return
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
    preset: str | None = None
    seed: int = 1
    flags: dict[str, bool] = {}
    # Органы мозга (BrainConfig.organs): None — как в сборке по умолчанию.
    organs: list[str] | None = None
    # Развитие при создании, с (brain/develop.py); 0 — без развития.
    develop: float = 0.0


class Switch(BaseModel):
    key: str
    enabled: bool


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


class Letter(BaseModel):
    letter: str


@app.post("/api/letter")
async def show_letter(body: Letter) -> dict:
    return await _run(lambda: host.show_letter(body.letter.strip().lower()), "смотрит на букву")


class Correction(BaseModel):
    shown: str
    right: str


@app.post("/api/letter/correct")
async def correct_letter(body: Correction) -> dict:
    return await _run(lambda: host.correct_letter(body.shown.strip().lower(), body.right.strip().lower()),
                      "учит букву")


class Phrase(BaseModel):
    phrase: str


# Сценарий «Запомни и повтори» — отдельный блок E1 (tools/e1_recall.py:
# счётчик позиции, ячейки слов, ворота BG, реле MD), ещё НЕ в мозге
# (перенос — шаг 4 docs/CHAT_PLAN.md). Считается на CPU в своём потоке,
# мозг в вебе живёт дальше; одна фраза ~5 мин (15 с времени блока, шаг 0.1 мс).
_e1_lock = threading.Lock()


def _e1_recall(phrase: str) -> dict:
    import sys
    tools = str(Path(__file__).resolve().parent.parent / "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    import e1_recall

    if not _e1_lock.acquire(blocking=False):
        raise ValueError("блок E1 уже повторяет другую фразу")
    try:
        return e1_recall.main({"phrase": phrase, "threads": "2"})
    finally:
        _e1_lock.release()


@app.post("/api/repeat")
async def repeat(body: Phrase) -> dict:
    try:
        return await asyncio.to_thread(_e1_recall, body.phrase.strip().upper())
    except ValueError as error:
        raise HTTPException(400, str(error)) from error


@app.post("/api/stop")
async def stop() -> dict:
    if host.session is not None:
        host.session.halt = True
    # Долгая работа потока мозга (развитие) — прервать после текущего окна.
    if host.busy:
        host.stopping.set()
    return {"ok": True}


@app.post("/api/live")
async def live(body: Flag) -> dict:
    host.live = body.enabled
    host.emit(host.status())
    return {"live": host.live}


@app.post("/api/homeostasis")
async def homeostasis(body: Flag) -> dict:
    """Медленный (биологический) гомеостаз готового мозга; ускоренный — только развитие при создании."""

    host.slow_homeo = body.enabled
    host._homeo_counts, host._homeo_ms = None, 0.0
    host.emit(host.status())
    return {"homeostasis": host.slow_homeo}


class Develop(BaseModel):
    seconds: float


@app.post("/api/develop")
async def develop_more(body: Develop) -> dict:
    """Доразвить готовый мозг (ускоренный гомеостаз) в пределах бюджета 30 с на мозг."""

    return await _run(lambda: host.develop_more(body.seconds), "доразвитие")


@app.get("/api/health")
async def health(ms: float = 500.0) -> dict:
    return await _run(lambda: host.health(ms), "здоровье")


@app.get("/api/places")
async def cell_places() -> dict:
    """Места клеток для карты нейронов (brain/places.py)."""

    return await _read(places.places)


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
async def sizes(neurons: int, small: bool = False) -> dict:
    cfg = BrainConfig()
    flags.configure(cfg, {"small_columns": small})
    total = anatomy.total_for(max(1, neurons), cfg)
    return {"neurons": neurons, "total": total, "min": anatomy.min_size(cfg)}


@app.get("/api/flags")
async def flag_state() -> dict:
    if host.mind is None:
        raise HTTPException(400, "мозг не выбран")
    return host.flag_state()


@app.post("/api/flag")
async def switch(body: Switch) -> dict:
    return await _run(lambda: host.switch(body.key, body.enabled), "флаг")


@app.post("/api/load")
async def load(body: Load) -> dict:
    return {"brain": await _run(lambda: host.load(body.name), "загрузка")}


@app.post("/api/new")
async def new(body: New) -> dict:
    name = body.name.strip()

    if not name:
        raise HTTPException(400, "нужно имя")

    return {"brain": await _run(lambda: host.new(name, body.neurons, body.preset, body.seed, body.flags,
                                                     body.organs, body.develop),
                                "новый мозг")}


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

    config = uvicorn.Config(app, host=os.environ.get("BRAIN_HOST", "127.0.0.1"),
                            port=int(os.environ.get("BRAIN_PORT", "8000")))
    server = uvicorn.Server(config)
    handle_exit = server.handle_exit

    def on_exit(sig, frame) -> None:
        # Ctrl+C: uvicorn сначала ждёт открытые запросы (создание мозга с развитием —
        # минуты), а lifespan (host.shutdown) — после них. Флаг «стоп» — сразу.
        host.closing = True
        host.stopping.set()
        if host.session is not None:
            host.session.halt = True
        handle_exit(sig, frame)

    server.handle_exit = on_exit
    try:
        server.run()
    except KeyboardInterrupt:
        pass  # uvicorn повторяет сигнал после остановки — как uvicorn.run


if __name__ == "__main__":
    main()
