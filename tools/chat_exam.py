"""
Экзамен чата (docs/CHAT_PLAN.md, шаг 1) — мера прогресса к первому вопросу.
Работает на любом «теле»; учитель только предъявляет слова и команды,
содержимое ответа не подставляет.

Формат — по готовому: задача «ответ на вопрос» Spaun (A5, состояния QAP/QAK)
и парные ассоциации (cued recall) — стандартный тест памяти человека.

Части:
  повтори   фраза 2–3 слова (А–Д) по одному слову в слот 1200 мс (слово 200 мс),
            потом команда «повтори»; ответ снимается в окнах 0–300 мс позиций
            1..4; второй «повтори» без повторной записи. Мера: фраза целиком
            (оба вывода), доля позиций, «молчал» отдельно от «ошибся»; молчание
            после конца фразы — часть верного ответа (стоп).
  пары      учёба: N пар «A B» (A 200 мс, сразу B 200 мс, пауза 800 мс) × reps;
            знакомые: «A ?» → B; отрицательные: невыученное A → молчание или
            «не знаю», ложный ответ — любое другое слово; сохранение: после
            второй порции пар первая спрашивается снова.
  звенья    где теряется сигнал: у E1 — записано ли слово в ячейку позиции,
            открылось ли реле чтения, вышел ли ответ; у мозга — держится ли
            слово-подсказка после конца предъявления, поднялась ли популяция
            ответа над фоном, победила ли, вышла ли за порог.

Тела:
  e1     блок E1 (tools/e1_recall.py, НЕ в мозге) — только «повтори»; пары не
         умеет (честный 0). Зёрна считаются параллельно на CPU (2 потока).
  brain  целый мозг; порт слов — заглушка ввода/вывода как VisionSystemDummy /
         MotorSystemDummy Spaun: слово = 40 клеток L2/3 E ассоциативной (смысловой)
         коры, предъявление — учитель-зажим 40 Гц (как слово в E1; ISSUES 7),
         ответ — какая популяция слова горит. Настоящий порт — шаг 2.
         mind=working (пресет, только зерно 1) или bare (голый мозг размера size).

Порог «сказано» — 10 Гц на клетку популяции, одна популяция выше порога
(как в E1); две и больше — «ошибся».

    PYTHONPATH=tools .venv/bin/python tools/chat_exam.py [body=e1|brain] [seeds=1,2,3]
        [trials=8] [pairs=4] [reps=3] [neg=3] [mind=working|bare] [size=small] [parts=repeat,pairs]
"""

from __future__ import annotations

import math
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor

SLOT_MS = 1200.0     # слот слова — как T блока E1
WORD_MS = 200.0      # слово горит — как в E1
READ_MS = 300.0      # окно ответа позиции — как в E1
THR_HZ = 10.0        # «сказано» — как в E1
WORDS = "АБВГД"
POS = 4              # окон ответа при «повтори» (3 слова + проверка молчания)
LMAX = 3
# Пары «понятие — свойство»; учитель предъявляет их вместе, вопрос — первое слово.
PAIRS = [("кошка", "мяукает"), ("собака", "лает"), ("корова", "мычит"), ("утка", "крякает"),
         ("лошадь", "ржёт"), ("овца", "блеет"), ("свинья", "хрюкает"), ("змея", "шипит"),
         ("пчела", "жужжит"), ("волк", "воет"), ("петух", "кукарекает"), ("лягушка", "квакает")]
UNKNOWN = "не знаю"


def wilson(k: int, n: int) -> tuple[float, float]:
    """95% интервал доли (Wilson 1927)."""

    if n == 0:
        return 0.0, 1.0
    z, p = 1.96, k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, c - h), min(1.0, c + h)


def share(k: int, n: int) -> str:
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {k / max(n, 1):.2f} [95% {lo:.2f}–{hi:.2f}]"


def phrases(seed: int, n: int) -> list[str]:
    rng = random.Random(seed)
    return ["".join(rng.choice(WORDS) for _ in range(2 + i % 2)) for i in range(n)]


# ==============================================================
# ТЕЛО: БЛОК E1
# ==============================================================

def e1_repeat(seed: int, texts: list[str]) -> dict:
    import e1_recall

    return e1_recall.main({"phrase": ",".join(texts), "seed": str(seed), "threads": "2"})


# ==============================================================
# ТЕЛО: ЦЕЛЫЙ МОЗГ С ПОРТОМ СЛОВ-ЗАГЛУШКОЙ
# ==============================================================

class BrainBody:
    def __init__(self, seed: int, mind: str, size: str):
        import torch

        from brain.config import BrainConfig
        from brain.mind import Mind
        from brain.presets import WORKING
        from brain.session import Session

        self.torch = torch
        if mind == "working":
            cfg = WORKING.config(seed)
            cfg.preset = "working"
            self.mind = WORKING.build(seed, cfg=cfg)
        else:
            cfg = BrainConfig()
            cfg.seed, cfg.size = seed, size
            self.mind = Mind(cfg)
        self.mind.cfg.autosave = False
        self.b = b = self.mind.brain
        self.s = Session(self.mind)
        vocab = list(WORDS) + ["повтори", "?", UNKNOWN] + [w for p in PAIRS for w in p]
        pop = b.areas["association"].layers["L23"]["E"]
        gen = torch.Generator().manual_seed(1000 + seed)
        cells = torch.randperm(pop.size, generator=gen)[: 40 * len(vocab)].view(len(vocab), 40) + pop.start
        self.cells = {w: c.to(b.device) for w, c in zip(vocab, cells)}
        self.vocab = vocab
        self.b.teach_hz = torch.zeros(b.n_slots, device=b.device)
        self.thr = {w: THR_HZ for w in vocab}
        self.blank(1000.0)

    def blank(self, ms: float):
        return self.b.observe(ms, self.s._blank(ms))

    def word(self, w: str, ms: float = WORD_MS):
        """Слово горит ms: его 40 клеток стреляют пуассоном 40 Гц (учитель-зажим)."""

        b = self.b
        b.teach_pop.zero_()
        b.teach_pop[self.cells[w]] = True
        b.teach_hz.zero_()
        b.teach_hz[self.cells[w]] = 40.0
        b.teach_on = True
        try:
            return self.blank(ms)
        finally:
            b.teach_on = False

    def rates(self, r) -> dict[str, float]:
        return {w: float(r[c].mean()) for w, c in self.cells.items()}

    def said(self, rates: dict[str, float], exclude=("повтори", "?")) -> str:
        """Сказанное в окне: '' — молчал, '?' — несколько слов сразу, иначе слово."""

        over = [w for w, hz in rates.items() if hz > self.thr[w] and w not in exclude]
        return over[0] if len(over) == 1 else ("" if not over else "?")

    def repeat(self, texts: list[str]) -> dict:
        out = {"said": [], "held": []}
        for text in texts:
            self.blank(400.0)
            for w in text:                                # запись: слот на слово
                self.word(w)
                self.blank(SLOT_MS - WORD_MS)
            runs = []
            for _ in range(2):                            # два «повтори» без повторной записи
                self.blank(300.0)
                self.word("повтори")
                self.blank(400.0 - WORD_MS)
                line = ""
                for _ in range(POS):
                    w = self.said(self.rates(self.blank(READ_MS)))
                    line += w if w and w in WORDS else ("·" if w == "" else "?")
                    self.blank(SLOT_MS - READ_MS)
                runs.append(line)
            out["said"].append(runs)
        return out

    def teach(self, pairs) -> None:
        for a, c in pairs:
            self.word(a)
            self.word(c)
            self.blank(SLOT_MS - 2 * WORD_MS)

    def ask(self, cue: str) -> dict:
        """«cue ?»: ответ в окне 600–900 мс; звенья — подсказка держится 0–100 мс после
        конца слова, популяции в окне ответа."""

        self.word(cue)
        hold = self.rates(self.blank(100.0))[cue] > self.thr[cue]
        self.word("?")
        self.blank(100.0)
        r = self.rates(self.blank(READ_MS))
        self.blank(SLOT_MS - 900.0)
        return {"said": self.said(r, exclude=("повтори", "?", cue)), "hold": hold, "rates": r, "cue": cue}


class WordBody(BrainBody):
    """
    Словесный мозг (cfg.organs = ("words",), CHAT_PLAN шаг 2): слово — орган
    «слова» (brain/body.Words) в L4 смысловой коры; популяции слов L2/3 и
    порог «сказано» — скрининг tools/word_port.py (пластичность выключена на
    время скрининга, потом как была).
    """

    def __init__(self, seed: int, size: str, screen: int = 4):
        import torch

        import word_port
        from brain.body import Words

        self.torch = torch
        self.mind, self.b = word_port.build({"seed": seed, "size": size})
        b = self.b
        self.port = Words.of(b)
        vocab = list(WORDS) + ["повтори", "?", UNKNOWN] + [w for p in PAIRS for w in p]
        if len(vocab) != b.cfg.words_count:
            raise ValueError(f"словарь {len(vocab)} слов, а порт на {b.cfg.words_count}")
        self.vocab, self.index = vocab, {w: i for i, w in enumerate(vocab)}
        plastic = b.plasticity
        b.plasticity = False
        pops, thr = word_port.screen(b, self.port, screen, seed)
        b.plasticity = plastic
        self.cells = {w: pops[i] for i, w in enumerate(vocab)}
        self.thr = {w: float(thr[i]) for i, w in enumerate(vocab)}
        self.blank(1000.0)

    def blank(self, ms: float):
        return self.b.observe(ms, self.port.hear(None, ms))

    def word(self, w: str, ms: float = WORD_MS):
        return self.b.observe(ms, self.port.hear(self.index[w], ms))


def brain_exam(seed: int, opts: dict) -> dict:
    t0 = time.time()
    mind = opts.get("mind", "words")
    body = (WordBody(seed, opts.get("size", "small"), int(opts.get("screen", 4))) if mind == "words"
            else BrainBody(seed, mind, opts.get("size", "small")))
    base = body.rates(body.blank(1000.0))
    res = {"seed": seed, "base": base, "thr": body.thr}
    parts = opts.get("parts", "repeat,pairs").split(",")
    if "repeat" in parts:
        texts = phrases(seed, int(opts.get("trials", 4)))
        res["repeat"] = {"phrases": texts, **body.repeat(texts)}
    if "pairs" in parts:
        n, reps, neg = int(opts.get("pairs", 4)), int(opts.get("reps", 3)), int(opts.get("neg", 3))
        rng = random.Random(seed)
        order = PAIRS[:]
        rng.shuffle(order)
        first, second, unseen = order[:n], order[n:2 * n], order[2 * n:2 * n + neg]
        for _ in range(reps):
            body.teach(first)
        known1 = [(c, body.ask(a)) for a, c in first]
        negative = [(None, body.ask(a)) for a, _ in unseen]
        for _ in range(reps):
            body.teach(second)
        known2 = [(c, body.ask(a)) for a, c in second]
        retained = [(c, body.ask(a)) for a, c in first]
        res["pairs"] = {"known1": known1, "negative": negative, "known2": known2, "retained": retained}
    res["seconds"] = time.time() - t0
    return res


# ==============================================================
# ОТЧЁТ
# ==============================================================

def score_repeat(texts: list[str], said: list[list[str]]) -> dict:
    """said[испытание][вывод] — строка POS знаков: слово, '·' молчал, '?' несколько."""

    whole = right = silent = wrong = stop_ok = stop_n = 0
    for text, runs in zip(texts, said):
        ok = True
        for line in runs:
            for k in range(POS):
                c = line[k] if k < len(line) else "·"
                if k < len(text):
                    right += c == text[k]
                    silent += c == "·"
                    wrong += c not in (text[k], "·")
                    ok &= c == text[k]
                else:
                    stop_n += 1
                    stop_ok += c == "·"
                    ok &= c == "·"
        whole += ok
    n_pos = sum(len(t) for t in texts) * 2
    return {"whole": (whole, len(texts)), "right": (right, n_pos), "silent": (silent, n_pos),
            "wrong": (wrong, n_pos), "stop": (stop_ok, stop_n)}


def report_repeat(name: str, results: list[tuple[list[str], list[list[str]]]], links: dict | None) -> None:
    tot = {}
    for texts, said in results:
        for k, v in score_repeat(texts, said).items():
            tot[k] = (tot.get(k, (0, 0))[0] + v[0], tot.get(k, (0, 0))[1] + v[1])
    print(f"\n[повтори] {name}: фраза целиком (оба вывода) {share(*tot['whole'])}")
    print(f"  позиции: верно {share(*tot['right'])}; молчал {share(*tot['silent'])}; ошибся {share(*tot['wrong'])}")
    print(f"  стоп (молчит после конца фразы): {share(*tot['stop'])}")
    if links:
        for k, v in links.items():
            print(f"  звено «{k}»: {share(*v)}")
    for texts, said in results:
        print("   " + "  ".join(f"{t}→{'|'.join(s)}" for t, s in zip(texts, said)))


def report_pairs(rs: list[dict]) -> None:
    def tally(key: str, negative: bool = False):
        ok = silent = wrong = n = 0
        link = {"подсказка держится": 0, "ответ над фоном": 0, "ответ победил": 0, "ответ вышел": 0}
        for r in rs:
            base = r["base"]
            for want, a in r["pairs"][key]:
                n += 1
                said = a["said"]
                if negative:
                    ok += said in ("", UNKNOWN)
                    wrong += said not in ("", UNKNOWN)
                    continue
                ok += said == want
                silent += said == ""
                wrong += said not in (want, "")
                link["подсказка держится"] += a["hold"]
                rate = a["rates"]
                link["ответ над фоном"] += rate[want] > base[want] + 2.0 * max(base[want], 1.0)
                rival = max(hz for w, hz in rate.items() if w not in (want, "повтори", "?", a["cue"]))
                link["ответ победил"] += rate[want] > rival
                link["ответ вышел"] += rate[want] > r["thr"][want]
        return ok, silent, wrong, n, link

    for key, title in (("known1", "знакомые (первая порция)"), ("known2", "знакомые (вторая порция)"),
                       ("retained", "сохранение первой после второй")):
        ok, silent, wrong, n, link = tally(key)
        print(f"\n[пары] {title}: верно {share(ok, n)}; молчал {share(silent, n)}; ошибся {share(wrong, n)}")
        print("  звенья: " + "; ".join(f"{k} {share(v, n)}" for k, v in link.items()))
    ok, _, wrong, n, _ = tally("negative", negative=True)
    print(f"\n[пары] отрицательные (невыученное): молчал или «не знаю» {share(ok, n)}; ложный ответ {share(wrong, n)}")


def main(opts: dict) -> None:
    body = opts.get("body", "e1")
    seeds = [int(x) for x in opts.get("seeds", "1,2,3").split(",")]
    parts = opts.get("parts", "repeat,pairs").split(",")
    t0 = time.time()
    print(f"Экзамен чата: тело {body}, зёрна {seeds}, части {parts}", flush=True)
    if body == "e1":
        n = int(opts.get("trials", 8))
        jobs = {s: phrases(s, n) for s in seeds}
        if "repeat" in parts:
            with ProcessPoolExecutor(len(seeds)) as pool:
                out = dict(zip(seeds, pool.map(e1_repeat, seeds, [jobs[s] for s in seeds])))
            links = {"записано в ячейку позиции": [0, 0], "реле чтения открылось": [0, 0]}
            for s in seeds:
                for text, st, rl in zip(jobs[s], out[s]["stored"], out[s]["relay"]):
                    for k in range(len(text)):
                        links["записано в ячейку позиции"][0] += st[k]
                        links["реле чтения открылось"][0] += rl[k]
                        links["записано в ячейку позиции"][1] += 1
                        links["реле чтения открылось"][1] += 1
            report_repeat("блок E1 (не в мозге)", [(jobs[s], out[s]["said"]) for s in seeds], links)
            leak = sum(out[s]["leak"] for s in seeds), sum(out[s]["leak_windows"] for s in seeds)
            print(f"  ответ во время записи («говорит», когда должен молчать): {share(*leak)}")
        if "pairs" in parts:
            print("\n[пары] блок E1: памяти пар нет — не умеет (0; ответ не снимался)")
    else:
        rs = [brain_exam(s, opts) for s in seeds]
        if "repeat" in parts:
            report_repeat(f"мозг ({opts.get('mind', 'words')})", [(r["repeat"]["phrases"], r["repeat"]["said"]) for r in rs], None)
        if "pairs" in parts:
            report_pairs(rs)
        for r in rs:
            hot = sorted(r["base"].items(), key=lambda x: -x[1])[:3]
            print(f"  зерно {r['seed']}: фон популяций слов {sum(r['base'].values()) / len(r['base']):.1f} Гц "
                  f"(выше всех {', '.join(f'{w} {hz:.1f}' for w, hz in hot)}); {r['seconds']:.0f} с")
    print(f"\nвсего {time.time() - t0:.0f} с")


if __name__ == "__main__":
    main(dict(kv.split("=", 1) for kv in sys.argv[1:]))
