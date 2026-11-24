"""
Порт слов словесного мозга (docs/CHAT_PLAN.md, шаг 2): петля без памяти —
предъявленное слово считывается обратно.

Мозг без глаза, уха и руки (cfg.organs = ("words",)); слово — рецепторы
органа «words» (brain/body.Words), вход — L4 смысловой коры (association).

Считыватель — как поиск клеток понятий у человека (Quiroga et al. 2005,
https://doi.org/10.1038/nature03687): сначала скрининг — каждое слово
screen раз, у каждого слова отбираются 40 клеток L2/3 E с наибольшей
избирательностью (частота на своё слово минус средняя на чужие, в единицах
разброса); мозг при этом не учится (пластичность выключена). Потом
проверка — test новых предъявлений каждого слова: слово = популяция с
наибольшей частотой в окне 0–300 мс от начала слова; «сказано» — популяция
выше своей тишины на 5 sd и одна [своё: нет готового]. L4 — то же по
местам слов (вход), чтобы видеть, где теряется.

Готово (шаг 2): считывание ≥ 95% по популяциям L2/3.

    PYTHONPATH=. .venv/bin/python tools/word_port.py [seed=1] [screen=4] [test=4] [psp=2.0] [k=10] [organs=words]
"""

from __future__ import annotations

import sys
import time

import torch

from brain.body import Words
from brain.config import BrainConfig
from brain.mind import Mind

ON_MS, OFF_MS, READ_MS = 200.0, 600.0, 300.0
POP = 40
SD_THR = 5.0


def build(opts: dict):
    cfg = BrainConfig()
    cfg.seed = int(opts.get("seed", 1))
    cfg.size = opts.get("size", "small")
    cfg.organs = tuple(x for x in opts.get("organs", "words").split(",") if x)
    cfg.words_psp_mv = float(opts.get("psp", cfg.words_psp_mv))
    cfg.words_k = int(opts.get("k", cfg.words_k))
    mind = Mind(cfg)
    mind.cfg.autosave = False
    return mind, mind.brain


LAST_SEL: torch.Tensor | None = None   # избирательность последнего скрининга [слово, клетка L2/3], в разбросах
SILENT: list[torch.Tensor] = []      # частоты в окнах тишины READ_MS (последние в паузе)


def present(b, port: Words, w: int) -> torch.Tensor:
    """Слово ON_MS, тишина до OFF_MS; частоты всех клеток в окне 0–READ_MS.
    Последние READ_MS паузы — окно тишины (SILENT) для порога «сказано»."""

    r = b.observe(ON_MS, port.hear(w, ON_MS)) * ON_MS
    r += b.observe(READ_MS - ON_MS, port.hear(None, READ_MS - ON_MS)) * (READ_MS - ON_MS)
    gap = OFF_MS - READ_MS + ON_MS - READ_MS
    b.observe(gap, port.hear(None, gap))
    SILENT.append(b.observe(READ_MS, port.hear(None, READ_MS)))
    return r / READ_MS


def block(b, port: Words, n: int, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """n показов каждого слова вперемешку: (слова, частоты всех клеток [показ, клетка])."""

    order = torch.arange(b.cfg.words_count).repeat(n)
    order = order[torch.randperm(len(order), generator=gen)]
    return order, torch.stack([present(b, port, int(w)) for w in order])


def screen(b, port: Words, n: int, seed: int, gen: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Скрининг: популяции слов L2/3 E (номера клеток мозга, [слово, POP]) и порог
    «сказано» по слову, Гц. Порог [своё: нет готового]: популяция выше своей
    тишины на SD_THR разбросов (детекция сигнала; окна тишины скрининга).
    """

    W = b.cfg.words_count
    L23 = b.areas["association"].layers["L23"]["E"]
    gen = gen or torch.Generator().manual_seed(seed)
    SILENT.clear()
    y, R = block(b, port, n, gen)
    X = R[:, L23.slice]
    mean = torch.stack([X[y == w].mean(0) for w in range(W)])                  # [слово, клетка]
    sd = X.std(0) + 0.5
    pops = []
    for w in range(W):
        other = (mean.sum(0) - mean[w]) / (W - 1)
        pops.append(((mean[w] - other) / sd).argsort(descending=True)[:POP])
    pops = torch.stack(pops)                                                   # [слово, POP], от начала L2/3
    S = torch.stack(SILENT)[:, L23.slice]
    S = torch.stack([S[:, pops[w]].mean(1) for w in range(W)], 1)              # [окно, слово]
    thr = S.mean(0) + SD_THR * S.std(0)
    global LAST_SEL
    LAST_SEL = torch.stack([(mean[w] - (mean.sum(0) - mean[w]) / (W - 1)) / sd for w in range(W)])   # [слово, клетка L2/3]
    overlap = len(pops.flatten()) - len(pops.flatten().unique())
    print(f"скрининг: {n} показа на слово; клеток в двух популяциях и больше {overlap}; тишина популяций "
          f"{float(S.mean()):.1f} Гц, разброс окна {float(S.std(0).mean()):.2f} Гц → порог {float(thr.mean()):.1f} Гц "
          f"(фон + {SD_THR:.0f} sd)", flush=True)
    return pops + L23.start, thr


def main(opts: dict) -> dict:
    t0 = time.time()
    mind, b = build(opts)
    port = Words.of(b)
    W = b.cfg.words_count
    b.plasticity = False
    L23 = b.areas["association"].layers["L23"]["E"]
    L4 = b.areas["association"].layers["L4"]["E"]
    seed = int(opts.get("seed", 1))
    gen = torch.Generator().manual_seed(seed)
    print(f"словесный мозг: органы {b.cfg.organs}, зон {len(b.areas)}, нейронов {b.n}; слов {W}, "
          f"ВПСП {b.cfg.words_psp_mv} мВ × {b.cfg.words_k} рецепторов, {b.cfg.words_hz} Гц; сборка {time.time() - t0:.0f} с",
          flush=True)
    b.observe(1000.0, port.hear(None, 1000.0))
    pops, thr = screen(b, port, int(opts.get("screen", 4)), seed, gen)
    pops = pops - L23.start

    y, R = block(b, port, int(opts.get("test", 4)), gen)
    X = R[:, L23.slice]
    rate = torch.stack([X[:, pops[w]].mean(1) for w in range(W)], 1)          # [показ, слово]
    acc = float((rate.argmax(1) == y.to(rate.device)).float().mean())
    over = rate > thr
    said = torch.where(over.sum(1) == 1, over.float().argmax(1), torch.full_like(y.to(rate.device), -1))
    said_ok = float((said == y.to(rate.device)).float().mean())
    silent = float((over.sum(1) == 0).float().mean())
    own = rate.gather(1, y.to(rate.device)[:, None]).squeeze(1)
    rival = rate.scatter(1, y.to(rate.device)[:, None], -1.0).max(1).values
    # Вход: L4 по плиткам слов (кто получает рецепторы слова — по проводке порта).
    path = next(p for p in b.projections if p.name.startswith("words -> ") and p.post is L4)
    tile = (path.pre[:, 0].long() - port.offset) // port.per                  # слово клетки L4
    live = path.weight[:, 0] > 0
    X4 = torch.stack([R[:, L4.slice][:, (tile == w) & live].mean(1) for w in range(W)], 1)
    acc4 = float((X4.argmax(1) == y.to(X4.device)).float().mean())
    base4 = float(X4.mean())
    print(f"\nпроверка {int(opts.get('test', 4))} новых показа на слово (случай {1 / W:.3f}):")
    print(f"  L2/3 популяции: слово по наибольшей {acc:.2f}; «сказано» (одна выше порога) верно {said_ok:.2f}, "
          f"молчал {silent:.2f}; своя {float(own.mean()):.1f} Гц, лучшая чужая {float(rival.mean()):.1f} Гц, "
          f"фон L2/3 E {float(X.mean()):.1f} Гц")
    print(f"  L4 места слов (вход): слово по наибольшей {acc4:.2f}; своё место "
          f"{float(X4.gather(1, y.to(X4.device)[:, None]).mean()):.1f} Гц, среднее места {base4:.1f} Гц")
    print(f"всего {time.time() - t0:.0f} с")
    return {"acc": acc, "said": said_ok, "silent": silent, "acc4": acc4}


if __name__ == "__main__":
    main(dict(kv.split("=", 1) for kv in sys.argv[1:]))
