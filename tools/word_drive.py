"""
Шаг 4а (docs/CHAT_PLAN.md): сколько сигнала слова кора может отдать в стриатум.

На клетку стриатума сходятся тысячи корковых нейронов, каждый вход слабый
(Kincaid, Zheng & Wilson 1998, крыса, https://doi.org/10.1523/JNEUROSCI.18-12-04722.1998).
E1 же получает слово с 20 источников по 40 Гц (фон 2 Гц): 800 спайков/с на
слово, 40/с в тишине, контраст 20×. Замер без правок: у каждого слова —
избирательность клеток L2/3 E смысловой коры (как в скрининге word_port:
(частота на своё − средняя на чужие) / разброс), и для групп «первые 20»,
«первые 40», «избирательность > 1, > 2 sd» — сколько клеток, суммарная
частота на слово (окно 0–300 мс) и в тишине, контраст.

Мозг — как в tools/e1_in_brain.py (разгон 2.5 с, адаптация adapt окон по 1 с).

    PYTHONPATH=tools:. .venv/bin/python tools/word_drive.py [seed=1] [adapt=1] [screen=4]
"""

from __future__ import annotations

import sys
import time

import torch

import word_port
from brain.body import Words


def main(opts: dict) -> None:
    seed = int(opts.get("seed", 1))
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    t0 = time.time()
    mind, b = word_port.build({"seed": seed, "size": "small"})
    b.cfg.event_overflow = "warn"
    port = Words.of(b)
    b.cfg.homeostasis = True
    b.run(2500.0, port.hear(None, 2500.0))
    for _ in range(int(opts.get("adapt", 1))):
        b.homeostasis_update(b.observe(1000.0, port.hear(None, 1000.0)))
    b.cfg.homeostasis = False
    b.plasticity = False
    W = b.cfg.words_count
    L23 = b.areas["association"].layers["L23"]["E"]
    gen = torch.Generator().manual_seed(seed)
    word_port.SILENT.clear()
    y, R = word_port.block(b, port, int(opts.get("screen", 4)), gen)
    X = R[:, L23.slice]
    S = torch.stack(word_port.SILENT)[:, L23.slice].mean(0)                    # тишина клетки, Гц
    mean = torch.stack([X[y == w].mean(0) for w in range(W)])
    sd = X.std(0) + 0.5
    print(f"зерно {seed}: L2/3 E смысловой {L23.size} клеток, слов {W}; тишина {float(S.mean()):.2f} Гц; "
          f"{time.time() - t0:.0f} с", flush=True)
    print("E1 ждёт: на слово 800 спайков/с (20 × 40 Гц), в тишине 40/с, контраст 20×")
    groups = {"первые 20": 20, "первые 40": 40, "изб. > 2 sd": 2.0, "изб. > 1 sd": 1.0, "изб. > 0.5 sd": 0.5}
    rows = {g: [] for g in groups}
    for w in range(W):
        other = (mean.sum(0) - mean[w]) / (W - 1)
        sel = (mean[w] - other) / sd
        order = sel.argsort(descending=True)
        for g, v in groups.items():
            idx = order[:v] if isinstance(v, int) else torch.nonzero(sel > v).squeeze(1)
            on = float(mean[w][idx].sum())
            off = float(S[idx].sum())
            rows[g].append((len(idx), on, off))
    for g, rr in rows.items():
        n = torch.tensor([r[0] for r in rr], dtype=torch.float)
        on = torch.tensor([r[1] for r in rr])
        off = torch.tensor([r[2] for r in rr])
        print(f"  {g:14s}: клеток {float(n.mean()):7.1f} (мин {int(n.min())}, макс {int(n.max())}); "
              f"на слово {float(on.mean()):8.0f} спайков/с; в тишине {float(off.mean()):7.0f}/с; "
              f"контраст {float((on / off.clamp(min=1e-6)).mean()):5.1f}×", flush=True)


if __name__ == "__main__":
    main(dict(kv.split("=", 1) for kv in sys.argv[1:]))
