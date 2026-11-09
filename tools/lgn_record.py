"""
Спайки реле LGN мозга на буквах — для замены 6 блока Chariker
(tools/block_chariker.py lgn_replay=файл): вместо LGN блока в L4 блока
идут спайки LGN мозга.

Мозг — как в letters_dc (reading_bench.build: LGN A, фон, гомеостаз зерна,
trn_scale 0.5), пластичность выкл. Показы — в порядке блока: буквы по
кругу, trials раз; каждый — пусто rest мс (растр глаза нулевой, как в
letters_dc), затем буква show мс по центру ямки со сдвигом ±1 px (как
letters_dc и блок). Шаг мозга 0.5 мс без пачек (CHUNK = 1, как
ic_latency): спайки по шагам.

Окно блока 16 × 16 px — столбцы ямки 20–35 (fovea_col 28 ↔ середина окна
8): клетка реле → её водитель (путь eye -> lgn) → (полярность, строка,
столбец) → индекс LGN блока pol · 256 + строка · 16 + столбец − 20.

Пишет: по показу спайки [n, 2] (шаг мозга от начала показа, индекс блока),
счёт всех 1792 реле за показ (различение LGN мозга на всём поле), буквы,
метки, сдвиги. Печатает временной строй LGN мозга: частоты фон / показ,
CV межспайковых интервалов, доля спайков в пачках (интервал ≤ 4 мс после
≥ 100 мс тишины — Lu, Guido & Sherman 1992,
https://doi.org/10.1152/jn.1992.68.6.2185).

    PYTHONPATH=. .venv/bin/python tools/lgn_record.py [letters=абвгдежзик] [trials=10] [rest=300] [show=350] [seed=1] out=файл.pt
"""

from __future__ import annotations

import sys
import time

import torch

from brain.engine import SpikingBrain
from brain.session import Session

SpikingBrain.CHUNK = 1

from tools.reading_bench import build  # noqa: E402


def main(opts: dict) -> None:
    seed = int(opts.get("seed", 1))
    opts = {"fit": "lgn-a", "homeo": f"lgn-a-s{seed}", "relay_v_t_mv": "-59", "relay_rest_mv": "-48",
            "trn_scale": "0.5", **opts}
    mind, b = build(opts)
    s = Session(mind, window_ms=10.0)
    cfg = b.cfg
    letters = opts.get("letters", "абвгдежзик")
    trials = int(opts.get("trials", 10))
    rest, show = float(opts.get("rest", 300)), float(opts.get("show", 350))
    gen = torch.Generator().manual_seed(seed + 1000)
    torch.manual_seed(seed)
    lgn = b.areas["lgn"].E
    eye = [x.name for x in b.sensors].index("eye")
    off = b.sensor_offsets[eye]
    driver = next(p for p in b.projections if p.name == "eye -> " + lgn.name).pre[:, 0].long().cpu() - off
    layer = cfg.retina_rows * cfg.retina_cols
    pol, rc = driver // layer, driver % layer
    row, col = rc // cfg.retina_cols, rc % cfg.retina_cols
    c0 = int(cfg.fovea_col) - 8
    inside = (col >= c0) & (col < c0 + 16)
    to_block = torch.where(inside, pol * 256 + row * 16 + (col - c0), -1)
    print(f"LGN мозга {lgn.size} реле, в окне блока {int(inside.sum())} (столбцы {c0}–{c0 + 15}); "
          f"шаг {cfg.dt_ms} мс", flush=True)

    def step_run(view: torch.Tensor) -> list[torch.Tensor]:
        out = []
        for t in range(view.shape[0]):
            b.run(cfg.dt_ms, view[t:t + 1])
            out.append(b.spikes[lgn.slice].nonzero()[:, 0].cpu())
        return out

    s.line, s.gaze = "", 0.0
    t0 = time.time()
    b.run(2500.0)
    print(f"разгон 2.5 с: {time.time() - t0:.0f} с", flush=True)

    labels = torch.arange(len(letters)).repeat(trials)
    n_rest, n_show = int(round(rest / cfg.dt_ms)), int(round(show / cfg.dt_ms))
    spikes, counts, shifts = [], [], []
    trains = [[] for _ in range(lgn.size)]  # времена спайков (мс) для временного строя
    t_ms = 0.0
    for j, k in enumerate(labels.tolist()):
        ch = letters[k]
        s.line = ""
        steps = step_run(s._blank(rest))
        shift = int(torch.randint(-1, 2, (1,), generator=gen))
        s.line, s.gaze = ch, s.screen.width(ch) / 2 + shift
        steps += step_run(s._view(show))
        s.line = ""
        rows_ = [torch.stack([torch.full_like(c, t), c], 1) for t, c in enumerate(steps) if len(c)]
        all_ = torch.cat(rows_) if rows_ else torch.zeros(0, 2, dtype=torch.long)
        for t, c in all_.tolist():
            trains[c].append(t_ms + t * cfg.dt_ms)
        t_ms += (n_rest + n_show) * cfg.dt_ms
        shown = all_[all_[:, 0] >= n_rest]
        counts.append(torch.bincount(shown[:, 1], minlength=lgn.size).float())
        blk = to_block[all_[:, 1]]
        spikes.append(torch.stack([all_[:, 0], blk], 1)[blk >= 0])
        shifts.append(shift)
        if (j + 1) % 10 == 0:
            bg = float((all_[:, 0] < n_rest).sum()) / lgn.size / (rest / 1000)
            sh = float(len(shown)) / lgn.size / (show / 1000)
            print(f"  показ {j + 1}/{len(labels)} «{ch}», {time.time() - t0:.0f} с: LGN мозга фон {bg:.1f}, "
                  f"показ {sh:.1f} Гц (все 1792)", flush=True)

    # Временной строй: CV интервалов (клетки с ≥ 20 спайками), доля спайков в пачках.
    cvs, burst, total = [], 0, 0
    for tr in trains:
        if len(tr) < 20:
            continue
        x = torch.tensor(tr)
        isi = x.diff()
        cvs.append(float(isi.std() / isi.mean()))
        pre = torch.cat([torch.tensor([1e9]), isi])
        post = torch.cat([isi, torch.tensor([1e9])])
        start = (pre >= 100.0) & (post <= 4.0)
        # Спайки пачки: начало и следующие подряд с интервалом ≤ 4 мс.
        inb = start.clone()
        for i in range(1, len(x)):
            if inb[i - 1] and isi[i - 1] <= 4.0:
                inb[i] = True
        burst += int(inb.sum())
        total += len(tr)
    cv = torch.tensor(cvs)
    print(f"временной строй LGN мозга: CV интервалов медиана {float(cv.median()):.2f} ({len(cvs)} клеток), "
          f"спайков в пачках {burst / max(total, 1):.1%}", flush=True)

    torch.save({"letters": letters, "labels": labels, "rest": rest, "show": show, "dt": cfg.dt_ms,
                "spikes": spikes, "counts": torch.stack(counts), "shifts": torch.tensor(shifts),
                "in_window": inside}, opts["out"])
    print(f"записано {opts['out']}, всего {time.time() - t0:.0f} с", flush=True)


if __name__ == "__main__":
    main(dict(kv.split("=", 1) for kv in sys.argv[1:]))
