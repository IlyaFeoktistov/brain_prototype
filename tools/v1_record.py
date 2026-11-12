"""
Спайки V1 L2/3 E мозга на буквах — для офлайн-обучения пути V1 → буквы
(tools/letters_offline.py): выучиваема ли буква из такого входа и за сколько
показов, без часов полного мозга (DESIGN «Обучение пути V1 → буквы»).

Мозг — как в letters_dc (reading_bench.build: LGN A, фон, гомеостаз lgn-d
зерна, trn_scale 0.1), пластичность выкл. Показ как в letters_dc: пусто rest
мс, затем буква show мс по центру ямки со сдвигом ±1 px. Шаг мозга 0.5 мс
(CHUNK = 1, как lgn_record): спайки по шагам, временной строй сохранён.

Пишет: по показу спайки [n, 2] (шаг от начала окна, индекс клетки V1 L2/3 E),
метки, сдвиги; входы пути visual -> letters (источники V1 L2/3 E на клетку букв
L4 E, врождённые веса, мВ) и место клеток букв на карте (столбец поля).

    PYTHONPATH=. .venv/bin/python tools/v1_record.py [letters=абвгдежзик] [trials=30] [rest=150] [show=350] [seed=1] out=файл.pt
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
    opts = {"fit": "lgn-a", "homeo": f"lgn-d-pop-s{seed}", "relay_v_t_mv": "-59", "trn_scale": "0.1", **opts}
    mind, b = build(opts)
    s = Session(mind, window_ms=10.0)
    cfg = b.cfg
    letters = opts.get("letters", "абвгдежзик")
    trials = int(opts.get("trials", 30))
    rest, show = float(opts.get("rest", 150)), float(opts.get("show", 350))
    gen = torch.Generator().manual_seed(seed + 2000)
    torch.manual_seed(seed)
    src = b.areas["visual"].layers["L23"]["E"]
    post = b.areas["letters"].layers["L4"]["E"]
    path = next(p for p in b.plastic if p.name == "visual -> letters")
    assert path.post is post
    pre = path.pre.long().cpu() - src.start
    assert int(pre.min()) >= 0 and int(pre.max()) < src.size

    def step_run(view: torch.Tensor) -> list[torch.Tensor]:
        out = []
        for t in range(view.shape[0]):
            b.run(cfg.dt_ms, view[t:t + 1])
            out.append(b.spikes[src.slice].nonzero()[:, 0].cpu())
        return out

    s.line, s.gaze = "", 0.0
    t0 = time.time()
    b.run(2500.0, s._blank(2500.0))
    order = torch.arange(len(letters)).repeat(trials)
    order = order[torch.randperm(len(order), generator=gen)]
    spikes, shifts = [], []
    for j, k in enumerate(order.tolist()):
        s.line = ""
        steps = step_run(s._blank(rest))
        shift = int(torch.randint(-1, 2, (1,), generator=gen))
        s.line, s.gaze = letters[k], s.screen.width(letters[k]) / 2 + shift
        steps += step_run(s._view(show))
        s.line = ""
        rows = [torch.stack([torch.full_like(c, t), c], 1) for t, c in enumerate(steps) if len(c)]
        spikes.append((torch.cat(rows) if rows else torch.zeros(0, 2, dtype=torch.long)).to(torch.int32))
        shifts.append(shift)
        if (j + 1) % 20 == 0:
            n_rest = int(round(rest / cfg.dt_ms))
            x = spikes[-1]
            print(f"  показ {j + 1}/{len(order)}, {time.time() - t0:.0f} с: V1 L2/3 фон "
                  f"{float((x[:, 0] < n_rest).sum()) / src.size / (rest / 1000):.2f}, показ "
                  f"{float((x[:, 0] >= n_rest).sum()) / src.size / (show / 1000):.2f} Гц", flush=True)
    torch.save({"spikes": spikes, "labels": order, "shifts": torch.tensor(shifts), "letters": letters,
                "dt_ms": cfg.dt_ms, "rest_ms": rest, "show_ms": show, "n_src": src.size,
                "pre": pre.to(torch.int32), "weight": path.weight.float().cpu(), "cap": float(path.cap),
                "field_col": b.field_cols[post.name].cpu(), "fovea_col": float(cfg.fovea_col),
                "seed": seed}, opts["out"])
    print(f"записано {len(order)} показов → {opts['out']}, {time.time() - t0:.0f} с", flush=True)


if __name__ == "__main__":
    main(dict(kv.split("=", 1) for kv in sys.argv[1:]))
