"""
Учится ли зона букв различать увиденное (звено 1 ROADMAP, DESIGN «Зона букв:
V1 — водитель, различие в токе, не в спайках»).

Мозг как в reading_bench (LGN A, фон, гомеостаз зерна). Три части:
    до        проверка: A и B вперемешку, n повторов, чтение, без пластичности;
    учёба     train показов каждой фразы вперемешку, тем же чтением —
              пластичность мозга включена (plastic=0 — контроль: то же время
              без пластичности);
    после     проверка на новых показах, как «до».
Учёба и проверка — разные показы; классификатор (ближайший центр по
корреляции, выброс одного повтора — l4_decode) видит только показы своей
проверки. Признаки — спайки клеток за каждую фиксацию (V1 L2/3 — для
сравнения, буквы L4 и L2/3) и ток возбуждения букв L4. Случайно —
перемешанные метки. Изменение пути V1 → буквы: средний |Δw| в долях
среднего веса.

    PYTHONPATH=. python tools/letters_learn.py [train=60] [n=12] [plastic=1] [seed=1] [fix=225] [step=5]
"""

from __future__ import annotations

import sys
import time

import torch

from brain.session import Session
from tools.letters_drive import loo
from tools.reading_bench import build

WIN = 10.0


def main(opts: dict) -> None:
    opts = {"fit": "lgn-a", "homeo": f"lgn-a-s{opts.get('seed', 1)}", "relay_v_t_mv": "-59",
            "relay_rest_mv": "-48", "trn_scale": "0.5", **opts}
    mind, b = build(opts)
    cfg = b.cfg
    s = Session(mind, window_ms=WIN)
    n_test, n_train = int(opts.get("n", 12)), int(opts.get("train", 60))
    fix, step = float(opts.get("fix", 225)), float(opts.get("step", 5))
    texts = (opts.get("a", "мама мыла раму"), opts.get("b", "папа пила чай"))
    gen = torch.Generator().manual_seed(int(opts.get("seed", 1)))
    vis, let = b.areas["visual"].layers, b.areas["letters"].layers
    stages = {"V1 L2/3": vis["L23"]["E"], "буквы L4": let["L4"]["E"], "буквы L2/3": let["L23"]["E"]}
    path = next(p for p in b.projections if p.name == "visual -> letters")
    w0 = path.weight.float().clone()

    s.line, s.gaze = "", 0.0
    b.run(2500.0)

    def gazes(line: str) -> list[float]:
        width = s.screen.width(line) / max(1, len(line))
        n = max(1, int((len(line) + step - 1) // step))
        return [i * step * width for i in range(n)][:3]

    def show(lab: int) -> dict[str, torch.Tensor]:
        """Один показ фразы чтением; признаки по фиксациям."""

        s.line = ""
        b.run(500.0, s._blank(500.0))
        line = texts[lab] + " "
        rows = {k: [] for k in [*stages, "ток букв L4"]}
        for i, g in enumerate(gazes(line)):
            if i:
                s.line, s.gaze = line, g
                b.run(cfg.saccade_ms, s._blank(cfg.saccade_ms))
            s.line, s.gaze = line, g
            acc = {k: 0 for k in rows}
            n_win = int(fix // WIN)
            for _ in range(n_win):
                r = b.observe(WIN, s._view(WIN)) * (WIN / 1000.0)
                for k, pop in stages.items():
                    acc[k] = acc[k] + r[pop.slice].cpu()
                acc["ток букв L4"] = acc["ток букв L4"] + b.i_syn[stages["буквы L4"].slice].float().cpu() / n_win
            for k in rows:
                rows[k].append(acc[k])
        return {k: torch.cat(v) for k, v in rows.items()}

    def shuffled(n: int) -> list[int]:
        order = [0, 1] * n
        return [order[i] for i in torch.randperm(len(order), generator=gen)]

    def test(name: str) -> None:
        b.plasticity = False
        order = shuffled(n_test)
        feats = [show(lab) for lab in order]
        y = torch.tensor(order)
        dw = float((path.weight.float() - w0).abs().mean() / w0.abs().mean())
        print(f"\n=== {name}: по {n_test} показов, |Δw| V1 → буквы {dw:.4f} среднего веса")
        print(f"  {'ступень':12s} {'сп/кл/фикс':>10s} {'точность':>9s} {'случайно':>9s}")
        for k in feats[0]:
            x = torch.stack([f[k] for f in feats])
            ch = float(torch.tensor([loo(x, y[torch.randperm(len(y), generator=gen)]) for _ in range(20)]).mean())
            print(f"  {k:12s} {float(x.mean()) * (1 if k.startswith('ток') else 1):10.3f} {loo(x, y):9.2f} {ch:9.2f}",
                  flush=True)

    test("до учёбы")
    plastic = opts.get("plastic", "1") == "1"
    t0 = time.time()
    for i, lab in enumerate(shuffled(n_train)):
        b.plasticity = plastic
        show(lab)
        if (i + 1) % 20 == 0:
            dw = float((path.weight.float() - w0).abs().mean() / w0.abs().mean())
            print(f"  учёба {i + 1}/{2 * n_train} показов, |Δw| {dw:.4f}, {time.time() - t0:.0f} с", flush=True)
    test(f"после учёбы ({2 * n_train} показов, пластичность {'вкл' if plastic else 'выкл — контроль'})")


if __name__ == "__main__":
    main(dict(kv.split("=", 1) for kv in sys.argv[1:]))
