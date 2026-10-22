"""
Протокол Campbell et al. 2020 (мышь, срез: оптогенетика терминалей TRN, 5
импульсов): все клетки зрительного TRN выстреливают разом с частотой 20 или
50 Гц, 5 импульсов; клетки реле LGN — в работающей сети. Мера — средний V
реле во время поезда минус тот же отрезок без поезда (контроль), и ответ на
первый и пятый импульс (провал V за 5 мс после импульса). Число входов TRN
на клетку реле — trn_inputs (остальные построенные — выключены).

Сверка (порядок величины, мышь): гиперполяризация 2.39 ± 0.24 мВ при 20 Гц,
4.03 ± 0.39 при 50 Гц; ответ к 5-му импульсу 47–56% первого
(https://doi.org/10.1152/jn.00757.2019). Депрессии TRN → LGN в модели нет —
её время восстановления не мерено.

    PYTHONPATH=. python tools/trn_train.py [inputs=1,2,5,10,20] [repeats=10]
"""

from __future__ import annotations

import sys

import torch

from brain.config import BrainConfig
from brain.mind import Mind


def main(inputs: list[int], repeats: int) -> None:
    cfg = BrainConfig()
    cfg.size = "small"
    cfg.autosave = False
    b = Mind(cfg).brain
    b.plasticity = False
    lgn, trn = b.areas["lgn"].E, b.areas["trn"].E
    proj = next(p for p in b.projections if p.post is lgn and p.name == "trn -> lgn")
    full = proj.weight.clone()
    b.run(2500.0)
    b.CHUNK, b._graphs = 1, {}
    dt = cfg.dt_ms
    every = torch.arange(trn.size, device=b.device)
    print(f"TRN → реле: {proj.weight.shape[1]} входов построено; V реле в покое сети {float(b.v[lgn.slice].mean()):.1f} мВ")

    def train(hz: float, kick: bool) -> tuple[torch.Tensor, list[int]]:
        period = int(round(1000 / hz / dt))
        span = 5 * period + int(10 / dt)
        trace = torch.zeros(span)
        for _ in range(repeats):
            for _ in range(int(100 / dt)):
                b.run(dt)
            for k in range(span):
                if kick and k % period == 0 and k < 5 * period:
                    b.kick("trn", every, 400.0, dt)
                b.run(dt)
                trace[k] += float(b.v[lgn.slice].mean()) / repeats
        return trace, [i * period for i in range(5)]

    for n in inputs:
        proj.weight.copy_(full)
        proj.weight[:, n:] = 0.0
        row = []
        for hz in (20.0, 50.0):
            on, pulses = train(hz, True)
            off, _ = train(hz, False)
            d = on - off
            window = int(5 / dt)
            first = -float(d[pulses[0]: pulses[0] + window].min())
            fifth = -float(d[pulses[4]: pulses[4] + window].min() - d[pulses[4]])
            mean = -float(d[: pulses[4] + window].mean())
            row.append(f"{hz:.0f} Гц: гиперполяризация {mean:5.2f} мВ, 1-й {first:4.2f}, 5-й {fifth:4.2f} ({fifth / max(first, 1e-6):.0%})")
        print(f"входов TRN {n:2d} | " + " | ".join(row))

    proj.weight.copy_(full)
    print("сверка (мышь, срез): 20 Гц 2.39 ± 0.24 мВ, 50 Гц 4.03 ± 0.39; 5-й ответ — 47–56% первого")


if __name__ == "__main__":
    opts = dict(kv.split("=", 1) for kv in sys.argv[1:])
    main([int(x) for x in opts.get("inputs", "1,2,5,10,20").split(",")], int(opts.get("repeats", 10)))
