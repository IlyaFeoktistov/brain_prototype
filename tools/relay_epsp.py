"""
ВПСП сетчатки на клетке реле LGN: форма одиночного (нарастание 10–90%,
пик, спад до половины) и сумма двух с интервалом 5 / 10 / 20 / 40 мс — пик и
доля клеток со спайком в 10 мс после второго; с триадой и без. Клетки — в
работающем мозге (TRN, кора, фон), спонтанные спайки сетчатки выключены.

Сверка: макака — спайк реле обычно от суммы ≥ 2 ВПСП за 40 мс, передача
второго при коротком интервале — до ~1 (Sincich et al. 2007); кошка, пара —
после паузы 30%, второй через 10 мс 47% (Usrey, Reppas & Reid 1998).

    PYTHONPATH=. python tools/relay_epsp.py [repeats=20] [trn_scale=1]
"""

from __future__ import annotations

import sys

import torch

from brain.config import BrainConfig
from brain.mind import Mind


def main(repeats: int, trn_scale: float = 1.0) -> None:
    cfg = BrainConfig()
    cfg.size = "small"
    cfg.autosave = False
    b = Mind(cfg).brain
    b.plasticity = False
    lgn = b.areas["lgn"].E
    # Анализ чувствительности: проводимость TRN → реле × trn_scale (нормировка
    # ТПСП при другом потенциале: ×19/11 при −63 мВ, ×19/29 при −45; E_ГАМК −74).
    for p in b.projections:
        if p.post is lgn and p.name == "trn -> lgn":
            p.weight.mul_(trn_scale)
    eye = [x.name for x in b.sensors].index("eye")
    off, size = b.sensor_offsets[eye], b.sensors[eye].size
    b.spontaneous[off: off + size] = 0.0
    b.run(2500.0)
    b.CHUNK, b._graphs = 1, {}
    dt = cfg.dt_ms
    cells = torch.arange(0, size, 4, device=b.device)
    quiet = torch.zeros(1, b.n_receptors, device=b.device)
    kick = quiet.clone()
    kick[0, off + cells] = 1.0
    triad = [p for p in b.projections if p.post is lgn and "триада" in p.name]

    def trial(gap_ms: float | None) -> tuple[torch.Tensor, float]:
        """Средний ход V (относительно V до первого) и доля клеток со спайком."""

        span = int(((gap_ms or 0) + 20) / dt)
        trace, fired = torch.zeros(span), 0.0

        for _ in range(repeats):
            for _ in range(int(80 / dt)):
                b.run(dt, quiet)
            v0 = b.v[lgn.start + cells].clone()
            spiked = torch.zeros(len(cells), dtype=torch.bool, device=b.device)

            for k in range(span):
                second = gap_ms is not None and k == int(gap_ms / dt)
                b.run(dt, kick if k == 0 or second else quiet)
                trace[k] += float((b.v[lgn.start + cells] - v0).mean()) / repeats
                if gap_ms is None or k >= int(gap_ms / dt):
                    spiked |= b.spikes[lgn.start + cells] > 0
            fired += float(spiked.float().mean()) / repeats

        return trace, fired

    for label, on in (("с триадой", True), ("без триады", False)):
        saved = [p.weight.clone() for p in triad]
        if not on:
            for p in triad:
                p.weight.zero_()

        print(f"\n{label}: V клеток реле до ВПСП {float(b.v[lgn.slice].mean()):.1f} мВ")
        one, p1 = trial(None)
        peak = float(one.max())
        k_peak = int(one.argmax())
        rise = [i for i, x in enumerate(one[: k_peak + 1].tolist()) if x >= 0.1 * peak]
        top = [i for i, x in enumerate(one[: k_peak + 1].tolist()) if x >= 0.9 * peak]
        half = next((i for i in range(k_peak, len(one)) if one[i] <= 0.5 * peak), len(one))
        print(f"   один ВПСП: пик {peak:.2f} мВ через {k_peak * dt:.1f} мс, нарастание 10–90% {(top[0] - rise[0]) * dt:.1f} мс, "
              f"спад до половины {(half - k_peak) * dt:.1f} мс, спайк у {p1:.0%} клеток")

        for gap in (5.0, 10.0, 20.0, 40.0):
            two, p2 = trial(gap)
            print(f"   два через {gap:4.0f} мс: пик суммы {float(two.max()):5.2f} мВ, спайк после второго у {p2:.0%}")

        for p, w in zip(triad, saved):
            p.weight.copy_(w)


if __name__ == "__main__":
    opts = dict(kv.split("=", 1) for kv in sys.argv[1:])
    main(int(opts.get("repeats", 20)), float(opts.get("trn_scale", 1.0)))
