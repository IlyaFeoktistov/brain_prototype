"""
Передача сетчатка → реле LGN мозга спайк в спайк — та же мера, что у
макаки: доля спайков водителя (сетчатки), за которыми реле стреляет в
окне 1–win мс (Carandini, Horton & Sincich 2007, https://doi.org/10.1167/7.14.20:
49 ± 14%; Sincich et al. 2007, https://doi.org/10.1523/JNEUROSCI.5077-06.2007:
> 95% спайков LGN — после ВПСП своего водителя), и обратная доля — спайков
реле, перед которыми был спайк водителя. Случайный уровень — та же мера со
спайками водителя, сдвинутыми на 50 мс (вероятность попасть в окно без
причинной связи). Без вмешательства; мозг как в letters_dc (без пластичности),
шаг 0.5 мс без пачек.

Диагностика (не правка): off=noise|l6|trn|in — у реле выключен один вход
(случайный фон движка, V1 L6 → LGN, TRN → LGN, интернейроны LGN → реле):
откуда спайки реле без спайка водителя перед ними.

Условия: пустой экран (поддерживаемая частота сетчатки) и буква по центру
ямки — по каждому показу пауза rest мс, буква show мс.

    PYTHONPATH=. .venv/bin/python tools/relay_transfer.py [letters=абвгд] [rest=300] [show=350] [win=5] [seed=1] [homeo=lgn-d-pop-s1] [off=noise|l6|trn|in]
"""

from __future__ import annotations

import sys

import torch

from brain.engine import SpikingBrain
from brain.session import Session

SpikingBrain.CHUNK = 1

from tools.reading_bench import build  # noqa: E402


def main(opts: dict) -> None:
    seed = int(opts.get("seed", 1))
    opts = {"fit": "lgn-a", "homeo": f"lgn-d-pop{'' if opts.get('tc_indep') == '0' else '-tcind'}-s{seed}", "relay_v_t_mv": "-59", "trn_scale": "0.1", **opts}
    mind, b = build(opts)
    s = Session(mind, window_ms=10.0)
    cfg = b.cfg
    letters = opts.get("letters", "абвгд")
    rest, show, win = float(opts.get("rest", 300)), float(opts.get("show", 350)), float(opts.get("win", 5))
    lgn = b.areas["lgn"].E
    eye = [x.name for x in b.sensors].index("eye")
    off = b.sensor_offsets[eye]
    driver = next(p for p in b.projections if p.name == "eye -> " + lgn.name).pre[:, 0].long() - off
    dt = cfg.dt_ms
    off_name = opts.get("off", "")
    if off_name == "noise":
        b.noise_gain[lgn.slice] = 0.0
    elif off_name:
        name = {"l6": "visual -> lgn", "trn": "trn -> lgn", "in": "lgn: I->E"}[off_name]
        next(p for p in b.projections if p.post is lgn and p.name == name).weight.zero_()
    if off_name:
        print(f"выключено у реле: {off_name}", flush=True)

    def record(view: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Растры водителя и реле [шагов, клеток реле]."""

        ret, rel = [], []
        for t in range(view.shape[0]):
            b.run(dt, view[t:t + 1])
            ret.append(view[t, off + driver] > 0)
            rel.append(b.spikes[lgn.slice] > 0)
        return torch.stack(ret), torch.stack(rel)

    s.line, s.gaze = "", 0.0
    b.run(2500.0, s._blank(2500.0))
    lo, hi = max(1, round(1.0 / dt)), round(win / dt)

    def transfer(ret: torch.Tensor, rel: torch.Tensor, shift: int = 0) -> tuple[float, float, int]:
        """Доля спайков водителя со спайком реле через 1–win мс; доля спайков реле с водителем перед ними."""

        if shift:
            ret = torch.roll(ret, shift, 0)
        T = ret.shape[0]
        after = torch.zeros_like(rel)
        for k in range(lo, hi + 1):
            after[:T - k] |= rel[k:]
        before = torch.zeros_like(ret)
        for k in range(lo, hi + 1):
            before[k:] |= ret[:T - k]
        n_ret = int(ret.sum())
        p_fwd = float((ret & after).sum()) / max(n_ret, 1)
        p_back = float((rel & before).sum()) / max(int(rel.sum()), 1)
        return p_fwd, p_back, n_ret

    acc = {"пустой экран": [], "буква": []}
    for ch in letters:
        s.line = ""
        r0 = record(s._blank(rest))
        s.line, s.gaze = ch, s.screen.width(ch) / 2
        r1 = record(s._view(show))
        s.line = ""
        acc["пустой экран"].append(r0)
        acc["буква"].append(r1)

    print(f"\nпередача сетчатка → реле, окно 1–{win:g} мс (макака: 0.49 ± 0.14 вперёд, > 0.95 назад)")
    for name, rs in acc.items():
        ret = torch.cat([x[0] for x in rs])
        rel = torch.cat([x[1] for x in rs])
        sec = ret.shape[0] * dt / 1000.0
        f, bk, n = transfer(ret, rel)
        fc, bc, _ = transfer(ret, rel, shift=round(50.0 / dt))
        print(f"  {name:13s} сетчатка {n / ret.shape[1] / sec:6.1f} Гц, реле {float(rel.sum()) / rel.shape[1] / sec:6.1f} Гц; "
              f"вперёд {f:.2f} (случайно {fc:.2f}), назад {bk:.2f} (случайно {bc:.2f})", flush=True)
        # Задержка спайка реле после спайка водителя (коррелограмма): шаги 0–20 мс.
        T = ret.shape[0]
        lags = [float((ret[:T - k] & rel[k:]).sum()) for k in range(0, round(20.0 / dt) + 1)]
        base = float(ret.float().mean() * rel.float().mean() * ret.shape[1] * T)
        print("    задержка реле после водителя, мс: " + ", ".join(
            f"{k * dt:g}:{x / max(base, 1e-9):.1f}" for k, x in enumerate(lags) if k <= 8 or k % 4 == 0)
              + "  (× случайного)", flush=True)
        if name == "буква":
            # Только реле, чей водитель под штрихом стреляет чаще поддерживаемой.
            hot = ret.float().mean(0) / (dt / 1000.0) > 2 * cfg.retina_maintained_hz
            f, bk, n = transfer(ret[:, hot], rel[:, hot])
            fc, bc, _ = transfer(ret[:, hot], rel[:, hot], shift=round(50.0 / dt))
            print(f"  {'под штрихом':13s} {int(hot.sum())} реле; вперёд {f:.2f} (случайно {fc:.2f}), "
                  f"назад {bk:.2f} (случайно {bc:.2f})", flush=True)


if __name__ == "__main__":
    main(dict(kv.split("=", 1) for kv in sys.argv[1:]))
