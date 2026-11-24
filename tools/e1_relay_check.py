"""
Шаг 4а (docs/CHAT_PLAN.md): расчёт реле MD E1 (tools/e1_gate.Relay, числа
E1: gsnr 16, фон bg 1, gctx 4 нС) — пропускает ли растормаженное реле
содержание само, без сигнала задачи. Если да, «задача на реле как второе
условие» без изменения ведущего входа невозможна.

Вход реле — суммарные спайки клеток своего слова (пуассон):
  0; 364/с — фон слова из коры (схождение, tools/word_drive.py);
  909/с — слово из коры; 1300/с — удерживаемое слово в ячейке E1 (вход
  реле чтения, замер на E1 отдельно).
SNr: тонически (60 Гц × 30) всё время, или пауза 150 мс (как переход
позиции, tools/e1_gate_check.py) с 300 до 450 мс. Меры: частота реле в
паузе, пик потенциала, мс выше −52 мВ; профиль — картинка.

    PYTHONPATH=tools:. .venv/bin/python tools/e1_relay_check.py [trials=32] [out=/tmp/e1b/relay.png]
"""

from __future__ import annotations

import sys

import torch

import plots
from e1_gate import NR, Relay
from e1_slot import DT, P

LEVELS = [0.0, 364.0, 909.0, 1300.0]
TOTAL, PAUSE = 600.0, (300.0, 450.0)
BIN = 5.0


def run(level: float, pause: bool, B: int) -> dict:
    md = Relay(B, "cpu", 16.0, 1.0, 4.0)
    nb = int(TOTAL / BIN)
    sb = int(round(BIN / DT))
    v_tr, r_tr = torch.zeros(nb), torch.zeros(nb)
    vmax = torch.full((B,), -80.0)
    above = torch.zeros(B)
    for b in range(nb):
        for k in range(sb):
            t = b * BIN + k * DT
            snr_on = torch.ones(B, dtype=torch.bool)
            if pause and PAUSE[0] <= t < PAUSE[1]:
                snr_on[:] = False
            ctx = torch.zeros(B, P)
            ctx[:, 0] = level                                  # слово 0 — испытуемое, остальные 0
            md.step_counts(torch.poisson((ctx * 1e-3 * DT).repeat_interleave(NR, 1)), snr_on)
            v0 = md.v.view(B, P, NR)[:, 0]
            v_tr[b] += float(v0.mean()) / sb
            r_tr[b] += float(md.spk.view(B, P, NR)[:, 0].mean()) / sb * (1000.0 / DT)
            if PAUSE[0] <= t < PAUSE[1]:
                vmax = torch.maximum(vmax, v0.max(1).values)
                above += (v0 > -52.0).float().mean(1) * DT
    in_p = slice(int(PAUSE[0] / BIN), int(PAUSE[1] / BIN))
    return {"v": v_tr, "r": r_tr, "rate_pause": float(r_tr[in_p].mean()), "vmax": float(vmax.mean()),
            "above": float(above.mean())}


def main(opts: dict) -> None:
    torch.set_num_threads(2)
    torch.manual_seed(1)
    B = int(opts.get("trials", 32))
    res = {}
    for lv in LEVELS:
        for pause in (False, True):
            r = run(lv, pause, B)
            res[(lv, pause)] = r
            print(f"вход {lv:6.0f}/с, SNr {'пауза 150 мс' if pause else 'тонически  '}: реле в окне паузы "
                  f"{r['rate_pause']:6.1f} Гц; пик потенциала {r['vmax']:6.1f} мВ; выше −52 мВ {r['above']:5.1f} мс", flush=True)
    t_ms = (torch.arange(int(TOTAL / BIN)) + 1) * BIN
    panels = [{"label": "потенциал реле\n(пауза SNr), мВ", "series": {f"вход {lv:.0f}/с": res[(lv, True)]["v"] for lv in LEVELS}},
              {"label": "частота реле\n(пауза SNr), Гц", "series": {f"вход {lv:.0f}/с": res[(lv, True)]["r"] for lv in LEVELS},
               "smooth": 2},
              {"label": "частота реле\n(SNr тонически), Гц", "series": {f"вход {lv:.0f}/с": res[(lv, False)]["r"] for lv in LEVELS},
               "smooth": 2}]
    out = opts.get("out", "/tmp/e1b/relay.png")
    plots.timeline(out, t_ms, panels, windows=[(PAUSE[0], PAUSE[1], "пауза SNr 150 мс")],
                   title="Реле MD E1: пропускает ли растормаженное реле содержание без сигнала задачи")
    print(out)


if __name__ == "__main__":
    main(dict(kv.split("=", 1) for kv in sys.argv[1:]))
