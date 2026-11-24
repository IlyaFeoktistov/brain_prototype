"""
Шаг 4а (docs/CHAT_PLAN.md): части E1 по отдельности на входе со статистикой
коры. Контроль перед звеном FSI.

Вход слова — суммарные спайки популяции слова, пуассон (форма вспышки во
времени не учтена: в коре ~100 мс, здесь ровно 200 мс):
  e1        — испытание E1: 800/с на слово, 40/с в тишине (20 × 40 / 2 Гц);
  cortex20  — первые 20 клеток коры: 125 / 27 (tools/word_drive.py, зерно 1);
  cortex162 — клетки избирательностью > 1 sd: 909 / 364.
Время: 400 мс тишины, 200 мс слово А, 400 мс тишины.

Меры:
  стриатум записи (tools/e1_bg.BG; вход — позиция + сумма всех слов):
    канал активной позиции (25 Гц × 40 клеток) — тишина / слово, Гц;
    канал неактивной позиции (2 Гц) — слово;
  реле MD (tools/e1_gate.Relay, ворота открыты всё время): реле слова А и
    реле чужих слов — тишина / слово, Гц.
Хорошо: стриатум активной позиции молчит в тишине и стреляет на слово;
реле А стреляет на слово, чужие реле молчат.

    PYTHONPATH=tools .venv/bin/python tools/e1_input_check.py [trials=16] [seed=1] [modes=e1,cortex20,cortex162]
"""

from __future__ import annotations

import sys

import torch

from e1_bg import BG, NMSN, NSRC
from e1_gate import NR, Relay
from e1_slot import DT, P

MODES = {"e1": (800.0, 40.0), "cortex20": (125.0, 27.0), "cortex162": (909.0, 364.0)}
SIL, WORD = 400.0, 200.0


def run(mode: str, B: int, fsi_k: int = 0, fsi_g: float = 7.5, fsi_gin: float | None = None,
        total_ms: float | None = None) -> dict:
    on, off = MODES[mode]
    bg = BG(B, "cpu", fsi_k=fsi_k, fsi_g=fsi_g, fsi_gin=fsi_gin)
    md = Relay(B, "cpu", 16.0, 1.0, 4.0)
    pos_hz = torch.tensor([25.0, 2.0, 2.0, 2.0, 2.0])
    total = total_ms or SIL + WORD + SIL
    acc = {k: torch.zeros(B, P) for k in ("msn_sil", "msn_word", "rel_sil", "rel_word")}
    acc["fsi_sil"], acc["fsi_word"] = torch.zeros(B, 1), torch.zeros(B, 1)
    for k in range(int(total / DT)):
        t = k * DT
        word = SIL <= t < SIL + WORD
        rate = torch.full((B, P), off)
        if word:
            rate[:, 0] = on                                       # слово А
        wc = torch.poisson(rate * 1e-3 * DT)                      # [B, P] спайки популяций слов за шаг
        pc = torch.poisson((pos_hz * NSRC * 1e-3 * DT).expand(B, P).contiguous())
        bg.step(pc, wc.sum(1))
        md.step_counts(wc.repeat_interleave(NR, 1), torch.zeros(B, dtype=torch.bool))
        msn = bg.msn.per_channel() / NMSN
        rel = md.spk.view(B, P, NR).mean(2)
        fs = bg.fsi.spk.mean(1, keepdim=True) if bg.fsi is not None else torch.zeros(B, 1)
        if word:
            acc["fsi_word"] += fs
            acc["msn_word"] += msn
            acc["rel_word"] += rel
        elif t >= 100.0:                                          # первые 100 мс — разгон
            acc["fsi_sil"] += fs
            acc["msn_sil"] += msn
            acc["rel_sil"] += rel
    n_sil = (2 * SIL - 100.0) / DT
    n_word = WORD / DT
    hz = lambda x, n: (x / n * 1000.0 / DT).mean(0)
    return {"msn_sil": hz(acc["msn_sil"], n_sil), "msn_word": hz(acc["msn_word"], n_word),
            "rel_sil": hz(acc["rel_sil"], n_sil), "rel_word": hz(acc["rel_word"], n_word),
            "fsi_sil": hz(acc["fsi_sil"], n_sil), "fsi_word": hz(acc["fsi_word"], n_word)}


def main(opts: dict) -> None:
    torch.set_num_threads(2)
    torch.manual_seed(int(opts.get("seed", 1)))
    B = int(opts.get("trials", 16))
    fsi_ks = [int(x) for x in opts.get("fsi_k", "0").split(",")]
    fsi_gs = [float(x) for x in opts.get("fsi_g", "7.5").split(",")]
    for mode in opts.get("modes", "e1,cortex20,cortex162").split(","):
      for fk in fsi_ks:
       for fg in (fsi_gs if fk else [0.0]):
        r = run(mode, B, fk, fg)
        on, off = MODES[mode]
        tag = f"FSI {fk} на клетку, {fg} нС" if fk else "без FSI"
        print(f"{mode:10s} (слово {on:.0f}/с, тишина {off:.0f}/с), {tag}; FSI тишина {float(r['fsi_sil'][0]):.0f}, слово {float(r['fsi_word'][0]):.0f} Гц:")
        print(f"   стриатум, активная позиция: тишина {float(r['msn_sil'][0]):6.1f} Гц, слово {float(r['msn_word'][0]):6.1f} Гц;"
              f" неактивная позиция на слове {float(r['msn_word'][1:].mean()):6.1f} Гц")
        print(f"   реле MD (ворота открыты): слово А — тишина {float(r['rel_sil'][0]):6.1f}, слово {float(r['rel_word'][0]):6.1f} Гц;"
              f" чужие слова — тишина {float(r['rel_sil'][1:].mean()):6.1f}, во время А {float(r['rel_word'][1:].mean()):6.1f} Гц",
              flush=True)


if __name__ == "__main__":
    main(dict(kv.split("=", 1) for kv in sys.argv[1:]))
