"""
Клетка стриатума (шипиковая, MSN) по Humphries, Lepora, Wood & Gurney 2009,
Front Comput Neurosci 3:26, https://doi.org/10.3389/neuro.10.026.2009 —
Izhikevich, подогнанная авторами к подробной модели MSN Moyer et al. 2007;
код авторов — ModelDB 128818 (MATLAB), числа — их файл
fit_model_results_NEWtuning.mat, схема шага — test_Moyer_model_fit.m
(Эйлер, dt 0.1 мс, без дофамина — «unmodified»).

  C dv/dt = k (v − vr)(v − vt) − u + I_syn,  du/dt = a (b (v − vr) − u)
  v ≥ vpeak → v = c, u += d
  k 1, a 0.01, b −20, c −55, vr −80, vpeak 40; C 15.229, vt −29.730, d 90.910
Синапсы (на каждое пришедшее событие G += PSP / τ, затем G·exp(−dt/τ)):
  корковый вход — AMPA и NMDA одновременно: PSP_AMPA 6.869 (Xsyn), PSP_NMDA =
  PSP_AMPA / 2, τ 6 и 160 мс, E 0; NMDA с магниевым блоком Moyer:
  B(v) = 1 / (1 + Mg/3.57 · exp(−0.062 v)), Mg 1;
  тормозной — PSP_GABA = PSP_AMPA / 1.4, τ 4 мс, E −60.
Авторы: клетка сама не бистабильна; двугорбость потенциала (down/up) даёт
вход через NMDA.

Проверки авторов (по данным Moyer, «unmod» в test_Moyer_model_fit.m):
  f-I: ток 220…300 пА → 0…16 Гц;
  f-f: 84 корковых + 84 тормозных пуассоновских входа по r Гц
       (событий/с = 168·r, 850…1350) → 0…10 Гц.

    PYTHONPATH=tools .venv/bin/python tools/msn_humphries.py
"""

from __future__ import annotations

import math
import sys

import torch

K, A, BB, CC, VR, VPEAK = 1.0, 0.01, -20.0, -55.0, -80.0, 40.0
CAP, VT, DD = 15.22942, -29.73032, 90.90962
PSP_AMPA = 6.86875
PSP_NMDA, PSP_GABA = PSP_AMPA / 2.0, PSP_AMPA / 1.4
TS_AMPA, TS_NMDA, TS_GABA = 6.0, 160.0, 4.0
E_AMPA, E_NMDA, E_GABA, MG = 0.0, 0.0, -60.0, 1.0
DT = 0.1

# Данные Moyer et al. 2007 (как в test_Moyer_model_fit.m, без дофамина)
FI_PA = [220, 225, 230, 235, 240, 245, 250, 255, 260, 265, 270, 275, 280, 285, 290, 295, 300]
FI_HZ = [0, 0, 0, 2, 4, 4, 6, 6, 8, 8, 10, 10, 12, 12, 14, 14, 16]
FF_EV = [850.08, 900.144, 950.208, 1000.272, 1050.336, 1100.4, 1150.464, 1200.528, 1250.592, 1300.656, 1350.72]
FF_HZ = [0, 0.7778, 1.4444, 2.5556, 3.0, 5.0, 5.3333, 7.5556, 7.8889, 9.5556, 10.2222]


class MSN:
    """Пачка клеток [B, n]; входы — числа пришедших событий за шаг (корковые, тормозные) и ток (пА)."""

    def __init__(self, shape: tuple, dev: str = "cpu"):
        self.v = torch.full(shape, VR, device=dev)
        self.u = torch.zeros(shape, device=dev)
        self.ga = torch.zeros(shape, device=dev)
        self.gn = torch.zeros(shape, device=dev)
        self.gg = torch.zeros(shape, device=dev)
        self.spk = torch.zeros(shape, device=dev)
        self.ea, self.en, self.eg = math.exp(-DT / TS_AMPA), math.exp(-DT / TS_NMDA), math.exp(-DT / TS_GABA)

    def step(self, ctx_events: torch.Tensor | float = 0.0, gaba_events: torch.Tensor | float = 0.0,
             i_pa: torch.Tensor | float = 0.0) -> None:
        self.ga = (self.ga + PSP_AMPA * ctx_events / TS_AMPA) * self.ea
        self.gn = (self.gn + PSP_NMDA * ctx_events / TS_NMDA) * self.en
        self.gg = (self.gg + PSP_GABA * gaba_events / TS_GABA) * self.eg
        v, u = self.v, self.u
        bn = 1.0 / (1.0 + (MG / 3.57) * torch.exp(-v * 0.062))
        i_syn = self.ga * (E_AMPA - v) + bn * self.gn * (E_NMDA - v) + self.gg * (E_GABA - v)
        v2 = v + DT * (K * (v - VR) * (v - VT) - u + i_syn + i_pa) / CAP
        u2 = u + DT * A * (BB * (v - VR) - u)
        s = v2 >= VPEAK
        self.v = torch.where(s, torch.full_like(v2, CC), v2)
        self.u = torch.where(s, u2 + DD, u2)
        self.spk = s.float()


def f_i(trials: int = 1) -> list[float]:
    """Ток → частота, как у авторов: 5 с, счёт с 1 до 5 с."""

    cells = MSN((len(FI_PA),))
    i = torch.tensor(FI_PA, dtype=torch.float)
    n = 0
    cnt = torch.zeros(len(FI_PA))
    for k in range(int(5000 / DT)):
        cells.step(0.0, 0.0, i)
        if k * DT >= 1000:
            cnt += cells.spk
    return (cnt / 4.0).tolist()


def f_f(trials: int = 8, seed: int = 1) -> list[float]:
    """Событий/с → частота: 84 корковых + 84 тормозных входа по r = событий/168 Гц."""

    g = torch.Generator().manual_seed(seed)
    r = torch.tensor(FF_EV) / 168.0
    cells = MSN((trials, len(FF_EV)))
    lam = (84 * r * 1e-3 * DT).expand(trials, -1)
    cnt = torch.zeros(trials, len(FF_EV))
    for k in range(int(5000 / DT)):
        cells.step(torch.poisson(lam, generator=g), torch.poisson(lam, generator=g))
        if k * DT >= 1000:
            cnt += cells.spk
    return (cnt / 4.0).mean(0).tolist()


def main(opts: dict) -> None:
    torch.set_num_threads(2)
    fi = f_i()
    print("f-I (ток пА → Гц): модель / Moyer")
    print("  " + "  ".join(f"{p}:{m:.0f}/{d}" for p, m, d in zip(FI_PA, fi, FI_HZ)))
    ff = f_f(int(opts.get("trials", 8)))
    print("f-f (событий/с → Гц): модель / Moyer")
    print("  " + "  ".join(f"{e:.0f}:{m:.1f}/{d:.1f}" for e, m, d in zip(FF_EV, ff, FF_HZ)))


if __name__ == "__main__":
    main(dict(kv.split("=", 1) for kv in sys.argv[1:]))
