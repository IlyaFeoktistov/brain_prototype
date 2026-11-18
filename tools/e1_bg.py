"""
E1, шаг 4а: выбор канала записи базальными ганглиями — стриатум D1 → SNr,
спайками. Не мозг — отдельная сеть на torch (как e1_slot/e1_gate).

Канал = позиция k (их P). Клетки:
  стриатум D1 канала k (NMSN клеток): покой −80 мВ — шипиковые клетки в
    покое глубоко гиперполяризованы и стреляют только при сходящемся входе
    коры (down/up-состояния: Wilson & Kawaguchi 1996, J Neurosci,
    https://doi.org/10.1523/JNEUROSCI.16-07-02397.1996 [сверить]). Входы: от
    группы k «текущей позиции» (NSRC клеток) и от клеток всех слов («пришло
    слово»), AMPA τ 2 мс, сила gmsn [подбор: каждый вход один — тихо, оба —
    стреляет];
  SNr канала k (NSNR клеток): сам стреляет ~60 Гц (пейсмейкер; anatomy.py:
    GPi/SNr ~60 Гц) — постоянный ток i_snr [подбор под 60 Гц]; стриатум k →
    SNr k, ГАМК τ 10 мс, сила g_d1 [подбор: стриатум ~20–40 Гц глушит SNr].
Клетка — LIF (0.5 нФ, 25 нС, порог −50, сброс −55, рефр. 2 мс), синапсы —
Эйлер, как в e1_slot. Долги: нет FSI, коллатералей, непрямого пути, GPe,
STN, дофамина (ISSUES 29).

Проверка (калибровка, без памяти): частоты стриатума и SNr канала при
(позиция k активна 25 Гц / нет 2 Гц) × (слово есть 40 Гц / нет 2 Гц).

    PYTHONPATH=tools python tools/e1_bg.py [gmsn=4.5] [g_d1=8] [i_snr=620]
"""

from __future__ import annotations

import sys

import torch

from e1_slot import DT, P

NMSN, NSNR, NSRC, NWORD = 40, 30, 40, 20


class LIF:
    """P каналов × n клеток, пачка B; возбуждение и торможение — проводимости (нС) со следом."""

    def __init__(self, B: int, n: int, dev: str, el: float, i_const: float = 0.0):
        self.B, self.n, self.dev, self.el, self.i_const = B, n, dev, el, i_const
        self.v = torch.full((B, P * n), el, device=dev)
        self.ref = torch.zeros(B, P * n, device=dev)
        self.ge = torch.zeros(B, P * n, device=dev)   # нС, AMPA τ 2
        self.gi = torch.zeros(B, P * n, device=dev)   # нС, ГАМК τ 10
        self.spk = torch.zeros(B, P * n, device=dev)

    def step(self, exc_ns: torch.Tensor, inh_ns: torch.Tensor) -> None:
        """exc_ns, inh_ns [B, P·n] — пришедшая за шаг проводимость (нС), Эйлер."""

        self.ge.mul_(1.0 - DT / 2.0).add_(exc_ns)
        self.gi.mul_(1.0 - DT / 10.0).add_(inh_ns)
        v = self.v
        i = self.ge * v + self.gi * (v + 70.0) - self.i_const
        v = torch.where(self.ref <= 0, v + DT * (-25.0 * (v - self.el) - i) / 0.5 * 1e-3, v)
        s = v > -50.0
        self.v = torch.where(s, -55.0, v)
        self.ref = torch.where(s, round(2.0 / DT) - 1, self.ref - 1)
        self.spk = s.float()

    def per_channel(self) -> torch.Tensor:
        return self.spk.view(self.B, P, self.n).sum(2)   # [B, P]


class BG:
    """Стриатум D1 + SNr, канал на позицию."""

    def __init__(self, B: int, dev: str, gmsn: float = 4.5, g_d1: float = 8.0, i_snr: float = 620.0):
        self.B, self.dev, self.gmsn, self.g_d1 = B, dev, gmsn, g_d1
        self.msn = LIF(B, NMSN, dev, -80.0)
        self.snr = LIF(B, NSNR, dev, -70.0, i_snr)

    def step(self, pos_counts: torch.Tensor, word_counts: torch.Tensor) -> None:
        """pos_counts [B, P] — спайки NSRC клеток каждой группы позиции; word_counts [B] — спайки всех клеток слов."""

        exc = self.gmsn * (pos_counts + word_counts[:, None]).repeat_interleave(NMSN, 1)
        self.msn.step(exc, torch.zeros_like(exc))
        inh = self.g_d1 * self.msn.per_channel().repeat_interleave(NSNR, 1)
        self.snr.step(torch.zeros_like(inh), inh)


def main(opts: dict) -> None:
    torch.set_num_threads(int(opts.get("threads", 2)))
    torch.manual_seed(int(opts.get("seed", 1)))
    B = 4
    bg = BG(B, "cpu", float(opts.get("gmsn", 4.5)), float(opts.get("g_d1", 8.0)), float(opts.get("i_snr", 620.0)))
    # канал 0: позиция активна? слово есть? — 4 сочетания по каналам 0..3, канал 4 — покой
    ph = float(opts.get("pos_hz", 25.0))
    pos_hz = torch.tensor([ph, ph, 2.0, 2.0, 2.0])
    word_on = bool(int(opts.get("word", 1)))
    print(f"gmsn {bg.gmsn} нС, стриатум → SNr {bg.g_d1} нС, ток SNr {bg.snr.i_const} пА")
    for word_hz in (2.0, 40.0):
        m = torch.zeros(B, P)
        s = torch.zeros(B, P)
        steps = int(500 / DT)
        for k in range(steps):
            pc = torch.poisson((pos_hz * NSRC * 1e-3 * DT).expand(B, P).contiguous())
            wc = torch.poisson(torch.full((B,), word_hz * NWORD * 1e-3 * DT) + 4 * 2.0 * NWORD * 1e-3 * DT)
            bg.step(pc, wc)
            if k >= steps // 2:
                m += bg.msn.per_channel()
                s += bg.snr.per_channel()
        m = m / NMSN / 0.25
        s = s / NSNR / 0.25
        print(f"  слово {'есть' if word_hz > 2 else 'нет '}: позиция активна → стриатум {m[:, 0].mean():.1f}, SNr {s[:, 0].mean():.1f} Гц;"
              f" позиция нет → стриатум {m[:, 2].mean():.1f}, SNr {s[:, 2].mean():.1f} Гц")
    del word_on


if __name__ == "__main__":
    main(dict(kv.split("=", 1) for kv in sys.argv[1:]))
