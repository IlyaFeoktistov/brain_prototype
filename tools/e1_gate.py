"""
E1, шаг 2: ворота записи в ячейку позиции — реле MD под тоническим
торможением SNr (PBWM: Frank, Loughry & O'Reilly 2001,
https://doi.org/10.3758/CABN.1.2.137; SNr → MD моносинаптически тормозит
клетки MD, проецирующиеся в ПФК, — макака, Tanibuchi et al. 2009
[сверить DOI]). Не мозг — отдельная сеть на torch, ячейка — tools/e1_slot.py
(Brunel & Wang 2001).

Устройство:
  слова (кора, учитель включает предъявленное — костыль) → реле MD канала
  позиции → группа слова в ячейке. Канал MD — единица ворот: SNr тормозит
  весь канал разом; внутри канала реле по словам (любое слово словаря
  проходит через канал своей позиции). Ворота открыты = SNr молчит (пауза
  SNr при выборе действия), закрыты = SNr тонически ~60 Гц.

Числа (что из данных, что подбор):
  слово: 20 клеток коры на слово, 40 Гц при предъявлении, иначе 2 Гц
    [подбор: порядок частот ассоциативной коры];
  реле MD: клетка LIF как E Brunel–Wang (0.5 нФ, 25 нС, порог −50), фон
    800 × bg Гц по 2.08 нС (опция bg, по умолчанию 3 — как фон коры) [долг: клетка коры, без T-тока — нет залпа
    отдачи после растормаживания];
  кора → реле: от всех 20 клеток своего слова, gctx нС (опция, по умолчанию 2.08) — ведущий вход
    (driver, Sherman & Guillery) [подбор];
  SNr → реле: 30 входов (как GPi → таламус в anatomy.py, Bodor 2008 —
    крупные окончания), 60 Гц (anatomy.py: GPi/SNr ~60 Гц), ГАМК-А τ 10 мс,
    реверсия −70; сила gsnr — опция [подбор: реле молчит при слове];
  реле → ячейка: группа слова получает от всех 20 реле своего слова, на
    синапс фона (2.08 нС) [подбор: запись нужна от ≥ +200 Гц на клетку].

Условия (по trials ячеек каждое):
  1 закрыто + слово А        → не записано
  2 открыто + слово А        → записано (0.4–0.9 с после)
  3 открыто без слова        → ячейка не загорается
  4 = 2, ворота закрыты дальше → держит А через 2.4 с без канала
  5 = 2, потом слово Б при закрытых → держит А, Б не записано

    PYTHONPATH=. python tools/e1_gate.py [trials=8] [seed=1] [gsnr=2.0] [bg=3] [gctx=2.08] [size=2]
"""

from __future__ import annotations

import sys
import time

import torch

from e1_slot import DT, P, SIZE, Slot

NW = 20          # клеток коры на слово
NR = 20          # реле MD на слово в канале
N_SNR, R_SNR = 30, 60.0


class Relay:
    """Реле MD одного канала: P слов × NR клеток, пачка B."""

    def __init__(self, B: int, dev: str, gsnr: float, bg: float = 3.0, gctx: float = 2.08):
        self.B, self.dev, self.gsnr, self.bg, self.gctx = B, dev, gsnr, bg, gctx
        z = lambda *sh: torch.zeros(*sh, device=dev)
        self.v = torch.full((B, P * NR), -70.0, device=dev)
        self.ref = z(B, P * NR)
        self.ext, self.ctx, self.gaba = z(B, P * NR), z(B, P * NR), z(B, P * NR)
        self.spk = z(B, P * NR)
        self.d2, self.d10 = float(torch.exp(torch.tensor(-DT / 2.0))), float(torch.exp(torch.tensor(-DT / 10.0)))

    def step(self, word_rate: torch.Tensor, snr_on: torch.Tensor) -> None:
        """word_rate [B, P] Гц клеток коры по словам; snr_on [B] — SNr тонически активен."""

        B, dev = self.B, self.dev
        self.ext.mul_(self.d2).add_(torch.poisson(torch.full((B, P * NR), 800 * self.bg * 1e-3 * DT, device=dev)))
        lam = (word_rate * NW * 1e-3 * DT).repeat_interleave(NR, 1)
        self.ctx.mul_(self.d2).add_(torch.poisson(lam))
        lam_snr = (snr_on.float() * N_SNR * R_SNR * 1e-3 * DT)[:, None].expand(B, P * NR)
        self.gaba.mul_(self.d10).add_(torch.poisson(lam_snr))
        v = self.v
        i = (2.08 * self.ext + self.gctx * self.ctx) * v + self.gsnr * self.gaba * (v + 70.0)
        v = torch.where(self.ref <= 0, v + DT * (-25.0 * (v + 70.0) - i) / 0.5 * 1e-3, v)
        s = v > -50.0
        self.v = torch.where(s, -55.0, v)
        self.ref = torch.where(s, 2.0, self.ref - DT)
        self.spk = s.float()

    def to_slot(self, lab: torch.Tensor) -> torch.Tensor:
        """Спайки реле → пришедшие на клетки ячейки [B, ne]: группа слова k — от всех NR реле слова k."""

        per_word = self.spk.view(self.B, P, NR).sum(2)          # [B, P]
        per_word = torch.cat([per_word, torch.zeros(self.B, 1, device=self.dev)], 1)
        return per_word[:, lab]


def main(opts: dict) -> None:
    dev = opts.get("device", "cpu")
    torch.set_num_threads(int(opts.get("threads", 2)))
    torch.manual_seed(int(opts.get("seed", 1)))
    T = int(opts.get("trials", 8))
    gsnr = float(opts.get("gsnr", 2.0))
    names = ["1 закрыто + А", "2/4 открыто + А", "3 открыто без слова", "5 открыто + А, потом Б закрыто"]
    C = len(names)
    B = C * T
    cond = torch.arange(C).repeat_interleave(T)
    bg = float(opts.get("bg", 3.0))
    slot, md = Slot(B, dev, size=float(opts.get("size", SIZE))), Relay(B, dev, gsnr, bg, float(opts.get("gctx", 2.08)))
    A, Bw = 0, 1
    # (от мс, до мс): ворота открыты? слово (−1 нет) — по условию
    open_ev = torch.tensor([False, True, True, True])[cond]
    word_ev = torch.tensor([A, A, -1, A])[cond]
    word_dis = torch.tensor([-1, -1, -1, Bw])[cond]
    rates = {}
    win = {"фон": (300, 500), "событие": (500, 600), "после": (1000, 1500), "отвлекатель": (1500, 1600),
           "долго": (2500, 3000)}
    cnt = {k: torch.zeros(B, P + 1) for k in win}
    md_cnt = {k: torch.zeros(B, P) for k in win}
    t0 = time.time()
    steps = int(3000 / DT)
    n = torch.tensor([slot.nsub] * P + [slot.nnon], dtype=torch.float)
    for k in range(steps):
        t = k * DT
        wr = torch.full((B, P), 2.0)
        snr = torch.ones(B, dtype=torch.bool)
        if 500 <= t < 600:
            snr = ~open_ev
            m = word_ev >= 0
            wr[m, word_ev[m]] = 40.0
        if 1500 <= t < 1600:
            m = word_dis >= 0
            wr[m, word_dis[m]] = 40.0
        md.step(wr.to(dev), snr.to(dev))
        slot.step(add_e=md.to_slot(slot.lab))
        for key, (a, b) in win.items():
            if a <= t < b:
                cnt[key] += (slot.spk_e @ slot.onehot).cpu()
                md_cnt[key] += md.spk.view(B, P, NR).sum(2).cpu()
    for key, (a, b) in win.items():
        rates[key] = cnt[key] / n / ((b - a) / 1000.0)
        md_cnt[key] = md_cnt[key] / NR / ((b - a) / 1000.0)
    print(f"Ворота MD под SNr: gsnr {gsnr} нС, фон реле 800 × {bg} Гц, кора → реле {md.gctx} нС, {T} ячеек на условие, {time.time() - t0:.0f} с")
    th = 10.0
    for c, name in enumerate(names):
        m = cond == c
        ev, af, lo = rates["событие"][m], rates["после"][m], rates["долго"][m]
        print(f"  {name}:")
        print(f"    реле MD в событие: слово А {md_cnt['событие'][m, A].mean():.1f}, прочие {md_cnt['событие'][m, 1:].mean():.1f} Гц;"
              f" в отвлекатель: Б {md_cnt['отвлекатель'][m, Bw].mean():.1f} Гц")
        print(f"    ячейка: фон макс {rates['фон'][m, :P].max(1).values.mean():.1f}; после (0.4–0.9 с) А {af[:, A].mean():.1f},"
              f" Б {af[:, Bw].mean():.1f}, прочие {af[:, 2:P].mean():.1f}; через 2.4 с А {lo[:, A].mean():.1f}, Б {lo[:, Bw].mean():.1f} Гц")
        holdA = lambda r: int(((r[:, A] > th) & (r[:, 1:P].max(1).values < th)).sum())
        none = lambda r: int((r[:, :P].max(1).values < th).sum())
        if c == 0:
            print(f"    → не записано: {none(af)}/{T}")
        elif c == 1:
            print(f"    → записано: {holdA(af)}/{T}; держит через 2.4 с при закрытых: {holdA(lo)}/{T}")
        elif c == 2:
            print(f"    → не загорелось: {none(af)}/{T} (и через 2.4 с {none(lo)}/{T})")
        else:
            print(f"    → записано: {holdA(af)}/{T}; после Б держит А, Б нет: {holdA(lo)}/{T}")


if __name__ == "__main__":
    main(dict(kv.split("=", 1) for kv in sys.argv[1:]))
