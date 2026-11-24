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


class FSI:
    """
    Быстрые тормозные интернейроны стриатума (PV, FS) — прямое торможение
    клеток стриатума корковым входом (Mallet et al. 2005,
    https://doi.org/10.1523/JNEUROSCI.5027-04.2005: FSI отвечают на кору
    раньше и при меньшей силе, чем клетки стриатума).
    Клетка — тормозная Brunel–Wang (как I в e1_slot: 0.2 нФ, 20 нС, порог
    −50, сброс −55, рефр. 1 мс). Вход — тот же корковый, что у стриатума
    (Humphries et al. 2010, https://doi.org/10.1371/journal.pcbi.1001011:
    «из тех же источников»), широкий: все позиции и все слова, та же сила на
    спайк, что у клеток стриатума [своё: числа коркового входа на FSI нет].
    Выход: на клетку стриатума k FSI (4–27 — Koós & Tepper 1999,
    https://doi.org/10.1038/8138 [DOI сверить]), вероятность связи 0.75
    (Planert et al. 2010, https://doi.org/10.1523/JNEUROSCI.5139-09.2010,
    74% пар) → пул ceil(k / 0.75); пиковая проводимость g_out нС на спайк
    (Gittis et al. 2010, https://doi.org/10.1523/JNEUROSCI.4870-09.2010:
    две группы 0.9 и 16.9 нС, средний ток ~500–580 пА ≈ 7.5 нС), спад ГАМК
    τ 10 мс как у LIF (у них 8 мс). Одна FSI тормозит клетки всех каналов
    (сотни–тысячи клеток на FSI).
    """

    def __init__(self, B: int, dev: str, k: int, g_out: float, gin: float, gen: torch.Generator | None = None):
        import math

        self.B, self.dev, self.g_out, self.gin = B, dev, g_out, gin
        self.n = max(k, math.ceil(k / 0.75))
        self.v = torch.full((B, self.n), -70.0, device=dev)
        self.ref = torch.zeros(B, self.n, device=dev)
        self.ge = torch.zeros(B, self.n, device=dev)
        self.spk = torch.zeros(B, self.n, device=dev)
        # Каждой клетке стриатума (P·NMSN) — k разных FSI из пула.
        gen = gen or torch.Generator().manual_seed(0)
        idx = torch.stack([torch.randperm(self.n, generator=gen)[:k] for _ in range(P * NMSN)])
        self.w = torch.zeros(self.n, P * NMSN, device=dev)
        self.w[idx, torch.arange(P * NMSN)[:, None].expand(-1, k)] = 1.0

    def step(self, cortex_counts: torch.Tensor) -> torch.Tensor:
        """cortex_counts [B] — спайки всего коркового входа за шаг (каждой FSI). Возвращает нС на клетки стриатума [B, P·NMSN]."""

        self.ge.mul_(1.0 - DT / 2.0).add_(self.gin * cortex_counts[:, None])
        v = self.v
        i = self.ge * v
        v = torch.where(self.ref <= 0, v + DT * (-20.0 * (v + 70.0) - i) / 0.2 * 1e-3, v)
        s = v > -50.0
        self.v = torch.where(s, -55.0, v)
        self.ref = torch.where(s, round(1.0 / DT) - 1, self.ref - 1)
        self.spk = s.float()
        return self.g_out * (self.spk @ self.w)


class MSNChannels:
    """
    Стриатум E1 клетками Humphries et al. 2009 (tools/msn_humphries.MSN, без
    дофамина): P каналов × NMSN. Вход — пришедшие корковые события (позиция +
    задача/слова) с авторской силой синапса AMPA+NMDA на событие; тормозного
    входа в E1 нет (у авторов фон — 84 корковых + 84 тормозных).
    """

    # Фон авторов (msn="humphries_bg"): нижняя точка их проверки f-f по данным Moyer — 84 корковых
    # + 84 тормозных пуассоновских входа по 850/168 = 5.06 Гц (850 событий/с → у Moyer 0 Гц, у переноса
    # 0.2 Гц); у каждой клетки свой, сигналы E1 — поверх.
    BG_N, BG_HZ = 84, 850.08 / 168.0

    def __init__(self, B: int, dev: str, background: bool = False):
        from msn_humphries import DT as MDT, MSN

        self.B = B
        self.cells = MSN((B, P * NMSN), dev)
        self.lam = torch.full((B, P * NMSN), self.BG_N * self.BG_HZ * 1e-3 * MDT, device=dev) if background else None

    @property
    def spk(self) -> torch.Tensor:
        return self.cells.spk

    def step_events(self, ctx_events: torch.Tensor) -> None:
        if self.lam is None:
            self.cells.step(ctx_events)
        else:
            self.cells.step(ctx_events + torch.poisson(self.lam), torch.poisson(self.lam))

    def per_channel(self) -> torch.Tensor:
        return self.cells.spk.view(self.B, P, NMSN).sum(2)


class BG:
    """Стриатум D1 + SNr, канал на позицию; по желанию — FSI (fsi_k > 0), клетки
    стриатума Humphries 2009 (msn="humphries") вместо LIF."""

    def __init__(self, B: int, dev: str, gmsn: float = 4.5, g_d1: float = 8.0, i_snr: float = 620.0,
                 fsi_k: int = 0, fsi_g: float = 7.5, fsi_gin: float | None = None, msn: str = "lif"):
        self.B, self.dev, self.gmsn, self.g_d1 = B, dev, gmsn, g_d1
        self.msn_model = msn
        self.msn = (MSNChannels(B, dev, background=(msn == "humphries_bg")) if msn.startswith("humphries")
                    else LIF(B, NMSN, dev, -80.0))
        self.snr = LIF(B, NSNR, dev, -70.0, i_snr)
        self.fsi = FSI(B, dev, fsi_k, fsi_g, gmsn if fsi_gin is None else fsi_gin) if fsi_k > 0 else None

    def step(self, pos_counts: torch.Tensor, word_counts: torch.Tensor) -> None:
        """pos_counts [B, P] — спайки NSRC клеток каждой группы позиции; word_counts [B] — спайки всех клеток слов."""

        events = (pos_counts + word_counts[:, None]).repeat_interleave(NMSN, 1)
        if self.msn_model.startswith("humphries"):
            self.msn.step_events(events)
        else:
            exc = self.gmsn * events
            inh = self.fsi.step(pos_counts.sum(1) + word_counts) if self.fsi is not None else torch.zeros_like(exc)
            self.msn.step(exc, inh)
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
