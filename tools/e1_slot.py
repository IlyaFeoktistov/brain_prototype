"""
E1, шаг 1: ячейка рабочей памяти «одна позиция — какое слово» — эталон
Brunel & Wang 2001 («Effects of neuromodulation in a cortical network model
of object working memory dominated by recurrent inhibition», J Comput
Neurosci 11:63, https://doi.org/10.1023/A:1011204814320; модель, расчёт —
по коду примера Brian2, https://github.com/brian-team/brian2/blob/master/examples/frompapers/Brunel_Wang_2001.py).
Не мозг — отдельная сеть на torch, испытания пачкой (как block_wang).

Почему этот блок: у Wang 2002 (сеть выбора, w+ 1.7) избранная группа после
стимула гаснет за ~2 с, а при w+ 1.9 с долей 0.15 сеть зажигается сама
(tools/block_wang.py, 2026-11-18) — окна памяти нет. Brunel & Wang 2001 —
та же модель клеток и синапсов, но настроена на память объекта: 5
избирательных групп по 10% E, w+ 2.1, торможение ГАМК 10 мс. Группа = слово.

Сеть (числа авторов): 800 E, 200 I, все со всеми (без i = j). LIF: покой
−70, порог −50, сброс −55; E 0.5 нФ / 25 нС / рефр. 2 мс; I 0.2 нФ / 20 нС
/ 1 мс. Фон: 800 входов × 3 Гц, AMPA → E 2.08, → I 1.62 нС. Внутри сети:
AMPA → E 0.104, → I 0.081; NMDA → E 0.327, → I 0.258 (подъём 2, спад 100 мс,
α 0.5/мс, Mg 1 мМ); ГАМК → E 1.25, → I 0.973 нС, τ 10 мс. Веса: в
избирательную группу от своей w+ = 2.1, от остальных (и неизбирательных)
w− = 1 − f(w+ − 1)/(1 − f); в неизбирательные и в I — 1.
Стимул «предъявить слово»: 80 входов × 25 Гц на группу 50 мс. Сброс: 800
входов × 25 Гц на все E и I 50 мс. Эйлер 0.1 мс (как Brian2 у авторов).

Контроль авторов (рис. 6b): фон низкий; слово 1 в 1.0 с — его группа держит
до следующего; слово 2 в 2.0 с — заменяет; сброс в 3.0 с — все к фону.

    PYTHONPATH=. python tools/e1_slot.py [trials=8] [seed=1] [size=1] [device=cpu]
"""

from __future__ import annotations

import sys
import time

import torch

DT = 0.1
NE, NI = 800, 200
F, P = 0.1, 5
NSUB = int(NE * F)                 # 80 клеток на слово
NNON = NE - P * NSUB               # 400 неизбирательных
WPLUS = 2.1
G = P + 1                          # группы E: 0..P−1 — слова, P — неизбирательные
C_EXT, R_EXT = 800, 3.0            # фон: входов × Гц
C_SEL, R_SEL = int(F * C_EXT), 25.0  # стимул слова
R_RESET = 25.0                     # сброс: C_EXT входов × 25 Гц


class Slot:
    """Пачка из B независимых ячеек; группы E пулятся (все со всеми)."""

    def __init__(self, B: int, dev: str, wplus: float = WPLUS, size: float = 1.0):
        # size — во сколько раз сеть больше примера Brian2 (800 E / 200 I);
        # внутренние проводимости ÷ size — как в коде примера (g × 800 / N_E).
        self.B, self.dev = B, dev
        self.ne, self.ni = int(NE * size), int(NI * size)
        self.nsub = int(self.ne * F)
        self.nnon = self.ne - P * self.nsub
        self.k = 1.0 / size
        z = lambda *sh: torch.zeros(*sh, device=dev)
        self.ve, self.vi = torch.full((B, self.ne), -70.0, device=dev), torch.full((B, self.ni), -70.0, device=dev)
        self.ref_e, self.ref_i = z(B, self.ne), z(B, self.ni)
        self.ext_e, self.ext_i = z(B, self.ne), z(B, self.ni)     # s_AMPA_ext (безразм.)
        self.ampa_e, self.ampa_i = z(B, G), z(B)          # s_AMPA по группе-получателю
        self.gaba_e, self.gaba_i = z(B), z(B)
        self.s_nmda, self.x_nmda = z(B, self.ne), z(B, self.ne)
        lab = torch.full((self.ne,), P, dtype=torch.long)
        for k in range(P):
            lab[k * self.nsub:(k + 1) * self.nsub] = k
        self.lab = lab.to(dev)
        self.onehot = torch.nn.functional.one_hot(lab, G).float().to(dev)  # [self.ne, G]
        wminus = 1.0 - F * (wplus - 1.0) / (1.0 - F)
        w = torch.ones(G, G)
        w[:P, :] = wminus
        for k in range(P):
            w[k, k] = wplus
        self.w = w.to(dev)  # [получатель, источник]
        self.decay = {t: float(torch.exp(torch.tensor(-DT / t))) for t in (2.0, 10.0)}
        self.spk_e = z(B, self.ne)
        self.spk_i = z(B, self.ni)
        self.count = z(B, G)

    def step(self, drive_e: torch.Tensor | None = None, drive_i: torch.Tensor | None = None,
             add_e: torch.Tensor | None = None) -> None:
        """drive_e [B, self.ne], drive_i [B, self.ni] — доп. пуассон (кГц входов) сверх фона;
        add_e [B, self.ne] — пришедшие спайки внешних путей на том же синапсе, что и фон (2.08 нС)."""

        B, dev = self.B, self.dev
        cnt = self.spk_e @ self.onehot          # спайки прошлого шага по группам
        n_i = self.spk_i.sum(1)
        d2, d10 = self.decay[2.0], self.decay[10.0]
        self.ampa_e.mul_(d2).add_(cnt @ self.w.T)
        self.ampa_i.mul_(d2).add_(cnt.sum(1))
        self.gaba_e.mul_(d10).add_(n_i)
        self.gaba_i.mul_(d10).add_(n_i)
        lam_e = torch.full((B, self.ne), C_EXT * R_EXT * 1e-3 * DT, device=dev)
        lam_i = torch.full((B, self.ni), C_EXT * R_EXT * 1e-3 * DT, device=dev)
        if drive_e is not None:
            lam_e = lam_e + drive_e * DT
        if drive_i is not None:
            lam_i = lam_i + drive_i * DT
        self.ext_e.mul_(d2).add_(torch.poisson(lam_e))
        self.ext_i.mul_(d2).add_(torch.poisson(lam_i))
        if add_e is not None:
            self.ext_e.add_(add_e)
        # NMDA у пресинаптической клетки: x += 1 на спайк, s с насыщением (Эйлер).
        self.x_nmda.mul_(d2).add_(self.spk_e)
        self.s_nmda.add_(DT * (-self.s_nmda / 100.0 + 0.5 * self.x_nmda * (1.0 - self.s_nmda)))
        s_pool = self.s_nmda @ self.onehot
        nmda_e = s_pool @ self.w.T                # [B, G]
        nmda_i = s_pool.sum(1)
        ve, vi = self.ve, self.vi
        pool = lambda x: x[:, self.lab]
        mg_e = 1.0 / (1.0 + torch.exp(-0.062 * ve) / 3.57)
        mg_i = 1.0 / (1.0 + torch.exp(-0.062 * vi) / 3.57)
        # нС · мВ = пА
        i_e = (2.08 * self.ext_e + 0.104 * self.k * pool(self.ampa_e)) * ve + 0.327 * self.k * pool(nmda_e) * mg_e * ve \
            + 1.25 * self.k * self.gaba_e[:, None] * (ve + 70.0)
        i_i = (1.62 * self.ext_i + 0.081 * self.k * self.ampa_i[:, None]) * vi + 0.258 * self.k * nmda_i[:, None] * mg_i * vi \
            + 0.973 * self.k * self.gaba_i[:, None] * (vi + 70.0)
        dve = (-25.0 * (ve + 70.0) - i_e) / 0.5 * 1e-3
        dvi = (-20.0 * (vi + 70.0) - i_i) / 0.2 * 1e-3
        ve = torch.where(self.ref_e <= 0, ve + DT * dve, ve)
        vi = torch.where(self.ref_i <= 0, vi + DT * dvi, vi)
        se, si = ve > -50.0, vi > -50.0
        self.ve = torch.where(se, -55.0, ve)
        self.vi = torch.where(si, -55.0, vi)
        self.ref_e = torch.where(se, 2.0, self.ref_e - DT)
        self.ref_i = torch.where(si, 1.0, self.ref_i - DT)
        self.spk_e, self.spk_i = se.float(), si.float()
        self.count.add_(self.spk_e @ self.onehot)

    def run(self, ms: float, word: torch.Tensor | None = None, reset: bool = False) -> torch.Tensor:
        """Прожить ms; word [B] — номер слова (−1 — нет). Частоты групп [B, G], Гц."""

        drive_e = drive_i = None
        if word is not None:
            on = (self.lab[None, :] == word.to(self.dev)[:, None]).float()
            drive_e = on * (C_SEL * R_SEL * 1e-3)
        if reset:
            drive_e = torch.full((self.B, self.ne), C_EXT * R_RESET * 1e-3, device=self.dev)
            drive_i = torch.full((self.B, self.ni), C_EXT * R_RESET * 1e-3, device=self.dev)
        self.count.zero_()
        for _ in range(int(round(ms / DT))):
            self.step(drive_e, drive_i)
        n = torch.tensor([self.nsub] * P + [self.nnon], device=self.dev, dtype=torch.float)
        return (self.count / n / (ms / 1000.0)).cpu()


def main(opts: dict) -> None:
    dev = opts.get("device", "cpu")
    torch.set_num_threads(int(opts.get("threads", 2)))
    torch.manual_seed(int(opts.get("seed", 1)))
    B = int(opts.get("trials", 8))
    slot = Slot(B, dev, float(opts.get("wplus", WPLUS)), float(opts.get("size", 1.0)))
    w1, w2 = torch.zeros(B, dtype=torch.long), torch.ones(B, dtype=torch.long)
    t0 = time.time()
    # (метка, длительность мс, слово, сброс)
    plan = [("фон", 1000, None, False), ("слово 1", 50, w1, False), ("держит 1", 950, None, False),
            ("слово 2", 50, w2, False), ("держит 2", 950, None, False), ("сброс", 50, None, True),
            ("после сброса", 950, None, False)]
    print(f"Brunel & Wang 2001, {B} ячеек, {slot.ne} E / {slot.ni} I, w+ {slot.w[0, 0]:.2f}, {dev}")
    rows = {}
    for name, ms, word, reset in plan:
        if ms > 100:  # окна по 250 мс, кроме первых 200 мс после события
            slot.run(200.0, word, reset)
            r = slot.run(ms - 200.0, word, reset)
        else:
            r = slot.run(ms, word, reset)
        rows[name] = r
        g = r.mean(0)
        print(f"  {name:13s} слова {' '.join(f'{float(x):5.1f}' for x in g[:P])} | неизб {float(g[P]):4.1f} Гц"
              f"   [{time.time() - t0:.0f} с]", flush=True)
    th = 10.0
    bg = rows["фон"][:, :P]
    h1, h2, rs = rows["держит 1"], rows["держит 2"], rows["после сброса"]
    print("Проверки (порог «держит» 10 Гц, окно — 200–950 мс после события):")
    print(f"  фон без зажигания: {int((bg.max(1).values < th).sum())}/{B}")
    print(f"  слово 1 держится, другие нет: {int(((h1[:, 0] > th) & (h1[:, 1:P].max(1).values < th)).sum())}/{B}")
    print(f"  слово 2 заменило 1: {int(((h2[:, 1] > th) & (h2[:, 0] < th)).sum())}/{B}")
    print(f"  сброс до фона: {int((rs[:, :P].max(1).values < th).sum())}/{B}")
    print(f"всего {time.time() - t0:.0f} с")


if __name__ == "__main__":
    main(dict(kv.split("=", 1) for kv in sys.argv[1:]))
