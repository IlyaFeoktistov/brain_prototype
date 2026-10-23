"""
Эталон блока «устойчивые избирательные состояния и выбор» — Wang 2002
(https://doi.org/10.1016/S0896-6273(02)01092-9), числа из воспроизведения
Brian2 (https://github.com/brian-team/brian2/blob/master/examples/frompapers/Wang_2002.py,
Wimmer & Stimberg). Не мозг —
отдельная сеть на torch: сколько испытаний — столько независимых сетей в
одном тензоре (шаг 1 сборки, DESIGN «Сборка из готовых блоков»).

Сеть: 1600 E (две избирательные группы по 240, остальные 1120 —
неизбирательные), 400 I; все со всеми. LIF: покой −70, порог −50, сброс −55;
E Cm 0.5 нФ, gL 25 нС, рефр. 2 мс; I Cm 0.2 нФ, gL 20 нС, рефр. 1 мс.
Синапсы: AMPA τ 2 мс, NMDA подъём 2 / спад 100 мс, насыщение α 0.5/мс, блок
магнием 1/(1 + e^(−0.062 V)·[Mg]/3.57), Mg 1 мМ, на E и на I; GABA τ 5 мс;
E_rev 0 / −70. Проводимости: фон → E 2.1, → I 1.62 нС (2.4 кГц пуассон);
E → E AMPA 0.05, NMDA 0.165; E → I AMPA 0.04, NMDA 0.13; I → E 1.3; I → I
1.0 нС; внутри группы ×1.7 (Jp), из другой группы ×Jm. Задержка 0.5 мс.
Стимул: группы 1 и 2 — по своему пуассону на клетку μ0(1 ± c/100) + шум σ 4
Гц (смена каждые 50 мс), μ0 40 Гц, с 1000 до 3000 мс; всего 4000 мс.
Шаг — Эйлер 0.1 мс (как у Brian2).

Проверка (как у Wang): фон до стимула (E ~2, I ~8–10 Гц у Wang), выбор —
группа с большей частотой в последние 500 мс стимула, доля верных по
когерентности (у Wang ~50% при 0, ~85% при 12.8, ~100% при 51.2),
удержание — частота победителя после стимула (3500–4000 мс).

    PYTHONPATH=. python tools/block_wang.py [trials=32] [coh=0,3.2,6.4,12.8,25.6,51.2] [seed=1]
"""

from __future__ import annotations

import sys
import time

import torch

DT = 0.1
NE, NI, SUB = 1600, 400, 240
JP = 1.7
JM = 1.0 - 0.15 * (JP - 1.0) / (1.0 - 0.15)
DELAY = 5  # шагов (0.5 мс)


class Net:
    def __init__(self, cohs: torch.Tensor, dev: str):
        self.B, self.dev = len(cohs), dev
        B = self.B
        self.coh = cohs.to(dev)[:, None]
        z = lambda *sh: torch.zeros(*sh, device=dev)
        self.ve = torch.full((B, NE), -52.0, device=dev)
        self.vi = torch.full((B, NI), -52.0, device=dev)
        self.ref_e, self.ref_i = z(B, NE), z(B, NI)
        # Входы одинаковы внутри группы (все со всеми) — по группам [B, 3]
        # для E и [B] для I; фон и стимул — свои у клетки.
        self.ampa_e, self.gaba_e, self.ampa_i, self.gaba_i = z(B, 3), z(B, 3), z(B), z(B)
        self.ext_e, self.ext_i = z(B, NE), z(B, NI)
        self.s_nmda = torch.full((B, NE), 100.0 * 0.01 * 0.2, device=dev)
        self.x_nmda = z(B, NE)
        self.ring_e, self.ring_i = z(DELAY, B, 3), z(DELAY, B)
        self.ring_spk = torch.zeros(DELAY, B, NE, device=dev)
        self.t = 0
        self.mu = z(B, 2)
        lab = torch.full((NE,), 2, dtype=torch.long)
        lab[:SUB], lab[SUB:2 * SUB] = 0, 1
        self.onehot = torch.nn.functional.one_hot(lab, 3).float().to(dev)  # [NE, 3]
        w = torch.ones(3, 3)
        w[0, 0] = w[1, 1] = JP
        w[0, 1] = w[0, 2] = w[1, 0] = w[1, 2] = JM
        self.w_pool = w.to(dev)  # [получатель, источник]
        self.d_ampa, self.d_gaba, self.d_x = [float(torch.exp(torch.tensor(-DT / t))) for t in (2.0, 5.0, 2.0)]
        self.count = z(B, 3)

    def step(self, stim_on: bool) -> None:
        B, dev = self.B, self.dev
        slot = self.t % DELAY
        arr_e, arr_i, arr_spk = self.ring_e[slot], self.ring_i[slot], self.ring_spk[slot]
        # Синапсы: AMPA и GABA от пришедших (задержка 0.5 мс) спайков.
        self.ampa_e.mul_(self.d_ampa).add_(0.05 * arr_e @ self.w_pool.T)
        self.ampa_i.mul_(self.d_ampa).add_(0.04 * arr_e.sum(1))
        self.gaba_e.mul_(self.d_gaba).add_(1.3 * arr_i[:, None])
        self.gaba_i.mul_(self.d_gaba).add_(1.0 * arr_i)
        # Фон 2.4 кГц и стимул — пуассон на клетку.
        self.ext_e.mul_(self.d_ampa).add_(2.1 * torch.poisson(torch.full((B, NE), 2.4 * DT, device=dev)))
        self.ext_i.mul_(self.d_ampa).add_(1.62 * torch.poisson(torch.full((B, NI), 2.4 * DT, device=dev)))
        if stim_on:
            rate = torch.cat([self.mu[:, :1].expand(B, SUB), self.mu[:, 1:].expand(B, SUB)], 1) / 1000.0
            self.ext_e[:, :2 * SUB].add_(2.1 * torch.poisson(rate.clamp(min=0.0) * DT))
        # NMDA: x += 1 на пришедший спайк своей клетки, s с насыщением.
        self.x_nmda.mul_(self.d_x).add_(arr_spk)
        self.s_nmda.add_(DT * (-self.s_nmda / 100.0 + 0.5 * self.x_nmda * (1.0 - self.s_nmda)))
        s_pool = self.s_nmda @ self.onehot  # [B, 3]
        nmda_e = s_pool @ self.w_pool.T     # [B, 3] на получателя
        nmda_i = s_pool.sum(1)
        # Мембраны (нС · мВ = пА; пА / нФ = 10⁻³ мВ/мс).
        ve, vi = self.ve, self.vi
        pool_e = lambda x: x @ self.onehot.T  # [B,3] → [B, NE]
        mg_e = 1.0 / (1.0 + torch.exp(-0.062 * ve) / 3.57)
        mg_i = 1.0 / (1.0 + torch.exp(-0.062 * vi) / 3.57)
        i_e = (pool_e(self.ampa_e) + self.ext_e) * ve + 0.165 * pool_e(nmda_e) * ve * mg_e \
            + pool_e(self.gaba_e) * (ve + 70.0)
        i_i = (self.ampa_i[:, None] + self.ext_i) * vi + 0.13 * nmda_i[:, None] * vi * mg_i \
            + self.gaba_i[:, None] * (vi + 70.0)
        dve = (-25.0 * (ve + 70.0) - i_e) / 0.5 * 1e-3
        dvi = (-20.0 * (vi + 70.0) - i_i) / 0.2 * 1e-3
        free_e, free_i = self.ref_e <= 0, self.ref_i <= 0
        ve = torch.where(free_e, ve + DT * dve, ve)
        vi = torch.where(free_i, vi + DT * dvi, vi)
        spk_e, spk_i = ve > -50.0, vi > -50.0
        self.ve.copy_(torch.where(spk_e, -55.0, ve))
        self.vi.copy_(torch.where(spk_i, -55.0, vi))
        self.ref_e.copy_(torch.where(spk_e, 2.0, self.ref_e - DT))
        self.ref_i.copy_(torch.where(spk_i, 1.0, self.ref_i - DT))
        se = spk_e.float()
        cnt = se @ self.onehot
        self.ring_e[slot] = cnt
        self.ring_i[slot] = spk_i.float().sum(1)
        self.ring_spk[slot] = se
        self.count.add_(cnt)
        self.count_i.add_(spk_i.float().sum(1))
        self.t += 1

    def run(self, ms: float, stim_on: bool) -> tuple[torch.Tensor, torch.Tensor]:
        """Прожить ms; частоты по группам E [B, 3] и I [B], Гц."""

        self.count = torch.zeros(self.B, 3, device=self.dev)
        self.count_i = torch.zeros(self.B, device=self.dev)
        steps = int(round(ms / DT))
        for k in range(steps):
            if stim_on and self.t % int(round(50.0 / DT)) == 0:
                c = self.coh / 100.0
                noise = 4.0 * torch.randn(self.B, 2, device=self.dev)
                self.mu = torch.cat([40.0 + c * 40.0, 40.0 - c * 40.0], 1) + noise
            self.step(stim_on)
        n = torch.tensor([SUB, SUB, NE - 2 * SUB], device=self.dev)
        return (self.count / n / (ms / 1000.0)).cpu(), (self.count_i / NI / (ms / 1000.0)).cpu()


def main(opts: dict) -> None:
    dev = opts.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(opts.get("seed", 1)))
    cohs = [float(c) for c in opts.get("coh", "0,3.2,6.4,12.8,25.6,51.2").split(",")]
    trials = int(opts.get("trials", 32))
    coh = torch.tensor(cohs).repeat_interleave(trials)
    # знак когерентности случайный: верный выбор — группа с большим стимулом
    sign = torch.where(torch.rand(len(coh)) < 0.5, -1.0, 1.0)
    net = Net(coh * sign, dev)
    print(f"Wang 2002: {len(coh)} сетей ({trials} испытаний × когерентности {cohs}), {dev}", flush=True)
    t0 = time.time()
    spont, spont_i = net.run(1000.0, False)
    print(f"  фон (0–1000 мс): E группы {float(spont[:, :2].mean()):.2f}, неизбир. {float(spont[:, 2].mean()):.2f}, "
          f"I {float(spont_i.mean()):.2f} Гц  [{time.time() - t0:.0f} с]", flush=True)
    net.run(1500.0, True)
    late, _ = net.run(500.0, True)
    print(f"  стимул, 2500–3000 мс: победитель {float(late[:, :2].max(1).values.mean()):.1f}, "
          f"проигравший {float(late[:, :2].min(1).values.mean()):.1f} Гц  [{time.time() - t0:.0f} с]", flush=True)
    net.run(500.0, False)
    after, _ = net.run(500.0, False)
    choice = torch.where(late[:, 0] > late[:, 1], 1.0, -1.0)
    correct = choice == sign
    winner = torch.where(choice > 0, 0, 1)
    held = after[torch.arange(len(coh)), winner]
    lost = after[torch.arange(len(coh)), 1 - winner]
    print(f"  после стимула (3500–4000 мс): победитель {float(held.mean()):.1f}, проигравший {float(lost.mean()):.1f} Гц; "
          f"удержали (> 15 Гц) {float((held > 15).float().mean()):.0%}")
    print("  когерентность → доля верных выборов:")
    for c in cohs:
        m = coh == c
        print(f"    {c:5.1f}%  {float(correct[m].float().mean()):.2f}  ({int(m.sum())} испытаний)")
    print(f"всего {time.time() - t0:.0f} с")


if __name__ == "__main__":
    main(dict(kv.split("=", 1) for kv in sys.argv[1:]))
