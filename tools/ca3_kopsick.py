"""
Эталон CA3 Kopsick et al. 2024 на torch — их сеть целиком, как в CARLsim6
feat/ca3net (контроль переноса; запускать их код нельзя: setupNetwork на
~10⁸ синапсов PC → PC съел всю ОЗУ, 2026-11-19).

Сеть — data/anatomy/hippocampus/ca3_kopsick.json (tools/ca3_kopsick_extract.py из
кода авторов). Счёт повторяет ядро CARLsim6 ветки feat/ca3net с флагами
авторов (JK_CA3_SNN, CSTP_DOUBLES; snn_gpu_module.cu, snn_manager.cpp):

  за 1 мс по порядку (SNN::advSimStep):
  1. спад STP каждой связи (Эйлер за 1 мс): u ×(1 − 1/τ_u), x += (1 − x)/τ_x;
     спад проводимости синапса ×(1 − 1/τ_d) (у каждой связи свой τ_d);
  2. выстрелившие на прошлом шаге — спайки этого шага (findFiring);
  3. доставка (задержка 1 мс): u⁺ = u + U(1 − u⁻), x⁺ = x − u⁺x⁻,
     Δg = вес × g × u⁺x⁻/U (u⁻, x⁻ — до спада этого шага, как в ядре);
  4. фон: каждой клетке lognormal(μ, σ своего типа) × U(0, 1) пА, ≤ 10 000;
  5. мембрана: 5 подшагов RK4 по 0.2 мс, Izhikevich 2007 (9 параметров),
     ток I = фон − g_AMPA·(v − 0) − g_GABAa·(v + 70) — по v начала подшага;
     v > vpeak → спайк, v = c, u += d; спайк на последнем подшаге —
     рефрактерность 1 мс (v держится c, u не меняется); v ≥ −90.

Отличие от ядра: проводимость у CARLsim хранится на каждом синапсе и при
спайке ЗАМЕНЯЕТСЯ новым приростом (setAMPASynGValue), здесь — сумма по
связи у получателя (прирост добавляется). Разница — остаток прошлого спайка
той же пары (e^(−ISI/τ_d), при ISI ≥ 50 мс и τ_d ≤ 9 мс < 0.4%).
STDP PC → PC (setESTDP у авторов) в покое не включаем: веса — начальные.

    PYTHONPATH=. .venv/bin/python tools/ca3_kopsick.py [ms=2000] [seed=1] [skip=100] [scale=1.0]

scale < 1 — меньше клеток с той же вероятностью связи (только для отладки
скрипта: меньше входов на клетку — другая динамика). degree=1 — вероятность
÷ scale: у клетки столько же входов, сколько у авторов (проверка, держит ли
покой CA3 размером с наш).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

NET = Path(__file__).resolve().parent.parent / "data" / "anatomy" / "hippocampus" / "ca3_kopsick.json"

# Частоты модели авторов в покое (Kopsick et al. 2024, табл. 3, «Model
# Firing Rate», Гц, среднее ± s.d.).
PAPER_HZ = {"CA3_Pyramidal": (0.42, 0.63), "CA3_Axo_Axonic": (19.17, 1.43), "CA3_Basket": (8.2, 3.22),
            "CA3_BC_CCK": (0.49, 0.65), "CA3_Bistratified": (14.08, 1.32), "CA3_Ivy": (1.40, 1.03),
            "CA3_MFA_ORDEN": (1.04, 1.02), "CA3_QuadD_LM": (5.70, 0.95)}

SUBSTEPS = 5
E_AMPA, E_GABAA = 0.0, -70.0


class CA3:
    def __init__(self, net: dict, seed: int, scale: float, device: torch.device, degree: bool = False):
        self.device = device
        gen = torch.Generator(device=device).manual_seed(seed)
        self.gen = gen
        groups = net["groups"]
        self.names = list(groups)
        sizes = [max(1, round(groups[n]["cells"] * scale)) for n in self.names]
        self.start = dict(zip(self.names, torch.tensor([0] + sizes[:-1]).cumsum(0).tolist()))
        self.size = dict(zip(self.names, sizes))
        n = self.n = sum(sizes)

        def per_cell(key):
            return torch.cat([torch.full((self.size[g],), float(groups[g][key])) for g in self.names]).to(device)

        self.C, self.k, self.vr, self.vt = per_cell("C"), per_cell("k"), per_cell("vr"), per_cell("vt")
        self.a, self.b, self.vpeak, self.c, self.d = per_cell("a"), per_cell("b"), per_cell("vpeak"), per_cell("c"), per_cell("d")
        self.ref = per_cell("refractory_ms").long()
        self.mu, self.sigma = per_cell("background_mu"), per_cell("background_sigma")
        self.cap = float(net["background_cap_pa"])
        self.type_of = torch.cat([torch.full((self.size[g],), i, dtype=torch.long) for i, g in enumerate(self.names)]).to(device)

        # Слоты STP: по связи и пресинаптической клетке; слоты проводимости:
        # по связи и получателю.
        links = net["links"]
        stp_pre, stp_U, stp_iu, stp_ix, stp_scale = [], [], [], [], []
        g_cell, g_decay, g_exc = [], [], []
        stp_off, g_off = [], []
        s_total = g_total = 0
        for link in links:
            pre, post = link["pre"], link["post"]
            npre, npost = self.size[pre], self.size[post]
            stp_off.append(s_total)
            g_off.append(g_total)
            stp_pre.append(torch.arange(npre) + self.start[pre])
            stp_U.append(torch.full((npre,), link["U"]))
            stp_iu.append(torch.full((npre,), 1.0 / link["tau_u_ms"]))
            stp_ix.append(torch.full((npre,), 1.0 / link["tau_x_ms"]))
            stp_scale.append(torch.full((npre,), link["weight"] * link["g"] / link["U"]))
            g_cell.append(torch.arange(npost) + self.start[post])
            g_decay.append(torch.full((npost,), 1.0 - 1.0 / link["tau_d_ms"]))
            g_exc.append(torch.full((npost,), groups[pre]["excitatory"]))
            s_total += npre
            g_total += npost
        cat = lambda xs, dt=None: torch.cat(xs).to(device=device, dtype=dt or xs[0].dtype)
        self.stp_pre, self.stp_U, self.stp_iu, self.stp_ix = cat(stp_pre), cat(stp_U), cat(stp_iu), cat(stp_ix)
        self.stp_scale = cat(stp_scale)
        self.g_cell, self.g_decay, self.g_exc = cat(g_cell), cat(g_decay), cat(g_exc)

        # Связи: Бернулли на каждую пару (connectRandom: drand48() < p, себя
        # тоже может). Строим по кускам пресинаптических клеток — синапсы
        # сразу упорядочены по пре (CSR), у синапса: слот проводимости и
        # слот STP.
        t0 = time.time()
        rows_g, rows_s, counts = [], [], torch.zeros(n, dtype=torch.long, device=device)
        by_pre = {}
        for i, link in enumerate(links):
            by_pre.setdefault(link["pre"], []).append(i)
        self.synapses = 0
        for g in self.names:
            npre = self.size[g]
            mine = by_pre.get(g, [])
            budget = 40_000_000
            width = sum(self.size[links[i]["post"]] for i in mine) or 1
            chunk = max(1, budget // width)
            for lo in range(0, npre, chunk):
                hi = min(npre, lo + chunk)
                pre_parts, g_parts, s_parts = [], [], []
                for i in mine:
                    link = links[i]
                    npost = self.size[link["post"]]
                    p = min(1.0, link["p"] / scale) if degree else link["p"]
                    mask = torch.rand((hi - lo, npost), generator=gen, device=device) < p
                    pre_l, post_l = mask.nonzero(as_tuple=True)
                    del mask
                    pre_l = pre_l + lo
                    pre_parts.append(pre_l)
                    g_parts.append((post_l + g_off[i]).int())
                    s_parts.append((pre_l + stp_off[i]).int())
                pre_l = torch.cat(pre_parts)
                order = torch.sort(pre_l, stable=True).indices
                rows_g.append(torch.cat(g_parts)[order])
                rows_s.append(torch.cat(s_parts)[order])
                counts[self.start[g] + lo:self.start[g] + hi] += torch.bincount(pre_l, minlength=hi)[lo:hi]
        self.syn_g = torch.cat(rows_g)
        self.syn_s = torch.cat(rows_s)
        del rows_g, rows_s
        self.out_count = counts
        self.out_start = torch.cumsum(counts, 0) - counts
        self.synapses = int(self.syn_g.numel())
        print(f"сеть: клеток {n}, синапсов {self.synapses:,}, сборка {time.time() - t0:.1f} с; "
              f"GPU {torch.cuda.memory_allocated() / 2**30:.2f} ГБ" if device.type == "cuda" else "", flush=True)

        # Состояние (CARLsim: v = vr, u = 0; STP u = 0, x = 1).
        self.v = self.vr.clone()
        self.u = torch.zeros(n, device=device)
        self.ref_c = torch.zeros(n, dtype=torch.long, device=device)
        self.cur_spike = torch.zeros(n, dtype=torch.bool, device=device)
        self.su = torch.zeros_like(self.stp_U)
        self.sx = torch.ones_like(self.stp_U)
        self.G = torch.zeros_like(self.g_decay)

    def step(self) -> torch.Tensor:
        # 1. Спад STP и проводимостей.
        u_prev, x_prev = self.su, self.sx
        u = u_prev * (1.0 - self.stp_iu)
        x = x_prev + (1.0 - x_prev) * self.stp_ix
        self.G.mul_(self.g_decay)

        # 2. Спайки шага — выстрелившие при прошлом счёте мембраны.
        fired = self.cur_spike.clone()
        self.cur_spike.zero_()

        # 3. Доставка: STP по слотам выстреливших пре, прирост g по синапсам.
        spiked = fired.index_select(0, self.stp_pre)
        u_new = torch.where(spiked, u + self.stp_U * (1.0 - u_prev), u)
        x_new = torch.where(spiked, x - u_new * x_prev, x)
        eff = u_new * x_prev * self.stp_scale
        self.su, self.sx = u_new, x_new
        idx = fired.nonzero(as_tuple=True)[0]
        if idx.numel():
            cnt = self.out_count.index_select(0, idx)
            total = int(cnt.sum())
            if total:
                base = torch.repeat_interleave(self.out_start.index_select(0, idx), cnt, output_size=total)
                offs = torch.arange(total, device=self.device) - torch.repeat_interleave(cnt.cumsum(0) - cnt, cnt, output_size=total)
                syn = base + offs
                self.G.index_add_(0, self.syn_g.index_select(0, syn).long(), eff.index_select(0, self.syn_s.index_select(0, syn).long()))

        g_e = torch.zeros(self.n, device=self.device).index_add_(0, self.g_cell, torch.where(self.g_exc, self.G, 0.0))
        g_i = torch.zeros(self.n, device=self.device).index_add_(0, self.g_cell, torch.where(self.g_exc, 0.0, self.G))

        # 4. Фон: lognormal × U(0, 1), потолок.
        ext = torch.exp(self.mu + self.sigma * torch.randn(self.n, generator=self.gen, device=self.device))
        ext = (ext * torch.rand(self.n, generator=self.gen, device=self.device)).clamp(max=self.cap)

        # 5. Мембрана: 5 подшагов RK4.
        h = 1.0 / SUBSTEPS
        inv_c = 1.0 / self.C
        for j in range(SUBSTEPS):
            last = j == SUBSTEPS - 1
            v, uu = self.v, self.u
            current = ext - g_e * (v - E_AMPA) - g_i * (v - E_GABAA)

            def fv(vv, ww):
                return (self.k * (vv - self.vr) * (vv - self.vt) - ww + current) * inv_c * h

            def fu(vv, ww):
                return self.a * (self.b * (vv - self.vr) - ww) * h

            k1, l1 = fv(v, uu), fu(v, uu)
            k2, l2 = fv(v + k1 / 2, uu + l1 / 2), fu(v + k1 / 2, uu + l1 / 2)
            k3, l3 = fv(v + k2 / 2, uu + l2 / 2), fu(v + k2 / 2, uu + l2 / 2)
            k4, l4 = fv(v + k3, uu + l3), fu(v + k3, uu + l3)
            v_next = v + (k1 + 2 * k2 + 2 * k3 + k4) / 6
            spike = v_next > self.vpeak
            v_next = torch.where(spike, self.c, v_next)
            u_next = uu + torch.where(spike, self.d, 0.0) + (l1 + 2 * l2 + 2 * l3 + l4) / 6
            v_next = v_next.clamp(min=-90.0)

            refr = self.ref_c > 0
            if last:
                v_next = torch.where(refr, self.c, v_next)
                self.ref_c = torch.where(refr, self.ref_c - 1, torch.where(spike, self.ref, self.ref_c))
            else:
                v_next = torch.where(refr, v, v_next)
            self.u = torch.where(refr, uu, u_next)
            self.v = v_next
            self.cur_spike |= spike & ~refr
        return fired


def main() -> None:
    opts = dict(a.split("=", 1) for a in sys.argv[1:])
    ms = int(opts.get("ms", 2000))
    seed = int(opts.get("seed", 1))
    skip = int(opts.get("skip", 100))
    scale = float(opts.get("scale", 1.0))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    net = json.loads(NET.read_text())
    ca3 = CA3(net, seed, scale, device, degree=opts.get("degree", "0") == "1")
    counts = torch.zeros(ca3.n, device=device)
    early = torch.zeros(ca3.n, device=device)
    t0 = time.time()

    def report(t_ms: int, final: bool = False) -> None:
        span = max(1, t_ms - skip)
        print(f"--- {t_ms} мс ({time.time() - t0:.0f} с), частоты {skip}–{t_ms} мс" + (" ИТОГ" if final else ""), flush=True)
        for i, g in enumerate(ca3.names):
            mask = ca3.type_of == i
            hz = counts[mask] / (span / 1000.0)
            paper = PAPER_HZ[g]
            silent = float((hz == 0).float().mean())
            print(f"  {g:18s} {float(hz.mean()):7.2f} ± {float(hz.std()):5.2f} Гц  (статья {paper[0]:.2f} ± {paper[1]:.2f}); "
                  f"молчат {silent:.0%}", flush=True)

    for t in range(ms):
        fired = ca3.step()
        if t >= skip:
            counts += fired
        if t < 500:
            early += fired
        if (t + 1) % 250 == 0 and t + 1 < ms:
            report(t + 1)
    report(ms, final=True)
    print("первые 500 мс (окно покоя авторов):")
    for i, g in enumerate(ca3.names):
        mask = ca3.type_of == i
        print(f"  {g:18s} {float((early[mask] / 0.5).mean()):7.2f} Гц")


if __name__ == "__main__":
    main()
