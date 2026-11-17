"""
Токи в блоке L5 руки при команде (DESIGN 2026-11-16 «Адресный вход в карту
руки → нажатие»: ток в участок L2/3 разгоняет его до 26 Гц, но блок L5 у
клавиши стреляет как в покое; ток в сам блок L5 жмёт нужную клавишу).
Вопрос: где теряется сигнал L2/3 → L5 — мало возбуждения (гипотеза A:
участок L2/3 уже полосы входов L5, суммарного тока не хватает), сильное
торможение (PV → L5 растёт вместе с командой) или несогласованные спайки
(гипотеза B: 26 Гц вразнобой, PSP не складываются).

Три состояния, перемешаны: покой (тока нет) / команда в L2/3 (ток amp23 мВ
клеткам L2/3 E в участке ±near карты у блока клавиши — КОСТЫЛЬ, ISSUES 24) /
ток amp5 мВ в сам блок L5 (±near5, положительный контроль). Окна 5 мс (шаг
счёта observe — 10 шагов по 0.5 мс), считаются только клетки блока L5 цели.

Токи — в мВ, как они входят в мембрану (engine: synaptic → dv):
- по источникам — поставка в равновесии: Σ вес × частота входа × dt/1000 ×
  syn_scale/(1 − decay_syn) × движущая сила по v клетки; торможение ×
  inh_gain. Без кратковременной депрессии и плато дендрита (верхняя оценка
  для путей с депрессией). Группы: (1) L2/3 E руки — из участка команды и
  вне его; (2) прочее возбуждение — местное (L5/L6/L4 руки), кора, ядра
  (таламус и др.), процедурные пути (по средней частоте популяции);
  (3) торможение — PV руки L5, прочие тормозные; среднее поле недостающих
  соседей — отдельно (одно на популяцию);
- итог — снимок в конце окна: возбуждающий ток (i_syn + NMDA) × движущая
  сила, тормозной (i_inh + i_ngf) × сила, среднее поле, ток команды; v,
  расстояние до порога v_t, адаптация.
Синхронность (B): входов L2/3 E на клетку блока, выстреливших в одном окне
5 мс — среднее и 95-й перцентиль по окнам; сколько таких PSP одновременно
нужно до порога (расстояние до v_t / средний PSP L2/3 → L5 в мВ у v
клетки); фано счёта спайков участка по окнам (1 — пуассон, ≫1 — залпы).

    PYTHONPATH=. .venv/bin/python tools/hand_l5_input.py [seed=1] [keys=пиржтс] [reps=2] [amp23=30] [amp5=10] [near=0.005] [near5=0.0022] [cmd=300] [rest=300]
"""

from __future__ import annotations

import sys
import time

import torch

from brain.session import Session
from brain.wiring import ProceduralProjection
from tools.reading_bench import build

WIN = 5.0


def main(opts: dict) -> None:
    seed = int(opts.get("seed", 1))
    opts = {"fit": "lgn-a", "homeo": f"lgn-d-pop-tcind-s{seed}", "relay_v_t_mv": "-59", "trn_scale": "0.1",
            "map_ee_sheet": "hand", **opts}
    mind, b = build(opts)
    cfg = b.cfg
    s = Session(mind, window_ms=WIN)
    va_bias = float(opts.get("va_bias", -76))
    reps, near, near5 = int(opts.get("reps", 2)), float(opts.get("near", 0.005)), float(opts.get("near5", 0.0022))
    amp23, amp5 = float(opts.get("amp23", 30)), float(opts.get("amp5", 10))
    cmd_ms, rest_ms = float(opts.get("cmd", 300)), float(opts.get("rest", 300))
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    A = b.areas
    if va_bias:
        b.bias[A["thalamus"].E.slice] += va_bias
    hand = A["hand"]
    E23, L5 = hand.layers["L23"]["E"], s.H
    PV5 = hand.layers["L5"]["PV"]
    c23, c5 = b.field_cols[E23.name].float().to(b.device), b.field_cols[L5.name].float().to(b.device)
    kb, cells = s.keyboard, s.hand.cells
    targets = [kb.layout.index(c) for c in opts.get("keys", "пиржтс")]

    # Население по номеру клетки: какой популяции принадлежит вход.
    pops = {}
    for p in b.projections:
        for P in (getattr(p, "pre_pop", None), p.post):
            if P is not None:
                pops[P.name] = P
    pops = sorted(pops.values(), key=lambda P: P.start)
    starts = torch.tensor([P.start for P in pops], device=b.device)
    cortex = {nm for nm, ar in A.items() if getattr(ar, "layers", None)}
    area_of = lambda P: P.name.split(":")[0].split(".")[0]  # noqa: E731

    into = [p for p in b.projections if p.post.start == L5.start and p.post.size == L5.size and p.source == "net"]
    print(f"va_bias {va_bias:+.0f}, map_ee_sheet {cfg.map_ee_sheet}; входы L5 E руки: "
          + ", ".join(f"{p.name}{' (проц.)' if isinstance(p, ProceduralProjection) else ''}" for p in into), flush=True)

    def group_of(pre_pop, sign: float) -> str:
        if sign < 0:
            return "торм: PV L5 руки" if pre_pop is PV5 else f"торм: прочее"
        if pre_pop is E23:
            return "L2/3 E руки"
        ar = area_of(pre_pop)
        if ar == "hand":
            return "возб: местное (L4/L5/L6 руки)"
        return "возб: кора" if ar in cortex else "возб: ядра (таламус и др.)"

    # Хранимые пути: pre [8 блоков × cells, k], вес и группа каждого входа.
    stored, proc = [], []
    for p in into:
        if isinstance(p, ProceduralProjection):
            proc.append(p)
            continue
        pre = p.pre.long().to(b.device)
        w = p.weight.float().to(b.device)
        pop_id = torch.searchsorted(starts, pre, right=True) - 1
        stored.append((p, pre, w, pop_id))
    groups = ["L2/3 E руки: участок", "L2/3 E руки: вне участка", "возб: местное (L4/L5/L6 руки)", "возб: кора",
              "возб: ядра (таламус и др.)", "возб: процедурные", "возб: фон Potjans (пуассон)", "торм: PV L5 руки", "торм: прочее"]
    for p, pre, w, pop_id in stored:
        names = sorted({group_of(pops[int(i)], float(w.mean())) for i in pop_id.unique()})
        k_in = pre.shape[1]
        print(f"  {p.name}: входов на клетку {k_in}, вес {float(w.mean()):+.3f} (ед. движка) → {', '.join(names)}", flush=True)
    for p in proc:
        print(f"  {p.name} (проц.): входов {p.k_in}, вес {p.value:+.3f} из {p.pre_pop.name}", flush=True)

    excite = lambda v: ((cfg.ampa_reversal_mv - v) / (cfg.ampa_reversal_mv - cfg.epsp_holding_mv)).clamp(min=0.0)  # noqa: E731
    force = lambda v: ((v - cfg.gaba_reversal_mv) / (cfg.ipsp_holding_mv - cfg.gaba_reversal_mv)).clamp(min=0.0)  # noqa: E731
    ss_e = b.syn_scale / (1.0 - b.decay_syn) * cfg.dt_ms / 1000.0       # Σ w·Гц → равновесный i_syn
    if b.nmda_on:
        f = (b.nmda_scale / (1.0 - b.decay_nmda) / (b.syn_scale / (1.0 - b.decay_syn)))[L5.slice]
        print(f"NMDA включён: на ту же поставку равновесный NMDA / AMPA = {float(f.mean()):.2f} (до блока Mg)", flush=True)
    else:
        print("NMDA выключен", flush=True)
    ss_i = b.syn_scale_inh / (1.0 - b.decay_syn_inh) * cfg.dt_ms / 1000.0

    # Пик PSP на единицу веса: i ← w·τm/τs, затухает τs; v = w·τm/(τm − τs)·(e^(−t/τm) − e^(−t/τs)).
    tm, ts = b.tau_m.float(), b.tau_syn.float()
    tpk = torch.log(tm / ts) * tm * ts / (tm - ts)
    peak = tm / (tm - ts) * (torch.exp(-tpk / tm) - torch.exp(-tpk / ts))

    s.line, s.gaze = "", 0.0
    b.run(2500.0, s._blank(2500.0))
    blank = s._blank(WIN)

    states = ["покой", "команда в L2/3", "ток в L5"]
    plan = [(st, k) for st in states for k in targets for _ in range(reps)]
    perm = torch.randperm(len(plan), generator=gen).tolist()
    plan = [plan[i] for i in perm]
    keep = {st: {"src": [], "tot": [], "coin": [], "need": [], "patchcnt": [], "blk": [], "hit": 0, "n": 0} for st in states}
    held = None
    t0 = time.time()
    for j, (st, k) in enumerate(plan):
        blk = torch.arange(L5.start + k * cells, L5.start + (k + 1) * cells, device=b.device)
        pos = float(c5[k * cells: (k + 1) * cells].mean())
        patch23 = ((c23 - pos).abs() <= near).nonzero().flatten() + E23.start
        in_patch = torch.zeros(b.n_slots, dtype=torch.bool, device=b.device)
        in_patch[patch23] = True
        for _ in range(int(rest_ms / WIN)):
            r = b.observe(WIN, blank)
            kk = s.hand.pressed(r[L5.slice])
            held = kk
        b.kick_current.zero_()
        if st == "команда в L2/3":
            b.kick_current[patch23] = amp23
        elif st == "ток в L5":
            p5 = ((c5 - pos).abs() <= near5).nonzero().flatten() + L5.start
            b.kick_current[p5] = amp5
        b.kick_left.fill_(round(cmd_ms / cfg.dt_ms) + 1)
        K = keep[st]
        K["n"] += 1
        hit = False
        for _ in range(int(cmd_ms / WIN)):
            r = b.observe(WIN, blank)
            kk = s.hand.pressed(r[L5.slice])
            hit |= kk == k and held != k
            held = kk
            # Входы могут указывать на фиктивные клетки (≥ n): их частота 0.
            r = torch.cat([r, torch.zeros(b.n_slots - b.n, device=r.device)])
            v = b.v[blk]
            ex, fo = excite(v), force(v) * b.inh_gain[blk]
            rows = blk - L5.start
            src = dict.fromkeys(groups, 0.0)
            coin = torch.zeros(len(blk), device=b.device)
            psp23 = []
            for p, pre, w, pop_id in stored:
                pr, ww, pid = pre[rows], w[rows], pop_id[rows]
                rate = r[pr]
                sup = ww * rate
                sign = float(ww.mean())
                if sign < 0:
                    g_pv = pid == pops.index(PV5)
                    src["торм: PV L5 руки"] += (sup * g_pv).sum(1) * ss_i[blk] * fo
                    src["торм: прочее"] += (sup * ~g_pv).sum(1) * ss_i[blk] * fo
                    continue
                g23 = pid == pops.index(E23)
                ip = in_patch[pr] & g23
                src["L2/3 E руки: участок"] += (sup * ip).sum(1) * ss_e[blk] * ex
                src["L2/3 E руки: вне участка"] += (sup * (g23 & ~ip)).sum(1) * ss_e[blk] * ex
                other = ~g23
                if bool(other.any()):
                    for i in pid[other].unique():
                        m = pid == i
                        src[group_of(pops[int(i)], sign)] += (sup * m).sum(1) * ss_e[blk] * ex
                if bool(g23.any()):
                    coin += ((rate > 0) & g23).float().sum(1)
                    psp23.append(ww[g23].mean())
            for p in proc:
                rate = float(r[p.pre_pop.slice].mean())
                if p.value < 0:
                    src["торм: прочее"] += p.k_in * p.value * rate * ss_i[blk] * fo
                else:
                    src["возб: процедурные"] += p.k_in * p.value * rate * ss_e[blk] * ex
            # Внешний вход колонки (Potjans: пуассон от остального мозга) — drive = noise × noise_gain.
            src["возб: фон Potjans (пуассон)"] = b.noise_mean[blk] * b.noise_gain[blk] * b.syn_scale[blk] / (1.0 - b.decay_syn[blk]) * ex
            K["src"].append(torch.stack([torch.as_tensor(src[g], device=b.device).expand(len(blk)).float() for g in groups]).mean(1).cpu())
            if b.nmda_on:
                blockmg = 1.0 / (1.0 + cfg.mg_mm / 3.57 * torch.exp(-0.062 * v)) if b.nmda_mg else torch.ones_like(v)
                nmda = (b.i_nmda[blk] + b.nmda_r[blk] * b.nmda_mf[blk]) * blockmg
            else:
                nmda = torch.zeros_like(v)
            br = b.i_dend.view(b.n_slots, max(1, cfg.dendrite_branches))[blk]
            plateau = torch.sigmoid((br - cfg.dendrite_threshold_mv) / cfg.dendrite_slope_mv).amax(1)
            dce, dci = b.last_dc_e, b.last_dc_i
            dce = dce[blk] if torch.is_tensor(dce) and dce.numel() > 1 else torch.zeros_like(v) + float(dce)
            dci = dci[blk] if torch.is_tensor(dci) and dci.numel() > 1 else torch.zeros_like(v) + float(dci)
            tot = torch.stack([
                b.i_syn[blk] * ex, nmda * ex, plateau, (b.i_inh[blk] + b.i_ngf[blk]) * force(v), dce * ex, dci * force(v),
                b.kick_current[blk] * (b.kick_left > 0).float(), b.bias[blk], b.adapt[blk], v, b.v_t[blk] - v,
                (r[blk] > 0).float(),
            ]).mean(1)
            K["tot"].append(tot.cpu())
            K["coin"].append(coin.cpu())
            # Нужно одновременных PSP L2/3 до порога: (v_t − v) / (PSP L2/3 → L5 в мВ у этого v).
            if psp23:
                psp_mv = float(torch.stack(psp23).mean()) * peak[blk] * ex
                K["need"].append(((b.v_t[blk] - v) / psp_mv.clamp_min(1e-6)).cpu())
            K["patchcnt"].append(float((r[patch23] > 0).float().sum()))
        b.kick_current.zero_()
        b.kick_left.zero_()
        K["hit"] += hit
        if j == 0:
            print(f"  первая попытка: {time.time() - t0:.0f} с", flush=True)
    print(f"попыток {len(plan)}, {time.time() - t0:.0f} с; поставка — равновесные токи, мВ (как в dv)\n", flush=True)

    head = f"{'':34s}" + "".join(f"{st:>16s}" for st in states)
    print(head)
    S = {st: torch.stack(keep[st]["src"]).mean(0) for st in states}
    for i, g in enumerate(groups):
        print(f"  {g:32s}" + "".join(f"{float(S[st][i]):16.2f}" for st in states))
    exc_sum = {st: float(S[st][:7].sum()) for st in states}
    inh_sum = {st: float(S[st][7:].sum()) for st in states}
    print(f"  {'Σ возб (поставка)':32s}" + "".join(f"{exc_sum[st]:16.2f}" for st in states))
    print(f"  {'Σ торм (поставка)':32s}" + "".join(f"{inh_sum[st]:16.2f}" for st in states))
    print("снимок в конце окна (блок L5 цели):")
    names = ["ток AMPA (i_syn)", "ток NMDA (с блоком Mg)", "плато дендрита (доля)", "ток торм (i_inh+ngf)", "среднее поле возб", "среднее поле торм", "ток команды",
             "bias (гомеостаз и др.)", "адаптация", "v, мВ", "до порога v_t, мВ", "доля выстреливших в окне"]
    T = {st: torch.stack(keep[st]["tot"]).mean(0) for st in states}
    for i, nm in enumerate(names):
        print(f"  {nm:32s}" + "".join(f"{float(T[st][i]):16.2f}" for st in states))
    print("синхронность входа L2/3 → блок L5:")
    for st in states:
        K = keep[st]
        c = torch.stack(K["coin"]).flatten()
        need = torch.stack(K["need"]).flatten() if K["need"] else torch.tensor([float("nan")])
        cnt = torch.tensor(K["patchcnt"])
        fano = float(cnt.var() / cnt.mean().clamp_min(1e-9))
        print(f"  {st:16s}: входов L2/3 в окне 5 мс на клетку среднее {float(c.mean()):.1f}, 95% {float(c.quantile(0.95)):.0f}, "
              f"макс {float(c.max()):.0f}; нужно одновременных PSP до порога медиана {float(need.median()):.0f}; "
              f"спайков участка в окне {float(cnt.mean()):.1f}, фано {fano:.2f}; попаданий {K['hit']}/{K['n']}", flush=True)
    print(f"всего {time.time() - t0:.0f} с")


if __name__ == "__main__":
    main(dict(kv.split("=", 1) for kv in sys.argv[1:]))
