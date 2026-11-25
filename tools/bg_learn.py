"""
б3а (docs/PBWM_TRANSFER.md): учатся ли ворота BG по дофамину — управляющая часть задачи sir2
(O'Reilly & Frank 2006, https://doi.org/10.1162/089976606775093909; код emer/leabra examples/sir2):
«control information is all that the Matrix layer requires».

Сеть — BG Lindahl & Hellgren Kotaleski 2016 (tools/bg_lindahl.py), норма дофамина, кора «активация»
(fig1_Beta_0_10000_tuned.json), каналы как «D1 и D2» в опыте выбора авторов (tools/bg_select.py):
2 набора у M1, M2, GI, SN; M1→SN, M2→GI, GI→SN — set-set. Канал 0 — «записать», канал 1 — «выдать».

Проба: покой rest мс, затем окно 100 мс с сигналом задачи k ∈ {S, I, R} (равновероятно):
  - вход задачи: каждая MSN i канала c получает добавочный пуассон с частотой фон_i · g[b, i, k]
    (фон — C1 у D1, C2 у D2; синапсы как у её входа коры: AMPA + NMDA одними спайками,
    задержка 2.5 мс) — так авторы подают вспышку коры (rate·(amp − 1), burst3);
  - решение: канал открыт, если SNr набора за окно ниже 14 Гц (порог опыта выбора авторов);
  - награда сразу: S → открыт только 0, I → ни один, R → только 1 (1 — верно, 0 — нет);
  - дофамин — Rescorla–Wagner: DA = r − V[b, k], V += α·DA (костыль: скаляр вместо VTA/SNc, ISSUES);
  - правило — след PBWM (leabra pbwm_paths.go, DWtMatrix): метка = pre_k · n_i (n_i — спайки MSN
    в окне; у PBWM f(act_post)·act_pre), для открывшегося канала +, для закрытого −0.7 (NotGatedLR);
    Δg = lr · DA · метка; у D2 знак DA обращён, при DA > 0 и метке < 0 — ×0.1 (GateNoGoPosLR);
    g ∈ [0, gmax]. Перевод «активности» PBWM в спайки (n_i) — допущение (ISSUES 42).

task=sir (б3б) — задача sir с одной ячейкой и отложенной наградой (sir2_env.go StepSIR без второй
ячейки): S (буква из 4) — только если хранить нечего, I (буква-помеха), R — только если есть что
вспомнить; ячейка ПФК — регистр-костыль: открыт канал 0 → записана буква пробы, открыт канал 1 →
выдано содержимое до записи; награда только на R (верно — буква последнего S); дофамин только на R;
след PBWM — tr += ntr − min(1, |ntr|)·tr у весов входа пробы (Decay 1), обучение по всем следам на R.

Строки пачки — независимые агенты (свои веса g, V, случайные задачи) на одной сети.

    PYTHONPATH=tools:. .venv/bin/python tools/bg_learn.py trials=100 B=32 seed=1 out=/tmp/bgl/learn_s1
"""

from __future__ import annotations

import json
import os
import sys
import time

import torch

from bg_lindahl import DT, Net
from bg_select import go_nogo_par
from bg_tune import oscillating

PAR = "~/.brain/lindahl/dump/fig1_Beta_0_10000_tuned.json"
THRESHOLD = 14.0
WIN = 100.0
TASKS = ["S", "I", "R"]
CORRECT = torch.tensor([[1, 0], [0, 0], [0, 1]], dtype=torch.bool)   # какие каналы должны открыться
DELAY_STEPS = 25   # 2.5 мс — задержка входа коры на MSN (conn C1_M1, C2_M2)


def main(o: dict) -> None:
    dev = o.get("device", "cuda")
    seed = int(o.get("seed", 1))
    B = int(o.get("B", 32))
    trials = int(o.get("trials", 100))
    rest = float(o.get("rest", 300.0))
    lr = float(o.get("lr", 0.02))
    alpha = float(o.get("alpha", 0.1))
    g0 = float(o.get("g0", 0.4))
    gmax = float(o.get("gmax", 2.0))
    torch.manual_seed(seed)
    sir = o.get("task", "ctrl") == "sir"
    nm = int(o.get("nm", 1)) if sir else 1        # полос памяти (у sir2 — 8; вывод k выдаёт память k)
    nch = 2 * nm                                  # каналы: 0..nm−1 — запись в полосу, nm..2nm−1 — вывод полосы
    par = go_nogo_par(json.load(open(os.path.expanduser(o.get("par", PAR)))), nch)
    # расписание входов коры — на всю длину прогона (в выгрузке — до sim_time 2000 мс)
    old_stop = float(par["simu"]["sim_time"])
    par["simu"]["sim_time"] = 1000.0 + trials * (rest + WIN) + 100.0
    for nd in par["node"].values():
        for st in nd.get("spike_setup") or []:
            if st.get("t_stop") is not None and float(st["t_stop"]) == old_stop:
                st["t_stop"] = par["simu"]["sim_time"]
    par = oscillating(par, par["tuning"]["upp"], par["tuning"]["state"])
    net = Net(par, seed, dev, 10, B=B, compile_=True, fp32=True)
    # MSN каналов: D1 (M1) и D2 (M2), набор c — каждая вторая клетка
    cells, base, w_ampa, w_nmda, chan, is_d2 = [], [], [], [], [], []
    for pop, inp in (("M1", "C1"), ("M2", "C2")):
        a, z = net.off[pop]
        ca, cn = par["conn"][f"{inp}_{pop}_ampa"], par["conn"][f"{inp}_{pop}_nmda"]
        idx = torch.arange(a, z)
        cells.append(idx)
        base.append(torch.full((z - a,), float(par["node"][inp]["rate"])))
        w_ampa.append(torch.full((z - a,), float(ca["weight"]["params"])))
        w_nmda.append(torch.full((z - a,), float(cn["weight"]["params"])))
        chan.append(torch.arange(z - a) % nch)
        is_d2.append(torch.full((z - a,), pop == "M2"))
    cells, base = torch.cat(cells).to(dev), torch.cat(base).to(dev)
    w_ampa, w_nmda = torch.cat(w_ampa).to(dev), torch.cat(w_nmda).to(dev)
    chan, is_d2 = torch.cat(chan).to(dev), torch.cat(is_d2).to(dev)
    nC = cells.numel()
    gen = torch.Generator(device="cpu").manual_seed(seed + 1000)
    g = (g0 * 2 * torch.rand(B, nC, 3, generator=gen)).to(dev)      # начальный разброс 0..2·g0
    if "sweep" in o:   # калибровка исследования: g0 строки — linspace(a, b, B), без учёбы (lr=0)
        lo_, hi_ = (float(x) for x in o["sweep"].split(","))
        g0s = torch.linspace(lo_, hi_, B)
        g = (g0s[:, None, None] * 2 * torch.rand(B, nC, 3, generator=gen)).to(dev)
    V = torch.zeros(B, 3, device=dev)
    sn = [torch.arange(net.off["SN"][0] + c, net.off["SN"][1], nch, device=dev) for c in range(nch)]
    sn_n = torch.tensor([len(x) for x in sn], device=dev, dtype=torch.float)
    steps_rest, steps_win = int(round(rest / DT)), int(round(WIN / DT))
    log = {k: [] for k in ("task", "open", "reward", "da", "g_mean")}
    t0 = time.time()
    # прогрев 1000 мс
    for _ in range(int(round(1000.0 / DT))):
        net.step()
    print(f"прогрев {time.time() - t0:.0f} с", flush=True)
    ar = torch.arange(B, device=dev)
    maint = torch.full((B,), -1, dtype=torch.long, device=dev)   # среда: буква последнего S (−1 — нет)
    reg = torch.full((B, nm), -1, dtype=torch.long, device=dev)  # полосы ПФК (костыль-регистры)
    trace = torch.zeros(B, nC, 3, device=dev)
    for tr in range(trials):
        for _ in range(steps_rest):
            net.step()
        if sir:
            # среда sir (sir2_env.go StepSIR, одна ячейка): S — только если хранить нечего, R — только если есть
            allowed = torch.stack([maint < 0, torch.ones_like(maint, dtype=torch.bool), maint >= 0], 1).float()
            task = torch.multinomial(allowed.cpu(), 1, generator=gen).squeeze(1).to(dev)
            letter = torch.where(task == 2, torch.full_like(task, -1), torch.randint(0, 4, (B,), generator=gen).to(dev))
        else:
            task = torch.randint(0, 3, (B,), generator=gen).to(dev)
        rate = base[None, :] * g[ar, :, task]                       # [B, nC], Гц
        lam = rate * (DT * 1e-3)
        n_msn = torch.zeros(B, nC, device=dev)
        n_sn = torch.zeros(B, nch, device=dev)
        for _ in range(steps_win):
            c = torch.poisson(lam)
            slot = (net.k + DELAY_STEPS) % net.D
            net.ring[slot, 0][:, cells] += (c * w_ampa).to(net.ring.dtype)
            net.ring[slot, 1][:, cells] += (c * w_nmda).to(net.ring.dtype)
            s = net.step()
            n_msn += s[:, cells].float()
            for ch in range(nch):
                n_sn[:, ch] += s[:, sn[ch]].float().sum(1)
        sn_hz = n_sn / sn_n / (WIN / 1000.0)
        opened = sn_hz < THRESHOLD                                   # [B, 2]
        if "debug" in o:
            print("SNr каналов", sn_hz.tolist(), "MSN спайков", float(n_msn.sum()), flush=True)
        gate_sign = torch.where(opened[:, chan], 1.0, -0.7)          # [B, nC]
        ntr = gate_sign * n_msn                                      # метка пробы (вход задачи k = task)
        sgn = torch.where(is_d2[None, :], -1.0, 1.0)
        if sir:
            # вывод: открыт канал 1 → содержимое ячейки до записи этой пробы
            # вывод: открытые каналы вывода выдают содержимое своих полос до записи этой пробы;
            # верно — открыт хоть один и все выданные буквы = буква последнего S
            o_out = opened[:, nm:]
            agree = ((reg == maint[:, None]) | ~o_out).all(1)
            is_r = task == 2
            good = o_out.any(1) & agree
            r = torch.where(is_r, good.float(), torch.full_like(task, float("nan"), dtype=torch.float))
            reg = torch.where(opened[:, :nm], letter[:, None], reg)  # запись: открыт канал полосы → буква пробы
            maint = torch.where(task == 0, letter, torch.where(is_r, torch.full_like(maint, -1), maint))
            # дофамин только на R (награда — только на Recall, sir2.go ApplyReward)
            da = torch.where(is_r, torch.nan_to_num(r) - V[:, 2], torch.zeros_like(V[:, 2]))
            V[:, 2] += alpha * da
            # след PBWM (DWtMatrix, Decay 1): tr += ntr − min(1, |ntr|)·tr — у весов входа этой пробы
            t_old = trace[ar, :, task]
            trace[ar, :, task] = t_old + ntr - ntr.abs().clamp(max=1.0) * t_old
            # обучение по всем следам при дофамине
            mult = torch.where(is_d2[None, :, None] & (da[:, None, None] > 0) & (trace < 0), 0.1, 1.0)
            g = (g + lr * da[:, None, None] * sgn[:, :, None] * trace * mult).clamp(0.0, gmax)
        else:
            r = (opened == CORRECT.to(dev)[task]).all(1).float()
            da = r - V[ar, task]
            V[ar, task] += alpha * da
            # след PBWM: метка = pre_k · n_i, знак по тому, открылся ли канал клетки
            mult = torch.where(is_d2[None, :] & (da[:, None] > 0) & (ntr < 0), 0.1, 1.0)
            dg = lr * da[:, None] * sgn * ntr * mult
            g[ar, :, task] = (g[ar, :, task] + dg).clamp(0.0, gmax)
        log["task"].append(task.cpu()); log["open"].append(opened.cpu()); log["reward"].append(r.cpu())
        log["da"].append(da.cpu())
        log["g_mean"].append(torch.stack([g[:, (chan == ch) & (is_d2 == d2), :].mean((0, 1)) for ch in range(nch)
                                          for d2 in (False, True)]).cpu())
        if (tr + 1) % 10 == 0:
            rr = torch.stack(log["reward"][-10:]).mean(0)
            tk = torch.stack(log["task"][-10:]); rw = torch.stack(log["reward"][-10:])
            if sir:
                op = torch.stack(log["open"][-10:]).float()
                per = " ".join(f"{TASKS[k]}: " + " ".join(f"{float(op[..., c][tk == k].mean()):.2f}" for c in range(nch)) + ";"
                               for k in range(3))
                rr = torch.nanmean(rw)
            else:
                per = " ".join(f"{TASKS[k]} {float(rw[tk == k].mean()):.2f}" for k in range(3))
            print(f"проба {tr + 1}: верно {float(rr.mean()):.2f} ({per}), SNr {float(sn_hz.mean()):.1f} Гц, "
                  f"{time.time() - t0:.0f} с", flush=True)
    if "sweep" in o:
        op = torch.stack(log["open"]).float().mean(0)   # [B, 2]
        for b in range(B):
            print(f"  g0 {float(g0s[b]):.3f}: открыт канал 0 {float(op[b, 0]):.2f}, канал 1 {float(op[b, 1]):.2f}")
    out = o.get("out")
    if out:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        torch.save({k: torch.stack(v) for k, v in log.items()} | {"opts": o}, out + ".pt")
        plot(out)


def plot(out: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = torch.load(out + ".pt", weights_only=False)
    task, rew, opened, gm = d["task"], d["reward"], d["open"], d["g_mean"]
    T = rew.shape[0]
    k = 10
    sm = lambda x: torch.nn.functional.avg_pool1d(x[None, None].float(), k, 1)[0, 0]
    fig, ax = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    ok = ~torch.isnan(rew)
    rz = torch.nan_to_num(rew)
    ax[0].plot(torch.arange(k - 1, T), sm(rz.sum(1) / ok.sum(1).clamp(min=1)), label="все", color="k")
    for i, name in enumerate(TASKS):
        m = ((task == i) & ok).float()
        if m.sum() == 0:
            continue
        frac = (rz * m).sum(1) / m.sum(1).clamp(min=1)
        ax[0].plot(torch.arange(k - 1, T), sm(frac), label=name)
    ax[0].set_ylabel("доля верных (окно 10 проб)")
    ax[0].set_ylim(0, 1.05)
    ax[0].legend()
    nch_p = opened.shape[2]
    for i, name in enumerate(TASKS):
        m = (task == i)
        for ch, ls in [(c, ("-", "--", ":", "-.")[c % 4]) for c in range(nch_p)]:
            f = (opened[:, :, ch] & m).float().sum(1) / m.float().sum(1).clamp(min=1)
            ax[1].plot(torch.arange(k - 1, T), sm(f), ls, label=f"{name}: канал {ch} открыт")
    ax[1].set_ylabel("доля открытий")
    ax[1].legend(fontsize=7, ncol=3)
    labels = [f"{'NoGo' if d2 else 'Go'} кан.{c}" for c in range(nch_p) for d2 in (False, True)]
    for j, lab in enumerate(labels):
        for i, name in enumerate(TASKS):
            ax[2].plot(gm[:, j, i], label=f"{lab} ← {name}")
    ax[2].set_ylabel("средний вес g")
    ax[2].set_xlabel("проба")
    ax[2].legend(fontsize=6, ncol=4)
    fig.suptitle("б3: ворота BG учатся по дофамину" + (f" — sir, полос {nch_p // 2} (каналы 0..{nch_p // 2 - 1} — запись, далее — вывод)"
                                                         if nch_p > 2 or "sir" in str(d["opts"]) else " (S → канал 0, I → ничего, R → канал 1)"))
    fig.tight_layout()
    fig.savefig(out + ".png", dpi=90)
    print(out + ".png")


if __name__ == "__main__":
    main(dict(kv.split("=", 1) for kv in sys.argv[1:]))
