"""
Офлайн-обучение пути V1 → буквы на записанных спайках V1 L2/3 E мозга
(tools/v1_record.py): выучиваема ли буква из такого входа конкурентным
обучением и за сколько показов — без часов полного мозга (DESIGN «Обучение
пути V1 → буквы»).

Вход — записанные спайки как есть (шаг 0.5 мс сведён к 1 мс, частоты и
временной строй сохранены, без усиления). Клетки букв L4 E — все 5680, с
настоящими входами пути (514 источников V1 L2/3 E на клетку) и врождёнными
весами (мВ). Модель клетки — LIF (τ 20 мс, покой −65, порог −50, сброс −65,
рефрактерность 2 мс), вход пре — скачок напряжения на вес синапса.
Прочих входов клетки мозга (фон, обратная связь, местные E) в записи нет —
их заменяет шум (σ noise мВ за мс) и общий сдвиг, подобранный один раз до
учёбы под частоту фона клеток букв L4 E в мозге 1.7 Гц (letters_dc,
2026-11-12) [упрощение офлайн-модели].

Учебные записи — первые 2/3 показов каждой буквы (по кругу, сколько надо
показов), метки — следующая 1/6, проверка — последняя 1/6: проверочные
записи в учёбе не участвуют. Каждые every учебных показов — замер без
пластичности: метки клеткам по ответу на показы меток (как у Diehl & Cook),
точность голосом клеток и корреляцией с центром буквы, с перемешанными
метками, доля избирательных клеток (F выше 95% перемешанных), спайков на
показ, |Δw| и корреляция выученного у клеток у ямки.

rule:
    native  родное правило мозга (engine.chemical_plasticity): на спайк
            поста w += A+ · след пре · средний врождённый вес, на спайк пре
            w −= A− · w · след поста (LTD от веса), следы τ 20 / 20 мс
            (AreaSpec letters), A+ 0.01, A− 0.01·51/103; раз в 50 мс сумма
            строки — к врождённой (engine._conserve), вес в [0, cap].
            Торможение — популяция PV (E:PV 4:1; E ← 117 PV по −0.21 мВ,
            PV ← 763 E по 0.45 мВ — числа местных связей L4 мозга), постоянное.
    vogels  native + пластичность PV → E (Vogels et al. 2011,
            https://doi.org/10.1126/science.1211095): спайк PV — w += η (след
            поста − α), спайк E — w += η · след PV, α = 2 · 1 Гц · 20 мс
            (норма cfg.inhibitory_target_hz), потолок 4 мВ (cfg.vogels_cap),
            η vogels_eta мВ [подбор, не по данным].
    clopath правило по напряжению как в движке (engine._clopath, числа —
            cfg.clopath_* из BrainConfig, Clopath et al. 2010,
            https://doi.org/10.1038/nn.2479): u — мембрана, во время спайка
            clopath_spike_mv на clopath_spike_ms; ū₋, ū₊ — фильтры u (τ 10 / 7
            мс), ū — деполяризация от покоя клетки за 1 с, x̄ пре += спайк/τx;
            на спайк пре w −= A_LTD·ū²/u_ref²·(ū₋ − θ₋)₊, каждый шаг w +=
            A_LTP·dt·x̄·(u − θ₊)₊·(ū₊ − θ₋)₊; θ₋ — покой клетки (V_REST +
            сдвиг), θ₊ абсолютный; вес в долях среднего врождённого, [0, cap];
            сохранения суммы нет (как в движке). Торможение — PV, как native.
            Отличие от мозга: клетка LIF, а не AdEx (выше θ₊ — только спайк).
    dc      Diehl & Cook 2015 (https://doi.org/10.3389/fncom.2015.00099, числа
            cfg.dc_*): следы пре τ 20, поста τ 20 / 40, ставятся в 1; спайк
            пре w −= ltd · пост₁, спайк поста w += ltp · пре · пост₂, в долях
            среднего врождённого веса (ltd 0.001, ltp 0.1, потолок 10);
            адаптивный порог +0.05 мВ на спайк; нормировка суммы весов клетки
            перед показом; торможение как у авторов — спайк E тормозит все
            прочие E (dc_inh мВ), PV нет.

    PYTHONPATH=. .venv/bin/python tools/letters_offline.py rec=файл.pt [rule=native|vogels|clopath|dc] [shows=3000] [every=200] [noise=2.0] [seed=1] [threads=2] [device=cpu|cuda]
"""

from __future__ import annotations

import math
import sys
import time

import torch

DT = 1.0
TAU_M, V_REST, V_TH, V_RESET, REFRACT = 20.0, -65.0, -50.0, -65.0, 2


def bins(spikes: torch.Tensor, steps: int, n: int, ratio: int) -> torch.Tensor:
    """Спайки [k, 2] (шаг записи, клетка) → [шаг 1 мс, клетка] (сколько спайков в шаге)."""

    x = torch.zeros(steps, n)
    t = (spikes[:, 0].long() // ratio).clamp(max=steps - 1)
    x.index_put_((t, spikes[:, 1].long()), torch.ones(len(t)), accumulate=True)
    return x


def main(opts: dict) -> None:
    torch.set_num_threads(int(opts.get("threads", 2)))
    dev = opts.get("device", "cpu")
    torch.set_default_device(dev)
    rec = torch.load(opts["rec"], map_location=dev)
    rule = opts.get("rule", "native")
    seed = int(opts.get("seed", 1))
    gen = torch.Generator(device=dev).manual_seed(seed)
    torch.manual_seed(seed)
    n_src = rec["n_src"]
    pre, w_born = rec["pre"].long(), rec["weight"]
    n_post = pre.shape[0]
    mean_w = float(w_born.mean())
    ratio = int(round(DT / rec["dt_ms"]))
    steps = int(round((rec["rest_ms"] + rec["show_ms"]) / DT))
    n_rest = int(round(rec["rest_ms"] / DT))
    labels, letters = rec["labels"], rec["letters"]
    L = len(letters)
    X = rec["spikes"]  # матрица показа собирается на лету (bins): плотные все — ~3 ГБ

    # Учёба / метки / проверка — свои записи у каждой буквы.
    split = {"train": [], "label": [], "test": []}
    for k in range(L):
        idx = (labels.cpu() == k).nonzero()[:, 0].tolist()
        a, c = len(idx) * 2 // 3, len(idx) * 5 // 6
        split["train"] += idx[:a]
        split["label"] += idx[a:c]
        split["test"] += idx[c:]

    # Матрица весов [пре, пост]; повторные входы — один синапс с суммой и потолком × кратность.
    W = torch.zeros(n_src, n_post)
    mult = torch.zeros(n_src, n_post)
    cols = torch.arange(n_post)[:, None].expand_as(pre)
    W.index_put_((pre, cols), w_born, accumulate=True)
    mult.index_put_((pre, cols), torch.ones_like(w_born), accumulate=True)
    mask = mult > 0
    cap = rec["cap"] * mult
    W0 = W.clone()
    target = W0.sum(0)

    # PV (native, vogels): E:PV 4:1, связи местные — в пределах ±3 столбцов поля.
    field = rec["field_col"].float()
    near = (field - rec["fovea_col"]).abs().argsort()[:20]
    use_pv = rule in ("native", "vogels", "clopath")
    from brain.config import BrainConfig
    cc = BrainConfig()
    cl_ltp_sum, cl_ltd_sum = [0.0], [0.0]
    if use_pv:
        n_pv = n_post // 4
        pv_field = field[torch.randperm(n_post, generator=gen)[:n_pv]]

        def local(dst: torch.Tensor, src_field: torch.Tensor, k: int) -> torch.Tensor:
            d = (dst[:, None] - src_field[None, :]).abs() <= 3
            score = torch.rand(d.shape, generator=gen) * d
            return score.topk(k, 1).indices

        e_from_pv = local(field, pv_field, 117)          # [E, 117]
        pv_from_e = local(pv_field, field, 763)          # [PV, 763]
        P = torch.zeros(n_pv, n_post)                    # PV → E, мВ (модуль торможения)
        P.index_put_((e_from_pv, torch.arange(n_post)[:, None].expand_as(e_from_pv)),
                     torch.full(e_from_pv.shape, 0.21), accumulate=True)
        P_mask = P > 0
        Q = torch.zeros(n_post, n_pv)                    # E → PV, мВ
        Q.index_put_((pv_from_e, torch.arange(n_pv)[:, None].expand_as(pv_from_e)),
                     torch.full(pv_from_e.shape, 0.45), accumulate=True)
    a_plus, a_minus = 0.01, 0.01 * 51.0 / 103.0
    d20, d40, d_pv_m = math.exp(-DT / 20.0), math.exp(-DT / 40.0), math.exp(-DT / 10.0)
    noise = float(opts.get("noise", 2.0))
    eta = float(opts.get("vogels_eta", 0.01))
    alpha = 2.0 * 1.0 * 0.020
    dc_inh = float(opts.get("dc_inh", 1.0))
    bias = torch.zeros(1)
    pv_bias = torch.zeros(1)
    theta = torch.zeros(n_post)
    # ū — средняя деполяризация за ~1 с — живёт между показами (как у мозга).
    cl_ubar = torch.full((n_post,), math.sqrt(cc.clopath_u_ref2))  # как engine: начало — √u_ref²

    pv_hz = [0.0]

    def run(x: torch.Tensor, learn: bool) -> torch.Tensor:
        """Один показ (пауза + буква); счёт спайков клеток за букву."""

        x = bins(x, steps, n_src, ratio)
        v = torch.full((n_post,), V_REST)
        ref = torch.zeros(n_post)
        xp, y1, y2 = torch.zeros(n_src), torch.zeros(n_post), torch.zeros(n_post)
        count = torch.zeros(n_post)
        s = torch.zeros(n_post, dtype=torch.bool)
        if use_pv:
            vp = torch.full((n_pv,), V_REST)
            refp = torch.zeros(n_pv)
            xpv = torch.zeros(n_pv)
            spv = torch.zeros(n_pv, dtype=torch.bool)
            pv_count = torch.zeros(())
        if learn and rule == "dc":
            W.mul_(target / W.sum(0).clamp_min(1e-9))
        if rule == "clopath":
            rest = V_REST + bias
            th_minus = rest + (cc.clopath_theta_minus_mv - cc.clopath_rest_mv)
            u_minus, u_plus = v.clone(), v.clone()
            ubar = cl_ubar
            xbar = torch.zeros(n_src)
            left = torch.zeros(n_post)
            spike_steps = round(cc.clopath_spike_ms / DT)
        for t in range(steps):
            active = x[t].nonzero()[:, 0]
            drive = (W.index_select(0, active) * x[t, active][:, None]).sum(0) if len(active) else 0.0
            inh = 0.0
            fired = s.nonzero()[:, 0]
            if use_pv:
                # Спайки прошлого шага: E → PV, PV → E.
                fired_pv = spv.nonzero()[:, 0]
                inh = P.index_select(0, fired_pv).sum(0) if len(fired_pv) else 0.0
                dpv = Q.index_select(0, fired).sum(0) if len(fired) else 0.0
                vp = vp + (V_REST + pv_bias - vp) * (DT / 10.0) + noise * torch.randn(n_pv) + dpv
                vp = torch.where(refp > 0, torch.full_like(vp, V_RESET), vp)
                spv = vp >= V_TH
                vp[spv] = V_RESET
                refp = torch.where(spv, torch.full_like(refp, REFRACT), (refp - 1).clamp_min(0))
                pv_count += spv.sum()
            elif rule == "dc" and len(fired):
                inh = dc_inh * (len(fired) - s.float())
            v = v + (V_REST + bias - v) * (DT / TAU_M) + noise * torch.randn(n_post) + drive - inh
            v = torch.where(ref > 0, torch.full_like(v, V_RESET), v)
            s = v >= V_TH + theta
            v[s] = V_RESET
            ref = torch.where(s, torch.full_like(ref, REFRACT), (ref - 1).clamp_min(0))
            if t >= n_rest:
                count += s.float()
            if not learn:
                continue
            post = s.nonzero()[:, 0]
            if rule in ("native", "vogels"):
                xp.mul_(d20).add_(x[t])
                if len(active):
                    W[active] -= a_minus * W[active] * y1[None, :]
                if len(post):
                    W[:, post] += (a_plus * mean_w) * xp[:, None] * mask[:, post]
                y1.mul_(d20).add_(s.float())
                if (t + 1) % 50 == 0:
                    W.mul_(target / W.sum(0).clamp_min(1e-9))
                    torch.minimum(W, cap, out=W)
                if rule == "vogels":
                    xpv.mul_(d20).add_(spv.float())
                    k = spv.nonzero()[:, 0]
                    if len(k):
                        P[k] = ((P[k] + eta * (y1[None, :] - alpha)).clamp(0.0, 4.0)) * P_mask[k]
                    if len(post):
                        P[:, post] = ((P[:, post] + eta * xpv[:, None]).clamp(0.0, 4.0)) * P_mask[:, post]
            elif rule == "clopath":
                left = (left - 1).clamp_min(0) + s.float() * spike_steps
                u = torch.where(left > 0, torch.full_like(v, cc.clopath_spike_mv), v)
                u_minus += (u - u_minus) * (DT / cc.clopath_tau_minus_ms)
                u_plus += (u - u_plus) * (DT / cc.clopath_tau_plus_ms)
                ubar += ((u - rest).clamp_min(0.0) - ubar) * (DT / cc.clopath_tau_ubar_ms)
                xbar += (x[t] - xbar * DT) / cc.clopath_tau_x_ms
                if len(active):
                    ltd = cc.clopath_a_ltd * ubar ** 2 / cc.clopath_u_ref2 * (u_minus - th_minus).clamp_min(0.0)
                    down = ltd[None, :] * x[t, active][:, None] * mask[active] * mean_w
                    W[active] = (W[active] - down).clamp_min(0.0)
                    cl_ltd_sum[0] += float(down.sum())
                ltp = cc.clopath_a_ltp * DT * (u - cc.clopath_theta_plus_mv).clamp_min(0.0) * (u_plus - th_minus).clamp_min(0.0)
                hot = (ltp > 0).nonzero()[:, 0]
                if len(hot):
                    up = xbar[:, None] * ltp[None, hot] * mask[:, hot] * mean_w
                    W[:, hot] = torch.minimum(W[:, hot] + up, cap[:, hot])
                    cl_ltp_sum[0] += float(up.sum())
            else:  # dc: следы ставятся в 1
                xp.mul_(d20)
                if len(active):
                    W[active] = (W[active] - (0.001 * mean_w) * y1[None, :] * mask[active]).clamp_min(0.0)
                    xp[active] = 1.0
                if len(post):
                    W[:, post] = torch.minimum(W[:, post] + (0.1 * mean_w) * xp[:, None] * y2[None, post] * mask[:, post],
                                               10.0 * mean_w * mult[:, post])
                y1.mul_(d20)
                y2.mul_(d40)
                y1[post] = 1.0
                y2[post] = 1.0
                theta[post] += 0.05
        if use_pv:
            pv_hz[0] = float(pv_count) / n_pv / (steps * DT / 1000.0)
        return count

    # Сдвиг фона: E — к 1.7 Гц (клетки букв L4 E мозга), PV — к 10 Гц [порядок для PV коры наяву].
    t0 = time.time()
    tr = split["train"]
    mix = torch.tensor(tr)[torch.randperm(len(tr), generator=gen)].tolist()
    for it in range(15):
        c = torch.stack([run(X[mix[(it * 4 + q) % len(mix)]], False) for q in range(4)])
        hz = float(c.mean()) / (rec["show_ms"] / 1000)
        bias += 3.0 * math.log(1.7 / max(hz, 0.05))
        msg = f"  сдвиг {it + 1}: E {hz:.2f} Гц, сдвиг {float(bias):.1f} мВ"
        if use_pv:
            pv_bias += 3.0 * math.log(10.0 / max(pv_hz[0], 0.05))
            msg += f"; PV {pv_hz[0]:.1f} Гц, сдвиг {float(pv_bias):.1f} мВ"
        print(msg, flush=True)

    def measure(n_seen: int) -> None:
        r_lab = torch.stack([run(X[i], False) for i in split["label"]])
        y_lab = labels[split["label"]]
        r_test = torch.stack([run(X[i], False) for i in split["test"]])
        y_test = labels[split["test"]]
        shuffle = y_lab[torch.randperm(len(y_lab), generator=gen)]
        z = lambda a: (a - a.mean(-1, keepdim=True)) / a.std(-1, keepdim=True).clamp_min(1e-9)

        mean_lab = torch.stack([r_lab[y_lab == k].mean(0) for k in range(L)])
        resp = mean_lab.max(0).values > 0
        share = float(torch.bincount(mean_lab.argmax(0)[resp], minlength=L).max()) / max(int(resp.sum()), 1)
        wn = (W[mask] / mean_w)
        wq = torch.quantile(wn[torch.randperm(len(wn), generator=gen)[:1_000_000]], torch.tensor([0.01, 0.5, 0.99]))
        at_cap = float((W[mask] >= 0.99 * cap[mask]).float().mean())

        def acc(ylab: torch.Tensor) -> tuple[float, float]:
            mean = torch.stack([r_lab[ylab == k].mean(0) for k in range(L)])
            lab = mean.argmax(0)
            act = mean.max(0).values > 0
            score = torch.stack([r_test[:, (lab == k) & act].mean(1) if ((lab == k) & act).any()
                                 else torch.full((len(y_test),), -1.0) for k in range(L)], 1)
            corr = z(r_test) @ z(mean).T
            return float((score.argmax(1) == y_test).float().mean()), float((corr.argmax(1) == y_test).float().mean())

        def F(r: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            g = [r[y == k] for k in range(L)]
            return torch.stack([a.mean(0) for a in g]).var(0) / torch.stack([a.var(0) for a in g]).mean(0).clamp_min(1e-9)

        f, f0 = F(r_lab, y_lab), F(r_lab, shuffle)
        sel = float((f > torch.quantile(f0, 0.95)).float().mean())
        (av, ac), (sv, sc) = acc(y_lab), acc(shuffle)
        dw = float(((W - W0).abs() * mask).sum() / W0.sum())
        d = (W - W0)[:, near].T
        d = d[:, d.abs().sum(0) > 0]
        c = (z(d) @ z(d).T) / d.shape[1]
        off = c[~torch.eye(len(near), dtype=torch.bool)]
        extra = f", PV→E среднее {float(P[P_mask].mean()):.2f} мВ" if rule == "vogels" else ""
        extra += f", θ {float(theta.mean()):.2f} мВ" if rule == "dc" else ""
        extra += (f", LTP {cl_ltp_sum[0] / mean_w:.0f} / LTD {cl_ltd_sum[0] / mean_w:.0f} ср. весов, ū {float(cl_ubar.mean()):.1f} мВ"
                  if rule == "clopath" else "")
        extra += (f"; веса 1/50/99% {float(wq[0]):.2f}/{float(wq[1]):.2f}/{float(wq[2]):.2f}, у потолка {at_cap:.2%}, "
                  f"частая буква у {share:.0%} клеток")
        print(f"после {n_seen} показов ({time.time() - t0:.0f} с): точность {av:.2f} / {ac:.2f} "
              f"(перемешано {sv:.2f} / {sc:.2f}), избирательных {sel:.0%}, спайков на показ "
              f"{float(r_test.sum(1).mean()):.0f}, |Δw| {dw:.1%}, корр. Δw у ямки {float(off.median()):.2f}{extra}", flush=True)

    # Потолок входа: корреляция по счёту V1 L2/3 за букву (как letters_dc).
    v1 = lambda ids: torch.stack([bins(X[i], steps, n_src, ratio)[n_rest:].sum(0) for i in ids])
    m = torch.stack([v1(split["label"])[labels[split["label"]] == k].mean(0) for k in range(L)])
    zz = lambda a: (a - a.mean(-1, keepdim=True)) / a.std(-1, keepdim=True).clamp_min(1e-9)
    ceil = float(((zz(v1(split["test"])) @ zz(m).T).argmax(1) == labels[split["test"]]).float().mean())
    print(f"правило {rule}; клеток букв {n_post}, входов {pre.shape[1]}, учебных записей {len(tr)}, "
          f"меток {len(split['label'])}, проверки {len(split['test'])}; V1 L2/3 (корреляция) {ceil:.2f}; "
          f"случайно {1 / L:.2f}", flush=True)
    shows, every = int(opts.get("shows", 3000)), int(opts.get("every", 200))
    measure(0)
    for j in range(shows):
        if j % len(tr) == 0:
            order = torch.tensor(tr)[torch.randperm(len(tr), generator=gen)].tolist()
        run(X[order[j % len(tr)]], True)
        if (j + 1) % every == 0:
            measure(j + 1)


if __name__ == "__main__":
    main(dict(kv.split("=", 1) for kv in sys.argv[1:]))
