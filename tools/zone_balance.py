"""
Откуда фон популяций коры (замер, мозг не меняется; мозг как в letters_dc:
LGN A, гомеостаз homeo заморожен, пустой экран). Вопрос: почему после
правки доставки путей ядер (b0f0dd4) регулятор зон с входом таламуса упёрся
в −10 мВ, а частота выше цели — избыток от таламуса или от своей цепи.

Для каждой популяции (pops=зона.слойТИП,…) — устоявшийся ток сомы, мВ, за
окна по 10 мс (seconds с): по каждому пути Σ вес × сила синапса × частота
источника → ток i_syn (τ_syn) × движущая сила (как engine._step): сила синапса — в начале окна, до его спайков; у путей
ядер — ресурс клетки-источника stp_x, у синапсов колонок — u·x/U следующего
спайка своей пары типов; торможение ×
inh_gain. Пути — по группам: таламус (ядра с депрессией cfg.nucleus_stp),
своя зона, другие зоны; отдельно — фон движка (шум), среднее поле
недостающих соседей (как в движке; по популяциям-источникам), bias вместе со сдвигом
гомеостаза, адаптация. Сверка — собранная сумма против тока движка (i_syn,
i_inh, i_ngf; среднее поле — та же оценка). У таламических путей — частота
и ресурс источника, входов на клетку, средний вес.

    PYTHONPATH=. .venv/bin/python tools/zone_balance.py [pops=…] [seconds=2] [seed=1] [homeo=lgn-d-s1]
"""

from __future__ import annotations

import sys
import time

import torch

from brain.engine import KINDS
from brain.session import Session
from tools.reading_bench import build, observe_pre

WIN = 10.0
POPS = ("prefrontal.L4E,prefrontal.L23E,prefrontal.L5E,goal.L23E,motor.L23E,motor.L4PV,"
        "auditory.L4E,association.L23E,visual.L4E")


def population(b, key: str):
    area, rest = key.split(".")
    for kind in KINDS:
        if rest.endswith(kind) and rest[: -len(kind)] in b.areas[area].layers:
            return b.areas[area].layers[rest[: -len(kind)]][kind], kind
    raise SystemExit(f"нет популяции {key}")


def main(opts: dict) -> None:
    seed = int(opts.get("seed", 1))
    opts = {"fit": "lgn-a", "homeo": f"lgn-d-s{seed}", "relay_v_t_mv": "-59", "trn_scale": "0.1", **opts}
    mind, b = build(opts)
    cfg = b.cfg
    s = Session(mind, window_ms=WIN)
    nuclei = {src for src, *_ in cfg.nucleus_stp}
    lim = cfg.homeostasis_max_mv
    shift = torch.load(__import__("brain.config", fromlist=["BRAIN_HOME"]).BRAIN_HOME / "fits" / f"{opts['homeo']}-homeo.pt")
    targets = {(layer, kind): hz for layer, *v in cfg.column_target_hz for kind, hz in zip(KINDS, v)}

    pops = []
    for key in opts.get("pops", POPS).split(","):
        pop, kind = population(b, key)
        sl = pop.slice
        paths = [p for p in b.projections if p.post is pop and p.source == "net" and p.weight.numel()]
        pops.append((key, pop, kind, sl, paths))

    def source(p) -> str:
        pre = getattr(p, "pre_pop", None)
        return pre.name.split(".")[0] if pre is not None else p.name.split(" -> ")[0].split(".")[0]

    pres = sorted({(pre.start, pre.size) for _, pre, *_ in b.mean_field})
    names_by_start = {(pre.start, pre.size): pre.name for _, pre, *_ in b.mean_field}
    mf_names = {i: names_by_start[k] for i, k in enumerate(pres)}

    def mf_rows(q: int):
        exc = b.mf_exc[q] - b.acetylcholine * b.mf_ach[q]
        inh = b.mf_inh[q]
        nxt = b.stp_u4 + b.stp_U4 * (1.0 - b.stp_u4)
        gain = nxt * b.stp_x4 / b.stp_U4
        gain = torch.where(b.is_lamp5[:, None], gain * b.ngf_d[:, None], gain)
        gain = torch.where(b.typed_pre[:, None], gain, 1.0)
        n = len(KINDS)
        G = torch.ones_like(b.mf_gain)
        G[:, :n] = torch.zeros_like(b.mf_gain[:, :n]).index_add_(0, b.mf_pre_id, gain) / b.mf_size[:, None]
        g = G[:, int(b.mf_post_type[q])]
        exc, inh = exc * g, inh * g
        if b.mf_stp_on:
            resource = torch.zeros_like(b.mf_rate).index_add_(0, b.mf_pre_id, b.stp_x) / b.mf_size
            exc = torch.where(b.mf_stp[q], exc * resource, exc)
        return exc, inh

    def window(view):
        # Сила синапсов — в начале окна, до его спайков (observe_pre).
        r, st = observe_pre(b, WIN, view)
        x0, u0, x40 = st["x"], st["u4"], st["x4"]
        per_step = cfg.dt_ms / 1000.0
        out = {}
        for key, pop, kind, sl, paths in pops:
            v = b.v[sl]
            excite = ((cfg.ampa_reversal_mv - v) / (cfg.ampa_reversal_mv - cfg.epsp_holding_mv)).clamp(min=0.0)
            force = ((v - cfg.gaba_reversal_mv) / (cfg.ipsp_holding_mv - cfg.gaba_reversal_mv)).clamp(min=0.0)
            k_e = b.syn_scale[sl] / (1.0 - b.decay_syn[sl])
            k_i = b.syn_scale_inh[sl] / (1.0 - b.decay_syn_inh[sl])
            post_type = KINDS.index(kind)
            rows = {}
            for p in paths:
                pre = p.pre.long()
                w = p.weight.float()
                if bool((b.stp_u_cell[pre] > 0).any()) and source(p) in nuclei:
                    g = x0[pre]
                else:
                    typed = b.typed_pre[pre]
                    slot = pre * len(KINDS) + post_type
                    u, x, U = (t.view(-1)[slot] for t in (u0, x40, b.stp_U4))
                    g = torch.where(typed, (u + U * (1.0 - u)) * x / U, 1.0)
                # На ветки дендрита (обратные пути, SOM → E; рабочая память →
                # смысл) — не в сому: отдельно, в ток сомы не входят.
                dend = getattr(p, "dendrite", False) or bool((b.dendrite_pre[pre] & b.dendrite_post[pop.start]).any())
                supply = (w * g * r[pre] * per_step).sum(1)
                if dend:
                    rows["~" + p.name] = ((w * g * r[pre] * per_step).sum(1) / (1.0 - b.decay_dend), False)
                    continue
                inh = float(w.mean()) < 0
                cur = supply * b.inh_gain[sl] * k_i * force if inh else supply * k_e * excite
                rows[p.name] = (cur, inh)
            q = int(b.mf_post_id[pop.start])
            noise = b.noise_mean[sl] * b.noise_gain[sl] * k_e * excite
            # Среднее поле — как engine._step: × (1 − ACh), × сила к типу
            # получателя, у путей ядер × средний ресурс источника.
            exc_q, inh_q = mf_rows(q)
            mf_parts = exc_q * b.mf_rate
            mf_e = mf_parts.sum() * excite
            mf_i = (inh_q * b.mf_rate).sum() * force + b.ngf_mf[sl] * force if b.mf_slow_on else (inh_q * b.mf_rate).sum() * force
            e_true = (b.i_syn[sl] + b.dc_mv[sl]) * excite + mf_e
            i_true = (b.i_inh[sl] + b.i_ngf[sl] + b.dc_inh_mv[sl]) * force + mf_i
            res = {name: float(c.mean()) for name, (c, _) in rows.items()}
            for i, val in enumerate((mf_parts * float(excite.mean())).tolist()):
                if abs(val) > 1e-4:
                    res["_mfsrc " + mf_names.get(i, "?")] = val
            res.update({"_noise": float(noise.mean()), "_mf_e": float(mf_e.mean()), "_mf_i": float(mf_i.mean()),
                        "_dc": float((b.dc_mv[sl] * excite).mean()),
                        "_bias": float(b.bias[sl].mean()), "_adapt": float(-b.adapt[sl].mean()),
                        "_e_true": float(e_true.mean()), "_i_true": float(i_true.mean()),
                        "_rate": float(r[sl].mean()), "_v": float(v.mean())})
            for p in paths:
                if source(p) in nuclei:
                    pre = p.pre.long()
                    res["_src_rate " + p.name] = float(r[pre].mean())
                    res["_src_x " + p.name] = float(x0[pre].mean())
            out[key] = res
        return out

    s.line = ""
    b.run(2500.0, s._blank(2500.0))
    t0 = time.time()
    n_win = int(float(opts.get("seconds", 2)) * 1000 / WIN)
    acc: dict = {}
    for _ in range(n_win):
        for key, res in window(s._blank(WIN)).items():
            d = acc.setdefault(key, {})
            for k, val in res.items():
                d[k] = d.get(k, 0.0) + val / n_win
    print(f"гомеостаз {opts['homeo']}, пустой экран {n_win * WIN / 1000:.0f} с, {time.time() - t0:.0f} с счёта")

    for key, pop, kind, sl, paths in pops:
        d = acc[key]
        layer = key.split(".")[1][: -len(kind)]
        sh = shift[sl]
        edge = float((sh <= -lim + 1e-3).float().mean())
        tgt = targets.get((layer, kind), float("nan"))
        print(f"\n=== {key}: {d['_rate']:.2f} Гц (цель {tgt}), мембрана {d['_v']:.1f} мВ, сдвиг гомеостаза "
              f"{float(sh.mean()):+.2f} мВ, у −{lim:.0f} мВ {edge:.0%}")
        groups = {"таламус": 0.0, "своя зона": 0.0, "другие зоны": 0.0}
        area = key.split(".")[0]
        lines = []
        dend_lines = []
        for p in paths:
            src = source(p)
            g = "таламус" if src in nuclei else "своя зона" if src == area else "другие зоны"
            if "~" + p.name in d:
                dend_lines.append((p.name, d["~" + p.name], g))
                continue
            groups[g] += d[p.name]
            lines.append((abs(d[p.name]), d[p.name], g, p))
        for g, val in groups.items():
            print(f"  {g:14s} {val:+8.2f}")
        print(f"  {'фон движка':14s} {d['_noise']:+8.2f}")
        print(f"  {'среднее поле':14s} {d['_mf_e']:+8.2f} возб., {d['_mf_i']:+8.2f} торм.")
        mf = sorted(((k[7:], v) for k, v in d.items() if k.startswith("_mfsrc ")), key=lambda kv: -abs(kv[1]))
        print("    по источникам: " + ", ".join(f"{k} {v:+.2f}" for k, v in mf[:6]))
        print(f"  {'bias+гомеост.':14s} {d['_bias']:+8.2f}   адаптация {d['_adapt']:+.2f}   постоянный dc {d['_dc']:+.2f}")
        e_sum = sum(v for _, v, _, _ in lines if v > 0) + d["_noise"] + d["_mf_e"] + d["_dc"]
        i_sum = sum(v for _, v, _, _ in lines if v < 0) + d["_mf_i"]
        print(f"  сверка: возб. собрано {e_sum:+.2f} / движок {d['_e_true']:+.2f}; торм. {i_sum:+.2f} / {d['_i_true']:+.2f}")
        if dend_lines:
            print("  на дендрит (не в сому; ток ветки, мВ): " + ", ".join(f"{n} {v:+.2f} [{g}]" for n, v, g in dend_lines))
        print("  пути (по |току|):")
        for _, val, g, p in sorted(lines, key=lambda t: -t[0])[:10]:
            extra = ""
            if g == "таламус":
                extra = (f"  источник {d['_src_rate ' + p.name]:.1f} Гц, ресурс {d['_src_x ' + p.name]:.2f}, "
                         f"{p.weight.shape[1]} вх., вес {float(p.weight.float().mean()):+.3f} мВ")
            print(f"    {p.name:38s} {val:+8.2f}  [{g}]{extra}")


if __name__ == "__main__":
    main(dict(kv.split("=", 1) for kv in sys.argv[1:]))
