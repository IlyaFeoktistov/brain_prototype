"""
Пересборка колонки по данным: одиночная колонка (слух, вход уха выключен —
ear=1 вернуть; 20 тыс., шаг 0.5 мс)
в анатомическом режиме — покой, толчок E и возврат, по одному механизму за
прогон. Частоты слоя — E/PV/SOM/VIP/LAMP5, g_I/g_E клеток E по слоям (как в
column_kick; сверка — Rudolph et al. 2007, бодрствующая кошка 1.7 ± 1.1).

    PYTHONPATH=. .venv/bin/python tools/column_check.py [ee=0] [параметр=значение …]

ach=0.75 — ацетилхолин бодрствования (занятость рецепторов в целом мозге
наяву ~0.75): ток рецепторов anatomy.RECEPTORS — всем клеткам зоны и своим
типам (VIP, SOM) — смещением (в стенде нет ядер-модуляторов).
fit=N — подбор только внешнего фона по популяциям (слой × тип) под
column_target_hz: N итераций по 1 с, фон × (цель / частота)^0.3 (в моделях
по данным фон извне подбирают под частоты in vivo — Billeh et al. 2020);
связи не трогаются. Печатает множители (для column_gain_external_pop).
inputs=L4.E,L23.VIP — после покоя разложить входы популяций: среднее поле
недостающих по источнику (мВ), построенные пути (Σ вес·Гц), фон, смещение.
ee=0 — без рекуррентного E → E внутри колонки (построенные синапсы и среднее
поле недостающих): нужна ли для разгона петля или он уже во входных токах.
Параметры — поля BrainConfig (кортеж — как в Python: "(1,1,1,1)").
"""

import math
import sys

import torch

from brain import anatomy
from brain.config import BrainConfig
from brain.engine import SpikingBrain, KINDS

opts = {"ee": "1", "seed": "0", "kicks": "10,30", "ach": "0", "fit": "0", "inputs": "", "area": "auditory", "ear": "0"}
cfg = BrainConfig()
cfg.compile_step = False
cfg.size = "20000"
cfg.dt_ms = 0.5

for kv in sys.argv[1:]:
    k, v = kv.split("=", 1)

    if k in opts:
        opts[k] = v
        continue

    old = getattr(cfg, k)
    setattr(cfg, k, (v == "1") if isinstance(old, bool) else eval(v) if isinstance(old, tuple) else type(old)(v))

cfg.seed = int(opts["seed"])
torch.manual_seed(cfg.seed)
cfg.areas = [a for a in anatomy.areas(cfg) if a.name == opts["area"]]
cfg.pathways = []
b = SpikingBrain(cfg)
b.plasticity = False
L = b.areas[opts["area"]].layers

if opts["ear"] == "0":
    # Рецепторы уха идут прямо в L4 (реле MGN в одиночной колонке нет) —
    # L4 E стреляли 12 Гц от уха, а не от колонки. ear=1 — оставить.
    b.ear_weight.zero_()

if opts["ee"] == "0":
    E = {g["E"].start for g in L.values()}

    for p in b.projections:
        if getattr(p, "post", None) is not None and p.post.start in E and p.pre.numel() and int(p.pre.flatten()[0]) in range(min(E), max(g["E"].stop for g in L.values())):
            if all(any(g["E"].start <= int(x) < g["E"].stop for g in L.values()) for x in p.pre.flatten()[:8].tolist()):
                p.weight.zero_()

    e_ids = sorted({int(b.mf_pre_id[s]) for s in E})
    q_ids = sorted({int(b.mf_post_id[s]) for s in E})

    for q in q_ids:
        b.mf_exc[q, e_ids] = 0.0


occupancy = float(opts["ach"])

if occupancy > 0:
    from brain import anatomy

    for substance, target, mv in anatomy.RECEPTORS:
        if substance != "acetylcholine" or (target and not target.startswith(":")):
            continue

        for g in L.values():
            for kind, pop in g.items():
                if not target or target[1:] == kind:
                    b.bias[pop.slice] += mv * occupancy


def ratio() -> str:
    nxt = b.stp_u4 + b.stp_U4 * (1 - b.stp_u4)
    gain = torch.where(b.typed_pre[:, None], nxt * b.stp_x4 / b.stp_U4, 1.0)
    gain = torch.where(b.is_lamp5[:, None], gain * b.ngf_d[:, None], gain)
    n = gain.shape[1]
    G = torch.ones(b.mf_size.shape[0], n + 1, device=b.device)
    G[:, :n] = torch.zeros(b.mf_size.shape[0], n, device=b.device).index_add_(0, b.mf_pre_id, gain) / b.mf_size[:, None]
    g = G.index_select(1, b.mf_post_type).t()
    dc_e = b.dc_mv + ((b.mf_exc - b.acetylcholine * b.mf_ach) * g @ b.mf_rate).index_select(0, b.mf_post_id)
    dc_i = b.dc_inh_mv + ((b.mf_inh * g) @ b.mf_rate).index_select(0, b.mf_post_id)
    e = (b.i_syn + dc_e) / (cfg.ampa_reversal_mv - cfg.epsp_holding_mv)
    i = -(b.i_inh + b.i_ngf + b.ngf_mf + dc_i) / (cfg.ipsp_holding_mv - cfg.gaba_reversal_mv)
    return " ".join(f"{l}:{float(i[g_['E'].slice].mean() / e[g_['E'].slice].mean().clamp(min=1e-6)):.1f}" for l, g_ in L.items())


def show(tag: str) -> None:
    b.run(100.0)
    gi = ratio()
    r = b.observe(300.0)
    print(f"  {tag:9s} gI/gE {gi} | " + " | ".join(
        f"{l} " + "/".join(f"{float(r[g[k].slice].mean()):.1f}" for k in KINDS if k in g) for l, g in L.items()), flush=True)


print(f"ee={opts['ee']} " + " ".join(sys.argv[1:]))
b.run(1500.0)
show("покой")
names = {int(b.mf_pre_id[pop.start]): f"{l}.{k}" for l, g in L.items() for k, pop in g.items()}

for spec in filter(None, opts["inputs"].split(",")):
    l, k = spec.split(".")
    pop = L[l][k]
    r = b.observe(500.0)
    q = int(b.mf_post_id[pop.start])
    parts = []

    for label, M in (("E", b.mf_exc), ("I", b.mf_inh), ("медл", b.mf_slow)):
        c = M[q] * b.mf_rate
        top = c.abs().topk(4).indices
        parts.append(f"{label} {float(c.sum()):+.1f}: " + ", ".join(f"{names.get(int(j), j)} {float(c[j]):+.1f}" for j in top if float(c[j]) != 0))

    built = []

    for p in b.projections:
        if getattr(p, "post", None) is not None and p.post.start == pop.start and p.pre.numel():
            built.append((p.name, float((p.weight.float() * r[p.pre.long()]).sum(1).mean())))

    print(f"  входы {spec} ({float(r[pop.slice].mean()):.1f} Гц): " + " | ".join(parts))
    print("     пути: " + ", ".join(f"{n.split(': ')[-1]} {v:+.0f}" for n, v in sorted(built, key=lambda x: -abs(x[1]))[:6]))
    print(f"     фон: {float(b.noise_mean[pop.slice].mean()):.2f} событий/шаг × {float(b.noise_gain[pop.slice].mean()):.3f}, смещение {float(b.bias[pop.slice].mean()):+.2f} мВ, порог {float((cfg.v_threshold - cfg.v_rest)):.0f}")

targets = {layer: dict(zip(KINDS, v)) for layer, *v in cfg.column_target_hz}
scale = {(l, k): 1.0 for l, g in L.items() for k in g}

for step in range(int(opts["fit"])):
    r = b.observe(1000.0)
    err = 0.0

    for l, g in L.items():
        for k, pop in g.items():
            hz = float(r[pop.slice].mean())
            factor = min(1.5, max(1 / 1.5, ((targets[l][k] + 0.5) / (hz + 0.5)) ** 0.3))
            scale[(l, k)] *= factor
            b.noise_gain[pop.slice] *= factor
            err += math.log((hz + 0.5) / (targets[l][k] + 0.5)) ** 2

    print(f"  подбор {step:2d}: ошибка {err / len(scale):.3f}", flush=True)

if int(opts["fit"]):
    show("после")
    print("  фон ×", " ".join(f"{l}.{k}={v:.3f}" for (l, k), v in scale.items()))

for mv in (float(x) for x in filter(None, opts["kicks"].split(","))):
    for g in L.values():
        b.bias[g["E"].slice] += mv
    b.run(100.0)
    for g in L.values():
        b.bias[g["E"].slice] -= mv
    show(f"толчок {mv:g}")
    b.run(1000.0)
    show("  +1.4 с")
