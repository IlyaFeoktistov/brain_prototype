"""
Пересборка колонки по данным: одиночная колонка (слух, 20 тыс., шаг 0.5 мс)
в анатомическом режиме — покой, толчок E и возврат, по одному механизму за
прогон. Частоты слоя — E/PV/SOM/VIP/LAMP5, g_I/g_E клеток E по слоям (как в
column_kick; сверка — Rudolph et al. 2007, бодрствующая кошка 1.7 ± 1.1).

    PYTHONPATH=. .venv/bin/python data/tools/column_check.py [ee=0] [параметр=значение …]

ee=0 — без рекуррентного E → E внутри колонки (построенные синапсы и среднее
поле недостающих): нужна ли для разгона петля или он уже во входных токах.
Параметры — поля BrainConfig (кортеж — как в Python: "(1,1,1,1)").
"""

import sys

import torch

from brain.config import BrainConfig, default_areas
from brain.engine import SpikingBrain, KINDS

opts = {"ee": "1", "seed": "0", "kicks": "10,30"}
cfg = BrainConfig()
cfg.layers = True
cfg.eye = False
cfg.compile_step = False
cfg.anatomy = True
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
cfg.areas = [a for a in default_areas() if a.name == "auditory"]
cfg.pathways = []
b = SpikingBrain(cfg)
b.plasticity = False
L = b.areas["auditory"].layers

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
    e = (b.i_syn + b.i_slow + dc_e) / (cfg.ampa_reversal_mv - cfg.epsp_holding_mv)
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

for mv in (float(x) for x in opts["kicks"].split(",")):
    for g in L.values():
        b.bias[g["E"].slice] += mv
    b.run(100.0)
    for g in L.values():
        b.bias[g["E"].slice] -= mv
    show(f"толчок {mv:g}")
    b.run(1000.0)
    show("  +1.4 с")
