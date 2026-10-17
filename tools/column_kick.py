"""
Одиночная колонка (слух, 20 тыс.) в анатомическом режиме: покой и толчок
L2/3–L6 E 10 и 30 мВ на 100 мс — устойчива ли (возвращается в покой).

    PYTHONPATH=. .venv/bin/python tools/column_kick.py ТОРМОЖЕНИЕ_E ФОН_E [ЗЕРНО] [II=…] [FROM_E=pv,som,vip,lamp5]

Частоты слоя — E/PV/SOM/VIP/LAMP5.
"""

import sys, torch
from brain.config import BrainConfig, default_areas
from brain.engine import SpikingBrain
to_e, ext_e, seed = float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 0
torch.manual_seed(seed)
cfg = BrainConfig(); cfg.layers=True; cfg.eye=False; cfg.compile_step=False; cfg.anatomy=True
cfg.size="20000"; cfg.dt_ms=0.5; cfg.seed = seed
cfg.column_gain_to_e = (to_e,) * 4
for kv in sys.argv[4:]:
    k, v = kv.split("=", 1)
    if k == "II":
        cfg.column_gain_ii = float(v)
    elif k == "FROM_E":
        cfg.column_gain_from_e = tuple(float(x) for x in v.split(","))
cfg.column_gain_external = (ext_e,) + tuple(cfg.column_gain_external[1:])
cfg.areas=[a for a in default_areas() if a.name=="auditory"]; cfg.pathways=[]
b = SpikingBrain(cfg); b.plasticity=False
L = b.areas["auditory"].layers
def ratio() -> str:
    """g_I/g_E клеток E по слоям: ток ÷ движущая сила при нормировке
    (возбуждение — к −70 мВ, 70 мВ до E_AMPA; торможение — к −55, 19 мВ до
    E_ГАМК), со средним полем недостающих соседей и фоном. Сверка: Rudolph
    et al. 2007, бодрствующая кошка — 1.7 ± 1.1 (медиана 1.4)."""
    cfg_ = b.cfg
    nxt = b.stp_u4 + b.stp_U4 * (1 - b.stp_u4)
    gain = torch.where(b.typed_pre[:, None], nxt * b.stp_x4 / b.stp_U4, 1.0)
    gain = torch.where(b.is_lamp5[:, None], gain * b.ngf_d[:, None], gain)
    n = gain.shape[1]
    G = torch.ones(b.mf_size.shape[0], n + 1, device=b.device)
    G[:, :n] = torch.zeros(b.mf_size.shape[0], n, device=b.device).index_add_(0, b.mf_pre_id, gain) / b.mf_size[:, None]
    g = G.index_select(1, b.mf_post_type).t()
    dc_e = b.dc_mv + ((b.mf_exc - b.acetylcholine * b.mf_ach) * g @ b.mf_rate).index_select(0, b.mf_post_id)
    dc_i = b.dc_inh_mv + ((b.mf_inh * g) @ b.mf_rate).index_select(0, b.mf_post_id)
    e = (b.i_syn + b.i_slow + dc_e) / (cfg_.ampa_reversal_mv - cfg_.epsp_holding_mv)
    i = -(b.i_inh + b.i_ngf + b.ngf_mf + dc_i) / (cfg_.ipsp_holding_mv - cfg_.gaba_reversal_mv)
    return " ".join(f"{l}:{float(i[g_['E'].slice].mean() / e[g_['E'].slice].mean().clamp(min=1e-6)):.1f}" for l, g_ in L.items())

def show(tag):
    gi = []
    for _ in range(4):
        b.run(25.0); gi.append(ratio())
    r = b.observe(300.0)
    lamp = torch.cat([g["LAMP5"].slice and b.ngf_d[g["LAMP5"].slice] for g in L.values() if "LAMP5" in g])
    print(f"  {tag:10s} gI/gE {gi[-1]}  сила LAMP5 ×{float(lamp.mean()):.2f} " + " | ".join(f"{l} " + "/".join(f"{float(r[g[k].slice].mean()):.1f}" for k in g) for l,g in L.items()), flush=True)
print(f"торможение E ×{to_e}, тормозные → тормозные ×{cfg.column_gain_ii}, E → PV/SOM/VIP ×{cfg.column_gain_from_e}, фон E ×{ext_e}, зерно {seed}")
b.run(300.0); show("покой")
for mv in (10.0, 30.0):
    for l in L.values(): b.bias[l["E"].slice] += mv
    b.run(100.0)
    for l in L.values(): b.bias[l["E"].slice] -= mv
    b.run(200.0); show(f"толчок {mv:g}")
    b.run(1000.0); show("  +1.6 с")
