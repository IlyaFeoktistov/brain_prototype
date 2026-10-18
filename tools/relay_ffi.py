"""
Прямое торможение реле MGN, как в опыте на срезе LGN (Blitz & Regehr 2005;
eNeuro 2024 — IPSC/EPSC реле при максимальной стимуляции тракта, медиана 0.53):
залп — все главные клетки IC разом (толчок на один шаг), на реле MGN —
возбуждающий (i_syn) и тормозной (i_inh) токи по шагам за 40 мс, среднее по
залпам за вычетом фона перед залпом. Токи — до множителя движущей силы, то
есть при «фиксации» у потенциалов нормировки (как voltage clamp). Печатает
начало, пик, интеграл и отношение пиков IPSC/EPSC; отдельно — что дают
интернейроны (mgn: I->E) и TRN (через выключение одного пути на прогон).
В опыте ВПСТ мерили при −70 мВ, ТПСТ при 0 мВ — сравнивать надо отношение
проводимостей (печатается).

    PYTHONPATH=. .venv/bin/python tools/relay_ffi.py [fit=ic-off] [local=0] [off=] [volleys=20] [kick_mv=60] [kick_ms=1] [every=4] [seed=0]

off=mgn — без интернейронов → реле, off=trn — без TRN → реле.
"""

import sys

import torch

from brain import anatomy
from brain.config import BrainConfig
from brain.mind import Mind

opts = {"fit": "ic-off", "local": "0", "off": "", "volleys": "20", "kick_mv": "60", "kick_ms": "1", "seed": "0"}
opts.update(dict(kv.split("=", 1) for kv in sys.argv[1:]))
cfg = BrainConfig()
cfg.save_path = "/nonexistent/brain.pt"
cfg.autosave = False
cfg.anatomy = True
cfg.hand = True
cfg.dt_ms = 0.5
cfg.seed = int(opts["seed"])
cfg.ic_local_inhibition = False
cfg.relay_local_inhibition = opts["local"] == "1"
if opts["fit"]:
    anatomy.use_fit(cfg, opts["fit"])
torch.manual_seed(cfg.seed)
b = Mind(cfg).brain
b.plasticity = False
ic, mgn = b.areas["ic"], b.areas["mgn"]
names = {"mgn": "mgn: I->E", "trn": "trn_mgn -> mgn"}
for key in filter(None, opts["off"].split(",")):
    for p in b.projections:
        if p.name == names[key]:
            p.weight.zero_()
    if key == "mgn":
        b.mf_inh[int(b.mf_post_id[mgn.E.start]), int(b.mf_pre_id[mgn.I.start])] = 0.0

relay = mgn.E.slice
steps = int(40 / cfg.dt_ms)
pre_steps = int(10 / cfg.dt_ms)
ex = torch.zeros(steps, device=b.device)
inh = torch.zeros(steps, device=b.device)
fired = []
b.observe(2500)

for _ in range(int(opts["volleys"])):
    b.observe(300)
    base_e, base_i = 0.0, 0.0
    for _ in range(pre_steps):
        b.observe(cfg.dt_ms)
        base_e += float(b.i_syn[relay].mean()) / pre_steps
        base_i += float(b.i_inh[relay].mean()) / pre_steps
    b.kick("ic", torch.arange(ic.E.size, device=b.device), float(opts["kick_mv"]), float(opts["kick_ms"]))
    for t in range(steps):
        r = b.observe(cfg.dt_ms)
        if t < 4:
            fired.append(float((r[ic.E.slice] > 0).float().mean()))
        ex[t] += (float(b.i_syn[relay].mean()) - base_e) / int(opts["volleys"])
        inh[t] += (float(b.i_inh[relay].mean()) - base_i) / int(opts["volleys"])


def onset(x: torch.Tensor) -> float:
    """Время до половины пика: порог 10% ловил фон между залпами."""

    peak = float(x.abs().max())
    return float((x.abs() >= 0.5 * peak).nonzero()[0]) * cfg.dt_ms


pe, pi = float(ex.max()), float(-inh.min())
print(f"залп: главных IC стреляло за 2 мс {sum(fired) / len(fired) * 4 * 100:.0f}% (средн. доля за шаг × 4)")
print(f"ВПСТ реле: половина пика {onset(ex):.1f} мс, пик {pe:.2f} на {float(ex.argmax()) * cfg.dt_ms:.1f} мс, интеграл {float(ex.sum()) * cfg.dt_ms:.1f}")
print(f"ТПСТ реле: половина пика {onset(inh):.1f} мс, пик {pi:.2f} на {float(inh.argmin()) * cfg.dt_ms:.1f} мс, интеграл {float(-inh.sum()) * cfg.dt_ms:.1f}")
print(f"ТПСТ / ВПСТ: пики {pi / max(pe, 1e-9):.2f}, интегралы {float(-inh.sum()) / max(float(ex.sum()), 1e-9):.2f}; задержка ТПСТ после ВПСТ {onset(inh) - onset(ex):+.1f} мс")
# В опыте ВПСТ — при −70 мВ, ТПСТ — при 0 мВ: движущая сила у обоих ~70 мВ,
# отношение токов = отношение проводимостей. У нас ток торможения нормирован к
# −55 мВ (19 мВ до реверсии ГАМК), возбуждения — к −70 (70 мВ до AMPA).
g_ratio = (pi / (cfg.ipsp_holding_mv - cfg.gaba_reversal_mv)) / max(pe / (cfg.ampa_reversal_mv - cfg.epsp_holding_mv), 1e-9)
print(f"отношение проводимостей gI/gE по пикам {g_ratio:.2f} (опыт, LGN мыши, интернейроны: 0.53, IQR 0.42–0.73)")
print("по шагам (мс: возб / торм): " + " ".join(f"{t * cfg.dt_ms:.0f}:{float(ex[t]):+.1f}/{float(inh[t]):+.1f}" for t in range(0, steps, int(opts.get("every", "4")))))
