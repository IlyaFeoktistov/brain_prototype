"""
Где начинается судорога гиппокампа: покой окнами по 10 мс до первого окна,
где CA3 или CA1 E > порога (50 Гц) hold окон подряд (устойчивый разгон, не
одиночный залп), — затем 300 мс до и 100 мс после:
частоты E/I CA3, CA1, зубчатой, энторинальной, смысловой L2/3; токи E CA3 и
CA1 (возбуждающий i_syn, тормозной i_inh — до движущей силы); доля клеток
CA3 E на предельной частоте (≥ 300 Гц в окне) и сколько клеток CA3 E
стреляет.

    PYTHONPATH=. .venv/bin/python tools/hippo_onset.py [seed=0] [max_s=20] [delays=0] [threshold=50] [hold=3] [warm=2500] [scale=ca3_ee*0.5]

scale=путь*множитель — диагностика: вес пути в этом мозге × множитель
(ca3_ee — коллатерали CA3, ca3_ie / ca1_ie — местное торможение пирамидных).
"""

import sys
from collections import deque

import torch

from brain.config import BrainConfig
from brain.mind import Mind

opts = {"seed": "0", "max_s": "20", "delays": "0", "threshold": "50", "hold": "3", "warm": "2500", "scale": ""}
opts.update(dict(kv.split("=", 1) for kv in sys.argv[1:]))
cfg = BrainConfig()
cfg.save_path = "/nonexistent/brain.pt"
cfg.autosave = False
cfg.anatomy = True
cfg.hand = True
cfg.dt_ms = 0.5
cfg.seed = int(opts["seed"])
cfg.path_delays = opts["delays"] == "1"
torch.manual_seed(cfg.seed)
b = Mind(cfg).brain
b.plasticity = False
# Путь: имя построенного пути, (зона, популяция) источника и получателя —
# среднее поле недостающих синапсов того же пути масштабируется вместе с ним.
names = {"ca3_ee": ("ca3: коллатерали", ("ca3", "E"), ("ca3", "E")),
         "ca3_ie": ("ca3: I->E", ("ca3", "I"), ("ca3", "E")),
         "ca1_ie": ("ca1: I->E", ("ca1", "I"), ("ca1", "E"))}
for item in filter(None, opts["scale"].split(",")):
    key, factor = item.split("*")
    name, (sa, sk), (da, dk) = names[key]
    hit = [p for p in b.projections if p.name == name]
    for p in hit:
        p.weight.mul_(float(factor))
    q = int(b.mf_post_id[getattr(b.areas[da], dk).start])
    pre = int(b.mf_pre_id[getattr(b.areas[sa], sk).start])
    table = b.mf_inh if sk == "I" else b.mf_exc
    table[q, pre] *= float(factor)
    print(f"scale {key} ×{factor}: {[p.name for p in hit]}, среднее поле [{q}, {pre}]")

A = b.areas
pops = {
    "CA3 E": A["ca3"].E, "CA3 I": A["ca3"].I, "CA1 E": A["ca1"].E, "CA1 I": A["ca1"].I,
    "DG E": A["dentate"].E, "EC E": A["entorhinal"].E, "смысл L2/3": A["association"].layers["L23"]["E"],
}
print("пути в CA3 E:", ", ".join(f"{p.name} (k {p.weight.shape[1]}, w̄ {float(p.weight.mean()):+.2f})" for p in b.projections if p.post is A["ca3"].E))
print("пути в CA1 E:", ", ".join(f"{p.name} (k {p.weight.shape[1]}, w̄ {float(p.weight.mean()):+.2f})" for p in b.projections if p.post is A["ca1"].E))
b.run(float(opts["warm"])) if float(opts["warm"]) > 0 else None
history = deque(maxlen=30)
th = float(opts["threshold"])
onset = None
run_len = 0
t = 0.0

def row(r) -> dict:
    out = {k: float(r[p.slice].mean()) for k, p in pops.items()}
    ca3 = A["ca3"].E.slice
    out["CA3 E i_syn"] = float(b.i_syn[ca3].mean())
    out["CA3 E i_inh"] = float(b.i_inh[ca3].mean())
    out["CA1 E i_syn"] = float(b.i_syn[A["ca1"].E.slice].mean())
    out["CA1 E i_inh"] = float(b.i_inh[A["ca1"].E.slice].mean())
    out["CA3 E на пределе"] = float((r[ca3] >= 300).float().mean())
    out["CA3 E стреляют"] = float((r[ca3] > 0).float().mean())
    out["ACh"] = float(b.acetylcholine)
    return out

while t < float(opts["max_s"]) * 1000 and onset is None:
    r = b.observe(10.0)
    t += 10.0
    rec = row(r)
    history.append((t, rec))
    run_len = run_len + 1 if (rec["CA3 E"] > th or rec["CA1 E"] > th) else 0
    if run_len >= int(opts["hold"]):
        onset = history[-int(opts["hold"])][0]

if onset is None:
    print(f"судороги нет за {opts['max_s']} с")
    sys.exit()

after = []
for _ in range(10):
    r = b.observe(10.0)
    t += 10.0
    after.append((t, row(r)))

keys = list(history[0][1].keys())
print(f"начало: {onset:.0f} мс покоя (после разгона {float(opts['warm']) / 1000:.1f} с)")
print("мс      " + " ".join(f"{k[:12]:>12s}" for k in keys))
for when, rec in list(history) + after:
    mark = " ←" if when == onset else ""
    print(f"{when - onset:+6.0f}  " + " ".join(f"{rec[k]:12.2f}" for k in keys) + mark)
