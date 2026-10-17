"""
Долгий замер анатомического мозга по окнам 100 мс: гиппокамп, L2/3 E/PV
смысла, ПФК, слуха, L4 PV моторной, таламус, MD — что перескакивает.

    PYTHONPATH=. .venv/bin/python tools/anatomy_timeline.py
"""

import torch
from brain.config import BrainConfig
from brain.mind import Mind
torch.manual_seed(0)
cfg = BrainConfig(); cfg.save_path="/nonexistent/b.pt"; cfg.autosave=False
cfg.anatomy=True; cfg.hand=True; cfg.size="small"; cfg.dt_ms=0.5
b = Mind(cfg).brain; b.plasticity=False; b.run(200.0)
cols = [("CA3", "ca3", None), ("CA1", "ca1", None), ("DG", "dentate", None), ("EC", "entorhinal", None),
        ("смыслE", "association", ("L23","E")), ("смыслPV", "association", ("L23","PV")),
        ("ПФК_E", "prefrontal", ("L23","E")), ("ПФК_PV", "prefrontal", ("L23","PV")),
        ("слухE", "auditory", ("L23","E")), ("слухPV", "auditory", ("L23","PV")),
        ("motL4PV", "motor", ("L4","PV")), ("таламус", "thalamus", None), ("MD", "md", None)]
print("мс   " + " ".join(f"{c:>7s}" for c, _, _ in cols))
for w in range(30):
    r = b.observe(100.0)
    vals = []
    for _, a, l in cols:
        s = b.areas[a].E.slice if l is None else b.areas[a].layers[l[0]][l[1]].slice
        vals.append(float(r[s].mean()))
    print(f"{(w+1)*100:4d} " + " ".join(f"{v:7.1f}" for v in vals), flush=True)
