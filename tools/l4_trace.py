"""
Где колонка смысловой L4 теряет различие слов (DESIGN «Межзонный путь A1 →
смысл»): слова голосом, обход ствола (IC выкл. флагом, inh_off — местное
торможение CN и MGN), свежий мозг small. На каждом уровне — вход пути A1
→ смысл (Σ вес × спайки источника), возбуждающий ток i_syn, тормозной
i_inh, мембрана v, спайки — у каждой клетки L4 E среднее за слово
(10-мс отсчёты). Разложение по клеткам: общее (среднее по всем показам),
словоспецифичное (среднее слова минус общее), шум (разброс повторов одного
слова). Печатает: доля словоспецифичного разброса в сумме с шумом (0 —
слово тонет в шуме, 1 — чисто) и различение (сходство повторов против
разных слов).

pv_off=1 — диагностика: торможение PV → E смысловой L4 выключено
(синапсы и среднее поле), входы те же.

    PYTHONPATH=. .venv/bin/python tools/l4_trace.py [words=кот,дом] [reps=4] [pv_off=0] [inh_off=cn,mgn] [seed=0]
"""

import random
import sys

import torch

from brain.body import Cochlea
from brain.config import BrainConfig
from brain.mind import Mind
from brain.world import Voice

opts = {"words": "кот,дом", "reps": "4", "pv_off": "0", "inh_off": "cn,mgn", "seed": "0"}
opts.update(dict(kv.split("=", 1) for kv in sys.argv[1:]))
cfg = BrainConfig()
cfg.save_path = "/nonexistent/brain.pt"
cfg.autosave = False
cfg.anatomy = True
cfg.hand = True
cfg.dt_ms = 0.5
cfg.ic_local_inhibition = False
cfg.seed = int(opts["seed"])
torch.manual_seed(cfg.seed)
b = Mind(cfg).brain
b.plasticity = False

for name in filter(None, opts["inh_off"].split(",")):
    nucleus = b.areas[name]
    for p in b.projections:
        if p.name == f"{name}: I->E":
            p.weight.zero_()
    b.mf_inh[int(b.mf_post_id[nucleus.E.start]), int(b.mf_pre_id[nucleus.I.start])] = 0.0

L4 = b.areas["association"].layers["L4"]
E, PV = L4["E"], L4["PV"]

if opts["pv_off"] == "1":
    for p in b.projections:
        if p.name == "association: L4.PV->L4.E":
            p.weight.zero_()
    b.mf_inh[int(b.mf_post_id[E.start]), int(b.mf_pre_id[PV.start])] = 0.0

i = [s.name for s in b.sensors].index("ear")
ear = Cochlea(cfg, b.sensor_offsets[i], b.sensors[i].size, b.device)
voice = Voice()
path = next(p for p in b.projections if p.name == "auditory -> association")
pre = path.pre.long()
words = opts["words"].split(",")
order = [w for w in words for _ in range(int(opts["reps"]))]
random.Random(cfg.seed).shuffle(order)
b.run(2500.0)
A1 = b.areas["auditory"].layers
levels = {"A1 L4 спайки": [], "A1 L2/3 спайки": [], "вход пути": [], "возб. ток": [], "торм. ток": [], "мембрана": [], "спайки": []}
WIN = 10.0

for word in order:
    b.observe(300.0)
    raster = ear.hear(*voice.say(word))
    steps = int(WIN / cfg.dt_ms)
    acc = {k: 0.0 for k in levels}
    n = 0

    for start in range(0, raster.shape[0] - steps + 1, steps):
        chunk = torch.zeros(steps, b.n_receptors, device=b.device)
        chunk[:, : raster.shape[1]] = raster[start: start + steps]
        r = b.observe(WIN, chunk)
        acc["A1 L4 спайки"] = acc["A1 L4 спайки"] + r[A1["L4"]["E"].slice]
        acc["A1 L2/3 спайки"] = acc["A1 L2/3 спайки"] + r[A1["L23"]["E"].slice]
        acc["вход пути"] = acc["вход пути"] + (path.weight.float() * r[pre]).sum(1)
        acc["возб. ток"] = acc["возб. ток"] + b.i_syn[E.slice].float()
        acc["торм. ток"] = acc["торм. ток"] + b.i_inh[E.slice].float()
        acc["мембрана"] = acc["мембрана"] + b.v[E.slice].float()
        acc["спайки"] = acc["спайки"] + r[E.slice]
        n += 1

    for k in levels:
        levels[k].append((acc[k] / n).cpu())

print(f"слова {words} × {opts['reps']}, PV → E смысловой L4 {'выкл' if opts['pv_off'] == '1' else 'вкл'}")
print(f"{'уровень':14s} {'среднее':>9s} {'слово/(слово+шум)':>18s} {'различение':>11s}")

for k, rows in levels.items():
    R = torch.stack(rows)  # [показ, клетка]
    common = R.mean(0)
    per_word = {w: R[[j for j, x in enumerate(order) if x == w]] for w in words}
    specific = torch.stack([v.mean(0) - common for v in per_word.values()])
    noise = torch.cat([v - v.mean(0, keepdim=True) for v in per_word.values()])
    s_var = float(specific.pow(2).mean())
    n_var = float(noise.pow(2).mean()) * len(order) / max(1, len(order) - len(words))
    X = R - common
    X = X / X.norm(dim=1, keepdim=True).clamp(min=1e-9)
    C = X @ X.t()
    same = [float(C[a, c]) for a in range(len(order)) for c in range(len(order)) if a != c and order[a] == order[c]]
    diff = [float(C[a, c]) for a in range(len(order)) for c in range(len(order)) if order[a] != order[c]]
    print(f"{k:14s} {float(common.mean()):+9.3f} {s_var / (s_var + n_var):18.3f} {sum(same) / len(same) - sum(diff) / len(diff):+11.3f}", flush=True)
