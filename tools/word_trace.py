"""
Где слово превращается в представление: слова голосом (world.Voice →
body.Cochlea), каждое reps раз в перемешанном порядке, свежий мозг small
(по умолчанию — диагностический режим: ic_local_inhibition выкл., DESIGN
«Временный обход»). По звеньям — прибавка частоты на слово (слово + after
мс против тишины до него), воспроизводимость (сходство узоров прибавки на
повторы одного слова), различимость (одно слово против разных; отдельно —
похожая пара pair=), задержка (первое 10-мс окно после начала слова, где
частота звена выше тишины на 3 σ окон тишины).

pv_off=association.L4 — диагностика: торможение PV → E этого слоя зоны
выключено (синапсы и среднее поле). chain=1 — звенья до выхода (смысл L5,
ПФК L4/L2/3/L5, премотор, стриатум, рука).
inh_off=cn,mgn — ещё и местное торможение главных клеток этих ядер выключено
(диагностика: на широкополосной речи гаснут CN и MGN, DESIGN).

    PYTHONPATH=. .venv/bin/python tools/word_trace.py [words=кот,кит,дом,мама] [pair=кот,кит] [reps=4] [after=100] [ic=0] [inh_off=] [seed=0]
"""

import random
import sys

import torch

from brain.body import Cochlea
from brain.config import BrainConfig
from brain.mind import Mind
from brain.world import Voice

opts = {"pv_off": "", "chain": "0", "inh_off": "", "words": "кот,кит,дом,мама", "pair": "кот,кит", "reps": "4", "after": "100", "ic": "0", "seed": "0"}
opts.update(dict(kv.split("=", 1) for kv in sys.argv[1:]))
cfg = BrainConfig()
cfg.save_path = "/nonexistent/brain.pt"
cfg.autosave = False
cfg.anatomy = True
cfg.hand = True
cfg.dt_ms = 0.5
cfg.ic_local_inhibition = opts["ic"] == "1"
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

for spec in filter(None, opts["pv_off"].split(",")):
    area, layer = spec.split(".")
    g = b.areas[area].layers[layer]
    for p in b.projections:
        if p.name == f"{area}: {layer}.PV->{layer}.E" or (layer == "L23" and p.name == f"{area}: I->E"):
            p.weight.zero_()
    b.mf_inh[int(b.mf_post_id[g["E"].start]), int(b.mf_pre_id[g["PV"].start])] = 0.0

i = [s.name for s in b.sensors].index("ear")
ear = Cochlea(cfg, b.sensor_offsets[i], b.sensors[i].size, b.device)
voice = Voice()
L = {a: b.areas[a].layers for a in ("auditory", "association", "prefrontal")}
stages = {
    "A1 L4": L["auditory"]["L4"]["E"], "A1 L2/3": L["auditory"]["L23"]["E"],
    "смысл L4": L["association"]["L4"]["E"], "смысл L2/3": L["association"]["L23"]["E"],
    "ПФК L2/3": L["prefrontal"]["L23"]["E"],
}

if opts["chain"] == "1":
    stages = {
        "A1 L4": L["auditory"]["L4"]["E"], "A1 L2/3": L["auditory"]["L23"]["E"],
        "смысл L4": L["association"]["L4"]["E"], "смысл L2/3": L["association"]["L23"]["E"],
        "смысл L5": L["association"]["L5"]["E"],
        "ПФК L4": L["prefrontal"]["L4"]["E"], "ПФК L2/3": L["prefrontal"]["L23"]["E"], "ПФК L5": L["prefrontal"]["L5"]["E"],
        "премотор L2/3": b.areas["premotor"].layers["L23"]["E"], "стриатум D1": b.areas["striatum_d1"].E,
        "рука L2/3": b.areas["hand"].layers["L23"]["E"],
    }
words = opts["words"].split(",")
order = [w for w in words for _ in range(int(opts["reps"]))]
random.Random(cfg.seed).shuffle(order)
WIN = 10.0
b.run(1500.0)
gains = {k: [] for k in stages}
latency = {k: [] for k in stages}

for word in order:
    quiet = [b.observe(WIN) for _ in range(30)]
    raster = ear.hear(*voice.say(word))
    steps = int(WIN / cfg.dt_ms)
    loud = []

    for start in range(0, raster.shape[0] - steps + 1, steps):
        chunk = torch.zeros(steps, b.n_receptors, device=b.device)
        chunk[:, : raster.shape[1]] = raster[start: start + steps]
        loud.append(b.observe(WIN, chunk))

    loud += [b.observe(WIN) for _ in range(int(float(opts["after"]) / WIN))]
    q, s = torch.stack(quiet), torch.stack(loud)

    for k, pop in stages.items():
        base = q[:, pop.slice].mean(1)
        trace = s[:, pop.slice].mean(1)
        gains[k].append(s[:, pop.slice].mean(0) - q[:, pop.slice].mean(0))
        over = (trace > base.mean() + 3 * base.std().clamp(min=1e-3)).nonzero()
        latency[k].append(float(over[0]) * WIN + WIN if len(over) else float("nan"))


def similarity(rows: list[torch.Tensor], a: str, c: str) -> float:
    R = torch.stack(rows).float().cpu()
    R = R - R.mean(0, keepdim=True)
    R = R / R.norm(dim=1, keepdim=True).clamp(min=1e-6)
    C = R @ R.t()
    ia = [j for j, w in enumerate(order) if w == a]
    ic = [j for j, w in enumerate(order) if w == c]
    vals = [float(C[x, y]) for x in ia for y in ic if x != y]
    return sum(vals) / len(vals)


a, c = opts["pair"].split(",")
print(f"слова {words} × {opts['reps']}, IC местное торможение {'вкл' if cfg.ic_local_inhibition else 'выкл (обход)'}, ещё выкл: {opts['inh_off'] or '—'}")
print(f"{'звено':14s} {'прибавка':>9s} {'повтор':>7s} {'разные':>7s} {'различ.':>8s} {a + '/' + c:>9s}  задержка")

for k in stages:
    R = gains[k]
    same = sum(similarity(R, w, w) for w in words) / len(words)
    diff = sum(similarity(R, x, y) for x in words for y in words if x != y) / (len(words) * (len(words) - 1))
    pair = similarity(R, a, c)
    lat = torch.tensor(latency[k])
    lat = lat[~lat.isnan()]
    when = f"{float(lat.median()):.0f} мс ({len(lat)}/{len(order)})" if len(lat) else "—"
    print(f"{k:14s} {float(torch.stack(R).mean()):+9.2f} {same:7.3f} {diff:7.3f} {same - diff:+8.3f} {pair:9.3f}  {when}", flush=True)
