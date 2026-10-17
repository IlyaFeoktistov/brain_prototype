"""
Сетка сил классов связей колонок — покой против ответа на слово. Мозг
собирается один раз; сила класса (тормозные → E, E → PV) меняется в живом
мозге: веса местных путей колонок и коэффициенты среднего поля × множитель.
Для каждой точки: частоты E в покое по слоям (среднее зон) и ответ / покой у
клеток, чьё поле на слове: LGN, V1 L4, V1 L2/3 (их входы L4 — на слове),
буквы L4 (входы из V1 L2/3 «на слове»). In vivo ответ на контрастный стимул
в поле — в разы выше спонтанной (LGN, простые и сложные клетки V1).

    PYTHONPATH=. .venv/bin/python tools/response_grid.py [to_e=3,2,1.5] [pv=1,0.66,0.44] [word=мама]
"""

import sys
import time

import torch

from brain.body import Eye
from brain.config import BrainConfig
from brain.engine import KINDS
from brain.mind import Mind
from brain.world import Screen

opts = {"to_e": "1,0.66,0.5", "pv": "1,0.66,0.44", "word": "мама", "settle": "1000", "rest": "1000", "ms": "300"}

for kv in sys.argv[1:]:
    k, v = kv.split("=", 1)
    opts[k] = v

torch.manual_seed(0)
cfg = BrainConfig()
cfg.save_path = "/nonexistent/brain.pt"
cfg.autosave = False
cfg.anatomy = True
cfg.hand = True
cfg.dt_ms = 0.5
start = time.time()
b = Mind(cfg).brain
b.plasticity = False
i = [s.name for s in b.sensors].index("eye")
eye = Eye(cfg, b.sensor_offsets[i], b.sensors[i].size, b.device)
screen = Screen(cfg.retina_rows, cfg.retina_cols)
blank, word = screen.picture(""), screen.picture(opts["word"])
print(f"мозг собран за {time.time() - start:.0f} с", flush=True)

# Зона каждой клетки (для «местных» путей колонки — пре и пост в одной зоне).
zone_of = torch.full((b.n_slots,), -1, dtype=torch.long, device=b.device)
column_zones = [a for a in b.areas.values() if a.layers is not None]

for z, area in enumerate(column_zones):
    zone_of[area.cells] = z

ctype = b.cell_type


def classes(pre_kinds, post_kinds):
    """Местные проекции колонок и элементы среднего поля данного класса."""

    projs = []

    for p in b.projections:
        if p.source != "net" or p.pre.numel() == 0:
            continue

        pre0 = int(p.pre.flatten()[0])
        post0 = p.post.start

        if zone_of[pre0] < 0 or zone_of[pre0] != zone_of[post0]:
            continue

        if int(ctype[pre0]) in pre_kinds and int(ctype[post0]) in post_kinds:
            projs.append(p)

    posts = sorted({(x.start, x.size) for x, *_ in b.mean_field})
    pres = sorted({(x.start, x.size) for _, x, *_ in b.mean_field})
    cells = []

    for q, (qs, _) in enumerate(posts):
        for j, (ps, _) in enumerate(pres):
            if zone_of[qs] >= 0 and zone_of[qs] == zone_of[ps] and int(ctype[ps]) in pre_kinds and int(ctype[qs]) in post_kinds:
                cells.append((q, j))

    return projs, cells


inh = [KINDS.index(k) for k in ("PV", "SOM", "VIP")]
to_e = classes(inh, [0])
e_pv = classes([0], [KINDS.index("PV")])
print(f"тормозные → E: {len(to_e[0])} путей, {len(to_e[1])} элементов поля; E → PV: {len(e_pv[0])}, {len(e_pv[1])}", flush=True)
current = {"to_e": 1.0, "pv": 1.0}


def scale(cls, factor: float) -> None:
    projs, cells = cls

    for p in projs:
        p.weight.mul_(factor)

    for q, j in cells:
        b.mf_exc[q, j] *= factor
        b.mf_inh[q, j] *= factor
        b.mf_ach[q, j] *= factor


def set_point(te: float, pv: float) -> None:
    scale(to_e, te / current["to_e"])
    scale(e_pv, pv / current["pv"])
    current.update(to_e=te, pv=pv)


# Клетки «на слове» по ступеням.
lgn = b.areas["lgn"].E
seen = (eye.rates(word) >= 5.0).nonzero().flatten().to(b.device) + eye.offset
drv = next(p for p in b.projections if p.name == "eye -> " + lgn.name)
lgn_on = torch.isin(drv.pre[:, 0], seen)
V = b.areas["visual"].layers
tc = next(p for p in b.projections if p.name == f"lgn -> {V['L4']['E'].name}")
l4_on = lgn_on[(tc.pre.long() - lgn.start)].float().mean(1) > 0.5
l4_cells = torch.arange(V["L4"]["E"].start, V["L4"]["E"].stop, device=b.device)
on_l4 = torch.zeros(b.n_slots, dtype=torch.bool, device=b.device)
on_l4[l4_cells[l4_on]] = True
p23 = next(p for p in b.projections if p.post.start == V["L23"]["E"].start and p.source == "net"
           and int(p.pre.flatten()[0]) >= V["L4"]["E"].start and int(p.pre.flatten()[0]) < V["L4"]["E"].stop)
l23_on = on_l4[p23.pre.long()].float().mean(1) > 0.3
l23_cells = torch.arange(V["L23"]["E"].start, V["L23"]["E"].stop, device=b.device)
on_l23 = torch.zeros(b.n_slots, dtype=torch.bool, device=b.device)
on_l23[l23_cells[l23_on]] = True
Lt = b.areas["letters"].layers["L4"]["E"]
pv1 = next(p for p in b.projections if p.name == "visual -> letters")
lt_on = on_l23[pv1.pre.long()].float().mean(1) > on_l23[l23_cells].float().mean() * 2
lt_cells = torch.arange(Lt.start, Lt.stop, device=b.device)[lt_on[: Lt.size]] if pv1.post.start == Lt.start else None
groups = {"LGN": torch.arange(lgn.start, lgn.stop, device=b.device)[lgn_on], "V1 L4": l4_cells[l4_on],
          "V1 L2/3": l23_cells[l23_on]}

if lt_cells is not None and len(lt_cells):
    groups["буквы L4"] = lt_cells

print("клеток на слове: " + ", ".join(f"{k} {len(v)}" for k, v in groups.items()), flush=True)
layers = ("L23", "L4", "L5", "L6")
print(f"{'торм→E':>6s} {'E→PV':>5s} | E покой " + " ".join(f"{l:>5s}" for l in layers) + " | ответ/покой "
      + " ".join(f"{k:>9s}" for k in groups) + " | макс популяция", flush=True)

for te in [float(x) for x in opts["to_e"].split(",")]:
    for pv in [float(x) for x in opts["pv"].split(",")]:
        set_point(te, pv)

        for name in b.areas:
            b.silence(name)

        b.observe(float(opts["settle"]), eye.look(blank, float(opts["settle"])))
        rest = b.observe(float(opts["rest"]), eye.look(blank, float(opts["rest"])))
        resp = b.observe(float(opts["ms"]), eye.look(word, float(opts["ms"])))
        e_rest = [sum(float(rest[a.layers[l]["E"].slice].mean()) for a in column_zones if l in a.layers)
                  / sum(1 for a in column_zones if l in a.layers) for l in layers]
        ratio = [float(resp[c].mean() + 0.5) / float(rest[c].mean() + 0.5) for c in groups.values()]
        top = max((float(rest[p.slice].mean()), p.name) for a in column_zones for g in a.layers.values() for p in g.values())
        print(f"{te:6.2f} {pv:5.2f} | " + "        ".join([""]) + " ".join(f"{x:5.2f}" for x in e_rest) + " |              "
              + " ".join(f"{x:9.2f}" for x in ratio) + f" | {top[1]} {top[0]:.0f}", flush=True)
