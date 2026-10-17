"""
Различают ли буквы клетки, чьё место — под буквой: буква стоит там, куда
её ставит мир при обучении (начало строки — в ямке, cfg.fovea_col), клетки
— с местом в пределах ±half px от середины буквы: letters_probe по всей зоне смешивает местные
клетки с далёкими, которые букву не видят, и разница тонет в их шуме.
Ответ — z по Пуассону, сходство ответов на одну букву против разных.

    PYTHONPATH=. .venv/bin/python data/tools/letters_local.py МОЗГ [half=8] [letters=аомсткл]
"""

import sys, torch
from brain.body import Eye
from brain.mind import Mind
from brain.world import Screen
m = Mind.load(sys.argv[1]); cfg, b = m.cfg, m.brain; b.plasticity = False
i = [s.name for s in b.sensors].index("eye")
eye = Eye(cfg, b.sensor_offsets[i], b.sensors[i].size, b.device)
scr = Screen(cfg.retina_rows, cfg.retina_cols); blank = scr.picture("")
opts = dict(kv.split("=", 1) for kv in sys.argv[2:])
half = float(opts.get("half", "8"))
x0 = int(round(cfg.fovea_col))  # начало строки — в ямке (session._picture)
mid = x0 + 4.0
show = opts.get("letters", "аомсткл")
stages = [("visual","L4"),("visual","L23"),("letters","L4"),("letters","L23")]
cells = {}
for z, l in stages:
    pop = b.areas[z].layers[l]["E"]
    cols = b.field_cols.get(pop.name)
    idx = torch.arange(pop.start, pop.stop, device=b.device)
    cells[f"{z}.{l}"] = idx[(cols - mid).abs() <= half] if cols is not None else idx
    print(z, l, "местных клеток", len(cells[f"{z}.{l}"]), "из", pop.size, "поле", b.field_width if cols is not None else None)
b.observe(2500, eye.look(blank, 2500))
shown, Z = [], {k: [] for k in cells}
for rep in range(3):
    for letter in show:
        b.observe(300, eye.look(blank, 300)); rest = b.observe(300, eye.look(blank, 300))
        seen = b.observe(300, eye.look(scr.picture(letter, x0), 300)); shown.append(letter)
        for k, c in cells.items():
            r, s = rest[c], seen[c]
            Z[k].append((s - r) * 0.3 / ((r + 0.5) * 0.3).sqrt())
same = torch.tensor([[a == x for x in shown] for a in shown]); eye_ = torch.eye(len(shown), dtype=torch.bool)
for k, zs in Z.items():
    R = torch.stack(zs).float().cpu(); mean_dz = float(R.mean())
    R = R - R.mean(1, keepdim=True); R = R / R.norm(dim=1, keepdim=True).clamp(min=1e-6); C = R @ R.t()
    print(f"{k}: средний z {mean_dz:+.2f}; одна {float(C[same & ~eye_].mean()):.3f} разные {float(C[~same].mean()):.3f} разница {float(C[same & ~eye_].mean() - C[~same].mean()):+.3f}")
