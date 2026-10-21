"""
Логнормальные врождённые веса пути V1 → буквы (cfg.interareal_weight_sigma):
свежий мозг без обучения, буквы в ямке — как letters_local. Печатает для
местных клеток L4 букв: покой / первые 50 мс / вся буква (Гц), различение
(сходство ответов на одну букву против разных, z по Пуассону), самую
горячую зону (разгон), и связаны ли сильные входы со штрихами — корреляция
между «вход по весам» (Σ вес × прибавка частоты источника V1 на эту букву)
и прибавкой самой клетки, и различие этого входа между буквами.

    PYTHONPATH=. .venv/bin/python tools/letters_weights.py [sigma=0] [letters=отл] [reps=3] [half=8] [seed=0]
"""

import sys

import torch

from brain.body import Eye
from brain.config import BrainConfig
from brain.mind import Mind
from brain.world import Screen

opts = {"sigma": "0", "letters": "отл", "reps": "3", "half": "8", "seed": "0"}
opts.update(dict(kv.split("=", 1) for kv in sys.argv[1:]))
cfg = BrainConfig()
cfg.save_path = "/nonexistent/brain.pt"
cfg.autosave = False
cfg.dt_ms = 0.5
cfg.seed = int(opts["seed"])
cfg.interareal_weight_sigma = float(opts["sigma"])
torch.manual_seed(cfg.seed)
m = Mind(cfg)
b = m.brain
b.plasticity = False
i = [s.name for s in b.sensors].index("eye")
eye = Eye(cfg, b.sensor_offsets[i], b.sensors[i].size, b.device)
scr = Screen(cfg.retina_rows, cfg.retina_cols)
blank = scr.picture("")
half = float(opts["half"])
x0 = int(round(cfg.fovea_col))  # начало строки — в ямке (session._picture)
mid = x0 + 4.0


def local(pop) -> torch.Tensor:
    cols = b.field_cols.get(pop.name)
    idx = torch.arange(pop.start, pop.stop, device=b.device)
    return idx[(cols - mid).abs() <= half] if cols is not None else idx


L4 = b.areas["letters"].layers["L4"]["E"]
cells = local(L4)
proj = next(p for p in b.projections if p.name == "visual -> letters")
rows = cells - L4.start
pre = proj.pre[rows].long()
w = proj.weight[rows].float()
top = w.flatten().topk(max(1, w.numel() // 100)).values.mean()
print(f"σ {cfg.interareal_weight_sigma}: вес среднее {float(w.mean()):.3f} медиана {float(w.median()):.3f} верхний 1% {float(top):.3f} (×{float(top / w.mean()):.1f})")

b.observe(2500, eye.look(blank, 2500))
shown, Z, drive, gain = [], [], [], []
rest_hz, early_hz, seen_hz, hottest = [], [], [], 0.0

for rep in range(int(opts["reps"])):
    for letter in opts["letters"]:
        b.observe(300, eye.look(blank, 300))
        rest = b.observe(300, eye.look(blank, 300))
        picture = scr.picture(letter, x0)
        early = b.observe(50, eye.look(picture, 50))
        late = b.observe(250, eye.look(picture, 250))
        seen = (early * 50 + late * 250) / 300
        shown.append(letter)
        r, s = rest[cells], seen[cells]
        Z.append((s - r) * 0.3 / ((r + 0.5) * 0.3).sqrt())
        rest_hz.append(float(r.mean()))
        early_hz.append(float(early[cells].mean()))
        seen_hz.append(float(s.mean()))
        # Вход по весам: прибавка источников V1 на букву через веса клетки.
        drive.append((w * (seen - rest)[pre]).sum(1))
        gain.append(s - r)

        for area in b.areas.values():
            hottest = max(hottest, float(seen[area.cells].mean()))

n = len(shown)
same = torch.tensor([[a == x for x in shown] for a in shown])
eye_ = torch.eye(n, dtype=torch.bool)


def similarity(rows_: list[torch.Tensor]) -> float:
    R = torch.stack(rows_).float().cpu()
    R = R - R.mean(1, keepdim=True)
    R = R / R.norm(dim=1, keepdim=True).clamp(min=1e-6)
    C = R @ R.t()
    return float(C[same & ~eye_].mean() - C[~same].mean())


D = torch.stack(drive)
G = torch.stack(gain)
corr = torch.corrcoef(torch.stack([D.flatten(), G.flatten()]).float().cpu())[0, 1]
print(f"  L4 букв: покой {sum(rest_hz) / n:.1f}  первые 50 мс {sum(early_hz) / n:.1f}  буква {sum(seen_hz) / n:.1f} Гц; горячее всех зон {hottest:.1f} Гц")
print(f"  различение ответа {similarity(Z):+.3f}; входа по весам {similarity(drive):+.3f}")
print(f"  вход по весам: {float(D.mean()):.1f} ± {float(D.std()):.1f} мВ·Гц (между буквами у клетки ±{float(D.std(0).mean()):.1f}); корреляция с ответом клетки {float(corr):+.3f}")
