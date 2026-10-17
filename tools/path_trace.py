"""
Межзонная передача A1 L2/3 → смысловая L4 (DESIGN «Слово по цепи»):
слова голосом, диагностический режим ствола (IC выкл. флагом, inh_off —
местное торможение CN и MGN выкл.), свежий мозг small. У каждой клетки
получателя — доставленный вход пути (Σ вес × спайки источника по
построенным синапсам, за слово) и ответ (прибавка частоты); для обоих —
прибавка, сходство повторов одного слова, разных и различение. Варианты в
одном мозге: путь как есть, выключен, вес ×2, ×4 (только сила синапса).
Сходимость — отдельным прогоном: interareal_share=0.42 (доля входов из
других зон ×2; построенных входов больше).

    PYTHONPATH=. .venv/bin/python tools/path_trace.py [words=кот,дом] [reps=4] [inh_off=cn,mgn] [interareal_share=0.21] [seed=0]
"""

import random
import sys

import torch

from brain.body import Cochlea
from brain.config import BrainConfig
from brain.mind import Mind
from brain.world import Voice

opts = {"words": "кот,дом", "reps": "4", "inh_off": "cn,mgn", "interareal_share": "0.21", "seed": "0"}
opts.update(dict(kv.split("=", 1) for kv in sys.argv[1:]))
cfg = BrainConfig()
cfg.save_path = "/nonexistent/brain.pt"
cfg.autosave = False
cfg.anatomy = True
cfg.hand = True
cfg.dt_ms = 0.5
cfg.ic_local_inhibition = False
cfg.interareal_share = float(opts["interareal_share"])
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

i = [s.name for s in b.sensors].index("ear")
ear = Cochlea(cfg, b.sensor_offsets[i], b.sensors[i].size, b.device)
voice = Voice()
post = b.areas["association"].layers["L4"]["E"]
path = next(p for p in b.projections if p.name == "auditory -> association")
w0 = path.weight.clone()
pre = path.pre.long()
words = opts["words"].split(",")
order = [w for w in words for _ in range(int(opts["reps"]))]
random.Random(cfg.seed).shuffle(order)
print(f"путь {path.name}: {tuple(path.weight.shape)} (клеток × входов), вес {float(w0.float().mean()):.3f}, "
      f"доля из других зон {cfg.interareal_share}")
b.run(1500.0)


def similarity(R: torch.Tensor) -> tuple[float, float]:
    R = R - R.mean(0, keepdim=True)
    R = R / R.norm(dim=1, keepdim=True).clamp(min=1e-6)
    C = R @ R.t()
    same = [float(C[x, y]) for x in range(len(order)) for y in range(len(order)) if x != y and order[x] == order[y]]
    diff = [float(C[x, y]) for x in range(len(order)) for y in range(len(order)) if order[x] != order[y]]
    return sum(same) / len(same), sum(diff) / len(diff)


for tag, scale in (("как есть", 1.0), ("выключен", 0.0), ("вес ×2", 2.0), ("вес ×4", 4.0)):
    path.weight.copy_(w0 * scale)
    b.run(500.0)
    drive, gain = [], []

    for word in order:
        quiet = b.observe(300.0)
        raster = ear.hear(*voice.say(word))
        sensory = torch.zeros(raster.shape[0], b.n_receptors, device=b.device)
        sensory[:, : raster.shape[1]] = raster
        loud = b.observe(raster.shape[0] * cfg.dt_ms, sensory)
        # Доставленный вход пути за слово (спайков × вес) — при весе как есть.
        drive.append((w0.float() * loud[pre]).sum(1).cpu() * raster.shape[0] * cfg.dt_ms / 1000.0)
        gain.append((loud[post.slice] - quiet[post.slice]).cpu())

    D, G = torch.stack(drive), torch.stack(gain)
    ds, dd = similarity(D)
    gs, gd = similarity(G)
    print(f"  {tag:9s} вход пути (вес как есть): {float(D.mean()):6.1f}, повтор {ds:+.3f} разные {dd:+.3f} различ. {ds - dd:+.3f} | "
          f"ответ L4: прибавка {float(G.mean()):+.2f} Гц, повтор {gs:+.3f} разные {gd:+.3f} различ. {gs - gd:+.3f}", flush=True)

path.weight.copy_(w0)
