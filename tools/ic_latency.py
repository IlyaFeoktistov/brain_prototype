"""
Латентность слухового пути против человека: тон 2 кГц, 90 дБ — первая волна у
нижнего двухолмия человека через ~6.5 мс (anatomy.HUMAN_IC_LATENCY_MS,
Møller & Jannetta 1982, https://doi.org/10.1016/0013-4694(82)90137-7).

Свежий мозг small, шаг 0.5 мс без пачек (SpikingBrain.CHUNK = 1), пластичность
выкл. 300 мс тишины, затем тон 50 мс (уровень тона — cochlea_speech_db = 90).
По звеньям (нерв, CN, IC, MGN, L4 слуховой коры) — клетки полосы 2 кГц ± 2
полосы: начало ответа популяции (первый шаг, где спайков больше, чем в
тишине, на 3 σ) и медиана первого спайка клеток, ответивших за 30 мс.

    PYTHONPATH=. .venv/bin/python tools/ic_latency.py [path_delays=0] [seed=0]
"""

import math
import sys

import torch

from brain import anatomy
from brain.body import Cochlea
from brain.config import BrainConfig
from brain.engine import SpikingBrain
from brain.mind import Mind

opts = {"path_delays": "0", "seed": "0", "freq": "2000", "db": "90"}

for kv in sys.argv[1:]:
    k, v = kv.split("=", 1)
    opts[k] = v

SpikingBrain.CHUNK = 1
cfg = BrainConfig()
cfg.save_path, cfg.autosave = "/nonexistent/brain.pt", False
cfg.seed = int(opts["seed"])
cfg.path_delays = opts["path_delays"] == "1"
cfg.cochlea_speech_db = float(opts["db"])
torch.manual_seed(cfg.seed)
b = Mind(cfg).brain
b.plasticity = False
i_ear = next(i for i, s in enumerate(b.sensors) if s.name == "ear")
ear = Cochlea(cfg, b.sensor_offsets[i_ear], b.sensors[i_ear].size, b.device)
rate, f = 44100, float(opts["freq"])
tone = torch.sin(2 * math.pi * f * torch.arange(int(0.05 * rate)) / rate)
sound = ear.hear(tone, rate)
dt = cfg.dt_ms
quiet, after = int(300 / dt), int(60 / dt)
raster = torch.zeros(quiet + after, sound.shape[1], device=b.device)
raster[quiet: quiet + sound.shape[0]] = sound
band = int(torch.argmin((ear.centres - f).abs()))
per = cfg.sensory_size // cfg.cochlea_bands


def in_band(index, size):
    """Клетки ядра (по месту j → полоса j // per) в своей полосе ± 2."""

    j = torch.arange(size, device=b.device) % cfg.sensory_size
    return ((j // per) - band).abs() <= 2


groups = {}
nerve = torch.arange(cfg.sensory_size, device=b.device)
groups["нерв"] = ("sensory", b.sensor_offsets[i_ear] + nerve[((nerve // per) - band).abs() <= 2])

for name in ("cn", "ic", "mgn"):
    E = b.areas[name].E
    groups[name.upper()] = ("net", E.start + torch.nonzero(in_band(None, E.size)).squeeze(1))

L4 = b.areas["auditory"].layers["L4"]["E"]
own = ((torch.arange(L4.size, device=b.device).float() + 0.5) * cfg.cochlea_bands / L4.size).long()
groups["A1 L4"] = ("net", L4.start + torch.nonzero((own - band).abs() <= 2).squeeze(1))

b.run(300.0)  # стартовый переход
counts = {k: [] for k in groups}
first = {k: torch.full((len(v[1]),), float("nan"), device=b.device) for k, v in groups.items()}

for step in range(raster.shape[0]):
    b.observe(dt, raster[step: step + 1])
    t = (step - quiet) * dt

    for k, (src, idx) in groups.items():
        fired = (raster[step, idx] > 0) if src == "sensory" else (b.counts[idx] > 0)
        counts[k].append(int(fired.sum()))

        if 0 <= t <= 30:
            first[k] = torch.where(torch.isnan(first[k]) & fired, torch.tensor(t, device=b.device), first[k])

print(f"тон {f:g} Гц, {opts['db']} дБ, полоса {band} ({float(ear.centres[band]):.0f} Гц) ± 2; "
      f"задержки путей {'вкл' if cfg.path_delays else 'выкл'}; шаг {dt} мс")
print(f"человек: приход в IC ~{anatomy.HUMAN_IC_LATENCY_MS} мс (волна V)\n")
print(f"{'звено':8s} {'клеток':>6s} {'тишина, сп/шаг':>15s} {'начало ответа, мс':>18s} {'медиана 1-го спайка, мс':>24s}")

for k in groups:
    c = torch.tensor(counts[k], dtype=torch.float)
    base = c[:quiet]
    mean, sd = float(base.mean()), float(base.std())
    above = torch.nonzero(c[quiet:] > mean + 3 * max(sd, 0.5)).squeeze(1)
    onset = float(above[0]) * dt if len(above) else float("nan")
    fs = first[k][~torch.isnan(first[k])]
    med = float(fs.median()) if len(fs) else float("nan")
    print(f"{k:8s} {len(groups[k][1]):6d} {mean:15.2f} {onset:18.1f} {med:24.1f}")
