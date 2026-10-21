"""
Профиль анатомического мозга: что съедает шаг (ядра видеокарты и Python).

    PYTHONPATH=. .venv/bin/python tools/profile_anatomy.py [ms=500] [warm=500] [ключ=значение настроек]

Мозг как в train_anatomy (dt 0.5 мс, рука), мир — указка на букве.
"""
import cProfile, pstats, sys, time
import torch
from torch.profiler import profile, ProfilerActivity
from brain.config import BrainConfig
from brain.mind import Mind
from brain.session import Session

opts = {"ms": "500", "warm": "500"}
cfg = BrainConfig(); cfg.save_path = "/nonexistent/b.pt"; cfg.autosave = False
cfg.dt_ms = 0.5

for kv in sys.argv[1:]:
    k, v = kv.split("=", 1)

    if k in opts:
        opts[k] = v
    else:
        old = getattr(cfg, k)
        setattr(cfg, k, (v == "1") if isinstance(old, bool) else type(old)(v))

t = time.time()
mind = Mind(cfg)
print(f"мозг собран за {time.time() - t:.0f} с")
session = Session(mind)
session.line, session.gaze = "м", 0.0
session.pointer = session.screen.width("м") / 2
engine = mind.brain
print(f"клеток {engine.n}, синапсов {engine.n_synapses}")
t = time.time(); session.live(float(opts["warm"])); torch.cuda.synchronize()
print(f"прогрев {opts['warm']} мс модели за {time.time() - t:.1f} с")
ms = float(opts["ms"])
t = time.time(); session.live(ms); torch.cuda.synchronize(); wall = time.time() - t
print(f"{ms:.0f} мс модели за {wall:.1f} с → {wall / ms * 1000:.0f} с на секунду модели")

pr = cProfile.Profile()

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    pr.enable(); session.live(100.0); torch.cuda.synchronize(); pr.disable()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30, max_name_column_width=70))
pstats.Stats(pr).sort_stats("tottime").print_stats(15)
