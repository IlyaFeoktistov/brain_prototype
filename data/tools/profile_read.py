"""
Профиль чтения и сна: 40 строк train-data, затем сон; скорость и горячие функции.

    .venv/bin/python data/tools/profile_read.py 0|1 [ripples]   (1 — нейронный гиппокамп, ripples — сон вспышками)
"""
import cProfile, pstats, sys, time
from pathlib import Path
import torch
from brain.config import BrainConfig
from brain.mind import Mind
cfg = BrainConfig(); cfg.size = "small"; cfg.autosave = False; cfg.save_path = "/nonexistent/b.pt"
cfg.neural_hippocampus = sys.argv[1] == "1"; cfg.neural_replay = len(sys.argv) > 2
m = Mind(cfg); m.calibrate()
lines = [l for l in Path("train-data.md").read_text().splitlines() if l.strip()][:40]
tokens = sum(len(m.language.tokenize(l)) for l in lines)
pr = cProfile.Profile()
t = time.time(); pr.enable()
m.language.read_text("\n".join(lines)); torch.cuda.synchronize()
read = time.time() - t
t2 = time.time()
m.language.sleep_until_rested(); torch.cuda.synchronize()
pr.disable()
print(f"чтение {tokens} токенов за {read:.1f} с ({tokens/read:.0f} ток/с), сон {time.time()-t2:.1f} с")
st = pstats.Stats(pr); st.sort_stats("tottime").print_stats(22)
