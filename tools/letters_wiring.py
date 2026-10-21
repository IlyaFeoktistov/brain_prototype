"""
Диагностика (не проводка мозга): теряется ли различие букв на пути V1 L2/3 →
буквы L4 из-за случайного смешения источников. Свежий мозг small; проба
(буквы × reps) — у каждого источника V1 L2/3 E прибавка на каждую букву;
затем веса тех же входов местных клеток букв перераспределяются (среднее
строки прежнее, потолок пути), проверка — на новых показах:

  A — как есть (контроль);
  B — вес ∝ информативности источника (разброс средней прибавки между
      буквами): преимущественно клетки, различающие штрихи;
  C — вес ∝ положительной прибавке источника на одну букву, своей у клетки
      (случайно): вход — сочетание штрихов этой буквы (так стал бы путь
      после хеббова обучения на ней).

Печатает: сколько источников клетки различают буквы (разброс между буквами
больше разброса между повторами), их доля в буквенной части входа, вход ↔
ответ, различение ответа клеток букв.

    PYTHONPATH=. .venv/bin/python tools/letters_wiring.py mode=A|B|C [letters=отл] [reps=3] [half=8] [seed=0]
"""

import sys

import torch

from brain.body import Eye
from brain.config import BrainConfig
from brain.mind import Mind
from brain.world import Screen

opts = {"mode": "A", "letters": "отл", "reps": "3", "half": "8", "seed": "0"}
opts.update(dict(kv.split("=", 1) for kv in sys.argv[1:]))
cfg = BrainConfig()
cfg.save_path = "/nonexistent/brain.pt"
cfg.autosave = False
cfg.dt_ms = 0.5
cfg.seed = int(opts["seed"])
torch.manual_seed(cfg.seed)
b = Mind(cfg).brain
b.plasticity = False
i = [s.name for s in b.sensors].index("eye")
eye = Eye(cfg, b.sensor_offsets[i], b.sensors[i].size, b.device)
scr = Screen(cfg.retina_rows, cfg.retina_cols)
blank = scr.picture("")
half = float(opts["half"])
x0 = int(round(cfg.fovea_col))  # начало строки — в ямке (session._picture)
mid = x0 + 4.0
letters = opts["letters"]
reps = int(opts["reps"])
L4 = b.areas["letters"].layers["L4"]["E"]
cols = b.field_cols[L4.name]
cells = torch.arange(L4.start, L4.stop, device=b.device)[(cols - mid).abs() <= half]
rows = cells - L4.start
proj = next(p for p in b.projections if p.name == "visual -> letters")
pre = proj.pre[rows].long()
psp = float(b._psp_peak(L4))


def show() -> tuple[list[str], torch.Tensor]:
    """Показы букв: (буквы по порядку, прибавки всех клеток [показ, клетка])."""

    shown, gains = [], []

    for _ in range(reps):
        for letter in letters:
            b.observe(300, eye.look(blank, 300))
            rest = b.observe(300, eye.look(blank, 300))
            seen = b.observe(300, eye.look(scr.picture(letter, x0), 300))
            shown.append(letter)
            gains.append(seen - rest)

    return shown, torch.stack(gains)


b.observe(2500, eye.look(blank, 2500))
shown, probe = show()
by_letter = torch.stack([probe[[j for j, x in enumerate(shown) if x == c]] for c in letters])  # [буква, повтор, клетка]
mean = by_letter.mean(1)  # [буква, клетка]
between = mean.std(0)  # разброс между буквами
within = by_letter.std(1).mean(0) / reps ** 0.5  # шум средней по повторам
informative = between > 2 * within

if opts["mode"] != "A":
    w = proj.weight[rows].float()
    row_mean = w.mean(1, keepdim=True)

    if opts["mode"] == "B":
        score = between[pre]
    else:
        own = torch.randint(len(letters), (len(rows),), device=b.device, generator=b.generator)
        score = mean[own[:, None], pre].clamp(min=0)

    score = score / score.mean(1, keepdim=True).clamp(min=1e-9)
    new = torch.where(score.sum(1, keepdim=True) > 0, row_mean * score, w)
    proj.weight[rows] = new.clamp(max=proj.cap).to(proj.weight.dtype)

shown, test = show()
same = torch.tensor([[a == x for x in shown] for a in shown])
eye_ = torch.eye(len(shown), dtype=torch.bool)


def similarity(R: torch.Tensor) -> float:
    R = R.float().cpu()
    R = R - R.mean(1, keepdim=True)
    R = R / R.norm(dim=1, keepdim=True).clamp(min=1e-6)
    C = R @ R.t()
    return float(C[same & ~eye_].mean() - C[~same].mean())


def letter_part(R: torch.Tensor) -> torch.Tensor:
    return (R - R.mean(0, keepdim=True)).float().cpu()


w = proj.weight[rows].float()
parts = w[None] * test[:, pre] * psp  # [показ, клетка, вход], мВ·Гц
drive = parts.sum(2)
from_info = (parts * informative[pre][None]).sum(2)
response = test[:, cells]
share = float(letter_part(from_info).norm() / letter_part(drive).norm().clamp(min=1e-9))
corr = float(torch.corrcoef(torch.stack([letter_part(drive).flatten(), letter_part(response).flatten()]))[0, 1])
top = response.mean(0).topk(max(1, len(cells) // 20)).values.mean()
print(f"режим {opts['mode']}: клеток букв {len(cells)}, входов {pre.shape[1]}, вес среднее {float(w.mean()):.3f} (верхний 1% ×{float(w.flatten().topk(w.numel() // 100).values.mean() / w.mean()):.1f})")
print(f"  источников, различающих буквы: {float(informative.float().mean()) * 100:.1f}% V1 L2/3 E, у клетки букв {float(informative[pre].float().sum(1).mean()):.0f} из {pre.shape[1]}")
print(f"  буквенная часть входа ±{float(letter_part(drive).std()):.1f} мВ·Гц, из неё от различающих {share * 100:.0f}%; различение входа {similarity(drive):+.3f}")
print(f"  ответ клеток букв: прибавка {float(response.mean()):+.2f} Гц (верхние 5% {float(top):+.1f}); вход ↔ ответ {corr:+.3f}; различение ответа {similarity(response):+.3f}")
