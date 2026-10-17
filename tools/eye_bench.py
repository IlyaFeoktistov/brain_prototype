"""
Глаз (ROADMAP чат-минимум 7.2): сетчатка и V1 на новом мозге.

1. Полосы: горизонтальная / вертикальная / косые — отвечают ли клетки V1
   своей ориентации (доля спайков по ориентациям).
2. Слова: узор V1 одного слова на двух показах против разных слов; одна
   буква в разных местах против разных букв в одном месте.

    PYTHONPATH=. .venv/bin/python tools/eye_bench.py [key=value ...]
"""

import sys

import torch

from brain.body import Eye
from brain.config import BrainConfig
from brain.mind import Mind
from brain.world import Screen

cfg = BrainConfig()
cfg.size = "small"
cfg.save_path = "/nonexistent/brain.pt"
cfg.eye = True

for kv in sys.argv[1:]:
    k, v = kv.split("=", 1)
    old = getattr(cfg, k)
    setattr(cfg, k, type(old)(v))

mind = Mind(cfg)
brain = mind.brain
mind.cfg.autosave = False
V = brain.areas["visual"].E
i = [s.name for s in brain.sensors].index("eye")
eye = Eye(cfg, brain.sensor_offsets[i], brain.sensors[i].size, brain.device)
screen = Screen(cfg.retina_rows, cfg.retina_cols)


def v1(picture: torch.Tensor) -> torch.Tensor:
    with brain.frozen():
        brain.run(200.0)
        rates = brain.observe(cfg.fixation_ms, eye.look(picture))
    return rates[V.slice].cpu()


def cosine(a, b) -> float:
    na, nb = a.norm(), b.norm()
    return float(a @ b / (na * nb)) if na and nb else 0.0


blank = torch.ones(cfg.retina_rows, cfg.retina_cols)
r = eye.rates(screen.picture("москва"))
print(f"сетчатка на «москва»: ON активных {int((r[: r.numel() // 2] > 5).sum())}, "
      f"OFF {int((r[r.numel() // 2:] > 5).sum())}, макс {float(r.max()):.0f} Гц; "
      f"V1 клеток {V.size}")
quiet = v1(blank)
print(f"пустой экран: V1 стреляет {int((quiet > 0).sum())} клеток, {float(quiet.mean()):.1f} Гц")

bars = {}

for name, angle in (("—", 0), ("/", 45), ("|", 90), ("\\", 135)):
    pic = blank.clone()
    rr, cc = torch.meshgrid(torch.arange(cfg.retina_rows), torch.arange(cfg.retina_cols), indexing="ij")
    rr, cc = rr.float() - cfg.retina_rows / 2, cc.float() - cfg.retina_cols / 2
    a = torch.tensor(angle * torch.pi / 180)
    # Расстояние до прямой через центр под углом angle (вверх — минус строк).
    dist = (rr * torch.cos(a) + cc * torch.sin(a)).abs()
    pic[dist < 0.6] = 0.0
    rates = v1(pic)
    by = [float(rates[o::4].sum()) for o in range(4)]
    total = sum(by) or 1.0
    bars[name] = by
    print(f"  полоса {name} ({float(rates[rates > 0].mean()) if (rates > 0).any() else 0:.0f} Гц): по ориентациям "
          + " / ".join(f"{100 * b / total:.0f}%" for b in by) + f", клеток {int((rates > 0).sum())}")

words = ["москва", "берлин", "человек", "может", "столица", "является"]
seen = {w: (v1(screen.picture(w)), v1(screen.picture(w))) for w in words}
same = [cosine(a, b) for a, b in seen.values()]
diff = [cosine(seen[a][0], seen[b][0]) for a in words for b in words if a < b]
print(f"слово само с собой {sum(same) / len(same):.2f}, разные слова {sum(diff) / len(diff):.2f}")

letters = "аоклмнвс"
at = lambda ch, pos: v1(screen.picture(" " * pos + ch))
shift = [cosine(at(c, 0), at(c, 4)) for c in letters[:4]]
other = [cosine(at(a, 2), at(b, 2)) for a, b in zip(letters[:4], letters[4:])]
print(f"буква в другом месте {sum(shift) / len(shift):.2f}, другая буква в том же месте "
      f"{sum(other) / len(other):.2f}")
