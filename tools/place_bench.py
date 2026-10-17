"""
Слово только на связях (ROADMAP чат-минимум 7.3а): место слова в смысловой
коре захватывается нейронами, а не таблицей.

Показ слова глазом: форма (зона букв) → смысл по выученному пути. Знакомое
слово зажигает свои клетки. Новое не зажигает ничего — новизна:
ацетилхолин нарастающе поднимает возбудимость зоны (стенд — ровный
нарастающий ток), первыми стреляют самые возбудимые клетки (распределение
памяти по возбудимости — Han et al. 2007, Yiu et al. 2014); связь «форма →
эти клетки» записывается (instar), их возбудимость падает — следующее новое
займут другие. Проверка: каждое слово показать снова — зажглось ли его
место.

    PYTHONPATH=. .venv/bin/python tools/place_bench.py МОЗГ [words=200] [ramp=25] [drop=4] [sigma=2] [rate=1]
"""

import random
import re
import sys
from pathlib import Path

import torch

from brain.body import Eye
from brain.mind import Mind
from brain.world import Screen

opts = {"words": "200", "ramp": "25", "drop": "4", "sigma": "2", "rate": "1", "grow": "8", "cap": "1"}

for kv in sys.argv[2:]:
    k, v = kv.split("=", 1)
    opts[k] = v

mind = Mind.load(sys.argv[1])
mind.cfg.autosave = False
cfg, brain, lang = mind.cfg, mind.brain, mind.language
A = brain.areas["association"].E
F = brain.areas["letters"].E
i = [s.name for s in brain.sensors].index("eye")
eye = Eye(cfg, brain.sensor_offsets[i], brain.sensors[i].size, brain.device)
screen = Screen(cfg.retina_rows, cfg.retina_cols)
path = brain.projection("letters -> association")
everyone = torch.arange(A.size, device=brain.device)
gen = torch.Generator(device=brain.device).manual_seed(1)
# Врождённая разнородность возбудимости клеток (σ мВ) — у живых нейронов
# порог и сопротивление разные.
brain.bias[A.slice] = torch.randn(A.size, device=brain.device, generator=gen) * float(opts["sigma"])


def show(picture: torch.Tensor, novelty: float = 0.0) -> torch.Tensor:
    """Фиксация на слове; novelty — нарастающий ток новизны в смысл, мВ."""

    raster = eye.look(picture)
    brain.counts.zero_()
    steps = 5
    part = cfg.fixation_ms / steps

    for j in range(steps):
        if novelty:
            brain.feel("association", everyone, novelty * (j + 1) / steps)
        brain.run(part, raster[int(j * raster.shape[0] / steps):])

    brain.feel("association", None, 0.0)
    return brain.counts[: brain.n].float().clone()


def capture(counts: torch.Tensor, cells: torch.Tensor) -> None:
    """Связь «форма → клетки места»: рост синапсов от активных клеток букв и instar."""

    letters = (counts[F.slice] > 0).nonzero().flatten()

    if len(letters):
        brain.grow([("letters -> association", letters, cells)], len(cells) * int(opts["grow"]))

    pre = (counts.index_select(0, path.pre.flatten().long()).view(path.pre.shape) > 0).float()
    post = torch.zeros(A.size, device=brain.device)
    post[cells] = 1.0
    rate = float(opts["rate"])
    path.weight.add_(rate * post[:, None] * (float(opts["cap"]) * pre - path.weight))


text = "\n".join(p.read_text(encoding="utf-8") for p in sorted(Path("data/corpus/qa").glob("*.txt")))
vocab = sorted({w for w in re.findall(r"[а-яё]+", text.lower()) if 3 <= len(w) <= 8})
rng = random.Random(0)
words = rng.sample(vocab, int(opts["words"]))
places: dict[str, torch.Tensor] = {}
sizes = []

for w in words:
    pic = screen.picture(w)

    with brain.frozen():
        seen = show(pic)

    known = (seen[A.slice] > 0).nonzero().flatten()

    if len(known) >= 2:
        places[w] = known
        continue

    with brain.frozen():
        seen = show(pic, float(opts["ramp"]))

    cells = (seen[A.slice] > 0).nonzero().flatten()
    sizes.append(len(cells))

    if not len(cells):
        continue

    places[w] = cells
    capture(seen, cells)
    brain.bias[A.start + cells] -= float(opts["drop"])

print(f"захвачено мест: {len(sizes)} из {len(words)}; клеток на место: среднее "
      f"{sum(sizes) / max(1, len(sizes)):.1f}, мин {min(sizes, default=0)}, макс {max(sizes, default=0)}")

right = silent = 0
overlap = []

for w in words:
    if w not in places:
        continue

    with brain.frozen():
        seen = show(screen.picture(w))

    fired = (seen[A.slice] > 0).nonzero().flatten()

    if not len(fired):
        silent += 1
        continue

    mine = places[w]
    share = len(set(fired.tolist()) & set(mine.tolist())) / len(set(fired.tolist()) | set(mine.tolist()))
    overlap.append(share)
    right += share >= 0.5

print(f"узнавание: своё место (≥ половины общих) {right}/{len(places)}, молчит {silent}, "
      f"среднее совпадение {sum(overlap) / max(1, len(overlap)):.2f}")
cells_used = torch.cat(list(places.values())) if places else torch.zeros(0)
print(f"клеток занято {len(set(cells_used.tolist()))}, мест делят клетки: "
      f"{len(cells_used) - len(set(cells_used.tolist()))}")
