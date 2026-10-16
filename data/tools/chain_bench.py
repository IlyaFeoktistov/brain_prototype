"""
Цепь от входа до выхода (анатомический мозг, ARCHITECTURE 11): слово на
экране → сетчатка → LGN → V1 → буквы → смысл → ПФК и цель → гиппокамп;
стриатум → GPi → таламус → премоторная → рука → клавиша. По окнам
win мс — частота каждой ступени против покоя: где сигнал появился, с какой
задержкой (первое окно, где частота выше покоя на 3 σ окон покоя) и где
пропал. Отладка идёт по этой таблице сверху вниз.

    PYTHONPATH=. .venv/bin/python data/tools/chain_bench.py [word=мама] [ms=300] [win=20] [warm=2500] [load=МОЗГ]
"""

import sys

import torch

from brain.body import Eye, Hand
from brain.config import BrainConfig
from brain.mind import Mind
from brain.world import Keyboard, Screen

opts = {"word": "мама", "ms": "300", "win": "20", "rest": "1000", "warm": "2500", "load": ""}
cfg = BrainConfig()
cfg.save_path = "/nonexistent/brain.pt"
cfg.autosave = False
cfg.anatomy = True
cfg.hand = True
cfg.dt_ms = 0.5

for kv in sys.argv[1:]:
    k, v = kv.split("=", 1)

    if k in opts:
        opts[k] = v
    else:
        old = getattr(cfg, k)
        setattr(cfg, k, (v == "1") if isinstance(old, bool) else type(old)(v))

mind = Mind.load(opts["load"]) if opts["load"] else Mind(cfg)
mind.cfg.autosave = False
cfg, brain = mind.cfg, mind.brain
brain.plasticity = False
i = [s.name for s in brain.sensors].index("eye")
eye = Eye(cfg, brain.sensor_offsets[i], brain.sensors[i].size, brain.device)
screen = Screen(cfg.retina_rows, cfg.retina_cols)
keyboard = Keyboard()
hand = Hand(len(keyboard), cfg.hand_cells)


def stages() -> list[tuple[str, slice]]:
    """Ступени по ходу сигнала: (имя, клетки)."""

    out = [("сетчатка", slice(eye.offset, eye.offset + eye.size))]
    order = ("lgn", "trn", "visual", "letters", "association", "category", "prefrontal", "goal", "md",
             "entorhinal", "dentate", "ca3", "ca1", "striatum_d1", "striatum_d2", "fsi", "gpe", "stn", "gpi",
             "thalamus", "premotor", "hand", "motor", "granule", "purkinje", "dcn", "io", "vta")

    for name in order:
        area = brain.areas.get(name)

        if area is None:
            continue

        if area.layers is None:
            out.append((name, area.E.slice))
            continue

        for layer, group in area.layers.items():
            out.append((f"{name}.{layer}", group["E"].slice))

    return out


rows = stages()
win = float(opts["win"])
blank = screen.picture("")
# Разгон: после сборки кора оседает ~2–3 с (как в anatomy_fit) — покой
# мерить после него, а не на переходном процессе.
brain.observe(float(opts["warm"]), eye.look(blank, float(opts["warm"])))
# Покой: пустой экран (сетчатка видит ровный фон — молчит, кроме спонтанной).
base = []

for _ in range(int(float(opts["rest"]) // win)):
    rates = brain.observe(win, eye.look(blank, win))
    base.append(torch.stack([rates[s].mean() for _, s in rows]))

base = torch.stack(base)
mean, sd = base.mean(0), base.std(0)
picture = screen.picture(opts["word"])
windows = int(float(opts["ms"]) // win)
trace = []
# Нажатие — как в мире (session.Session.live): набор клеток клавиши
# выстрелил наполовину за одно окно, а не за весь показ (за 300 мс фон
# руки ~5 Гц сам «нажмёт» любую клавишу).
keys = []
raster = eye.look(picture, windows * win)
steps = raster.shape[0] // windows

for w in range(windows):
    rates = brain.observe(win, raster[w * steps:(w + 1) * steps])
    trace.append(torch.stack([rates[s].mean() for _, s in rows]))
    key = hand.pressed(rates[brain.areas["hand"].E.slice])

    if key is not None and (not keys or keys[-1][1] != key):
        keys.append((w * win, key))

trace = torch.stack(trace)
print(f"слово «{opts['word']}», окна {win:g} мс; покой — среднее ± σ; задержка — первое окно > покой + 3σ\n")
print(f"{'ступень':16s} {'покой':>12s} {'пик':>7s} {'задержка':>9s}   по окнам, Гц")

for j, (name, _) in enumerate(rows):
    m, s = float(mean[j]), float(sd[j])
    over = (trace[:, j] > m + 3 * max(s, 0.05)).nonzero()
    lag = f"{float(over[0]) * win + win:6.0f} мс" if len(over) else "     —   "
    cells = " ".join(f"{float(v):5.1f}" for v in trace[:, j])
    print(f"{name:16s} {m:6.1f} ±{s:4.1f} {float(trace[:, j].max()):7.1f} {lag:>9s}   {cells}")

print("\nклавиши: " + (" ".join(f"{t:g} мс {keyboard.type(k)!r}" for t, k in keys) if keys else "не нажаты"))
