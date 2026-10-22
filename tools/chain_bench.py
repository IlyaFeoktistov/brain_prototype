"""
Цепь от входа до выхода (анатомический мозг, ARCHITECTURE 11): слово на
экране → сетчатка → LGN → V1 → буквы → смысл → ПФК и цель → гиппокамп;
стриатум → GPi → таламус → премоторная → рука → клавиша. По окнам
win мс — частота каждой ступени против покоя: где сигнал появился, с какой
задержкой (первое окно, где частота выше покоя на 3 σ окон покоя) и где
пропал. Отладка идёт по этой таблице сверху вниз.

    PYTHONPATH=. .venv/bin/python tools/chain_bench.py [word=мама] [ms=300] [win=20] [warm=2500] [load=МОЗГ] [trn_scale=1] [tc_depression=1] [fit=имя] [homeo=имя]

trn_scale — проводимость TRN → реле LGN ×k (анализ чувствительности, не число
мозга); tc_depression=0 — без депрессии синапсов LGN → кора (причинный тест); прочие ключи — поля BrainConfig (relay_v_t_mv=-59, seed=2).
"""

import sys

import torch

from brain.body import Eye, Hand
from brain.config import BrainConfig
from brain.mind import Mind
from brain.world import Keyboard, Screen

opts = {"word": "мама", "ms": "300", "win": "20", "rest": "1000", "warm": "2500", "load": "", "trn_scale": "1", "tc_depression": "1", "fit": "", "homeo": ""}
cfg = BrainConfig()
cfg.save_path = "/nonexistent/brain.pt"
cfg.autosave = False
cfg.dt_ms = 0.5

for kv in sys.argv[1:]:
    k, v = kv.split("=", 1)

    if k in opts:
        opts[k] = v
    else:
        old = getattr(cfg, k)
        setattr(cfg, k, (v == "1") if isinstance(old, bool) else float(v) if old is None else type(old)(v))

if opts["fit"]:
    # Фон из ~/.brain/fits/<имя>.json (anatomy_fit out=имя).
    from brain import anatomy

    anatomy.use_fit(cfg, opts["fit"])
mind = Mind.load(opts["load"]) if opts["load"] else Mind(cfg)
mind.cfg.autosave = False
cfg, brain = mind.cfg, mind.brain
brain.plasticity = False

if opts["homeo"]:
    # Сдвиги гомеостаза после развития (tools/develop.py out=имя) — заморожены.
    from brain.config import BRAIN_HOME

    brain.bias[: brain.n] += torch.load(BRAIN_HOME / "fits" / f"{opts["homeo"]}-homeo.pt").to(brain.device)

for p in brain.projections:
    if p.name == "trn -> lgn":
        p.weight.mul_(float(opts["trn_scale"]))
i = [s.name for s in brain.sensors].index("eye")
eye = Eye(cfg, brain.sensor_offsets[i], brain.sensors[i].size, brain.device)
screen = Screen(cfg.retina_rows, cfg.retina_cols)
keyboard = Keyboard()
hand = Hand(len(keyboard), cfg.hand_cells)


def stages() -> list[tuple[str, slice]]:
    """Ступени по ходу сигнала: (имя, клетки)."""

    out = []
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
# Клетки LGN, чей «водитель» из сетчатки видит слово, — отдельной строкой:
# среднее по всему LGN тонет в его качелях с ретикулярным ядром.
seen = (eye.rates(screen.picture(opts["word"])) >= 5.0).nonzero().flatten().to(brain.device) + eye.offset
driver = next(p for p in brain.projections if p.name == "eye -> " + brain.areas["lgn"].E.name)
on = torch.isin(driver.pre[:, 0], seen)
lgn = brain.areas["lgn"].E
rows += [("lgn: видят слово", torch.arange(lgn.start, lgn.stop, device=brain.device)[on]),
         ("lgn: остальные", torch.arange(lgn.start, lgn.stop, device=brain.device)[~on])]
print(f"клеток LGN, чей водитель видит слово: {int(on.sum())} из {lgn.size}")
# Клетки V1 L4, чьё поле на слове: больше половины входов от зрячих LGN.
l4 = brain.areas["visual"].layers["L4"]["E"]
tc = next(p for p in brain.projections if p.name == f"lgn -> {l4.name}")
share = on[(tc.pre.long() - lgn.start)].float().mean(1)
cells = torch.arange(l4.start, l4.stop, device=brain.device)
rows += [("V1 L4: поле на слове", cells[share > 0.5]), ("V1 L4: остальные", cells[share <= 0.5])]
print(f"клеток V1 L4 с полем на слове: {int((share > 0.5).sum())} из {l4.size}")
field = cells[share > 0.5]
seeing = torch.arange(lgn.start, lgn.stop, device=brain.device)[on]

if opts["tc_depression"] == "0":
    brain.stp_u_cell[lgn.slice] = 0.0
# Состояние в конце окна: ресурс синапсов зрячих LGN (депрессия; 1 — отдохнул)
# и средний V клеток поля V1 L4.
state = lambda: torch.stack([brain.stp_x[seeing].mean(), brain.v[field].mean()])
extra = []
win = float(opts["win"])
blank = screen.picture("")
# Разгон: после сборки кора оседает ~2–3 с (как в anatomy_fit) — покой
# мерить после него, а не на переходном процессе.
brain.observe(float(opts["warm"]), eye.look(blank, float(opts["warm"])))
# Покой: пустой экран (сетчатка видит ровный фон — молчит, кроме спонтанной).
base = []

def retina(raster) -> torch.Tensor:
    """Частота ганглиозных клеток (входной растр), Гц."""
    return raster[:, eye.offset:].mean() * 1000.0 / cfg.dt_ms


for _ in range(int(float(opts["rest"]) // win)):
    raster = eye.look(blank, win)
    rates = brain.observe(win, raster)
    base.append(torch.stack([retina(raster)] + [rates[s].mean() for _, s in rows]))

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
    part = raster[w * steps:(w + 1) * steps]
    rates = brain.observe(win, part)
    trace.append(torch.stack([retina(part)] + [rates[s].mean() for _, s in rows]))
    extra.append(state())
    key = hand.pressed(rates[brain.areas["hand"].output.slice])

    if key is not None and (not keys or keys[-1][1] != key):
        keys.append((w * win, key))

trace = torch.stack(trace)
print(f"слово «{opts['word']}», окна {win:g} мс; покой — среднее ± σ; задержка — первое окно > покой + 3σ\n")
print(f"{'ступень':16s} {'покой':>12s} {'пик':>7s} {'задержка':>9s}   по окнам, Гц")

for j, (name, _) in enumerate([("сетчатка", None)] + rows):
    m, s = float(mean[j]), float(sd[j])
    over = (trace[:, j] > m + 3 * max(s, 0.05)).nonzero()
    lag = f"{float(over[0]) * win + win:6.0f} мс" if len(over) else "     —   "
    cells = " ".join(f"{float(v):5.1f}" for v in trace[:, j])
    print(f"{name:16s} {m:6.1f} ±{s:4.1f} {float(trace[:, j].max()):7.1f} {lag:>9s}   {cells}")

extra = torch.stack(extra)
print(f"{'ресурс LGN слова':16s} {'':>12s} {'':>7s} {'':>9s}   " + " ".join(f"{float(v):5.2f}" for v in extra[:, 0]))
print(f"{'V поля L4, мВ':16s} {'':>12s} {'':>7s} {'':>9s}   " + " ".join(f"{float(v):5.1f}" for v in extra[:, 1]))
print("\nклавиши: " + (" ".join(f"{t:g} мс {keyboard.type(k)!r}" for t, k in keys) if keys else "не нажаты"))
