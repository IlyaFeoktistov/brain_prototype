"""
Чему учится путь V1 L2/3 → буквы L4 при обучении азбуке (протокол
train_anatomy stage=letters: указка под буквой dwell мс, затем буква ещё
200 мс, сразу следующая — пустого экрана нет). Свежий мозг small, буквы
«отл», местные клетки букв (±half px). Перед обучением — проба без
пластичности: шаблон буквы для входов клетки — положительная прибавка
источника на эту букву (как режим C letters_wiring, который даёт
различение 0.55).

По эпохам печатает:
- Δw пути отдельно за «указку» (dwell) и «после» (200 мс), и его
  выравнивание с шаблонами: у каждой клетки корреляция Δw с шаблоном каждой
  буквы, взят максимум по буквам; контроль — то же со шаблоном другой
  клетки (перемешаны);
- совпадения спайков в окне LTP (источник в своём или прошлом 5-мс окне,
  клетка — в этом; ~0–10 мс) и LTD (наоборот) — на синапс в секунду,
  отдельно у входов шаблона (верхние 10% шаблона клетки по её лучшей
  букве) и у прочих;
- после эпохи, без пластичности: матрица «клетка × буква» — сколько клеток
  предпочитают каждую букву, индекс избирательности, различение ответа.

pointer=0 — учёба без указки (буква на экране так же долго). Конкуренция
клеток букв: разреженность ответа на букву (Treves & Rolls; 1 — отвечают
все одинаково, к 0 — немногие) и устойчивость предпочтений — доля клеток,
что предпочитают ту же букву, что до обучения (у 3 букв случайно ~1/3):
высокая при неизменном различении — закреплён случайный начальный перевес.

ne= / ach= — диагностика третьего фактора (Seol et al. 2007: без
модуляторов STDP коры нет, β-адреналовые — LTP, M1-мускариновые — LTD,
оба — обычное окно): LTP × ne, LTD × ach. Амплитуды общие для всех путей —
в стенде из обучаемых меняется только путь коры (стриатуму без всплеска
дофамина след в вес не идёт).

gain= — внимание при учёбе как прибавка контраста (Reynolds et al. 2000:
в V4 ×1.51 эквивалентного контраста; Buffalo et al. 2010: ответ V1 +5%,
V4 +23%): вход LGN → V1 L4 (E и PV) × gain на время показа, на проверках
— 1.

ffi= — отношение суммарного возбуждающего тока пути V1 → буквы на клетку
PV к клетке E (сейчас ~1.84; D'Souza et al. 2013/2016, L2/3 мыши,
суммарный EPSC: V1 → LM 3.7×, V1 → PM 12.85×): ветка пути на PV × ffi /
нынешнее, на всё время.

    PYTHONPATH=. .venv/bin/python tools/letters_stdp.py [ffi=—] [gain=1] [ne=1] [ach=1] [epochs=3] [shows=30] [dwell=300] [letters=отл] [half=8] [seed=0] [pointer=1]
"""

import random
import sys

import torch

from brain.config import BrainConfig
from brain.mind import Mind
from brain.session import Session

opts = {"gain": "1", "ne": "1", "ach": "1", "pointer": "1", "epochs": "3", "shows": "30", "dwell": "300", "letters": "отл", "half": "8", "seed": "0"}
opts.update(dict(kv.split("=", 1) for kv in sys.argv[1:]))
cfg = BrainConfig()
cfg.save_path = "/nonexistent/brain.pt"
cfg.autosave = False
cfg.anatomy = True
cfg.hand = True
cfg.dt_ms = 0.5
cfg.seed = int(opts["seed"])
torch.manual_seed(cfg.seed)
mind = Mind(cfg)
b = mind.brain
# После сборки: anatomy.apply выводит LTD из LTP (Froemke & Dan), шаг
# пластичности ещё не записан в граф.
cfg.stdp_a_plus *= float(opts["ne"])
cfg.stdp_a_minus *= float(opts["ach"])
s = Session(mind)
rng = random.Random(cfg.seed)
letters = opts["letters"]
dwell = float(opts["dwell"])
half = float(opts["half"])
WIN = 5.0
x0 = cfg.fovea_col
mid = x0 + 4.0
L4 = b.areas["letters"].layers["L4"]["E"]
cols = b.field_cols[L4.name]
cells = torch.arange(L4.start, L4.stop, device=b.device)[(cols - mid).abs() <= half]
rows = cells - L4.start
proj = next(p for p in b.projections if p.name == "visual -> letters")
pre = proj.pre[rows].long()
thalamic = [p for p in b.projections if p.name.startswith("lgn -> visual")]
to_pv = next(p for p in b.projections if p.name == "visual -> letters.I")
ratio = float(to_pv.weight.float().sum(1).mean() / proj.weight.float().sum(1).mean())

if "ffi" in opts:
    to_pv.weight.mul_(float(opts["ffi"]) / ratio)
    print(f"ток пути на PV / на E: {ratio:.2f} → {float(opts['ffi']):.2f}")
PV4 = b.areas["letters"].layers["L4"]["PV"]
pv_cells = torch.arange(PV4.start, PV4.stop, device=b.device)[(b.field_cols[PV4.name] - mid).abs() <= half]


def live(ms: float) -> list[torch.Tensor]:
    """Прожить ms окнами WIN (сетчатка видит строку сессии); частоты окон."""

    out = []

    for _ in range(int(ms // WIN)):
        out.append(b.observe(WIN, s._view(WIN)))

    return out


def probe(reps: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
    """Без пластичности: прибавка на букву всех клеток [буква, клетка] и показы [показ, клетка]."""

    was = b.plasticity
    b.plasticity = False
    s.pointer = None
    gains = {c: [] for c in letters}
    hz = [0.0, 0.0, 0.0, 0.0]

    for _ in range(reps):
        for c in letters:
            s.line = ""
            live(300)
            rest = torch.stack(live(300)).mean(0)
            s.line, s.gaze = c, 0.0
            seen = torch.stack(live(300)).mean(0)
            gains[c].append(seen - rest)
            hz[0] += float(rest[cells].mean()) / (reps * len(letters))
            hz[1] += float(seen[cells].mean()) / (reps * len(letters))
            hz[2] += float(rest[pv_cells].mean()) / (reps * len(letters))
            hz[3] += float(seen[pv_cells].mean()) / (reps * len(letters))

    s.line = ""
    b.plasticity = was
    per = torch.stack([torch.stack(gains[c]).mean(0) for c in letters])
    shows = torch.stack([g for r in range(reps) for g in (gains[c][r] for c in letters)])
    probe.hz = hz
    return per, shows


def similarity(R: torch.Tensor, labels: list[str]) -> float:
    same = torch.tensor([[a == x for x in labels] for a in labels])
    eye_ = torch.eye(len(labels), dtype=torch.bool)
    R = R.float().cpu()
    R = R - R.mean(1, keepdim=True)
    R = R / R.norm(dim=1, keepdim=True).clamp(min=1e-6)
    C = R @ R.t()
    return float(C[same & ~eye_].mean() - C[~same].mean())


def alignment(dw: torch.Tensor, T: torch.Tensor) -> tuple[float, float]:
    """Средний по клеткам максимум (по буквам) корреляции Δw с шаблоном; контроль — шаблоны чужих клеток."""

    def best(templates: torch.Tensor) -> float:
        x = dw - dw.mean(1, keepdim=True)
        y = templates - templates.mean(2, keepdim=True)
        r = (x[None] * y).sum(2) / (x.norm(dim=1)[None] * y.norm(dim=2)).clamp(min=1e-9)
        return float(r.max(0).values.mean())

    return best(T), best(T[:, torch.randperm(T.shape[1], device=T.device)])


first_pref = None


def report(tag: str) -> None:
    global first_pref
    per, shows = probe()
    resp = per[:, cells]
    pref = resp.argmax(0)
    first_pref = pref if first_pref is None else first_pref
    r = resp.clamp(min=0)
    sparse = (r.mean(1) ** 2 / (r ** 2).mean(1).clamp(min=1e-9)).mean()
    counts = " ".join(f"{c}:{int((pref == j).sum())}" for j, c in enumerate(letters))
    top = resp.max(0).values
    rest = (resp.sum(0) - top) / (len(letters) - 1)
    index = (top - rest) / (top.abs() + rest.abs()).clamp(min=1e-6)
    strong = top > 2.0
    print(f"  {tag}: предпочитают {counts}; избирательность {float(index.mean()):+.2f} "
          f"(с прибавкой > 2 Гц — {int(strong.sum())} клеток, у них {float(index[strong].mean()) if strong.any() else 0:+.2f}); "
          f"различение ответа {similarity(shows[:, cells], [c for _ in range(3) for c in letters]):+.3f}; "
          f"L4 букв E покой {probe.hz[0]:.2f} → буква {probe.hz[1]:.2f}, PV {probe.hz[2]:.1f} → {probe.hz[3]:.1f} Гц; разреженность {float(sparse):.2f}; предпочтение как до обучения {float((pref == first_pref).float().mean()):.2f}", flush=True)


b.plasticity = False
s.line = ""
live(2500)
per, _ = probe()
T = per[:, pre].clamp(min=0)  # [буква, клетка, вход]
own = per[:, cells].argmax(0)
own_t = T[own, torch.arange(len(rows), device=b.device)]
key = own_t >= own_t.quantile(0.9, dim=1, keepdim=True)  # входы шаблона своей буквы
key = key & (own_t > 0)
print(f"клеток букв {len(cells)}, входов {pre.shape[1]}; входов шаблона у клетки {float(key.float().sum(1).mean()):.0f}")
report("до обучения")
b.plasticity = True
s.line = ""

gain = float(opts["gain"])

for epoch in range(int(opts["epochs"])):
    dw = {"указка": torch.zeros_like(own_t), "после": torch.zeros_like(own_t)}
    pairs = {(ph, kind, which): 0.0 for ph in dw for kind in ("LTP", "LTD") for which in ("шаблон", "прочие")}
    time_ms = {ph: 0.0 for ph in dw}
    rate_hz = {}

    for _ in range(int(opts["shows"])):
        s.line, s.gaze = rng.choice(letters), 0.0

        for phase, ms in (("указка", dwell), ("после", 200.0)):
            s.pointer = s.screen.width(s.line) / 2 if phase == "указка" and opts["pointer"] == "1" else None
            w0 = proj.weight[rows].float().clone()
            # Внимание на время показа: вход таламуса в V1 × gain.
            for p in thalamic:
                p.weight.mul_(gain)
            spikes_hz = 0.0
            prev_pre, prev_post = 0.0, 0.0

            for _ in range(int(ms // WIN)):
                r = b.observe(WIN, s._view(WIN)) * WIN / 1000.0  # спайков за окно
                a, c = r[pre], r[cells][:, None]
                # LTP: источник раньше клетки (своё или прошлое окно);
                # LTD: клетка раньше источника. Одно окно — в обоих.
                counts = {"LTP": (a + prev_pre) * c, "LTD": (c + prev_post) * a}

                for kind, n in counts.items():
                    for which, m in (("шаблон", key), ("прочие", ~key)):
                        pairs[(phase, kind, which)] += float((n * m).sum() / m.sum())

                prev_pre, prev_post = a, c
                spikes_hz += float(c.mean()) * 1000.0 / WIN / int(ms // WIN)

            for p in thalamic:
                p.weight.div_(gain)
            rate_hz[phase] = rate_hz.get(phase, 0.0) + spikes_hz / int(opts["shows"])
            dw[phase] += proj.weight[rows].float() - w0
            time_ms[phase] += ms

    print(f"эпоха {epoch + 1} ({int(opts['shows']) * (epoch + 1)} показов):")

    for phase, d in dw.items():
        got, null = alignment(d, T)
        print(f"  Δw за «{phase}»: |Δw| {float(d.abs().mean()):.4f}, среднее {float(d.mean()):+.4f}; у входов шаблона {float(d[key].mean()):+.4f}, у прочих {float(d[~key].mean()):+.4f}; "
              f"выравнивание с шаблоном {got:+.3f} (контроль {null:+.3f})")
        sec = time_ms[phase] / 1000.0
        print(f"     L4 букв при учёбе {rate_hz[phase]:.2f} Гц")
        print("     совпадений на синапс в с: " + ", ".join(
            f"{kind} шаблон {pairs[(phase, kind, 'шаблон')] / sec:.4f} / прочие {pairs[(phase, kind, 'прочие')] / sec:.4f}" for kind in ("LTP", "LTD")))

    report(f"после эпохи {epoch + 1}")
