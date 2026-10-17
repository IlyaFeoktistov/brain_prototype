"""
Где теряется различие букв по цепи V1 L4 → V1 L2/3 → буквы L4: свежий
мозг small без обучения, буквы в ямке (как letters_local), местные клетки
(место ±half px от середины буквы). На каждом уровне — различение
(сходство прибавок на одну букву против разных, как letters_local) и у
построенных путей — «вход» клетки: Σ вес × прибавка частоты источника
(собственная часть; среднее поле недостающих соседей одинаково для всех
клеток и узора не несёт). «Вход ↔ ответ» — корреляция буквенной части
(прибавка минус среднее клетки по буквам) входа и ответа по всем клеткам:
доходит ли различие входа до спайков. В V1 L2/3 ещё токи: возбуждающий
(i_syn) и тормозной (i_inh), средние по 10-мс отсчётам.

pv_off=1 — диагностика (не физиология): выход PV в V1 и в зоне букв
выключен — построенные синапсы PV → E и их среднее поле. Различение
появилось — дело в балансе возбуждения и торможения; нет — в структуре
входа.

    PYTHONPATH=. .venv/bin/python data/tools/v1_trace.py [letters=отл] [reps=3] [half=8] [pv_off=0] [seed=0]
"""

import sys

import torch

from brain.body import Eye
from brain.config import BrainConfig
from brain.mind import Mind
from brain.world import Screen

opts = {"letters": "отл", "reps": "3", "half": "8", "pv_off": "0", "seed": "0"}
opts.update(dict(kv.split("=", 1) for kv in sys.argv[1:]))
cfg = BrainConfig()
cfg.save_path = "/nonexistent/brain.pt"
cfg.autosave = False
cfg.anatomy = True
cfg.hand = True
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
V1 = b.areas["visual"].layers
LT = b.areas["letters"].layers


def local(pop) -> torch.Tensor:
    cols = b.field_cols.get(pop.name)
    idx = torch.arange(pop.start, pop.stop, device=b.device)
    return idx[(cols - mid).abs() <= half] if cols is not None else idx


if opts["pv_off"] == "1":
    pv = [g["PV"] for a in (V1, LT) for g in a.values()]
    for p in b.projections:
        if p.post is not None and any(getattr(p, "pre_pop", None) is q for q in pv) and p.post.name.endswith(".E"):
            p.weight.zero_()
    groups = sorted({int(b.mf_pre_id[q.start]) for q in pv})
    b.mf_inh[:, groups] = 0.0

levels = {
    "V1 L4 E": V1["L4"]["E"], "V1 L4 PV": V1["L4"]["PV"],
    "V1 L2/3 E": V1["L23"]["E"], "V1 L2/3 PV": V1["L23"]["PV"],
    "буквы L4 E": LT["L4"]["E"], "буквы L4 PV": LT["L4"]["PV"],
}
cells = {k: local(p) for k, p in levels.items()}
# Входы построенных путей: (уровень-получатель, путь).
paths = {"V1 L2/3 E": "visual: L4.E->E", "буквы L4 E": "visual -> letters"}
proj = {k: next(p for p in b.projections if p.name == n) for k, n in paths.items()}

b.observe(2500, eye.look(blank, 2500))
shown = []
gain = {k: [] for k in levels}
drive = {k: [] for k in paths}
cur = {"возб": [], "торм": []}
base = {}
l23 = cells["V1 L2/3 E"]


def currents(picture, ms: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Частоты за окно и средние токи E V1 L2/3 по 10-мс отсчётам."""

    hz, ex, inh = 0.0, 0.0, 0.0

    for _ in range(ms // 10):
        hz = hz + b.observe(10, eye.look(picture, 10))
        ex = ex + b.i_syn[l23].float()
        inh = inh + b.i_inh[l23].float()

    n = ms // 10
    return hz / n, ex / n, inh / n


for rep in range(int(opts["reps"])):
    for letter in opts["letters"]:
        b.observe(300, eye.look(blank, 300))
        rest, ex0, in0 = currents(blank, 300)
        seen, ex1, in1 = currents(scr.picture(letter, x0), 300)
        shown.append(letter)
        d = seen - rest

        for k, c in cells.items():
            gain[k].append(d[c])
            base.setdefault(k, []).append(float(rest[c].mean()))

        for k, p in proj.items():
            rows = cells[k] - levels[k].start
            # В мВ·Гц: вес → пик PSP получателя.
            drive[k].append((p.weight[rows].float() * d[p.pre[rows].long()]).sum(1) * float(b._psp_peak(levels[k])))

        cur["возб"].append(ex1 - ex0)
        cur["торм"].append(in1 - in0)

same = torch.tensor([[a == x for x in shown] for a in shown])
eye_ = torch.eye(len(shown), dtype=torch.bool)


def similarity(rows: list[torch.Tensor]) -> float:
    R = torch.stack(rows).float().cpu()
    R = R - R.mean(1, keepdim=True)
    R = R / R.norm(dim=1, keepdim=True).clamp(min=1e-6)
    C = R @ R.t()
    return float(C[same & ~eye_].mean() - C[~same].mean())


def letter_part(rows: list[torch.Tensor]) -> torch.Tensor:
    """Прибавка минус среднее клетки по всем показам — что зависит от буквы."""

    R = torch.stack(rows).float().cpu()
    return R - R.mean(0, keepdim=True)


def corr(a: torch.Tensor, c: torch.Tensor) -> float:
    return float(torch.corrcoef(torch.stack([a.flatten(), c.flatten()]))[0, 1])


print(f"pv_off={opts['pv_off']} буквы {opts['letters']} × {opts['reps']}, местных ±{half:g} px")

for k, rows in gain.items():
    R = torch.stack(rows)
    top = R.mean(0).topk(max(1, R.shape[1] // 20)).values.mean()
    print(f"  {k:12s} клеток {R.shape[1]:5d}  покой {sum(base[k]) / len(base[k]):5.1f}, прибавка {float(R.mean()):+5.2f} Гц, верхние 5% {float(top):+5.1f}  различение {similarity(rows):+.3f}")

for k, rows in drive.items():
    print(f"  вход {paths[k]:18s} различение {similarity(rows):+.3f}; буквенная часть входа ±{float(letter_part(rows).std()):.1f} (пик PSP мВ × Гц); вход ↔ ответ {corr(letter_part(rows), letter_part(gain[k])):+.3f}")

ex, inh = letter_part(cur["возб"]), letter_part(cur["торм"])
print(f"  V1 L2/3 E токи на букву: возб {float(torch.stack(cur['возб']).mean()):+.2f}, торм {float(torch.stack(cur['торм']).mean()):+.2f}; "
      f"различение возб {similarity(cur['возб']):+.3f}, торм {similarity(cur['торм']):+.3f}; "
      f"возб ↔ ответ {corr(ex, letter_part(gain['V1 L2/3 E'])):+.3f}, торм ↔ возб {corr(inh, ex):+.3f}")
