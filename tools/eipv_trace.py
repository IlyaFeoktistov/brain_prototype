"""
Согласованные E ↔ PV в колонке смысловой L4 (Znamenskiy et al. 2024, V1
мыши: 64% пар E–PV связаны взаимно; сила E → PV и PV → E у взаимных пар
коррелирует, r 0.57; сходство ответов объясняет ~15% вариации силы E → PV
и ~18% PV → E — r ≈ 0.4; вероятность связи от сходства не зависит).
Свежий мозг small; inh_off=cn,mgn — диагностика: местное торможение главных
клеток этих ядер выключено; fit=имя — фон из ~/.brain/fits/имя.json
(anatomy_fit out=имя).

1. Проба (PV → E смысловой L4 выкл.): средний ответ каждой клетки L4 E и
   PV на каждое слово; настройка пары — корреляция её ответов по словам.
2. Связи как есть: корреляция силы E → PV и PV → E со сходством пары, доля
   взаимно связанных пар среди связанных.
3. Проверка (PV вкл., слова × test): различение спайков смысла L4 и ПФК
   L4 / L2/3 — при связях как есть и при согласованных: сила каждой
   связи сдвинута к сходству пары (w · exp(k · сходство), сумма входа
   клетки прежняя), k — чтобы корреляция силы со сходством стала target.

geometry=1 (по умолчанию) — колонки без карты как лист коры (cfg.column_geometry): взаимность
по расстоянию между клетками пары, частоты в покое; ee_sigma — σ E → E
на листе, мкм (0 — партнёры E → E случайно по зоне).

    PYTHONPATH=. .venv/bin/python tools/eipv_trace.py [geometry=1] [ee_sigma=124.6] [words=кот,дом,кит,мама] [probe=4] [test=8] [target=0.4] [inh_off=] [fit=ic-off] [seed=0]
"""

import random
import sys

import torch

from brain.body import Cochlea
from brain.config import BrainConfig
from brain.mind import Mind
from brain.world import Voice

opts = {"geometry": "1", "ee_sigma": "124.6", "words": "кот,дом,кит,мама", "probe": "4", "test": "8", "target": "0.4", "inh_off": "", "fit": "", "seed": "0"}
opts.update(dict(kv.split("=", 1) for kv in sys.argv[1:]))
cfg = BrainConfig()
cfg.save_path = "/nonexistent/brain.pt"
cfg.autosave = False
cfg.dt_ms = 0.5
if opts["fit"]:
    from brain import anatomy

    anatomy.use_fit(cfg, opts["fit"])
cfg.column_geometry = opts["geometry"] == "1"
cfg.sigma_ee_um = float(opts["ee_sigma"])
cfg.seed = int(opts["seed"])
torch.manual_seed(cfg.seed)
b = Mind(cfg).brain
b.plasticity = False

for name in filter(None, opts["inh_off"].split(",")):
    nucleus = b.areas[name]
    for p in b.projections:
        if p.name == f"{name}: I->E":
            p.weight.zero_()
    b.mf_inh[int(b.mf_post_id[nucleus.E.start]), int(b.mf_pre_id[nucleus.I.start])] = 0.0

L4 = b.areas["association"].layers["L4"]
E, PV = L4["E"], L4["PV"]
e2pv = next(p for p in b.projections if p.name == "association: L4.E->L4.PV")
pv2e = next(p for p in b.projections if p.name == "association: L4.PV->L4.E")
w_e2pv, w_pv2e = e2pv.weight.clone(), pv2e.weight.clone()
mf_q, mf_p = int(b.mf_post_id[E.start]), int(b.mf_pre_id[PV.start])
mf_pv = float(b.mf_inh[mf_q, mf_p])
i = [s.name for s in b.sensors].index("ear")
ear = Cochlea(cfg, b.sensor_offsets[i], b.sensors[i].size, b.device)
voice = Voice()
words = opts["words"].split(",")
rng = random.Random(cfg.seed)
PFC = b.areas["prefrontal"].layers
watch = {"смысл L4 E": E, "ПФК L4 E": PFC["L4"]["E"], "ПФК L2/3 E": PFC["L23"]["E"]}


def hear(word: str) -> torch.Tensor:
    b.observe(300.0)
    raster = ear.hear(*voice.say(word))
    sensory = torch.zeros(raster.shape[0], b.n_receptors, device=b.device)
    sensory[:, : raster.shape[1]] = raster
    return b.observe(raster.shape[0] * cfg.dt_ms + 100.0, torch.cat([sensory, torch.zeros(200, b.n_receptors, device=b.device)]))


def discrimination(rows: list[torch.Tensor], order: list[str]) -> float:
    R = torch.stack(rows).float()
    X = R - R.mean(0)
    X = X / X.norm(dim=1, keepdim=True).clamp(min=1e-9)
    C = X @ X.t()
    same = [float(C[a, c]) for a in range(len(order)) for c in range(len(order)) if a != c and order[a] == order[c]]
    diff = [float(C[a, c]) for a in range(len(order)) for c in range(len(order)) if order[a] != order[c]]
    return sum(same) / len(same) - sum(diff) / len(diff)


# 1. Проба без PV → E: настройка клеток по словам.
pv2e.weight.zero_()
b.mf_inh[mf_q, mf_p] = 0.0
b.run(2500.0)
resp = {w: [] for w in words}
order = [w for w in words for _ in range(int(opts["probe"]))]
rng.shuffle(order)

for w in order:
    resp[w].append(hear(w).cpu())

tuning = torch.stack([torch.stack(resp[w]).mean(0) for w in words])  # [слово, нейрон]
tune_e = tuning[:, E.slice]
tune_pv = tuning[:, PV.slice]


def zscore(x: torch.Tensor) -> torch.Tensor:
    x = x - x.mean(0, keepdim=True)
    return x / x.norm(dim=0, keepdim=True).clamp(min=1e-9)


ze, zp = zscore(tune_e), zscore(tune_pv)


def pair_similarity(proj) -> torch.Tensor:
    """Сходство настройки пары для каждого синапса пути (получатель × вход)."""

    pre = proj.pre.long().cpu()
    if proj is e2pv:  # E → PV: получатель PV, вход E
        return (zp[:, :, None] * ze[:, pre - E.start].permute(0, 1, 2)).sum(0) if False else \
            torch.einsum("wp,wpk->pk", zp, ze[:, pre - E.start])
    return torch.einsum("we,wek->ek", ze, zp[:, pre - PV.start])


sim_e2pv, sim_pv2e = pair_similarity(e2pv), pair_similarity(pv2e)


def corr(a: torch.Tensor, c: torch.Tensor) -> float:
    return float(torch.corrcoef(torch.stack([a.flatten().float(), c.flatten().float()]))[0, 1])


# 2. Структура как есть: связь силы со сходством, взаимность.
pre_pv = pv2e.pre.long().cpu() - PV.start  # PV-входы каждой E
pre_e = e2pv.pre.long().cpu() - E.start  # E-входы каждой PV
inbound = set(zip(torch.arange(PV.size)[:, None].expand_as(pre_e).flatten().tolist(), pre_e.flatten().tolist()))
recip = sum((int(p), e) in inbound for e, row in enumerate(pre_pv.tolist()) for p in row) / pre_pv.numel()
pairs_e, pairs_p = [], []
for e, row in enumerate(pre_pv.tolist()):
    for p in row:
        pairs_e.append(e); pairs_p.append(int(p))
pe, pp = torch.tensor(pairs_e), torch.tensor(pairs_p)
mutual = torch.tensor([(int(p), int(e)) in inbound for e, p in zip(pe.tolist(), pp.tolist())])
if cfg.column_geometry:
    pos_e = b._sheet_place(next(s for s in cfg.areas if s.name == "association"), b.areas["association"], E).cpu()
    pos_p = b._sheet_place(next(s for s in cfg.areas if s.name == "association"), b.areas["association"], PV).cpu()
    d = (pos_e[pe] - pos_p[pp]).norm(dim=1) * 1000
    bins = [0, 50, 100, 200, 400, 10000]
    print("взаимность PV → E по расстоянию пары: " + ", ".join(
        f"{bins[j]}–{bins[j + 1]} мкм {float(mutual[(d >= bins[j]) & (d < bins[j + 1])].float().mean()) * 100:.0f}% ({int(((d >= bins[j]) & (d < bins[j + 1])).sum())})"
        for j in range(len(bins) - 1) if bool(((d >= bins[j]) & (d < bins[j + 1])).any())) + f"; медиана расстояния {float(d.median()):.0f} мкм")
rest = b.observe(1000.0)
print("покой: " + ", ".join(f"{a} {l} E {float(rest[b.areas[a].layers[l]['E'].slice].mean()):.1f} / PV {float(rest[b.areas[a].layers[l]['PV'].slice].mean()):.1f}"
                        for a in ("association", "prefrontal") for l in ("L23", "L4")))
print(f"проба: L4 E избирательных (разброс по словам > 0) — {float((tune_e.std(0) > 0).float().mean()) * 100:.0f}%")
print(f"связи как есть: E → PV ↔ сходство {corr(w_e2pv.cpu(), sim_e2pv):+.3f}, PV → E ↔ сходство {corr(w_pv2e.cpu().abs(), sim_pv2e):+.3f}; "
      f"взаимных среди связей PV → E — {recip * 100:.0f}% (V1: 64% пар)")


def cotune(w0: torch.Tensor, sim: torch.Tensor) -> torch.Tensor:
    """Сила связи × exp(k · сходство), сумма входа клетки прежняя; k — до целевой корреляции."""

    target = float(opts["target"])
    mag = w0.abs().cpu().float()

    for k in [x / 4 for x in range(1, 200)]:
        new = mag * torch.exp(k * sim)
        new = new * (mag.sum(1, keepdim=True) / new.sum(1, keepdim=True))
        if corr(new, sim) >= target:
            break

    return (new * w0.sign().cpu()).to(w0.dtype).to(w0.device), k


tuned_e2pv, k1 = cotune(w_e2pv, sim_e2pv)
tuned_pv2e, k2 = cotune(w_pv2e, sim_pv2e)


def test(tag: str) -> None:
    b.run(1000.0)
    order = [w for w in words for _ in range(int(opts["test"]))]
    rng.shuffle(order)
    rows = {k: [] for k in watch}

    for w in order:
        r = hear(w)
        for k, pop in watch.items():
            rows[k].append(r[pop.slice].cpu())

    print(f"  {tag:14s} " + "; ".join(f"{k} {float(torch.stack(v).mean()):.1f} Гц, различение {discrimination(v, order):+.3f}" for k, v in rows.items()), flush=True)


# 3. Проверка с PV → E.
b.mf_inh[mf_q, mf_p] = mf_pv
e2pv.weight.copy_(w_e2pv)
pv2e.weight.copy_(w_pv2e)
test("как есть")
e2pv.weight.copy_(tuned_e2pv)
pv2e.weight.copy_(tuned_pv2e)
print(f"  согласованные: E → PV ↔ сходство {corr(tuned_e2pv.cpu(), sim_e2pv):+.3f} (k {k1}), PV → E ↔ сходство "
      f"{corr(tuned_pv2e.cpu().abs(), sim_pv2e):+.3f} (k {k2})")
test("согласованные")
