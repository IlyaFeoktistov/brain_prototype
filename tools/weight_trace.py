"""
Что правило пластичности делает с весами цепи (DESIGN «Учится ли цепь
фактам»): тот же диагностический режим и факты 06 голосом, что learn_trace.
До учёбы и после каждого прохода по участкам: распределение (мин, медиана,
макс, σ, доли у нуля и у потолка, гистограмма), парное сравнение тех же
синапсов (наклон «после ~ до»: < 1 — стягиваются к среднему; сдвиг к
краям), сохранение суммы входа у каждой клетки. Во время фактов — пары
спайков у выборки синапсов (окна 5 мс): «вход раньше клетки» (своё или
прошлое окно источника, окно клетки) и «клетка раньше входа»; средний Δw
у причинных (первых больше), обратных и без близких спайков.

    PYTHONPATH=. .venv/bin/python tools/weight_trace.py [passes=2] [sample=20000] [inh_off=cn,mgn] [pv_off=association.L4] [seed=0]
"""

import sys
from pathlib import Path

import torch

from brain.body import Cochlea
from brain.config import BrainConfig
from brain.mind import Mind
from brain.world import Voice

opts = {"passes": "2", "sample": "20000", "inh_off": "cn,mgn", "pv_off": "association.L4", "pause": "500", "seed": "0"}
opts.update(dict(kv.split("=", 1) for kv in sys.argv[1:]))
cfg = BrainConfig()
cfg.save_path = "/nonexistent/brain.pt"
cfg.autosave = False
cfg.dt_ms = 0.5
cfg.seed = int(opts["seed"])
torch.manual_seed(cfg.seed)
b = Mind(cfg).brain

for name in filter(None, opts["inh_off"].split(",")):
    nucleus = b.areas[name]
    for p in b.projections:
        if p.name == f"{name}: I->E":
            p.weight.zero_()
    b.mf_inh[int(b.mf_post_id[nucleus.E.start]), int(b.mf_pre_id[nucleus.I.start])] = 0.0

for spec in filter(None, opts["pv_off"].split(",")):
    area, layer = spec.split(".")
    g = b.areas[area].layers[layer]
    for p in b.projections:
        if p.name == f"{area}: {layer}.PV->{layer}.E":
            p.weight.zero_()
    b.mf_inh[int(b.mf_post_id[g["E"].start]), int(b.mf_pre_id[g["PV"].start])] = 0.0

i = [s.name for s in b.sensors].index("ear")
ear = Cochlea(cfg, b.sensor_offsets[i], b.sensors[i].size, b.device)
voice = Voice()
NAMES = ("auditory -> association", "association -> prefrontal", "association: E->E", "entorhinal -> ca3", "ca3 -> ca1", "ca3: коллатерали")
paths = {n: p for p in b.projections for n in NAMES if p.name == n}
start = {n: p.weight.float().clone() for n, p in paths.items()}
birth = {n: p.birth.float().sum(1) if getattr(p, "birth", None) is not None else None for n, p in paths.items()}
g = torch.Generator(device=b.device).manual_seed(5)
pick = {n: torch.randint(0, p.weight.numel(), (int(opts["sample"]),), device=b.device, generator=g) for n, p in paths.items()}
pairs = {n: torch.zeros(2, int(opts["sample"]), device=b.device) for n in paths}  # [причинные, обратные]


def report(tag: str) -> None:
    print(f"{tag}:")

    for n, p in paths.items():
        w = p.weight.float()
        w0 = start[n]
        cap = float(getattr(p, "cap", 0.0) or w.abs().max())
        q = torch.quantile(w.flatten()[torch.randperm(w.numel(), device=w.device)[:200000]], torch.tensor([0.5], device=w.device))
        at0 = float((w.abs() < 1e-6).float().mean())
        atcap = float((w.abs() > 0.98 * cap).float().mean()) if cap > 0 else 0.0
        hist = torch.histc(w.flatten().clamp(-cap, cap), bins=24, min=min(0.0, float(w.min())), max=max(cap, float(w.max())))
        bars = "".join(" ▁▂▃▄▅▆▇█"[min(8, int(8 * float(h) / float(hist.max().clamp(min=1))))] for h in hist)
        line = (f"  {n:26s} мин {float(w.min()):+.3f} мед {float(q[0]):+.3f} макс {float(w.max()):+.3f} σ {float(w.std()):.3f} "
                f"у нуля {at0 * 100:4.1f}% у потолка {atcap * 100:4.1f}% |{bars}|")

        if tag != "до учёбы":
            x, y = w0.flatten(), w.flatten()
            slope = float(((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum().clamp(min=1e-12))
            corr = float(torch.corrcoef(torch.stack([x, y]))[0, 1])
            line += f"\n      после ~ до: наклон {slope:+.3f}, корр. {corr:+.3f}"

            if birth[n] is not None:
                dev = (w.sum(1) - birth[n]).abs() / birth[n].abs().clamp(min=1e-9)
                line += f"; сумма входа клетки: отклонение медиана {float(dev.median()) * 100:.2f}%, макс {float(dev.max()) * 100:.1f}%"

        print(line, flush=True)


def body(path: str) -> list[str]:
    return [l for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")]


facts = body("data/course/06-факты.txt")
b.plasticity = False
b.run(2500.0)
report("до учёбы")
b.plasticity = True
WIN = 5.0
prev = None

for p_ in range(int(opts["passes"])):
    for f in facts:
        raster = ear.hear(*voice.say(f))
        steps = int(WIN / cfg.dt_ms)

        for s in range(0, raster.shape[0] - steps + 1, steps):
            chunk = torch.zeros(steps, b.n_receptors, device=b.device)
            chunk[:, : raster.shape[1]] = raster[s: s + steps]
            spikes = (b.observe(WIN, chunk) > 0).float()

            if prev is not None:
                for n, p in paths.items():
                    flat = pick[n]
                    pre = p.pre.flatten().long()[flat]
                    post = p.post.start + flat // p.weight.shape[1]
                    pairs[n][0] += (spikes[pre] + prev[pre]) * spikes[post]  # вход раньше (или вместе)
                    pairs[n][1] += (spikes[post] + prev[post]) * spikes[pre]  # клетка раньше (или вместе)

            prev = spikes

        b.observe(float(opts["pause"]))
        prev = None

    report(f"после прохода {p_ + 1}")

b.plasticity = False
print("Δw по порядку спайков (выборка синапсов, оба прохода):")

for n, p in paths.items():
    d = (p.weight.float() - start[n]).flatten()[pick[n]]
    ltp, ltd = pairs[n]
    groups = {"причинные": ltp > ltd, "обратные": ltd > ltp, "без пар": (ltp + ltd) == 0}
    print(f"  {n:26s} " + "; ".join(f"{k} {int(m.sum())}: Δw {float(d[m].mean()) if bool(m.any()) else float('nan'):+.4f}" for k, m in groups.items()), flush=True)
