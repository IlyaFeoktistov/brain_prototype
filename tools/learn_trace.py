"""
Учится ли цепь фактам (DESIGN «qa_ear»): диагностический режим (IC выкл.
флагом, inh_off — местное торможение CN и MGN, pv_off — PV → E смысловой
L4), факты 06 голосом × passes с пластичностью. По участкам цепи:

- изменение весов: доля изменившихся синапсов, сумма роста и спада
  отдельно (среднее ≈ 0 может прятать взаимное гашение), |Δw| к весу;
- связь с фактами: корреляция Δw синапса с совместной активностью его
  пары на фактах (Σ по фактам (r_пре − среднее) · (r_пост − среднее));
- частоты CA3 / CA1 во время фактов и в тишине;
- представление: узор смысла и ПФК после questions вопросов до учёбы и
  после — сходство (1 — не изменился).

    PYTHONPATH=. .venv/bin/python tools/learn_trace.py [passes=2] [questions=5] [inh_off=cn,mgn] [pv_off=association.L4] [seed=0]
"""

import re
import sys
from pathlib import Path

import torch

from brain.body import Cochlea
from brain.config import BrainConfig
from brain.mind import Mind
from brain.world import Voice

opts = {"passes": "2", "questions": "5", "inh_off": "cn,mgn", "pv_off": "association.L4", "pause": "500", "recall": "300", "seed": "0"}
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
NAMES = ("auditory -> association", "association: E->E", "association -> prefrontal", "entorhinal -> ca3",
         "ca3: коллатерали", "ca3 -> ca1", "ca1 -> association")
paths = {n: p for p in b.projections for n in NAMES if p.name == n}
missing = [n for n in NAMES if n not in paths]
print("участки:", ", ".join(f"{n} {tuple(p.weight.shape)} {'учится' if p.plastic else 'не учится'}" for n, p in paths.items()),
      ("; нет: " + ", ".join(missing)) if missing else "")
before = {n: p.weight.float().clone() for n, p in paths.items()}


def body(path: str) -> list[str]:
    return [l for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")]


def hear(text: str) -> torch.Tensor:
    raster = ear.hear(*voice.say(text))
    sensory = torch.zeros(raster.shape[0], b.n_receptors, device=b.device)
    sensory[:, : raster.shape[1]] = raster
    return b.observe(raster.shape[0] * cfg.dt_ms, sensory)


facts = body("data/course/06-факты.txt")
questions = [l.split("|")[0] for l in body("data/course/07-вопрос-ответ.txt")][: int(opts["questions"])]
L23 = {a: b.areas[a].layers["L23"]["E"] for a in ("association", "prefrontal")}


def recall() -> dict[str, torch.Tensor]:
    out = {a: [] for a in L23}
    for q in questions:
        b.observe(500.0)
        hear(re.sub(r"[^а-яё ?]", " ", q.lower()))
        r = b.observe(float(opts["recall"]))
        for a, pop in L23.items():
            out[a].append(r[pop.slice].cpu())
    return {a: torch.stack(v) for a, v in out.items()}


b.plasticity = False
b.run(2500.0)
quiet = b.observe(1000.0)
print(f"тишина: ca3 {float(quiet[b.areas['ca3'].E.slice].mean()):.1f} Гц, ca1 {float(quiet[b.areas['ca1'].E.slice].mean()):.1f}")
pattern0 = recall()
b.plasticity = True
during = []

for p in range(int(opts["passes"])):
    for f in facts:
        during.append(hear(f).cpu())
        b.observe(float(opts["pause"]))

b.plasticity = False
R = torch.stack(during)
print(f"во время фактов: ca3 {float(R[:, b.areas['ca3'].E.slice].mean()):.1f} Гц, ca1 {float(R[:, b.areas['ca1'].E.slice].mean()):.1f}, "
      f"смысл L2/3 {float(R[:, L23['association'].slice].mean()):.1f}, ПФК L2/3 {float(R[:, L23['prefrontal'].slice].mean()):.1f}")
Rc = R - R.mean(0, keepdim=True)

for n, p in paths.items():
    w0 = before[n]
    d = (p.weight.float() - w0).cpu()
    changed = d.abs() > 1e-6
    up, down = float(d.clamp(min=0).sum()), float(d.clamp(max=0).sum())
    pre = p.pre.long().cpu()
    post = torch.arange(p.post.start, p.post.stop)[:, None].expand_as(pre)
    # Совместная активность пары на фактах (ковариация по фактам).
    co = (Rc[:, pre] * Rc[:, post]).sum(0)
    x, y = d.flatten(), co.flatten()
    corr = float(torch.corrcoef(torch.stack([x, y]))[0, 1]) if bool(changed.any()) else float("nan")
    print(f"  {n:26s} изменилось {float(changed.float().mean()) * 100:5.1f}%  рост {up:+9.2f} спад {down:+9.2f}  "
          f"|Δw|/вес {float(d.abs().mean() / w0.abs().mean().clamp(min=1e-9)) * 100:6.3f}%  Δw ↔ совм. активность {corr:+.3f}", flush=True)

pattern1 = recall()

for a in L23:
    x, y = pattern0[a], pattern1[a]
    sims = [float(torch.nn.functional.cosine_similarity(x[k] - x.mean(0), y[k] - y.mean(0), dim=0)) for k in range(len(questions))]
    print(f"  узор {a} после вопроса: до ↔ после {sum(sims) / len(sims):+.3f} (1 — не изменился)")
