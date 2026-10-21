"""
Тормозная пластичность (Vogels et al. 2011, по частотам — engine.
inhibitory_plasticity) только на синапсах PV → E смысловой L4: делает ли
она торможение согласованным с возбуждением клетки и возвращает ли
различение слов (DESIGN «Колонка смысловой L4»). Слова голосом, обход
ствола, свежий мозг small; прочая пластичность выключена. Эпоха учёбы —
слова × reps, правило после каждого слова (окно — слово + after мс);
после эпохи — проверка без учёбы на отдельных показах (слова × test):
различение и доля слова у тормозного тока и спайков L4 E, частота L4 E,
средняя сила PV → E (ток веса). Норма правила — целевая частота L4 E
(column_target_hz), скорость и потолок — cfg.inhibitory_rate / vogels_cap.

    PYTHONPATH=. .venv/bin/python tools/vogels_trace.py [words=кот,дом,кит,мама] [reps=8] [test=4] [epochs=4] [inh_off=cn,mgn] [seed=0]
"""

import random
import sys

import torch

from brain.body import Cochlea
from brain.config import BrainConfig
from brain.mind import Mind
from brain.world import Voice

opts = {"words": "кот,дом,кит,мама", "reps": "8", "test": "4", "epochs": "4", "inh_off": "cn,mgn", "after": "100", "seed": "0"}
opts.update(dict(kv.split("=", 1) for kv in sys.argv[1:]))
cfg = BrainConfig()
cfg.save_path = "/nonexistent/brain.pt"
cfg.autosave = False
cfg.dt_ms = 0.5
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

E = b.areas["association"].layers["L4"]["E"]
pv = None

for p in b.projections:
    p.vogels = p.name == "association: L4.PV->L4.E"
    if p.vogels:
        p.cap = cfg.vogels_cap
        pv = p

cfg.inhibitory_target_hz = dict((l, e) for l, e, *_ in cfg.column_target_hz)["L4"]
i = [s.name for s in b.sensors].index("ear")
ear = Cochlea(cfg, b.sensor_offsets[i], b.sensors[i].size, b.device)
voice = Voice()
words = opts["words"].split(",")
rng = random.Random(cfg.seed)
b.run(2500.0)
print(f"PV → E смысловой L4: {tuple(pv.weight.shape)}, вес {float(pv.weight.float().mean()):+.3f}; норма {cfg.inhibitory_target_hz} Гц, "
      f"скорость {cfg.inhibitory_rate}, потолок {cfg.vogels_cap}")


def present(word: str, learn: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Слово на ухо; (спайки L4 E, тормозной ток L4 E — средние за слово, частоты окна)."""

    b.observe(300.0)
    raster = ear.hear(*voice.say(word))
    sensory = torch.zeros(raster.shape[0], b.n_receptors, device=b.device)
    sensory[:, : raster.shape[1]] = raster
    steps = int(10.0 / cfg.dt_ms)
    spikes, inh, n = 0.0, 0.0, 0
    window = 0.0

    for start in range(0, raster.shape[0] - steps + 1, steps):
        r = b.observe(10.0, sensory[start: start + steps])
        spikes = spikes + r[E.slice]
        inh = inh + b.i_inh[E.slice].float()
        window = window + r
        n += 1

    tail = b.observe(float(opts["after"]))
    rates = (window * 10.0 + tail * float(opts["after"])) / (n * 10.0 + float(opts["after"]))

    if learn:
        b.plasticity = True
        b.inhibitory_plasticity(rates)
        b.plasticity = False

    return (spikes / n).cpu(), (inh / n).cpu()


def measure(tag: str) -> None:
    order = [w for w in words for _ in range(int(opts["test"]))]
    rng.shuffle(order)
    S, I = [], []

    for w in order:
        s, inh = present(w, False)
        S.append(s)
        I.append(inh)

    def stats(rows):
        R = torch.stack(rows)
        common = R.mean(0)
        groups = {w: R[[j for j, x in enumerate(order) if x == w]] for w in words}
        spec = torch.stack([g.mean(0) - common for g in groups.values()]).pow(2).mean()
        noise = torch.cat([g - g.mean(0, keepdim=True) for g in groups.values()]).pow(2).mean() * len(order) / (len(order) - len(words))
        X = R - common
        X = X / X.norm(dim=1, keepdim=True).clamp(min=1e-9)
        C = X @ X.t()
        same = [float(C[a, c]) for a in range(len(order)) for c in range(len(order)) if a != c and order[a] == order[c]]
        diff = [float(C[a, c]) for a in range(len(order)) for c in range(len(order)) if order[a] != order[c]]
        return float(spec / (spec + noise)), sum(same) / len(same) - sum(diff) / len(diff), float(common.mean())

    sf, sd, sm = stats(S)
    inf, ind, im = stats(I)
    print(f"  {tag:10s} спайки L4 E {sm:5.1f} Гц: доля слова {sf:.3f}, различение {sd:+.3f} | тормозной ток {im:+6.1f}: доля слова {inf:.3f}, "
          f"различение {ind:+.3f} | вес PV → E {float(pv.weight.float().mean()):+.3f}", flush=True)


measure("до учёбы")

for epoch in range(int(opts["epochs"])):
    order = [w for w in words for _ in range(int(opts["reps"]))]
    rng.shuffle(order)

    for w in order:
        present(w, True)

    measure(f"эпоха {epoch + 1}")
