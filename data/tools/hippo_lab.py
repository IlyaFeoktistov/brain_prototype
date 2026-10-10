"""
Лаборатория гиппокампа: только смысловая кора (вход и выход) и гиппокамп,
без языка, цели и рабочей памяти — динамика за секунды.

    .venv/bin/python data/tools/hippo_lab.py [нейронов] [episodes=N] [key=value ...]

«Слово» — concept_cells клеток смысловой коры (как место слова в мозге). N эпизодов (по умолчанию 20) вида «частое редкое ? ответ»
(частое — одно из 3, «?» — общее для всех): запись по одному разу, затем
подсказка первыми тремя словами — ранг ответа по выученному току CA1 →
смысл и по спайкам коры; затем вспышки CA3 (сон) — какие слова будит и
принадлежат ли они одному эпизоду.
"""

import random
import sys
import time

import torch

from brain.config import AreaSpec, BrainConfig, hippocampus_areas, hippocampus_pathways
from brain.concepts import Concepts
from brain.engine import SpikingBrain

total = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 20000
cfg = BrainConfig()
cfg.size = str(total)
cfg.autosave = False
cfg.neural_hippocampus = True

EPISODES = 20
for kv in sys.argv[1:]:
    if kv.startswith("episodes="):
        EPISODES = int(kv.split("=")[1])
    elif "=" in kv:
        k, v = kv.split("=", 1)
        old = getattr(cfg, k)
        setattr(cfg, k, (v == "1") if isinstance(old, bool) else type(old)(v))
        print(f"{k} = {getattr(cfg, k)}")

cfg.areas = [
    AreaSpec("auditory", "слух (не используется)", 0.01, k_ee=10, ee_plastic=False),
    AreaSpec("association", "смысл", 0.66, k_ee=100),
] + hippocampus_areas()
names = {a.name for a in cfg.areas}
cfg.pathways = [p for p in hippocampus_pathways() if p.src in names and p.dst in names]

t0 = time.time()
brain = SpikingBrain(cfg)
rng = random.Random(1)
assoc = brain.areas["association"].E
CELLS, WORD_MS, REST_MS, WORD_MV = cfg.concept_cells, cfg.stimulus_ms, cfg.rest_ms, cfg.concept_mv

# Словарь: места слов — компактные пятна на карте смысловой коры, как в
# мозге (Concepts), но не рядом с прозвучавшим перед ним: иначе ответ
# лежит на карте возле «?», и топографический вход EC смешивает их коды
# без всякой памяти ([замер] без записи CA1 давала узор ответа ×17).
WORDS = 4 + 2 * EPISODES
common = [0, 1, 2]
QUESTION = 3
episodes = [[common[i % 3], 4 + i, QUESTION, 4 + EPISODES + i] for i in range(EPISODES)]
places = Concepts(assoc.size, CELLS, cfg.seed, brain.device)
for i in rng.sample(range(WORDS), WORDS):
    places.get(str(i), allocate=True)
vocab = [places.get(str(i), allocate=False).to(brain.device) for i in range(WORDS)]


def hear(word: int | None, ms: float) -> torch.Tensor:
    if word is None:
        brain.feel("association", None, 0.0)
    else:
        brain.feel("association", vocab[word], WORD_MV)
    rates = brain.observe(ms)
    brain.feel("association", None, 0.0)
    return rates


def share(rates, area):
    return float((rates[brain.areas[area].E.slice] > 0).float().mean())


# Калибровка: случайные слова, гомеостаз находит рабочую точку.
brain.plasticity = True
brain.developing = True
for _ in range(cfg.calibration_tokens):
    rates = hear(rng.randrange(WORDS), WORD_MS)
    brain.homeostasis(rates)
    if _ >= cfg.calibration_tokens - 20:
        calib = {n: calib.get(n, 0.0) + share(rates, n) / 20 for n in ("entorhinal", "dentate", "ca3", "ca1")} if "calib" in dir() else {n: share(rates, n) / 20 for n in ("entorhinal", "dentate", "ca3", "ca1")}
brain.developing = False
print("конец калибровки активны: " + ", ".join(f"{n} {v:.1%}" for n, v in calib.items()))
brain.eligibility.zero_()

# Запись: каждый эпизод один раз, бодрствование — ацетилхолин высокий.
brain.set_acetylcholine(1.0)
log = {n: [] for n in ("entorhinal", "dentate", "ca3", "ca1")}
# Узор CA3 каждого эпизода — клетки, стрелявшие на любом его слове.
CA3 = brain.areas["ca3"].E
traces, answers, answers1 = [], [], []
CA1 = brain.areas["ca1"].E
for episode in episodes:
    trace = torch.zeros(CA3.size, dtype=torch.bool, device=brain.device)
    for word in episode:
        rates = hear(word, WORD_MS)
        brain.homeostasis(rates)
        brain.encode(cfg.hippocampus_encode)
        trace |= rates[CA3.slice] > 0
        for n in log:
            log[n].append(share(rates, n))
    # Узор CA3 на слове-ответе — его и должна достроить подсказка.
    answers.append(rates[CA3.slice] > 0)
    answers1.append(rates[CA1.slice] > 0)
    traces.append(trace)
    hear(None, REST_MS)
traces = torch.stack(traces).float()

# Выученные связи CA3 → CA3: вес пар клеток одного эпизода против прочих.
proj = next(p for p in brain.plastic if p.name == "ca3: E->E")
pre_local = proj.pre.long() - CA3.start
member = traces.bool()
inside = torch.zeros_like(proj.weight, dtype=torch.bool)
for trace in member:
    inside |= trace[:, None] & trace[pre_local]
w = proj.weight.float()
print(f"CA3 → CA3: вес внутри эпизода {float(w[inside].mean()):.3f} ({int(inside.sum())} синапсов), "
      f"прочие {float(w[~inside].mean()):.3f}, потолок {proj.cap}")
print("при записи активны: " + ", ".join(f"{n} {sum(v) / len(v):.1%}" for n, v in log.items()))

# Вспоминание: первые три слова слушает (ацетилхолин высокий), на
# последнем открывает ворота «говори» — ацетилхолин падает, CA3 и выход свободны.
ranks, spike_ranks, completion, completion1 = [], [], [], []
with brain.frozen():
    for episode in episodes[:20]:
        hear(None, REST_MS)
        brain.set_acetylcholine(1.0)
        for word in episode[:2]:
            hear(word, WORD_MS)
        brain.set_acetylcholine(0.0)
        rates = hear(episode[2], WORD_MS)
        # Достраивание: доля клеток своего ответа, стреляющих на подсказку,
        # против той же доли для ответов других эпизодов (специфичность;
        # просто «против случайного» завышали самые возбудимые клетки зоны).
        index = episodes.index(episode)
        for zone, store, out in ((CA3, answers, completion), (CA1, answers1, completion1)):
            fired = (rates[zone.slice] > 0).float()
            patterns = torch.stack(store).float()
            if fired.sum() == 0 or patterns[index].sum() == 0:
                continue
            hit = (patterns @ fired) / patterns.sum(1).clamp(min=1)
            others = torch.cat([hit[:index], hit[index + 1:]])
            out.append(float(hit[index] / (others.mean() + 1e-3)))
        drive = brain.learned_drive(rates, "association", source="ca1")
        score = torch.stack([drive[v].mean() for v in vocab])
        ranks.append(float((score > score[episode[3]]).sum() + 0.5 * ((score == score[episode[3]]).sum() - 1)))
        # После вопроса открыты ворота «говори» (ровный ток на всю кору, как
        # при ответе): что вспыхнет само — с подсказкой CA1.
        brain.feel("association", torch.arange(assoc.size, device=brain.device), cfg.speak_gate_mv)
        after = brain.observe(WORD_MS)
        brain.feel("association", None, 0.0)
        spikes = torch.stack([after[assoc.start + v].sum() for v in vocab])
        spike_ranks.append(int((spikes > spikes[episode[3]]).sum()) if spikes.sum() > 0 else WORDS)

top1 = sum(r == 0 for r in ranks)
print(f"вспоминание по току CA1 (первые 20 эпизодов из {EPISODES}): ответ первым {top1}/20, "
      f"средний ранг {sum(ranks) / 20 + 1:.1f} из {WORDS} (случайно ~{WORDS // 2})")
print(f"достраивание: клеток своего ответа стреляет ×{sum(completion) / max(1, len(completion)):.2f} к чужим ответам в CA3 "
      f"(по {len(completion)} эпизодам); в CA1 ×{sum(completion1) / max(1, len(completion1)):.2f} (по {len(completion1)})")
print(f"по спайкам коры при воротах «говори»: первым {sum(r == 0 for r in spike_ranks)}/20, "
      f"молчала {sum(r == WORDS for r in spike_ranks)}/20")

# Сон: вспышки CA3, ацетилхолина нет.
brain.set_acetylcholine(0.0)
ca3 = brain.areas["ca3"].E
everyone = torch.arange(ca3.size, device=brain.device)
with brain.frozen():
    for mv in [cfg.ripple_mv * f for f in (0.5, 0.75, 1.0, 1.5)]:
        stats, pure, lift = {n: [] for n in ("ca3", "ca1", "association")}, [], []
        for _ in range(5):
            hear(None, REST_MS)
            brain.feel("ca3", everyone, mv)
            # UP-состояние медленного сна (как Language._ripple).
            brain.feel("association", torch.arange(assoc.size, device=brain.device), cfg.ripple_up_state_mv)
            rates = brain.observe(cfg.ripple_ms)
            brain.feel("ca3", None, 0.0)
            brain.feel("association", None, 0.0)
            for n in stats:
                stats[n].append(share(rates, n))
            # CA3: во сколько раз узор вспышки покрывает узор лучшего эпизода
            # больше, чем случайный набор клеток того же размера.
            fired = (rates[CA3.slice] > 0).float()
            if fired.sum() > 0:
                size = traces.sum(1).clamp(min=1)
                shuffled = fired[torch.randperm(fired.numel(), device=brain.device)]
                # Лучший эпизод против лучшего же для перемешанного набора.
                lift.append(float((traces @ fired / size).max() / ((traces @ shuffled / size).max() + 1e-9)))
            # Кора — только свои слова эпизодов (без частых и «?»).
            spikes = torch.stack([rates[assoc.start + v].sum() for v in vocab])
            own = spikes.clone()
            own[: 4] = 0
            if own.sum() > 0:
                pure.append(max(float(own[e].sum() / own.sum()) for e in episodes))
        print(f"вспышка {mv:.0f} мВ: " + ", ".join(f"{n} {sum(v) / len(v):.2%}" for n, v in stats.items())
              + (f" | CA3 лучший эпизод ×{sum(lift) / len(lift):.2f} к перемешанному" if lift else "")
              + (f" | свои слова одного эпизода {sum(pure) / len(pure):.0%} (кора в {len(pure)}/5)" if pure else " | кора молчит"))

print("смещение: " + ", ".join(f"{n} {float(brain.bias[brain.areas[n].E.slice].mean()):.1f}"
                              for n in ("entorhinal", "dentate", "ca3", "ca1")))
print(f"{brain.n:,} нейронов, {time.time() - t0:.0f} с")
