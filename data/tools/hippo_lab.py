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

# Словарь: места слов не пересекаются.
order = list(range(assoc.size))
rng.shuffle(order)
WORDS = 4 + 2 * EPISODES
vocab = [torch.tensor(order[i * CELLS:(i + 1) * CELLS], device=brain.device) for i in range(WORDS)]
common = [0, 1, 2]
QUESTION = 3
episodes = [[common[i % 3], 4 + i, QUESTION, 4 + EPISODES + i] for i in range(EPISODES)]


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
for episode in episodes:
    for word in episode:
        rates = hear(word, WORD_MS)
        brain.homeostasis(rates)
        brain.encode(cfg.hippocampus_encode)
        for n in log:
            log[n].append(share(rates, n))
    hear(None, REST_MS)
print("при записи активны: " + ", ".join(f"{n} {sum(v) / len(v):.1%}" for n, v in log.items()))

# Вспоминание: первые три слова слушает (ацетилхолин высокий), на
# последнем открывает ворота «говори» — ацетилхолин падает, CA3 и выход свободны.
ranks, spike_ranks = [], []
with brain.frozen():
    for episode in episodes[:20]:
        hear(None, REST_MS)
        brain.set_acetylcholine(1.0)
        for word in episode[:2]:
            hear(word, WORD_MS)
        brain.set_acetylcholine(0.0)
        rates = hear(episode[2], WORD_MS)
        drive = brain.learned_drive(rates, "association", source="ca1")
        score = torch.stack([drive[v].mean() for v in vocab])
        ranks.append(int((score > score[episode[3]]).sum()))
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
print(f"по спайкам коры при воротах «говори»: первым {sum(r == 0 for r in spike_ranks)}/20, "
      f"молчала {sum(r == WORDS for r in spike_ranks)}/20")

# Сон: вспышки CA3, ацетилхолина нет.
brain.set_acetylcholine(0.0)
ca3 = brain.areas["ca3"].E
everyone = torch.arange(ca3.size, device=brain.device)
with brain.frozen():
    for mv in [cfg.ripple_mv * f for f in (0.5, 0.75, 1.0, 1.5)]:
        stats, pure = {n: [] for n in ("ca3", "ca1", "association")}, []
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
            spikes = torch.stack([rates[assoc.start + v].sum() for v in vocab])
            if spikes.sum() > 0:
                # Доля спайков коры на словах одного (лучшего) эпизода.
                pure.append(max(float(spikes[e].sum() / spikes.sum()) for e in episodes))
        print(f"вспышка {mv:.0f} мВ: " + ", ".join(f"{n} {sum(v) / len(v):.2%}" for n, v in stats.items())
              + (f" | слова одного эпизода {sum(pure) / len(pure):.0%} (кора стреляла в {len(pure)}/5)" if pure else " | кора молчит"))

print("смещение: " + ", ".join(f"{n} {float(brain.bias[brain.areas[n].E.slice].mean()):.1f}"
                              for n in ("entorhinal", "dentate", "ca3", "ca1")))
print(f"{brain.n:,} нейронов, {time.time() - t0:.0f} с")
