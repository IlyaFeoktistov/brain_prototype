"""
Замер: новый мозг -> 2 прохода train-data -> экзамен фактов.

    python -m brain.measure размер имя [key=value ...] [then=папка:слов] [load=путь]

Печатает память, пик видеопамяти, скорость чтения без сна и факты;
обученный мозг сохраняет в data/measure/имя.pt (там же — для экзамена
с другими настройками ответа).

then=папка:слов — потом прочитать первые слова корпуса (один проход,
обычно больше слов, чем мест) и пересдать экзамен: сколько выученного
переживает переполнение ёмкости (без дремоты при чтении). Мозг до этой
фазы сохраняется в data/measure/имя-before.pt; load=путь — начать с него
(без создания и учёбы), key=value применяются к загруженному.
"""
import sys, time
from pathlib import Path
import torch
from brain.commands import Session
from brain.config import BrainConfig
from brain.mind import Mind
from brain import evaluate

size = sys.argv[1] if len(sys.argv) > 1 else "medium"
cfg = BrainConfig()
cfg.size = size
cfg.autosave = False
cfg.save_path = "/nonexistent/brain.pt"
# Настройки: medium.py размер имя key=value ...
name = sys.argv[2] if len(sys.argv) > 2 else size
then = None
load = None
overrides = []
for kv in sys.argv[3:]:
    k, v = kv.split("=")
    if k == "then":
        folder, _, words = v.partition(":")
        then = (Path(folder), int(words or 10000))
    elif k == "load":
        load = v
    else:
        overrides.append((k, v))


def apply(cfg):
    for k, v in overrides:
        old = getattr(cfg, k)
        setattr(cfg, k, (v == "1") if isinstance(old, bool) else type(old)(v))
        print(f"{k} = {getattr(cfg, k)}")


t0 = time.time()
if load:
    mind = Mind.load(load)
    mind.cfg.autosave = False
    apply(mind.cfg)
    mind.brain.recompile()
else:
    apply(cfg)
    mind = Mind(cfg)
    mind.calibrate()
torch.cuda.synchronize()
c = mind.language.concepts
print(f"{size}: {mind.brain.n:,} нейронов, {mind.brain.synapse_count():,} синапсов, "
      f"{mind.brain.memory_bytes()/2**20:.0f} МБ, мест ~{c.size // c.cells:,}, "
      f"создан за {time.time()-t0:.0f} с, пик {torch.cuda.max_memory_allocated()/2**30:.2f} ГБ, "
      f"резерв {torch.cuda.memory_reserved()/2**30:.2f} ГБ", flush=True)

session = Session(mind, lambda text, kind: kind != "progress" and print(text, flush=True))
slept = [0.0]
_sleep = mind.language.sleep_until_rested
def timed_sleep(*a, **k):
    t = time.time(); r = _sleep(*a, **k); slept[0] += time.time() - t; return r
mind.language.sleep_until_rested = timed_sleep
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
tokens = 0
for _ in range(0 if load else 2):
    text = Path("train-data.md").read_text(encoding="utf-8")
    tokens += session.read_text("train-data", text).tokens
dt = time.time() - t0
if not load:
    print(f"чтение: {tokens} токенов за {dt:.0f} с (сон {slept[0]:.0f} с), "
          f"без сна {tokens/(dt-slept[0]):.1f} ток/с, "
          f"синапсов {mind.brain.synapse_count():,}, пик {torch.cuda.max_memory_allocated()/2**30:.2f} ГБ, "
          f"резерв {torch.cuda.memory_reserved()/2**30:.2f} ГБ", flush=True)

tok = mind.language.tokenize
qs = evaluate.facts(Path("train-data.md"), tok)


def exam():
    for kind, score in evaluate.run(mind, qs, 3).items():
        print(score.line(kind)); print("\n".join(score.examples), flush=True)
    health = mind.language.health
    bad = [snap for snap in health.log]
    print(f"здоровье: нарушений {len(bad)} в последних фразах из {health.sentences}"
          + (f" — {bad[-1].problems[0]}" if bad else ""), flush=True)
    usage = c.usage
    print(f"места: слов {len(c):,}, клеток занято {(usage > 0).float().mean():.0%}, "
          f"делят место {(usage > 1).float().mean():.0%} клеток", flush=True)


exam()

if then is not None:
    if not load:
        mind.save(f"data/measure/{name}-before.pt")
    folder, limit = then
    text = "\n".join(p.read_text(encoding="utf-8") for p in sorted(folder.glob("*.txt")))
    words = text.split()
    text = " ".join(words[:limit])
    # Без дремоты: иначе фаза идёт 20 минут, из них 17 — сон.
    mind.cfg.nap_while_reading = False
    t0, slept[0] = time.time(), 0.0
    report = session.read_text(folder.name, text)
    print(f"потом {folder.name}: {report.tokens} токенов за {time.time() - t0:.0f} с "
          f"(сон {slept[0]:.0f} с)", flush=True)
    exam()
mind.save(f"data/measure/{name}.pt")
