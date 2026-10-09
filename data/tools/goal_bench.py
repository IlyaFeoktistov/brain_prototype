import sys, time; sys.path.insert(0, ".")
from brain.config import BrainConfig
from brain.mind import Mind
kw = dict(a.split("=") for a in sys.argv[1:])
cfg = BrainConfig(); cfg.size = "small"; cfg.calibration_tokens = 150; cfg.autosave = False
spec = next(s for s in cfg.areas if s.name == "goal")
for k, v in kw.items():
    setattr(spec, k, float(v) if k != 'columns' else int(v))
m = Mind(cfg); m.calibrate(); L = m.language; B = m.brain
G = L.goal.E
def alive(places, ms=500):
    r = B.observe(ms)[G.slice]
    return [round(float(r[p].mean()), 0) for p in places], round(float((r > 0).float().mean()), 3)
print(kw, "тишина 3 с, доля активных:", [alive([], 1000)[1] for _ in range(3)])
def load(words):
    places = []
    for w in words:
        p = L._goal_place(w, True); places.append(p)
        B.feel("goal", p, cfg.goal_load_mv); B.observe(cfg.stimulus_ms); B.feel("goal", None, 0.0); B.observe(cfg.gap_ms)
    return places
q = load(["столица", "франции", "?"])
for t in [1, 2, 4, 8, 16]:
    hz, frac = alive(q, 500)
    print(f"вопрос из 3 через ~{t*0.5:.1f} с: Гц {hz}")
    B.observe(max(0, (t * 2 - t) * 500 - 500)) if t > 1 else None
q = load(["столица", "франции", "?"])
q2 = load([f"слово{i}" for i in range(8)])
for t in range(4):
    hz, frac = alive(q + q2)
    print(f"вопрос + 8 слов, {t+1}×0.5 с: живых из вопроса {sum(h > 5 for h in hz[:3])}/3, новых {sum(h > 5 for h in hz[3:])}/8, Гц {hz}")
stream = load([f"текст{i}" for i in range(40)])
hz, frac = alive(stream, 300)
print(f"после потока 40 слов: живых {sum(h > 5 for h in hz)}/40, последние 10: {hz[-10:]}, первые 5: {hz[:5]}")
B.observe(3000)
hz, frac = alive(stream, 500)
print(f"и через 3 с тишины: живых {sum(h > 5 for h in hz)}/40, доля клеток {frac}")
