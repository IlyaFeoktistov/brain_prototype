"""
Один фон на все размеры мозга: частоты покоя ядер и популяций колонок и
ответ LGN на слово — на мозге с колонками разного размера при том же фоне
(anatomy.NUCLEUS_BIAS_MV, COLUMN_EXTERNAL_POP).

    PYTHONPATH=. python tools/size_check.py column=20000 seed=1 out=a.json   # замер одного мозга
    PYTHONPATH=. python tools/size_check.py compare=a.json,b.json,c.json     # размер A, A другим зерном, размер B

Критерии (заданы до замера):
  (а) среднее отношение частот «размер B / размер A» по всем популяциям —
      в пределах 10% (|среднее ln| ≤ 0.1);
  (б) разброс между размерами не больше разброса между зёрнами одного
      размера × 1.5 (среднеквадратичный ln отношения);
  (в) LGN под словом выше фона ≥ 3 раз на каждом размере.
Частоты — (r + 0.5): ядра, молчащие в покое, не дают бесконечных отношений.
"""

from __future__ import annotations

import json
import math
import sys

import torch

from brain.config import BrainConfig, log_path
from brain.mind import Mind
from brain.session import Session


def measure(column: int, seed: int, warm: float, ms: float, text: str) -> dict:
    cfg = BrainConfig()
    cfg.size = "small"
    cfg.min_column_neurons = column
    cfg.seed = seed
    cfg.autosave = False
    mind = Mind(cfg)
    b = mind.brain
    b.plasticity = False
    rates: dict[str, float] = {}

    b.run(warm)
    r = b.observe(ms)

    for name, area in b.areas.items():
        if area.layers is None:
            rates[name] = float(r[area.E.slice].mean())
        else:
            for layer, group in area.layers.items():
                for kind, pop in group.items():
                    rates[f"{name}.{layer}.{kind}"] = float(r[pop.slice].mean())

    # Ответ LGN на слово в ямке (как tools/lgn_trace.py).
    s = Session(mind)
    s.line, s.gaze = text + " ", 0.0
    eye = [x.name for x in b.sensors].index("eye")
    lgn = b.areas["lgn"].E
    seen = s.eye.rates(s._picture(s.screen, cfg.fovea_col))
    driven = (seen[torch.arange(lgn.size) % b.sensors[eye].size] > 0.3 * cfg.retina_rate_hz).to(b.device)

    for _ in range(int(warm / s.window_ms)):
        b.observe(s.window_ms, s._view(s.window_ms))

    hz = torch.zeros(b.n, device=b.device)
    steps = int(ms / s.window_ms)

    for _ in range(steps):
        hz += b.observe(s.window_ms, s._view(s.window_ms)) / steps

    word = hz[lgn.slice]
    return {"column": column, "seed": seed, "neurons": b.n, "rates": rates,
            "lgn_word": float(word[driven].mean()), "lgn_rest": float(word[~driven].mean())}


def compare(a: dict, a2: dict, b: dict) -> None:
    keys = sorted(set(a["rates"]) & set(a2["rates"]) & set(b["rates"]))
    ln = lambda x, y, k: math.log((x["rates"][k] + 0.5) / (y["rates"][k] + 0.5))
    sizes = [ln(b, a, k) for k in keys]
    seeds = [ln(a2, a, k) for k in keys]
    rms = lambda v: math.sqrt(sum(x * x for x in v) / len(v))
    mean = sum(sizes) / len(sizes)

    print(f"мозги: A {a['neurons']} (колонка {a['column']}), A' — зерно {a2['seed']}, B {b['neurons']} (колонка {b['column']}); популяций {len(keys)}")
    print(f"(а) среднее ln(B/A) = {mean:+.3f} → {'да' if abs(mean) <= 0.1 else 'НЕТ'} (≤ 0.1)")
    print(f"(б) разброс: размеры {rms(sizes):.3f}, зёрна {rms(seeds):.3f} → {'да' if rms(sizes) <= 1.5 * rms(seeds) else 'НЕТ'} (≤ ×1.5)")

    for x in (a, a2, b):
        ratio = (x["lgn_word"] + 0.5) / (x["lgn_rest"] + 0.5)
        print(f"(в) колонка {x['column']}, зерно {x['seed']}: LGN слово {x['lgn_word']:.1f} / фон {x['lgn_rest']:.1f} Гц → ×{ratio:.1f} {'да' if ratio >= 3 else 'НЕТ'}")

    worst = sorted(keys, key=lambda k: -abs(ln(b, a, k)))[:12]
    print("сильнее всех расходятся (A / A' / B, Гц):")

    for k in worst:
        print(f"   {k:28s} {a['rates'][k]:7.1f} {a2['rates'][k]:7.1f} {b['rates'][k]:7.1f}")


if __name__ == "__main__":
    opts = dict(kv.split("=", 1) for kv in sys.argv[1:])

    if "compare" in opts:
        files = [json.load(open(log_path(f) if "/" not in f else f, encoding="utf-8")) for f in opts["compare"].split(",")]
        compare(*files)
    else:
        out = measure(int(opts.get("column", 20000)), int(opts.get("seed", 1)), float(opts.get("warm", 2500)),
                      float(opts.get("ms", 2000)), opts.get("text", "мама мыла раму"))
        path = log_path(opts.get("out", f"size-{out['column']}-{out['seed']}.json"))
        path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
        print(f"{out['neurons']} нейронов, LGN слово {out['lgn_word']:.1f} / фон {out['lgn_rest']:.1f} Гц → {path}")
