"""
Эталон мозга для перестройки кода: перенос не должен менять поведение ни на бит.

Сценарий (GPU, мозг small — как в опытах, зерно 1; гонять в очереди за опытами): сборка → пусто 200 мс → буква «а» у
ямки 200 мс с учителем и правилом с учителем на пути V1 → буквы → пусто
100 мс с STDP → буква «б» 100 мс без пластичности. Слепок — по каждому
окну частоты всех клеток, затем все веса и состояние (state_dict).

    PYTHONPATH=. .venv/bin/python tests/golden.py write   # записать эталон
    PYTHONPATH=. .venv/bin/python tests/golden.py         # сравнить с эталоном

Короткий вариант: GOLDEN_SCALE=0.1 — окна × 0.1 (20/20/10/10 мс), путь тот
же (сборка, учитель, STDP, все веса и состояние); эталон в своём файле
golden-x0.1.pt. Полный сценарий на пути PyTorch идёт дольше 30 мин;
короткого хватает, чтобы проверить «правка под флагом при выключенном флаге
не меняет вычислений» (ветки, синапсы, пластичность считаются каждый шаг).
Для проверки правки под флагом: эталон пишется на коммите до правки
(git worktree, туда же этот файл), сравнение — на новом.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Детерминированный GPU: свои ядра CUDA складывают входы atomicAdd во float —
# порядок от запуска к запуску разный, последние биты плывут. Эталон — путь
# PyTorch (без ядер и компиляции) с детерминированными алгоритмами:
# перенос кода не трогает brain/csrc, ядра сверяет tools/kernel_check.py.
os.environ["BRAIN_KERNELS"] = "0"
os.environ["BRAIN_COMPILE"] = "0"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

torch.use_deterministic_algorithms(True)

from brain.config import BRAIN_HOME, BrainConfig
from brain.mind import Mind
from brain.session import Session
from brain import anatomy

SCALE = float(os.environ.get("GOLDEN_SCALE", "1"))
# ~2.6 ГБ — вне git, рядом с весами; короткий вариант — свой файл.
GOLDEN = BRAIN_HOME / ("golden.pt" if SCALE == 1 else f"golden-x{SCALE:g}.pt")


def flatten(prefix: str, x, out: dict) -> None:
    """Все тензоры вложенного словаря/списка — плоско по пути."""

    if isinstance(x, torch.Tensor):
        out[prefix] = x.detach().cpu().clone()
    elif isinstance(x, dict):
        for k, v in x.items():
            flatten(f"{prefix}.{k}", v, out)
    elif isinstance(x, (list, tuple)):
        for i, v in enumerate(x):
            flatten(f"{prefix}[{i}]", v, out)
    elif isinstance(x, (int, float, bool)):
        out[prefix] = torch.tensor(x)


def scenario() -> dict[str, torch.Tensor]:
    torch.manual_seed(1)
    cfg = BrainConfig()
    cfg.size = "small"
    cfg.seed = 1
    cfg.autosave = False
    anatomy.use_fit(cfg, "lgn-a")
    cfg.us_paths = ("visual -> letters",)
    t0 = time.time()
    mind = Mind(cfg)
    b = mind.brain
    print(f"сборка {time.time() - t0:.0f} с; окна × {SCALE:g}", flush=True)
    s = Session(mind, window_ms=10.0)
    out: dict[str, torch.Tensor] = {}

    s.line, s.gaze = "", 0.0
    b.plasticity = False
    out["r0"] = b.observe(200.0 * SCALE, s._blank(200.0 * SCALE)).cpu()

    path = b.us[0]
    P = path.post
    b.teach_pop.zero_()
    b.teach_target.zero_()
    b.teach_pop[P.start:P.start + 20] = True
    b.teach_target[P.start:P.start + 10] = True
    s.line = "а"
    s.gaze = s.screen.width("а") / 2
    b.teach_on = b.us_on = True
    out["r1"] = b.observe(200.0 * SCALE, s._view(200.0 * SCALE)).cpu()
    b.teach_on = b.us_on = False

    s.line = ""
    b.plasticity = True
    out["r2"] = b.observe(100.0 * SCALE, s._blank(100.0 * SCALE)).cpu()
    b.plasticity = False

    s.line = "б"
    s.gaze = s.screen.width("б") / 2
    out["r3"] = b.observe(100.0 * SCALE, s._view(100.0 * SCALE)).cpu()

    print(f"окна прожиты, всего {time.time() - t0:.0f} с", flush=True)
    for p in b.projections:
        out[f"w.{p.name}"] = p.weight.detach().cpu().clone()
    flatten("state", b.state_dict(), out)
    return out


def main() -> None:
    got = scenario()
    if sys.argv[1:] == ["write"]:
        torch.save(got, GOLDEN)
        spikes = float(sum(got[f"r{i}"].sum() for i in range(4)))
        print(f"эталон записан: {GOLDEN} ({len(got)} тензоров, сумма частот {spikes:.1f})")
        return
    ref = torch.load(GOLDEN)
    bad = [k for k in ref if k not in got or ref[k].shape != got[k].shape or not torch.equal(ref[k], got[k])]
    extra = [k for k in got if k not in ref]
    for k in bad[:20]:
        if k in got and ref[k].shape == got[k].shape and ref[k].is_floating_point():
            d = (ref[k].float() - got[k].float()).abs()
            print(f"  РАЗНИЦА {k}: отличий {int((d > 0).sum())} из {d.numel()}, макс {float(d.max()):.3g}")
        else:
            print(f"  РАЗНИЦА {k}" + ("" if k in got else " (нет)"))
    if extra:
        print(f"  новые ключи: {extra[:10]}")
    print("совпадает бит в бит" if not bad else f"НЕ СОВПАДАЕТ: {len(bad)} из {len(ref)}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
