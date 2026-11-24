"""
Картинки по трассам tools/e1_in_brain.py (trace=…): весь путь слова по времени —
кора (популяции слов и пороги «сказано»), вход E1 из коры, стриатум E1, ячейки
слов E1 по позициям, реле чтения и ответ E1, средние частоты зон мозга.
Рисует общий модуль tools/plots.py.

    PYTHONPATH=tools .venv/bin/python tools/e1_in_brain_plot.py /tmp/e1b/tr.БД.pt [out.png]
"""

from __future__ import annotations

import sys

import numpy as np
import torch

import plots


def main(path: str, out: str | None) -> str:
    d = torch.load(path, weights_only=False)
    words, vocab = d["words"], d["vocab"]
    n = d["src"].shape[0]
    t_ms = (np.arange(n) + 1) * d["chunk_ms"]
    K = 5                                                        # сглаживание 5 × 5 мс = 25 мс
    shown = words + ["повтори"]
    pops = {w: d["pops"][:, vocab.index(w)] for w in shown}
    thr = {w: d["thr"][vocab.index(w)] for w in shown}
    panels = [
        {"label": "кора: популяции\nслов L2/3, Гц", "series": pops, "thr": thr, "smooth": K, "height": 1.3},
        {"label": "вход E1 из коры\n(20 клеток слова), Гц", "series": {w: d["src"][:, j] for j, w in enumerate(words)},
         "hlines": [(40, "40 Гц — слово в испытании E1"), (2, "2 Гц — фон E1")], "smooth": K},
        {"label": "стриатум E1\n(запись/чтение), Гц", "series": {w: d["msn"][:, j] for j, w in enumerate(words)},
         "smooth": K},
    ]
    held = d["held"].numpy()                                     # [кусок, позиция, слово]
    for pos in range(len(d["text"])):
        h = plots.smooth(held[:, pos, :], 10)
        share = h / np.maximum(h.sum(1, keepdims=True), 1e-9)
        panels.append({"label": f"ячейка {pos + 1} (ждём «{d['text'][pos]}»):\nдоля слова",
                       "series": {w: share[:, j] for j, w in enumerate(words)}, "height": 0.7})
    panels += [
        {"label": "реле чтения E1\nпо словам, Гц", "series": {w: d["rel"][:, j] for j, w in enumerate(words)},
         "smooth": K},
        {"label": "ответ E1\nпо словам, Гц", "series": {w: d["ans"][:, j] for j, w in enumerate(words)}, "smooth": K},
        {"label": "зоны (E)", "series": {"areas": d["areas"]}, "heat": True,
         "rows": d.get("area_names"), "smooth": K},
    ]
    reads = [(p0, p0 + 300, "") for p0, kind, _ in d["periods"] if kind == "r"]
    title = (f"«{d['text']}»: путь слова кора → E1 → кора (серое — слово звучит, голубое — окна чтения; "
             f"пунктир — порог «сказано»)")
    return plots.timeline(out or path.replace(".pt", ".png"), t_ms, panels, windows=d["heard"],
                          windows_alt=reads, title=title)


if __name__ == "__main__":
    print(main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None))
