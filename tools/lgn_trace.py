"""
Почему клетки LGN под словом не стреляют чаще фона: строка на экране, взгляд
неподвижен, частоты клеток реле, чей рецептор видит штрих («под словом»),
против прочих — при выключенных по очереди входах реле (веса пути → 0).

    PYTHONPATH=. python tools/lgn_trace.py brain=проверка-веба.pt ms=1000

Человек и примат: ведомая клетка LGN отвечает на стимул в поле в разы чаще
спонтанной (~10 Гц покоя против десятков Гц на контраст; Usrey, Reppas &
Reid 1999; Kaplan & Shapley 1984). Здесь видно, какой вход это ломает.
"""

from __future__ import annotations

import sys

import torch

from brain.mind import Mind
from brain.session import Session


def main(brain: str = "проверка-веба.pt", ms: float = 1000.0, text: str = "мама мыла раму") -> None:
    mind = Mind.load(brain)
    mind.cfg.autosave = False
    b = mind.brain
    s = Session(mind)
    s.line, s.gaze = text + " ", 0.0
    eye = next(i for i, x in enumerate(b.sensors) if x.name == "eye")
    offset, size = b.sensor_offsets[eye], b.sensors[eye].size
    lgn = b.areas["lgn"]
    rates = s.eye.rates(s._picture(s.screen, s.cfg.fovea_col))
    receptor = torch.arange(lgn.E.size) % size
    driven = rates[receptor] > 0.3 * s.cfg.retina_rate_hz

    # Входы реле по путям: имя пути → проекции.
    groups: dict[str, list] = {}
    for p in b.projections:
        if p.post is lgn.E:
            groups.setdefault(p.name.split(" (")[0] if "триада" not in p.name else "триада", []).append(p)

    print(f"LGN E {lgn.E.size}, под словом {int(driven.sum())}, рецепторов {size}; входы реле:")
    for name, ps in groups.items():
        print(f"  {name}: {sum(p.weight.numel() for p in ps)} синапсов, средний вес {float(torch.cat([p.weight.flatten() for p in ps]).mean()):+.3f}")

    def run(label: str) -> None:
        with b.frozen():
            counts = torch.zeros(b.n, device=b.device)
            steps = int(round(ms / s.window_ms))
            seen_rec = torch.zeros(size, device=b.device)

            for _ in range(steps):
                view = s._view(s.window_ms)
                seen_rec += view[:, offset:offset + size].sum(0)
                r = b.observe(s.window_ms, view)
                counts += r * s.window_ms / 1000.0

            hz = counts / (ms / 1000.0)
            rec = seen_rec / (ms / 1000.0)
            e = hz[lgn.E.slice].cpu()
            i = hz[lgn.I.slice].mean().item()
            print(f"{label:28s} рецепторы {rec[receptor[driven]].mean():6.1f} / {rec[receptor[~driven]].mean():5.1f} Гц | "
                  f"LGN E под словом {e[driven].mean():6.1f}, фон {e[~driven].mean():5.1f} Гц | LGN I {i:5.1f} | "
                  f"TRN {hz[b.areas['trn'].E.slice].mean().item():5.1f}")

    run("всё включено")

    for name, ps in groups.items():
        saved = [p.weight.clone() for p in ps]
        for p in ps:
            p.weight.zero_()
        run(f"без «{name}»")
        for p, w in zip(ps, saved):
            p.weight.copy_(w)


if __name__ == "__main__":
    args = dict(a.split("=", 1) for a in sys.argv[1:])
    main(args.get("brain", "проверка-веба.pt"), float(args.get("ms", 1000)), args.get("text", "мама мыла раму"))
