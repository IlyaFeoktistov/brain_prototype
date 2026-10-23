"""
Чем возбуждается зона букв L4 E при чтении и доходит ли до неё различие
фраз (DESIGN «Вспышка и различение по популяции»: V1 различает A/B, буквы —
нет).

Предъявления как в l4_decode (A и B вперемешку, n повторов, чтение). У
каждой клетки букв L4 E за фиксацию: вход каждого пути — Σ вес × спайки
источника (мВ за фиксацию, без формы синапса и депрессии: мера поставки, не
ток), возбуждающий и тормозный ток (i_syn, i_inh — среднее по окнам), спайки.
Прочее возбуждение — фон и среднее поле недостающих соседей (в i_syn, не в
путях). Различие фраз — точность ближайшего центра с выбросом одного повтора
по вектору входа пути V1, по токам и по спайкам: если вход V1 различает, а
спайки нет — сигнал доходит, но тонет в прочем возбуждении.

Один причинный контроль: без=v1 — веса visual → letters (E и PV) в ноль.

    PYTHONPATH=. python tools/letters_drive.py [n=12] [fix=225] [step=5] [seed=1] [без=v1]
"""

from __future__ import annotations

import sys

import torch

from brain.session import Session
from tools.reading_bench import build

WIN = 10.0


def loo(x: torch.Tensor, y: torch.Tensor) -> float:
    """Ближайший центр класса по корреляции, выброс одного повтора."""

    def corr(a: torch.Tensor) -> torch.Tensor:
        a = a - a.mean(1, keepdim=True)
        a = a / a.norm(dim=1, keepdim=True).clamp_min(1e-9)
        return a @ a.T

    hits = 0
    for i in range(len(y)):
        keep = torch.arange(len(y)) != i
        cents = torch.stack([x[keep & (y == c)].mean(0) for c in (0, 1)])
        hits += int(int(corr(torch.cat([x[i:i + 1], cents]))[0, 1:].argmax()) == int(y[i]))
    return hits / len(y)


def main(opts: dict) -> None:
    opts = {"fit": "lgn-a", "homeo": f"lgn-a-s{opts.get('seed', 1)}", "relay_v_t_mv": "-59",
            "relay_rest_mv": "-48", "trn_scale": "0.5", **opts}
    mind, b = build(opts)
    cfg = b.cfg
    s = Session(mind, window_ms=WIN)
    n_rep, fix, step = int(opts.get("n", 12)), float(opts.get("fix", 225)), float(opts.get("step", 5))
    texts = (opts.get("a", "мама мыла раму"), opts.get("b", "папа пила чай"))
    gen = torch.Generator().manual_seed(int(opts.get("seed", 1)))
    E = b.areas["letters"].layers["L4"]["E"]
    paths = [p for p in b.projections if p.post is E and p.source == "net"]
    if opts.get("без") == "v1":
        for p in b.projections:
            if p.name.startswith("visual -> letters"):
                p.weight.zero_()

    s.line, s.gaze = "", 0.0
    b.run(2500.0)

    def gazes(line: str) -> list[float]:
        width = s.screen.width(line) / max(1, len(line))
        n = max(1, int((len(line) + step - 1) // step))
        return [i * step * width for i in range(n)]

    order = [0, 1] * n_rep
    order = [order[i] for i in torch.randperm(len(order), generator=gen)]
    keys = [p.name for p in paths] + ["i_syn", "i_inh", "спайки"]
    feats = {k: [] for k in keys}
    for lab in order:
        s.line = ""
        b.run(500.0, s._blank(500.0))
        line = texts[lab] + " "
        rows = {k: [] for k in keys}
        for i, g in enumerate(gazes(line)[:3]):
            if i:
                s.line, s.gaze = line, g
                b.run(cfg.saccade_ms, s._blank(cfg.saccade_ms))
            s.line, s.gaze = line, g
            acc = {k: 0 for k in keys}
            n_win = int(fix // WIN)
            for _ in range(n_win):
                r = b.observe(WIN, s._view(WIN)) * (WIN / 1000.0)
                for p in paths:
                    acc[p.name] = acc[p.name] + (p.weight.float() * r[p.pre.long()]).sum(1).cpu()
                acc["i_syn"] = acc["i_syn"] + b.i_syn[E.slice].float().cpu() / n_win
                acc["i_inh"] = acc["i_inh"] + b.i_inh[E.slice].float().cpu() / n_win
                acc["спайки"] = acc["спайки"] + r[E.slice].cpu()
            for k in keys:
                rows[k].append(acc[k])
        for k in keys:
            feats[k].append(torch.cat(rows[k]))
    y = torch.tensor(order)
    chance = lambda x: float(torch.tensor([loo(x, y[torch.randperm(len(y), generator=gen)])
                                           for _ in range(20)]).mean())

    print(f"\n=== Зона букв L4 E ({E.size} клеток), «{texts[0]}» против «{texts[1]}», "
          f"по {n_rep} повторов, без={opts.get('без', '—')}")
    print(f"  {'вход':28s} {'среднее/фикс':>12s} {'разброс клеток':>14s} {'A−B / шум':>10s}"
          f" {'точность':>9s} {'случайно':>9s}")
    for k in keys:
        x = torch.stack(feats[k])
        ma, mb = x[y == 0].mean(0), x[y == 1].mean(0)
        noise = torch.cat([x[y == 0] - ma, x[y == 1] - mb]).std(0).mean()
        snr = float((ma - mb).abs().mean() / noise.clamp_min(1e-9))
        print(f"  {k:28s} {float(x.mean()):12.3f} {float(x.mean(0).std()):14.3f} {snr:10.2f}"
              f" {loo(x, y):9.2f} {chance(x):9.2f}", flush=True)


if __name__ == "__main__":
    main(dict(kv.split("=", 1) for kv in sys.argv[1:]))
