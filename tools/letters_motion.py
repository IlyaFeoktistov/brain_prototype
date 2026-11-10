"""
Движение глаза при фиксации и буквы (замер, мозг не меняется; мозг как в
letters_dc, гомеостаз lgn-c). Вопрос (DESIGN, l4_inputs): ответ V1 L4 на
неподвижную букву — всплеск 0–50 мс, дальше вход гасит депрессия таламус →
L4; у приматов при фиксации картинку сдвигают дрейф и микросаккады, и ответ
V1 держится (Snodderly, Kagan & Gur 2001, https://doi.org/10.1017/S0952523801182118).
Возвращает ли движение букву в V1 L4 / L2/3 и дальше — по различению, не
по частоте.

Движение (как tools/fixation_motion.py): дрейф макаки — пик ~33–45 угл.
мин/с (Malevich, Buonocore & Hafed 2020, https://doi.org/10.7554/eLife.57595);
1 px ≈ 2.5 угл. мин [оценка] → дрейф 15 px/с, случайное блуждание по
горизонтали (направление меняется с вероятностью turn за 10 мс);
микросаккада ~10 угл. мин → 4 px, на ms_at мс, направление случайно. Сдвиг
картинки — по горизонтали и на целый пиксель (Session._picture). Траектория
— своя у каждого показа.

Условия перемешаны в одном мозге: неподвижно / дрейф / дрейф + микросаккада;
по каждому — label + test показов каждой буквы. Ступени: LGN, V1 L4 E, V1
L2/3 E, буквы L4 E, буквы L2/3 E. Частоты (пауза — последние 100 мс, 0–50,
50–show мс) и различение букв по корреляции (как letters_dc) по окнам 0–50,
50–show и за весь показ.

    PYTHONPATH=. .venv/bin/python tools/letters_motion.py [letters=абвгдежзик] [label=5] [test=5] [drift=15] [micro=4] [ms_at=110] [turn=0.1] [seed=1]
"""

from __future__ import annotations

import sys
import time

import torch

from brain.session import Session
from tools.reading_bench import build

WIN = 10.0


def main(opts: dict) -> None:
    seed = int(opts.get("seed", 1))
    opts = {"fit": "lgn-a", "homeo": f"lgn-c-s{seed}", "relay_v_t_mv": "-59", "trn_scale": "0.1", **opts}
    mind, b = build(opts)
    s = Session(mind, window_ms=WIN)
    letters = opts.get("letters", "абвгдежзик")
    n_label, n_test = int(opts.get("label", 5)), int(opts.get("test", 5))
    rest, show = float(opts.get("rest", 300)), float(opts.get("show", 350))
    drift, micro = float(opts.get("drift", 15)), float(opts.get("micro", 4))
    ms_at, turn = float(opts.get("ms_at", 110)), float(opts.get("turn", 0.1))
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    vis, let = b.areas["visual"].layers, b.areas["letters"].layers
    stages = {"LGN": b.areas["lgn"].E, "V1 L4": vis["L4"]["E"], "V1 L2/3": vis["L23"]["E"],
              "буквы L4": let["L4"]["E"], "буквы L2/3": let["L23"]["E"]}
    n_win, early = int(show // WIN), int(50 // WIN)

    def trajectory(kind: str) -> list[float]:
        if kind == "неподвижно":
            return [0.0] * n_win
        x, sign, out = 0.0, (1.0 if float(torch.rand(1, generator=gen)) < 0.5 else -1.0), []
        jump = micro * (1.0 if float(torch.rand(1, generator=gen)) < 0.5 else -1.0)
        for t in range(n_win):
            if float(torch.rand(1, generator=gen)) < turn:
                sign = -sign
            x += sign * drift * WIN / 1000.0
            out.append(x + (jump if kind == "дрейф + микросаккада" and t * WIN >= ms_at else 0.0))
        return out

    conditions = ("неподвижно", "дрейф", "дрейф + микросаккада")
    plan = [(c, k, part) for part in ("метки", "проверка") for c in conditions for k in range(len(letters))
            for _ in range(n_label if part == "метки" else n_test)]
    order = torch.randperm(len(plan), generator=gen).tolist()

    s.line, s.gaze = "", 0.0
    b.run(2500.0, s._blank(2500.0))
    t0 = time.time()
    rec = {c: {"y": [], "part": [], "bg": [], "e": [], "l": []} for c in conditions}
    for j, idx in enumerate(order):
        c, k, part = plan[idx]
        ch = letters[k]
        s.line = ""
        b.run(rest - 100.0, s._blank(rest - 100.0))
        bg = b.observe(100.0, s._blank(100.0))
        base = s.screen.width(ch) / 2 + float(torch.randint(-1, 2, (1,), generator=gen))
        path = trajectory(c)
        e = l = 0
        for t in range(n_win):
            s.line, s.gaze = ch, base + path[t]
            r = b.observe(WIN, s._view(WIN))
            if t < early:
                e = e + r / early
            else:
                l = l + r / (n_win - early)
        s.line = ""
        R = rec[c]
        R["y"].append(k)
        R["part"].append(part == "метки")
        R["bg"].append({n: bg[p.slice].cpu() for n, p in stages.items()})
        R["e"].append({n: e[p.slice].cpu() for n, p in stages.items()})
        R["l"].append({n: l[p.slice].cpu() for n, p in stages.items()})
        if (j + 1) % 50 == 0:
            print(f"  показов {j + 1}/{len(order)}, {time.time() - t0:.0f} с", flush=True)

    z = lambda x: (x - x.mean(-1, keepdim=True)) / x.std(-1, keepdim=True).clamp_min(1e-9)

    def decode(X: torch.Tensor, y: torch.Tensor, lab: torch.Tensor) -> float:
        mean = torch.stack([X[lab][y[lab] == k].mean(0) for k in range(len(letters))])
        return float(((z(X[~lab]) @ z(mean).T).argmax(1) == y[~lab]).float().mean())

    print(f"\n=== движение глаза и буквы: {len(letters)} букв, метки {n_label} / проверка {n_test} на условие; "
          f"дрейф {drift:g} px/с, микросаккада {micro:g} px на {ms_at:g} мс; случайно {1 / len(letters):.2f}")
    print(f"  {'условие':22s} {'ступень':11s} {'пауза':>6s} {'0–50':>6s} {'50–' + str(int(show)):>7s}   "
          f"различение 0–50 / 50–{int(show)} / весь показ")
    for c in conditions:
        R = rec[c]
        y, lab = torch.tensor(R["y"]), torch.tensor(R["part"])
        for n in stages:
            B = torch.stack([x[n] for x in R["bg"]])
            E = torch.stack([x[n] for x in R["e"]])
            L = torch.stack([x[n] for x in R["l"]])
            W = (E * early + L * (n_win - early)) / n_win
            print(f"  {c:22s} {n:11s} {float(B.mean()):6.2f} {float(E.mean()):6.2f} {float(L.mean()):7.2f}   "
                  f"{decode(E, y, lab):.2f} / {decode(L, y, lab):.2f} / {decode(W, y, lab):.2f}", flush=True)
    print(f"всего {time.time() - t0:.0f} с")


if __name__ == "__main__":
    main(dict(kv.split("=", 1) for kv in sys.argv[1:]))
