"""
Где буква теряется между V1 L4 и V1 L2/3 мозга (замер, без вмешательства;
мозг как в letters_dc: LGN A, гомеостаз lgn-b, без пластичности). На каждом
показе буквы для каждой клетки V1 L2/3 E копятся:
  - вход каждого пути в клетку (Σ вес × частота источника за окно, мВ·Гц
    — мера поставки, без кратковременной пластичности коры);
  - ток возбуждения и торможения сомы (с движущей силой) на конце окна;
  - мембрана на конце окна;
  - спайки.
По каждой величине — различение букв тем же классификатором по корреляции,
что в letters_dc (центр буквы — средний ответ на метках, ответ — центр с
наибольшей корреляцией); для сравнения — спайки V1 L4 E. Если буква есть
во входе от L4 и пропадает в токах — её гасит сумма с другими входами;
есть в мембране, но не в спайках — порог; нет уже во входе — путь L4 → L2/3.
Плюс средние величины на фоне (пауза) и на показе.

    PYTHONPATH=. .venv/bin/python tools/l23_trace.py [letters=абвгдежзик] [label=5] [test=5] [seed=1]
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
    opts = {"fit": "lgn-a", "homeo": f"lgn-b-s{seed}", "relay_v_t_mv": "-59", "trn_scale": "0.1", **opts}
    mind, b = build(opts)
    cfg = b.cfg
    s = Session(mind, window_ms=WIN)
    letters = opts.get("letters", "абвгдежзик")
    n_label, n_test = int(opts.get("label", 5)), int(opts.get("test", 5))
    show_ms, rest_ms = float(opts.get("show", 350)), float(opts.get("rest", 150))
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    vis = b.areas["visual"].layers
    post = vis["L23"]["E"]
    l4 = vis["L4"]["E"]
    paths = [p for p in b.projections if p.post is post and p.source == "net"]
    print("входы V1 L2/3 E:")
    for p in paths:
        print(f"  {p.name}: {p.weight.shape[1]} входов на клетку, вес {float(p.weight.float().mean()):+.3f} мВ", flush=True)
    names = [p.name for p in paths]
    keys = [f"вход: {n}" for n in names] + ["ток возбуждения", "ток торможения", "сумма токов", "мембрана", "спайки",
                                            "спайки V1 L4 E"]

    def window(view: torch.Tensor) -> list[torch.Tensor]:
        r = b.observe(WIN, view)
        v = b.v[post.slice]
        excite = ((cfg.ampa_reversal_mv - v) / (cfg.ampa_reversal_mv - cfg.epsp_holding_mv)).clamp(min=0.0)
        force = ((v - cfg.gaba_reversal_mv) / (cfg.ipsp_holding_mv - cfg.gaba_reversal_mv)).clamp(min=0.0)
        e = (b.i_syn[post.slice] + b.dc_mv[post.slice] if torch.is_tensor(b.dc_mv) else b.i_syn[post.slice]) * excite
        # i_inh и i_ngf — со знаком (торможение отрицательно).
        i = (b.i_inh[post.slice] + b.i_ngf[post.slice]) * force
        out = [(p.weight.float() * r[p.pre.long()]).sum(1) for p in paths]
        out += [e, i, e + i, v.clone(), r[post.slice] * WIN / 1000.0, r[l4.slice] * WIN / 1000.0]
        return [x.float() for x in out]

    def show(ch: str) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        s.line = ""
        bg = None
        for _ in range(int(rest_ms // WIN)):
            w = window(s._blank(WIN))
            bg = w if bg is None else [a + c for a, c in zip(bg, w)]
        s.line, s.gaze = ch, s.screen.width(ch) / 2 + float(torch.randint(-1, 2, (1,), generator=gen))
        acc = None
        for _ in range(int(show_ms // WIN)):
            w = window(s._view(WIN))
            acc = w if acc is None else [a + c for a, c in zip(acc, w)]
        s.line = ""
        n_bg, n_sh = int(rest_ms // WIN), int(show_ms // WIN)
        mean_bg = [x / n_bg for x in bg]
        mean_sh = [x / n_sh for x in acc]
        # Спайки — счёт за показ, остальное — среднее по окнам.
        mean_sh[-2], mean_sh[-1] = acc[-2], acc[-1]
        return [x.cpu() for x in mean_sh], [x.cpu() for x in mean_bg]

    s.line, s.gaze = "", 0.0
    b.run(2500.0, s._blank(2500.0))
    t0 = time.time()
    rows, bgs, labels = [], [], []
    for rep in range(n_label + n_test):
        for k in torch.randperm(len(letters), generator=gen).tolist():
            sh, bg = show(letters[k])
            rows.append(sh)
            bgs.append(bg)
            labels.append(k)
        print(f"  проход {rep + 1}/{n_label + n_test}, {time.time() - t0:.0f} с", flush=True)

    y = torch.tensor(labels)
    lab = torch.arange(len(y)) < n_label * len(letters)
    z = lambda x: (x - x.mean(-1, keepdim=True)) / x.std(-1, keepdim=True).clamp_min(1e-9)
    print(f"\nV1 L2/3 E: различение букв по величине (корреляция; случайно {1 / len(letters):.2f}); "
          f"среднее по клеткам — пауза → показ:")
    for j, key in enumerate(keys):
        X = torch.stack([r[j] for r in rows])
        B = torch.stack([r[j] for r in bgs])
        mean = torch.stack([X[lab][y[lab] == k].mean(0) for k in range(len(letters))])
        acc = float(((z(X[~lab]) @ z(mean).T).argmax(1) == y[~lab]).float().mean())
        unit = "спайков за показ" if key.startswith("спайки") else ("мВ" if key in ("мембрана",) or key.startswith("ток") or key == "сумма токов" else "мВ·Гц")
        bg_txt = f"{float(B.mean()):9.2f} → " if not key.startswith("спайки") else " " * 13
        print(f"  {key:44s} {acc:5.2f}   {bg_txt}{float(X.mean()):9.2f} {unit}", flush=True)
    print(f"всего {time.time() - t0:.0f} с")


if __name__ == "__main__":
    main(dict(kv.split("=", 1) for kv in sys.argv[1:]))
