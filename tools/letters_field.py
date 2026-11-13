"""
Буквы по всему полю зоны букв — первая ступень LCD (Dehaene et al. 2005:
детекторы букв в нескольких позициях; DESIGN 2026-11-13 «Проект: ступень
локальных сочетаний букв»). Правило и учитель — как в letters_us (cfg.us_*,
engine._us, engine.teach_*), путь V1 → буквы (visual -> letters).

Мозг как в letters_us. Группы: каждая клетка L4 E зоны букв навсегда
приписана к букве случайной перестановкой (не рангом расстояния — тот
разводит буквы по сторонам ямки). Показ: буква 350 мс, центр — случайное
место строки (4…52 px), пауза 150 мс. Учитель: клетки, в окне входов
которых центр буквы (|место − центр| ≤ band px; band 6 — половина окна
V1 → буквы 12 px): своя группа стреляет teacher_hz, прочие в полосе
молчат; вне полосы клетки живут сами.

Проверка (учитель и правило ВЫКЛ): буква в случайных местах, ответ —
группа с наибольшей средней частотой среди клеток той же полосы; точность
всего и по местам (7 корзин по 7 px), контроль — перемешанные метки ×100.

    PYTHONPATH=. .venv/bin/python tools/letters_field.py [train=50] [test=10] [seed=1] [band=6] [out=…] [load=…] [learn=1] [shows=N]
"""

from __future__ import annotations

import sys
import time

import torch

from brain.session import Session
from tools.reading_bench import build


def main(opts: dict) -> None:
    seed = int(opts.get("seed", 1))
    opts = {"fit": "lgn-a", "homeo": f"lgn-d-pop-tcind-s{seed}", "relay_v_t_mv": "-59", "trn_scale": "0.1",
            "us": "visual -> letters", **opts}
    mind, b = build(opts)
    s = Session(mind, window_ms=10.0)
    letters = opts.get("letters", "абвгдежзик")
    L = len(letters)
    n_train, n_test = int(opts.get("train", 50)), int(opts.get("test", 10))
    band = float(opts.get("band", 6.0))
    learn = opts.get("learn", "1") == "1"
    show_ms, rest_ms = 350.0, 150.0
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    dev = b.device
    path = b.us[0]
    P = path.post
    cols = b.field_cols[P.name].to(dev)
    perm = torch.randperm(P.size, generator=torch.Generator().manual_seed(1000 + seed)).to(dev)
    group = torch.empty(P.size, dtype=torch.long, device=dev)
    group[perm] = torch.arange(P.size, device=dev) % L                  # постоянная группа клетки
    if opts.get("load"):
        path.weight.copy_(torch.load(opts["load"], map_location=dev)["weight"])
    w0 = path.weight.clone()
    width = b.field_width
    print(f"буквы «{letters}»: учёба {n_train} показов каждой в случайных местах, учитель и правило {'вкл' if learn else 'ВЫКЛ'}; "
          f"полоса учителя ±{band:.0f} px; клеток {P.size} ({P.size / L / width:.1f} каждой буквы на px)", flush=True)

    s.line, s.gaze = "", 0.0
    b.run(2500.0, s._blank(2500.0))
    lo, hi = 4.0, width - 4.0

    def show(k: int, centre: float, train: bool) -> torch.Tensor:
        """Буква k с центром в месте centre строки; частоты клеток полосы по группам."""

        s.line = letters[k]
        # Картинка сдвинута так, что точка взгляда — в fovea_col: центр
        # буквы (width/2 от начала строки) попадает в fovea_col − (gaze − w/2).
        s.gaze = s.screen.width(letters[k]) / 2 + (b.cfg.fovea_col - centre)
        inband = (cols - centre).abs() <= band
        b.teach_pop.zero_()
        b.teach_target.zero_()
        b.teach_pop[P.start:P.stop] = inband
        b.teach_target[P.start:P.stop] = inband & (group == k)
        b.teach_on = b.us_on = train
        r = b.observe(show_ms, s._view(show_ms))[P.slice]
        b.teach_on = b.us_on = False
        s.line = ""
        b.run(rest_ms, s._blank(rest_ms))
        return torch.stack([r[inband & (group == g)].float().mean() for g in range(L)])

    t0 = time.time()
    order = torch.arange(L).repeat(n_train)
    order = order[torch.randperm(len(order), generator=gen)]
    places = lo + torch.rand(len(order), generator=gen) * (hi - lo)
    if "shows" in opts:
        # Другие показы учёбы (места и порядок) при продолжении; генератор
        # зерна израсходован как прежде — проверка та же, что у прежних кусков.
        tg = torch.Generator().manual_seed(int(opts["shows"]))
        order = order[torch.randperm(len(order), generator=tg)]
        places = lo + torch.rand(len(order), generator=tg) * (hi - lo)
    if n_train and learn:
        for j, (k, c) in enumerate(zip(order.tolist(), places.tolist())):
            show(k, c, True)
            if (j + 1) % 100 == 0:
                dw = float((path.weight - w0).abs().mean() / w0.mean())
                print(f"  учёба {j + 1}/{len(order)}, {time.time() - t0:.0f} с: |Δw| {dw:.1%}", flush=True)
    if opts.get("out"):
        torch.save({"weight": path.weight.cpu(), "group": group.cpu(), "opts": opts}, opts["out"])

    idx = torch.arange(L).repeat(n_test)
    idx = idx[torch.randperm(len(idx), generator=gen)]
    where = lo + torch.rand(len(idx), generator=gen) * (hi - lo)
    G = torch.stack([show(k, c, False) for k, c in zip(idx.tolist(), where.tolist())]).cpu()
    pred = G.argmax(1)
    ok = (pred == idx).float()
    null = torch.tensor([float((pred == idx[torch.randperm(len(idx))]).float().mean()) for _ in range(100)])
    print(f"\nпроверка {n_test} показов каждой в случайных местах (случайно {1 / L:.2f}; перемешано×100: 95% {float(null.quantile(0.95)):.2f}):")
    print(f"  буква по группам полосы {float(ok.mean()):.2f}; |Δw| {float((path.weight - w0).abs().mean() / w0.mean()):.1%}; "
          f"частота полосы на букве {float(G.mean()):.2f} Гц")
    bins = torch.clamp(((where - lo) / (hi - lo) * 7).long(), max=6)
    print("  по местам (центр корзины, px: верно/показов): " + "  ".join(
        f"{float(lo + (i + 0.5) * (hi - lo) / 7):.0f}: {int(ok[bins == i].sum())}/{int((bins == i).sum())}" for i in range(7)))
    conf = torch.zeros(L, L, dtype=torch.long)
    for a, p_ in zip(idx.tolist(), pred.tolist()):
        conf[a, p_] += 1
    print("  верно по буквам: " + " ".join(f"{letters[k]} {int(conf[k, k])}/{int(conf[k].sum())}" for k in range(L)))
    print(f"всего {time.time() - t0:.0f} с")


if __name__ == "__main__":
    main(dict(kv.split("=", 1) for kv in sys.argv[1:]))
