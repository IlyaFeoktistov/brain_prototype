"""
Зона букв мозга учится буквам с учителем (DESIGN 2026-11-12): путь V1 → буквы
(visual -> letters) — правило cfg.us_* (engine._us; Pfister et al. 2006 /
Urbanczik & Senn 2014), учитель — зажим спайков (engine.teach_*): 200 клеток
букв L4 E у ямки, по 20 на букву (группы чередуются по расстоянию до центра —
у каждой клетки во всех местах поля).

Мозг как в reading_bench (гомеостаз зерна), путь глаз → LGN → V1 → буквы.
Показ: буква 350 мс по центру ямки (сдвиг ±1 px), пауза 150 мс.
    учёба     train показов каждой буквы вперемешку: учитель вкл (группа буквы
              стреляет cfg.teacher_hz, прочие 180 молчат), правило вкл;
              learn=0 — контроль: те же показы, учителя и правила нет;
    проверка  test новых показов каждой буквы: учитель и правило ВЫКЛ —
              буква = группа с наибольшим числом спайков (только зрительный
              путь); и по одним весам: счёт V1 L2/3 за показ × веса пути,
              сумма по группе (без динамики клеток букв).
Сохранение: out=файл — веса пути после учёбы (повторная проверка — load=файл).

    PYTHONPATH=. .venv/bin/python tools/letters_us.py [train=30] [test=10] [learn=1] [seed=1] [out=…] [load=…] [detail=1]
"""

from __future__ import annotations

import sys
import time

import torch

from brain.session import Session
from tools.reading_bench import build


def main(opts: dict) -> None:
    seed = int(opts.get("seed", 1))
    opts = {"fit": "lgn-a", "homeo": f"lgn-d-pop{'' if opts.get('tc_indep') == '0' else '-tcind'}-s{seed}",
            "relay_v_t_mv": "-59", "trn_scale": "0.1", "us": "visual -> letters", **opts}
    mind, b = build(opts)
    s = Session(mind, window_ms=10.0)
    letters = opts.get("letters", "абвгдежзик")
    L = len(letters)
    n_train, n_test = int(opts.get("train", 30)), int(opts.get("test", 10))
    learn = opts.get("learn", "1") == "1"
    show_ms, rest_ms = 350.0, 150.0
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    path = b.us[0]
    L4 = path.post
    v1 = b.areas["visual"].layers["L23"]["E"]
    dev = b.device
    near = (b.field_cols[L4.name] - b.cfg.fovea_col).abs().argsort()[:20 * L].to(dev)
    group = torch.arange(len(near), device=dev) % L          # чередование по расстоянию
    cells = near + L4.start
    if opts.get("load"):
        path.weight.copy_(torch.load(opts["load"], map_location=dev)["weight"])
    w0 = path.weight.clone()
    print(f"буквы «{letters}», учёба {n_train} показов каждой, учитель и правило {'вкл' if learn else 'ВЫКЛ — контроль'}; "
          f"путь {path.name}: {path.pre.shape[1]} входов на клетку; учимых клеток {len(near)} (по {len(near) // L} на букву)", flush=True)

    s.line, s.gaze = "", 0.0
    b.run(2500.0, s._blank(2500.0))
    b.teach_pop.zero_()
    b.teach_pop[cells] = True

    blank: list[torch.Tensor] = []                            # частоты учимых на паузе после показа, Гц

    def show(k: int, train: bool) -> tuple[torch.Tensor, torch.Tensor]:
        s.line = letters[k]
        s.gaze = s.screen.width(letters[k]) / 2 + float(torch.randint(-1, 2, (1,), generator=gen))
        b.teach_target.zero_()
        b.teach_target[cells[group == k]] = True
        b.teach_on = b.us_on = train
        r = b.observe(show_ms, s._view(show_ms)) * (show_ms / 1000.0)
        b.teach_on = b.us_on = False
        s.line = ""
        rb = b.observe(rest_ms, s._blank(rest_ms))
        blank.append(rb[cells].float())
        return r[cells].float(), r[v1.slice].float()

    t0 = time.time()
    order = torch.arange(L).repeat(n_train)
    order = order[torch.randperm(len(order), generator=gen)]
    if n_train and learn:
        for j, k in enumerate(order.tolist()):
            r, _ = show(k, True)
            if (j + 1) % 50 == 0:
                dw = float((path.weight - w0).abs().mean() / w0.mean())
                own = float(r[group == k].sum()) / max(float(r.sum()), 1.0)
                print(f"  учёба {j + 1}/{len(order)}, {time.time() - t0:.0f} с: |Δw| {dw:.1%}, доля спайков своей группы {own:.0%}", flush=True)
    elif n_train:
        for k in order.tolist():
            show(k, False)
    if opts.get("out"):
        torch.save({"weight": path.weight.cpu(), "opts": opts}, opts["out"])

    # Проверка без учителя.
    blank.clear()
    idx = torch.arange(L).repeat(n_test)
    idx = idx[torch.randperm(len(idx), generator=gen)]
    outs = [show(k, False) for k in idx.tolist()]
    R = torch.stack([o[0] for o in outs])                     # [показ, учимые клетки]
    X = torch.stack([o[1] for o in outs])                     # [показ, V1 L2/3]
    y = idx.to(dev)
    g = torch.stack([R[:, group == k].sum(1) for k in range(L)], 1)
    acc = float((g.argmax(1) == y).float().mean())
    silent = float((g.sum(1) == 0).float().mean())
    # По весам: вход клетки = Σ w · счёт источника V1 L2/3 за показ.
    pre = path.pre[near].long() - v1.start
    ok = (pre >= 0) & (pre < v1.size)
    w = path.weight[near].float() * ok
    drive = torch.stack([(w * X[i][pre.clamp(0, v1.size - 1)]).sum(1) for i in range(len(X))])
    gw = torch.stack([drive[:, group == k].sum(1) for k in range(L)], 1)
    acc_w = float(((gw - gw.mean(1, keepdim=True)).argmax(1) == y).float().mean())
    null = torch.tensor([float((g.argmax(1) == y[torch.randperm(len(y), device=dev)]).float().mean()) for _ in range(100)])
    print(f"\nпроверка {n_test} показов каждой, учитель ВЫКЛ (случайно {1 / L:.2f}; перемешано×100: 95% {float(null.quantile(0.95)):.2f}):")
    print(f"  буква по спайкам групп {acc:.2f} (без спайков {silent:.0%}), по весам {acc_w:.2f}; спайков учимых клеток на показ "
          f"{float(R.sum(1).mean()):.1f}; |Δw| {float((path.weight - w0).abs().mean() / w0.mean()):.1%}", flush=True)
    conf = torch.zeros(L, L, dtype=torch.long)
    for a, p_ in zip(y.tolist(), g.argmax(1).tolist()):
        conf[a, p_] += 1
    print("  верно по буквам: " + " ".join(f"{letters[k]} {int(conf[k, k])}/{int(conf[k].sum())}" for k in range(L)))
    if opts.get("detail") == "1":
        # Где перекос (DESIGN 2026-11-13): матрица путаницы, ответ групп на
        # буквы, фон групп на паузе, ответ − фон, активность V1 L2/3 на букву.
        print("\n  путаница (строка — показанная буква, столбец — ответ):")
        print("     " + " ".join(f"{c:>3}" for c in letters))
        for k in range(L):
            print(f"  {letters[k]}  " + " ".join(f"{int(conf[k, j]):3d}" for j in range(L)))
        rate = torch.stack([R[:, group == k].mean(1) for k in range(L)], 1) / (show_ms / 1000.0)   # [показ, группа], Гц на клетку
        Bk = torch.stack(blank)
        base = torch.stack([Bk[:, group == k].mean(1) for k in range(L)], 1).mean(0)               # [группа], Гц
        M = torch.stack([rate[y == k].mean(0) for k in range(L)], 0).cpu()                         # [буква, группа]
        print("\n  ответ групп, Гц на клетку (строка — буква, столбец — группа); последняя строка — фон на паузе:")
        print("     " + " ".join(f"{c:>5}" for c in letters))
        for k in range(L):
            print(f"  {letters[k]}  " + " ".join(f"{float(M[k, j]):5.2f}" for j in range(L)))
        print("  фон " + " ".join(f"{float(x):5.2f}" for x in base.cpu()))
        D = M - base.cpu()[None, :]
        print("\n  ответ − фон, Гц (строка — буква, столбец — группа); * — максимум строки:")
        print("     " + " ".join(f"{c:>6}" for c in letters))
        for k in range(L):
            j = int(D[k].argmax())
            print(f"  {letters[k]}  " + " ".join(f"{float(D[k, i]):+5.2f}{'*' if i == j else ' '}" for i in range(L)))
        own = D.diagonal()
        other = (D.sum(1) - own) / (L - 1)
        print("  своя − средняя чужая (по буквам): " + " ".join(f"{letters[k]} {float(own[k] - other[k]):+.2f}" for k in range(L)))
        print("  средний ответ группы на все буквы − фон: " + " ".join(f"{letters[k]} {float(D[:, k].mean()):+.2f}" for k in range(L)))
        v1r = torch.stack([X[y == k].sum(1).mean() for k in range(L)]).cpu() / v1.size / (show_ms / 1000.0)
        print("  V1 L2/3, средняя частота на букве, Гц: " + " ".join(f"{letters[k]} {float(v1r[k]):.2f}" for k in range(L)))
    print(f"всего {time.time() - t0:.0f} с")


if __name__ == "__main__":
    main(dict(kv.split("=", 1) for kv in sys.argv[1:]))
