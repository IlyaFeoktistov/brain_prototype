"""
Блок Diehl & Cook в зоне букв мозга (DESIGN «Сборка из готовых блоков»,
блок 1): учится ли зона букв различать отдельные буквы на входе от нашей V1.

Мозг как в reading_bench (LGN A, фон, гомеостаз зерна); на пути V1 → буквы
(visual -> letters) — правило блока (cfg.dc_paths, engine._dc) с адаптивным
порогом клеток букв L4 E, перед каждым учебным показом — нормировка суммы
весов клетки (engine.dc_normalize). Показ как у авторов: буква 350 мс, пусто
150 мс; буква — по центру ямки со случайным сдвигом ±1 px.

Части:
    учёба     train показов каждой буквы вперемешку, правило блока включено
              (learn=0 — контроль: те же показы без правила);
    метки     label показов каждой буквы без правила: клетке — буква с
              наибольшим средним ответом (как у авторов);
    проверка  test новых показов каждой буквы без правила: ответ — буква с
              наибольшим средним ответом своих клеток.
Те же метки и проверка — для V1 L2/3 (не учится; сколько различает сам
вход) и букв L2/3.

    PYTHONPATH=. python tools/letters_dc.py [letters=абвгдежзик] [train=40] [label=10] [test=10] [learn=1] [seed=1]
"""

from __future__ import annotations

import sys
import time

import torch

from brain.session import Session
from tools.reading_bench import build


def main(opts: dict) -> None:
    opts = {"fit": "lgn-a", "homeo": f"lgn-a-s{opts.get('seed', 1)}", "relay_v_t_mv": "-59",
            "relay_rest_mv": "-48", "trn_scale": "0.5", "dc": "visual -> letters", **opts}
    mind, b = build(opts)
    s = Session(mind, window_ms=10.0)
    letters = opts.get("letters", "абвгдежзик")
    n_train, n_label, n_test = int(opts.get("train", 40)), int(opts.get("label", 10)), int(opts.get("test", 10))
    learn = opts.get("learn", "1") == "1"
    show_ms, rest_ms = float(opts.get("show", 350)), float(opts.get("rest", 150))
    gen = torch.Generator().manual_seed(int(opts.get("seed", 1)))
    torch.manual_seed(int(opts.get("seed", 1)))
    vis, let = b.areas["visual"].layers, b.areas["letters"].layers
    stages = {"LGN": b.areas["lgn"].E, "V1 L4": vis["L4"]["E"], "V1 L2/3": vis["L23"]["E"],
              "буквы L4": let["L4"]["E"], "буквы L2/3": let["L23"]["E"]}
    L4 = let["L4"]["E"]
    path = b.dc[0]
    w0 = path.weight.clone()
    print(f"буквы «{letters}», учёба {n_train} показов каждой, правило блока {'вкл' if learn else 'выкл — контроль'}; "
          f"путь {path.name}: {path.pre.shape[1]} входов на клетку", flush=True)

    s.line, s.gaze = "", 0.0
    b.run(2500.0)

    def show(ch: str, train: bool) -> dict[str, torch.Tensor]:
        s.line = ch
        s.gaze = s.screen.width(ch) / 2 + float(torch.randint(-1, 2, (1,), generator=gen))
        if train:
            b.dc_normalize()
        b.dc_on = train
        r = b.observe(show_ms, s._view(show_ms)) * (show_ms / 1000.0)
        s.line = ""
        b.run(rest_ms, s._blank(rest_ms))
        b.dc_on = False
        return {k: r[pop.slice].cpu() for k, pop in stages.items()}

    t0 = time.time()
    order = torch.arange(len(letters)).repeat(n_train)
    order = order[torch.randperm(len(order), generator=gen)]
    spikes = 0.0
    for j, k in enumerate(order.tolist()):
        out = show(letters[k], learn)
        spikes += float(out["буквы L4"].sum())
        if (j + 1) % 50 == 0:
            theta = (b.v_t - b.v_t_base)[L4.slice]
            dw = float((path.weight - w0).abs().mean() / w0.mean())
            print(f"  учёба {j + 1}/{len(order)}, {time.time() - t0:.0f} с: спайков букв L4 на показ {spikes / 50:.1f}, "
                  f"θ {float(theta.mean()):.2f} мВ (макс {float(theta.max()):.2f}), |Δw| {dw:.1%} среднего", flush=True)
            spikes = 0.0

    def collect(n: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        idx = torch.arange(len(letters)).repeat(n)
        idx = idx[torch.randperm(len(idx), generator=gen)]
        outs = [show(letters[k], False) for k in idx.tolist()]
        return {key: torch.stack([o[key] for o in outs]) for key in stages}, idx

    r_lab, y_lab = collect(n_label)
    r_test, y_test = collect(n_test)
    print(f"\nметки {n_label} и проверка {n_test} показов каждой буквы, без правила "
          f"(случайно {1 / len(letters):.2f}):")
    for key in stages:
        mean = torch.stack([r_lab[key][y_lab == k].mean(0) for k in range(len(letters))])
        label = mean.argmax(0)
        active = mean.max(0).values > 0
        score = torch.stack([r_test[key][:, (label == k) & active].mean(1) if ((label == k) & active).any()
                             else torch.full((len(y_test),), -1.0) for k in range(len(letters))], 1)
        acc = float((score.argmax(1) == y_test).float().mean())
        # Классификатор по корреляции: центр буквы — средний ответ на метках,
        # ответ — центр с наибольшей корреляцией (как l4_decode).
        z = lambda x: (x - x.mean(-1, keepdim=True)) / x.std(-1, keepdim=True).clamp_min(1e-9)
        corr = z(r_test[key]) @ z(mean).T / mean.shape[1]
        acc_c = float((corr.argmax(1) == y_test).float().mean())
        print(f"  {key:12s} точность {acc:.2f} (голос клеток), {acc_c:.2f} (корреляция); клеток с ответом {int(active.sum())}, букв с клетками "
              f"{len(set(label[active].tolist()))}/{len(letters)}, спайков на показ {float(r_test[key].sum(1).mean()):.1f}",
              flush=True)
    print(f"всего {time.time() - t0:.0f} с")


if __name__ == "__main__":
    main(dict(kv.split("=", 1) for kv in sys.argv[1:]))
