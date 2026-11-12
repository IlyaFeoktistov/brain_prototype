"""
Где буква теряется между V1 L2/3 и зоной букв (замер, без вмешательства;
мозг как в letters_dc: LGN A, гомеостаз lgn-d, без пластичности, путь
V1 → буквы врождённый). Четыре вопроса:

  1. проводка: сколько клеток V1 L2/3 E подключено к клетке букв L4 E, веса,
     какая доля всех V1 L2/3 E вообще куда-то подключена, и сколько входов
     клетки реально получает спайки на показе;
  2. информация на входе: различают ли буквы спайки именно подключённых
     клеток V1 L2/3 E (а не всей популяции) и вход пути у клетки букв
     (Σ вес × частота источника) — что зона реально получает;
  3. токи: вход каждого построенного пути, среднее поле недостающих
     соседей пути V1 (ток по частоте всей популяции — буквы в нём нет),
     ток возбуждения и торможения сомы с движущей силой, их сумма;
  4. ответ: мембрана, спайки букв L4 E, спайки букв L2/3 E.

По каждой величине — различение тем же классификатором по корреляции, что
в letters_dc и l23_trace (центр буквы — средний ответ на метках, ответ —
центр с наибольшей корреляцией). Если вход пути различает, а спайки нет —
буква теряется внутри зоны (суммирование, порог); если не различают уже
спайки подключённых клеток — проводка (какие клетки подключены).

    PYTHONPATH=. .venv/bin/python tools/letters_trace.py [letters=абвгдежзик] [label=5] [test=5] [seed=1]
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
    opts = {"fit": "lgn-a", "homeo": f"lgn-d-pop-s{seed}", "relay_v_t_mv": "-59", "trn_scale": "0.1", **opts}
    mind, b = build(opts)
    cfg = b.cfg
    s = Session(mind, window_ms=WIN)
    letters = opts.get("letters", "абвгдежзик")
    n_label, n_test = int(opts.get("label", 5)), int(opts.get("test", 5))
    show_ms, rest_ms = float(opts.get("show", 350)), float(opts.get("rest", 150))
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    vis, let = b.areas["visual"].layers, b.areas["letters"].layers
    post, l23 = let["L4"]["E"], let["L23"]["E"]
    src = vis["L23"]["E"]
    paths = [p for p in b.projections if p.post is post and p.source == "net"]
    v1 = next(p for p in paths if p.pre.min() >= src.start and p.pre.max() < src.start + src.size)
    tau = float(b.tau_m[post.start])

    # 1. Проводка.
    pre = v1.pre.long()
    w = v1.weight.float()
    connected = torch.unique(pre)
    print(f"путь {v1.name}: клеток букв L4 E {post.size}, входов на клетку {pre.shape[1]}, "
          f"разных источников у клетки (медиана) {float(torch.tensor([len(torch.unique(r)) for r in pre]).float().median()):.0f}; "
          f"вес {float(w.mean()):.3f} мВ (5–95% {float(w.quantile(0.05)):.3f}–{float(w.quantile(0.95)):.3f}); "
          f"подключено V1 L2/3 E {len(connected)} из {src.size} ({len(connected) / src.size:.0%})", flush=True)
    print("входы букв L4 E (построенные):")
    for p in paths:
        print(f"  {p.name}: {p.weight.shape[1]} входов на клетку, вес {float(p.weight.float().mean()):+.3f} мВ", flush=True)
    field = [(pp, pr, per, ach) for pp, pr, per, _, ach, _ in b.mean_field if pp is post]
    for pp, pr, per, _ in field:
        print(f"  среднее поле от {pr.name}: {per:.1f} мВ на кГц частоты источника", flush=True)
    v1_field = sum(per for pp, pr, per, _ in field if pr is src)

    keys = ["спайки V1 L2/3 E (все)", "спайки V1 L2/3 E (подключённые)"]
    keys += [f"вход: {p.name}" for p in paths]
    keys += ["V1: построенное, мВ", "V1: среднее поле, мВ", "активных входов V1 (спайк за показ)",
             "ток возбуждения", "ток торможения", "сумма токов", "мембрана", "спайки букв L4 E", "спайки букв L2/3 E"]
    counts = {"спайки V1 L2/3 E (все)", "спайки V1 L2/3 E (подключённые)", "спайки букв L4 E", "спайки букв L2/3 E",
              "активных входов V1 (спайк за показ)"}

    def window(view: torch.Tensor, fired: torch.Tensor) -> list[torch.Tensor]:
        r = b.observe(WIN, view)
        v = b.v[post.slice]
        excite = ((cfg.ampa_reversal_mv - v) / (cfg.ampa_reversal_mv - cfg.epsp_holding_mv)).clamp(min=0.0)
        force = ((v - cfg.gaba_reversal_mv) / (cfg.ipsp_holding_mv - cfg.gaba_reversal_mv)).clamp(min=0.0)
        e = (b.i_syn[post.slice] + b.dc_mv[post.slice] if torch.is_tensor(b.dc_mv) else b.i_syn[post.slice]) * excite
        i = (b.i_inh[post.slice] + b.i_ngf[post.slice]) * force
        spikes = r * WIN / 1000.0
        fired |= spikes[src.slice] > 0
        out = [spikes[src.slice], spikes[connected]]
        out += [(p.weight.float() * r[p.pre.long()]).sum(1) for p in paths]
        built = (w * r[pre]).sum(1) / 1000.0 * tau
        out += [built, torch.full_like(built, v1_field * float(r[src.slice].mean()) / 1000.0), torch.zeros_like(built)]
        out += [e, i, e + i, v.clone(), spikes[post.slice], spikes[l23.slice]]
        return [x.float() for x in out]

    def show(ch: str) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        s.line = ""
        n_bg, n_sh = int(rest_ms // WIN), int(show_ms // WIN)
        fired = torch.zeros(src.size, dtype=torch.bool, device=b.device)
        bg = None
        for _ in range(n_bg):
            x = window(s._blank(WIN), fired)
            bg = x if bg is None else [a + c for a, c in zip(bg, x)]
        fired.zero_()
        s.line, s.gaze = ch, s.screen.width(ch) / 2 + float(torch.randint(-1, 2, (1,), generator=gen))
        acc = None
        for _ in range(n_sh):
            x = window(s._view(WIN), fired)
            acc = x if acc is None else [a + c for a, c in zip(acc, x)]
        s.line = ""
        sh = [a / n_sh for a in acc]
        bgm = [a / n_bg for a in bg]
        for j, key in enumerate(keys):
            if key in counts:
                sh[j], bgm[j] = acc[j], bg[j] * (n_sh / n_bg)
        active = fired[pre - src.start].float().sum(1)
        sh[keys.index("активных входов V1 (спайк за показ)")] = active
        bgm[keys.index("активных входов V1 (спайк за показ)")] = torch.zeros_like(active)
        return [x.cpu() for x in sh], [x.cpu() for x in bgm]

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
    print(f"\nразличение букв по величине (корреляция; случайно {1 / len(letters):.2f}); "
          f"среднее по клеткам — пауза → показ (спайки — счёт за {show_ms:.0f} мс):")
    for j, key in enumerate(keys):
        X = torch.stack([r[j] for r in rows])
        B = torch.stack([r[j] for r in bgs])
        if float(X.std()) == 0.0:
            acc = float("nan")
        else:
            mean = torch.stack([X[lab][y[lab] == k].mean(0) for k in range(len(letters))])
            acc = float(((z(X[~lab]) @ z(mean).T).argmax(1) == y[~lab]).float().mean())
        print(f"  {key:44s} {acc:5.2f}   {float(B.mean()):9.3f} → {float(X.mean()):9.3f}", flush=True)

    # Какие источники подключены: отвечают ли на буквы (показ − пауза, |t| > 2)
    # и различают ли их (разброс средних по буквам больше разброса повторов).
    X = torch.stack([r[0] for r in rows])
    B = torch.stack([r[0] for r in bgs])
    d = X - B
    t = d.mean(0) / (d.std(0) / len(d) ** 0.5).clamp_min(1e-9)
    means = torch.stack([X[y == k].mean(0) for k in range(len(letters))])
    within = torch.stack([X[y == k].var(0) for k in range(len(letters))]).mean(0)
    n_rep = n_label + n_test
    informative = means.var(0) > 2.0 * within / n_rep
    mask = torch.zeros(src.size, dtype=torch.bool)
    mask[(connected - src.start).cpu()] = True
    for name, m in (("все", torch.ones_like(mask)), ("подключённые", mask)):
        print(f"V1 L2/3 E {name}: отвечают вверх {float((t[m] > 2).float().mean()):.0%}, "
              f"различают буквы {float(informative[m].float().mean()):.0%}", flush=True)
    resp = (t > 2).to(pre.device)
    info = informative.to(pre.device)
    print(f"у клетки букв L4 E из {pre.shape[1]} входов V1: отвечающих {float(resp[pre - src.start].float().sum(1).mean()):.1f}, "
          f"различающих {float(info[pre - src.start].float().sum(1).mean()):.1f}", flush=True)
    print(f"всего {time.time() - t0:.0f} с")


if __name__ == "__main__":
    main(dict(kv.split("=", 1) for kv in sys.argv[1:]))
