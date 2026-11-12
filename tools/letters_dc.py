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

rule=native — вместо блока родное правило мозга на том же пути: STDP сразу
в вес (modulator "hebbian", engine.chemical_plasticity) и сохранение суммы
входа клетки (engine._conserve), всё как у мозга. Включена общая
пластичность — после каждого учебного показа веса всех прочих пластичных
путей (обратный association → letters и др.) возвращаются к исходным:
учится только V1 → буквы, V1 не пластична (источник тот же). Контроль с
перемешанными метками (клеткам — метки по перемешанным показам) —
печатается рядом. Дополнительно: доля избирательных к букве клеток (F по
показам меток против перемешанных), распределение весов пути (доли
среднего врождённого), схожесть выученного у клеток у ямки — Δw, собранный
по источникам V1 L2/3, корреляция между клетками (≈1 — все выучили одно).

    PYTHONPATH=. python tools/letters_dc.py [letters=абвгдежзик] [train=40] [label=10] [test=10] [learn=1] [rule=dc|native] [seed=1]
"""

from __future__ import annotations

import sys
import time

import torch

from brain.session import Session
from tools.reading_bench import build


def main(opts: dict) -> None:
    native = opts.get("rule", "dc") == "native"
    opts = {"fit": "lgn-a", "homeo": f"lgn-d-s{opts.get('seed', 1)}", "relay_v_t_mv": "-59",
            "trn_scale": "0.1", **({} if native else {"dc": "visual -> letters"}), **opts}
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
    path = next(p for p in b.plastic if p.name == "visual -> letters") if native else b.dc[0]
    w0 = path.weight.clone()
    # Прочие пластичные пути — к исходным весам после каждого учебного показа.
    others = [(p, p.weight.clone()) for p in b.plastic if p is not path] if native else []
    rule = "родное (STDP + сохранение суммы)" if native else "блока D&C"
    print(f"буквы «{letters}», учёба {n_train} показов каждой, правило {rule} {'вкл' if learn else 'выкл — контроль'}; "
          f"путь {path.name}: {path.pre.shape[1]} входов на клетку", flush=True)

    s.line, s.gaze = "", 0.0
    b.run(2500.0, s._blank(2500.0))

    def show(ch: str, train: bool) -> dict[str, torch.Tensor]:
        s.line = ch
        s.gaze = s.screen.width(ch) / 2 + float(torch.randint(-1, 2, (1,), generator=gen))
        if train and not native:
            b.dc_normalize()
        b.dc_on = train and not native
        b.plasticity = train and native
        r = b.observe(show_ms, s._view(show_ms)) * (show_ms / 1000.0)
        s.line = ""
        b.dc_on = False
        b.plasticity = False
        for p, w in others:
            p.weight.copy_(w)
        # Фон — пауза после показа (частоты, Гц): для распределения частот.
        r0 = b.observe(rest_ms, s._blank(rest_ms))
        out = {k: r[pop.slice].cpu() for k, pop in stages.items()}
        out.update({"фон " + k: r0[pop.slice].cpu() for k, pop in stages.items()})
        return out

    t0 = time.time()
    order = torch.arange(len(letters)).repeat(n_train)
    order = order[torch.randperm(len(order), generator=gen)]
    spikes = 0.0
    for j, k in enumerate(order.tolist()):
        out = show(letters[k], learn)
        spikes += float(out["буквы L4"].sum())
        if (j + 1) % 50 == 0 or j + 1 == 10:
            theta = (b.v_t - getattr(b, "v_t_base", b.v_t))[L4.slice]
            dw = float((path.weight - w0).abs().mean() / w0.mean())
            print(f"  учёба {j + 1}/{len(order)}, {time.time() - t0:.0f} с: спайков букв L4 на показ {spikes / ((j % 50) + 1):.1f}, "
                  f"θ {float(theta.mean()):.2f} мВ (макс {float(theta.max()):.2f}), |Δw| {dw:.1%} среднего", flush=True)
            if (j + 1) % 50 == 0:
                spikes = 0.0

    def collect(n: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        idx = torch.arange(len(letters)).repeat(n)
        idx = idx[torch.randperm(len(idx), generator=gen)]
        outs = [show(letters[k], False) for k in idx.tolist()]
        return {key: torch.stack([o[key] for o in outs]) for key in outs[0]}, idx

    r_lab, y_lab = collect(n_label)
    r_test, y_test = collect(n_test)
    print(f"\nметки {n_label} и проверка {n_test} показов каждой буквы, без правила "
          f"(случайно {1 / len(letters):.2f}):")
    def F(r: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Отношение разброса средних по буквам к разбросу внутри буквы, на клетку.
        groups = [r[y == k] for k in range(len(letters))]
        between = torch.stack([g.mean(0) for g in groups]).var(0)
        within = torch.stack([g.var(0) for g in groups]).mean(0).clamp_min(1e-9)
        return between / within

    shuffle = y_lab[torch.randperm(len(y_lab), generator=gen)]
    for key in stages:
        f, f0 = F(r_lab[key], y_lab), F(r_lab[key], shuffle)
        cut = float(torch.quantile(f0, 0.95))
        print(f"  {key:12s} избирательных клеток (F выше 95% перемешанных) {float((f > cut).float().mean()):.0%}", flush=True)
    for key, labels in [(k, y_lab) for k in stages] + [(k + " (перемешано)", shuffle) for k in ("буквы L4", "буквы L2/3")]:
        name, key = key, key.replace(" (перемешано)", "")
        mean = torch.stack([r_lab[key][labels == k].mean(0) for k in range(len(letters))])
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
        print(f"  {name:12s} точность {acc:.2f} (голос клеток), {acc_c:.2f} (корреляция); клеток с ответом {int(active.sum())}, букв с клетками "
              f"{len(set(label[active].tolist()))}/{len(letters)}, спайков на показ {float(r_test[key].sum(1).mean()):.1f}",
              flush=True)
    # Частоты клеток (все показы меток и проверки): фон — пауза после показа,
    # показ — счёт за показ / show. Молчащая — ни одного спайка за все окна;
    # отвечает — парная разность показ − фон по показам, |t| > 2.
    print(f"\nчастоты клеток, Гц ({2 * len(y_lab)} показов; фон — {rest_ms:.0f} мс после показа):")
    for key in stages:
        on = torch.cat([r_lab[key], r_test[key]]) / (show_ms / 1000.0)
        off = torch.cat([r_lab["фон " + key], r_test["фон " + key]])
        d = on - off
        t = d.mean(0) / (d.std(0) / len(d) ** 0.5).clamp_min(1e-9)
        q = lambda x: f"медиана {float(x.mean(0).median()):.2f}, среднее {float(x.mean()):.2f}, молчат {float((x.sum(0) == 0).float().mean()):.0%}"
        print(f"  {key:12s} клеток {on.shape[1]}; фон: {q(off)}; показ: {q(on)}; "
              f"отвечают вверх {float((t > 2).float().mean()):.0%}, вниз {float((t < -2).float().mean()):.0%}", flush=True)
    w = path.weight.float() / w0.float().mean()
    q = torch.quantile(w.flatten()[torch.randperm(w.numel(), generator=gen)[:1_000_000]].to(w.device),
                       torch.tensor([0.01, 0.1, 0.5, 0.9, 0.99], device=w.device))
    print(f"\nвеса пути (доли среднего врождённого): перцентили 1/10/50/90/99 "
          f"{' / '.join(f'{float(x):.2f}' for x in q)}, макс {float(w.max()):.2f}, у потолка "
          f"{float((path.weight >= path.cap * 0.99).float().mean()):.2%}, |Δw| {float((path.weight - w0).abs().mean() / w0.mean()):.1%}")
    # Что выучили клетки у ямки: Δw, собранный по источникам (V1 L2/3 E),
    # корреляция между клетками; клетки у ямки — ближайшие к центру 20.
    src = vis["L23"]["E"]
    assert path.post is L4
    rows = (b.field_cols[L4.name] - b.cfg.fovea_col).abs().argsort()[:20].to(path.weight.device)
    dw = (path.weight - w0).float()[rows]
    pre = path.pre[rows].long() - src.start
    ok = (pre >= 0) & (pre < src.stop - src.start)
    vec = torch.zeros(len(rows), src.stop - src.start, device=dw.device)
    vec.scatter_add_(1, pre.clamp(0, src.stop - src.start - 1), dw * ok)
    vec = vec[:, vec.abs().sum(0) > 0]
    zv = (vec - vec.mean(1, keepdim=True)) / vec.std(1, keepdim=True).clamp_min(1e-12)
    c = zv @ zv.T / vec.shape[1]
    off = c[~torch.eye(len(rows), dtype=torch.bool, device=c.device)]
    print(f"выученное у {len(rows)} клеток букв L4 у ямки: корреляция Δw между клетками медиана {float(off.median()):.2f} "
          f"(10–90%: {float(torch.quantile(off, 0.1)):.2f}…{float(torch.quantile(off, 0.9)):.2f})")
    print(f"всего {time.time() - t0:.0f} с")


if __name__ == "__main__":
    main(dict(kv.split("=", 1) for kv in sys.argv[1:]))
