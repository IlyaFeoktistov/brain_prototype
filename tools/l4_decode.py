"""
Теряет ли V1 L4 зрительную информацию или её не видит мера сходства
reading_bench (одна пара коротких ответов при 0–1 спайке на клетку).

1. Вспышка: пусто 500 мс, фраза у точки взгляда flash мс, n повторов.
   PSTH окнами 5 мс: LGN зрячие, поле L4, вне строки, V1 L2/3, буквы.
   Начало ответа — первое окно выше фона на 3 sd фона (фон — 100 мс до
   вспышки), пик, ширина на половине высоты, спайков на клетку сверх фона
   в 0–50 / 50–100 / 100–flash мс. Сравнение — с ответом на вспышку у
   макак, не с решётками: начало L4Cα ~44 мс, L4Cβ ~20 мс позже
   (Nowak et al. 1995, https://doi.org/10.1017/S0952523800008282).
2. Различение по популяции: A и B вперемешку по n повторов в режиме
   чтения (фиксации fix мс, саккада cfg.saccade_ms, шаг step знаков).
   Признаки ступени — число спайков каждой клетки на каждой фиксации,
   в раннем окне (early мс от начала фиксации) и за всю фиксацию.
   Классификатор — ближайший центр класса по корреляции, с выбросом
   одного повтора (центры без него); случайный уровень — перемешанные
   метки. Рядом — средняя корреляция пар внутри фразы против между
   фразами по всем парам повторов (та же мера, что в reading_bench,
   но усреднённая — меньше шума).
   Если L4 различает, а L2/3 нет — следующий тест: передача L4 → L2/3.

    PYTHONPATH=. python tools/l4_decode.py [n=12] [flash=300] [fix=225] [step=5] [early=50]
        [seed=1] [fit=lgn-a] [homeo=lgn-a-s1] [relay_v_t_mv=-59] [relay_rest_mv=-48] [trn_scale=0.5]
"""

from __future__ import annotations

import sys

import torch

from brain.session import Session
from tools.reading_bench import build

WIN = 5.0


def main(opts: dict) -> None:
    # Конфигурация LGN A, фон и гомеостаз зерна 1 — как в reading_bench.
    opts = {"fit": "lgn-a", "homeo": f"lgn-a-s{opts.get('seed', 1)}", "relay_v_t_mv": "-59",
            "relay_rest_mv": "-48", "trn_scale": "0.5", **opts}
    mind, b = build(opts)
    cfg = b.cfg
    s = Session(mind, window_ms=WIN)
    n_rep = int(opts.get("n", 12))
    flash, fix = float(opts.get("flash", 300)), float(opts.get("fix", 225))
    step, early = float(opts.get("step", 5)), float(opts.get("early", 50))
    a_text, b_text = opts.get("a", "мама мыла раму"), opts.get("b", "папа пила чай")
    gen = torch.Generator().manual_seed(int(opts.get("seed", 1)))

    lgn = b.areas["lgn"].E
    vis, let = b.areas["visual"].layers, b.areas["letters"].layers
    l4 = vis["L4"]["E"]
    eye_i = [x.name for x in b.sensors].index("eye")
    off = b.sensor_offsets[eye_i]
    driver = (next(p for p in b.projections if p.name == "eye -> " + lgn.name).pre[:, 0].long() - off)
    tc = next(p for p in b.projections if p.name == f"lgn -> {l4.name}")
    tc_pre = tc.pre.long() - lgn.start
    stages = {"LGN": lgn, "V1 L4": l4, "V1 L2/3": vis["L23"]["E"],
              "буквы L4": let["L4"]["E"], "буквы L2/3": let["L23"]["E"]}

    s.line, s.gaze = "", 0.0
    b.run(2500.0)

    def ink(line: str, gaze: float) -> torch.Tensor:
        """Клетки реле, чей водитель видит штрих при этом взгляде."""

        s.line, s.gaze = line, gaze
        seen = s.eye.rates(s._picture(s.screen, cfg.fovea_col)) >= 5.0
        return seen.to(driver.device)[driver]

    def gazes(line: str) -> list[float]:
        width = s.screen.width(line) / max(1, len(line))
        n = max(1, int((len(line) + step - 1) // step))
        return [i * step * width for i in range(n)]

    # ---------- 1. Вспышка ----------
    line = a_text + " "
    seeing = ink(line, 0.0)
    field = seeing[tc_pre].float().mean(1) > 0.5
    outside = (~seeing[tc_pre]).all(1)
    groups = {"LGN зрячие": (lgn, seeing), "L4 поле": (l4, field), "L4 вне поля": (l4, outside),
              "V1 L2/3 E": (vis["L23"]["E"], None), "буквы L4 E": (let["L4"]["E"], None),
              "буквы L2/3 E": (let["L23"]["E"], None)}
    pre_bins, post_bins = int(100 // WIN), int(flash // WIN)
    psth = {k: torch.zeros(pre_bins + post_bins) for k in groups}
    for _ in range(n_rep):
        s.line = ""
        b.run(400.0, s._blank(400.0))
        for t in range(pre_bins + post_bins):
            if t == pre_bins:
                s.line, s.gaze = line, 0.0
            r = b.observe(WIN, s._view(WIN))
            for k, (pop, m) in groups.items():
                x = r[pop.slice] if m is None else r[pop.slice][m]
                psth[k][t] += float(x.mean()) / n_rep
    print(f"\n=== Вспышка «{a_text}», {flash:g} мс, повторов {n_rep}, окна {WIN:g} мс "
          f"(клеток поля L4 {int(field.sum())}, вне поля {int(outside.sum())})")
    print(f"  {'группа':14s} {'фон Гц':>7s} {'начало мс':>9s} {'пик Гц':>7s} {'пик мс':>7s} {'ширина мс':>9s}"
          f" {'сп/кл 0–50':>10s} {'50–100':>7s} {f'100–{flash:g}':>8s}")
    for k, h in psth.items():
        base, resp = h[:pre_bins], h[pre_bins:]
        mu, sd = float(base.mean()), float(base.std())
        above = (resp > mu + 3 * max(sd, 1e-6)).nonzero()
        onset = f"{float(above[0]) * WIN:9.0f}" if len(above) else f"{'—':>9s}"
        pk = int(resp.argmax())
        half = resp > mu + (float(resp[pk]) - mu) / 2
        width = float(half.sum()) * WIN
        extra = lambda a, z: float((resp[int(a // WIN):int(z // WIN)] - mu).sum()) * WIN / 1000.0
        print(f"  {k:14s} {mu:7.1f} {onset} {float(resp[pk]):7.1f} {pk * WIN:7.0f} {width:9.0f}"
              f" {extra(0, 50):10.2f} {extra(50, 100):7.2f} {extra(100, flash):8.2f}")
    print("  PSTH поля L4 (Гц, окна 5 мс, 0 — вспышка): " + " ".join(
        f"{float(v):.1f}" for v in psth["L4 поле"][pre_bins - 4: pre_bins + 40]), flush=True)

    # ---------- 2. Различение по популяции ----------
    order = [0, 1] * n_rep
    perm = torch.randperm(len(order), generator=gen)
    order = [order[i] for i in perm]
    texts = (a_text, b_text)
    feats = {(k, w): [] for k in stages for w in ("ранние", "вся")}
    spikes = {k: 0.0 for k in stages}
    for lab in order:
        s.line = ""
        b.run(500.0, s._blank(500.0))
        line = texts[lab] + " "
        rows = {(k, w): [] for k in stages for w in ("ранние", "вся")}
        for i, g in enumerate(gazes(line)):
            if i:
                s.line, s.gaze = line, g
                b.run(cfg.saccade_ms, s._blank(cfg.saccade_ms))
            s.line, s.gaze = line, g
            acc_e = {k: 0 for k in stages}
            acc = {k: 0 for k in stages}
            for t in range(int(fix // WIN)):
                r = b.observe(WIN, s._view(WIN)) * (WIN / 1000.0)   # спайки за окно
                for k, pop in stages.items():
                    x = r[pop.slice].cpu()
                    acc[k] = acc[k] + x
                    if t < early // WIN:
                        acc_e[k] = acc_e[k] + x
            for k in stages:
                rows[(k, "ранние")].append(acc_e[k])
                rows[(k, "вся")].append(acc[k])
                spikes[k] += float(acc[k].mean())
        for key, v in rows.items():
            feats[key].append(torch.cat(v[:3]))
    labels = torch.tensor(order)
    n_fix = 3 * len(order)

    def corr_rows(x: torch.Tensor) -> torch.Tensor:
        x = x - x.mean(1, keepdim=True)
        x = x / x.norm(dim=1, keepdim=True).clamp_min(1e-9)
        return x @ x.T

    def loo(x: torch.Tensor, y: torch.Tensor) -> float:
        hits = 0
        for i in range(len(y)):
            keep = torch.arange(len(y)) != i
            cents = torch.stack([x[keep & (y == c)].mean(0) for c in (0, 1)])
            sims = corr_rows(torch.cat([x[i:i + 1], cents]))[0, 1:]
            hits += int(int(sims.argmax()) == int(y[i]))
        return hits / len(y)

    print(f"\n=== Различение «{a_text}» против «{b_text}»: по {n_rep} повторов, чтение, "
          f"фиксации {fix:g} мс, ранние — первые {early:g} мс")
    print(f"  {'ступень':12s} {'сп/кл/фикс':>10s} {'окно':>7s} {'точность':>9s} {'случайно':>15s}"
          f" {'внутри':>7s} {'между':>7s}")
    for k in stages:
        for w in ("ранние", "вся"):
            x = torch.stack(feats[(k, w)])
            acc = loo(x, labels)
            shuf = []
            for _ in range(20):
                shuf.append(loo(x, labels[torch.randperm(len(labels), generator=gen)]))
            shuf = torch.tensor(shuf)
            c = corr_rows(x)
            same = labels[:, None] == labels[None, :]
            off_diag = ~torch.eye(len(labels), dtype=torch.bool)
            within, between = float(c[same & off_diag].mean()), float(c[~same].mean())
            print(f"  {k:12s} {spikes[k] / n_fix:10.2f} {w:>7s} {acc:9.2f} "
                  f"{float(shuf.mean()):6.2f} ± {float(shuf.std()):4.2f} {within:7.3f} {between:7.3f}", flush=True)


if __name__ == "__main__":
    main(dict(kv.split("=", 1) for kv in sys.argv[1:]))
