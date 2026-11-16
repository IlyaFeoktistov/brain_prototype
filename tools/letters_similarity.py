"""
Сходство представлений буквы у ямки и на сдвиге — без учёбы и без правки
весов (DESIGN 2026-11-16, после letters_continuity regul=1: регуляция держит
ямку, переноса на +3 px нет).

Вопрос: есть ли в нынешней анатомии сигнал, который правило следа может
перенести, — похоже ли представление буквы на +3 px на ту же букву у ямки
больше, чем на другие буквы? Мерится на уровнях пути:
    V1 L4 E и V1 L2/3 E в окне ±win px вокруг ямки;
    V1 L2/3 E — только входы 200 клеток букв (то, что видит путь);
    200 клеток букв L4 E у ямки (группы как в letters_us, веса phi60).
Показы: буквы letters= на сдвигах 0, +shift, −shift (±1 px дрожь, как
в letters_continuity), n раз каждой, 350 мс буква + 150 мс пауза.

Меры (признаки — частота клетки за показ):
    у ямки — различимость (ближайший центр по корреляции, выброс одного);
    перенос — центры по показам у ямки, распознаются показы на сдвиге
              (это и есть вопрос: узнаётся ли сдвинутая буква по образцу
              ямки); случай — перемешанные метки, 95%;
    r средних — корреляция средних ответов: та же буква 0 ↔ сдвиг, разные
              буквы 0 ↔ сдвиг, разные буквы обе у ямки; потолок — r двух
              половин показов одного условия у ямки (шум);
    активные — доля общих клеток среди 20% самых частых (Жаккар);
    победители — для клеток букв: группа с наибольшим числом спайков
              (из 10) по местам.

    PYTHONPATH=. .venv/bin/python tools/letters_similarity.py [letters=бгжк] [shift=3] [n=8] [win=12] [seed=1] [load=…]
"""

from __future__ import annotations

import sys
import time

import torch

from brain.session import Session
from tools.reading_bench import BRAIN_HOME, build
from tools.word_read import LETTERS


def corr_rows(x: torch.Tensor) -> torch.Tensor:
    x = x - x.mean(1, keepdim=True)
    x = x / x.norm(dim=1, keepdim=True).clamp_min(1e-9)
    return x @ x.T


def nearest(cents: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Номер центра с наибольшей корреляцией для каждой строки x."""

    return corr_rows(torch.cat([cents, x]))[len(cents):, : len(cents)].argmax(1)


def main(opts: dict) -> None:
    seed = int(opts.get("seed", 1))
    opts = {"fit": "lgn-a", "homeo": f"lgn-d-pop-tcind-s{seed}", "relay_v_t_mv": "-59", "trn_scale": "0.1",
            "us": "visual -> letters", **opts}
    mind, b = build(opts)
    s = Session(mind, window_ms=10.0)
    shown = opts.get("letters", "бгжк")
    ks = [LETTERS.index(c) for c in shown]
    L = len(shown)
    shift, n, win = float(opts.get("shift", 3)), int(opts.get("n", 8)), float(opts.get("win", 12))
    show_ms, rest_ms = 350.0, 150.0
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    dev = b.device

    path = b.us[0]
    P = path.post
    f = opts.get("load", str(BRAIN_HOME / f"letters_us_phi60_s{seed}.pt"))
    path.weight.copy_(torch.load(f, map_location=dev)["weight"])
    near = (b.field_cols[P.name] - b.cfg.fovea_col).abs().argsort()[:20 * len(LETTERS)].to(dev)
    group = (torch.arange(len(near), device=dev) % len(LETTERS)).cpu()
    l4 = b.areas["visual"].layers["L4"]["E"]
    l23 = b.areas["visual"].layers["L23"]["E"]
    pops: dict[str, torch.Tensor] = {}
    for name, pop in (("V1 L4", l4), ("V1 L2/3", l23)):
        cols = b.field_cols[pop.name].to(dev)
        pops[f"{name} ±{win:.0f} px"] = (((cols - b.cfg.fovea_col).abs() <= win).nonzero().flatten() + pop.start).cpu()
    pre = path.pre[near].long().flatten()
    pops["V1 L2/3 входы 200"] = pre[(pre >= l23.start) & (pre < l23.start + l23.size)].unique().cpu()
    pops["буквы 200"] = (near + P.start).cpu()
    print(f"веса {path.name}: {f}; буквы «{shown}», сдвиги 0/±{shift:.0f} px, {n} показов; клеток: "
          + ", ".join(f"{k} {len(v)}" for k, v in pops.items()), flush=True)

    s.line, s.gaze = "", 0.0
    b.run(2500.0, s._blank(2500.0))
    places = [0.0, shift, -shift]
    order = torch.arange(L).repeat(n * len(places))
    where = torch.tensor(places).repeat_interleave(L).repeat(n)
    perm = torch.randperm(len(order), generator=gen)
    order, where = order[perm], where[perm]
    t0 = time.time()
    X = []
    for j, (i, sh) in enumerate(zip(order.tolist(), where.tolist())):
        s.line = shown[i]
        s.gaze = s.screen.width(shown[i]) / 2 + float(torch.randint(-1, 2, (1,), generator=gen)) + sh
        X.append(b.observe(show_ms, s._view(show_ms)).float().cpu())
        s.line = ""
        b.observe(rest_ms, s._blank(rest_ms))
        if j == 3:
            print(f"  первые 4 показа: {time.time() - t0:.0f} с; частота букв 200 {float(torch.stack(X)[:, pops['буквы 200']].mean()):.1f} Гц", flush=True)
    X = torch.stack(X)
    print(f"показы: {time.time() - t0:.0f} с\n", flush=True)

    def means(x: torch.Tensor, sh: float) -> torch.Tensor:
        return torch.stack([x[(where == sh) & (order == q)].mean(0) for q in range(L)])

    for name, idx in pops.items():
        x = X[:, idx]
        y = order
        m0 = where == 0
        # у ямки — выброс одного
        x0, y0 = x[m0], y[m0]
        hits = 0
        for t in range(len(y0)):
            keep = torch.arange(len(y0)) != t
            c = torch.stack([x0[keep & (y0 == q)].mean(0) for q in range(L)])
            hits += int(nearest(c, x0[t:t + 1])[0] == y0[t])
        c0 = means(x, 0.0)
        line = f"{name}: частота {float(x.mean()):.2f} Гц; у ямки {hits / len(y0):.2f}"
        for sh in (shift, -shift):
            ms = where == sh
            pred = nearest(c0, x[ms])
            acc = float((pred == y[ms]).float().mean())
            null = torch.tensor([float((pred == y[ms][torch.randperm(int(ms.sum()), generator=gen)]).float().mean())
                                 for _ in range(200)])
            line += f"; перенос {sh:+.0f} px {acc:.2f} [95% {float(null.quantile(0.95)):.2f}]"
        print(line + f" (случай {1 / L:.2f})", flush=True)
        # корреляции средних
        R0 = corr_rows(c0)
        off = ~torch.eye(L, dtype=torch.bool)
        half = []
        for q in range(L):
            rows = x0[y0 == q]
            half.append(float(corr_rows(torch.stack([rows[0::2].mean(0), rows[1::2].mean(0)]))[0, 1]))
        parts = [f"потолок (половины у ямки) {sum(half) / L:.2f}", f"разные буквы у ямки {float(R0[off].mean()):.2f}"]
        for sh in (shift, -shift):
            cs = means(x, sh)
            C = corr_rows(torch.cat([c0, cs]))[:L, L:]
            parts.append(f"{sh:+.0f} px: та же {float(C.diag().mean()):.2f} / разные {float(C[off].mean()):.2f}")
        print("  r средних: " + "; ".join(parts), flush=True)
        # общие активные клетки
        k = max(1, int(0.2 * x.shape[1]))
        top = lambda v: set(v.topk(k).indices.tolist())  # noqa: E731
        jac = lambda a, c: len(a & c) / len(a | c)  # noqa: E731
        T0 = [top(v) for v in c0]
        parts = [f"разные у ямки {sum(jac(T0[a], T0[c]) for a in range(L) for c in range(L) if a != c) / (L * L - L):.2f}"]
        for sh in (shift, -shift):
            Ts = [top(v) for v in means(x, sh)]
            same = sum(jac(T0[a], Ts[a]) for a in range(L)) / L
            diff = sum(jac(T0[a], Ts[c]) for a in range(L) for c in range(L) if a != c) / (L * L - L)
            parts.append(f"{sh:+.0f} px: та же {same:.2f} / разные {diff:.2f}")
        print(f"  общие среди 20% частых (Жаккар): " + "; ".join(parts), flush=True)
        if name == "буквы 200":
            for sh in places:
                ms = where == sh
                g = torch.stack([x[ms][:, group == q].sum(1) for q in range(len(LETTERS))], 1)
                win10 = g.argmax(1)
                win4 = g[:, ks].argmax(1)
                desc = []
                for q in range(L):
                    w = win10[y[ms] == q]
                    cnt = torch.bincount(w, minlength=len(LETTERS))
                    desc.append(f"{shown[q]}→" + ",".join(f"{LETTERS[c]}{int(cnt[c])}" for c in cnt.argsort(descending=True)[:3] if cnt[c] > 0))
                print(f"  победители {sh:+.0f} px: из 4 верно {float((win4 == y[ms]).float().mean()):.2f}; из 10: " + " ".join(desc), flush=True)
    print(f"всего {time.time() - t0:.0f} с")


if __name__ == "__main__":
    main(dict(kv.split("=", 1) for kv in sys.argv[1:]))
