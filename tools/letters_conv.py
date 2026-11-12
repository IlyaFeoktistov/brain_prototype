"""
Конвергенция V1 L2/3 → зона букв L4 (замер, без вмешательства; мозг как в
letters_dc: LGN A, гомеостаз lgn-d, без пластичности, путь врождённый).
Вопросы: сколько из входов клетки букв активны и избирательны, как они лежат
на поле, теряется ли буква в равновесной сумме входов или её нет и в самом
наборе входов, и что делает сдвиг буквы.

Показы: буквы × сдвиги взгляда (по умолчанию −2…+2 px) × повторы; пауза
rest мс перед буквой, отрезки 0–50 и 50–show мс. По каждому показу — счёт
спайков V1 L2/3 E, букв L4 E и число чернильных пикселей.

Печатает:
  1. геометрия: у клеток букв у ямки — окно пути (столбцы, строки), разных
     источников, доля клеток V1 L2/3 в полосе, взятая клеткой; совпадение
     наборов входов у двух клеток на одном месте;
  2. по отрезкам: активных входов у клетки (источник растёт на показе
     против паузы, |t| > 2 по показам), избирательных (F по буквам) —
     против перемешанных меток;
  3. различение (корреляция, как letters_dc; метки — первая половина
     повторов, проверка — вторая, внутри своего сдвига):
       V1 L2/3 все;
       набор входов одной клетки букв у ямки (вектор счётов её источников —
       что могла бы вычитать любая настройка весов; среднее по cells
       клеткам);
       сумма входов (Σ вес × счёт) по всем клеткам букв и по клеткам у ямки;
       спайки букв L4 E;
  4. сумма у клеток у ямки: корреляция по показам между клетками одного
     места (сумма у всех одна и та же?) и с числом чернил;
  5. сдвиг: перенос с центра (сдвиг 0) на прочие — метки на 0, проверка
     на сдвиге s; и сходство «та же буква, другой сдвиг» против «другая
     буква, тот же сдвиг».

    PYTHONPATH=. .venv/bin/python tools/letters_conv.py [letters=абвгдежзик] [shifts=-2,-1,0,1,2] [reps=4] [cells=20] [seed=1]
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
    shifts = [int(x) for x in opts.get("shifts", "-2,-1,0,1,2").split(",")]
    reps = int(opts.get("reps", 4))
    n_cells = int(opts.get("cells", 20))
    show_ms, rest_ms, early_ms = float(opts.get("show", 350)), float(opts.get("rest", 150)), 50.0
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    src = b.areas["visual"].layers["L23"]["E"]
    post = b.areas["letters"].layers["L4"]["E"]
    path = next(p for p in b.projections if p.post is post and p.source == "net"
                and int(p.pre.min()) >= src.start and int(p.pre.max()) < src.start + src.size)
    pre = (path.pre.long() - src.start).cpu()
    w = path.weight.float().cpu()
    centre = b.field_cols[post.name].cpu()
    v1_pos = b.map_pos[src.name].cpu()
    fovea = cfg.fovea_col
    near = (centre - fovea).abs().argsort()[: n_cells]
    # Матрица весов клетка букв × клетка V1 L2/3 (повторы входов складываются).
    W = torch.zeros(post.size, src.size)
    W.index_put_((torch.arange(post.size)[:, None].expand_as(pre), pre), w, accumulate=True)

    # 1. Геометрия.
    rows = torch.stack([v1_pos[pre[c], 0] for c in near])
    cols = torch.stack([v1_pos[pre[c], 1] for c in near])
    uniq = torch.tensor([len(torch.unique(pre[c])) for c in near]).float()
    in_band = torch.tensor([int(((v1_pos[:, 1] - centre[c]).abs() <= 6.0).sum()) for c in near]).float()
    a, c2 = near[0], near[1]
    sa, sb = set(pre[a].tolist()), set(pre[c2].tolist())
    print(f"путь {path.name}: {pre.shape[1]} входов, вес {float(w.mean()):.3f} мВ; клеток букв {post.size}, "
          f"мест по полю {len(torch.unique(centre.round(decimals=2)))}, на одно место ~{post.size / s.screen.cols:.0f} клеток")
    print(f"клетки у ямки ({n_cells}, место {float(centre[near].min()):.1f}–{float(centre[near].max()):.1f}): "
          f"столбцы входов {float(cols.min()):.1f}–{float(cols.max()):.1f} (размах у клетки {float((cols.max(1).values - cols.min(1).values).mean()):.1f}), "
          f"строки {float(rows.min()):.0f}–{float(rows.max()):.0f}; разных источников {float(uniq.mean()):.0f} из {float(in_band.mean()):.0f} "
          f"клеток V1 L2/3 в полосе ({float((uniq / in_band).mean()):.0%}); общих источников у двух соседних "
          f"{len(sa & sb) / len(sa | sb):.0%} (Жаккар)", flush=True)

    def segment(view, n: int) -> torch.Tensor:
        acc = None
        for _ in range(n):
            r = b.observe(WIN, view(WIN)) * (WIN / 1000.0)
            acc = r if acc is None else acc + r
        return acc

    def show(ch: str, shift: int) -> dict:
        s.line = ""
        out = {"пауза": segment(s._blank, int(rest_ms // WIN))}
        s.line, s.gaze = ch, s.screen.width(ch) / 2 + shift
        ink = float((s._picture(s.screen, fovea) < 0.5).sum())
        out["0–50"] = segment(s._view, int(early_ms // WIN))
        out["50–350"] = segment(s._view, int((show_ms - early_ms) // WIN))
        s.line = ""
        return {k: (v[src.slice].cpu(), v[post.slice].cpu()) for k, v in out.items()} | {"ink": ink}

    s.line, s.gaze = "", 0.0
    b.run(2500.0, s._blank(2500.0))
    t0 = time.time()
    shows, lab_letter, lab_shift, lab_rep = [], [], [], []
    for rep in range(reps):
        order = [(k, sh) for k in range(len(letters)) for sh in shifts]
        for j in torch.randperm(len(order), generator=gen).tolist():
            k, sh = order[j]
            shows.append(show(letters[k], sh))
            lab_letter.append(k)
            lab_shift.append(sh)
            lab_rep.append(rep)
        print(f"  повтор {rep + 1}/{reps}, {time.time() - t0:.0f} с", flush=True)

    y, sh_, rp = torch.tensor(lab_letter), torch.tensor(lab_shift), torch.tensor(lab_rep)
    ink = torch.tensor([x["ink"] for x in shows])
    z = lambda x: (x - x.mean(-1, keepdim=True)) / x.std(-1, keepdim=True).clamp_min(1e-9)
    L = len(letters)

    def acc(X: torch.Tensor, train: torch.Tensor, test: torch.Tensor) -> float:
        keep = X.std(0) > 0
        X = X[:, keep]
        if X.shape[1] < 2:
            return float("nan")
        mean = torch.stack([X[train & (y == k)].mean(0) for k in range(L)])
        return float(((z(X[test]) @ z(mean).T).argmax(1) == y[test]).float().mean())

    def within(X: torch.Tensor) -> float:
        # Метки — первая половина повторов, проверка — вторая, внутри каждого сдвига.
        res = [acc(X, (sh_ == v) & (rp < reps // 2), (sh_ == v) & (rp >= reps // 2)) for v in shifts]
        return sum(res) / len(res)

    for part in ("0–50", "50–350"):
        V = torch.stack([x[part][0] for x in shows])          # [показ, V1 L2/3]
        P = torch.stack([x["пауза"][0] for x in shows])
        S = torch.stack([x[part][1] for x in shows])          # [показ, буквы L4]
        scale = (show_ms - early_ms) / rest_ms if part == "50–350" else early_ms / rest_ms
        # 2. Активные и избирательные входы.
        d = V - P * scale
        t = d.mean(0) / (d.std(0) / len(d) ** 0.5).clamp_min(1e-9)
        means = torch.stack([V[y == k].mean(0) for k in range(L)])
        within_var = torch.stack([V[y == k].var(0) for k in range(L)]).mean(0)
        sel = means.var(0) > 2.0 * within_var / (len(shifts) * reps)
        yp = y[torch.randperm(len(y), generator=gen)]
        means0 = torch.stack([V[yp == k].mean(0) for k in range(L)])
        within0 = torch.stack([V[yp == k].var(0) for k in range(L)]).mean(0)
        sel0 = means0.var(0) > 2.0 * within0 / (len(shifts) * reps)
        act = torch.stack([(t[pre[c]] > 2).float().sum() for c in near]).mean()
        sl = torch.stack([sel[pre[c]].float().sum() for c in near]).mean()
        sl0 = torch.stack([sel0[pre[c]].float().sum() for c in near]).mean()
        print(f"\n=== {part} мс")
        print(f"  у клетки букв у ямки из {pre.shape[1]} входов: активных {float(act):.1f}, избирательных {float(sl):.1f} "
              f"(перемешано {float(sl0):.1f})", flush=True)
        # 3. Различение.
        summed = V @ W.T                                       # [показ, клетка букв]: Σ вес × счёт
        sets = [within(V[:, torch.unique(pre[c])]) for c in near]
        print(f"  различение (случайно {1 / L:.2f}), внутри сдвига:")
        print(f"    V1 L2/3 все                                {within(V):.2f}")
        print(f"    набор входов одной клетки у ямки           {sum(sets) / len(sets):.2f} "
              f"(клетки {min(sets):.2f}–{max(sets):.2f})")
        print(f"    сумма входов, все клетки букв              {within(summed):.2f}")
        print(f"    сумма входов, клетки у ямки ({n_cells})         {within(summed[:, near]):.2f}")
        print(f"    спайки букв L4 E                           {within(S):.2f}", flush=True)
        # 4. Сумма у клеток у ямки: одна ли она у всех, и чернила.
        F = summed[:, near]
        Fz = (F - F.mean(0)) / F.std(0).clamp_min(1e-9)
        cc = (Fz.T @ Fz) / (len(F) - 1)
        off = cc[~torch.eye(len(near), dtype=torch.bool)]
        ink_r = float(torch.corrcoef(torch.stack([F.mean(1), ink]))[0, 1])
        print(f"  сумма у клеток у ямки: корреляция между клетками по показам {float(off.mean()):.2f} "
              f"(5–95% {float(off.quantile(0.05)):.2f}–{float(off.quantile(0.95)):.2f}); "
              f"со числом чернил {ink_r:+.2f}", flush=True)
        # 5. Сдвиг.
        line = "  перенос с центра (метки — сдвиг 0, все повторы):"
        for v in shifts:
            if v == 0:
                continue
            train, test = sh_ == 0, sh_ == v
            line += f" {v:+d}: V1 {acc(V, train, test):.2f} / сумма {acc(summed, train, test):.2f};"
        print(line, flush=True)
        M = torch.stack([V[(y == k) & (sh_ == v)].mean(0) for k in range(L) for v in shifts])
        Mz = z(M)
        sim = Mz @ Mz.T / M.shape[1]
        kk = torch.tensor([k for k in range(L) for _ in shifts])
        vv = torch.tensor([v for _ in range(L) for v in shifts])
        same_letter = (kk[:, None] == kk[None, :]) & (vv[:, None] != vv[None, :])
        same_shift = (kk[:, None] != kk[None, :]) & (vv[:, None] == vv[None, :])
        print(f"  V1 L2/3, сходство средних ответов: та же буква, другой сдвиг {float(sim[same_letter].mean()):.2f}; "
              f"другая буква, тот же сдвиг {float(sim[same_shift].mean()):.2f}", flush=True)
    print(f"всего {time.time() - t0:.0f} с")


if __name__ == "__main__":
    main(dict(kv.split("=", 1) for kv in sys.argv[1:]))
