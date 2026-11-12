"""
Выучиваема ли буква из записи V1 L2/3 E без учителя — предел без спайков
(DESIGN «Обучение пути V1 → буквы»): k-средних по счётам клеток за букву
(350 мс), векторы нормированы (z по клеткам), центры — по учебным записям,
метки центрам — по записям меток (большинство), проверка — на отложенных
записях (разбиение как tools/letters_offline.py: 2/3, 1/6, 1/6). Для
сравнения — с учителем (центр буквы по учебным). Плюс доля разброса счёта,
зависящая от буквы (между буквами / всего), по всей V1 и у входов клетки
букв у ямки.

    PYTHONPATH=. .venv/bin/python tools/v1_cluster.py rec=файл.pt [reps=5]
"""

from __future__ import annotations

import sys

import torch


def main(opts: dict) -> None:
    torch.set_num_threads(1)
    rec = torch.load(opts["rec"])
    n, letters = rec["n_src"], rec["letters"]
    L = len(letters)
    nr = int(round(rec["rest_ms"] / rec["dt_ms"]))
    X = torch.stack([torch.bincount(s[s[:, 0] >= nr][:, 1].long(), minlength=n).float() for s in rec["spikes"]])
    B = torch.stack([torch.bincount(s[s[:, 0] < nr][:, 1].long(), minlength=n).float() for s in rec["spikes"]])
    y = rec["labels"]
    tr, lab, te = [], [], []
    for k in range(L):
        idx = (y == k).nonzero()[:, 0].tolist()
        a, c = len(idx) * 2 // 3, len(idx) * 5 // 6
        tr += idx[:a]
        lab += idx[a:c]
        te += idx[c:]

    m = torch.stack([X[y == k].mean(0) for k in range(L)])
    between = m.var(0)
    within = torch.stack([X[y == k].var(0) for k in range(L)]).mean(0)
    pre, field = rec["pre"].long(), rec["field_col"].float()
    inputs = pre[int((field - rec["fovea_col"]).abs().argmin())].unique()
    print(f"счёт V1 L2/3 на клетку за букву {float(X.mean()):.2f}, фон за то же время "
          f"{float(B.mean()) * rec['show_ms'] / rec['rest_ms']:.2f}; доля разброса от буквы: вся V1 "
          f"{float(between.sum() / (between.sum() + within.sum())):.3f}, входы клетки букв у ямки ({len(inputs)}) "
          f"{float(between[inputs].sum() / (between[inputs].sum() + within[inputs].sum())):.3f}")

    z = lambda a: (a - a.mean(1, keepdim=True)) / a.std(1, keepdim=True).clamp_min(1e-9)
    Z = z(X)
    centre = torch.stack([Z[[i for i in tr if y[i] == k]].mean(0) for k in range(L)])
    print(f"с учителем (центр буквы): {float(((Z[te] @ centre.T).argmax(1) == y[te]).float().mean()):.2f}; "
          f"случайно {1 / L:.2f}")
    gen = torch.Generator().manual_seed(int(opts.get("seed", 1)))
    reps = int(opts.get("reps", 5))
    for K in (10, 30, 100):
        accs = []
        for _ in range(reps):
            C = Z[torch.tensor(tr)[torch.randperm(len(tr), generator=gen)[:K]]].clone()
            for _ in range(50):
                a = (Z[tr] @ C.T).argmax(1)
                for c in range(K):
                    if (a == c).any():
                        C[c] = z(Z[tr][a == c].mean(0, keepdim=True))[0]
            al = (Z[lab] @ C.T).argmax(1)
            names = torch.tensor([int(torch.bincount(y[lab][al == c], minlength=L).argmax()) if (al == c).any() else -1
                                  for c in range(K)])
            accs.append(float((names[(Z[te] @ C.T).argmax(1)] == y[te]).float().mean()))
        print(f"k-средних K={K}: точность {sum(accs) / reps:.2f} ({min(accs):.2f}–{max(accs):.2f}, {reps} запусков)")


if __name__ == "__main__":
    main(dict(kv.split("=", 1) for kv in sys.argv[1:]))
