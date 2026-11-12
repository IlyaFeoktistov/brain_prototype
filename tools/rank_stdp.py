"""
Обучение по порядку первых спайков — Masquelier & Thorpe 2007
(https://doi.org/10.1371/journal.pcbi.0030031, Methods «S2 cells», «STDP
model»), офлайн, без мозга: проверка, выучивает ли готовый рецепт стыка
«V1 → обучаемая зона» букву из записей V1 (DESIGN 2026-11-12).

Как у авторов (одна позиция — буква в ямке, без размножения клеток по
позициям и масштабам, т. е. без свёртки):
  вход — одна волна: у каждого входа не больше одного спайка, важен лишь
    порядок (задержка ~ 1/сила; функция «сила → задержка» результата не
    меняет — авторы);
  обучаемые клетки — без утечки, вес входа прибавляется к потенциалу по
    приходу его спайка, порог thr · (число входов) (у авторов 64 = ¼ · 16·16);
  «победитель забирает всё»: первая дошедшая до порога клетка — победитель,
    прочим спайк в этой волне запрещён; учится только победитель;
  STDP: вход пришёл не позже спайка победителя — Δw = a⁺ w (1 − w), позже
    или не пришёл — Δw = a⁻ w (1 − w), a⁻ = −¾ a⁺; a⁺ = 2⁻⁶, ×2 каждые
    400 спайков победителей до 2⁻²; начальные веса 0.8 ± 0.05.
Чтение — как у D&C в letters_offline (метки клеткам по отдельным показам,
голос клеток; контроль — 100 перестановок меток) и отдельно ближайший центр
по потенциалам клеток (у авторов — классификатор с учителем, RBF).

Вход:
  pixels=1 — контроль на MNIST: пиксель ярче — спайк раньше, пиксели с
    яркостью 0 молчат (как «нулевая свёртка» у авторов);
  rec=файл.pt — записи V1 (tools/v1_record.py): первый спайк каждой клетки
    V1 L2/3 E за окно win мс после начала буквы; не стрелявшие — молчат.
    Печатает и этап 2: сколько клеток стреляет за 0–50 / 0–win мс, как
    часто первый спайк — фоновый (по паузе перед буквой), и различение
    буквы ближайшим центром по рангу первых спайков и по счёту спайков.

    PYTHONPATH=. .venv/bin/python tools/rank_stdp.py pixels=1 [images=6000] [cells=50] [thr=0.25] [shows=4000] [every=1000]
    PYTHONPATH=. .venv/bin/python tools/rank_stdp.py rec=файл.pt [win=50] [cells=50] [thr=0.25] [shows=4000] [every=1000]
"""

from __future__ import annotations

import gzip
import math
import os
import sys
import time

import torch


def mnist(images: int) -> tuple[torch.Tensor, torch.Tensor]:
    root = os.path.expanduser("~/.brain/mnist")
    with gzip.open(f"{root}/train-images-idx3-ubyte.gz") as f:
        img = torch.frombuffer(bytearray(f.read()[16:]), dtype=torch.uint8).view(-1, 784)[:images].float()
    with gzip.open(f"{root}/train-labels-idx1-ubyte.gz") as f:
        lab = torch.frombuffer(bytearray(f.read()[8:]), dtype=torch.uint8)[:images].long()
    # Задержка: ярче — раньше; 0 — молчит (inf).
    lat = torch.where(img > 0, 1.0 / img, torch.full_like(img, math.inf))
    return lat, lab


def v1(path: str, win: float) -> tuple[torch.Tensor, torch.Tensor, dict]:
    rec = torch.load(path, map_location="cpu")
    n, dt = rec["n_src"], rec["dt_ms"]
    rest, show = rec["rest_ms"], rec["show_ms"]
    lats, counts, bg_first = [], [], []
    for sp in rec["spikes"]:
        t = sp[:, 0].float() * dt - rest  # мс от начала буквы
        c = sp[:, 1].long()
        lat = torch.full((n,), math.inf)
        m = (t >= 0) & (t < win)
        # Первый спайк каждой клетки в окне.
        lat.scatter_reduce_(0, c[m], t[m], reduce="amin")
        lats.append(lat)
        cnt = torch.zeros(n)
        cnt.index_add_(0, c[(t >= 0) & (t < show)], torch.ones(int(((t >= 0) & (t < show)).sum())))
        counts.append(cnt)
        # Фон: спайки в паузе той же длины перед буквой.
        b = (t < 0) & (t >= -min(win, rest))
        bg_first.append(float(torch.unique(c[b]).numel()) / n * (win / min(win, rest)))
    info = {"bg_share": sum(bg_first) / len(bg_first), "counts": torch.stack(counts)}
    return torch.stack(lats), rec["labels"].long(), info


def main(opts: dict) -> None:
    torch.set_num_threads(int(opts.get("threads", 2)))
    seed = int(opts.get("seed", 1))
    gen = torch.Generator().manual_seed(seed)
    if opts.get("pixels"):
        lat, labels = mnist(int(opts.get("images", 6000)))
        info = None
    else:
        lat, labels, info = v1(opts["rec"], float(opts.get("win", 50)))
    n_show, n_in = lat.shape
    L = int(labels.max()) + 1

    # Учёба / метки / проверка — как letters_offline: у каждой буквы 2/3 / 1/6 / 1/6.
    split = {"train": [], "label": [], "test": []}
    for k in range(L):
        idx = (labels == k).nonzero()[:, 0]
        idx = idx[torch.randperm(len(idx), generator=gen)].tolist()
        a, c = len(idx) * 2 // 3, len(idx) * 5 // 6
        split["train"] += idx[:a]
        split["label"] += idx[a:c]
        split["test"] += idx[c:]

    fired = torch.isfinite(lat)
    print(f"входов {n_in}, показов {n_show}, классов {L}; стреляет входов за волну: медиана "
          f"{float(fired.sum(1).float().median()):.0f} ({float(fired.float().mean()):.1%})", flush=True)

    z = lambda a: (a - a.mean(-1, keepdim=True)) / a.std(-1, keepdim=True).clamp_min(1e-9)

    def centroid(feat: torch.Tensor) -> float:
        """Ближайший центр (корреляция) по меткам → проверка."""

        m = torch.stack([feat[split["label"]][labels[split["label"]] == k].mean(0) for k in range(L)])
        pred = (z(feat[split["test"]]) @ z(m).T).argmax(1)
        return float((pred == labels[split["test"]]).float().mean())

    # Этап 2: несёт ли букву порядок первых спайков (ранг: раньше — больше; молчащие — 0).
    rank = torch.where(fired, 1.0 / (1.0 + lat.argsort(1).argsort(1).float()), torch.zeros_like(lat))
    early = fired.float()
    msg = f"различение ближайшим центром: по рангу первых спайков {centroid(rank):.2f}, по «стрелял/нет» {centroid(early):.2f}"
    if info is not None:
        msg += f", по счёту спайков за показ {centroid(info['counts']):.2f}; доля клеток с фоновым спайком за то же окно {info['bg_share']:.1%}"
    print(msg + f"; случайно {1 / L:.2f}", flush=True)

    # Обучаемые клетки (Masquelier & Thorpe 2007).
    cells = int(opts.get("cells", 50))
    thr = float(opts.get("thr", 0.25)) * float(fired.sum(1).float().median())
    W = (0.8 + 0.05 * torch.randn(cells, n_in, generator=gen)).clamp(0.0, 1.0)
    a_plus = 2.0 ** -6
    wins = [0]

    rule = opts.get("rule", "stdp")
    if rule == "hebb":
        # Частотный вариант тех же авторов (уравнения 1–2): ответ клетки — нормированное
        # скалярное произведение Y = W·X / |X|² (X — счёт спайков входов за показ),
        # k-WTA по наибольшему ответу (k = 1), победитель: ΔW = a · Y · (X̂ − W), X̂ = X / |X|.
        X = (info["counts"] if info is not None else torch.where(fired, 1.0 / lat, torch.zeros_like(lat)))
        Xn = X / X.norm(dim=1, keepdim=True).clamp_min(1e-9)
        W = torch.rand(cells, n_in, generator=gen)
        W = W / W.norm(dim=1, keepdim=True)
        a_h = float(opts.get("a", 0.05))

    def wave(i: int, learn: bool) -> tuple[int, torch.Tensor]:
        if rule == "hebb":
            y = W @ Xn[i]
            winner = int(y.argmax())
            if learn:
                W[winner] += a_h * float(y[winner]) * (Xn[i] - W[winner])
                wins[0] += 1
            return winner, y
        """Одна волна: победитель (или −1) и итоговый потенциал клеток (на всех входах волны)."""

        order = lat[i].argsort()
        order = order[torch.isfinite(lat[i, order])]
        pot = W[:, order].cumsum(1)  # потенциал после каждого входа
        cross = (pot >= thr)
        hit = cross.any(1)
        winner = -1
        if hit.any():
            first = torch.where(hit, cross.float().argmax(1), torch.full((cells,), 10 ** 9))
            winner = int(first.argmin())
            if learn:
                before = torch.zeros(n_in, dtype=torch.bool)
                before[order[: int(first[winner]) + 1]] = True
                w = W[winner]
                a = a_plus * (2.0 ** min(4, wins[0] // 400))
                W[winner] = (w + torch.where(before, a, -0.75 * a) * w * (1.0 - w)).clamp(0.0, 1.0)
                wins[0] += 1
        return winner, pot[:, -1] if pot.shape[1] else torch.zeros(cells)

    def measure(n_seen: int) -> None:
        res = [wave(i, False) for i in range(n_show)]
        resp = torch.stack([torch.nn.functional.one_hot(torch.tensor(max(w, 0)), cells).float() * (w >= 0) for w, _ in res])
        pot = torch.stack([p for _, p in res])

        def vote(y_lab: torch.Tensor) -> float:
            r = resp[split["label"]]
            m = torch.stack([r[y_lab == k].mean(0) for k in range(L)])
            lab = m.argmax(0)
            act = m.max(0).values > 0
            t = resp[split["test"]]
            score = torch.stack([t[:, (lab == k) & act].sum(1) if ((lab == k) & act).any()
                                 else torch.zeros(len(t)) for k in range(L)], 1)
            got = score.max(1).values > 0
            return float(((score.argmax(1) == labels[split["test"]]) & got).float().mean())

        y_lab = labels[split["label"]]
        acc = vote(y_lab)
        null = torch.tensor([vote(y_lab[torch.randperm(len(y_lab), generator=gen)]) for _ in range(100)])
        used = int((resp.sum(0) > 0).sum())
        print(f"после {n_seen} волн ({time.time() - t0:.0f} с): голос клеток {acc:.2f} (перемешано×100 среднее "
              f"{float(null.mean()):.2f}, 95% {float(null.quantile(0.95)):.2f}); ближайший центр по потенциалам "
              f"{centroid(pot):.2f}; побеждали {used} из {cells} клеток; без победителя "
              f"{float((resp.sum(1) == 0).float().mean()):.0%}; a⁺ {a_plus * 2 ** min(4, wins[0] // 400):.4f}; "
              f"веса 0/1: {float((W < 0.05).float().mean()):.0%} / {float((W > 0.95).float().mean()):.0%}", flush=True)

    t0 = time.time()
    shows, every = int(opts.get("shows", 4000)), int(opts.get("every", 1000))
    tr = split["train"]
    measure(0)
    for j in range(shows):
        if j % len(tr) == 0:
            order_tr = torch.tensor(tr)[torch.randperm(len(tr), generator=gen)].tolist()
        wave(order_tr[j % len(tr)], True)
        if (j + 1) % every == 0:
            measure(j + 1)


if __name__ == "__main__":
    main(dict(kv.split("=", 1) for kv in sys.argv[1:]))
