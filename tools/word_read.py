"""
Доходит ли слово до ассоциативной коры — измерение без правок мозга (ROADMAP,
звено 1; DESIGN 2026-11-13 «От букв к слову»).

Мозг как в letters_us (гомеостаз зерна, путь глаз → LGN → V1 → буквы), веса
пути V1 → буквы — после учёбы с учителем 60 показов (load=, по умолчанию
~/.brain/letters_us_phi60_s{seed}.pt). Учитель, правило с учителем и вся
пластичность выключены — только чтение.

Показ слова — фиксации по буквам слева направо, как при чтении: взгляд на
центре буквы fix мс (по умолчанию 350 — как при учёбе букв), между буквами
саккада (cfg.saccade_ms пустого входа), после слова пауза rest мс. Взгляд
ставит стенд, не двухолмие (измеряется путь, не выбор саккады). Сдвиг
взгляда ±1 px — как при учёбе.

Слова (words=, по умолчанию): разные буквы, перестановки тех же букв,
различие в одну букву:
    вид див дви — перестановки; жди бди зди — одна буква (первая);
    кжб бжк — перестановка.

Ступени (E клетки, частота за окно): группы учимых клеток букв (10 средних
частот — код слова в выученных буквах); зона букв L4 (группы учимых клеток —
какую букву «видит» каждая фиксация), зона букв L2/3, L5; ассоциативная кора
L4, L2/3, L5, L6. Окна: каждая фиксация отдельно и всё слово.

Различение слова: ближайший центр (корреляция) с исключением одного показа;
контроль — перемешанные метки ×100 (95%). Отдельно — только внутри
наборов перестановок (порядок букв при том же наборе) и внутри набора «одна
буква».

    PYTHONPATH=. .venv/bin/python tools/word_read.py [seed=1] [reps=8] [fix=350] [rest=300] [words=…] [load=…]
"""

from __future__ import annotations

import sys
import time

import torch

from brain.session import Session
from tools.reading_bench import BRAIN_HOME, build

LETTERS = "абвгдежзик"
WORDS = "вид,див,дви,жди,бди,зди,кжб,бжк"


def nearest_centroid(X: torch.Tensor, y: torch.Tensor) -> float:
    """Доля верных по ближайшему центру (корреляция), без показа-себя."""

    Z = X - X.mean(1, keepdim=True)
    Z = Z / Z.norm(dim=1, keepdim=True).clamp_min(1e-9)
    n, k = len(y), int(y.max()) + 1
    ok = 0
    for i in range(n):
        keep = torch.ones(n, dtype=torch.bool)
        keep[i] = False
        cents = torch.stack([X[keep & (y == c)].mean(0) for c in range(k)])
        cents = cents - cents.mean(1, keepdim=True)
        cents = cents / cents.norm(dim=1, keepdim=True).clamp_min(1e-9)
        ok += int(int((cents @ Z[i]).argmax()) == int(y[i]))
    return ok / n


def decode(X: torch.Tensor, y: torch.Tensor, gen: torch.Generator, perms: int = 100) -> tuple[float, float]:
    """Точность и 95% перемешанных меток."""

    acc = nearest_centroid(X, y)
    null = torch.tensor([nearest_centroid(X, y[torch.randperm(len(y), generator=gen)]) for _ in range(perms)])
    return acc, float(null.quantile(0.95))


def main(opts: dict) -> None:
    seed = int(opts.get("seed", 1))
    load = opts.get("load", str(BRAIN_HOME / f"letters_us_phi60_s{seed}.pt"))
    opts = {"fit": "lgn-a", "homeo": f"lgn-d-pop-tcind-s{seed}", "relay_v_t_mv": "-59", "trn_scale": "0.1",
            "us": "visual -> letters", **opts}
    mind, b = build(opts)
    s = Session(mind, window_ms=10.0)
    words = opts.get("words", WORDS).split(",")
    reps, fix_ms, rest_ms = int(opts.get("reps", 8)), float(opts.get("fix", 350)), float(opts.get("rest", 300))
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    dev = b.device

    path = b.us[0]
    path.weight.copy_(torch.load(load, map_location=dev)["weight"])
    L4 = path.post
    L = len(LETTERS)
    near = (b.field_cols[L4.name] - b.cfg.fovea_col).abs().argsort()[:20 * L].to(dev)
    group = torch.arange(len(near), device=dev) % L
    cells = near + L4.start

    A = b.areas
    stages = {"буквы L4 E": A["letters"].layers["L4"]["E"], "буквы L2/3 E": A["letters"].layers["L23"]["E"],
              "буквы L5 E": A["letters"].layers["L5"]["E"]}
    for layer in ("L4", "L23", "L5", "L6"):
        stages[f"смысл {layer.replace('L23', 'L2/3')} E"] = A["association"].layers[layer]["E"]
    print(f"веса пути {path.name}: {load}", flush=True)
    print("пути из зоны букв в ассоциативную: " + ", ".join(
        p.name for p in b.projections if "letters" in p.name.split("->")[0] and "association" in p.name.split("->")[-1]), flush=True)
    print(f"слова {words}, по {reps} показов; фиксация {fix_ms:.0f} мс, саккада {b.cfg.saccade_ms:.0f} мс, пауза {rest_ms:.0f} мс", flush=True)

    s.line, s.gaze = "", 0.0
    b.run(2500.0, s._blank(2500.0))
    blank_rates = {k: [] for k in stages}

    def show(word: str) -> tuple[list[dict], torch.Tensor]:
        """Фиксации по буквам: частоты ступеней по фиксациям и группы букв L4."""

        s.line = word
        per_fix, groups = [], []
        x = 0.0
        for i, ch in enumerate(word):
            w = s.screen.width(ch)
            s.gaze = x + w / 2 + float(torch.randint(-1, 2, (1,), generator=gen))
            x += w
            r = b.observe(fix_ms, s._view(fix_ms))
            groups.append(torch.stack([r[cells[group == g]].float().mean() for g in range(L)]))
            per_fix.append({"буквы, группы": groups[-1], **{k: r[v.slice].float() for k, v in stages.items()}})
            if i < len(word) - 1:
                b.run(b.cfg.saccade_ms, s._blank(b.cfg.saccade_ms))
        s.line = ""
        rb = b.observe(rest_ms, s._blank(rest_ms))
        for k, v in stages.items():
            blank_rates[k].append(float(rb[v.slice].float().mean()))
        return per_fix, torch.stack(groups)

    t0 = time.time()
    order = torch.arange(len(words)).repeat(reps)
    order = order[torch.randperm(len(order), generator=gen)]
    shows = []
    for j, w in enumerate(order.tolist()):
        shows.append(show(words[w]))
        if (j + 1) % 16 == 0:
            print(f"  показ {j + 1}/{len(order)}, {time.time() - t0:.0f} с", flush=True)
    y = order

    # Какую букву видит зона букв на каждой фиксации (группа с наибольшей частотой).
    n_fix = len(words[0])
    seen_ok = torch.zeros(n_fix)
    for (pf, G), wi in zip(shows, y.tolist()):
        for i in range(n_fix):
            seen_ok[i] += int(int(G[i].argmax()) == LETTERS.index(words[wi][i]))
    print(f"\nзона букв L4, группа-победитель = буква фиксации (случайно {1 / L:.2f}): "
          + " ".join(f"фиксация {i + 1} {float(seen_ok[i]) / len(y):.2f}" for i in range(n_fix)))
    conf = {}
    for (pf, G), wi in zip(shows, y.tolist()):
        for i in range(n_fix):
            key = (words[wi][i], LETTERS[int(G[i].argmax())])
            conf[key] = conf.get(key, 0) + 1
    per_letter = {}
    for (true, got), c in conf.items():
        tot, ok = per_letter.get(true, (0, 0))
        per_letter[true] = (tot + c, ok + (c if true == got else 0))
    print("  верно по буквам: " + " ".join(f"{k} {ok}/{tot}" for k, (tot, ok) in sorted(per_letter.items())))

    sets = {"все слова": list(range(len(words)))}
    groups_by_letters: dict[str, list[int]] = {}
    for i, w in enumerate(words):
        groups_by_letters.setdefault("".join(sorted(w)), []).append(i)
    perm = [i for g in groups_by_letters.values() if len(g) > 1 for i in g]
    if perm:
        sets["перестановки (внутри набора)"] = perm
    tails: dict[str, list[int]] = {}
    for i, w in enumerate(words):
        tails.setdefault(w[1:], []).append(i)
    one = [i for g in tails.values() if len(g) > 1 for i in g]
    if one:
        sets["одна буква"] = one

    print(f"\nразличение слов (ближайший центр, без показа-себя; [95% перемешанных]); случайно — 1/число слов набора:")
    print(f"  {'ступень':<16} {'окно':<12} " + "  ".join(f"{name:<28}" for name in sets))
    for k in ["буквы, группы", *stages]:
        for win in ["всё слово"] + [f"фиксация {i + 1}" for i in range(n_fix)]:
            if win == "всё слово":
                X = torch.stack([torch.stack([f[k] for f in pf]).mean(0) for pf, _ in shows]).cpu()
            else:
                i = int(win.split()[-1]) - 1
                X = torch.stack([pf[i][k] for pf, _ in shows]).cpu()
            cols = []
            for name, idx in sets.items():
                if name == "перестановки (внутри набора)":
                    # Точность внутри каждого набора перестановок отдельно, средняя по показам.
                    tot, hit, nul = 0, 0.0, 0.0
                    for g in groups_by_letters.values():
                        if len(g) < 2:
                            continue
                        m = torch.isin(y, torch.tensor(g))
                        yy = torch.tensor([g.index(int(v)) for v in y[m]])
                        a, q = decode(X[m], yy, gen)
                        hit += a * int(m.sum()); nul += q * int(m.sum()); tot += int(m.sum())
                    cols.append(f"{hit / tot:.2f} [{nul / tot:.2f}]")
                    continue
                m = torch.isin(y, torch.tensor(idx))
                yy = torch.tensor([idx.index(int(v)) for v in y[m]])
                if name == "одна буква":
                    tot, hit, nul = 0, 0.0, 0.0
                    for g in tails.values():
                        if len(g) < 2:
                            continue
                        mm = torch.isin(y, torch.tensor(g))
                        y2 = torch.tensor([g.index(int(v)) for v in y[mm]])
                        a, q = decode(X[mm], y2, gen)
                        hit += a * int(mm.sum()); nul += q * int(mm.sum()); tot += int(mm.sum())
                    cols.append(f"{hit / tot:.2f} [{nul / tot:.2f}]")
                    continue
                a, q = decode(X[m], yy, gen)
                cols.append(f"{a:.2f} [{q:.2f}]")
            rate = float(X.mean())
            print(f"  {k:<16} {win:<12} " + "  ".join(f"{c:<28}" for c in cols) + f" ({rate:.2f} Гц)")
    print("\nфон на паузе после слова, Гц: " + ", ".join(f"{k} {sum(v) / len(v):.2f}" for k, v in blank_rates.items()))
    print(f"всего {time.time() - t0:.0f} с")


if __name__ == "__main__":
    main(dict(kv.split("=", 1) for kv in sys.argv[1:]))
