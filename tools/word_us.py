"""
Слово → смысл с учителем на уровне слова (DESIGN 2026-11-15): путь буквы
L2/3 → смысл L4 (letters -> association) учится тем же правилом, что буквы
(Urbanczik & Senn 2014, engine._us); учитель — группа клеток смысла «этого
слова», пока мозг читает слово. Основание: чтению учатся, сопоставляя
написанное с уже знакомым словом (самообучение через перекодирование — Share
1995, https://doi.org/10.1016/0010-0277(94)00645-2); в сетях, обученных на
словах сверху, промежуточные детекторы возникают сами (Hannagan et al. 2021,
https://doi.org/10.1073/pnas.2104779118). Проводка не меняется: правило само
поднимает входы от обученных клеток букв и опускает прочие.

Мозг как в word_read: веса V1 → буквы (load=, по умолчанию
letters_us_phi60_s{seed}.pt) и L4 → L2/3 букв (l23=, по умолчанию
letters_us_l23_60_s{seed}.pt) — после учёбы, заморожены. Слово — фиксации по
буквам слева направо (fix мс, саккада cfg.saccade_ms, пауза rest мс), взгляд
ставит стенд, сдвиг ±1 px.

Группы: per клеток смысла L4 E на слово, случайная перестановка (у смысла нет
карты). Учёба: train показов каждого слова вперемешку; учитель вкл на
фиксациях (группа слова стреляет cfg.teacher_hz, прочие учимые молчат),
правило вкл. Проверка без учителя и правила: test показов каждого слова;
слово = группа с наибольшим числом спайков за всё слово; рядом — ближайший
центр по всем клеткам смысла L4 E (как word_read). Известное ограничение:
по одной букве за фиксацию правило видит набор букв, а не порядок —
перестановки (вид/див/дви) первым шагом не различить. Поэтому набор по
умолчанию (SETS) — 8 слов без перестановок и повторов букв внутри слова,
каждая из 10 выученных букв в 2–3 словах: это проверка распознавания
НАБОРА букв, не порядка (порядок — ступень LCD, направление А).

Контроли: shuffle=1 — учитель на каждом показе у случайной группы (связь
«слово → группа» перемешана); прежняя проводка (без field=) против
field="letters -> association:window:1" (cfg.field_input). Сверх точности:
вклад обученных клеток букв L2/3 у ямки (200 ближайших) в ток учимых
клеток (Σ w·частота пре) и различимость этого тока по словам.

    PYTHONPATH=. .venv/bin/python tools/word_us.py [train=20] [test=8] [seed=1] [per=20] [fix=350] [rest=300]
        [words=…] [load=…] [l23=…] [out=…] [wload=…] [learn=1]
        [shuffle=0] [field=путь:режим:px]
"""

from __future__ import annotations

import sys
import time

import torch

from brain.session import Session
from tools.reading_bench import BRAIN_HOME, build
from tools.word_read import decode

# Наборы букв без перестановок и повторов (см. выше).
SETS = "газ,век,бег,жди,вид,баз,дек,жик"

PATH = "letters -> association"


def main(opts: dict) -> None:
    seed = int(opts.get("seed", 1))
    opts = {"fit": "lgn-a", "homeo": f"lgn-d-pop-tcind-s{seed}", "relay_v_t_mv": "-59", "trn_scale": "0.1",
            "us": PATH, **opts}
    mind, b = build(opts)
    s = Session(mind, window_ms=10.0)
    words = opts.get("words", SETS).split(",")
    W = len(words)
    n_train, n_test = int(opts.get("train", 20)), int(opts.get("test", 8))
    per = int(opts.get("per", 20))
    fix_ms, rest_ms = float(opts.get("fix", 350)), float(opts.get("rest", 300))
    learn = opts.get("learn", "1") == "1"
    shuffle = opts.get("shuffle", "0") == "1"
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    dev = b.device

    for key, name, default in (("load", "visual -> letters", f"letters_us_phi60_s{seed}.pt"),
                               ("l23", "letters: L4.E->E", f"letters_us_l23_60_s{seed}.pt")):
        f = opts.get(key, str(BRAIN_HOME / default))
        next(p for p in b.projections if p.name == name).weight.copy_(torch.load(f, map_location=dev)["weight"])
        print(f"веса {name}: {f} (заморожены)", flush=True)
    path = b.us[0]
    P = path.post
    if opts.get("wload"):
        path.weight.copy_(torch.load(opts["wload"], map_location=dev)["weight"])
    w0 = path.weight.clone()
    src = path.pre_pop
    near = (b.field_cols[src.name].to(dev) - b.cfg.fovea_col).abs().argsort()[:200] + src.start
    from_trained = torch.isin(path.pre.long(), near)
    pick = torch.randperm(P.size, generator=torch.Generator().manual_seed(2000 + seed))[: per * W].to(dev)
    group = torch.arange(len(pick), device=dev) % W
    cells = pick + P.start
    print(f"слова {words}; учится {path.name}: {P.name}, {path.pre.shape[1]} входов на клетку, φ {tuple(round(x, 4) for x in path.us_phi)}; "
          f"учимых {len(cells)} (по {per} на слово); учёба {n_train} показов каждого, учитель и правило {'вкл' if learn else 'ВЫКЛ'}"
          f"{', метки ПЕРЕМЕШАНЫ' if shuffle else ''}; входов от 200 обученных клеток букв у ямки {float(from_trained.float().mean()):.1%}", flush=True)

    b.teach_pop.zero_()
    b.teach_pop[cells] = True
    s.line, s.gaze = "", 0.0
    b.run(2500.0, s._blank(2500.0))

    def show(wi: int, train: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Слово по буквам; спайки учимых клеток, частоты смысла L4 E и источника пути за всё слово."""

        word = words[wi]
        s.line = word
        b.teach_target.zero_()
        tg = int(torch.randint(W, (1,), generator=gen)) if shuffle and train else wi
        b.teach_target[cells[group == tg]] = True
        spikes = torch.zeros(len(cells), device=dev)
        rates = torch.zeros(P.size, device=dev)
        pre_rates = torch.zeros(src.size, device=dev)
        x = 0.0
        for i, ch in enumerate(word):
            w = s.screen.width(ch)
            s.gaze = x + w / 2 + float(torch.randint(-1, 2, (1,), generator=gen))
            x += w
            b.teach_on = b.us_on = train
            r = b.observe(fix_ms, s._view(fix_ms))
            b.teach_on = b.us_on = False
            spikes += r[cells].float() * (fix_ms / 1000.0)
            rates += r[P.slice].float() / len(word)
            pre_rates += r[src.slice].float() / len(word)
            if i < len(word) - 1:
                b.run(b.cfg.saccade_ms, s._blank(b.cfg.saccade_ms))
        s.line = ""
        b.run(rest_ms, s._blank(rest_ms))
        return spikes, rates, pre_rates

    t0 = time.time()
    order = torch.arange(W).repeat(n_train)
    order = order[torch.randperm(len(order), generator=gen)]
    if n_train and learn:
        for j, wi in enumerate(order.tolist()):
            sp = show(wi, True)[0]
            if (j + 1) % 40 == 0:
                dw = float((path.weight - w0).abs().mean() / w0.mean())
                own = float(sp[group == wi].sum()) / max(float(sp.sum()), 1.0)
                print(f"  учёба {j + 1}/{len(order)}, {time.time() - t0:.0f} с: |Δw| {dw:.1%}, доля спайков своей группы {own:.0%}", flush=True)
    if opts.get("out"):
        torch.save({"weight": path.weight.cpu(), "opts": opts}, opts["out"])

    idx = torch.arange(W).repeat(n_test)
    idx = idx[torch.randperm(len(idx), generator=gen)]
    outs = [show(wi, False) for wi in idx.tolist()]
    S = torch.stack([o[0] for o in outs])
    X = torch.stack([o[1] for o in outs]).cpu()
    g = torch.stack([S[:, group == k].sum(1) for k in range(W)], 1)
    y = idx.to(dev)
    acc = float((g.argmax(1) == y).float().mean())
    null = torch.tensor([float((g.argmax(1) == y[torch.randperm(len(y), device=dev)]).float().mean()) for _ in range(100)])
    nc, nc95 = decode(X, idx, gen)
    # Ток учимых клеток от пути: Σ w · частота пре, всего и от обученных клеток букв.
    R = torch.stack([o[2] for o in outs])
    wc, pc, fc = path.weight[pick], path.pre[pick].long() - src.start, from_trained[pick]
    drive = torch.stack([(wc * r[pc]).sum(1) for r in R])
    drive_tr = torch.stack([(wc * r[pc] * fc).sum(1) for r in R])
    share = float(drive_tr.sum() / drive.sum().clamp(min=1e-9))
    dc, dc95 = decode(drive.cpu(), idx, gen)
    dt_, dt95 = decode(drive_tr.cpu(), idx, gen)
    # Несут ли слово сами обученные клетки букв (частоты за слово).
    dl, dl95 = decode(R[:, near - src.start].cpu(), idx, gen)
    print(f"\nпроверка {n_test} показов каждого слова без учителя (случайно {1 / W:.2f}):")
    print(f"  слово по группам смысла {acc:.2f} (перемешано 95% {float(null.quantile(0.95)):.2f}); "
          f"спайков учимых на слово {float(S.sum(1).mean()):.1f}, без спайков {float((g.sum(1) == 0).float().mean()):.0%}")
    print(f"  ближайший центр по всем клеткам смысла L4 E: {nc:.2f} [95% {nc95:.2f}]; |Δw| {float((path.weight - w0).abs().mean() / w0.mean()):.1%}")
    print(f"  ток пути на учимые клетки: от обученных клеток букв {share:.1%}; слово по току: весь {dc:.2f} [95% {dc95:.2f}], "
          f"от обученных {dt_:.2f} [95% {dt95:.2f}]; слово по 200 обученным клеткам букв {dl:.2f} [95% {dl95:.2f}]")
    conf = torch.zeros(W, W, dtype=torch.long)
    for a, p_ in zip(idx.tolist(), g.argmax(1).tolist()):
        conf[a, p_] += 1
    print("  спайки группы на слово, среднее (строка — показанное, столбец — группа):")
    print("        " + " ".join(f"{w:>5}" for w in words))
    for k in range(W):
        print(f"    {words[k]} " + " ".join(f"{float(g[idx.to(dev) == k, j].mean()):5.1f}" for j in range(W)))
    print("  верно по словам: " + " ".join(f"{words[k]} {int(conf[k, k])}/{int(conf[k].sum())}" for k in range(W)))
    print("  путаница (строка — показанное, столбец — ответ):")
    for k in range(W):
        print(f"    {words[k]} " + " ".join(f"{int(conf[k, j]):2d}" for j in range(W)))
    print(f"всего {time.time() - t0:.0f} с")


if __name__ == "__main__":
    main(dict(kv.split("=", 1) for kv in sys.argv[1:]))
