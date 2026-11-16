"""
Перенос узнавания буквы на соседнее место по непрерывности во времени
(DESIGN 2026-11-13 «Временная непрерывность», опыт 2; ROADMAP направление А).

Основание: инвариантность к месту выучивается из того, что видно подряд —
правило следа (Földiák 1991, https://doi.org/10.1162/neco.1991.3.2.194;
VisNet — Wallis & Rolls 1997, https://doi.org/10.1016/S0301-0082(96)00054-8);
на нейронах нижневисочной коры макаки — Li & DiCarlo 2008
(https://doi.org/10.1126/science.1160028): подмена предмета во время саккады
за часы меняет избирательность нейрона в подменённом месте.

Правило следа в форме правила с учителем (Urbanczik & Senn 2014, engine._us):
цель клетки на фиксации после саккады — её собственная частота на прошлой
фиксации (engine.teach_hz), предсказание — φ(V) от нынешнего входа:
ΔW ∝ (след − φ(V)) · x. Учитель-группа (зажим teacher_hz) не нужен: у ямки
клетки уже выучены (letters_us_phi60), их ответ и есть след. КОСТЫЛЬ: след
считает стенд по спайкам прошлой фиксации (как зажим учителя — ISSUES 7);
у клетки это могло бы быть плато NMDA (ROADMAP: удержание разряда).

Опыт (путь V1 → буквы L4, 200 клеток у ямки по 20 на букву из 10, группы
как в letters_us; показываются буквы letters=):
    учёба    train раз каждой: буква у ямки (сдвиг 0, ±1 px) fix мс — без
             правила, частоты клеток → след; саккада cfg.saccade_ms; та же
             буква со сдвигом shift px — правило вкл, цель — след; пауза;
             swap=1 — контроль Li & DiCarlo: на сдвиге показана другая буква
             (пары по порядку: 1↔2, 3↔4, …), цель — след первой;
    trace=wta — конкурентный след (у Földiák выход «победитель берёт всё»):
             цель — группа с наибольшим числом спайков у ямки из всех 10
             (выбирает активность сети, показанную букву стенд не знает)
             стреляет cfg.teacher_hz, прочие молчат; shuffle=1 — контроль:
             цель отдаётся случайной группе вместо победителя (связь «прошлая
             фиксация → нынешняя» разорвана). Выбор группы-победителя
             стендом — КОСТЫЛЬ (как учитель, ISSUES 7);
    regul=1 — регуляция возбудимости 200 клеток у ямки на учёбе (engine.
             homeostasis_update, scope cell; у Földiák 1990 порог клетки
             тянется к заданной частоте: Δt = γ(y − p),
             https://doi.org/10.1007/BF02331346): уставка — собственная
             частота клетки в проверке «до» (уставка своя у клетки — Hengen
             et al. 2016), шаг — раз за попытку (~1 с, как окно develop);
             learn=0 — те же попытки без правила (проверка, что регуляция
             сама не портит ямку);
    проверка до и после учёбы: test показов каждой на сдвигах 0, +shift,
             −shift (необученная сторона); буква = группа с наибольшим числом
             спайков среди показываемых букв.

    PYTHONPATH=. .venv/bin/python tools/letters_continuity.py [letters=бгжк] [shift=3] [train=15] [test=8]
        [seed=1] [swap=0] [trace=rate|wta] [shuffle=0] [regul=0] [learn=1] [fix=350] [rest=150] [load=…] [out=…]
"""

from __future__ import annotations

import sys
import time

import torch

from brain.session import Session
from tools.reading_bench import BRAIN_HOME, build
from tools.word_read import LETTERS


def main(opts: dict) -> None:
    seed = int(opts.get("seed", 1))
    opts = {"fit": "lgn-a", "homeo": f"lgn-d-pop-tcind-s{seed}", "relay_v_t_mv": "-59", "trn_scale": "0.1",
            "us": "visual -> letters", **opts}
    mind, b = build(opts)
    s = Session(mind, window_ms=10.0)
    shown = opts.get("letters", "бгжк")
    ks = [LETTERS.index(c) for c in shown]
    L = len(shown)
    shift = float(opts.get("shift", 3))
    n_train, n_test = int(opts.get("train", 15)), int(opts.get("test", 8))
    swap = opts.get("swap", "0") == "1"
    wta = opts.get("trace", "rate") == "wta"
    shuffle = opts.get("shuffle", "0") == "1"
    homeo = opts.get("regul", "0") == "1"
    learn = opts.get("learn", "1") == "1"
    fix_ms, rest_ms = float(opts.get("fix", 350)), float(opts.get("rest", 150))
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    dev = b.device

    path = b.us[0]
    P = path.post
    f = opts.get("load", str(BRAIN_HOME / f"letters_us_phi60_s{seed}.pt"))
    path.weight.copy_(torch.load(f, map_location=dev)["weight"])
    w0 = path.weight.clone()
    # Группы — как при учёбе у ямки (letters_us: 20 на букву из 10, чередование по расстоянию).
    near = (b.field_cols[P.name] - b.cfg.fovea_col).abs().argsort()[:20 * len(LETTERS)].to(dev)
    group = torch.arange(len(near), device=dev) % len(LETTERS)
    cells = near + P.start
    partner = [i ^ 1 if (i ^ 1) < L else i for i in range(L)]
    print(f"веса {path.name}: {f}; буквы «{shown}», сдвиг {shift:+.0f} px; учёба {n_train} раз каждой"
          f"; след {'конкурентный (победитель ' + str(b.cfg.teacher_hz) + ' Гц)' if wta else 'частота'}{', цель СЛУЧАЙНОЙ группе' if shuffle else ''}"
          f"{', ПОДМЕНА на сдвиге: ' + ' '.join(f'{shown[i]}→{shown[partner[i]]}' for i in range(L)) if swap else ''}", flush=True)

    b.teach_pop.zero_()
    b.teach_pop[cells] = True
    s.line, s.gaze = "", 0.0
    b.run(2500.0, s._blank(2500.0))

    def view(i: int, sh: float, ms: float, target: torch.Tensor | None = None) -> torch.Tensor:
        """Буква shown[i] со сдвигом sh px (> 0 — левее ямки); частоты 200 клеток, Гц."""

        s.line = shown[i]
        s.gaze = s.screen.width(shown[i]) / 2 + float(torch.randint(-1, 2, (1,), generator=gen)) + sh
        if target is not None and learn:
            hz = torch.zeros(b.n_slots, device=dev)
            hz[cells] = target
            b.teach_hz = hz
            b.teach_on = b.us_on = True
        r = b.observe(ms, s._view(ms))
        b.teach_on = b.us_on = False
        b.teach_hz = None
        acc_rates.append((r.float(), ms))
        return r[cells].float()

    acc_rates: list = []

    def rest() -> None:
        s.line = ""
        acc_rates.append((b.observe(rest_ms, s._blank(rest_ms)).float(), rest_ms))

    def window_rates() -> torch.Tensor:
        """Средняя частота за накопленные окна (Гц, все клетки); сбрасывает накопленное."""

        tot = sum(ms for _, ms in acc_rates)
        out = sum(r * ms for r, ms in acc_rates) / tot
        acc_rates.clear()
        return out

    def check(tag: str) -> None:
        for sh in (0.0, shift, -shift):
            idx = torch.arange(L).repeat(n_test)
            idx = idx[torch.randperm(len(idx), generator=gen)]
            R = []
            for i in idx.tolist():
                R.append(view(i, sh, fix_ms))
                rest()
            R = torch.stack(R)
            g = torch.stack([R[:, group == k].sum(1) for k in ks], 1)
            y = idx.to(dev)
            acc = float((g.argmax(1) == y).float().mean())
            null = torch.tensor([float((g.argmax(1) == y[torch.randperm(len(y), device=dev)]).float().mean()) for _ in range(100)])
            conf = torch.zeros(L, L, dtype=torch.long)
            for a, p_ in zip(y.tolist(), g.argmax(1).tolist()):
                conf[a, p_] += 1
            side = "у ямки" if sh == 0 else ("обученная сторона" if sh == shift else "НЕобученная сторона")
            print(f"  {tag} сдвиг {sh:+.0f} px ({side}): буква {acc:.2f} [перемешано 95% {float(null.quantile(0.95)):.2f}; случай {1 / L:.2f}]; "
                  f"частота своей группы {float(torch.stack([R[y == j][:, group == ks[j]].mean() for j in range(L)]).mean()):.1f} Гц; "
                  f"путаница {conf.tolist()}", flush=True)

    t0 = time.time()
    check("до")
    if homeo:
        # Уставка — частота клетки за проверку «до»; регулируются только 200 клеток у ямки.
        setpoint = window_rates()
        b.cfg.homeostasis_scope = "cell"
        b.homeostasis_update(setpoint)
        tgt = torch.full_like(b.homeo_target, float("nan"))
        tgt[cells] = setpoint[cells]
        b.homeo_target = tgt
        b.homeo_rate = torch.where(tgt.isnan(), 0.0, tgt)
        b.bias[: b.n] -= b.homeo_shift
        b.homeo_shift.zero_()
        print(f"  регуляция: уставки 200 клеток {float(setpoint[cells].mean()):.2f} Гц (мин {float(setpoint[cells].min()):.2f}, "
              f"макс {float(setpoint[cells].max()):.2f}); правило {'вкл' if learn else 'ВЫКЛ'}", flush=True)
    acc_rates.clear()
    order = torch.arange(L).repeat(n_train)
    order = order[torch.randperm(len(order), generator=gen)]
    hits = 0
    for j, i in enumerate(order.tolist()):
        trace = view(i, 0.0, fix_ms)
        if wta:
            win = int(torch.stack([trace[group == k].sum() for k in range(len(LETTERS))]).argmax())
            hits += int(win == ks[i])
            if shuffle:
                win = int(torch.randint(len(LETTERS), (1,), generator=gen))
            trace = (group == win).float() * b.cfg.teacher_hz
        s.line = ""
        b.run(b.cfg.saccade_ms, s._blank(b.cfg.saccade_ms))
        view(partner[i] if swap else i, shift, fix_ms, target=trace)
        rest()
        if homeo:
            b.homeostasis_update(window_rates())
        else:
            acc_rates.clear()
        if (j + 1) % 20 == 0:
            dw = float((path.weight[near] - w0[near]).abs().mean() / w0.mean())
            print(f"  учёба {j + 1}/{len(order)}, {time.time() - t0:.0f} с: |Δw| учимых {dw:.1%}; "
                  f"след своей группы {float(trace[group == ks[i]].mean()):.1f} Гц, прочих {float(trace[group != ks[i]].mean()):.1f}"
                  + (f"; сдвиг порогов по группам, мВ: " + " ".join(f"{shown[q]} {float(b.homeo_shift[cells[group == ks[q]]].mean()):+.2f}" for q in range(L))
                     + f", размах клеток {float(b.homeo_shift[cells].min()):+.1f}…{float(b.homeo_shift[cells].max()):+.1f}" if homeo else ""), flush=True)
    if wta and learn:
        print(f"  победитель у ямки = показанная буква: {hits}/{len(order)} (из 10 групп)", flush=True)
    if opts.get("out"):
        torch.save({"weight": path.weight.cpu(), "opts": opts}, opts["out"])
    check("после")
    print(f"всего {time.time() - t0:.0f} с")


if __name__ == "__main__":
    main(dict(kv.split("=", 1) for kv in sys.argv[1:]))
