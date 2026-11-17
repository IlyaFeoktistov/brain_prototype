"""
Адресный вход в карту руки → нажатие одной клавиши (DESIGN 2026-11-16,
«Проводка E → E руки по расстоянию»: при VA без тока и E → E по расстоянию
рука почти молчит — 0.1–0.3 ложных начала в с). Вопрос: проводит ли карта
адресный вход в нужную клавишу и только в неё, и молчит ли после.

КОСТЫЛЬ опыта: «команда» — постоянный ток amp мВ (engine.kick_current,
как bias) клеткам L2/3 E руки в участке ±near карты у блока клавиши. Это
не премоторная команда: путь премоторная → рука — обучаемый путь коры без
топографии по клавишам (anatomy LEARNED_CORTEX), адресность ему ещё
предстоит выучить (лепет руки). Здесь проверяется только выход: участок
L2/3 → L5 блока → Hand.

Мозг: map_ee_sheet=hand, va_bias −76 (как в разборе), пустой экран.
Попытка: пауза rest мс → команда cmd мс на клавишу A → после post мс.
Окна 10 мс; нажатие — условие Hand.pressed, с правилом отпускания (как
Session). По силе amp: попаданий (A нажата за команду), задержка первого
нажатия A, повторов A за команду, чужих нажатий за команду (соседние ±1
клавиши отдельно), нажатий после снятия и в паузах (контроль без команды).

    PYTHONPATH=. .venv/bin/python tools/hand_command.py [seed=1] [amps=2,5,10] [keys=пиржтс] [reps=3] [cmd=300] [rest=500] [post=500] [near=0.005] [layer=L23|L5]
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
    opts = {"fit": "lgn-a", "homeo": f"lgn-d-pop-tcind-s{seed}", "relay_v_t_mv": "-59", "trn_scale": "0.1",
            "map_ee_sheet": "hand", **opts}
    mind, b = build(opts)
    s = Session(mind, window_ms=WIN)
    va_bias = float(opts.get("va_bias", -76))
    amps = [float(x) for x in opts.get("amps", "2,5,10").split(",")]
    reps, near = int(opts.get("reps", 3)), float(opts.get("near", 0.005))
    cmd_ms, rest_ms, post_ms = float(opts.get("cmd", 300)), float(opts.get("rest", 500)), float(opts.get("post", 500))
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    A = b.areas
    if va_bias:
        b.bias[A["thalamus"].E.slice] += va_bias
    E, L5 = A["hand"].layers["L23"]["E"], s.H
    if opts.get("layer", "L23") == "L5":
        # Контроль считывания: ток самому L5 у клавиши (минуя L2/3).
        E = L5
    cE, c5 = b.field_cols[E.name].float().to(b.device), b.field_cols[L5.name].float()
    kb = s.keyboard
    keys, cells = len(kb), s.hand.cells
    targets = [kb.layout.index(c) for c in opts.get("keys", "пиржтс")]
    print(f"va_bias {va_bias:+.0f}, map_ee_sheet {b.cfg.map_ee_sheet}; клавиши "
          + ", ".join(f"«{kb.type(k)}» #{k}" for k in targets) + f"; сила {amps} мВ, {reps} попыток, команда {cmd_ms:.0f} мс", flush=True)

    s.line, s.gaze = "", 0.0
    b.run(2500.0, s._blank(2500.0))
    blank = s._blank(WIN)

    nb = max(1, b.cfg.dendrite_branches)

    def plateau(idx: torch.Tensor) -> float:
        """Доля клеток idx с плато самой горячей ветки > 0.5 (детектор — состояние плато, иначе сигмоида тока)."""

        if b.cfg.dendrite_coincidence:
            pl = b.dend_plateau.view(b.n_slots, nb)[idx].amax(1)
        else:
            pl = torch.sigmoid((b.i_dend.view(b.n_slots, nb)[idx].amax(1) - b.cfg.dendrite_threshold_mv) / b.cfg.dendrite_slope_mv)
        return float((pl > 0.5).float().mean())

    def windows(ms: float) -> tuple[list[int | None], list[float], list[float]]:
        """Нажатая клавиша (или None) по окнам; частота участка L2/3 и доля блока L5 цели — заполняются снаружи; плато участка — в cur["pl"]."""

        out, patch, blk = [], [], []
        cur["pl"] = []
        for _ in range(int(ms / WIN)):
            r = b.observe(WIN, blank)
            out.append(s.hand.pressed(r[L5.slice]))
            patch.append(float(r[cur["patch"]].mean()) if cur["patch"] is not None else 0.0)
            blk.append(float((r[L5.slice][cur["blk"]] > 0).float().mean()) if cur["blk"] is not None else 0.0)
            cur["pl"].append(plateau(cur["patch"]) if cur["patch"] is not None else 0.0)
        return out, patch, blk

    def onsets(seq: list[int | None], held: int | None) -> tuple[list[tuple[int, int]], int | None]:
        ev = []
        for i, k in enumerate(seq):
            if k is not None and k != held:
                ev.append((i, k))
            held = k
        return ev, held

    cur: dict = {"patch": None, "blk": None}
    plan = [(a, k) for a in amps for k in targets for _ in range(reps)]
    perm = torch.randperm(len(plan), generator=gen).tolist()
    plan = [plan[i] for i in perm]
    res: dict = {a: {"hit": 0, "n": 0, "lat": [], "rep": [], "other": 0, "nb": 0, "post": 0, "patch": [], "blk": [], "pl": [], "pl0": []} for a in amps}
    rest_ev, rest_ms_tot = 0, 0.0
    held = None
    t0 = time.time()
    for j, (amp, k) in enumerate(plan):
        pos = float(c5[k * cells: (k + 1) * cells].mean())
        patch = ((cE - pos).abs() <= near).nonzero().flatten()
        cur["patch"], cur["blk"] = (patch + E.start), torch.arange(k * cells, (k + 1) * cells, device=b.device)
        seq, _, _ = windows(rest_ms)
        pl_rest = sum(cur["pl"]) / len(cur["pl"])
        ev, held = onsets(seq, held)
        rest_ev += len(ev)
        rest_ms_tot += rest_ms
        b.kick_current.zero_()
        b.kick_current[patch + E.start] = amp
        b.kick_left.fill_(round(cmd_ms / b.cfg.dt_ms) + 1)
        seq, pr, bl = windows(cmd_ms)
        b.kick_current.zero_()
        b.kick_left.zero_()
        ev, held = onsets(seq, held)
        R = res[amp]
        R["n"] += 1
        mine = [i for i, kk in ev if kk == k]
        R["hit"] += bool(mine)
        if mine:
            R["lat"].append(mine[0] * WIN)
        R["rep"].append(len(mine))
        R["other"] += sum(1 for _, kk in ev if kk != k)
        R["nb"] += sum(1 for _, kk in ev if abs(kk - k) == 1)
        R["patch"].append(sum(pr) / len(pr))
        R["blk"].append(sum(bl) / len(bl))
        R["pl"].append(sum(cur["pl"]) / len(cur["pl"]))
        R["pl0"].append(pl_rest)
        seq, _, _ = windows(post_ms)
        ev, held = onsets(seq, held)
        R["post"] += len(ev)
        if j == 0:
            print(f"  первая попытка: {time.time() - t0:.0f} с", flush=True)
    print(f"попыток {len(plan)}, {time.time() - t0:.0f} с\n", flush=True)
    print(f"пауза (без команды): начал нажатий {rest_ev / (rest_ms_tot / 1000):.2f} в с ({rest_ev} за {rest_ms_tot / 1000:.0f} с)")
    for a in amps:
        R = res[a]
        lat = torch.tensor(R["lat"]) if R["lat"] else torch.tensor([float("nan")])
        print(f"сила {a:4.1f} мВ: попаданий {R['hit']}/{R['n']}; задержка медиана {float(lat.median()):.0f} мс "
              f"[{float(lat.min()):.0f}…{float(lat.max()):.0f}]; нажатий A за команду {sum(R['rep']) / R['n']:.1f}; "
              f"чужих за команду {R['other'] / R['n']:.2f} (соседних ±1 {R['nb'] / R['n']:.2f}); после снятия ({post_ms:.0f} мс) "
              f"{R['post'] / R['n']:.2f}; участок L2/3 {sum(R['patch']) / R['n']:.1f} Гц, блок L5 выстрелил {sum(R['blk']) / R['n']:.0%} окон-клеток; "
              f"плато участка L2/3: команда {sum(R['pl']) / R['n']:.0%}, пауза перед ней {sum(R['pl0']) / R['n']:.0%}",
              flush=True)
    print(f"всего {time.time() - t0:.0f} с")


if __name__ == "__main__":
    main(dict(kv.split("=", 1) for kv in sys.argv[1:]))
