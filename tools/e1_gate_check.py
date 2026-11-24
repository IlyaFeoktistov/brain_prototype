"""
Шаг 4а (docs/CHAT_PLAN.md, docs/PBWM_TRANSFER.md): что открывает ворота E1.
Сеть E1 отдельно (tools/e1_in_brain.E1Net, без мозга, CPU), вход слов — как
от коры со схождением (пуассон: 364/с на слово в тишине, 909/с на своё слово
200 мс — tools/word_drive.py, зерно 1), фраза «БД».

Четыре условия (позиция → стриатум записи и чтения; S — задача записи, окно
0–300 мс каждого такта записи; R — задача чтения, весь вывод):
  1 фон            — без позиции, без S/R;
  2 позиция        — позиция + фон, без S/R;
  3 задача         — S/R + фон, без позиции;
  4 всё            — S/R + позиция + фон.
Меры (Гц на клетку стриатума, все каналы): стриатум записи в окне S /
вне окна в тактах записи / во время вывода; стриатум чтения во время R /
вне R; ответ E1 в тактах записи (ранний ответ) и во время R; фаза такта, на
которой стриатум записи вне S сильнее всего. Картинка — лента по времени
(tools/plots.py) для всех четырёх условий.

Критерий исправления: позиция без S/R не открывает ворота; S — только
запись; R — только чтение; сдвиг позиции сам не вызывает записи.

Условия независимы — каждое своим процессом (по 2 потока), потом сводка:
    for c in 1 2 3 4; do PYTHONPATH=tools:. .venv/bin/python tools/e1_gate_check.py cond=$c & done; wait
    PYTHONPATH=tools:. .venv/bin/python tools/e1_gate_check.py report=1 [out=/tmp/e1b/gate.png]
(одно условие ~5 мин: шаг E1 ~1.7 мс × 150 тыс. шагов)
"""

from __future__ import annotations

import sys

import torch

import plots
from e1_bg import NMSN
from e1_in_brain import LMAX, NANS, S_WIN, T, E1Net
from e1_slot import DT, P

ON, OFF = 909.0, 364.0
TEXT = [1, 4]                         # «БД»: номера слов
BIN_MS = 10.0
CONDS = {"1 фон": (0.0, "off", "off"), "2 позиция": (1.0, "off", "off"),
         "3 задача": (0.0, "task", "task"), "4 всё": (1.0, "task", "task")}


ADV = "burst"   # форма сброса при сдвигах счётчика: burst (по умолчанию E1) | inhib (вариант E1, opt reset_adv)


MSN_MODEL = "lif"   # клетка стриатума: lif (E1) | humphries (Humphries et al. 2009)


def run(pos_to_bg: float, s_mode: str, r_mode: str) -> dict:
    net = E1Net(len(TEXT), s_mode=s_mode, msn=MSN_MODEL)
    if ADV == "inhib":
        net.reset_e_adv = None      # сдвиг — торможение ячеек (как nengo InputGatedMemory), без залпа всех позиций
    net.pos_to_bg, net.r_mode = pos_to_bg, r_mode
    steps_bin = int(round(BIN_MS / DT))
    n_bins = int(net.total // BIN_MS)
    rec = {k: torch.zeros(n_bins) for k in ("msn_w", "msn_r", "ans", "pos", "s", "r")}
    words = [(400.0 + i * T, 400.0 + i * T + 200.0, w) for i, w in enumerate(TEXT)]
    for b in range(n_bins):
        for k in range(steps_bin):
            t = b * BIN_MS + k * DT
            rate = torch.full((1, P), OFF)
            for a, z, w in words:
                if a <= t < z:
                    rate[0, w] = ON
            ans = net.step(t, torch.poisson(rate * 1e-3 * DT))
            rec["msn_w"][b] += net.bg_w.msn.spk.sum()
            rec["msn_r"][b] += net.bg_r.msn.spk.sum()
            rec["ans"][b] += ans.sum()
            rec["pos"][b] += net.cur.spk_e[:, net.src_cur].sum()
            rec["s"][b] += float(net.s_on)
            rec["r"][b] += float(net.task_on)
    hz = 1000.0 / BIN_MS
    out = {"msn_w": rec["msn_w"] / (P * NMSN) * hz, "msn_r": rec["msn_r"] / (P * NMSN) * hz,
           "ans": rec["ans"] / (P * NANS) * hz, "pos": rec["pos"] / net.src_cur.numel() * hz,
           "s": rec["s"] / steps_bin, "r": rec["r"] / steps_bin, "net": net}
    return out


def main(opts: dict) -> None:
    torch.set_num_threads(2)
    seed = int(opts.get("seed", 1))
    global ADV, MSN_MODEL
    ADV = opts.get("adv", "burst")
    MSN_MODEL = opts.get("msn", "lif")
    tag = ("" if ADV == "burst" else f"_{ADV}") + ("" if MSN_MODEL == "lif" else f"_{MSN_MODEL}")
    names = list(CONDS)
    if "cond" in opts:
        name = names[int(opts["cond"]) - 1]
        torch.manual_seed(seed)
        r = run(*CONDS[name])
        net = r.pop("net")
        torch.save({**r, "rec_starts": net.rec_starts}, f"/tmp/e1b/gate{tag}_{opts['cond']}.pt")
        print(f"{name}: готово", flush=True)
        return
    res = {}
    for i, name in enumerate(names):
        r = torch.load(f"/tmp/e1b/gate{tag}_{i + 1}.pt", weights_only=False)
        res[name] = r
        t = torch.arange(len(r["msn_w"])) * BIN_MS
        write_period = torch.zeros(len(t), dtype=torch.bool)
        s_win = torch.zeros(len(t), dtype=torch.bool)
        for i in range(LMAX):
            p0 = 400.0 + i * T
            write_period |= (t >= p0) & (t < p0 + T)
            s_win |= (t >= p0) & (t < p0 + S_WIN)
        read_phase = t >= r["rec_starts"][0]
        r_on = r["r"] > 0.5
        m = lambda x, sel: float(x[sel].mean()) if sel.any() else float("nan")
        out_s = write_period & ~s_win
        # Фаза такта максимума стриатума записи вне окна S.
        ph = (t[out_s] - 400.0) % T
        peak_ph = float(ph[r["msn_w"][out_s].argmax()]) if out_s.any() else float("nan")
        # Пик (Гц, бин 10 мс) и самая длинная серия бинов > 20 Гц (мс) — короткий залп может открыть ворота
        # при безопасном среднем.
        def peak(x, sel):
            if not sel.any():
                return "—"
            v = x.clone()
            v[~sel] = 0.0
            on = (v > 20.0).int().tolist()
            run_, best = 0, 0
            for o in on:
                run_ = run_ + 1 if o else 0
                best = max(best, run_)
            return f"пик {float(v.max()):5.0f} Гц, дольше всего {best * BIN_MS:4.0f} мс > 20 Гц"
        print(f"{name:10s}: ВНЕ окон — стриатум записи вне S {peak(r['msn_w'], ~s_win)}; "
              f"стриатум чтения вне R {peak(r['msn_r'], ~r_on)}; ответ E1 в тактах записи {peak(r['ans'], write_period)}")
        print(f"{name:10s}: стриатум записи — в окне S {m(r['msn_w'], s_win):6.1f}, вне окна в тактах записи "
              f"{m(r['msn_w'], out_s):6.1f} (макс. на фазе {peak_ph:.0f} мс), при выводе {m(r['msn_w'], read_phase):6.1f};"
              f" стриатум чтения — при R {m(r['msn_r'], r_on):6.1f}, вне R {m(r['msn_r'], ~r_on):6.1f};"
              f" ответ E1 — в тактах записи {m(r['ans'], write_period):6.1f}, при R {m(r['ans'], r_on):6.1f} Гц", flush=True)
    any_r = next(iter(res.values()))
    t_ms = (torch.arange(len(any_r["msn_w"])) + 1) * BIN_MS
    panels = [{"label": f"{name}\nГц на клетку", "series": {"стриатум записи": r["msn_w"], "стриатум чтения": r["msn_r"],
                                                         "ответ E1": r["ans"]}, "smooth": 3}
              for name, r in res.items()]
    panels.append({"label": "источники позиции\n(условие 4), Гц", "series": {"позиция": res["4 всё"]["pos"]}, "smooth": 3})
    s_windows = [(400.0 + i * T, 400.0 + i * T + S_WIN, "S") for i in range(LMAX)]
    r_windows = [(rs + 150.0, rs + 400.0 + 4 * T, "") for rs in any_r["rec_starts"]]
    out = opts.get("out", "/tmp/e1b/gate.png")
    plots.timeline(out, t_ms, panels, windows=s_windows, windows_alt=r_windows,
                   title="Ворота E1: кто открывает стриатум записи и чтения (серое — окно S, голубое — R)")
    print(out)


if __name__ == "__main__":
    main(dict(kv.split("=", 1) for kv in sys.argv[1:]))
