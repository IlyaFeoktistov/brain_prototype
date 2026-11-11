"""
Из чего ответ реле LGN мозга на штрих буквы (замер, без вмешательства;
DESIGN, итог блока Chariker: реле отвечают на штрих слабо, показ / фон
~1.15). Мозг как в letters_dc (reading_bench.build: LGN A, фон, гомеостаз
зерна, trn_scale 0.5), пластичность выкл. Буквы по n показов: пусто rest
мс, буква show мс по центру ямки (как letters_dc и lgn_record).

Реле — по водителю (путь eye -> lgn): «штрих» — частота его клетки
сетчатки на показе ≥ 0.3 · retina_rate_hz, «рядом» — больше 0, «пусто» — 0
(своя активность сетчатки — у всех). По группам, окна: пауза (последние 200 мс), показ 0–50
и 50–show мс:
  - вход каждого пути в реле, мВ: доля пути в Σ вес × частота источника
    (× ресурс депрессии у путей с ней) × ток сомы своего знака
    (возбуждение i_syn × движущая сила, торможение i_inh × движущая сила);
  - среднее поле (возб. / торм.), фон bias, T-ток, адаптация, мембрана,
    частота реле, частота источников.
Окно 10 мс, токи — на конце окна (среднее по многим окнам и клеткам).

    PYTHONPATH=. .venv/bin/python tools/lgn_sources.py [letters=абвгдежзик] [n=3] [rest=300] [show=350] [seed=1]
"""

from __future__ import annotations

import sys

import torch

from brain.session import Session
from tools.reading_bench import build, observe_pre

WIN = 10.0


def main(opts: dict) -> None:
    seed = int(opts.get("seed", 1))
    opts = {"fit": "lgn-a", "homeo": f"lgn-d-s{seed}", "relay_v_t_mv": "-59",
            "trn_scale": "0.1", **opts}
    mind, b = build(opts)
    cfg = b.cfg
    s = Session(mind, window_ms=WIN)
    letters = opts.get("letters", "абвгдежзик")
    n_rep = int(opts.get("n", 3))
    rest, show = float(opts.get("rest", 300)), float(opts.get("show", 350))
    gen = torch.Generator().manual_seed(seed + 1000)
    torch.manual_seed(seed)
    lgn = b.areas["lgn"].E
    eye_i = [x.name for x in b.sensors].index("eye")
    off = b.sensor_offsets[eye_i]
    paths = [p for p in b.projections if p.post is lgn]
    eye_p = next(p for p in paths if p.name == "eye -> " + lgn.name)
    driver = eye_p.pre[:, 0].long() - off
    stp_paths = {p.name for p in paths if p.source == "net" and bool((b.stp_u_cell[p.pre.long()] > 0).any())}
    print("входы реле LGN:")
    for p in paths:
        w = p.weight.float()
        print(f"  {p.name}: {p.source}, {w.shape[1]} входов на клетку, вес {float(w.mean()):+.3f} мВ"
              f"{', депрессия' if p.name in stp_paths else ''}", flush=True)
    q = int(b.mf_post_id[lgn.start])
    src_pops = {}
    for p in paths:
        if p.source == "net":
            pre = p.pre.long()
            src_pops[p.name] = (int(pre.min()), int(pre.max()) + 1)

    names = [p.name for p in paths]
    keys = names + ["среднее поле возб.", "среднее поле торм.", "фон bias", "T-ток", "адаптация",
                    "мембрана v, мВ", "частота реле, Гц"] + [f"частота источника «{n}», Гц" for n in names]
    groups = ("штрих", "рядом", "пусто")

    def sample(r: torch.Tensor, st: dict, rec_rate: torch.Tensor, masks: dict) -> dict:
        v = b.v[lgn.slice]
        excite = ((cfg.ampa_reversal_mv - v) / (cfg.ampa_reversal_mv - cfg.epsp_holding_mv)).clamp(min=0.0)
        force = ((v - cfg.gaba_reversal_mv) / (cfg.ipsp_holding_mv - cfg.gaba_reversal_mv)).clamp(min=0.0)
        e_cur = b.i_syn[lgn.slice] * excite
        i_cur = (b.i_inh[lgn.slice] + b.i_ngf[lgn.slice]) * force
        drives, src_hz = [], []
        for p in paths:
            pre = p.pre.long()
            if p.source == "sensory":
                src = rec_rate[pre - off]
            else:
                src = r[pre]
                if p.name in stp_paths:
                    src = src * st["x"][pre]
            drives.append((p.weight.float() * src).sum(1))
            src_hz.append(src.mean(1) if p.source == "net" else rec_rate[pre - off].mean(1))
        drives = torch.stack(drives)                                  # [путей, клеток]
        pos = drives.clamp(min=0).sum(0).clamp_min(1e-9)
        neg = (-drives).clamp(min=0).sum(0).clamp_min(1e-9)
        mv = torch.where(drives >= 0, drives / pos * e_cur, drives / neg * i_cur)
        mf_e = float((b.mf_exc[q] * b.mf_rate).sum())
        mf_i = float((b.mf_inh[q] * b.mf_rate).sum())
        t_open = b.t_mask[lgn.slice] & (v >= cfg.t_vh_mv)
        t_cur = torch.where(t_open, cfg.t_gt_ratio * b.t_h[lgn.slice] * (cfg.t_et_mv - v), 0.0)
        out = {}
        for g, m in masks.items():
            if not bool(m.any()):
                continue
            row = [mv[j][m].mean() for j in range(len(paths))]
            row += [v.new_tensor(mf_e), v.new_tensor(mf_i), b.bias[lgn.slice][m].mean(), t_cur[m].mean(),
                    (-b.adapt[lgn.slice][m]).mean(), v[m].mean(), r[lgn.slice][m].mean()]
            row += [h[m].mean() for h in src_hz]
            out[g] = torch.stack([x.float() for x in row]).cpu()
        return out

    acc, cnt = {}, {}

    def add(win: str, res: dict) -> None:
        for g, x in res.items():
            acc[(win, g)] = acc.get((win, g), 0) + x
            cnt[(win, g)] = cnt.get((win, g), 0) + 1

    s.line, s.gaze = "", 0.0
    b.run(2500.0, s._blank(2500.0))
    n_rec = 2 * cfg.retina_rows * cfg.retina_cols
    # Своя активность сетчатки (SensorSpec.spontaneous_hz) — движок добавляет
    # её к растру глаза (спайк, если есть хоть один из двух).
    p_sp = b.spontaneous[off:off + n_rec] if b.spontaneous is not None else torch.zeros(n_rec, device=b.device)

    def retina_hz(view: torch.Tensor) -> torch.Tensor:
        p_v = view[:, off:off + n_rec].mean(0)
        return (p_v + p_sp - p_v * p_sp) * (1000.0 / cfg.dt_ms)

    n_rest, n_show, early = int(rest // WIN), int(show // WIN), int(50 // WIN)
    for _ in range(n_rep):
        for ch in letters:
            shift = int(torch.randint(-1, 2, (1,), generator=gen))
            s.line, s.gaze = ch, s.screen.width(ch) / 2 + shift
            rates = s.eye.rates(s._picture(s.screen, cfg.fovea_col)).to(b.device)[driver]
            masks = {"штрих": rates >= 0.3 * cfg.retina_rate_hz, "рядом": (rates > 0) & (rates < 0.3 * cfg.retina_rate_hz),
                     "пусто": rates == 0}
            s.line = ""
            for t in range(n_rest):
                view = s._blank(WIN)
                r, st = observe_pre(b, WIN, view)
                if t >= n_rest - 20:
                    add("пауза", sample(r, st, retina_hz(view), masks))
            s.line = ch
            for t in range(n_show):
                view = s._view(WIN)
                r, st = observe_pre(b, WIN, view)
                rec = retina_hz(view)
                add("0–50 мс" if t < early else f"50–{int(show)} мс", sample(r, st, rec, masks))
            s.line = ""

    wins = ("пауза", "0–50 мс", f"50–{int(show)} мс")
    for g in groups:
        print(f"\n=== реле «{g}» (вход, мВ; пауза / 0–50 / 50–{int(show)} мс)", flush=True)
        for j, name in enumerate(keys):
            vals = [float(acc[(w, g)][j] / cnt[(w, g)]) if (w, g) in acc else float("nan") for w in wins]
            print(f"  {name:46s} " + " ".join(f"{x:9.2f}" for x in vals), flush=True)


if __name__ == "__main__":
    main(dict(kv.split("=", 1) for kv in sys.argv[1:]))
