"""
Из чего ответ клетки V1 L4 E на штрих в своём поле (замер, без
вмешательства; мозг как в letters_dc, гомеостаз lgn-d). Продолжение
tools/l4_response.py: клетка со штрихом в поле прибавляет ~1 Гц — какой
вход гасит ответ.

Группы клеток на показе — как l4_response (по входам LGN: «в поле» — у ≥
половины водитель ≥ 2 × поддерживаемой, «против» — < ½, «вне»). По каждому
пути в L4 E — устоявшийся ток сомы, мВ: Σ вес × сила синапса × частота
источника за окно → ток i_syn (τ_syn) × движущая сила:
  сила синапса — как в engine._step, в начале окна (observe_pre): у пути
  таламуса — ресурс реле (stp_x,
  депрессия), у синапсов коры — u·x/(1 − u)/U своей пары типов (stp_u4,
  stp_x4), торможение × усиление клетки (inh_gain);
  отдельно — внешний фон колонки (шум движка), ток недостающих соседей
  (среднее поле), bias, адаптация; проверка — сумма против реального тока
  сомы на конце окна. Входы на дендрит (dendrite_*) не учтены.
Окна: пауза (последние 100 мс), показ 0–50 и 50–show мс; у «в поле» —
распределение по клеткам (10 / 50 / 90%) и ход по 10 мс первых 100 мс:
частота реле-входов, их ресурс, ток таламуса, ток PV, мембрана, частота.

    PYTHONPATH=. .venv/bin/python tools/l4_inputs.py [letters=абвгдежзик] [n=5] [rest=300] [show=350] [seed=1]
"""

from __future__ import annotations

import sys
import time

import torch

from brain.engine import KINDS
from brain.session import Session
from tools.reading_bench import build, observe_pre

WIN = 10.0


def main(opts: dict) -> None:
    seed = int(opts.get("seed", 1))
    opts = {"fit": "lgn-a", "homeo": f"lgn-d-pop{'' if opts.get('tc_indep') == '0' else '-tcind'}-s{seed}", "relay_v_t_mv": "-59", "trn_scale": "0.1", **opts}
    mind, b = build(opts)
    cfg = b.cfg
    s = Session(mind, window_ms=WIN)
    letters = opts.get("letters", "абвгдежзик")
    n_rep = int(opts.get("n", 5))
    rest, show = float(opts.get("rest", 300)), float(opts.get("show", 350))
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    lgn = b.areas["lgn"].E
    L4 = b.areas["visual"].layers["L4"]["E"]
    sl = L4.slice
    eye_i = [x.name for x in b.sensors].index("eye")
    driver = next(p for p in b.projections if p.name == "eye -> " + lgn.name).pre[:, 0].long() - b.sensor_offsets[eye_i]
    paths = [p for p in b.projections if p.post is L4 and p.source == "net"]
    tc = next(p for p in paths if p.name == f"lgn -> {L4.name}")
    tc_pre = tc.pre.long() - lgn.start
    m = cfg.retina_maintained_hz
    e_type = KINDS.index("E")
    # Перевод поставки (мВ на спайк × спайков за шаг) в устоявшийся ток i_syn.
    k_e = (b.syn_scale[sl] / (1.0 - b.decay_syn[sl]))
    k_i = (b.syn_scale_inh[sl] / (1.0 - b.decay_syn_inh[sl]))
    q = int(b.mf_post_id[L4.start])
    print("входы V1 L4 E:")
    for p in paths:
        print(f"  {p.name}: {p.weight.shape[1]} входов на клетку, вес {float(p.weight.float().mean()):+.3f} мВ", flush=True)

    names = [p.name for p in paths]
    keys = names + ["внешний фон колонки", "среднее поле возб.", "среднее поле торм.", "bias", "адаптация",
                    "Σ возб. (собрано)", "Σ торм. (собрано)", "возб. сомы (движок)", "торм. сомы (движок)",
                    "мембрана, мВ", "частота, Гц", "частота реле-входов, Гц", "ресурс реле-входов"]

    def window(view: torch.Tensor) -> torch.Tensor:
        r, st = observe_pre(b, WIN, view)
        v = b.v[sl]
        excite = ((cfg.ampa_reversal_mv - v) / (cfg.ampa_reversal_mv - cfg.epsp_holding_mv)).clamp(min=0.0)
        force = ((v - cfg.gaba_reversal_mv) / (cfg.ipsp_holding_mv - cfg.gaba_reversal_mv)).clamp(min=0.0)
        per_step = cfg.dt_ms / 1000.0
        rows = []
        for p in paths:
            pre = p.pre.long()
            w = p.weight.float()
            if p is tc:
                g = st["x"][pre]
            else:
                typed = b.typed_pre[pre]
                slot = pre * len(KINDS) + e_type
                u = st["u4"].view(-1)[slot]
                x = st["x4"].view(-1)[slot]
                U = b.stp_U4.view(-1)[slot]
                nxt = u + U * (1.0 - u)      # сила следующего спайка (как у среднего поля)
                g = torch.where(typed, nxt * x / U, 1.0)
            supply = (w * g * r[pre] * per_step).sum(1)
            if float(w.mean()) < 0:
                rows.append(supply * b.inh_gain[sl] * k_i * force)
            else:
                rows.append(supply * k_e * excite)
        noise = b.noise_mean[sl] * b.noise_gain[sl] * k_e * excite
        mf_e = (b.mf_exc[q] * b.mf_rate).sum() * excite
        mf_i = (b.mf_inh[q] * b.mf_rate).sum() * force
        exc = sum(x for x in rows if float(x.mean()) >= 0) + noise + mf_e
        inh = sum(x for x in rows if float(x.mean()) < 0) + mf_i
        e_true = (b.i_syn[sl] + b.dc_mv[sl]) * excite
        i_true = (b.i_inh[sl] + b.i_ngf[sl] + b.dc_inh_mv[sl]) * force
        rows += [noise, mf_e, mf_i, b.bias[sl].expand_as(v), -b.adapt[sl], exc, inh, e_true, i_true, v.clone(),
                 r[sl], r[lgn.slice][tc_pre].mean(1), st["x"][lgn.start + tc_pre].mean(1)]
        return torch.stack([x.float().expand_as(v) for x in rows]).cpu()   # [величин, клеток]

    s.line, s.gaze = "", 0.0
    b.run(2500.0, s._blank(2500.0))
    t0 = time.time()
    n_rest, n_show, n_early = int(rest // WIN), int(show // WIN), int(50 // WIN)
    sums = {}                      # (окно, группа) → [величин]
    counts = {}
    field_rows = {"пауза": [], "0–50 мс": [], f"50–{int(show)} мс": []}
    course = torch.zeros(n_show, len(keys))
    course_n = 0
    for rep in range(n_rep):
        for k in torch.randperm(len(letters), generator=gen).tolist():
            ch = letters[k]
            s.line, s.gaze = ch, s.screen.width(ch) / 2 + float(torch.randint(-1, 2, (1,), generator=gen))
            d = s.eye.rates(s._picture(s.screen, cfg.fovea_col)).to(b.device)[driver][tc_pre]
            groups = {"в поле": ((d >= 2 * m).float().mean(1) >= 0.5).cpu(),
                      "против": ((d < 0.5 * m).float().mean(1) >= 0.5).cpu()}
            groups["вне"] = ~(groups["в поле"] | groups["против"])
            s.line = ""
            pause = [window(s._blank(WIN)) for _ in range(n_rest)][-10:]
            s.line = ch
            shown = [window(s._view(WIN)) for _ in range(n_show)]
            s.line = ""
            parts = {"пауза": torch.stack(pause).mean(0), "0–50 мс": torch.stack(shown[:n_early]).mean(0),
                     f"50–{int(show)} мс": torch.stack(shown[n_early:]).mean(0)}
            for w, X in parts.items():
                for g, mask in groups.items():
                    if mask.any():
                        sums[(w, g)] = sums.get((w, g), 0) + X[:, mask].sum(1)
                        counts[(w, g)] = counts.get((w, g), 0) + int(mask.sum())
                field_rows[w].append(X[:, groups["в поле"]])
            if groups["в поле"].any():
                course += torch.stack([x[:, groups["в поле"]].mean(1) for x in shown])
                course_n += 1
        print(f"  проход {rep + 1}/{n_rep}, {time.time() - t0:.0f} с", flush=True)

    wins = list(field_rows)
    for g in ("в поле", "против", "вне"):
        print(f"\n=== V1 L4 E «{g}»: ток сомы по путям, мВ (среднее по парам клетка × показ; " + " / ".join(wins) + ")")
        for j, key in enumerate(keys):
            vals = [float(sums[(w, g)][j] / counts[(w, g)]) for w in wins]
            print(f"  {key:40s} " + " ".join(f"{x:9.2f}" for x in vals), flush=True)
    print(f"\n=== «в поле»: распределение по клеткам (10 / 50 / 90%), " + " | ".join(wins))
    for j, key in enumerate(keys):
        cells = []
        for w in wins:
            X = torch.cat(field_rows[w], 1)[j]
            qv = torch.quantile(X, torch.tensor([0.1, 0.5, 0.9]))
            cells.append(" ".join(f"{float(x):7.2f}" for x in qv))
        print(f"  {key:40s} " + " | ".join(cells), flush=True)
    print(f"\n=== «в поле»: ход по 10 мс показа (первые 100 мс; среднее по {course_n} показам)")
    show_keys = [f"lgn -> {L4.name}"] + [n for n in names if "PV" in n] + ["Σ возб. (собрано)", "Σ торм. (собрано)", "мембрана, мВ",
                                                                         "частота, Гц", "частота реле-входов, Гц", "ресурс реле-входов"]
    c = course / max(course_n, 1)
    print("  " + " " * 40 + " ".join(f"{t * WIN:6.0f}" for t in range(10)))
    for key in show_keys:
        j = keys.index(key)
        print(f"  {key:40s} " + " ".join(f"{float(c[t, j]):6.2f}" for t in range(10)), flush=True)
    print(f"всего {time.time() - t0:.0f} с")


if __name__ == "__main__":
    main(dict(kv.split("=", 1) for kv in sys.argv[1:]))
