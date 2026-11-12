"""
Почему клетки V1 почти не молчат на пустом экране (замер, мозг не меняется;
мозг как в letters_dc, гомеостаз lgn-d-pop заморожен). У бодрствующей макаки
на пустом сером экране молчат 35% клеток L2 и 72% L3 (Gur & Snodderly 2008,
https://doi.org/10.1113/jphysiol.2008.151795), у мозга — 1–2% за 20 с.

По каждой клетке популяции (pops) за seconds с пустого экрана, шаг мозга
(CHUNK = 1): средний потенциал мембраны вне рефрактерности, его размах (СКО),
расстояние до порога экспоненты v_t в мВ и в СКО (z), частота. Клетки — по
z: z < 1 — постоянный ток держит у порога; z ≥ 3 — порог переходят только
редкие выбросы. Плюс постоянные для всей популяции части входа, мВ
(установившийся вклад): внешний фон колонки (шум движка, среднее и СКО
одного шага), постоянный ток недостающих соседей (dc_mv), bias со сдвигом
гомеостаза — и их разброс по клеткам (у одинаковых для всех клеток частей
разброса нет — молчащих они не создают).

Сверка: у бодрствующей макаки на пустом экране (фиксация, whole-cell, 31
клетка V1) медиана расстояния мембраны до порога спайка 13.9 мВ, медиана
асимметрии распределения потенциала +0.72 — хвост в деполяризацию, спайки от
редких крупных коррелированных событий (Tan et al. 2014,
https://doi.org/10.1038/nn.3684, PMC4067243). Поэтому ещё: асимметрия
потенциала каждой клетки; для traced клеток у ямки — полная запись мембраны:
корреляция потенциала и счёта спайков (окна 50 мс) между соседями, крупные
деполяризации (выше среднего клетки на 2 СКО: доля времени, событий в
секунду). Порог в мозге — v_t экспоненты AdEx, в данных — начало спайка
(сопоставимо с точностью до нескольких мВ).

    PYTHONPATH=. .venv/bin/python tools/silence_check.py [pops=visual.L4E,visual.L23E] [seconds=3] [traced=300] [seed=1] [bg=independent|shared|local|network] [bg_patch=4]
"""

from __future__ import annotations

import sys
import time

import torch

from brain.engine import SpikingBrain
from brain.session import Session

SpikingBrain.CHUNK = 1

from tools.reading_bench import build  # noqa: E402


def main(opts: dict) -> None:
    seed = int(opts.get("seed", 1))
    opts = {"fit": "lgn-a", "homeo": f"lgn-d-pop{'' if opts.get('tc_indep') == '0' else '-tcind'}-s{seed}", "relay_v_t_mv": "-59", "trn_scale": "0.1", **opts}
    mind, b = build(opts)
    s = Session(mind, window_ms=10.0)
    cfg = b.cfg
    dt = cfg.dt_ms
    pops = {}
    for name in opts.get("pops", "visual.L4E,visual.L23E").split(","):
        area, rest = name.split(".")
        pops[name] = b.areas[area].layers[rest[:-1] if rest.endswith("E") else rest[:-2]][rest[-1] if rest.endswith("E") else rest[-2:]]
    s.line, s.gaze = "", 0.0
    b.run(2500.0, s._blank(2500.0))
    steps = int(float(opts.get("seconds", 3)) * 1000 / dt)
    acc = {k: [torch.zeros(p.size, device=b.device) for _ in range(4)] for k, p in pops.items()}
    # Как у Tan et al.: без спайка — без подъёма к нему (V > v_t), рефрактерности и 3 мс после.
    clean = {k: [torch.zeros(p.size, device=b.device) for _ in range(4)] for k, p in pops.items()}
    after = {k: torch.zeros(p.size, device=b.device) for k, p in pops.items()}
    tail = int(round(3.0 / dt))
    n_tr = int(opts.get("traced", 300))
    traced = {k: (b.field_cols[p.name] - cfg.fovea_col).abs().argsort()[:n_tr].to(b.device) if p.name in b.field_cols
              else torch.arange(n_tr, device=b.device) for k, p in pops.items()}
    traces = {k: torch.zeros(steps, n_tr, device=b.device) for k in pops}
    fired = {k: torch.zeros(steps, n_tr, device=b.device) for k in pops}
    spikes = {k: torch.zeros(p.size, device=b.device) for k, p in pops.items()}
    # Проводимость входа в долях утечки: ток модели нормирован к точке замера PSP,
    # g_E/g_L = I_E / (E_AMPA − V_замера EPSP), g_I/g_L = |I_I| / (V_замера IPSP − E_ГАМК);
    # τ_эфф = τ_m / (1 + g_E + g_I). NMDA и плато дендрита не входят.
    # По каждой клетке: Σg_E, Σg_I, Σфон, Σg_E², Σg_I², Σg_E·g_I — среднее, СКО и корреляция E/I во времени.
    cond = {k: [torch.zeros(p.size, device=b.device) for _ in range(8)] for k, p in pops.items()}
    g_tr = {k: torch.zeros(steps, 2, n_tr, device=b.device) for k in pops}
    e_span = cfg.ampa_reversal_mv - cfg.epsp_holding_mv
    i_span = cfg.ipsp_holding_mv - cfg.gaba_reversal_mv
    view = s._blank(steps * dt)
    t0 = time.time()
    for t in range(steps):
        b.run(dt, view[t:t + 1])
        for k, p in pops.items():
            v = b.v[p.slice]
            free = (b.refractory[p.slice] <= 0).float()
            a = acc[k]
            a[0] += free
            a[1] += v * free
            a[2] += v * v * free
            a[3] += v * v * v * free
            ref = b.refractory[p.slice] > 0
            after[k] = torch.where(ref, float(tail), (after[k] - 1).clamp(min=0))
            ok = ((~ref) & (after[k] <= 0) & (v <= b.v_t[p.slice])).float()
            cl = clean[k]
            cl[0] += ok
            cl[1] += v * ok
            cl[2] += v * v * ok
            cl[3] += v * v * v * ok
            spikes[k] += b.spikes[p.slice]
            ge = (b.i_syn[p.slice] + b.last_dc_e[p.slice]) / e_span
            gi = (b.i_inh[p.slice] + b.i_ngf[p.slice] + b.last_dc_i[p.slice]).abs() / i_span
            bg = b.noise_mean[p.slice] / dt * b.noise_gain[p.slice] * b.tau_m[p.slice] / e_span
            c = cond[k]
            c[0] += ge
            c[1] += gi
            c[2] += bg
            c[3] += ge * ge
            c[4] += gi * gi
            c[5] += ge * gi
            # Доли без импульсов: недостающие соседи (среднее поле + постоянная часть) — у E и у I.
            c[6] += b.last_dc_e[p.slice] / e_span if torch.is_tensor(b.last_dc_e) else 0.0
            c[7] += b.last_dc_i[p.slice].abs() / i_span if torch.is_tensor(b.last_dc_i) else 0.0
            g_tr[k][t, 0] = ge[traced[k]]
            g_tr[k][t, 1] = gi[traced[k]]
            traces[k][t] = v[traced[k]]
            fired[k][t] = b.spikes[p.slice][traced[k]]
    print(f"пустой экран {steps * dt / 1000:.1f} с, {time.time() - t0:.0f} с", flush=True)
    for k, p in pops.items():
        n, s1, s2, s3 = acc[k]
        mean = s1 / n.clamp_min(1)
        sd = (s2 / n.clamp_min(1) - mean ** 2).clamp_min(1e-9).sqrt()
        skew = (s3 / n.clamp_min(1) - 3 * mean * sd ** 2 - mean ** 3) / sd ** 3
        vt = b.v_t[p.slice]
        gap = vt - mean
        z = gap / sd
        hz = spikes[k] / (steps * dt / 1000)
        q = lambda x: " / ".join(f"{float(v):.2f}" for v in torch.quantile(x.float(), torch.tensor([0.1, 0.5, 0.9], device=x.device)))
        tau = float(b.tau_m[p.start])
        ext_mean = b.noise_mean[p.slice] / dt * b.noise_gain[p.slice] * tau
        ext_sd = b.noise_std[p.slice] * b.noise_gain[p.slice]
        print(f"\n{k} ({p.size} клеток): частота 10/50/90% {q(hz)} Гц, молчат {float((hz == 0).float().mean()):.1%}")
        print(f"  мембрана, мВ: средняя {q(mean)}, СКО {q(sd)}, до порога v_t {q(gap)} мВ = {q(z)} СКО")
        for lo, hi in ((-99, 1), (1, 2), (2, 3), (3, 99)):
            m = (z >= lo) & (z < hi)
            if m.any():
                print(f"  z {lo if lo > -99 else '−∞'}…{hi if hi < 99 else '∞'}: клеток {float(m.float().mean()):.0%}, "
                      f"частота медиана {float(hz[m].median()):.2f} Гц, молчат {float((hz[m] == 0).float().mean()):.0%}")
        m0, m1, m2, m3 = clean[k]
        cmean = m1 / m0.clamp_min(1)
        csd = (m2 / m0.clamp_min(1) - cmean ** 2).clamp_min(1e-9).sqrt()
        cskew = (m3 / m0.clamp_min(1) - 3 * cmean * csd ** 2 - cmean ** 3) / csd ** 3
        print(f"  асимметрия потенциала {q(skew)}; без спайков (как Tan) {q(cskew)}, до порога без спайков {q(vt - cmean)} мВ "
              f"(макака на пустом экране: асимметрия +0.72, до порога 13.9 мВ)")
        # Соседи у ямки: корреляция потенциала (без шагов спайка/рефрактерности — как есть) и счёта за 50 мс.
        tr, fr = traces[k], fired[k]
        zt = (tr - tr.mean(0)) / tr.std(0).clamp_min(1e-6)
        cv = (zt.T @ zt) / len(zt)
        off = cv[~torch.eye(n_tr, dtype=torch.bool, device=cv.device)]
        w = int(50 / dt)
        cnt = fr[: len(fr) // w * w].view(-1, w, n_tr).sum(1)
        ok = cnt.std(0) > 0
        cc = torch.corrcoef(cnt[:, ok].T)
        offc = cc[~torch.eye(int(ok.sum()), dtype=torch.bool, device=cc.device)]
        high = tr > (tr.mean(0) + 2 * tr.std(0))
        starts = (high[1:] & ~high[:-1]).float().sum(0) / (steps * dt / 1000)
        # По расстоянию на карте (px) и общий ритм: спектр суммарного счёта traced клеток.
        if p.name in b.map_pos:
            pos = b.map_pos[p.name][traced[k]]
            dist = torch.cdist(pos, pos)
            eye = torch.eye(n_tr, dtype=torch.bool, device=cv.device)
            parts = []
            for lo, hi in ((0, 1), (1, 2), (2, 4), (4, 8), (8, 99)):
                m = (dist >= lo) & (dist < hi) & ~eye
                if m.any():
                    parts.append(f"{lo}–{hi if hi < 99 else '…'} px {float(cv[m].median()):+.2f}")
            print("  корреляция потенциала по расстоянию: " + ", ".join(parts))
        pop_rate = fr.sum(1)
        pop_rate = pop_rate[: len(pop_rate) // 2 * 2].view(-1, 2).sum(1)  # шаги по 1 мс
        spec = torch.fft.rfft(pop_rate - pop_rate.mean()).abs() ** 2
        freq = torch.fft.rfftfreq(len(pop_rate), d=0.001).to(spec.device)
        band = (freq >= 1) & (freq <= 100)
        peak = float(freq[band][spec[band].argmax()])
        share = float(spec[band][(freq[band] - peak).abs() <= 2].sum() / spec[band].sum())
        cv_pop = float(pop_rate.float().std() / pop_rate.float().mean().clamp_min(1e-9))
        print(f"  общий ритм: пик спектра суммарного счёта {peak:.1f} Гц, доля мощности ±2 Гц у пика {share:.0%}; "
              f"CV суммарного счёта за 1 мс {cv_pop:.2f}")
        print(f"  {n_tr} клеток у ямки: корреляция потенциала соседей медиана {float(off.median()):+.3f}, "
              f"счёта за 50 мс {float(offc.nanmedian()):+.3f}; выше среднего + 2 СКО {float(high.float().mean()):.1%} времени, "
              f"событий {float(starts.median()):.1f} в с")
        r = torch.corrcoef(torch.stack([hz, -z]))[0, 1]
        print(f"  корреляция частоты с близостью к порогу (−z): {float(r):+.2f}")
        print(f"  постоянные части, мВ (медиана; разброс 10–90%): внешний фон {q(ext_mean)} (СКО шага {float(ext_sd.median()):.2f}), "
              f"dc соседей {q(b.dc_mv[p.slice])}, bias+гомеостаз {q(b.bias[p.slice])}, v_t {q(vt)}")
        ge, gi, gb, ge2, gi2, gei, gde, gdi = (c / steps for c in cond[k])
        sde = (ge2 - ge ** 2).clamp_min(0).sqrt()
        sdi = (gi2 - gi ** 2).clamp_min(0).sqrt()
        rei = (gei - ge * gi) / (sde * sdi).clamp_min(1e-9)
        q5 = lambda x: " / ".join(f"{float(v):.2f}" for v in torch.quantile(x.float(), torch.tensor([0.05, 0.5, 0.95], device=x.device)))
        print(f"  проводимость входа / g_L по клеткам (5 / 50 / 95%): g_E среднее {q5(ge)} (внешний фон {q5(gb)}), СКО {q5(sde)}; "
              f"g_I среднее {q5(gi)}, СКО {q5(sdi)}")
        print(f"    без импульсов (среднее поле недостающих соседей + постоянная часть): g_E {q5(gde)}, g_I {q5(gdi)}")
        print(f"    g_I/g_E {q5(gi / ge.clamp_min(1e-6))}, СКО_I/СКО_E {q5(sdi / sde.clamp_min(1e-9))} "
              f"(σ_e/σ_i ниже √((V_t−E_I)/(E_E−V_t)) = {((-50 - cfg.gaba_reversal_mv) / (cfg.ampa_reversal_mv + 50)) ** 0.5:.2f} — спайк по спаду торможения, Piwkowska 2008); "
              f"корр. g_E–g_I во времени {q5(rei)}")
        print(f"    полная / покоя {q5(1 + ge + gi)}, τ_эфф {q5(tau / (1 + ge + gi))} мс (τ_m {tau:.1f}); "
              f"среднее по клеткам: g_E {float(ge.mean()):.2f}, g_I {float(gi.mean()):.2f}, СКО {float(sde.mean()):.3f} / {float(sdi.mean()):.3f}")
        # STA проводимостей за 50 мс до спайка (traced клетки; спайки после ≥ 100 мс тишины, как Rudolph 2007),
        # в долях g_L относительно своего среднего клетки.
        gt, fr_t = g_tr[k], fired[k]
        win = int(50 / dt)
        quiet = int(100 / dt)
        sums, n_sp = torch.zeros(win, 2, device=b.device), 0
        mean_c = gt.mean(0)
        idx = fr_t.nonzero()
        last = torch.full((n_tr,), -10 ** 9, device=b.device, dtype=torch.long)
        for t_s, c_s in idx.tolist():
            ok_sp = t_s - int(last[c_s]) >= quiet and t_s >= win
            last[c_s] = t_s
            if ok_sp:
                sums += gt[t_s - win:t_s, :, c_s] - mean_c[:, c_s]
                n_sp += 1
        if n_sp:
            sta = sums / n_sp
            pts = [50, 20, 10, 5, 2, 1]
            row = lambda j: " ".join(f"{float(sta[win - int(ms / dt), j]):+.3f}" for ms in pts)
            tot = sta[:, 0] + sta[:, 1]
            print(f"    STA за {'/'.join(str(x) for x in pts)} мс до спайка ({n_sp} спайков после ≥100 мс тишины), Δ от среднего клетки: "
                  f"g_E {row(0)}; g_I {row(1)}; полная {' '.join(f'{float(tot[win - int(ms / dt)]):+.3f}' for ms in pts)}")


if __name__ == "__main__":
    main(dict(kv.split("=", 1) for kv in sys.argv[1:]))
