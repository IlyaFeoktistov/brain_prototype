"""
Передаточная функция клетки букв для правила с учителем (DESIGN 2026-11-13):
φ(V) правила (engine._us) должна быть частотой самой клетки при мембране V
(Urbanczik & Senn 2014, https://doi.org/10.1016/j.neuron.2013.11.030: сома —
пуассон с частотой φ(V_дендрит)). Замер прежней φ (us_phi_check): центр у
v_t и максимум 100 Гц завышают частоту ~×8 у всех клеток.

Замер — без учителя и правила, веса врождённые, фон сети обычный: те же 200
клеток букв L4 E у ямки, что в letters_us; показы буквы 350 мс + пауза 150
мс. Мембрана V в начале каждого куска движка (5 мс) и спайки клетки за этот
кусок → мгновенная частота при V (разрешение 5 мс — шаг куска; допущение).
Подгонка φ(V) = max · σ((V − v_t − центр) / наклон) по пуассоновскому
правдоподобию (для каждой пары центр/наклон максимум — в закрытом виде) на
нечётных показах; проверка на чётных (независимые интервалы): по корзинам V
и по клеткам, прежняя φ — рядом. Корзины V с < min_n наблюдений не
показываются (экстраполяция туда — без опоры).

    PYTHONPATH=. .venv/bin/python tools/us_phi_fit.py [shows=6] [seed=1] [min_n=500]
"""

from __future__ import annotations

import sys

import torch

from brain.session import Session
from tools.reading_bench import build


def main(opts: dict) -> None:
    seed = int(opts.get("seed", 1))
    opts = {"fit": "lgn-a", "homeo": f"lgn-d-pop-tcind-s{seed}", "relay_v_t_mv": "-59", "trn_scale": "0.1",
            "us": "visual -> letters", **opts}
    mind, b = build(opts)
    cfg = b.cfg
    s = Session(mind, window_ms=10.0)
    letters = "абвгдежзик"
    L = len(letters)
    shows, min_n = int(opts.get("shows", 6)), int(opts.get("min_n", 500))
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    L4 = b.us[0].post
    dev = b.device
    near = (b.field_cols[L4.name] - cfg.fovea_col).abs().argsort()[:20 * L].to(dev)
    cells = near + L4.start
    vt = b.v_t[cells]
    chunk = b.CHUNK * cfg.dt_ms
    dt_s = chunk / 1000.0

    V, N, part = [], [], []                     # V − v_t в начале куска, спайки за кусок, № показа

    def window(ms: float, raster: torch.Tensor, show: int) -> None:
        for i in range(int(round(ms / chunk))):
            dv = (b.v[cells] - vt).clone()
            r = b.observe(chunk, raster[i * b.CHUNK:(i + 1) * b.CHUNK])
            V.append(dv)
            N.append(r[cells] * dt_s)
            part.append(show)

    s.line, s.gaze = "", 0.0
    b.run(2500.0, s._blank(2500.0))
    order = torch.arange(L).repeat(shows)
    order = order[torch.randperm(len(order), generator=gen)]
    for j, k in enumerate(order.tolist()):
        s.line = ""
        window(150.0, s._blank(150.0), j)
        s.line = letters[k]
        s.gaze = s.screen.width(letters[k]) / 2 + float(torch.randint(-1, 2, (1,), generator=gen))
        window(350.0, s._view(350.0), j)
    V = torch.stack(V).cpu().double()            # [кусок, клетка]
    N = torch.stack(N).cpu().double().round()
    part = torch.tensor(part)
    fit, val = (part % 2 == 1), (part % 2 == 0)

    def nll(v, n, mx, c, sl):
        lam = mx * torch.sigmoid((v - c) / sl) * dt_s
        return float((lam - n * torch.log(lam.clamp(min=1e-12))).sum())

    vf, nf = V[fit].flatten(), N[fit].flatten()
    best = None
    for c in torch.arange(-12.0, 15.01, 0.25):
        for sl in torch.arange(0.25, 6.01, 0.25):
            sg = torch.sigmoid((vf - c) / sl)
            mx = float(nf.sum() / (sg.sum() * dt_s))
            if mx > 2000.0:                      # выше рефрактерного предела 1/τ_реф — не частота клетки
                continue
            l = nll(vf, nf, mx, float(c), float(sl))
            if best is None or l < best[0]:
                best = (l, mx, float(c), float(sl))
    _, mx, c, sl = best
    old = (cfg.us_max_khz * 1000.0, 0.0, cfg.us_slope_mv)
    new = (mx, c, sl)
    print(f"наблюдений {V.numel()} (клеток {V.shape[1]}, кусков {V.shape[0]} по {chunk} мс), спайков {int(N.sum())}; "
          f"подгонка — нечётные показы, проверка — чётные")
    print(f"прежняя φ: max {old[0]:.0f} Гц, центр v_t{old[1]:+.2f} мВ, наклон {old[2]:.2f} мВ")
    print(f"новая φ:   max {mx:.1f} Гц, центр v_t{c:+.2f} мВ, наклон {sl:.2f} мВ "
          f"(край сетки: центр −12…+15, наклон 0.25…6)")

    vv, nv = V[val].flatten(), N[val].flatten()
    print("\nпроверка (чётные показы) по корзинам V − v_t: n | частота | новая φ | прежняя φ (Гц)")
    edges = torch.arange(-20.0, 6.01, 2.0)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (vv >= lo) & (vv < hi)
        if int(m.sum()) < min_n:
            continue
        rate = float(nv[m].mean() / dt_s)
        pn = float((new[0] * torch.sigmoid((vv[m] - new[1]) / new[2])).mean())
        po = float((old[0] * torch.sigmoid((vv[m] - old[1]) / old[2])).mean())
        print(f"  [{float(lo):+.0f}, {float(hi):+.0f}) n {int(m.sum()):7d} | {rate:6.2f} | {pn:6.2f} | {po:6.2f}")
    m = vv >= edges[-1]
    print(f"  ≥ {float(edges[-1]):+.0f}: n {int(m.sum())} (мало — без оценки)" if int(m.sum()) < min_n else
          f"  ≥ {float(edges[-1]):+.0f}: n {int(m.sum())} | {float(nv[m].mean() / dt_s):.2f}")

    rate_c = N[val].mean(0) / dt_s
    for name, (a, cc, ss) in (("новая", new), ("прежняя", old)):
        phi_c = (a * torch.sigmoid((V[val] - cc) / ss)).mean(0)
        err = rate_c - phi_c
        q = torch.quantile(err, torch.tensor([0.05, 0.5, 0.95], dtype=err.dtype))
        print(f"по клеткам, {name} φ: частота {float(rate_c.mean()):.2f}, φ {float(phi_c.mean()):.2f} Гц, ошибка "
              f"{float(err.mean()):+.2f} (5/50/95%: {' / '.join(f'{float(x):+.2f}' for x in q)}), ошибка < 0 у "
              f"{float((err < 0).float().mean()):.0%}; корр. частота–φ по клеткам {float(torch.corrcoef(torch.stack([rate_c, phi_c]))[0, 1]):.2f}")


if __name__ == "__main__":
    main(dict(kv.split("=", 1) for kv in sys.argv[1:]))
