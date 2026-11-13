"""
Предсказание правила с учителем против настоящей частоты клеток букв
(DESIGN 2026-11-13). Правило (engine._us): ΔW ∝ (S − φ(V)·dt) · след пре,
φ(V) = us_max · сигмоида((V − v_t) / us_slope). Если φ систематически выше
настоящей частоты, ошибка в среднем отрицательна — правило гасит все входы.

Без учителя и правила: те же 200 клеток букв L4 E у ямки, что в letters_us
(по 20 на букву), показы буквы 350 мс / пауза 150 мс. Каждые 5 мс (кусок
движка) — мембрана V клеток; за окно — спайки. По клетке: настоящая частота,
средняя φ(V) в Гц, ошибка (частота − φ), среднее и квантили V − v_t, доля
времени V > v_t. Отдельно: пауза перед буквой (последние 100 мс) и буква.
Два состояния весов: врождённые (начало учёбы) и load=… (после учёбы).

    PYTHONPATH=. .venv/bin/python tools/us_phi_check.py [shows=3] [seed=1] [load=~/.brain/letters_us_s1.pt]
"""

from __future__ import annotations

import os
import sys

import torch

from brain.session import Session
from tools.reading_bench import build


def main(opts: dict) -> None:
    seed = int(opts.get("seed", 1))
    load = os.path.expanduser(opts.pop("load", "~/.brain/letters_us_s1.pt"))
    opts = {"fit": "lgn-a", "homeo": f"lgn-d-pop-tcind-s{seed}", "relay_v_t_mv": "-59", "trn_scale": "0.1",
            "us": "visual -> letters", **opts}
    mind, b = build(opts)
    cfg = b.cfg
    s = Session(mind, window_ms=10.0)
    letters = "абвгдежзик"
    L = len(letters)
    shows = int(opts.get("shows", 3))
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    path = b.us[0]
    L4 = path.post
    dev = b.device
    near = (b.field_cols[L4.name] - cfg.fovea_col).abs().argsort()[:20 * L].to(dev)
    group = (torch.arange(len(near), device=dev) % L).cpu()
    cells = near + L4.start
    vt = b.v_t[cells]
    chunk = b.CHUNK * cfg.dt_ms
    w_innate = path.weight.clone()

    def window(ms: float, raster: torch.Tensor | None) -> dict:
        n = int(round(ms / chunk))
        steps = b.CHUNK
        phi = torch.zeros(len(cells), device=dev)
        above = torch.zeros(len(cells), device=dev)
        vs = []
        spikes = torch.zeros(len(cells), device=dev)
        for i in range(n):
            part = None if raster is None else raster[i * steps:(i + 1) * steps]
            r = b.observe(chunk, part)
            spikes += r[cells] * (chunk / 1000.0)
            dv = b.v[cells] - vt
            phi += cfg.us_max_khz * 1000.0 * torch.sigmoid((dv - cfg.us_center_mv) / cfg.us_slope_mv)
            above += (dv > 0).float()
            vs.append(dv)
        return {"rate": spikes / (ms / 1000.0), "phi": phi / n, "above": above / n, "dv": torch.stack(vs)}

    def measure(tag: str) -> None:
        s.line, s.gaze = "", 0.0
        b.run(2500.0, s._blank(2500.0))
        pre, on = [], []
        order = torch.arange(L).repeat(shows)
        order = order[torch.randperm(len(order), generator=gen)]
        for k in order.tolist():
            s.line = ""
            b.observe(50.0, s._blank(50.0))
            pre.append(window(100.0, s._blank(100.0)))
            s.line = letters[k]
            s.gaze = s.screen.width(letters[k]) / 2 + float(torch.randint(-1, 2, (1,), generator=gen))
            on.append(window(350.0, s._view(350.0)))
        print(f"\n=== {tag} ===")
        for name, ws in (("пауза", pre), ("буква", on)):
            rate = torch.stack([w["rate"] for w in ws]).mean(0).cpu()
            phi = torch.stack([w["phi"] for w in ws]).mean(0).cpu()
            above = torch.stack([w["above"] for w in ws]).mean(0).cpu()
            dv = torch.cat([w["dv"] for w in ws]).cpu()
            err = rate - phi
            q = torch.quantile(dv.flatten(), torch.tensor([0.05, 0.5, 0.95]))
            print(f"{name}: частота {float(rate.mean()):.2f} Гц, φ {float(phi.mean()):.2f} Гц, ошибка (частота − φ) "
                  f"{float(err.mean()):+.2f} Гц; клеток с ошибкой < 0: {float((err < 0).float().mean()):.0%}; "
                  f"ошибка по клеткам 5/50/95%: {' / '.join(f'{float(x):+.1f}' for x in torch.quantile(err, torch.tensor([0.05, 0.5, 0.95])))}")
            print(f"       V − v_t 5/50/95%: {' / '.join(f'{float(x):+.1f}' for x in q)} мВ; среднее {float(dv.mean()):+.1f}; "
                  f"доля времени выше v_t {float(above.mean()):.1%}")
            print("       по группам (частота / φ / ошибка): " + "  ".join(
                f"{letters[k]} {float(rate[group == k].mean()):.1f}/{float(phi[group == k].mean()):.1f}/{float(err[group == k].mean()):+.1f}"
                for k in range(L)))

    print(f"φ: us_max {cfg.us_max_khz * 1000:.0f} Гц, наклон {cfg.us_slope_mv} мВ, центр v_t{cfg.us_center_mv:+.2f}; v_t клеток "
          f"{float(vt.mean()):.1f} мВ; v_reset {cfg.v_reset}; кусок замера {chunk} мс", flush=True)
    measure("врождённые веса (начало учёбы)")
    if os.path.exists(load):
        path.weight.copy_(torch.load(load, map_location=dev)["weight"])
        measure(f"после учёбы ({os.path.basename(load)})")
    path.weight.copy_(w_innate)


if __name__ == "__main__":
    main(dict(kv.split("=", 1) for kv in sys.argv[1:]))
