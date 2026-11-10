"""
Ответ клеток V1 L4 E мозга на штрих в своём поле (замер, без вмешательства;
мозг как в letters_dc, гомеостаз lgn-c). Вопрос: L4 отвечает на букву
слабо или кодирует её разреженно (какие клетки, а не сколько спайков).

Поле клетки L4 — её входы от реле LGN (путь lgn -> visual.L4.E), у реле —
водитель сетчатки. На показе буквы клетка L4:
  «в поле»   — у ≥ половины её входов водитель стреляет ≥ 2 × поддерживаемой
               (штрих своего знака в поле);
  «против»   — у ≥ половины водитель < ½ поддерживаемой (свет против знака);
  «вне»      — прочие.
По каждой группе (пары «клетка × показ»): частота в паузе перед показом и
на показе, отношение; по клеткам — доля с ответом вверх / вниз / без
(парная разность показ − пауза по своим показам в группе, |t| > 2, у
клетки ≥ 3 таких показов). Отдельно — клетки, у которых сдвиг гомеостаза
у предела (cfg.homeostasis_max_mv). Различение букв всей популяцией L4 E —
корреляцией, как в letters_dc.

    PYTHONPATH=. .venv/bin/python tools/l4_response.py [letters=абвгдежзик] [label=10] [test=10] [rest=300] [show=350] [seed=1]
"""

from __future__ import annotations

import sys
import time

import torch

from brain.config import BRAIN_HOME
from brain.session import Session
from tools.reading_bench import build


def main(opts: dict) -> None:
    seed = int(opts.get("seed", 1))
    opts = {"fit": "lgn-a", "homeo": f"lgn-c-s{seed}", "relay_v_t_mv": "-59", "trn_scale": "0.1", **opts}
    mind, b = build(opts)
    cfg = b.cfg
    s = Session(mind, window_ms=10.0)
    letters = opts.get("letters", "абвгдежзик")
    n_label, n_test = int(opts.get("label", 10)), int(opts.get("test", 10))
    rest, show = float(opts.get("rest", 300)), float(opts.get("show", 350))
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    lgn = b.areas["lgn"].E
    L4 = b.areas["visual"].layers["L4"]["E"]
    eye_i = [x.name for x in b.sensors].index("eye")
    driver = next(p for p in b.projections if p.name == "eye -> " + lgn.name).pre[:, 0].long().cpu() - b.sensor_offsets[eye_i]
    tc = next(p for p in b.projections if p.name == f"lgn -> {L4.name}")
    tc_pre = (tc.pre.long() - lgn.start).cpu()                          # [клеток L4, входов]
    shift = torch.load(BRAIN_HOME / "fits" / f"{opts['homeo']}-homeo.pt")[L4.slice]
    at_limit = shift.abs() >= cfg.homeostasis_max_mv - 1e-3
    m = cfg.retina_maintained_hz
    print(f"V1 L4 E: {L4.size} клеток, входов LGN {tc_pre.shape[1]}; сдвиг гомеостаза у предела {float(at_limit.float().mean()):.0%}",
          flush=True)

    s.line, s.gaze = "", 0.0
    b.run(2500.0, s._blank(2500.0))
    t0 = time.time()
    R0, R1, G, Y = [], [], [], []
    for rep in range(n_label + n_test):
        for k in torch.randperm(len(letters), generator=gen).tolist():
            ch = letters[k]
            s.line = ""
            r0 = b.observe(rest, s._blank(rest))[L4.slice].cpu()
            s.line, s.gaze = ch, s.screen.width(ch) / 2 + float(torch.randint(-1, 2, (1,), generator=gen))
            rates = s.eye.rates(s._picture(s.screen, cfg.fovea_col))[driver]   # водитель каждого реле, Гц
            r1 = b.observe(show, s._view(show))[L4.slice].cpu()
            s.line = ""
            d = rates[tc_pre]
            strong = (d >= 2 * m).float().mean(1) >= 0.5
            against = (d < 0.5 * m).float().mean(1) >= 0.5
            g = torch.full((L4.size,), 2, dtype=torch.long)
            g[against] = 1
            g[strong] = 0
            R0.append(r0)
            R1.append(r1)
            G.append(g)
            Y.append(k)
        print(f"  проход {rep + 1}/{n_label + n_test}, {time.time() - t0:.0f} с", flush=True)

    R0, R1, G, y = torch.stack(R0), torch.stack(R1), torch.stack(G), torch.tensor(Y)
    print(f"\nответ V1 L4 E по группам (пары клетка × показ; пауза → показ, Гц):")
    for gi, name in enumerate(("в поле", "против", "вне")):
        for lim_name, cells in (("все", torch.ones(L4.size, dtype=torch.bool)), ("у предела", at_limit),
                                ("не у предела", ~at_limit)):
            mask = (G == gi) & cells[None]
            if not bool(mask.any()):
                continue
            a, c = R0[mask], R1[mask]
            # По клеткам: свои показы в группе.
            n = mask.sum(0)
            dif = torch.where(mask, R1 - R0, 0.0)
            mean = dif.sum(0) / n.clamp(min=1)
            var = (torch.where(mask, (R1 - R0 - mean) ** 2, 0.0)).sum(0) / (n - 1).clamp(min=1)
            t = mean / (var / n.clamp(min=1)).sqrt().clamp_min(1e-9)
            ok = n >= 3
            up = float(((t > 2) & ok).sum()) / max(int(ok.sum()), 1)
            down = float(((t < -2) & ok).sum()) / max(int(ok.sum()), 1)
            print(f"  {name:7s} {lim_name:13s} пар {int(mask.sum()):7d}: {float(a.mean()):6.2f} → {float(c.mean()):6.2f} "
                  f"(×{float(c.mean()) / max(float(a.mean()), 1e-9):.2f}); молчат на показе {float((c == 0).float().mean()):.0%}; "
                  f"клеток ≥3 показов {int(ok.sum())}: вверх {up:.0%}, вниз {down:.0%}", flush=True)

    lab = torch.arange(len(y)) < n_label * len(letters)
    z = lambda x: (x - x.mean(-1, keepdim=True)) / x.std(-1, keepdim=True).clamp_min(1e-9)
    print(f"\nразличение букв популяцией L4 E (корреляция; метки {n_label}, проверка {n_test} показов; случайно {1 / len(letters):.2f}):")
    for name, cells in (("все", torch.ones(L4.size, dtype=torch.bool)), ("у предела", at_limit), ("не у предела", ~at_limit)):
        X = R1[:, cells]
        mean = torch.stack([X[lab][y[lab] == k].mean(0) for k in range(len(letters))])
        acc = float(((z(X[~lab]) @ z(mean).T).argmax(1) == y[~lab]).float().mean())
        print(f"  {name:13s} {int(cells.sum()):5d} клеток: {acc:.2f}", flush=True)
    print(f"всего {time.time() - t0:.0f} с")


if __name__ == "__main__":
    main(dict(kv.split("=", 1) for kv in sys.argv[1:]))
