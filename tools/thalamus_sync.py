"""
Синхронность реле таламуса и секторов TRN (CHAT_PLAN шаг 3, цикл 42):
торможение внутри TRN (cfg.trn_intra_sectors) во всех секторах против
только MD. Мозг целиком (organs по умолчанию — глаз, ухо, рука; пустой экран)
или словесный (organs=words); разгон 2.5 с, адаптация с гомеостазом окнами
по 1 с, запись без гомеостаза с шагом 1 мс.

По каждому реле (LGN, MGN, MD, таламус VA/VL, VL мозжечка) и сектору TRN:
частота, CV популяции по окнам 5 мс (у независимых ≈ 1/√(N·r·0.005)),
корреляция пар (окно 5 мс, до 200 клеток; у мышей наяву в MD 0.004–0.010 —
цикл 23), пик спектра популяции 5–200 Гц и пик / медиана. Плюс: ПФК L2/3 E
(CV поп.), CA3 PC — окна 5 мс с ≥ 50% PC.

    PYTHONPATH=. python tools/thalamus_sync.py [seed=42] [adapt=30] [record=3] [trn=trn_md] [organs=words]
"""

from __future__ import annotations

import sys

import torch

from brain.config import BrainConfig
from brain.mind import Mind
from tools import develop
from tools.pfc_isolate import pair_corr, spectrum


def main(opts: dict) -> None:
    cfg = BrainConfig()
    cfg.size = "small"
    cfg.seed = int(opts.get("seed", 42))
    cfg.autosave = False
    cfg.homeostasis = True
    cfg.event_overflow = "warn"
    if opts.get("organs"):
        cfg.organs = tuple(opts["organs"].split(","))
    if "trn" in opts:
        cfg.trn_intra_sectors = tuple(x for x in opts["trn"].split(",") if x)
    print(f"зерно {cfg.seed}, органы {cfg.organs}, торможение внутри TRN: {cfg.trn_intra_sectors}", flush=True)
    mind = Mind(cfg)
    b = mind.brain
    b.plasticity = False
    if "eye" in cfg.organs:
        from brain.session import Session

        s = Session(mind)
    else:
        s = develop.Silence(b)

    b.run(2500.0, s._blank(2500.0))
    for _ in range(int(opts.get("adapt", 30))):
        b.homeostasis_update(b.observe(1000.0, s._blank(1000.0)))
    b.cfg.homeostasis = False
    b.CHUNK = int(round(1.0 / b.cfg.dt_ms))
    b._graphs.clear()

    zones = [z for z in ("lgn", "trn", "mgn", "trn_mgn", "md", "trn_md", "thalamus", "vl_cb", "trn_motor") if z in b.areas]
    pops = {z: b.areas[z].E.slice for z in zones}
    pops["ПФК L2/3 E"] = b.areas["prefrontal"].layers["L23"]["E"].slice
    gen = torch.Generator().manual_seed(cfg.seed)
    pick = {}
    for z, sl in pops.items():
        n = sl.stop - sl.start
        pick[z] = (torch.randperm(n, generator=gen)[:200] + sl.start).to(b.device)
    ca3 = b.areas["ca3"].E.slice
    ms = int(float(opts.get("record", 3)) * 1000)
    pop = {z: torch.zeros(ms, device=b.device) for z in pops}
    spk = {z: torch.zeros(ms, len(pick[z]), device=b.device) for z in pops}
    ca3_frac = torch.zeros(ms, device=b.device)

    for t in range(ms):
        r = b.observe(1.0, s._blank(1.0))
        fired = r > 0
        for z, sl in pops.items():
            pop[z][t] = fired[sl].float().sum()
            spk[z][t] = fired[pick[z]].float()
        ca3_frac[t] = fired[ca3].float().mean()

    print("\nзона           Гц     CV поп. 5 мс (незав.)  корр. пар 5 мс   пик спектра (пик / медиана)")
    for z, sl in pops.items():
        n = sl.stop - sl.start
        y = pop[z]
        hz = float(y.sum()) / n / (ms / 1000)
        y5 = y[: ms // 5 * 5].view(-1, 5).sum(1)
        cv = float(y5.std() / y5.mean()) if float(y5.mean()) > 0 else float("nan")
        cv0 = 1.0 / max(1e-9, n * hz * 0.005) ** 0.5
        x5 = spk[z][: ms // 5 * 5].view(-1, 5, spk[z].shape[1]).sum(1)
        f, ratio = spectrum(y)
        print(f"{z:12s} {hz:6.2f}   {cv:5.2f} ({cv0:4.2f})          {pair_corr(x5):6.3f}          {f:5.0f} Гц (×{ratio:.1f})", flush=True)
    # Доля PC в окне 5 мс ≈ сумма долей по 1 мс (PC не стреляет дважды за 5 мс в покое).
    c5 = ca3_frac[: ms // 5 * 5].view(-1, 5).sum(1)
    print(f"CA3 PC: окон 5 мс с ≥ 50% PC — {int((c5 >= 0.5).sum())}, с ≥ 20% — {int((c5 >= 0.2).sum())} из {len(c5)}; наибольшая доля {float(c5.max()):.1%}")
    print(f"потеряно событий за прогон: мозг {float(b.dropped):.3g}, " + ", ".join(f"{k} {float(x.dropped):.3g}" for k, x in b.circuits.items()))


if __name__ == "__main__":
    main(dict(a.split("=", 1) for a in sys.argv[1:]))
