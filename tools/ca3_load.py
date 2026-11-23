"""
Нагрузка CA3 по источникам и перерезки входов EC / DG в одном мозге (CHAT_PLAN
шаг 3, цикл 31, пункты Б и В).

Фон сети CA3 (Kopsick et al. 2024) — по статье «lognormal background current to
model the upstream inputs CA3 receives … from dentate gyrus and entorhinal
cortex», подобран под частоты типов in vivo (препринт
https://doi.org/10.1101/2024.03.27.586909, Methods «Full-scale CA3 SNN»).
В мозге явные пути EC → CA3 и DG → CA3 идут сверх своей величины при частотах
покоя (circuit.set_rest_input): по среднему не задвоены, по колебаниям —
добавлены к колебаниям фона.

Мозг — как в tools/chain_sync.py (organs=words): разгон 2.5 с, адаптация с
гомеостазом adapt × 1 с, затем без гомеостаза записи по record с подряд:
  всё → без EC → без DG → без EC и DG (только фон и прочие входы) → всё снова
(шум прогона: другой шум, то же состояние). Каждое условие начинается с одного
снимка состояния после адаптации (engine._snapshot). Перерезка: веса построенных синапсов источника → CA3 и его
среднее поле = 0, вход покоя зоны уменьшен на долю источника (по ней
вычитался) — так CA3 получает ровно фон авторов вместо пути. Перед записью
settle мс после перерезки (и без неё).

Меры по записи (PC CA3):
- вход мозга сверх покоя, пА (= (ext − rest) × gain, что идёт в ток клетки):
  среднее, σ во времени у клетки, σ общей части (среднее по PC за 1 мс) —
  против фона авторов (среднее и σ за 1 мс у PC);
- оценка по источникам: вход покоя пути × (частота источника / покоя − 1),
  пА (линейно, без депрессии синапсов) — что тянет вверх;
- доля времени PC у порога (v > vt − 5 мВ, vt — порог Izhikevich), частоты
  типов, CV популяции PC по окнам 5 мс, корреляция пар, доля окон-залпов
  (> 5× средней).

    PYTHONPATH=. python tools/ca3_load.py [seed=42] [adapt=30] [record=3] [settle=500] [extpop=…]
"""

from __future__ import annotations

import math
import sys

import torch

from brain import anatomy
from brain.config import BrainConfig
from brain.mind import Mind
from tools import develop
from tools.ca3_margin import rest_by_source

SOURCES = {"EC": "entorhinal", "DG": "dentate"}


class Cut:
    """Перерезка путей источника → CA3 (построенные синапсы, среднее поле, вход покоя)."""

    def __init__(self, b, src: str, rest_src: torch.Tensor):
        area, ca3 = b.areas[src], b.areas["ca3"].cells
        self.b, self.c, self.rest = b, b.circuits["ca3"], rest_src.to(b.device)
        inside = lambda pop: area.cells.start <= pop.start < area.cells.stop
        self.weights = []
        for p in b.projections:
            if p.source != "net" or not hasattr(p, "pre") or p.post.start < ca3.start or p.post.stop > ca3.stop:
                continue
            try:
                first = int(p.pre.flatten()[0])
            except (TypeError, AttributeError):
                continue
            if area.cells.start <= first < area.cells.stop:
                self.weights.append(p)
        self.cells = []
        for post, pre, *_ in b.mean_field:
            if ca3.start <= post.start < ca3.stop and inside(pre):
                q, i = int(b.mf_post_id[post.start]), int(b.mf_pre_id[pre.start])
                self.cells.append((q, i))
        self.saved = None

    def apply(self) -> None:
        mats = (self.b.mf_exc, self.b.mf_inh, self.b.mf_ach, self.b.mf_slow)
        self.saved = ([p.weight.clone() for p in self.weights],
                      [[float(m[q, i]) for m in mats] for q, i in self.cells])
        for p in self.weights:
            p.weight.zero_()
        for q, i in self.cells:
            for m in mats:
                m[q, i] = 0.0
        self.c.rest_mv.sub_(self.rest)

    def undo(self) -> None:
        mats = (self.b.mf_exc, self.b.mf_inh, self.b.mf_ach, self.b.mf_slow)
        ws, vals = self.saved
        for p, w in zip(self.weights, ws):
            p.weight.copy_(w)
        for (q, i), v in zip(self.cells, vals):
            for m, x in zip(mats, v):
                m[q, i] = x
        self.c.rest_mv.add_(self.rest)


def record(b, s, ms: float, by_src: dict, pc: slice, sample: torch.Tensor) -> dict:
    """Запись ms мс: вход CA3, порог, частоты; c.step обёрнут на время записи."""

    c = b.circuits["ca3"]
    n_pc = pc.stop - pc.start
    dev0 = b.device
    # Шаг мозга записан в CUDA-граф: накопители — на месте (суммы и суммы
    # квадратов), обёртка попадает в граф при его пересъёмке (b._graphs = {}).
    z = lambda *shape: torch.zeros(*shape, device=dev0)
    acc = {"dev": z(1), "dev2": z(1), "dc": z(len(sample)), "dc2": z(len(sample)),
           "bc": z(len(sample)), "bc2": z(len(sample)), "bg": z(1), "near": z(1), "steps": z(1),
           "inh": z(len(c.names)), "inh2": z(len(c.names))}
    n_type = torch.zeros(len(c.names), device=dev0).index_add_(0, c.type_of, torch.ones(c.n, device=dev0))
    vt = c.vt[pc]
    step = c.step

    def wrapped(ext_mv, fired_prev):
        out = step(ext_mv, fired_prev)
        d = ((ext_mv - c.rest_mv) * c.gain)[pc]
        m = d.mean()
        acc["dev"].add_(m)
        acc["dev2"].add_(m * m)
        ds = d[sample]
        acc["dc"].add_(ds)
        acc["dc2"].add_(ds * ds)
        bg = c.ext[pc]
        acc["bg"].add_(bg.mean())
        bs = bg[sample]
        acc["bc"].add_(bs)
        acc["bc2"].add_(bs * bs)
        acc["near"].add_((c.v[pc] > vt - 5.0).float().mean())
        acc["steps"].add_(1.0)
        # Вход сверх покоя по типам (среднее по типу за шаг).
        per = torch.zeros(len(c.names), device=dev0).index_add_(0, c.type_of, (ext_mv - c.rest_mv) * c.gain) / n_type
        acc["inh"].add_(per)
        acc["inh2"].add_(per * per)
        return out

    c.step = wrapped
    b._graphs = {}
    bins = int(ms / 5)
    series = torch.zeros(bins, device=b.device)
    cells = torch.zeros(bins, len(sample), device=b.device)
    counts = torch.zeros(len(c.names), device=b.device)
    src_spk = {k: 0.0 for k in SOURCES}
    try:
        for t in range(bins):
            r = b.observe(5.0, s._blank(5.0)) * 0.005
            x = r[b.areas["ca3"].cells]
            counts.index_add_(0, c.type_of, x)
            series[t] = x[pc].sum()
            cells[t] = x[pc][sample]
            for k, name in SOURCES.items():
                src_spk[k] += float(r[b.areas[name].E.slice].sum())
    finally:
        del c.step
        b._graphs = {}
    sec = ms / 1000.0
    n = float(acc["steps"])
    if n == 0:
        raise RuntimeError("шаг сети CA3 не вызывался — обёртка не попала в шаг мозга")
    sd = lambda s1, s2: ((s2 / n - (s1 / n) ** 2).clamp(min=0.0)).sqrt()
    m = float(series.mean())
    x = cells - cells.mean(0)
    sdx = x.std(0)
    live = sdx > 0
    corr = float("nan")
    if int(live.sum()) > 2:
        x = x[:, live] / sdx[live]
        cc = (x.T @ x) / (x.shape[0] - 1)
        corr = float(cc[~torch.eye(cc.shape[0], dtype=torch.bool, device=cc.device)].mean())
    rates = {g.replace("CA3_", ""): float(counts[i]) / int((c.type_of == i).sum()) / sec for i, g in enumerate(c.names)}
    src_hz = {k: src_spk[k] / b.areas[SOURCES[k]].E.size / sec for k in SOURCES}
    gain_pc = float(c.gain[pc].mean())
    est = {k: float(by_src.get(SOURCES[k], torch.zeros(1))[: n_pc].mean()) * gain_pc
           * (src_hz[k] / anatomy.RATES_HZ[SOURCES[k]] - 1.0) for k in SOURCES}
    return {
        "dev": float(acc["dev"]) / n, "dev_sd_cell": float(sd(acc["dc"], acc["dc2"]).mean()),
        "dev_sd_common": float(sd(acc["dev"], acc["dev2"])),
        "bg": float(acc["bg"]) / n, "bg_sd_cell": float(sd(acc["bc"], acc["bc2"]).mean()),
        "near": float(acc["near"]) / n,
        "types_dev": {g.replace("CA3_", ""): (float(acc["inh"][i]) / n, float(sd(acc["inh"][i], acc["inh2"][i])))
                      for i, g in enumerate(c.names)}, "rates": rates, "src_hz": src_hz, "est": est,
        "cv": float(series.std() / series.mean()) if m > 0 else float("nan"), "corr": corr,
        "burst": float((series > 5 * m).float().mean()) if m > 0 else 0.0,
    }


def main(opts: dict) -> None:
    develop.override_paths(opts.get("path", ""))
    cfg = BrainConfig()
    cfg.size = "small"
    cfg.seed = int(opts.get("seed", 42))
    cfg.autosave = False
    cfg.homeostasis = True
    cfg.event_overflow = "warn"
    cfg.organs = ("words",)
    develop.override_external(cfg, opts.get("extpop", ""))
    b = Mind(cfg).brain
    b.plasticity = False
    s = develop.Silence(b)

    area = b.areas["ca3"]
    by_src = {k: v.to(b.device) for k, v in rest_by_source(b, area).items()}
    c = b.circuits["ca3"]
    pc_n = int((c.type_of == 0).sum())
    pc = slice(0, pc_n)
    gain_pc = float(c.gain[pc].mean())
    print(f"зерно {cfg.seed}{', внешний фон ' + opts['extpop'] if opts.get('extpop') else ''}; "
          f"вход CA3 PC при частотах покоя по источникам, пА (мВ × gain {gain_pc:.1f}): "
          + ", ".join(f"{k} {float(v[:pc_n].mean()) * gain_pc:.0f}"
                      for k, v in sorted(by_src.items(), key=lambda kv: -float(kv[1][:pc_n].mean()))), flush=True)
    mu, sg = float(c.mu[0]), float(c.sigma[0])
    print(f"фон авторов у PC: среднее {math.exp(mu + sg * sg / 2) / 2:.0f} пА, σ за 1 мс "
          f"{math.sqrt(math.exp(2 * mu + 2 * sg * sg) / 3 - (math.exp(mu + sg * sg / 2) / 2) ** 2):.0f} пА "
          "(lognormal × U(0, 1), без потолка)", flush=True)

    b.run(2500.0, s._blank(2500.0))
    for _ in range(int(opts.get("adapt", 30))):
        b.homeostasis_update(b.observe(1000.0, s._blank(1000.0)))
    b.cfg.homeostasis = False
    shift = getattr(b, "homeo_shift", None)
    if shift is not None:
        print(f"гомеостаз: сдвиг CA3 {float(shift[area.cells].mean()):+.1f} мВ, EC "
              f"{float(shift[b.areas['entorhinal'].cells].mean()):+.1f} мВ", flush=True)

    sample = torch.randperm(pc_n, generator=torch.Generator().manual_seed(1))[:300].to(b.device)
    cuts = {k: Cut(b, SOURCES[k], by_src.get(SOURCES[k], torch.zeros_like(c.rest_mv))) for k in SOURCES}
    ms = float(opts.get("record", 3)) * 1000.0
    settle = float(opts.get("settle", 500))
    plan = [("всё", ()), ("без EC", ("EC",)), ("без DG", ("DG",)), ("без EC и DG", ("EC", "DG")), ("всё снова", ())]
    # Каждое условие — с одного снимка состояния после адаптации (подряд
    # идущие записи уплывали: повтор «всё» в конце давал PC 28 Гц против 8).
    snap = b._snapshot()
    for name, cut in plan:
        b._restore(snap)
        for k in cut:
            cuts[k].apply()
        b.observe(settle, s._blank(settle))
        r = record(b, s, ms, by_src, pc, sample)
        for k in cut:
            cuts[k].undo()
        print(f"{name:12s} PC {r['rates']['Pyramidal']:5.2f} Гц, CV {r['cv']:4.2f}, корр. {r['corr']:.3f}, "
              f"залпов {r['burst']:5.1%} окон; у порога {r['near']:5.1%} | вход сверх покоя {r['dev']:+6.1f} пА "
              f"(σ у клетки {r['dev_sd_cell']:.0f}, общая {r['dev_sd_common']:.1f}); фон {r['bg']:.0f} "
              f"(σ {r['bg_sd_cell']:.0f}) | EC {r['src_hz']['EC']:.2f} Гц ~{r['est']['EC']:+.0f} пА, "
              f"DG {r['src_hz']['DG']:.2f} Гц ~{r['est']['DG']:+.0f} пА | "
              + " ".join(f"{g} {v:.1f}" for g, v in list(r["rates"].items())[1:]), flush=True)
        print("             вход сверх покоя по типам, пА (среднее / σ общей): " + " ".join(
            f"{g} {m:+.0f}/{v:.0f}" for g, (m, v) in r["types_dev"].items()), flush=True)


if __name__ == "__main__":
    main(dict(a.split("=", 1) for a in sys.argv[1:]))
