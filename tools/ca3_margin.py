"""
Запас устойчивости сети CA3 (Kopsick, brain/circuit.py) к превышению входа
EC над покоем — отдельно от мозга (CHAT_PLAN шаг 3, цикл 9).

В мозге в CA3 идёт фон авторов (их покой целиком) + (вход мозга − вход при
частотах покоя) × множитель мВ → пА. Здесь: вход при покое (rest_mv) и его
доля от EC, множитель (gain) — из настоящего словесного мозга (сборка на GPU,
те же зерно и проводка CA3); затем мозг удаляется, и сеть CA3 (та же статистика
проводки, свой экземпляр — генератор CPU) живёт одна на GPU (если свободна; иначе CPU, 2 потока) с входом rest_mv + (k − 1) × rest_EC —
постоянный ток (без колебаний EC). k = 1 — покой авторов (проверка эталона),
k = 0 — EC молчит.

Меры по k: частоты типов, CV популяционной частоты PC по окнам 5 мс,
корреляция пар PC (окно 5 мс), доля окон-залпов (> 5× средней) и доля PC в
залпе.

trace=путь.pt — вход EC с волнами мозга (tools/dg_sync.py save_ec=): доля EC
× k × r(t)/r̄ по окнам 5 мс.

    PYTHONPATH=. python tools/ca3_margin.py [seed=42] [ms=2000] [k=0,1,1.1,1.2,1.3,1.5,2] [trace=ec.pt] [bgvar=1]
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import torch

from brain import anatomy, circuit
from brain.config import BrainConfig
from brain.mind import Mind


def rest_by_source(b, area) -> dict[str, torch.Tensor]:
    """Вход при частотах покоя по источникам (как engine._rest_input), мВ на клетку зоны."""

    out: dict[str, torch.Tensor] = {}
    cells = area.cells
    tau = b.tau_m

    def add(name, sl, value):
        out.setdefault(name, torch.zeros(b.n_slots, device=b.device))[sl] += value

    for p in b.projections:
        if p.source != "net" or p.post.start < cells.start or p.post.stop > cells.stop or not hasattr(p, "pre"):
            continue
        try:
            first = int(p.pre.flatten()[0])
        except (TypeError, AttributeError):
            continue
        src = next(a for a in b.areas.values() if a.cells.start <= first < a.cells.stop)
        hz = anatomy.RATES_HZ.get(src.name, 1.0)
        w = p.weight.float().clamp(min=0.0).sum(1)
        add(src.name, p.post.slice, w * hz / 1000.0 * tau[p.post.slice])

    for post, pre, coeff, rate, _, _ in b.mean_field:
        if cells.start <= post.start < cells.stop and coeff > 0:
            src = next(a for a in b.areas.values() if a.cells.start <= pre.start < a.cells.stop)
            add(src.name, post.slice, torch.full((post.size,), coeff * rate, device=b.device))

    return {k: v[cells].cpu() for k, v in out.items()}


def main(opts: dict) -> None:
    seed = int(opts.get("seed", 42))
    cfg = BrainConfig()
    cfg.size = "small"
    cfg.seed = seed
    cfg.autosave = False
    cfg.organs = ("words",)
    b = Mind(cfg).brain
    area = b.areas["ca3"]
    c_brain = b.circuits["ca3"]
    by_src = rest_by_source(b, area)
    rest = c_brain.rest_mv.cpu().clone()
    gain = c_brain.gain.cpu().clone()
    spec = next(s for s in cfg.areas if s.name == "ca3")
    E = SimpleNamespace(start=area.E.start, size=area.E.size, stop=area.E.stop)
    I = SimpleNamespace(start=area.I.start, size=area.I.size, stop=area.I.stop)
    dt = cfg.dt_ms
    del b, c_brain
    torch.cuda.empty_cache()

    print(f"зерно {seed}: вход CA3 при покое по источникам (среднее по клеткам, мВ):",
          ", ".join(f"{k} {float(v.mean()):.1f}" for k, v in sorted(by_src.items(), key=lambda kv: -float(kv[1].mean()))),
          f"| всего {float(rest.mean()):.1f}", flush=True)
    ec = by_src.get("entorhinal", torch.zeros_like(rest))

    # На GPU, если свободна (мозг уже удалён): на CPU 23 млн синапсов не
    # укладываются в короткий прогон.
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(2)
    gen = torch.Generator(device=dev).manual_seed(seed)
    net = circuit.load(spec.circuit)
    c = circuit.TypedCircuit(net, SimpleNamespace(E=E, I=I), dt, dev, gen, compact=dev.type == "cuda", capacity=0.02)
    # bgvar= — разброс фона авторов вокруг среднего (1 — как у авторов).
    c.bg_scale = float(opts.get("bgvar", 1.0))
    c.gain = gain.to(dev)
    c.rest_mv = rest.to(dev)
    ec_dev = by_src.get("entorhinal", torch.zeros_like(rest)).to(dev)
    steps = int(float(opts.get("ms", 2000)) / dt)
    per_bin = max(1, round(5.0 / dt))
    pc = c.type_of == 0
    n_pc = int(pc.sum())
    sample = torch.randperm(n_pc, generator=torch.Generator().manual_seed(1))[:300].to(dev)
    print(f"сеть CA3 на {dev}: клеток {c.n}, синапсов {c.synapses}; {steps} шагов по {dt} мс", flush=True)

    for k in [float(x) for x in opts.get("k", "0,1,1.1,1.2,1.3,1.5,2").split(",")]:
        c.reset_state()
        ext = c.rest_mv + (k - 1.0) * ec_dev
        # trace= — частота EC мозга по окнам 5 мс: доля EC входа идёт × r(t)/r̄
        # (средний вход тот же, волны — мозга); k — множитель среднего.
        wave = None
        if opts.get("trace"):
            r = torch.load(opts["trace"]).float().to(dev)
            wave = r / r.mean()
        fired = torch.zeros(c.n, dtype=torch.bool, device=dev)
        counts = torch.zeros(len(c.names), device=dev)
        bins = steps // per_bin
        pop = torch.zeros(bins, device=dev)
        cells = torch.zeros(bins, len(sample), device=dev)
        skip = int(200 / dt)
        for t in range(steps):
            if wave is not None:
                ext = c.rest_mv + (k * float(wave[(t // per_bin) % len(wave)]) - 1.0) * ec_dev
            _, fired = c.step(ext, fired)
            if t < skip:
                continue
            counts.index_add_(0, c.type_of, fired.float())
            j = (t - skip) // per_bin
            if j < bins:
                pop[j] += fired[pc].sum()
                cells[j] += fired[pc][sample].float()
        sec = (steps - skip) * dt / 1000.0
        bins = (steps - skip) // per_bin
        pop, cells = pop[:bins], cells[:bins]
        rates = {g: float(counts[i]) / c.size[g] / sec for i, g in enumerate(c.names)}
        m = float(pop.mean())
        cv = float(pop.std() / pop.mean()) if m > 0 else float("nan")
        burst = pop > 5 * m if m > 0 else torch.zeros_like(pop, dtype=torch.bool)
        part = float((pop[burst] / n_pc).mean()) if bool(burst.any()) else 0.0
        x = cells - cells.mean(0)
        sd = x.std(0)
        live = sd > 0
        corr = float("nan")
        if int(live.sum()) > 2:
            x = x[:, live] / sd[live]
            cc = (x.T @ x) / (x.shape[0] - 1)
            corr = float(cc[~torch.eye(cc.shape[0], dtype=torch.bool, device=cc.device)].mean())
        print(f"k {k:4.2f} (EC +{float(((k - 1) * ec_dev)[pc].mean()):6.2f} мВ у PC): PC {rates[c.names[0]]:6.2f} Гц, CV {cv:5.2f}, "
              f"корр. пар {corr:6.3f}, залпов {float(burst.float().mean()):5.1%} окон по {part:5.1%} PC | "
              + " ".join(f"{g.replace('CA3_', '')} {r:.1f}" for g, r in list(rates.items())[1:]), flush=True)


if __name__ == "__main__":
    main(dict(a.split("=", 1) for a in sys.argv[1:]))
