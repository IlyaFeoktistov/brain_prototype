"""
Стенд звена LGN → V1 L4: влияет ли синхронность спайков LGN на ответ L4 при
том же числе спайков.

Мозг целиком (small, фон fit=, LGN — опции BrainConfig) пишет 1.5 с пустого
экрана и 0.5 с слова: спайки всех клеток, что шлют входы в L4 (E и PV) извне
— LGN, L2/3, L5, L6 и прочее. Затем в том же мозге L4 считается заново с
входами, зажатыми по записи (engine.clamp_*), а спайки LGN — в одном из
вариантов (у каждой клетки LGN число спайков то же):

    запись     как записано;
    разброс    каждый спайк сдвинут случайно на ±jitter мс (синхронность ↓);
    сжатие=Δ   спайк сдвинут к началу своего окна Δ мс — спайки разных клеток
               в одном окне совпадают (синхронность ↑);
    без=X      источник X (lgn, L6, L23, L5 — возбуждающие) молчит весь прогон:
               разность тока с записью — вклад источника в работающей сети
               (не прямой: без него меняется и сама L4).

L4 — связанная сеть (E ↔ E, PV тормозит всех), поэтому вариант — отдельный
прогон всего L4, а не группа клеток.

Меры (окна 20 мс, клетки поля — больше половины входов LGN от клеток, чей
водитель на слове чаще пустого на 20 Гц): частота поля L4 E и прочих L4 E,
L4 PV, V поля, ток возбуждения и торможения поля (i_syn, i_inh), ресурс
синапсов LGN — входов поля (stp_x; у варианта — от его же спайков); синхронность
входов LGN на клетке поля — доля входных спайков, у которых другой вход той же
клетки в ±1 мс, и та же доля для случайного сдвига ±20 мс (уровень случая).
Сверка: кошка — соседние клетки LGN часто синхронны до 1 мс (Alonso, Usrey &
Reid 1996, https://doi.org/10.1038/383815a0); L4 гонит синхронный вход
таламуса (крыса, in vivo — Bruno & Sakmann 2006,
https://doi.org/10.1126/science.1124593).

    PYTHONPATH=. python tools/l4_link_bench.py [seeds=1,2,3] [word=мама] [fit=lgn-a] [homeo=имя]
        [relay_v_t_mv=-59] [relay_rest_mv=-48] [trn_scale=0.5] [variants=запись,разброс=10,сжатие=5,сжатие=10]
"""

from __future__ import annotations

import sys

import torch

from brain import anatomy
from brain.config import BrainConfig
from brain.mind import Mind
from brain.session import Session

BLANK_MS, WORD_MS, SETTLE_MS, WIN_MS = 1500.0, 500.0, 500.0, 20.0


def shifted(spikes: torch.Tensor, variant: str, dt: float, g: torch.Generator) -> torch.Tensor:
    """Растр [шаги × клетки] с теми же числами спайков у клеток, время — по варианту."""

    if variant == "запись":
        return spikes
    kind, _, val = variant.partition("=")
    t, c = spikes.nonzero(as_tuple=True)
    steps = spikes.shape[0]

    if kind == "разброс":
        k = int(round(float(val) / dt))
        t = (t + torch.randint(-k, k + 1, t.shape, generator=g)).clamp(0, steps - 1)
    elif kind == "сжатие":
        k = int(round(float(val) / dt))
        t = t // k * k
    else:
        raise ValueError(variant)
    out = torch.zeros_like(spikes)
    # Совпавшие во времени спайки одной клетки — на соседний шаг (число то же).
    for _ in range(8):
        taken = out[t, c]
        out[t[~taken], c[~taken]] = True
        t, c = t[taken], c[taken]
        if not len(t):
            break
        t = (t + 1).clamp(max=steps - 1)
    return out


def coincidence(raster: torch.Tensor, inputs: torch.Tensor, dt: float) -> float:
    """Доля входных спайков LGN, у которых другой вход той же клетки поля — в ±1 мс."""

    k = int(round(1.0 / dt))
    r = raster.float()
    near = r.clone()
    for lag in range(1, k + 1):
        near[lag:] += r[:-lag]
        near[:-lag] += r[lag:]
    hit, total = 0.0, 0.0
    for row in inputs:                       # входы одной клетки поля (номера клеток LGN)
        s, n = r[:, row], near[:, row]
        sum_s, sum_n = s.sum(1), n.sum(1)
        # Для каждого спайка входа i — есть ли спайк другого входа в ±1 мс.
        other = sum_n[:, None] - n
        hit += float((s * (other > 0)).sum())
        total += float(s.sum())
    return hit / max(total, 1.0)


def run_seed(seed: int, opts: dict, variants: list[str]):
    cfg = BrainConfig()
    cfg.size = "small"
    cfg.seed = seed
    cfg.autosave = False
    for k in ("relay_v_t_mv", "relay_rest_mv"):
        if k in opts:
            setattr(cfg, k, float(opts[k]))
    if opts.get("fit"):
        anatomy.use_fit(cfg, opts["fit"])
    mind = Mind(cfg)
    b = mind.brain
    b.plasticity = False
    for p in b.projections:
        if p.name == "trn -> lgn":
            p.weight.mul_(float(opts.get("trn_scale", 1.0)))
    if opts.get("homeo"):
        # Сдвиги гомеостаза после развития (tools/develop.py out=имя) — заморожены.
        from brain.config import BRAIN_HOME

        b.bias[: b.n] += torch.load(BRAIN_HOME / "fits" / f"{opts["homeo"]}-homeo.pt").to(b.device)
    torch.manual_seed(seed)
    b.run(2500.0)
    b.CHUNK, b._graphs = 1, {}
    dt = cfg.dt_ms
    lgn = b.areas["lgn"].E
    l4 = b.areas["visual"].layers["L4"]
    l4e, l4pv = l4["E"], l4["PV"]
    inside = torch.zeros(b.n_slots, dtype=torch.bool, device=b.device)
    for pop in (l4e, l4pv):
        inside[pop.slice] = True
    inputs = [p for p in b.projections if p.post in (l4e, l4pv) and p.source == "net"]
    pre = torch.unique(torch.cat([p.pre.long().flatten() for p in inputs]))
    pre = pre[~inside[pre]]
    in_lgn = (pre >= lgn.start) & (pre < lgn.start + lgn.size)
    eye_i = [x.name for x in b.sensors].index("eye")
    off, size = b.sensor_offsets[eye_i], b.sensors[eye_i].size
    spont = float(b.spontaneous[off])
    b.spontaneous[off: off + size] = 0.0

    s = Session(mind)
    s.gaze = 0.0
    n_blank, n_word = int(BLANK_MS / dt), int(WORD_MS / dt)
    views, rec = [], []
    for t in range(n_blank + n_word):
        s.line = opts.get("word", "мама") + " " if t >= n_blank else ""
        view = s._view(dt)
        retina = view[:, off: off + size]
        retina.copy_(((retina + (torch.rand_like(retina) < spont).float()) > 0).float())
        b.run(dt, view)
        views.append(view.clone())
        rec.append((b.spikes[pre] > 0).cpu())
    rec = torch.stack(rec)

    # Клетки поля: больше половины входов LGN — от клеток, чей водитель на слове чаще пустого на 20 Гц.
    driver = next(p for p in b.projections if p.name == "eye -> " + lgn.name)
    retina = torch.stack([v[0, off: off + size].cpu() for v in views]).float()
    gain = (retina[n_blank:].mean(0) - retina[:n_blank].mean(0)) * 1000 / dt
    sees = gain[(driver.pre[:, 0].long() - off).cpu()] >= 20.0
    tc = next(p for p in b.projections if p.name == f"lgn -> {l4e.name}")
    tc_pre = (tc.pre.long() - lgn.start).cpu()
    field = sees[tc_pre].float().mean(1) > 0.5
    field_idx = field.nonzero().squeeze(1)
    other_idx = (~field).nonzero().squeeze(1)
    # Входы LGN клеток поля — как столбцы растра pre.
    col_of = torch.full((lgn.size,), -1, dtype=torch.long)
    lgn_cols = in_lgn.cpu().nonzero().squeeze(1)
    col_of[(pre[in_lgn] - lgn.start).cpu()] = torch.arange(len(lgn_cols))
    field_inputs = col_of[tc_pre[field_idx]]
    # Клетки LGN — входы поля (для ресурса синапсов).
    field_lgn = (torch.unique(tc_pre[field_idx]) + lgn.start).to(b.device)

    b.clamp_on = True
    b.clamp_mask[pre] = True
    b.recompile()
    g = torch.Generator().manual_seed(seed)
    win = int(WIN_MS / dt)
    out = {}

    vis = b.areas["visual"].layers
    sources = {"lgn": lgn, "L6": vis["L6"]["E"], "L23": vis["L23"]["E"], "L5": vis["L5"]["E"]}

    for variant in variants:
        kind, _, val = variant.partition("=")
        clamp = rec.clone()
        if kind == "без":
            # Источник молчит весь прогон (его спайки и среднее поле — 0).
            src = sources[val]
            cols = ((pre >= src.start) & (pre < src.start + src.size)).cpu()
            clamp[:, cols] = False
            lgn_r = clamp[:, lgn_cols]
        else:
            lgn_r = shifted(rec[:, lgn_cols], variant, dt, g)
            clamp[:, lgn_cols] = lgn_r
        clamp = clamp.to(b.device)
        settle = int(SETTLE_MS / dt)
        trace = []
        for t in range(n_blank + n_word):
            b.clamp_now[pre] = clamp[t]
            b.run(dt, views[t])
            if t >= settle:
                trace.append(torch.stack([
                    b.spikes[l4e.start + field_idx.to(b.device)].float().mean(),
                    b.spikes[l4e.start + other_idx.to(b.device)].float().mean(),
                    b.spikes[l4pv.slice].float().mean(),
                    b.v[l4e.start + field_idx.to(b.device)].mean(),
                    b.i_syn[l4e.start + field_idx.to(b.device)].mean(),
                    b.i_inh[l4e.start + field_idx.to(b.device)].mean(),
                    b.stp_x[field_lgn].mean(),
                ]).cpu())
        trace = torch.stack(trace)
        n = trace.shape[0] // win * win
        w = trace[:n].view(-1, win, trace.shape[1]).mean(1)
        w[:, :3] *= 1000 / dt
        blank_w = (n_blank - settle) // win
        sync = {}
        for part, (a, z) in (("пусто", (settle, n_blank)), ("слово", (n_blank, n_blank + n_word))):
            sync[part] = coincidence(lgn_r[a:z], field_inputs, dt)
            sync[part + " случай"] = coincidence(shifted(lgn_r[a:z], "разброс=20", dt, g), field_inputs, dt)
        out[variant] = (w[:blank_w], w[blank_w:], sync)
    return out, len(field_idx)


def main() -> None:
    opts = dict(kv.split("=", 1) for kv in sys.argv[1:] if not kv.startswith("variants="))
    variants = next((kv.split("=", 1)[1] for kv in sys.argv[1:] if kv.startswith("variants=")),
                    "запись,разброс=10,сжатие=5,сжатие=10").split(",")
    seeds = [int(x) for x in opts.pop("seeds", "1,2,3").split(",")]
    opts.setdefault("fit", "lgn-a")
    opts.setdefault("relay_v_t_mv", "-59")
    opts.setdefault("relay_rest_mv", "-48")
    opts.setdefault("trn_scale", "0.5")
    names = ("поле L4 E, Гц", "прочие L4 E, Гц", "L4 PV, Гц", "V поля, мВ", "возб. поля", "торм. поля", "ресурс LGN поля")
    print(f"настройки: {opts}; варианты: {variants}")

    for seed in seeds:
        out, n_field = run_seed(seed, opts, variants)
        print(f"\n=== зерно {seed}: клеток поля L4 {n_field}")
        for variant, (blank, word, sync) in out.items():
            print(f"\n  {variant}: синхронность входов LGN поля (доля в ±1 мс) — пусто {sync['пусто']:.3f} "
                  f"(случай {sync['пусто случай']:.3f}), слово {sync['слово']:.3f} (случай {sync['слово случай']:.3f})")
            for j, name in enumerate(names):
                m, sd = float(blank[:, j].mean()), float(blank[:, j].std())
                print(f"    {name:16s} пусто {m:7.2f} ±{sd:5.2f} | слово по окнам: "
                      + " ".join(f"{float(v):6.2f}" for v in word[:, j]), flush=True)


if __name__ == "__main__":
    main()
