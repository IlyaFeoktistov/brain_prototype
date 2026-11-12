"""
Чтение с саккадами против неподвижного взгляда: передаётся ли смена
картинки по цепи сетчатка → LGN → V1 L4 → L2/3 → зона букв.

Синапс таламус → кора при спонтанной LGN ~10 Гц уже у предела депрессии
(docs/DESIGN.md «Разложение возбуждения поля L4»): L4 отвечает на смену
входа ~80–100 мс. При чтении взгляд меняется саккадами (фиксации ~200–250 мс
— Rayner 1998, https://doi.org/10.1037/0033-2909.124.3.372).

Режимы (одно зерно, один фон — оба в одном мозге по очереди):
    неподвижно  взгляд на начале строки всё предъявление;
    чтение      фиксации fix мс, между ними саккада (cfg.saccade_ms пустого
                входа — подавление на время скачка) на step знаков вперёд.
Саккады — по расписанию стенда (зрительная цепь отдельно от двухолмия).

Порядок: пусто, A, пусто, B, пусто, A (каждый режим). По фиксации (окно
после начала фиксации, ms): LGN — частота клеток, чей водитель видит штрих
этой фиксации, и их ресурс синапсов; V1 L4 E — поле фиксации (больше
половины входов LGN — зрячие этой фиксации) и вне поля всей строки (ни
одного зрячего входа); V1 L2/3 E, зона букв L4 и L2/3 E. Различение: вектор
частот по фиксациям на каждой ступени (LGN, V1 L4, V1 L2/3, буквы L4, L2/3),
сходство (корреляция) A с повтором A против A с B — повтор должен быть ближе.
По ступеням видно, где пропадает различение и не упирается ли мера в шум
счёта (если и LGN не различает — мера, а не цепь).

    PYTHONPATH=. python tools/reading_bench.py [a=мама мыла раму] [b=папа пила чай] [fix=225]
        [step=5] [seed=1] [fit=lgn-a] [homeo=имя] [relay_v_t_mv=-59] [relay_rest_mv=-48] [trn_scale=0.5]
"""

from __future__ import annotations

import sys

import torch

from brain import anatomy
from brain.config import BRAIN_HOME, BrainConfig
from brain.mind import Mind
from brain.session import Session

WIN = 10.0


def observe_pre(b, ms: float, view):
    """
    Частоты окна (b.observe) и состояние кратковременной пластичности до
    окна: stp_x (пути ядер), stp_u4, stp_x4 (синапсы колонок). Силу синапса
    для «частота × сила» брать отсюда: состояние в конце окна уже истощено
    спайками, которые входят в частоту того же окна (у реле LGN при 8 Гц ток
    таламус → L4 выходит вдвое ниже точного).
    """

    state = {"x": b.stp_x.clone(), "u4": b.stp_u4.clone(), "x4": b.stp_x4.clone()}
    return b.observe(ms, view), state


def build(opts: dict):
    cfg = BrainConfig()
    cfg.size = "small"
    cfg.seed = int(opts.get("seed", 1))
    cfg.autosave = False
    for k in ("relay_v_t_mv", "relay_rest_mv"):
        if k in opts:
            setattr(cfg, k, float(opts[k]))
    if opts.get("fit"):
        anatomy.use_fit(cfg, opts["fit"])
    if opts.get("clopath"):
        # Правило по напряжению на этих путях вместо STDP (cfg.clopath_paths).
        cfg.clopath_paths = tuple(opts["clopath"].split(","))
    if opts.get("dc"):
        # Пути с правилом блока Diehl & Cook (cfg.dc_paths) вместо STDP.
        cfg.dc_paths = tuple(opts["dc"].split(","))
    if opts.get("bg"):
        # Структура фона недостающих входов (cfg.background_mode; bg_patch — участок, px).
        cfg.background_mode = opts["bg"]
        cfg.background_patch_px = float(opts.get("bg_patch", cfg.background_patch_px))
    if opts.get("tonic"):
        # Постоянная тормозная проводимость (cfg.tonic_inh_g): зона.слой.тип:g, через запятую.
        cfg.tonic_inh_g = tuple((*k.split("."), float(g)) for k, g in (e.split(":") for e in opts["tonic"].split(",")))
    if opts.get("bg_scale"):
        # Доза внешнего фона (cfg.background_scale).
        cfg.background_scale = float(opts["bg_scale"])
    if opts.get("ee_skip"):
        # Зоны без человеческой связности E → E (cfg.column_ee_human_skip).
        cfg.column_ee_human_skip = tuple(opts["ee_skip"].split(","))
    if opts.get("nmda"):
        # NMDA у клеток колонок, E и тормозных (cfg.nmda, cfg.nmda_inh_ratio).
        cfg.nmda = True
        cfg.nmda_chariker = opts["nmda"] == "chariker"
    if opts.get("lgn_l6"):
        # Прямой вход LGN → V1 L6 (cfg.lgn_l6): доля клеток[,входов[,по данным]].
        parts = opts["lgn_l6"].split(",")
        cfg.lgn_l6 = (float(parts[0]), int(parts[1]) if len(parts) > 1 else 20, float(parts[2]) if len(parts) > 2 else 85.0)
    for k, v in opts.items():
        if (k.startswith("clopath_") and k != "clopath_paths") or (k.startswith("dc_") and k != "dc_paths"):
            setattr(cfg, k, float(v))
    mind = Mind(cfg)
    b = mind.brain
    b.plasticity = False
    for p in b.projections:
        if p.name == "trn -> lgn":
            p.weight.mul_(float(opts.get("trn_scale", 1.0)))
    if opts.get("homeo"):
        # Сдвиги гомеостаза после развития (tools/develop.py) — заморожены.
        b.bias[: b.n] += torch.load(BRAIN_HOME / "fits" / f"{opts['homeo']}-homeo.pt").to(b.device)
    return mind, b


def main(opts: dict) -> None:
    mind, b = build(opts)
    cfg = b.cfg
    s = Session(mind, window_ms=WIN)
    lgn = b.areas["lgn"].E
    vis, let = b.areas["visual"].layers, b.areas["letters"].layers
    l4 = vis["L4"]["E"]
    eye_i = [x.name for x in b.sensors].index("eye")
    off = b.sensor_offsets[eye_i]
    driver = (next(p for p in b.projections if p.name == "eye -> " + lgn.name).pre[:, 0].long() - off)
    tc = next(p for p in b.projections if p.name == f"lgn -> {l4.name}")
    tc_pre = tc.pre.long() - lgn.start
    a_text, b_text = opts.get("a", "мама мыла раму"), opts.get("b", "папа пила чай")
    fix, step = float(opts.get("fix", 225)), float(opts.get("step", 5))
    s.line, s.gaze = "", 0.0
    b.run(2500.0, s._blank(2500.0))

    def ink(line: str, gaze: float) -> torch.Tensor:
        """Клетки реле, чей водитель видит штрих при этом взгляде."""

        s.line, s.gaze = line, gaze
        seen = s.eye.rates(s._picture(s.screen, cfg.fovea_col)) >= 5.0
        return seen.to(driver.device)[driver]

    def gazes(line: str) -> list[float]:
        # Ширина знака — по ширине строки; фиксации каждые step знаков.
        width = s.screen.width(line) / max(1, len(line))
        n = max(1, int((len(line) + step - 1) // step))
        return [i * step * width for i in range(n)]

    stages = {"LGN": lgn, "V1 L4": l4, "V1 L2/3": vis["L23"]["E"],
              "буквы L4": let["L4"]["E"], "буквы L2/3": let["L23"]["E"]}
    names = ("LGN зрячие, Гц", "ресурс зрячих", "L4 поле, Гц", "L4 вне строки", "V1 L2/3 E", "буквы L4 E", "буквы L2/3 E")

    for mode in ("неподвижно", "чтение"):
        vectors = {}
        for label, text in (("A", a_text), ("B", b_text), ("A2", a_text)):
            s.line = ""
            b.observe(500.0, s._view(500.0))      # пустой экран между фразами
            stage_vec = {k: [] for k in stages}
            rows = []
            line = text + " "
            points = gazes(line)
            any_ink = torch.zeros(lgn.size, dtype=torch.bool, device=b.device)
            for g in points:
                any_ink |= ink(line, g)
            outside = (~any_ink[tc_pre]).all(1)
            for i, g in enumerate(points):
                if mode == "чтение" and i:
                    s.line, s.gaze = line, g
                    b.run(cfg.saccade_ms, s._blank(cfg.saccade_ms))
                gaze = g if mode == "чтение" else points[0]
                seeing = ink(line, gaze)
                field = seeing[tc_pre].float().mean(1) > 0.5
                trace, acc = [], {k: 0 for k in stages}
                for _ in range(int(fix // WIN)):
                    r = b.observe(WIN, s._view(WIN))
                    pick = lambda m: r[lgn.slice][m].mean() if m.any() else r.new_tensor(0.0)
                    trace.append(torch.stack([
                        pick(seeing), b.stp_x[lgn.slice][seeing].mean() if seeing.any() else r.new_tensor(0.0),
                        r[l4.slice][field].mean() if field.any() else r.new_tensor(0.0),
                        r[l4.slice][outside].mean() if outside.any() else r.new_tensor(0.0),
                        r[vis["L23"]["E"].slice].mean(), r[let["L4"]["E"].slice].mean(), r[let["L23"]["E"].slice].mean(),
                    ]).cpu())
                    for k, pop in stages.items():
                        acc[k] = acc[k] + r[pop.slice].cpu()
                for k in stages:
                    stage_vec[k].append(acc[k])
                rows.append((g, torch.stack(trace), int(field.sum()), int(outside.sum())))
            vectors[label] = {k: torch.cat(v) for k, v in stage_vec.items()}
            print(f"\n=== {mode}: «{text}» ({label}), фиксаций {len(points)}, {fix:g} мс, окна {WIN:g} мс")
            for g, tr, nf, no in rows:
                print(f"  взгляд {g:5.1f}: клеток поля L4 {nf}, вне строки {no}")
                for j, name in enumerate(names):
                    fmt = "{:5.2f}" if j == 1 else "{:5.1f}"
                    print(f"    {name:16s} " + " ".join(fmt.format(float(v)) for v in tr[:, j]))
        print(f"\n{mode}: сходство A с повтором A против A с B (повтор должен быть ближе)")
        for k in stages:
            va, va2, vb = (vectors[x][k] for x in ("A", "A2", "B"))
            n = min(len(va), len(va2), len(vb))
            c = lambda x, y: float(torch.corrcoef(torch.stack([x[:n], y[:n]]))[0, 1])
            print(f"  {k:12s} A–A2 {c(va, va2):6.3f}   A–B {c(va, vb):6.3f}", flush=True)


if __name__ == "__main__":
    main(dict(kv.split("=", 1) for kv in sys.argv[1:]))
