"""
Анатомический мозг (флаг BrainConfig.anatomy; ARCHITECTURE, раздел 11):
кора — колонки по слоям (cfg.layers), гиппокамп — из нейронов, подкорковые
цепи — каждая целиком по данным: таламус с ретикулярным ядром и реле
органов чувств, базальные ганглии с STN и быстрыми тормозными стриатума,
ядра модуляторов (дофамин, норадреналин, серотонин, ацетилхолин, орексин),
миндалина (базолатеральная и центральная), мозжечок (гранулярные,
Пуркинье, ядра, нижняя олива).

Каждое ядро — главные клетки (у тормозных ядер их выход — со знаком «−» в
путях); числа — частоты, PSP, пейсмейкеры — с источником. Где число не
найдено — «[сверить]» или «[не измерено]».

Частоты in vivo (бодрствование) для стенда — RATES_HZ.
"""

from __future__ import annotations

from dataclasses import replace

from brain.config import AreaSpec, PathwaySpec

# Частоты покоя у бодрствующих, Гц (стенд anatomy_bench).
RATES_HZ = {
    # Реле таламуса: тоническая частота наяву ~10 Гц (Swadlow & Gusev 2001 —
    # VB кролика; LGN мыши — Niell & Stryker 2010 [сверить]).
    "lgn": 10.0, "mgn": 10.0, "thalamus": 15.0, "md": 10.0,
    # Ретикулярное ядро наяву ~10–30 Гц (Steriade 2005 [сверить]).
    "trn": 15.0,
    # Базальные ганглии (бодрствующая крыса/мышь; сводка — Lindahl &
    # Hellgren Kotaleski 2016, Table 1): MSN ~0.5, FSI ~15, GPe ~40, STN
    # ~15, SNr/GPi ~60 Гц.
    "striatum_d1": 0.5, "striatum_d2": 0.5, "fsi": 15.0, "gpe": 40.0, "stn": 15.0, "gpi": 60.0,
    # Модуляторы: VTA ~4.5 (Grace & Bunney 1984), LC ~2 наяву (Aston-Jones &
    # Bloom 1981), шов ~2 (Jacobs & Fornal 1999), ядро Мейнерта ~5 (Hangya
    # et al. 2015), орексин ~3 (Lee, Hassani & Jones 2005).
    "vta": 4.5, "lc": 2.0, "raphe": 2.0, "nbm": 5.0, "orexin": 3.0,
    # Миндалина: базолатеральная — главные клетки <1 Гц (Paré & Gaudreau
    # 1996), центральная ~5 [сверить].
    "amygdala": 0.5, "cea": 5.0,
    # Мозжечок: Пуркинье простые спайки ~50, ядра ~30, олива ~1 Гц
    # (Häusser & Clark 1997; Raman et al. 2000; Lang et al. 1999).
    "purkinje": 50.0, "dcn": 30.0, "io": 1.0,
}


def _area(name, title, share, **kw) -> AreaSpec:
    """Подкорковое ядро: без обучаемых связей внутри, без гомеостаза по словам."""

    return AreaSpec(name, title, share, k_ee=10, ee_plastic=False, ee_weight=(0.0, 0.0), target_activity=0.0, **kw)


def areas(cfg) -> list[AreaSpec]:
    out = [
        # ---- таламус ----
        # Реле органов чувств: клеток — сколько рецепторов (один к одному).
        _area("lgn", "латеральное коленчатое тело (реле зрения)", 0.0,
              cells=2 * cfg.retina_rows * cfg.retina_cols if cfg.eye else 0),
        _area("mgn", "медиальное коленчатое тело (реле слуха)", 0.0, cells=cfg.sensory_size),
        _area("md", "медиодорсальное ядро таламуса (реле ПФК)", 0.003),
        # Ретикулярное ядро — только тормозные (ГАМК), тормозят реле.
        _area("trn", "ретикулярное ядро таламуса", 0.003),
        # ---- базальные ганглии ----
        # Быстрые тормозные стриатума — 1–2% клеток (Tepper et al. 2010).
        _area("fsi", "стриатум: быстрые тормозные (PV)", 0.001),
        # STN — пейсмейкер ~10 Гц в срезе (Bevan & Wilson 1999).
        _area("stn", "субталамическое ядро", 0.002, autonomous_hz=10.0, tonic=True),
        # ---- модуляторы ----
        # Шов — серотонин, пейсмейкер ~1–2 Гц (Vandermaelen & Aghajanian 1983).
        _area("raphe", "дорсальное ядро шва (серотонин)", 0.002, autonomous_hz=1.5, tonic=True),
        # Ядро Мейнерта — ацетилхолин коры.
        _area("nbm", "базальное ядро Мейнерта (ацетилхолин)", 0.002),
        # Орексин — латеральный гипоталамус, держит бодрствование
        # (сам по себе стреляет — Eggermann et al. 2003).
        _area("orexin", "латеральный гипоталамус (орексин)", 0.002, autonomous_hz=3.0, tonic=True),
        # ---- миндалина ----
        # Центральное ядро — тормозные клетки, выход миндалины.
        _area("cea", "центральное ядро миндалины (выход)", 0.002),
    ]

    if cfg.hand:
        out += [
            # Пуркинье — пейсмейкеры ~40–50 Гц в срезе (Raman & Bean 1999);
            # ядра мозжечка ~10–20 (Raman et al. 2000); олива ~1 Гц.
            _area("purkinje", "мозжечок: клетки Пуркинье", 0.01, autonomous_hz=45.0, tonic=True),
            _area("dcn", "ядра мозжечка", 0.003, autonomous_hz=15.0, tonic=True),
            _area("io", "нижняя олива (лазающие волокна)", 0.002, autonomous_hz=1.0, tonic=True),
        ]

    return out


def _path(src, dst, k, psp, direction="", **kw) -> PathwaySpec:
    """Путь с весом из PSP (мВ; знак — знак синапса), без отдельного торможения."""

    return PathwaySpec(src, dst, k=k, weight=(0.0, 0.0), k_inhibitory=0, predictive=False,
                       psp_mv=psp, direction=direction, **kw)


def pathways(cfg) -> list[PathwaySpec]:
    out = []
    # ---- таламус (Sherman & Guillery 2006) ----
    # L6 → реле: много слабых облегчающихся входов (модуляторы; ~30–50%
    # синапсов клетки реле — Van Horn, Erişir & Sherman 2000); L6 → TRN —
    # в 2–3 раза сильнее (Golshani, Liu & Jones 2001). Реле → TRN —
    # коллатерали; TRN → реле — ГАМК, IPSP ~1–2 мВ [сверить].
    relays = [("visual", "lgn"), ("auditory", "mgn"), ("prefrontal", "md"), ("goal", "md"), ("motor", "thalamus")]

    for cortex, relay in relays:
        out += [
            _path(cortex, relay, 50, 0.3, "ct"),
            _path(cortex, "trn", 50, 0.7, "ct"),
        ]

    for relay in ("lgn", "mgn", "md", "thalamus"):
        out += [
            _path(relay, "trn", 20, 1.5),
            _path("trn", relay, 20, -1.5),
        ]

    # Реле ПФК: MD → ПФК и цель (в L4, как вход таламуса).
    out += [
        _path("md", "prefrontal", 20, 0.6, "thalamic"),
        _path("md", "goal", 20, 0.6, "thalamic"),
    ]
    # ---- базальные ганглии (Gerfen & Surmeier 2011; Lindahl & Hellgren
    # Kotaleski 2016 — сводка числа входов и PSP [сверить по таблице]) ----
    # Кора (L5) → FSI — сильно (Mallet et al. 2005), FSI → MSN — крупные IPSP
    # (Koós & Tepper 1999); кора → STN — «гиперпрямой» путь (Nambu et al.
    # 2002); STN → GPe, GPi; GPe → STN, GPi, FSI (Mallet et al. 2012).
    for cortex in ("prefrontal", "motor"):
        out += [
            _path(cortex, "fsi", 50, 1.0),
            _path(cortex, "stn", 50, 0.5),
        ]
    out += [
        _path("fsi", "striatum_d1", 10, -1.5),
        _path("fsi", "striatum_d2", 10, -1.5),
        _path("stn", "gpe", 20, 1.0),
        _path("stn", "gpi", 20, 1.0),
        _path("gpe", "stn", 20, -1.0),
        _path("gpe", "fsi", 10, -1.0),
    ]
    # Таламус VA/VL → рука (как моторная кора): «давай» из базальных ганглиев.
    if cfg.hand:
        out.append(_path("thalamus", "hand", 20, 0.6, "thalamic"))
    # ---- модуляторы ----
    # Ядро Мейнерта: вход — ПФК и миндалина (Zaborszky et al. 1997);
    # орексин → LC, Мейнерт (Hagan et al. 1999; Eggermann et al. 2001); VLPO
    # тормозит орексин и шов (Sherin et al. 1998).
    out += [
        _path("prefrontal", "nbm", 30, 0.5),
        _path("cea", "nbm", 20, -0.5),
        _path("orexin", "lc", 20, 1.0),
        _path("orexin", "nbm", 20, 1.0),
        _path("vlpo", "orexin", 20, -1.0),
        _path("vlpo", "raphe", 20, -1.0),
        _path("prefrontal", "raphe", 20, 0.5),
    ]
    # ---- миндалина (LeDoux 2000; Paré et al. 2004) ----
    # Базолатеральная → центральная; центральная (ГАМК) → LC, VTA, Мейнерт.
    out += [
        _path("amygdala", "cea", 30, 1.0),
        _path("cea", "lc", 20, -0.8),
        _path("cea", "vta", 20, -0.8),
        _path("amygdala", "prefrontal", 30, 0.4, "lateral"),
    ]
    # ---- мозжечок (Ito 1984; Marr 1969; Albus 1971) ----
    # Параллельные волокна → Пуркинье: очень много слабых (~0.05 мВ
    # [сверить]); Пуркинье → ядра — крупные IPSP (сходятся ~30–50 на
    # клетку ядра; Person & Raman 2012); олива → Пуркинье — одно лазающее
    # волокно, сложный спайк; ядра → таламус VA/VL → моторная кора.
    if cfg.hand:
        out += [
            _path("granule", "purkinje", 1000, 0.05),
            _path("purkinje", "dcn", 40, -2.0),
            _path("io", "purkinje", 1, 10.0),
            _path("dcn", "thalamus", 20, 1.0),
            _path("dcn", "io", 10, -1.0),
        ]
    return out


# Рецепторы (вещество, зона[:тип], мВ при полной занятости). Знак — по
# данным, величина [не измерено в наших единицах]:
# - ацетилхолин: никотиновые возбуждают VIP (Fu et al. 2014; Letzkus et al.
#   2011), мускариновые M1 деполяризуют пирамидные клетки (McCormick & Prince
#   1986);
# - серотонин: 5-HT1A гиперполяризуют пирамидные клетки ПФК (Araneda &
#   Andrade 1991);
# - орексин возбуждает LC и Мейнерт (Hagan et al. 1999; Eggermann et al. 2001).
RECEPTORS = (
    ("acetylcholine", ":VIP", 5.0),
    ("acetylcholine", "", 2.0),
    ("serotonin", "prefrontal", -3.0),
    ("serotonin", "goal", -3.0),
    ("orexin", "lc", 10.0),
    ("orexin", "nbm", 5.0),
)

MODULATORS = (("serotonin", "raphe"), ("acetylcholine", "nbm"), ("orexin", "orexin"))


def apply(cfg) -> None:
    """Достроить cfg до анатомического мозга (вызывается из Mind)."""

    names = {a.name for a in cfg.areas}
    cfg.areas = cfg.areas + [a for a in areas(cfg) if a.name not in names and (a.cells or a.share)]
    names = {a.name for a in cfg.areas}

    if cfg.hand and "purkinje" in names:
        # Выход мозжечка — через Пуркинье и ядра, а не прямой путь к руке.
        cfg.pathways = [p for p in cfg.pathways if not (p.src == "granule" and p.dst == "hand")]

    cfg.pathways = cfg.pathways + [p for p in pathways(cfg) if p.src in names and p.dst in names]
    cfg.extra_modulators = MODULATORS
    cortex = [a for a in cfg.layered_areas if a in names]
    receptors = []

    for substance, target, mv in RECEPTORS:
        if target.startswith(":"):
            receptors += [(substance, f"{area}{target}", mv) for area in cortex]
        elif target == "":
            receptors += [(substance, area, mv) for area in cortex]
        else:
            receptors.append((substance, target, mv))

    cfg.receptors = tuple(cfg.receptors) + tuple(receptors)
