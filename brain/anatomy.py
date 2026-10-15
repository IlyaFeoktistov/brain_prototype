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
    # Гиппокамп наяву (Mizuseki & Buzsáki 2013): CA1 ~1.5, CA3 ~1, зубчатая
    # — редко, ~0.3 Гц; энторинальная L2/3 ~2 [сверить]. Гранулярные клетки
    # мозжечка — редко, ~0.5 (Chadderton, Margrie & Häusser 2004). VLPO наяву
    # ~1, во сне 5–10 (Szymusiak et al. 1998).
    "ca1": 1.5, "ca3": 1.0, "dentate": 0.3, "entorhinal": 2.0, "granule": 0.5, "vlpo": 1.0,
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

# [согласовано с частотами in vivo] 2026-10-15, data/tools/anatomy_fit.py
# (20 + 15 + 15 шагов, весь мозг small, шаг 0.5 мс, с плато и таламусом по
# данным; внешний фон L2/3 и L5 ушёл почти в 0 — вероятно двойной учёт: вход
# Potjans уже содержит таламус и другие зоны, у нас они явные — разобрать): фон ядер, мВ
# (заменяет входы ствола и коры, которых нет), и фон колонок по слою и типу.
# Лучшая ошибка 0.37; от шага к шагу ядра сна и бодрствования (LC, шов,
# орексин, Мейнерт, VLPO) перескакивают разом — их взаимное торможение —
# переключатель (Saper et al. 2001). Выше нормы: PV L2/3 (~40 против 6.4).
NUCLEUS_BIAS_MV = (('vta', 19.2), ('amygdala', 16.57), ('lc', 7.35), ('vlpo', 17.3), ('striatum_d1', 10.46), ('striatum_d2', 23.91), ('gpe', 7.8), ('gpi', 19.45), ('thalamus', 57.07), ('entorhinal', -55.92), ('dentate', 13.96), ('ca3', 19.35), ('ca1', -24.87), ('granule', 9.28), ('lgn', 42.76), ('mgn', 81.87), ('md', 23.14), ('trn', -19.71), ('fsi', 12.21), ('stn', 24.84), ('raphe', -1.92), ('nbm', 15.58), ('orexin', -1.23), ('cea', 17.59), ('purkinje', 66.63), ('dcn', 66.48), ('io', 3.26))
COLUMN_EXTERNAL_POP = (('auditory', 'L23', 'E', 0.738), ('auditory', 'L23', 'PV', 0.0), ('auditory', 'L23', 'SOM', 0.155), ('auditory', 'L23', 'VIP', 0.296), ('auditory', 'L4', 'E', 0.764), ('auditory', 'L4', 'PV', 0.084), ('auditory', 'L4', 'SOM', 1.252), ('auditory', 'L4', 'VIP', 0.297), ('auditory', 'L5', 'E', 0.744), ('auditory', 'L5', 'PV', 0.0), ('auditory', 'L5', 'SOM', 1.189), ('auditory', 'L5', 'VIP', 0.415), ('auditory', 'L6', 'E', 1.327), ('auditory', 'L6', 'PV', 0.313), ('auditory', 'L6', 'SOM', 0.797), ('auditory', 'L6', 'VIP', 0.657), ('association', 'L23', 'E', 0.517), ('association', 'L23', 'PV', 0.0), ('association', 'L23', 'SOM', 0.155), ('association', 'L23', 'VIP', 0.044), ('association', 'L4', 'E', 3.624), ('association', 'L4', 'PV', 0.086), ('association', 'L4', 'SOM', 1.24), ('association', 'L4', 'VIP', 0.321), ('association', 'L5', 'E', 0.847), ('association', 'L5', 'PV', 0.0), ('association', 'L5', 'SOM', 1.237), ('association', 'L5', 'VIP', 0.787), ('association', 'L6', 'E', 1.488), ('association', 'L6', 'PV', 0.218), ('association', 'L6', 'SOM', 0.745), ('association', 'L6', 'VIP', 0.596), ('prefrontal', 'L23', 'E', 0.268), ('prefrontal', 'L23', 'PV', 0.0), ('prefrontal', 'L23', 'SOM', 0.155), ('prefrontal', 'L23', 'VIP', 0.006), ('prefrontal', 'L4', 'E', 0.004), ('prefrontal', 'L4', 'PV', 0.0), ('prefrontal', 'L4', 'SOM', 1.047), ('prefrontal', 'L4', 'VIP', 0.001), ('prefrontal', 'L5', 'E', 0.392), ('prefrontal', 'L5', 'PV', 0.0), ('prefrontal', 'L5', 'SOM', 1.136), ('prefrontal', 'L5', 'VIP', 0.189), ('prefrontal', 'L6', 'E', 1.365), ('prefrontal', 'L6', 'PV', 0.077), ('prefrontal', 'L6', 'SOM', 0.613), ('prefrontal', 'L6', 'VIP', 0.594), ('motor', 'L23', 'E', 0.0), ('motor', 'L23', 'PV', 0.0), ('motor', 'L23', 'SOM', 0.0), ('motor', 'L23', 'VIP', 0.0), ('motor', 'L5', 'E', 0.0), ('motor', 'L5', 'PV', 0.0), ('motor', 'L5', 'SOM', 0.0), ('motor', 'L5', 'VIP', 0.267), ('motor', 'L6', 'E', 1.312), ('motor', 'L6', 'PV', 0.0), ('motor', 'L6', 'SOM', 0.001), ('motor', 'L6', 'VIP', 0.205), ('goal', 'L23', 'E', 0.335), ('goal', 'L23', 'PV', 0.0), ('goal', 'L23', 'SOM', 0.155), ('goal', 'L23', 'VIP', 0.01), ('goal', 'L4', 'E', 0.004), ('goal', 'L4', 'PV', 0.0), ('goal', 'L4', 'SOM', 1.066), ('goal', 'L4', 'VIP', 0.002), ('goal', 'L5', 'E', 0.293), ('goal', 'L5', 'PV', 0.0), ('goal', 'L5', 'SOM', 1.163), ('goal', 'L5', 'VIP', 0.324), ('goal', 'L6', 'E', 1.301), ('goal', 'L6', 'PV', 0.086), ('goal', 'L6', 'SOM', 0.565), ('goal', 'L6', 'VIP', 0.587), ('limbic', 'L23', 'E', 0.95), ('limbic', 'L23', 'PV', 0.0), ('limbic', 'L23', 'SOM', 0.155), ('limbic', 'L23', 'VIP', 0.466), ('limbic', 'L5', 'E', 1.446), ('limbic', 'L5', 'PV', 0.0), ('limbic', 'L5', 'SOM', 1.074), ('limbic', 'L5', 'VIP', 0.713), ('limbic', 'L6', 'E', 1.114), ('limbic', 'L6', 'PV', 0.342), ('limbic', 'L6', 'SOM', 0.606), ('limbic', 'L6', 'VIP', 0.6), ('category', 'L23', 'E', 0.725), ('category', 'L23', 'PV', 0.0), ('category', 'L23', 'SOM', 0.155), ('category', 'L23', 'VIP', 0.275), ('category', 'L4', 'E', 3.267), ('category', 'L4', 'PV', 0.104), ('category', 'L4', 'SOM', 1.166), ('category', 'L4', 'VIP', 0.333), ('category', 'L5', 'E', 0.733), ('category', 'L5', 'PV', 0.0), ('category', 'L5', 'SOM', 1.189), ('category', 'L5', 'VIP', 0.845), ('category', 'L6', 'E', 1.321), ('category', 'L6', 'PV', 0.281), ('category', 'L6', 'SOM', 0.758), ('category', 'L6', 'VIP', 0.661), ('visual', 'L23', 'E', 0.309), ('visual', 'L23', 'PV', 0.0), ('visual', 'L23', 'SOM', 0.155), ('visual', 'L23', 'VIP', 0.133), ('visual', 'L4', 'E', 4.249), ('visual', 'L4', 'PV', 0.024), ('visual', 'L4', 'SOM', 1.052), ('visual', 'L4', 'VIP', 0.298), ('visual', 'L5', 'E', 1.702), ('visual', 'L5', 'PV', 0.0), ('visual', 'L5', 'SOM', 1.204), ('visual', 'L5', 'VIP', 0.373), ('visual', 'L6', 'E', 2.135), ('visual', 'L6', 'PV', 0.049), ('visual', 'L6', 'SOM', 0.746), ('visual', 'L6', 'VIP', 0.475), ('letters', 'L23', 'E', 0.626), ('letters', 'L23', 'PV', 0.0), ('letters', 'L23', 'SOM', 0.155), ('letters', 'L23', 'VIP', 0.181), ('letters', 'L4', 'E', 3.197), ('letters', 'L4', 'PV', 0.025), ('letters', 'L4', 'SOM', 0.935), ('letters', 'L4', 'VIP', 0.228), ('letters', 'L5', 'E', 0.901), ('letters', 'L5', 'PV', 0.0), ('letters', 'L5', 'SOM', 1.204), ('letters', 'L5', 'VIP', 0.429), ('letters', 'L6', 'E', 1.576), ('letters', 'L6', 'PV', 0.063), ('letters', 'L6', 'SOM', 0.71), ('letters', 'L6', 'VIP', 0.423), ('hand', 'L23', 'E', 0.0), ('hand', 'L23', 'PV', 0.0), ('hand', 'L23', 'SOM', 0.0), ('hand', 'L23', 'VIP', 0.0), ('hand', 'L5', 'E', 0.0), ('hand', 'L5', 'PV', 0.0), ('hand', 'L5', 'SOM', 0.001), ('hand', 'L5', 'VIP', 0.521), ('hand', 'L6', 'E', 1.166), ('hand', 'L6', 'PV', 0.0), ('hand', 'L6', 'SOM', 0.002), ('hand', 'L6', 'VIP', 0.388))


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

    if not cfg.nucleus_bias_mv:
        cfg.nucleus_bias_mv = NUCLEUS_BIAS_MV

    if not cfg.column_gain_external_pop:
        cfg.column_gain_external_pop = COLUMN_EXTERNAL_POP
