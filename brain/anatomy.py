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

# Частоты в медленном сне, Гц (переключатель сна, стенды sleep_switch и
# sleep_fit): VLPO 5–10 (Szymusiak et al. 1998); LC ~вдвое реже, чем наяву
# (Aston-Jones & Bloom 1981), шов — тоже (Jacobs & Fornal 1999); орексин
# почти молчит (Lee, Hassani & Jones 2005); холинергические Мейнерта в
# медленном сне редки (Lee et al. 2005 [сверить число]).
SLEEP_RATES_HZ = {"vlpo": 7.0, "lc": 1.0, "raphe": 1.0, "orexin": 0.1, "nbm": 0.5}


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
    # ---- переключатель сна (Saper, Chou & Scammell 2001) ----
    # VLPO (ГАМК, галанин) тормозит ядра бодрствования синапсами — LC, шов,
    # орексин (Sherin et al. 1998); обратно ядра бодрствования тормозят VLPO
    # не синапсом, а веществом — норадреналин, серотонин, ацетилхолин
    # (Gallopin et al. 2000; рецепторы — RECEPTORS). Орексин возбуждает шов
    # (Brown et al. 2001), как LC и Мейнерт.
    out += [
        _path("vlpo", "lc", 20, -1.0),
        _path("orexin", "raphe", 20, 1.0),
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
# - орексин возбуждает LC и Мейнерт (Hagan et al. 1999; Eggermann et al. 2001);
# - VLPO тормозят норадреналин (α2), ацетилхолин (мускариновые) и у
#   большинства клеток серотонин, гистамин — нет (Gallopin et al. 2000):
#   обратная ветвь переключателя сна. Величина [сверить по статье];
# - аденозин (A1) тормозит орексиновые клетки (Liu & Gao 2007 — через вход,
#   без сдвига потенциала покоя; у нас — током клетки) и холинергические
#   базального мозга (Arrigoni et al. 2006); A2A деполяризует часть клеток
#   VLPO (Gallopin et al. 2005), A1 гиперполяризует LC (Regenold & Illes
#   1990) — единицы мВ [сверить]; заменяют подобранные под старый стенд
#   +35 / −15 из cfg.receptors (с ними VLPO во сне ~30 Гц против 5–10).
RECEPTORS = (
    ("acetylcholine", ":VIP", 5.0),
    ("acetylcholine", "", 2.0),
    ("serotonin", "prefrontal", -3.0),
    ("serotonin", "goal", -3.0),
    ("orexin", "lc", 10.0),
    ("orexin", "nbm", 5.0),
    ("noradrenaline", "vlpo", -5.0),
    ("serotonin", "vlpo", -5.0),
    ("acetylcholine", "vlpo", -5.0),
    ("adenosine", "orexin", -5.0),
    ("adenosine", "nbm", -5.0),
    ("adenosine", "vlpo", 5.0),
    ("adenosine", "lc", -5.0),
)

MODULATORS = (("serotonin", "raphe"), ("acetylcholine", "nbm"), ("orexin", "orexin"))

# [согласовано с частотами in vivo] 2026-10-15. Оба набора пишут скрипты
# подбора сами (anatomy.write_fitted), руками не править.
# NUCLEUS_BIAS_MV — фон каждого ядра, мВ: постоянный ток вместо входов
# ствола и коры, которых в модели нет. Подбор — data/tools/anatomy_fit.py
# (весь мозг small, шаг 0.5 мс, частоты наяву RATES_HZ), кроме узла сна
# (VLPO, LC, шов, орексин, Мейнерт): он двустабилен, его подбирает
# data/tools/sleep_fit.py по двум состояниям — наяву (RATES_HZ) и в
# медленном сне (SLEEP_RATES_HZ).
# COLUMN_EXTERNAL_POP — множитель внешнего фона колонки (вход из мозга вне
# модели, Potjans минус явные пути) по зоне, слою и типу клетки; подбор —
# anatomy_fit под column_target_hz. Внешний фон L2/3 и L5 ушёл почти в 0 —
# вероятно двойной учёт (вход Potjans уже содержит таламус и другие зоны,
# у нас они явные) — разобрать. Выше нормы: PV L2/3 (~40 против 6.4).
NUCLEUS_BIAS_MV = (
    ('vta', 17.97),
    ('amygdala', 10.77),
    ('lc', 20.3),
    ('vlpo', 18.56),
    ('striatum_d1', 6.88),
    ('striatum_d2', 18.37),
    ('gpe', 8.07),
    ('gpi', 20.38),
    ('thalamus', 58.48),
    ('entorhinal', -53.19),
    ('dentate', 11.0),
    ('ca3', 18.42),
    ('ca1', -8.32),
    ('granule', -1.88),
    ('lgn', 43.78),
    ('mgn', 82.6),
    ('md', 23.13),
    ('trn', -15.99),
    ('fsi', 19.8),
    ('stn', 26.89),
    ('raphe', -3.33),
    ('nbm', 14.79),
    ('orexin', -1.07),
    ('cea', 15.77),
    ('purkinje', 68.71),
    ('dcn', 71.43),
    ('io', 1.8),
)
# (зона, слой, тип клетки, множитель внешнего фона)
COLUMN_EXTERNAL_POP = (
    ('auditory', 'L23', 'E', 1.97), ('auditory', 'L23', 'PV', 0.0), ('auditory', 'L23', 'SOM', 1.65), ('auditory', 'L23', 'VIP', 0.126),
    ('auditory', 'L4', 'E', 1.627), ('auditory', 'L4', 'PV', 0.082), ('auditory', 'L4', 'SOM', 1.38), ('auditory', 'L4', 'VIP', 0.112),
    ('auditory', 'L5', 'E', 1.571), ('auditory', 'L5', 'PV', 0.0), ('auditory', 'L5', 'SOM', 1.407), ('auditory', 'L5', 'VIP', 0.262),
    ('auditory', 'L6', 'E', 1.966), ('auditory', 'L6', 'PV', 0.515), ('auditory', 'L6', 'SOM', 0.739), ('auditory', 'L6', 'VIP', 0.493),
    ('association', 'L23', 'E', 1.007), ('association', 'L23', 'PV', 0.0), ('association', 'L23', 'SOM', 1.619), ('association', 'L23', 'VIP', 0.002),
    ('association', 'L4', 'E', 4.059), ('association', 'L4', 'PV', 0.064), ('association', 'L4', 'SOM', 1.411), ('association', 'L4', 'VIP', 0.128),
    ('association', 'L5', 'E', 0.942), ('association', 'L5', 'PV', 0.0), ('association', 'L5', 'SOM', 1.827), ('association', 'L5', 'VIP', 0.642),
    ('association', 'L6', 'E', 1.81), ('association', 'L6', 'PV', 0.258), ('association', 'L6', 'SOM', 0.755), ('association', 'L6', 'VIP', 0.394),
    ('prefrontal', 'L23', 'E', 2.382), ('prefrontal', 'L23', 'PV', 0.0), ('prefrontal', 'L23', 'SOM', 1.65), ('prefrontal', 'L23', 'VIP', 0.019),
    ('prefrontal', 'L4', 'E', 0.01), ('prefrontal', 'L4', 'PV', 0.0), ('prefrontal', 'L4', 'SOM', 1.221), ('prefrontal', 'L4', 'VIP', 0.004),
    ('prefrontal', 'L5', 'E', 1.004), ('prefrontal', 'L5', 'PV', 0.0), ('prefrontal', 'L5', 'SOM', 1.276), ('prefrontal', 'L5', 'VIP', 0.371),
    ('prefrontal', 'L6', 'E', 1.608), ('prefrontal', 'L6', 'PV', 0.594), ('prefrontal', 'L6', 'SOM', 0.686), ('prefrontal', 'L6', 'VIP', 0.542),
    ('motor', 'L23', 'E', 0.0), ('motor', 'L23', 'PV', 0.0), ('motor', 'L23', 'SOM', 0.0), ('motor', 'L23', 'VIP', 0.0),
    ('motor', 'L5', 'E', 0.0), ('motor', 'L5', 'PV', 0.0), ('motor', 'L5', 'SOM', 0.0), ('motor', 'L5', 'VIP', 0.086),
    ('motor', 'L6', 'E', 1.398), ('motor', 'L6', 'PV', 0.0), ('motor', 'L6', 'SOM', 0.0), ('motor', 'L6', 'VIP', 0.087),
    ('goal', 'L23', 'E', 2.321), ('goal', 'L23', 'PV', 0.0), ('goal', 'L23', 'SOM', 1.65), ('goal', 'L23', 'VIP', 0.031),
    ('goal', 'L4', 'E', 0.022), ('goal', 'L4', 'PV', 0.0), ('goal', 'L4', 'SOM', 1.119), ('goal', 'L4', 'VIP', 0.01),
    ('goal', 'L5', 'E', 1.838), ('goal', 'L5', 'PV', 0.0), ('goal', 'L5', 'SOM', 1.319), ('goal', 'L5', 'VIP', 0.449),
    ('goal', 'L6', 'E', 1.78), ('goal', 'L6', 'PV', 0.619), ('goal', 'L6', 'SOM', 0.631), ('goal', 'L6', 'VIP', 0.55),
    ('limbic', 'L23', 'E', 3.665), ('limbic', 'L23', 'PV', 0.0), ('limbic', 'L23', 'SOM', 1.65), ('limbic', 'L23', 'VIP', 0.167),
    ('limbic', 'L5', 'E', 2.161), ('limbic', 'L5', 'PV', 0.0), ('limbic', 'L5', 'SOM', 1.414), ('limbic', 'L5', 'VIP', 0.505),
    ('limbic', 'L6', 'E', 1.86), ('limbic', 'L6', 'PV', 0.625), ('limbic', 'L6', 'SOM', 0.621), ('limbic', 'L6', 'VIP', 0.445),
    ('category', 'L23', 'E', 2.183), ('category', 'L23', 'PV', 0.0), ('category', 'L23', 'SOM', 1.65), ('category', 'L23', 'VIP', 0.145),
    ('category', 'L4', 'E', 3.089), ('category', 'L4', 'PV', 0.119), ('category', 'L4', 'SOM', 0.989), ('category', 'L4', 'VIP', 0.138),
    ('category', 'L5', 'E', 0.737), ('category', 'L5', 'PV', 0.0), ('category', 'L5', 'SOM', 1.272), ('category', 'L5', 'VIP', 0.643),
    ('category', 'L6', 'E', 1.531), ('category', 'L6', 'PV', 0.433), ('category', 'L6', 'SOM', 0.751), ('category', 'L6', 'VIP', 0.521),
    ('visual', 'L23', 'E', 0.335), ('visual', 'L23', 'PV', 0.0), ('visual', 'L23', 'SOM', 1.65), ('visual', 'L23', 'VIP', 0.033),
    ('visual', 'L4', 'E', 4.166), ('visual', 'L4', 'PV', 0.002), ('visual', 'L4', 'SOM', 0.856), ('visual', 'L4', 'VIP', 0.087),
    ('visual', 'L5', 'E', 1.687), ('visual', 'L5', 'PV', 0.0), ('visual', 'L5', 'SOM', 1.278), ('visual', 'L5', 'VIP', 0.345),
    ('visual', 'L6', 'E', 1.941), ('visual', 'L6', 'PV', 0.003), ('visual', 'L6', 'SOM', 0.621), ('visual', 'L6', 'VIP', 0.389),
    ('letters', 'L23', 'E', 2.094), ('letters', 'L23', 'PV', 0.0), ('letters', 'L23', 'SOM', 1.65), ('letters', 'L23', 'VIP', 0.073),
    ('letters', 'L4', 'E', 3.347), ('letters', 'L4', 'PV', 0.002), ('letters', 'L4', 'SOM', 0.969), ('letters', 'L4', 'VIP', 0.059),
    ('letters', 'L5', 'E', 0.864), ('letters', 'L5', 'PV', 0.0), ('letters', 'L5', 'SOM', 1.184), ('letters', 'L5', 'VIP', 0.298),
    ('letters', 'L6', 'E', 1.582), ('letters', 'L6', 'PV', 0.01), ('letters', 'L6', 'SOM', 0.681), ('letters', 'L6', 'VIP', 0.277),
    ('hand', 'L23', 'E', 0.0), ('hand', 'L23', 'PV', 0.0), ('hand', 'L23', 'SOM', 0.0), ('hand', 'L23', 'VIP', 0.0),
    ('hand', 'L5', 'E', 0.0), ('hand', 'L5', 'PV', 0.0), ('hand', 'L5', 'SOM', 0.0), ('hand', 'L5', 'VIP', 0.372),
    ('hand', 'L6', 'E', 1.439), ('hand', 'L6', 'PV', 0.0), ('hand', 'L6', 'SOM', 0.001), ('hand', 'L6', 'VIP', 0.281),
)


def format_fitted(bias, pop) -> str:
    """Найденные числа — текстом для anatomy.py: ядро на строку, слой зоны на строку."""

    lines = ["NUCLEUS_BIAS_MV = ("]
    lines += [f"    ({name!r}, {round(mv, 2)})," for name, mv in bias]
    lines += [")", "# (зона, слой, тип клетки, множитель внешнего фона)", "COLUMN_EXTERNAL_POP = ("]
    rows: dict = {}

    for area, layer, kind, gain in pop:
        rows.setdefault((area, layer), []).append(f"({area!r}, {layer!r}, {kind!r}, {round(gain, 3)}),")

    lines += ["    " + " ".join(row) for row in rows.values()]
    lines.append(")")

    return "\n".join(lines) + "\n"


def write_fitted(bias, pop) -> str:
    """Заменить NUCLEUS_BIAS_MV и COLUMN_EXTERNAL_POP в этом файле найденными."""

    from pathlib import Path

    path = Path(__file__)
    text = path.read_text(encoding="utf-8")
    start = text.index("NUCLEUS_BIAS_MV = ")
    tail = text.index("COLUMN_EXTERNAL_POP = ")
    # Конец — закрывающая скобка в начале строки (или конец однострочной записи).
    end = text.index("\n)\n", tail) + 3 if text[tail:].startswith("COLUMN_EXTERNAL_POP = (\n") else text.index("\n", tail) + 1
    path.write_text(text[:start] + format_fitted(bias, pop) + text[end:], encoding="utf-8")

    return str(path)


def apply(cfg) -> None:
    """Достроить cfg до анатомического мозга (вызывается из Mind)."""

    names = {a.name for a in cfg.areas}
    cfg.areas = cfg.areas + [a for a in areas(cfg) if a.name not in names and (a.cells or a.share)]
    names = {a.name for a in cfg.areas}

    if cfg.hand and "purkinje" in names:
        # Выход мозжечка — через Пуркинье и ядра, а не прямой путь к руке.
        cfg.pathways = [p for p in cfg.pathways if not (p.src == "granule" and p.dst == "hand")]

    # Старый переключатель сна (VLPO ↔ LC через тормозные клетки
    # получателя) — заменён ветвями по данным: VLPO → LC синапсом ГАМК,
    # LC → VLPO норадреналином (RECEPTORS).
    cfg.pathways = [p for p in cfg.pathways if {p.src, p.dst} != {"vlpo", "lc"}]
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
