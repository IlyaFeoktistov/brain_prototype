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
    # Слуховой ствол: T-звёздчатые клетки кохлеарного ядра в тишине ~10–30
    # Гц, нижнее двухолмие наяву ~5–10 Гц [сверить по бодрствующим].
    "cn": 20.0, "ic": 8.0,
    # Ретикулярное ядро наяву ~10–30 Гц (Steriade 2005 [сверить]).
    "trn": 15.0, "trn_mgn": 15.0, "trn_md": 15.0, "trn_motor": 15.0,
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
    # Промежуточный слой верхнего двухолмия при фиксации — единицы Гц до
    # залпа саккады (Munoz & Wurtz 1995 [сверить число]).
    "colliculus": 5.0,
    # Часы: SCN днём ~6–10 Гц, ночью ~1–3 (Meijer et al. 1998, грызуны
    # in vivo) — к 8 ч утра (рождение мозга) середина, ~6 [сверить]; DMH —
    # ~5 [не найдено, сверить].
    "scn": 6.0, "dmh": 5.0,
    # Латеральная уздечка наяву — десятки Гц у обезьяны (Matsumoto &
    # Hikosaka 2007) — 20 [сверить].
    "lhb": 20.0,
}

# Частоты в медленном сне, Гц (переключатель сна, стенды sleep_switch и
# sleep_fit): VLPO 5–10 (Szymusiak et al. 1998); LC ~вдвое реже, чем наяву
# (Aston-Jones & Bloom 1981), шов — тоже (Jacobs & Fornal 1999); орексин
# почти молчит (Lee, Hassani & Jones 2005); холинергические Мейнерта в
# медленном сне редки (Lee et al. 2005 [сверить число]).
SLEEP_RATES_HZ = {"vlpo": 7.0, "lc": 1.0, "raphe": 1.0, "orexin": 0.1, "nbm": 0.5}


# Местный контур ядра: (доля местных тормозных, (входов E → I, ВПСП мВ,
# входов I → E, ТПСП мВ, входов I → I, ТПСП мВ, частота тормозных Гц[,
# входов E → E, ВПСП мВ — местное возбуждение главных, если есть])). Чего нет в таблице — у
# ядра местных тормозных нет (0): GPe, GPi/SNr, STN, стриатум (его быстрые
# тормозные — своя зона fsi), ретикулярное, VLPO, SCN, центральная
# миндалина, Пуркинье, ядра мозжечка, олива, LC, шов, Мейнерт, орексин, DMH
# — главные клетки или сами тормозные (знак выхода — в путях), или
# местного торможения в данных нет [сверить шов и LC — там есть ГАМК-клетки].
NUCLEI = {
    # Реле таламуса: местные тормозные ~25% клеток у приматов (Arcelli et
    # al. 1997), ТПСП на реле ~1–2 мВ (Cox, Zhou & Sherman 1998 [сверить]).
    **{name: (0.25, (20, 1.0, 20, -1.5, 10, -1.0, 10.0)) for name in ("lgn", "mgn", "md", "thalamus")},
    # VTA: ~35% ГАМК-клеток (Nair-Roberts et al. 2008), тормозят дофаминовые
    # [величины сверить].
    "vta": (0.35, (10, 0.5, 20, -1.0, 10, -1.0, 15.0)),
    # Базолатеральная миндалина: ~20% тормозных (McDonald 1985), корзинчатые
    # → главные ~1 мВ (Woodruff & Sah 2007 [сверить]).
    "amygdala": (0.20, (30, 1.0, 30, -1.0, 10, -1.0, 10.0)),
    # Гиппокамп: тормозных ~10% (Freund & Buzsáki 1996); пирамидная →
    # тормозная — крупные ВПСП (Gulyás et al. 1993), корзинчатая →
    # пирамидная ~1.5 мВ (Miles et al. 1996), у пирамидной клетки сотни
    # тормозных связей, ~1700 тормозных синапсов (Megías et al. 2001) —
    # 200 связей; PV-корзинчатая получает до ~16 тыс. возбуждающих синапсов,
    # в основном от местных пирамидных (Gulyás et al. 1999) — ~2000 связей:
    # торможение сразу следует за возбуждением (с 50 входами рекуррентная
    # петля CA3 уходила в «горящее» состояние). Зубчатая: тормозных ~2%
    # гранулярных [сверить], мшистые волокна бьют в основном по тормозным
    # (Acsády et al. 1998) — разреженность кода от этого, а не от «целевой
    # активности».
    **{name: (0.10, (2000, 1.0, 200, -1.5, 20, -1.0, 15.0)) for name in ("ca3", "ca1")},
    "entorhinal": (0.20, (30, 1.0, 30, -1.5, 10, -1.0, 15.0)),
    # Корзинчатых в зубчатой — 3600 на 635 тыс. гранулярных у крысы
    # (Seress & Pokorny 1981: 1 на 160–220) — 0.57%.
    "dentate": (0.0057, (30, 1.5, 30, -2.0, 10, -1.0, 15.0)),
    # Двухолмие, промежуточный слой: местные ГАМК-клетки ~30% (Mize 1992
    # [сверить]).
    "colliculus": (0.30, (20, 1.0, 20, -1.0, 10, -1.0, 10.0)),
    # Клетки Гольджи держат гранулярный код редким (Eccles 1967; Brickley et
    # al. 1996 — тоническое торможение): их ~1 на 400 гранулярных, но
    # каждая — на сотнях клеток [сверить]; ~4 входа на гранулярную.
    "granule": (0.01, (50, 0.5, 4, -2.0, 0, 0.0, 8.0)),
    # Вентральное кохлеарное ядро: тормозные (глицин) — L-звёздчатые (местные,
    # узкополосные, их большинство) и D-звёздчатые (широкополосные, ~10%;
    # Ngodup et al. 2020; доля — cfg.cn_d_stellate), и вход клубочковых клеток
    # DCN; доля тормозных среди всех клеток VCN ~10–15% [сверить: знаменателя
    # в данных нет]. Сумма: 10 входов × −1 мВ ≈ суммарный ТПСП мультиполярных
    # при оптической активации всех глициновых −7.6 ± 3.1 мВ (Xie & Manis
    # 2014). Выход D-звёздчатых тоже местный по полосе — долг (у них он
    # широкий).
    "cn": (0.15, (10, 1.0, 10, -1.0, 5, -1.0, 20.0)),
    # Нижнее двухолмие: ГАМК-клеток ~25% (Oliver et al. 1994; Merchán et
    # al. 2005), местные связи [величины сверить]. Местное возбуждение
    # главных есть, но слабее местного торможения: карты входов вдоль
    # изочастотной оси, E:I суммарного заряда 0.38 (P13–15) → ~0.1 (P19–22,
    # 3.9 / 37.8 пКл; Sturm, Nguyen & Kandler 2014, мышь): 4 × 0.5 мВ против
    # 20 × −1 мВ торможения [сверить число и ВПСП].
    "ic": (0.25, (20, 1.0, 20, -1.0, 10, -1.0, 10.0, 4, 0.5)),
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
        # Слуховой ствол: волокно нерва → кохлеарное ядро → нижнее двухолмие
        # → MGN; клеток — как волокон, место клетки — полоса улитки.
        _area("cn", "кохлеарное ядро (T-звёздчатые)", 0.0, cells=cfg.sensory_size),
        _area("ic", "нижнее двухолмие", 0.0, cells=cfg.sensory_size),
        _area("mgn", "медиальное коленчатое тело (реле слуха)", 0.0, cells=cfg.sensory_size),
        _area("md", "медиодорсальное ядро таламуса (реле ПФК)", 0.003),
        # Ретикулярное ядро — только тормозные (ГАМК), тормозят реле. Оно
        # разделено на секторы по модальностям: зрительный сектор связан с
        # LGN и V1, слуховой — с MGN и слуховой корой и т. д. (Guillery &
        # Harting 2003; Pinault 2004). Одно общее ядро связывало все реле:
        # моторная и слуховая петли через него тормозили LGN, весь таламус
        # качался вместе ~25 Гц. trn — зрительный сектор.
        _area("trn", "ретикулярное ядро: зрительный сектор", 0.00075),
        _area("trn_mgn", "ретикулярное ядро: слуховой сектор", 0.00075),
        _area("trn_md", "ретикулярное ядро: сектор MD (ПФК)", 0.00075),
        _area("trn_motor", "ретикулярное ядро: моторный сектор", 0.00075),
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
        # Латеральная уздечка — «хуже ожидаемого» (Matsumoto & Hikosaka 2007).
        _area("lhb", "латеральная уздечка (хуже ожидаемого)", 0.002),
        # Дорсомедиальный гипоталамус — выход суточных часов к узлу сна
        # (Chou et al. 2003; Aston-Jones et al. 2001).
        _area("dmh", "дорсомедиальный гипоталамус (часы → бодрствование)", 0.002),
    ]

    if cfg.eye:
        # Верхнее двухолмие, промежуточный слой: карта целей саккады — клетка
        # стоит в своём месте поля зрения, её залп ведёт взгляд туда (Robinson
        # 1972; амплитуда — средний вектор выстреливших, Lee, Rohrer & Sparks
        # 1988). Клеток — по 4 на столбец сетчатки.
        out.append(_area("colliculus", "верхнее двухолмие (куда посмотреть)", 0.0, cells=4 * cfg.retina_cols))

    if cfg.hand:
        # Дорсальная премоторная кора, «зона Экснера» (BA 6, основание
        # средней лобной извилины): буква → движение руки при письме —
        # её поражение даёт аграфию при сохранном чтении (Exner 1881;
        # Roux et al. 2009 — стимуляция у человека; Planton et al. 2013 —
        # метаанализ письма: левая дорсальная премоторная, верхняя теменная,
        # мозжечок). Агранулярная колонка (как BA 6). Учится, как вся кора.
        out.append(AreaSpec(
            "premotor", "премоторная кора, зона Экснера (буква → движение руки)",
            0.045, k_ee=10, target_activity=0.0,
        ))

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
    sector = {"lgn": "trn", "mgn": "trn_mgn", "md": "trn_md", "thalamus": "trn_motor"}

    for cortex, relay in relays:
        out += [
            _path(cortex, relay, 50, 0.3, "ct"),
            _path(cortex, sector[relay], 50, 0.7, "ct"),
        ]

    for relay, trn in sector.items():
        out += [
            _path(relay, trn, 20, 1.5),
            _path(trn, relay, 20, -1.5),
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
    # Входов и PSP — по измерениям, собранным в модели Lindahl & Hellgren
    # Kotaleski 2016 (eNeuro; параметры — github mickelindahl/bgmodel);
    # проводимости пересчитаны в PSP на пассивной мембране получателя
    # (ёмкость, утечка, реверсия ГАМК — оттуда же): D1 → GPi/SNr ~500
    # входов по 2.2 мВ (Connelly et al. 2010), D2 → GPe ~500 по 2.2 (Sims et
    # al. 2008), STN → GPe 30 по 3.5 (Hanson & Jaeger 2002), GPe → GPe 25 по
    # 1.2 (Sims 2008), GPe → GPi/SNr 32 крупных перисоматических (Connelly
    # 2010), STN → GPi/SNr 30 (Shen & Johnson 2006). У этих двух — сильная
    # кратковременная депрессия (Tsodyks: U 0.2 / 0.35, восстановление ~1 /
    # 0.8 с): ~27 и ~15 мВ — первый спайк после тишины, а при частотах in
    # vivo (GPe 40, STN 15 Гц) синапс устаёт до ~12% и ~19% — 3.3 и 2.9 мВ.
    # Пока — устоявшаяся величина; кратковременная пластичность путей ядер —
    # долг (нужна залпам «давай»). GPe → STN в модели не измерено — 20 по 1 мВ
    # [сверить по Baufreton et al. 2009]; FSI → MSN — крупные IPSP (Koós &
    # Tepper 1999), ~10 входов (Lindahl: 11–16).
    out += [
        _path("fsi", "striatum_d1", 16, -1.5),
        _path("fsi", "striatum_d2", 11, -1.5),
        _path("striatum_d1", "gpi", 500, -2.2),
        _path("striatum_d2", "gpe", 500, -2.2),
        _path("stn", "gpe", 30, 3.5),
        _path("stn", "gpi", 30, 2.9),
        _path("gpe", "gpe", 25, -1.2, name="gpe: коллатерали"),
        _path("gpe", "gpi", 32, -3.3),
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
    # ---- часы → узел сна (Saper, Scammell & Lu 2005) ----
    # SCN → (субпаравентрикулярная зона) → DMH; DMH → LC (Aston-Jones et
    # al. 2001), глутаматом → орексин, ГАМК → VLPO (Chou et al. 2003). PSP
    # [сверить]; прежний прямой SCN → LC (вес 3, подбирался) убран.
    out += [
        _path("scn", "dmh", 20, 1.0),
        _path("dmh", "lc", 20, 0.5),
        _path("dmh", "orexin", 20, 0.5),
        _path("dmh", "vlpo", 20, -1.0),
    ]
    # ---- награда ----
    # Уздечка → RMTg (ГАМК) → VTA — одним путём со знаком «−» (Jhou et al.
    # 2009); базальные ганглии → уздечка (через бледный шар, Hong &
    # Hikosaka 2008) — выход GPi. ПФК → VTA (Carr & Sesack 2000). PSP
    # [сверить].
    out += [
        _path("lhb", "vta", 20, -1.0),
        _path("gpi", "lhb", 20, 0.5),
        _path("prefrontal", "vta", 30, 0.3),
    ]
    # ---- коллатерали внутри тормозных ядер ----
    # MSN → MSN (Taverna, Ilijic & Surmeier 2008; числа входов — Lindahl &
    # Hellgren Kotaleski 2016): D1→D1 ~364, D1→D2 ~84, D2→D1 ~393, D2→D2
    # ~504, ТПСП ~0.5 мВ [сверить]; FSI → FSI ~9 (Gittis et al. 2010), ~1 мВ
    # [сверить].
    out += [
        _path("striatum_d1", "striatum_d1", 50, -0.5, k_data=364, name="striatum_d1: коллатерали"),
        _path("striatum_d1", "striatum_d2", 50, -0.5, k_data=84),
        _path("striatum_d2", "striatum_d1", 50, -0.5, k_data=393),
        _path("striatum_d2", "striatum_d2", 50, -0.5, k_data=504, name="striatum_d2: коллатерали"),
        _path("fsi", "fsi", 9, -1.0, name="fsi: коллатерали"),
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
    # ---- гиппокамп (Amaral, Ishizuka & Claiborne 1990; сводка — Rolls
    # 2013; входы CA1 — Megías et al. 2001) ----
    # Входов на клетку (k_data) и единичный ВПСП: EC → гранулярная ~4000
    # по ~0.1 мВ (McNaughton, Barnes & Andersen 1981); мшистые волокна → CA3
    # ~46 крупных «детонаторов» ~1 мВ (Henze et al. 2002; сильно облегчаются);
    # EC → CA3 ~3600 по ~0.1 [сверить]; CA3 → CA1 (Шаффер) ~20 тыс. по 0.13
    # (Sayer, Friedlander & Redman 1990); EC слой III → CA1 ~1700 по ~0.1
    # [сверить]. Вход коры в EC и выход CA1 в кору — как пути коры (0.15 мВ),
    # входов [не найдено, сверить]. Запись эпизода — пластичность при
    # ацетилхолине бодрствования (Hasselmo 2006), как было.
    # Прямое торможение (k_data_inhibitory): входы путей идут и на местные
    # тормозные клетки — они в тонусе (~15 Гц наяву) ещё до того, как
    # пирамиды проснутся; без этого рекуррентная петля CA3 (72 мВ на 1 Гц)
    # убегала из 0 в 400 Гц раньше, чем тормозные доходили до порога, а
    # покоя ~1 Гц не было ни при каком фоне. Доля синапсов пути на
    # тормозных (Takács et al. 2012, электронная микроскопия): Шаффер 7.1%,
    # EC в lacunosum-moleculare 9.2%; на одну тормозную клетку — 9
    # пирамидных (тормозных ~10%): Шаффер 20000 × 0.071/0.929 × 9 ≈ 13.8
    # тыс. — сходится с ~15 тыс. возбуждающих синапсов PV-клетки CA1
    # (Gulyás et al. 1999: 16.3 тыс., из них ГАМК 6.4%). EC → CA1 1700 ×
    # 0.092/0.908 × 9 ≈ 1550; EC → CA3 — та же доля [сверить]: 3600 → 3300.
    # Мшистые волокна: тонких окончаний на тормозные в 10 раз больше, чем
    # крупных на пирамиды (Acsády et al. 1998) — 46 × 10 × 9 ≈ 4100 [часть
    # — на тормозные хилуса, сверить]. PSP на тормозной: Шаффер и EC — как
    # на пирамидной [сверить]; одиночный тонкий мшистый синапс — 0.3 мВ
    # [сверить; облегчения, как у «детонатора», нет — Tóth et al. 2000].
    out += [
        _path("association", "entorhinal", 100, 0.15, k_data=1000, topographic=1.5),
        _path("prefrontal", "entorhinal", 50, 0.15, k_data=500),
        _path("entorhinal", "dentate", 200, 0.1, k_data=4000),
        _path("dentate", "ca3", 46, 1.0, k_data_inhibitory=4100, psp_inhibitory_mv=0.3),
        _path("entorhinal", "ca3", 200, 0.1, k_data=3600, plastic=True, modulator="acetylcholine",
              k_data_inhibitory=3300, psp_inhibitory_mv=0.1),
        _path("ca3", "ca1", 200, 0.13, k_data=20000, plastic=True, modulator="acetylcholine", ach_suppressed=True,
              k_data_inhibitory=13800, psp_inhibitory_mv=0.13),
        _path("entorhinal", "ca1", 200, 0.1, k_data=1700, k_data_inhibitory=1550, psp_inhibitory_mv=0.1),
        _path("ca1", "association", 200, 0.15, plastic=True, modulator="acetylcholine", ach_suppressed=True),
        # Рекуррентные коллатерали CA3: ~12 тыс. синапсов (Amaral 1990) —
        # ~6 тыс. связей по ~2 синапса, связь ~0.6 мВ (Guzman et al. 2016;
        # Miles & Wong 1986 — до ~1), STDP симметричное (Mishra et al. 2016 —
        # AreaSpec.ee_symmetric), пишутся при ацетилхолине. Прежние 200 по
        # 0–0.05 / потолок 3 подбирались стендом (связь зоны выключена в apply).
        _path("ca3", "ca3", 200, 0.6, k_data=6000, plastic=True, modulator="acetylcholine",
              ach_suppressed=True, name="ca3: коллатерали"),
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
    # [сверить]); Пуркинье → ядра — крупные IPSP (сходятся ~40 на клетку
    # ядра, спад IPSC ~2.5 мс; Person & Raman 2012), ~2 мВ [сверить] у
    # первого спайка; при частоте Пуркинье in vivo (~50 Гц) синапс устаёт на
    # ~60% (Telgkamp & Raman 2002) — 0.8 мВ (без этого торможение ядер было
    # ~80 мВ, и подбор навешивал на ядра +89 мВ фона; кратковременная
    # пластичность путей ядер — долг, как у базальных ганглиев); олива →
    # Пуркинье — одно лазающее
    # волокно, сложный спайк; ядра → таламус VA/VL → моторная кора.
    if cfg.hand:
        out += [
            # Учится по лазающему волокну (modulator "climbing", engine.
            # chemical_plasticity): LTD при совпадении с ним, слабая LTP без
            # него (Ito 1989; Coesmans et al. 2004).
            _path("granule", "purkinje", 1000, 0.05, plastic=True, modulator="climbing"),
            _path("purkinje", "dcn", 40, -0.8),
            _path("io", "purkinje", 1, 10.0),
            _path("dcn", "thalamus", 20, 1.0),
            # Коллатерали мшистых волокон — к ядрам мозжечка (Eccles 1967):
            # ядра возбуждает тот же вход, что гранулярные клетки (PSP
            # [сверить]).
            _path("premotor", "dcn", 20, 0.5),
            _path("hand", "dcn", 20, 0.5),
            _path("visual", "dcn", 20, 0.5),
            _path("dcn", "io", 10, -1.0),
            # Олива — сравнение: «что увидел» (зрительная кора → претектум /
            # мезодиэнцефальное соединение → олива) и «что делал» (моторная
            # кора → красное ядро → олива); ядра мозжечка гасят ожидаемое
            # (De Zeeuw et al. 1998). PSP [сверить].
            _path("letters", "io", 20, 0.5),
            _path("premotor", "io", 20, 0.5),
        ]
    out += cortex_chain(cfg)
    # ---- взгляд (Hikosaka, Takikawa & Kawagoe 2000) ----
    # Карту двухолмия даёт сетчатка (боковое зрение, BrainConfig.sensors);
    # V1 L5 → двухолмие (Lund 1975) — долг: карты ямки и поля в разных
    # координатах. Лобное
    # глазное поле (здесь — ПФК) → двухолмие (Sommer & Wurtz 2000); выход
    # базальных ганглиев (SNr) тонически тормозит двухолмие, снятие
    # торможения отпускает саккаду (Hikosaka & Wurtz 1983); копия саккады —
    # двухолмие → MD → лобная кора (Sommer & Wurtz 2002). PSP [сверить].
    out += [
        _path("prefrontal", "colliculus", 50, 0.3),
        _path("gpi", "colliculus", 20, -1.5),
        _path("colliculus", "md", 20, 0.5),
    ]
    return out


# Кора целиком (ARCHITECTURE 11): чтение — сетчатка → LGN → V1 → буквы →
# смысл → ПФК и цель; письмо — смысл → буквы (правописание: лексический путь
# в обе стороны, Rapcsak & Beeson 2004) → зона Экснера → рука (Planton et
# al. 2013). Все связи коры учатся (STDP), число входов и PSP — по данным
# (engine._cortical_path: 21% входа клетки — из других зон, Markov 2011;
# PSP 0.15 мВ). Направления по слоям — Felleman & Van Essen 1991: прямой
# (ff) — к высшей ступени, обратный (fb) — к низшей, боковой — равные.
LEARNED_CORTEX = (
    # чтение
    ("letters", "association", "ff"),
    ("auditory", "association", "ff"),
    ("association", "prefrontal", "ff"), ("prefrontal", "association", "fb"),
    ("association", "goal", "ff"), ("goal", "association", "fb"),
    ("association", "category", "ff"), ("category", "association", "fb"),
    ("limbic", "association", "fb"),
    ("association", "motor", "lateral"),
    # письмо
    ("association", "letters", "fb"),
    ("letters", "premotor", "ff"), ("prefrontal", "premotor", "fb"),
    ("premotor", "hand", "fb"), ("hand", "premotor", "ff"),
)

# Пути коры прежней схемы между теми же зонами (веса подбирались под
# стенды) — заменяются LEARNED_CORTEX.
REPLACED_CORTEX = {(s, d) for s, d, _ in LEARNED_CORTEX}
REPLACED_OLD = {("striatum_d1", "gpi"), ("striatum_d2", "gpe"), ("gpe", "gpi"), ("scn", "lc"),
                          # гиппокамп — по данным (pathways)
                          ("association", "entorhinal"), ("prefrontal", "entorhinal"), ("entorhinal", "dentate"),
                          ("dentate", "ca3"), ("entorhinal", "ca3"), ("ca3", "ca1"), ("entorhinal", "ca1"),
                          ("ca1", "association")}


def cortex_chain(cfg) -> list[PathwaySpec]:
    out = [
        PathwaySpec(src, dst, k=1, weight=(0.0, 0.0), plastic=True, k_inhibitory=1, predictive=False,
                    modulator="hebbian", direction=direction, learned=True)
        for src, dst, direction in LEARNED_CORTEX
    ]
    # Кора → стриатум: в стриатум проецирует вся кора (из L5; Kemp & Powell
    # 1970; McGeorge & Faull 1989). У клетки стриатума ~5–10 тыс. корковых
    # синапсов, почти каждый — от своего аксона, ВПСП единичного входа ~0.1–
    # 0.3 мВ (Kincaid, Zheng & Wilson 1998 [сверить величину]) — 0.2;
    # недостающее — фон ядра. Учатся по трём факторам: D1 — от дофамина, D2 —
    # от его провала (Reynolds & Wickens 2002; Shen et al. 2008).
    for src in ("association", "goal", "motor", "premotor", "hand"):
        out += [
            _path(src, "striatum_d1", 50, 0.2, plastic=True),
            _path(src, "striatum_d2", 50, 0.2, plastic=True, modulator="dopamine_d2"),
        ]
    # Моторный таламус: VA → премоторная (Jones 2007), как VL → M1; L6
    # премоторной и руки → своё реле и ретикулярное (Sherman & Guillery).
    out += [
        _path("thalamus", "premotor", 20, 0.6, "thalamic"),
        _path("premotor", "thalamus", 50, 0.3, "ct"),
        _path("premotor", "trn_motor", 50, 0.7, "ct"),
        _path("hand", "thalamus", 50, 0.3, "ct"),
        _path("hand", "trn_motor", 50, 0.7, "ct"),
        # Гиперпрямой путь и вход быстрых тормозных стриатума — от моторной
        # коры, как от ПФК (Nambu et al. 2002; Mallet et al. 2005).
        _path("premotor", "stn", 50, 0.5),
        _path("premotor", "fsi", 50, 1.0),
    ]
    # Кора → мост → мшистые волокна → гранулярные клетки (копия команды;
    # Schmahmann & Pandya 1997): у гранулярной клетки ~4 мшистых волокна
    # (Eccles 1967) — 2 из зрительной коры (буква), по одному из премоторной
    # и моторной руки; клетке нужны 2 совпавших (Chadderton et al. 2004).
    out += [
        _path("premotor", "granule", 1, 7.0),
        _path("hand", "granule", 1, 7.0),
    ]
    return out


# Рецепторы (вещество, зона[:тип], мВ при полной занятости). Знак — по
# данным, величина [не измерено в наших единицах]:
# - ацетилхолин: никотиновые возбуждают VIP (Fu et al. 2014; Letzkus et al.
#   2011), мускариновые M1 деполяризуют пирамидные клетки (McCormick & Prince
#   1986) и SOM (Kawaguchi 1997; Fanselow et al. 2008; Chen, Sugihara & Sur
#   2015) — без этого SOM в покое молчат (вход E → SOM слабый и облегчающийся,
#   при 1–2 Гц почти не действует), и VIP, которых тормозят в основном SOM,
#   горят 20–47 Гц (VIP ⊣ SOM ⊣ VIP); величина как у VIP [сверить];
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
    ("acetylcholine", ":SOM", 5.0),
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
# ствола и коры, которых в модели нет. Подбор — tools/anatomy_fit.py
# (весь мозг small, шаг 0.5 мс, частоты наяву RATES_HZ), кроме узла сна
# (VLPO, LC, шов, орексин, Мейнерт): он двустабилен, его подбирает
# tools/sleep_fit.py по двум состояниям — наяву (RATES_HZ) и в
# медленном сне (SLEEP_RATES_HZ).
# COLUMN_EXTERNAL_POP — множитель внешнего фона колонки (вход из мозга вне
# модели, Potjans минус явные пути) по зоне, слою и типу клетки; подбор —
# anatomy_fit под column_target_hz. Внешний фон L2/3 и L5 ушёл почти в 0 —
# вероятно двойной учёт (вход Potjans уже содержит таламус и другие зоны,
# у нас они явные) — разобрать. Выше нормы: PV L2/3 (~40 против 6.4).
NUCLEUS_BIAS_MV = (
    ('lc', 20.3),
    ('vlpo', 18.56),
    ('raphe', -3.33),
    ('nbm', 14.79),
    ('orexin', -1.07),
    ('vta', 28.12),
    ('amygdala', 9.25),
    ('scn', -6.69),
    ('striatum_d1', 12.87),
    ('striatum_d2', 27.83),
    ('gpe', 55.53),
    ('gpi', 136.68),
    ('thalamus', 65.78),
    ('entorhinal', -10.92),
    ('dentate', -122.69),
    ('ca3', 72.89),
    ('ca1', 98.02),
    ('granule', 6.18),
    ('lgn', 28.01),
    ('mgn', 19.44),
    ('md', 21.82),
    ('trn', 14.03),
    ('trn_mgn', 11.24),
    ('trn_md', 15.33),
    ('trn_motor', 12.09),
    ('fsi', 27.9),
    ('stn', 21.19),
    ('cea', 15.43),
    ('lhb', 10.93),
    ('dmh', 12.84),
    ('colliculus', 40.1),
    ('purkinje', 15.42),
    ('dcn', 68.94),
    ('io', 0.28),
    ('cn', 18.1),
    ('ic', 11.06),
)
# (зона, слой, тип клетки, множитель внешнего фона)
COLUMN_EXTERNAL_POP = (
    ('auditory', 'L23', 'E', 0.533), ('auditory', 'L23', 'PV', 0.088),
    ('auditory', 'L4', 'E', 0.141), ('auditory', 'L4', 'PV', 0.126),
    ('auditory', 'L5', 'E', 0.712), ('auditory', 'L5', 'PV', 0.309),
    ('auditory', 'L6', 'E', 1.087), ('auditory', 'L6', 'PV', 0.56),
    ('association', 'L23', 'E', 2.0), ('association', 'L23', 'PV', 0.376),
    ('association', 'L4', 'E', 1.515), ('association', 'L4', 'PV', 0.796),
    ('association', 'L5', 'E', 1.687), ('association', 'L5', 'PV', 1.468),
    ('association', 'L6', 'E', 1.223), ('association', 'L6', 'PV', 1.547),
    ('prefrontal', 'L23', 'E', 0.359), ('prefrontal', 'L23', 'PV', 0.042),
    ('prefrontal', 'L4', 'E', 0.114), ('prefrontal', 'L4', 'PV', 0.08),
    ('prefrontal', 'L5', 'E', 0.466), ('prefrontal', 'L5', 'PV', 0.185),
    ('prefrontal', 'L6', 'E', 1.062), ('prefrontal', 'L6', 'PV', 0.334),
    ('motor', 'L23', 'E', 2.874), ('motor', 'L23', 'PV', 0.658),
    ('motor', 'L4', 'E', 0.094), ('motor', 'L4', 'PV', 0.193),
    ('motor', 'L5', 'E', 1.997), ('motor', 'L5', 'PV', 2.048),
    ('motor', 'L6', 'E', 1.29), ('motor', 'L6', 'PV', 2.363),
    ('goal', 'L23', 'E', 0.374), ('goal', 'L23', 'PV', 0.044),
    ('goal', 'L4', 'E', 0.113), ('goal', 'L4', 'PV', 0.082),
    ('goal', 'L5', 'E', 0.462), ('goal', 'L5', 'PV', 0.198),
    ('goal', 'L6', 'E', 1.06), ('goal', 'L6', 'PV', 0.346),
    ('limbic', 'L23', 'E', 3.561), ('limbic', 'L23', 'PV', 1.148),
    ('limbic', 'L5', 'E', 2.126), ('limbic', 'L5', 'PV', 2.488),
    ('limbic', 'L6', 'E', 1.395), ('limbic', 'L6', 'PV', 2.74),
    ('category', 'L23', 'E', 2.45), ('category', 'L23', 'PV', 0.438),
    ('category', 'L4', 'E', 1.59), ('category', 'L4', 'PV', 1.151),
    ('category', 'L5', 'E', 1.71), ('category', 'L5', 'PV', 1.646),
    ('category', 'L6', 'E', 1.225), ('category', 'L6', 'PV', 1.754),
    ('visual', 'L23', 'E', 0.411), ('visual', 'L23', 'PV', 0.059),
    ('visual', 'L4', 'E', 0.128), ('visual', 'L4', 'PV', 0.073),
    ('visual', 'L5', 'E', 0.494), ('visual', 'L5', 'PV', 0.134),
    ('visual', 'L6', 'E', 1.177), ('visual', 'L6', 'PV', 0.243),
    ('letters', 'L23', 'E', 1.543), ('letters', 'L23', 'PV', 0.236),
    ('letters', 'L4', 'E', 0.828), ('letters', 'L4', 'PV', 0.264),
    ('letters', 'L5', 'E', 1.798), ('letters', 'L5', 'PV', 0.418),
    ('letters', 'L6', 'E', 1.283), ('letters', 'L6', 'PV', 0.692),
    ('hand', 'L23', 'E', 1.948), ('hand', 'L23', 'PV', 0.276),
    ('hand', 'L4', 'E', 0.092), ('hand', 'L4', 'PV', 0.182),
    ('hand', 'L5', 'E', 1.968), ('hand', 'L5', 'PV', 0.905),
    ('hand', 'L6', 'E', 1.143), ('hand', 'L6', 'PV', 1.184),
    ('premotor', 'L23', 'E', 0.295), ('premotor', 'L23', 'PV', 0.035),
    ('premotor', 'L5', 'E', 1.803), ('premotor', 'L5', 'PV', 0.33),
    ('premotor', 'L6', 'E', 1.22), ('premotor', 'L6', 'PV', 0.412),
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


def save_fit(name: str, bias, pop) -> str:
    """
    Фон диагностического режима (обход, не норма) — в ~/.brain/fits/<имя>.json,
    не в этот файл: здесь — фон мозга без обходов.
    """

    import json

    from brain.config import BRAIN_HOME

    path = BRAIN_HOME / "fits" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"bias": [list(b) for b in bias], "pop": [list(p) for p in pop]}, ensure_ascii=False), encoding="utf-8")

    return str(path)


def use_fit(cfg, name: str) -> None:
    """Взять фон из ~/.brain/fits/<имя>.json (save_fit) вместо чисел этого файла."""

    import json

    from brain.config import BRAIN_HOME

    data = json.loads((BRAIN_HOME / "fits" / f"{name}.json").read_text(encoding="utf-8"))
    cfg.nucleus_bias_mv = tuple(tuple(b) for b in data["bias"])
    cfg.column_gain_external_pop = tuple(tuple(p) for p in data["pop"])


def apply(cfg) -> None:
    """Достроить cfg до анатомического мозга (вызывается из Mind)."""

    names = {a.name for a in cfg.areas}
    cfg.areas = cfg.areas + [a for a in areas(cfg) if a.name not in names and (a.cells or a.share)]
    # Тормозные клетки ядер — без сдвига к порогу: 8 мВ (i_excitability) —
    # подгонка старой ассоциативной зоны («торможение раньше залпа»), не
    # данные; в ядрах ствола и таламуса она давала тормозным клеткам тот же
    # вход, что главным, и фору к порогу — прямое торможение съедало ответ
    # (DESIGN «Аудит», «MGN: ряд силы интернейронов»). Колонки коры —
    # своё расстояние до порога по типу (pv_gap_mv и др.).
    cfg.i_excitability = 0.0
    # Окно STDP коры по данным: L2/3 зрительной коры, пары спайков — LTP
    # +103% · exp(−Δt/13.5 мс), LTD −51% · exp(Δt/34.5 мс) (Froemke & Dan
    # 2002 [сверить амплитуды]): узкое окно LTP без сдвига пика, широкое LTD,
    # по интегралу LTD чуть сильнее (1.26) — некоррелированный фон веса не
    # растит (Song, Miller & Abbott 2000). У человека (L2/3 височной коры)
    # окно LTP шире (Testa-Silva et al. 2010; Verhoog et al. 2013) — долг.
    # Прежнее окно (LTP exp(−t/80) − exp(−t/30), LTD τ 40) подбиралось под
    # переходы слов старого мозга и по интегралу было втрое за LTP.
    cfg.stdp_tau_plus, cfg.stdp_ltp_rise, cfg.stdp_tau_minus = 13.5, 0.0, 34.5
    cfg.stdp_a_minus = cfg.stdp_a_plus * 51.0 / 103.0
    cfg.stdp_ltd_weight_dependent = True
    cfg.heterosynaptic_conservation = True
    names = {a.name for a in cfg.areas}

    if cfg.hand and "purkinje" in names:
        # Выход мозжечка — через Пуркинье и ядра, а не прямой путь к руке.
        cfg.pathways = [p for p in cfg.pathways if not (p.src == "granule" and p.dst == "hand")]

    # Старый переключатель сна (VLPO ↔ LC через тормозные клетки
    # получателя) — заменён ветвями по данным: VLPO → LC синапсом ГАМК,
    # LC → VLPO норадреналином (RECEPTORS).
    cfg.pathways = [p for p in cfg.pathways if {p.src, p.dst} != {"vlpo", "lc"}]
    # Пути коры прежней схемы между теми же зонами, прежний вход коры в
    # стриатум и связи базальных ганглиев (веса подбирались под стенды) —
    # заменяют cortex_chain и pathways (по данным).
    cfg.pathways = [
        p for p in cfg.pathways
        if (p.src, p.dst) not in REPLACED_CORTEX
        and not (p.dst in ("striatum_d1", "striatum_d2") and p.src in cfg.layered_areas)
        and (p.src, p.dst) not in REPLACED_OLD
    ]

    # Мшистые волокна гранулярной клетки: 2 из зрительной коры (было 4 —
    # теперь ещё 2 из моторной коры, cortex_chain).
    # Синапс мшистого волокна на гранулярной клетке — крупный: клетке нужны
    # 2–3 совпавших входа из ~4 (Chadderton, Margrie & Häusser 2004) —
    # ~7 мВ [сверить]; прежние 15 подбирались стендом.
    cfg.pathways = [replace(p, k=2, psp_mv=7.0, k_inhibitory=0) if (p.src, p.dst) in (("complex", "granule"), ("visual", "granule")) else p
                    for p in cfg.pathways]
    # GPi → таламус —
    # крупные терминали (Bodor et al. 2008), ~2 мВ [сверить]; смысл →
    # базолатеральная миндалина — путь коры (0.15 мВ), учится дофамином
    # (раньше вес ставил код — language._learn_value).
    # Ядра по данным (NUCLEI): местный контур, без возбуждающих связей
    # внутри «по умолчанию» (у реле таламуса и STN их нет, тормозные ядра
    # друг друга не возбуждают; коллатерали — путями: CA3, MSN, GPe), без
    # подобранных фона, торможения, обратной связи, целевой активности и
    # гомеостаза — частоту ядра держит согласуемый фон (частоты in vivo).
    # Привыкание LC — α2-ауторецептор (химия), не подобранная адаптация.
    layered = set(cfg.layered_areas) | {"complex", "visual", "letters", "hand", "premotor"}
    cfg.areas = [
        replace(
            a, ee_plastic=False, ee_weight=(0.0, 0.0), background=None, inhibition=1.0, feedback=1.0,
            target_activity=0.0, homeostasis=1.0, inhibition_adapts=False,
            adapt_b=0.0 if a.name == "lc" else a.adapt_b,
            local_inhibitory=NUCLEI.get(a.name, (0.0, ()))[0], local_psp=NUCLEI.get(a.name, (0.0, ()))[1],
        ) if a.name not in layered else a
        for a in cfg.areas
    ]
    # Клетки окончания звука — клетки самой слуховой коры (Scholl et al.
    # 2010), не отдельное ядро: в анатомическом мозге пока без них.
    cfg.areas = [a for a in cfg.areas if a.name != "offset"]
    cfg.pathways = [p for p in cfg.pathways if "offset" not in (p.src, p.dst)]
    # GPi → таламус: крупные окончания с многими синапсами (Bodor et al.
    # 2008 — 8.5 синапса на окончание у SNr → таламус; у GPi не мерено);
    # моносинаптический IPSP 1.6 мс (Uno, Ozawa & Yoshida 1978, кошка);
    # только ГАМК-А, депрессирует на 10 и 40 Гц, вход GPi — у 32 из 42 клеток
    # VA/VL (Koster & Sherman 2024, мышь; PPR и амплитуды не приведены).
    # Сходимость, амплитуда, коэффициент депрессии не найдены [сверить] —
    # при 60 Гц GPi ~70 мВ торможения, подбор держит VA/VL фоном ~+55 мВ.
    # Долг: у кошки клетки VA/VL с входом GPi и с входом мозжечка — разные
    # части ядра (Uno 1978), у нас — один таламус.
    fixed = {
        ("gpi", "thalamus"): dict(psp_mv=-2.0, k_inhibitory=0),
        ("association", "amygdala"): dict(psp_mv=0.15, k_inhibitory=0, plastic=True, modulator="dopamine"),
    }
    cfg.pathways = [replace(p, **fixed[(p.src, p.dst)]) if (p.src, p.dst) in fixed else p for p in cfg.pathways]

    if "premotor" in names:
        cfg.layered_areas = tuple(cfg.layered_areas) + ("premotor",)
        cfg.microcircuit_areas = tuple(cfg.microcircuit_areas) + ("premotor",)
        cfg.agranular_areas = tuple(cfg.agranular_areas) + ("premotor",)

    # Рука — колонка по данным: «колонки клавиш» (связи внутри набора клеток
    # клавиши 8 мВ) и торможение ×4 подбирались под стенд — убраны; выбор
    # одной клавиши — работа PV колонки и базальных ганглиев.
    cfg.areas = [replace(a, columns=0, column_weight=0.0, inhibition=1.0, feedback=1.0) if a.name == "hand" else a
                 for a in cfg.areas]
    # V1 → буквы: обучаемый путь по данным (входов и PSP — как у всех путей
    # коры), с картой поля (окно 12 px — поле V4 в 2–3 раза шире поля V1,
    # Dehaene et al. 2005); прежние 0–0.5 / потолок 3 подбирались стендом.
    cfg.pathways = [
        replace(p, learned=True, direction="ff", modulator="hebbian")
        if (p.src, p.dst) in (("visual", "letters"), ("complex", "letters")) else p
        for p in cfg.pathways
    ]

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
