"""
Флаги мозга для веба: что можно включить при создании и что — на ходу.

Два вида (не смешивать — иначе мозг нельзя воспроизвести из его файла):
  «проводка» — задаёт сборку (связи, клетки, их число); только при
    создании. Значение хранится в конфиге мозга (cfg, сохраняется в файле
    мозга), поэтому у загруженного мозга видно, с чем он собран;
  «на ходу» — меняет работу уже собранной сети; переключается в любой
    момент, состояние тоже в cfg (сохраняется с мозгом).

У каждого флага — статус отладки (что о нём известно по замерам) и где
разобран. Флаги — опыты, не механизм по умолчанию: по умолчанию всё как в
пресете «рабочий».
"""

from __future__ import annotations

from dataclasses import dataclass

# Колонка коры при флаге «малые колонки» — не меньше этого числа клеток:
# ниже в слое остаются единицы клеток каждого типа. Это НЕ по данным:
# min_column_neurons 20 000 выбран потому, что в малых зонах (400–950 клеток)
# почти весь вход — средний ток вместо соседей и сеть идёт пачками
# (brain/config.py); 1000 клеток — тот же режим, только позволяет собрать
# мозг поменьше (ядра по данным остаются своего размера, ~99 тыс.).
SMALL_COLUMN = 1000

# Зона, где подобранный ток таламуса VA/VLa (cfg.nucleus_bias_mv) гонит руку.
VA_AREA = "thalamus"


@dataclass(frozen=True)
class Flag:
    key: str
    title: str
    kind: str          # "wiring" — только при создании; "runtime" — на ходу
    status: str        # "debugged" | "partial" | "not_debugged" | "not_data"
    note: str          # что известно по замерам
    where: str         # где разобран


FLAGS: tuple[Flag, ...] = (
    Flag("small_columns", "малые колонки коры (меньше 20 000 клеток)", "wiring", "not_data",
         "Позволяет мозг меньше ~320 тыс. нейронов (до ~110 тыс.: ядра по данным своего размера). "
         "В малой колонке почти весь вход — средний ток вместо соседей, сеть идёт пачками; "
         "подгонки фона посчитаны на колонках 20 тыс. — с пресетом нельзя.",
         "brain/config.py min_column_neurons"),
    Flag("map_ee_sheet", "E→E руки по расстоянию на листе (σ Campagnola)", "wiring", "partial",
         "Вместо «k ближайших» — связи L2/3 руки по расстоянию. Без тока VA ложных нажатий 0.1–0.3/с "
         "вместо 4–5; с током VA залпы таламуса всё равно дают 12–15/с. Гомеостаз пресета подобран без него.",
         "docs/DESIGN.md, ISSUES 21–24"),
    Flag("column_built_shares", "доли тормозных типов среди построенных (ровно Potjans)", "wiring", "partial",
         "Проводка колонки ровно Potjans (×0.99–1.03), но рабочая точка мозга под неё не согласована: "
         "у предела гомеостаза 28% клеток, рука жмёт ~8/с.",
         "docs/ISSUES.md 25"),
    Flag("nmda_chariker", "NMDA по Chariker 2016 (без блока магнием, τ 80 мс)", "wiring", "not_debugged",
         "Флаг опыта V1 (модель Chariker, Shapley & Young 2016); в мозге целиком не проверялся.",
         "brain/config.py nmda_chariker"),
    Flag("dendrite_coincidence", "дендрит — детектор совпадений (плато от ~15 синапсов за 5 мс)", "runtime", "partial",
         "Убирает постоянное плато ветвей моторных зон (100% → 0%). Гомеостаз пресета подобран без него — "
         "с ним частоты моторных зон ниже целей; свой гомеостаз есть только вне пресета.",
         "docs/ISSUES.md 26, DESIGN «Аудит веток дендрита»"),
    Flag("va_current_off", "выключить подобранный ток таламуса VA (+83 мВ)", "runtime", "not_debugged",
         "Ток VA держит таламус и гонит карту руки залпами — отсюда шум клавиш в чате. Без него VA молчит, "
         "но рука всё равно жмёт ~9–12/с от VLp, если не включён флаг E→E руки; моторные зоны ниже целей.",
         "docs/ISSUES.md 23"),
)
BY_KEY = {f.key: f for f in FLAGS}


def value(cfg, key: str) -> bool:
    if key == "small_columns":
        return cfg.min_column_neurons < 20000
    if key == "map_ee_sheet":
        return "hand" in cfg.map_ee_sheet
    return bool(getattr(cfg, key))


def configure(cfg, flags: dict[str, bool]) -> None:
    """Флаги до сборки (все виды) — в конфиг."""

    for key, on in flags.items():
        if key not in BY_KEY:
            raise ValueError(f"нет флага {key}")
        if key == "small_columns":
            cfg.min_column_neurons = SMALL_COLUMN if on else 20000
        elif key == "map_ee_sheet":
            cfg.map_ee_sheet = ("hand",) if on else ()
        else:
            setattr(cfg, key, bool(on))


def va_bias_mv(cfg) -> float:
    return dict(cfg.nucleus_bias_mv).get(VA_AREA, 0.0)


def after_build(mind) -> None:
    """После сборки нового мозга: флаги на ходу, которые меняют состояние (ток VA)."""

    if mind.cfg.va_current_off:
        b = mind.brain
        b.bias[b.areas[VA_AREA].E.slice] -= va_bias_mv(mind.cfg)


def switch(mind, key: str, on: bool) -> None:
    """Переключить флаг на ходу у собранного мозга."""

    flag = BY_KEY.get(key)
    if flag is None:
        raise ValueError(f"нет флага {key}")
    if flag.kind != "runtime":
        raise ValueError(f"«{flag.title}» задаёт проводку — только при создании мозга")
    cfg, b = mind.cfg, mind.brain
    if bool(getattr(cfg, key)) == on:
        return
    if key == "dendrite_coincidence":
        cfg.dendrite_coincidence = on
        b.set_dendrite_coincidence(on)
    elif key == "va_current_off":
        cfg.va_current_off = on
        b.bias[b.areas[VA_AREA].E.slice] += -va_bias_mv(cfg) if on else va_bias_mv(cfg)


def describe(cfg=None) -> list[dict]:
    """Флаги для веба; с cfg — значения у этого мозга (с чем он собран и что включено)."""

    return [{"key": f.key, "title": f.title, "kind": f.kind, "status": f.status, "note": f.note, "where": f.where,
             **({"value": value(cfg, f.key)} if cfg is not None else {})} for f in FLAGS]
