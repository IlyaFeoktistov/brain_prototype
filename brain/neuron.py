"""
Карточка нейрона: что знает одна клетка.

В нейробиологии это «избирательность» (tuning): на какие стимулы клетка
отвечает. Здесь у каждого нейрона три стороны:

    отвечает на   в ансамблях каких слов он стреляет (прототипы readout)
    предсказывает куда толкают его выученные исходящие синапсы — какие
                  слова он будит следующими (это и есть его «знание»)
    будят его     выученные входящие связи: от каких слов и зон

Только чтение: мозг не меняется.
"""

import torch

from brain.limbic import FACES


def _readout(language, area: str):
    return {"association": language.semantic, "motor": language.mouth}.get(area)


def _tuning(readout, local: int, top: int) -> tuple[list[dict], int]:
    """В прототипах каких слов этот нейрон и с каким весом (доля от сильнейшего в слове)."""

    n = len(readout.tokens)

    if not n:
        return [], 0

    idx, val = readout.idx[:n], readout.val[:n]
    hit = idx == local
    rows = hit.any(1).nonzero().flatten()

    if not len(rows):
        return [], 0

    mine = (val * hit).sum(1)[rows]
    share = mine / val[rows].max(1).values.clamp(min=1e-9)
    order = share.argsort(descending=True)[:top]

    return [
        {"word": readout.tokens[int(rows[i])], "share": round(float(share[i]), 3)}
        for i in order
    ], len(rows)


def _learned(brain, proj) -> torch.Tensor:
    reference = proj.birth if brain.cfg.learned_from_birth else proj.innate_max
    return proj.weight - reference


def _area_of(brain, index: int):
    for area in brain.areas.values():
        for kind, pop in area.populations:
            if pop.start <= index < pop.stop:
                return area, kind, pop

    return None, None, None


def _words(language, vector: torch.Tensor, top: int) -> list[dict]:
    return [
        {"word": word, "score": round(score, 3)}
        for word, score in language.semantic.decode(vector, top=top)
    ]


def describe(mind, index: int, top: int = 10) -> dict | None:
    brain, language = mind.brain, mind.language
    area, kind, pop = _area_of(brain, index)

    if area is None:
        return None

    local = index - pop.start
    rates = language.last_rates
    card = {
        "index": index,
        "area": area.name,
        "title": area.title,
        "type": kind,
        "local": local,
        "state": {
            "v": round(float(brain.v[index]), 2),
            "bias": round(float(brain.bias[index]), 2),
            "adapt": round(float(brain.adapt[index]), 2),
            # Частота на последнем услышанном/сказанном слове.
            "rate": None if rates is None else round(float(rates[index]), 1),
        },
    }

    # ---- отвечает на слова ----
    readout = _readout(language, area.name) if kind == "E" else None

    if readout is not None:
        card["tuning"], card["tuningWords"] = _tuning(readout, local, top)

    if area.name == "limbic" and kind == "E":
        card["feeling"] = next(
            (mood for mood in sorted(FACES)
             if bool((language._feeling_neurons(mood) == local).any())),
            None,
        )

    # ---- синапсы ----
    meaning = brain.areas["association"].E
    outgoing, incoming = {}, {}
    predicts = torch.zeros(meaning.size, device=brain.device)
    woken = torch.zeros(meaning.size, device=brain.device)

    for proj in brain.projections:
        if proj.source != "net":
            continue

        if hasattr(proj, "outgoing"):
            # Процедурный путь: связи вычисляются, вес у всех одинаковый.
            if proj.pre_pop.start <= index < proj.pre_pop.stop:
                count = len(proj.outgoing(index))
                _add(outgoing.setdefault(proj.post.name, _blank()),
                     torch.full((count,), proj.value, device=brain.device), None)
            if proj.post.start <= index < proj.post.stop:
                count = len(proj.incoming(index - proj.post.start))
                _add(incoming.setdefault(proj.pre_pop.name, _blank()),
                     torch.full((count,), proj.value, device=brain.device), None)
            continue

        learned = _learned(brain, proj) if proj.plastic else None
        out_mask = proj.pre == index

        if bool(out_mask.any()):
            weights = proj.weight[out_mask]
            entry = outgoing.setdefault(proj.post.name, _blank())
            _add(entry, weights, None if learned is None else learned[out_mask])

            if learned is not None and proj.post is meaning:
                rows = torch.arange(proj.post.size, device=brain.device)[:, None].expand_as(out_mask)
                predicts.index_add_(0, rows[out_mask], learned[out_mask].clamp(min=0.0))

        if proj.post.start <= index < proj.post.stop:
            row = index - proj.post.start
            source = proj.pre_pop.name
            entry = incoming.setdefault(source, _blank())
            _add(entry, proj.weight[row], None if learned is None else learned[row])

            if learned is not None and proj.pre_pop is meaning:
                woken.index_add_(
                    0,
                    (proj.pre[row] - meaning.start).long(),
                    learned[row].clamp(min=0.0),
                )

    card["outgoing"] = _finish(outgoing)
    card["incoming"] = _finish(incoming)
    card["predicts"] = _words(language, predicts, top)
    card["wokenBy"] = _words(language, woken, top)

    return card


# Выученным считается синапс, который опыт заметно сдвинул от рождения.
_LEARNED_EPS = 1e-3


def _blank() -> dict:
    return {"count": 0, "sum": 0.0, "plastic": 0, "stronger": 0, "weaker": 0}


def _add(entry: dict, weights: torch.Tensor, learned: torch.Tensor | None) -> None:
    entry["count"] += int(weights.numel())
    entry["sum"] += float(weights.sum())

    if learned is not None:
        entry["plastic"] += int(learned.numel())
        entry["stronger"] += int((learned > _LEARNED_EPS).sum())
        entry["weaker"] += int((learned < -_LEARNED_EPS).sum())


def _finish(groups: dict) -> list[dict]:
    return [
        {
            "population": name,
            "count": entry["count"],
            "mean": round(entry["sum"] / max(1, entry["count"]), 3),
            "plastic": entry["plastic"],
            "stronger": entry["stronger"],
            "weaker": entry["weaker"],
        }
        for name, entry in sorted(groups.items(), key=lambda item: -item[1]["count"])
    ]
