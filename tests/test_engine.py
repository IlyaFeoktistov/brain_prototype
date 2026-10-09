"""Движок: устойчивость, разреженность, read-only, сохранение, структурная пластичность."""

import os

import torch

from brain.mind import Mind
from tests.conftest import FACTS, make_mind


def test_no_nan_and_no_lost_spikes(mind):
    mind.language.read_text(FACTS)
    brain = mind.brain

    assert torch.isfinite(brain.net_weight).all()
    assert torch.isfinite(brain.v[: brain.n]).all()
    # Таблица событий не должна переполняться (иначе спайки теряются).
    assert float(brain.dropped) == 0


def test_word_activates_sparse_ensemble():
    # [замер] на tiny (2k) смысловая зона молчит хотя бы на одно из слов
    # «париж/берлин/компьютер» в 19 из 40 seed (с лимбической зоной —
    # 17 из 40): слишком мало клеток для надёжного ансамбля. На small
    # (20k) — 0 из 10 seed.
    mind = make_mind("small")
    mind.language.read_text("Столица Франции Париж.")
    probe = mind.language.probe("париж")

    for area in mind.brain.areas.values():
        active = float((probe["rates"][area.E.slice] > 0).float().mean())
        # Разреженный код: отвечает малая часть зоны, но не ноль.
        assert active < 0.20, f"{area.name}: {active:.1%} — зона взорвалась"

    # Смысл: стреляет место слова (понятие), не меньше половины его клеток.
    meaning = mind.brain.areas["association"].E
    place = mind.language.concepts.get("париж", allocate=False)
    assert int((probe["rates"][meaning.start + place] > 0).sum()) >= mind.cfg.concept_cells // 2


def test_same_word_same_ensemble(mind):
    language = mind.language
    # У слова есть место в смысле, только когда мозг его слышал.
    language.read_text("Берлин и компьютер.")
    a = language.probe("берлин")["meaning"]
    b = language.probe("берлин")["meaning"]
    c = language.probe("компьютер")["meaning"]

    cos = lambda x, y: float(x @ y / (x.norm() * y.norm() + 1e-9))
    assert cos(a, b) > cos(a, c) + 0.1


def test_diagnostics_do_not_change_brain(mind):
    language = mind.language
    language.read_text(FACTS)

    weights = mind.brain.net_weight.clone()
    eligibility = mind.brain.eligibility.clone()
    tokens = language.experienced_tokens
    words = len(language.semantic)

    language.debug_prediction("столица франции")
    language.compare("открыть файл", "закрыть файл")
    language.debug_word("париж")
    language.generate("столица германии")

    assert torch.equal(weights, mind.brain.net_weight)
    assert torch.equal(eligibility, mind.brain.eligibility)
    assert language.experienced_tokens == tokens
    assert len(language.semantic) == words


def test_save_load_roundtrip(mind, tmp_dir):
    mind.language.read_text(FACTS)
    path = os.path.join(tmp_dir, "roundtrip.pt")
    mind.save(path)

    loaded = Mind.load(path)

    assert torch.equal(mind.brain.net_weight.cpu(), loaded.brain.net_weight.cpu())
    assert mind.language.semantic.tokens == loaded.language.semantic.tokens
    assert mind.language.confidence.total == loaded.language.confidence.total
    assert loaded.cfg.save_path == path
    # Загруженный мозг живой: читает дальше без ошибок.
    loaded.language.read_text("Столица Испании Мадрид.")


def test_rewire_replaces_only_weak_synapses(mind):
    brain = mind.brain
    proj = brain.plastic[0]

    proj.weight[:5, 0] = 0.0      # ослабли до нуля — должны отмереть
    proj.weight[5:10, 0] = 2.5    # сильные — память, не трогать
    strong_pre = proj.pre[5:10, 0].clone()

    grown = brain.rewire()

    assert grown >= 5
    assert (proj.weight[:5, 0] > 0).all()
    assert torch.equal(proj.pre[5:10, 0], strong_pre)
    assert (proj.weight[5:10, 0] == 2.5).all()
    # Таблицы пересобраны — симуляция продолжает работать.
    mind.language.read_text("Столица Франции Париж.")
    assert float(brain.dropped) == 0


def test_similar_forms_sound_similar(mind):
    ear = mind.language.ear

    def sound(word):
        return ear.encode(word).sum(0)

    cos = lambda x, y: float(x @ y / (x.norm() * y.norm() + 1e-9))
    # Слово звучит по буквам: формы похожи, чужое слово — нет.
    assert cos(sound("франция"), sound("франции")) > cos(sound("франция"), sound("кошка")) + 0.3


def test_synapses_grow_where_transition_is_learned(mind):
    language, brain = mind.language, mind.brain
    mind.cfg.grow_every_sentences = 1
    language.read_text("Столица Италии Рим.")

    proj = next(p for p in brain.plastic if p.name == "association: E->E")
    start = brain.areas["association"].E.start
    italy = set((language.concepts.get("италии", allocate=False) + start).tolist())
    rome = language.concepts.get("рим", allocate=False)

    links = sum(int(pre) in italy for pre in proj.pre[rome].flatten().tolist())
    assert links >= mind.cfg.grow_synapses // 2
    assert brain.synapses_grown > 0


def test_procedural_wiring_saves_memory_and_reads():
    """Врождённые пути с постоянным весом — из хеша: памяти меньше, мозг живёт."""

    stored = make_mind("tiny")
    procedural = make_mind("tiny", procedural_wiring=True)
    brain = procedural.brain

    assert brain.procedural
    assert brain.memory_bytes() < stored.brain.memory_bytes()
    assert brain.synapse_count() == stored.brain.synapse_count()

    report = procedural.language.read_text(FACTS)
    assert report.tokens > 0

    # Вычисляемая строка шага = связи пути.
    neuron = brain.procedural[0].pre_pop.start
    post, weight = brain.proc_rows.rows(torch.tensor([neuron], device=brain.device))
    expected = torch.cat([
        p.outgoing(neuron) for p in brain.procedural
        if p.pre_pop.start <= neuron < p.pre_pop.stop
    ])
    assert sorted(post[0][weight[0] != 0].tolist()) == sorted(expected.tolist())


def test_health_is_clean_and_catches_runaway(mind):
    """Стенд: здоровый мозг без нарушений; очаг (кора стреляет в тишине) — пойман."""

    language = mind.language
    language.read_text("Кошка мяукает.\nСобака лает.")
    assert language.health.sentences >= 2
    assert not language.health.log
    assert "association" in language.health.report()

    # Очаг: в паузе смысловая кора стреляет (как после долгого чтения).
    import torch
    rates = torch.zeros(mind.brain.n, device=mind.brain.device)
    rates[language.meaning.E.slice][:200] = 240.0
    language.health.word(rates * 0)
    language.health.pause(rates)
    snap = language.health.sentence("очаг")
    assert any("в тишине" in p for p in snap.problems)
