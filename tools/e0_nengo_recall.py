"""
E0 (docs/SPAUN_TRANSFER.md): эталон в Nengo — сеть держит фразу из слов и
повторяет её по порядку, сама останавливаясь. Устройство — как в Spaun 2.0
(github.com/xchoo/spaun2.0, _spaun/), на готовых сетях Nengo:

- слово — семантический указатель (вектор D), позиции POS1, POS2 = POS1 ⊛ INC,
  … (vocabulator/info_encoding: Pos_Inc_Network);
- запоминание: слово ⊛ позиция (CircularConvolution) копится в интеграторе
  «список» (обратная связь 1 через 100 мс, как InputGatedMemory Spaun);
  позиция при записи подаётся вместе со словом — упрощение (в Spaun её
  ведёт тот же счётчик, что при выводе);
- вывод (serial_recall_net): список ⊛ ~позиция → очистка (ассоциативная
  память с порогом и победителем, WTAAssocMem) → узел «сказано» (заглушка выхода вместо руки: слово
  сказано, когда очистка держит его ~0.25 с; затем сигнал «готово» 0.12 с);
- счётчик позиции (Pos_Inc_Network, MemoryBlock): две памяти с воротами;
  «готово» открывает первую — она берёт очищенную текущую ⊛ INC (очистка
  позиции, как cleanup_mode у Spaun), потом вторая копирует её;
- стоп (Output_Classification «знаю / не знаю / стоп»): базальные ганглии
  Nengo (Gurney, Prescott & Redgrave 2001; Stewart et al. 2010) выбирают
  между «продолжать» (полезность — насколько очистка узнала слово) и «стоп»
  (постоянная 0.5); выход таламуса «стоп» копится и защёлкивается.

Учитель только предъявляет фразу и даёт сигнал «повтори» (вход задачи);
следующее слово и момент остановки выбирает сеть.

    .venv-nengo/bin/python tools/e0_nengo_recall.py [D=128] [trials=4] [seed=1] [len=3,4,5,5]
"""

from __future__ import annotations

import sys
import time

import numpy as np
import nengo
import nengo_spa as spa

WORDS = ["МАМА", "МЫЛА", "РАМУ", "ПАПА", "ЧИТАЛ", "КНИГУ", "КОТ", "СПАЛ", "ДОМА", "УТРОМ"]
ITEM_ON, ITEM_GAP = 0.3, 0.1        # с: слово на входе, пауза
SAY_HOLD, DONE_LEN = 0.25, 0.12     # с: держать слово до «сказано», длина «готово»
DEBUG = False


def make_vocab(D: int, n_pos: int, rng: np.random.RandomState) -> spa.Vocabulary:
    v = spa.Vocabulary(D, pointer_gen=rng, strict=False)
    for i in range(len(WORDS)):
        v.populate(f"W{i}")
    v.add("INC", v.create_pointer().unitary())
    v.add("POS1", v.create_pointer().unitary())
    for k in range(2, n_pos + 1):
        v.add(f"POS{k}", (v[f"POS{k - 1}"] * v["INC"]).unitary())
    return v


def trial(D: int, phrase: list[int], seed: int) -> dict:
    rng = np.random.RandomState(seed)
    v = make_vocab(D, len(phrase) + 2, rng)
    words = np.array([v[f"W{i}"].v for i in range(len(WORDS))])
    pos = [v[f"POS{k + 1}"].v for k in range(len(phrase) + 2)]
    inc = v["INC"].v

    t_enc = len(phrase) * (ITEM_ON + ITEM_GAP)
    t_rec = t_enc + 0.3                      # сигнал «повтори»
    t_end = t_rec + (len(phrase) + 2) * 0.6
    scale_item = 0.45                        # вклад слова в «список»: 5 слов ≈ норма 1

    def stim_fn(t):
        k = int(t // (ITEM_ON + ITEM_GAP))
        if k < len(phrase) and (t - k * (ITEM_ON + ITEM_GAP)) < ITEM_ON:
            return np.concatenate([words[phrase[k]], pos[k]])
        return np.zeros(2 * D)

    said: list[tuple[float, int]] = []
    st = {"cur": -1, "since": 0.0, "done_until": -1.0, "emitted": False}

    def say_fn(t, x):
        """Заглушка выхода: узнанное очисткой слово держится SAY_HOLD → сказано, «готово» на DONE_LEN."""

        if t < t_rec:
            return [0.0]
        sims = words @ x
        k = int(np.argmax(sims))
        if sims[k] < 0.5:
            st["cur"], st["emitted"] = -1, False
        elif k != st["cur"]:
            st["cur"], st["since"], st["emitted"] = k, t, False
        elif not st["emitted"] and t - st["since"] >= SAY_HOLD:
            said.append((round(t, 3), k))
            st["emitted"] = True
            st["done_until"] = t + DONE_LEN
        return [1.0 if t < st["done_until"] else 0.0]

    model = nengo.Network(seed=seed)
    with model:
        stim = nengo.Node(stim_fn)
        recall = nengo.Node(lambda t: 1.0 if t >= t_rec else 0.0)
        reset = nengo.Node(lambda t: pos[0] if t < t_rec - 0.05 else np.zeros(D))

        # Запись: слово ⊛ позиция → интегратор «список».
        enc = nengo.networks.CircularConvolution(200, D)
        nengo.Connection(stim[:D], enc.input_a)
        nengo.Connection(stim[D:], enc.input_b)
        lst = nengo.networks.EnsembleArray(50, D, radius=0.5)
        nengo.Connection(lst.output, lst.input, synapse=0.1)
        nengo.Connection(enc.output, lst.input, synapse=0.1, transform=0.1 * scale_item / ITEM_ON)

        # Счётчик позиции: mem1 открыта при «готово», mem2 — когда «готово» нет.
        mem1 = nengo.networks.InputGatedMemory(50, D, difference_gain=15.0)
        mem2 = nengo.networks.InputGatedMemory(50, D, difference_gain=15.0)
        done = nengo.Ensemble(100, 1)
        bias = nengo.Node(1.0)
        inc_c = nengo.networks.CircularConvolution(100, D)
        nengo.Connection(mem2.output, inc_c.input_a)
        nengo.Connection(nengo.Node(inc), inc_c.input_b)
        # Очистка позиции (Spaun: InputGatedCleanupMemory, cleanup_mode=1 счётчика): без неё шум ⊛ INC копится.
        pos_clean = spa.WTAAssocMem(0.3, v, mapping=[f"POS{k + 1}" for k in range(len(pos))],
                                    function=lambda x: x > 0)
        nengo.Connection(inc_c.output, pos_clean.input)
        nengo.Connection(pos_clean.output, mem1.input)
        nengo.Connection(reset, mem1.input)
        nengo.Connection(mem1.output, mem2.input)
        nengo.Connection(bias, mem1.gate)
        nengo.Connection(done, mem1.gate, transform=-1.0)
        # До «повтори»: mem1 открыта и берёт POS1 (сброс), путь ⊛ INC заглушён.
        resetting = nengo.Node(lambda t: 1.0 if t < t_rec - 0.05 else 0.0)
        nengo.Connection(resetting, mem1.gate, transform=-1.0)
        for e in inc_c.all_ensembles:
            nengo.Connection(resetting, e.neurons, transform=-5.0 * np.ones((e.n_neurons, 1)))
        nengo.Connection(done, mem2.gate)

        # Вывод: список ⊛ ~позиция → очистка.
        dec = nengo.networks.CircularConvolution(100, D, invert_b=True)
        nengo.Connection(lst.output, dec.input_a)
        # Читаемая позиция — тоже через очистку: остаток прежней позиции в mem2 иначе читает прежнее слово.
        read_pos = spa.WTAAssocMem(0.3, v, mapping=[f"POS{k + 1}" for k in range(len(pos))],
                                   function=lambda x: x > 0)
        nengo.Connection(mem2.output, read_pos.input)
        nengo.Connection(read_pos.output, dec.input_b)
        # Очистка с победителем (у Spaun — порог + разность полезностей): одно слово или ничего.
        clean = spa.WTAAssocMem(0.5, v, mapping=[f"W{i}" for i in range(len(WORDS))],
                                function=lambda x: x > 0)
        nengo.Connection(dec.output, clean.input, transform=2.0)   # слово ~0.4 → ~0.8 при пороге 0.5

        say = nengo.Node(say_fn, size_in=D, size_out=1)
        nengo.Connection(clean.output, say, synapse=0.01)
        nengo.Connection(say, done, synapse=0.005)

        # Стоп: BG выбирает «продолжать» (узнано) против «стоп» (0.5), только после «повтори».
        known = nengo.Ensemble(100, 1)
        nengo.Connection(clean.output, known, transform=[words.sum(0)], synapse=0.02)
        bg = nengo.networks.BasalGanglia(2)
        th = nengo.networks.Thalamus(2)
        nengo.Connection(known, bg.input[0])
        nengo.Connection(recall, bg.input[1], transform=0.5)
        nengo.Connection(bg.output, th.input)
        stop = nengo.Ensemble(100, 1)
        nengo.Connection(th.output[1], stop, transform=0.15, synapse=0.1)   # «не узнал» должно держаться ~0.3 с
        nengo.Connection(stop, stop, synapse=0.1)                     # защёлка
        # До «повтори» стоп заглушён (задача «вывод» ещё не начата).
        nengo.Connection(recall, stop.neurons, transform=-5.0 * np.ones((stop.n_neurons, 1)),
                         function=lambda r: 1.0 - r)
        for e in clean.all_ensembles:
            nengo.Connection(stop, e.neurons, transform=-5.0 * np.ones((e.n_neurons, 1)),
                             function=lambda s: max(0.0, s - 0.3) * 3)

        p_m1 = nengo.Probe(mem1.output, synapse=0.02)
        p_m2 = nengo.Probe(mem2.output, synapse=0.02)
        p_rp = nengo.Probe(read_pos.output, synapse=0.02)
        p_done = nengo.Probe(done, synapse=0.01)
        p_dec = nengo.Probe(dec.output, synapse=0.02)
        p_lst = nengo.Probe(lst.output, synapse=0.02)
        p_stop = nengo.Probe(stop, synapse=0.02)
        p_known = nengo.Probe(known, synapse=0.02)

    n_neurons = sum(e.n_neurons for e in model.all_ensembles)
    t0 = time.time()
    with nengo.Simulator(model, progress_bar=False, seed=seed) as sim:
        sim.run(t_end)
    wall = time.time() - t0
    if DEBUG:
        P = np.array(pos)
        tr = sim.trange()
        for name, tt in (("после записи", t_enc + 0.1), ("при «повтори»", t_rec), ("в конце", t_end - 0.01)):
            L = sim.data[p_lst][np.searchsorted(tr, tt)]
            print("   список %s: |L| %.2f, сходство с word⊛pos: %s" % (name, np.linalg.norm(L), [
                round(float(L @ (v[f"W{w}"] * v[f"POS{k + 1}"]).v), 2) for k, w in enumerate(phrase)]))
        for t in np.arange(t_rec - 0.2, t_end, 0.1):
            i = np.searchsorted(tr, t)
            m1, m2 = P @ sim.data[p_m1][i], P @ sim.data[p_m2][i]
            dw = words @ sim.data[p_dec][i]
            rp = spa.SemanticPointer(sim.data[p_rp][i])
            ideal = words @ (spa.SemanticPointer(sim.data[p_lst][i]) * ~rp).v
            print(f"   t {t:4.1f}: done {sim.data[p_done][i, 0]:+.2f} mem1 POS{int(m1.argmax()) + 1} {m1.max():.2f} "
                  f"mem2 POS{int(m2.argmax()) + 1} {m2.max():.2f} |mem2| {np.linalg.norm(sim.data[p_m2][i]):.2f} "
                  f"чтение POS{int((P @ sim.data[p_rp][i]).argmax()) + 1} {(P @ sim.data[p_rp][i]).max():.2f}; "
                  f"развязка {WORDS[int(dw.argmax())]} {dw.max():.2f} (точно {WORDS[int(ideal.argmax())]} {ideal.max():.2f}); узнано {sim.data[p_known][i, 0]:.2f} стоп {sim.data[p_stop][i, 0]:.2f}")
    stop_t = next((float(sim.trange()[i]) for i in range(len(sim.trange())) if sim.data[p_stop][i, 0] > 0.5), None)
    return {"phrase": phrase, "said": said, "stop_t": stop_t, "t_rec": t_rec, "t_end": t_end,
            "wall": wall, "neurons": n_neurons}


def main(opts: dict) -> None:
    global DEBUG
    DEBUG = opts.get("debug", "0") != "0"
    D = int(opts.get("D", 128))
    seed = int(opts.get("seed", 1))
    lens = [int(x) for x in opts.get("len", "3,4,5,5").split(",")][: int(opts.get("trials", 4))]
    rng = np.random.RandomState(seed)
    tot_pos = ok_pos = 0
    for j, L in enumerate(lens):
        phrase = list(rng.choice(len(WORDS), L, replace=False))
        r = trial(D, phrase, seed * 100 + j)
        got = [k for _, k in r["said"]]
        hits = sum(1 for i, k in enumerate(got[:L]) if k == phrase[i])
        tot_pos += L
        ok_pos += hits
        last = r["said"][-1][0] if r["said"] else None
        stop_s = "нет" if r["stop_t"] is None else f"{r['stop_t']:.2f} с"
        print(f"проба {j + 1}: фраза {' '.join(WORDS[i] for i in phrase)}\n"
              f"   сказано {' '.join(WORDS[k] for k in got) or '—'} (времена {[t for t, _ in r['said']]}); "
              f"верно по позициям {hits}/{L}, лишних {max(0, len(got) - L)}; "
              f"стоп {stop_s} (повтори в {r['t_rec']:.2f}, "
              f"последнее слово {last}); нейронов {r['neurons']}, {r['t_end']:.1f} с модели за {r['wall']:.0f} с", flush=True)
    print(f"итого верно по позициям {ok_pos}/{tot_pos}")


if __name__ == "__main__":
    main(dict(kv.split("=", 1) for kv in sys.argv[1:]))
