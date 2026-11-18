"""
E1, шаг 4: запись фразы «позиция × слово» — канал записи выбирают
базальные ганглии из активности сети. Не мозг — отдельная сеть на torch.

Блоки:
  счётчик позиции — tools/e1_counter.py (текущая/следующая, каналы G1/G2;
    сдвиг по расписанию — ISSUES 28);
  слова — 20 клеток на слово, 40 Гц при предъявлении, иначе 2 Гц (учитель
    только предъявляет);
  стриатум D1 → SNr, канал на позицию — tools/e1_bg.py: стриатум k
    стреляет только при совпадении «группа k текущей позиции» + «есть
    слово», глушит SNr k;
  реле MD позиции k — tools/e1_gate.py (SNr k — спайки клеток SNr канала);
  ячейка слов позиции k — Brunel & Wang 2001 (tools/e1_slot.py), группа =
    слово.
Номер позиции никто не передаёт: канал открывается совпадением в стриатуме.

Время (period T = 1200 мс): позиция 1 в текущую 100–150 мс (начало фразы);
слово i — с 400 + i·T, 200 мс; после слова сдвиг счётчика: сброс
следующей +300…350, G1 +400…600, сброс текущей +700…750, G2 +800…1000.

Проверки (порог 10 Гц): позиция i < L держит своё слово одна (сразу после
записи и в конце — стирают ли последующие сдвиги); позиции ≥ L пусты.
Испытание 0 — «А А А…» (одно слово на разных позициях), 1 — «А Б В…»,
остальные — случайные фразы (разные слова на одной позиции).

    PYTHONPATH=tools python tools/e1_phrase.py [trials=8] [seed=1] [L=3] [reset=burst|inhib]
"""

from __future__ import annotations

import sys
import time

import torch

from e1_bg import BG, NSRC, NWORD
from e1_counter import relay_input, sources
from e1_gate import NR, Relay
from e1_slot import C_EXT, C_SEL, DT, P, R_RESET, R_SEL, Slot

T = 1200.0


def main(opts: dict) -> None:
    dev = opts.get("device", "cpu")
    torch.set_num_threads(int(opts.get("threads", 2)))
    torch.manual_seed(int(opts.get("seed", 1)))
    B = int(opts.get("trials", 8))
    L = int(opts.get("L", 3))
    phrase = torch.randint(0, P, (B, L))
    phrase[0] = 0
    phrase[1] = torch.arange(L) % P
    # счётчик
    cur, nxt = Slot(B, dev), Slot(B, dev)
    g1, g2 = Relay(B, dev, 16.0, 1.0, 4.0), Relay(B, dev, 16.0, 1.0, 4.0)
    src_cur, src_nxt = sources(cur, NSRC), sources(nxt, NSRC)
    # выбор BG и запись: позиции — в пачке (B·P)
    bg = BG(B, dev)
    md = Relay(B * P, dev, 16.0, 1.0, 4.0)
    ws = Slot(B * P, dev)
    # Форма сброса ячеек счётчика (шаг 5в, tools/e1_counter_reset.py): burst — залп авторов
    # Brunel–Wang на все E и I; inhib — тот же залп только на тормозные (как сброс nengo
    # InputGatedMemory — торможение клеток памяти): без вспышки всех позиций и хвоста NMDA.
    reset_mode = opts.get("reset", "burst")
    reset_e = None if reset_mode == "inhib" else torch.full((B, cur.ne), C_EXT * R_RESET * 1e-3)
    reset_i = torch.full((B, cur.ni), C_EXT * R_RESET * 1e-3)
    start = (cur.lab[None, :] == 0).float().expand(B, cur.ne) * (C_SEL * R_SEL * 1e-3)
    closed, opened = torch.ones(B, dtype=torch.bool), torch.zeros(B, dtype=torch.bool)
    total = 400.0 + L * T + 800.0
    # окна: после записи слова i — [w_i+1000, w_i+1200); конец — последние 500 мс
    wins = [(400.0 + i * T + 1000.0, 400.0 + i * T + 1200.0) for i in range(L)] + [(total - 500.0, total)]
    cnt = torch.zeros(len(wins), B, P, P)          # окно, испытание, позиция, слово
    cur_cnt = torch.zeros(len(wins), B, P)
    msn_cnt = torch.zeros(L, B, P)                 # слово i × канал
    t0 = time.time()
    for step in range(int(round(total / DT))):
        t = step * DT
        rel = t - 400.0
        i_word = int(rel // T) if rel >= 0 else -1
        ph = rel - i_word * T if rel >= 0 else -1.0
        # слова
        wr = torch.full((B, P), 2.0)
        if 0 <= i_word < L and 0 <= ph < 200:
            wr[torch.arange(B), phrase[:, i_word]] = 40.0
        wc = torch.poisson(wr * NWORD * 1e-3 * DT)           # [B, P] спайки клеток слов
        # счётчик
        de_c = di_c = de_n = di_n = None
        if 100 <= t < 150:
            de_c = start
        adv = 0 <= i_word < L
        if adv and 300 <= ph < 350:
            de_n, di_n = reset_e, reset_i
        if adv and 700 <= ph < 750:
            de_c, di_c = reset_e, reset_i
        g1.step_counts(relay_input(cur.spk_e, src_cur, 1), opened if adv and 400 <= ph < 600 else closed)
        g2.step_counts(relay_input(nxt.spk_e, src_nxt, 0), opened if adv and 800 <= ph < 1000 else closed)
        # BG: позиция — спайки NSRC клеток каждой группы текущей; слово — все клетки слов
        bg.step(cur.spk_e[:, src_cur].sum(2), wc.sum(1))
        # реле позиций: реле слова j слушает 20 клеток слова j; SNr — своего канала
        ctx = wc.repeat_interleave(NR, 1)[:, None, :].expand(B, P, P * NR).reshape(B * P, P * NR)
        md.step_counts(ctx, None, bg.snr.per_channel().reshape(B * P))
        nxt.step(de_n, di_n, add_e=g1.to_slot(nxt.lab))
        cur.step(de_c, di_c, add_e=g2.to_slot(cur.lab))
        ws.step(add_e=md.to_slot(ws.lab))
        if adv and ph < 200:
            msn_cnt[i_word] += bg.msn.per_channel()
        for k, (a, b) in enumerate(wins):
            if a <= t < b:
                cnt[k] += (ws.spk_e @ ws.onehot)[:, :P].view(B, P, P)
                cur_cnt[k] += (cur.spk_e @ cur.onehot)[:, :P]
    rates = cnt / ws.nsub / torch.tensor([b - a for a, b in wins])[:, None, None, None] * 1000.0
    print(f"Запись фразы: {B} испытаний, L = {L}, {time.time() - t0:.0f} с")
    m = (msn_cnt / 40 / 0.2).mean(1)               # [L, P], Гц
    for i in range(L):
        print(f"  стриатум во время слова {i + 1}, по каналам 1–{P}: {' '.join(f'{float(x):5.1f}' for x in m[i])} Гц")
    held = lambda r, pos, word: (r[:, pos, :].argmax(1) == word) & ((r[:, pos, :] > 10).sum(1) == 1)
    ok_all = torch.ones(B, dtype=torch.bool)
    fin = rates[-1]
    for i in range(L):
        just = held(rates[i], i, phrase[:, i])
        end = held(fin, i, phrase[:, i])
        ok_all &= end
        print(f"  позиция {i + 1}: держит своё слово сразу после записи {int(just.sum())}/{B}, в конце {int(end.sum())}/{B};"
              f" своё {fin[torch.arange(B), i, phrase[:, i]].mean():.1f} Гц")
    for i in range(L, P):
        empty = (fin[:, i, :] < 10).all(1)
        ok_all &= empty
        print(f"  позиция {i + 1} (за концом фразы): пуста {int(empty.sum())}/{B}, макс {fin[:, i, :].max(1).values.mean():.1f} Гц")
    print(f"  фраза записана целиком без лишнего: {int(ok_all.sum())}/{B}")
    print("  испытания: " + "; ".join(f"{''.join('АБВГД'[w] for w in phrase[b].tolist())}"
                                      f"{'✓' if ok_all[b] else '✗'}" for b in range(B)))


if __name__ == "__main__":
    main(dict(kv.split("=", 1) for kv in sys.argv[1:]))
