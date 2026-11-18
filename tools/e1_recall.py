"""
E1, шаг 5: вывод фразы по порядку (A3 Spaun на словах). Запись — как в
tools/e1_phrase.py; вывод:

  «повтори» (учитель предъявляет команду — заглушка правила BG «повтори»):
    сброс счётчика, позиция 1 в текущую; клетки задачи «повторяю» (20
    клеток, 40 Гц с 150 мс после «повтори», иначе 2 Гц) — состояние задачи, как ps_task у Spaun;
  канал чтения позиции k: стриатум D1 чтения k (совпадение «группа k
    текущей позиции» + «задача повторяю», те же числа, что у записи) →
    SNr чтения k → реле MD чтения k: реле слова j слушает 40 клеток группы j
    ячейки слов k → слой ответа;
  слой ответа: группа на слово (40 клеток LIF, без памяти) — какая группа
    стреляет, то слово и «сказано»; сила реле → ответ g_ans [подбор].
  Сдвиг счётчика при выводе — то же расписание, что при записи (ISSUES 28):
    «сказано» пока не запускает сдвиг сам.

Проверки: во время записи ответ молчит; при выводе в окне позиции k
(0–300 мс) — одно слово, верное; после конца фразы — молчание (позиции
пусты; удерживающего «стопа» ещё нет); вывод дважды подряд без повторной
записи. Испытания: длины фразы 2 и 3 вперемешку; 0 — «ААА», 1 — «АБВ».

    PYTHONPATH=tools python tools/e1_recall.py [trials=8] [seed=1] [g_ans=8] [reads=4] [repeats=2]
"""

from __future__ import annotations

import sys
import time

import torch

from e1_bg import BG, LIF, NSRC, NWORD
from e1_counter import relay_input, sources
from e1_gate import NR, Relay
from e1_slot import C_EXT, C_SEL, DT, P, R_RESET, R_SEL, Slot

T = 1200.0
NANS = 40


def main(opts: dict) -> None:
    dev = opts.get("device", "cpu")
    torch.set_num_threads(int(opts.get("threads", 2)))
    torch.manual_seed(int(opts.get("seed", 1)))
    B = int(opts.get("trials", 8))
    LMAX = 3
    reads = int(opts.get("reads", 4))
    repeats = int(opts.get("repeats", 2))
    g_ans = float(opts.get("g_ans", 8.0))
    phrase = torch.randint(0, P, (B, LMAX))
    phrase[0] = 0
    phrase[1] = torch.arange(LMAX)
    length = torch.where(torch.arange(B) % 2 == 0, LMAX, LMAX - 1)
    length[0] = length[1] = LMAX
    cur, nxt = Slot(B, dev), Slot(B, dev)
    g1, g2 = Relay(B, dev, 16.0, 1.0, 4.0), Relay(B, dev, 16.0, 1.0, 4.0)
    src_cur, src_nxt = sources(cur, NSRC), sources(nxt, NSRC)
    bg_w, bg_r = BG(B, dev), BG(B, dev)
    md_w, md_r = Relay(B * P, dev, 16.0, 1.0, 4.0), Relay(B * P, dev, 16.0, 1.0, 4.0)
    ws = Slot(B * P, dev)
    src_ws = sources(ws, NSRC)                         # [P слов, NSRC]
    ans = LIF(B, NANS, dev, -70.0)
    reset_e = torch.full((B, cur.ne), C_EXT * R_RESET * 1e-3)
    reset_i = torch.full((B, cur.ni), C_EXT * R_RESET * 1e-3)
    start = (cur.lab[None, :] == 0).float().expand(B, cur.ne) * (C_SEL * R_SEL * 1e-3)
    closed, opened = torch.ones(B, dtype=torch.bool), torch.zeros(B, dtype=torch.bool)
    # расписание: периоды (начало, вид, номер); вид — "w" запись слова i, "r" чтение позиции k
    periods = [(400.0 + i * T, "w", i) for i in range(LMAX)]
    rec_starts = []
    t_end = 400.0 + LMAX * T
    for r in range(repeats):
        rs = t_end + 300.0
        rec_starts.append(rs)
        periods += [(rs + 400.0 + k * T, "r", (r, k)) for k in range(reads)]
        t_end = rs + 400.0 + reads * T
    total = t_end + 200.0
    ans_read = torch.zeros(repeats, reads, B, P)
    ans_write = torch.zeros(LMAX, B, P)           # ответ в 0–600 мс после записываемого слова
    dbg = torch.zeros(repeats, reads, 4, B, P)   # стриатум чтения, SNr чтения, реле чтения (по каналам), текущая позиция
    t0 = time.time()
    pi = 0
    for step in range(int(round(total / DT))):
        t = step * DT
        while pi + 1 < len(periods) and t >= periods[pi + 1][0]:
            pi += 1
        p0, kind, idx = periods[pi]
        ph = t - p0 if t >= p0 else -1.0
        in_period = 0 <= ph < T
        # слова и задача
        wr = torch.full((B, P), 2.0)
        if kind == "w" and 0 <= ph < 200:
            m = idx < length
            wr[torch.arange(B)[m], phrase[m, idx]] = 40.0
        wc = torch.poisson(wr * NWORD * 1e-3 * DT)
        task_on = any(rs + 150.0 <= t < rs + 400.0 + reads * T for rs in rec_starts)
        tc = torch.poisson(torch.full((B,), (40.0 if task_on else 2.0) * NWORD * 1e-3 * DT))
        # счётчик: начало фразы, «повтори», сдвиги
        de_c = di_c = de_n = di_n = None
        if 100 <= t < 150 or any(rs + 100 <= t < rs + 150 for rs in rec_starts):
            de_c = start
        if any(rs <= t < rs + 50 for rs in rec_starts):
            de_c, di_c, de_n, di_n = reset_e, reset_i, reset_e, reset_i
        if in_period and 300 <= ph < 350:
            de_n, di_n = reset_e, reset_i
        if in_period and 700 <= ph < 750:
            de_c, di_c = reset_e, reset_i
        g1.step_counts(relay_input(cur.spk_e, src_cur, 1), opened if in_period and 400 <= ph < 600 else closed)
        g2.step_counts(relay_input(nxt.spk_e, src_nxt, 0), opened if in_period and 800 <= ph < 1000 else closed)
        pos = cur.spk_e[:, src_cur].sum(2)
        bg_w.step(pos, wc.sum(1))
        bg_r.step(pos, tc)
        # запись
        ctx_w = wc.repeat_interleave(NR, 1)[:, None, :].expand(B, P, P * NR).reshape(B * P, P * NR)
        md_w.step_counts(ctx_w, None, bg_w.snr.per_channel().reshape(B * P))
        # чтение: реле слова j канала k слушает NSRC клеток группы j ячейки k
        ctx_r = ws.spk_e[:, src_ws].sum(2).repeat_interleave(NR, 1)        # [B·P, P·NR]
        md_r.step_counts(ctx_r, None, bg_r.snr.per_channel().reshape(B * P))
        to_ans = md_r.spk.view(B, P, P, NR).sum((1, 3))                      # [B, слово]
        ans.step(g_ans * to_ans.repeat_interleave(NANS, 1), torch.zeros(B, P * NANS))
        nxt.step(de_n, di_n, add_e=g1.to_slot(nxt.lab))
        cur.step(de_c, di_c, add_e=g2.to_slot(cur.lab))
        ws.step(add_e=md_w.to_slot(ws.lab))
        a = ans.per_channel()
        if kind == "r" and 0 <= ph < 300:
            ans_read[idx[0], idx[1]] += a
            dbg[idx[0], idx[1], 0] += bg_r.msn.per_channel() / 40
            dbg[idx[0], idx[1], 1] += bg_r.snr.per_channel() / 30
            dbg[idx[0], idx[1], 2] += md_r.spk.view(B, P, P * NR).sum(2) / (P * NR)
            dbg[idx[0], idx[1], 3] += (cur.spk_e @ cur.onehot)[:, :P] / cur.nsub
        if kind == "w" and 0 <= ph < 600:
            ans_write[idx] += a
    ans_read = ans_read / NANS / 0.3
    ans_write = ans_write / NANS / 0.6
    print(f"Вывод фразы: {B} испытаний, длины {length.tolist()}, g_ans {g_ans}, {time.time() - t0:.0f} с")
    dbg = dbg / 0.3
    if int(opts.get("debug", 0)):
        names = ["стриатум чт.", "SNr чт.", "реле чт.", "текущая поз."]
        for r in range(repeats):
            for k in range(reads):
                print(f"  [отладка] вывод {r + 1}, окно поз. {k + 1}: " + "; ".join(
                    f"{n} {' '.join(f'{float(x):4.0f}' for x in dbg[r, k, j].mean(0))}" for j, n in enumerate(names))
                    + f"; ответ {' '.join(f'{float(x):4.0f}' for x in ans_read[r, k].mean(0))}")
    leak = (ans_write.max(2).values > 10)
    print(f"  ответ во время записи (0–600 мс после слова): «сказал» (> 10 Гц) в {int(leak.sum())}/{leak.numel()} окнах,"
          f" макс группы в среднем {ans_write.max(2).values.mean():.1f} Гц (должен молчать)")
    th = 10.0
    for r in range(repeats):
        ok = torch.ones(B, dtype=torch.bool)
        line = []
        for k in range(reads):
            a = ans_read[r, k]
            said = (a > th)
            one = said.sum(1) == 1
            right = one & (a.argmax(1) == phrase[:, min(k, LMAX - 1)]) & (k < length)
            silent = ~said.any(1) & (k >= length)
            good = right | silent
            ok &= good
            line.append(f"поз. {k + 1}: верно {int((good & (k < length)).sum())}/{int((k < length).sum())},"
                        f" молчит после конца {int((good & (k >= length)).sum())}/{int((k >= length).sum())}")
        print(f"  вывод {r + 1}: " + "; ".join(line) + f" → фраза целиком {int(ok.sum())}/{B}")
    for b in range(B):
        outs = []
        for r in range(repeats):
            w = ""
            for k in range(reads):
                a = ans_read[r, k, b]
                w += "АБВГД"[int(a.argmax())] if (a > th).sum() == 1 else ("·" if (a > th).sum() == 0 else "?")
            outs.append(w)
        print(f"    {''.join('АБВГД'[x] for x in phrase[b, :length[b]].tolist()):3s} → " + " | ".join(outs))


if __name__ == "__main__":
    main(dict(kv.split("=", 1) for kv in sys.argv[1:]))
