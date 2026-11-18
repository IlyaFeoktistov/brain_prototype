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

    PYTHONPATH=tools python tools/e1_recall.py [trials=8] [seed=1] [g_ans=8] [reads=4] [repeats=2] [trace=0]

trace=1 — диагностика без изменений сети (тот же поток случайных чисел):
временные ряды по цепочке, бины 50 мс, среднее по испытаниям, позиции 1 и 2,
запись против чтения — см. trace_report.
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


class Trace:
    """Бины по времени: средние по испытаниям величины цепочки по каналам (позициям) 0..P−1.
    «Своё» слово канала k — слово фразы на позиции k (испытания короче k не считаются)."""

    NAMES = ["поз Гц", "поз v", "стрЧ Гц", "стрЧ v", "стрЧ Iвх", "SNrЧ Гц", "стрЗ Гц", "SNrЗ Гц",
             "MDЧ своё", "MDЧ чуж", "MDЧ v", "MDЧ Iкора", "MDЧ Isnr", "MDЗ Гц", "яч своё", "отв своё", "отв чуж",
             "отв Iвх"]

    def __init__(self, nb: int, B: int, phrase: torch.Tensor, length: torch.Tensor):
        self.x = torch.zeros(nb, len(self.NAMES), P)
        self.n = torch.zeros(nb)
        self.B = B
        L = phrase.shape[1]
        self.valid = (torch.arange(P)[None, :] < length[:, None]).float()        # [B, P]
        w = torch.zeros(B, P, dtype=torch.long)
        w[:, :L] = phrase
        self.w = w

    def add(self, b: int, cur, ws, bg_w, bg_r, md_r, md_w, ans) -> None:
        B, w, val = self.B, self.w, self.valid
        nv = val.sum(0).clamp(min=1)
        own = lambda x: (x.gather(2, w[:, :, None]).squeeze(2) * val).sum(0) / nv     # x [B, канал, слово]
        other = lambda x: ((x.sum(2) - x.gather(2, w[:, :, None]).squeeze(2)) / (P - 1) * val).sum(0) / nv
        grp = lambda v, nn: v.view(B, P, nn).mean(2).mean(0)
        k = 1000.0 / DT    # доля спайков за шаг → Гц
        r = [(cur.spk_e @ cur.onehot)[:, :P].mean(0) / cur.nsub * k,
             ((cur.ve @ cur.onehot)[:, :P] / cur.nsub).mean(0),
             bg_r.msn.per_channel().mean(0) / 40 * k,
             grp(bg_r.msn.v, 40),
             grp(-bg_r.msn.ge * bg_r.msn.v, 40),
             bg_r.snr.per_channel().mean(0) / 30 * k,
             bg_w.msn.per_channel().mean(0) / 40 * k,
             bg_w.snr.per_channel().mean(0) / 30 * k]
        sp = md_r.spk.view(B, P, P, NR).mean(3) * k
        r += [own(sp), other(sp),
              own(md_r.v.view(B, P, P, NR).mean(3)),
              own((-md_r.gctx * md_r.ctx * md_r.v).view(B, P, P, NR).mean(3)),
              own((md_r.gsnr * md_r.gaba * (md_r.v + 70.0)).view(B, P, P, NR).mean(3)),
              md_w.spk.view(B, P, P * NR).mean(2).mean(0) * k,
              own((ws.spk_e @ ws.onehot)[:, :P].view(B, P, P) / ws.nsub * k)]
        a3 = (ans.spk.view(B, P, NANS).mean(2) * k)[:, None, :].expand(B, P, P)
        g3 = (-ans.ge * ans.v).view(B, P, NANS).mean(2)[:, None, :].expand(B, P, P)
        r += [own(a3), other(a3), own(g3)]
        self.x[b] += torch.stack(r)
        self.n[b] += 1


def trace_report(tr: Trace, TB: float, periods: list, rec_starts: list) -> None:
    x = tr.x / tr.n.clamp(min=1)[:, None, None]
    rs = rec_starts[0]
    # (заголовок, начало окна, нулевой момент, канал)
    wins = [("НАЧАЛО фразы: позиция 1 в текущую +100…150", 0.0, 0.0, 0),
            ("ЗАПИСЬ слова 1 (0 — начало слова)", periods[0][0] - 100, periods[0][0], 0),
            ("ЗАПИСЬ слова 2", periods[1][0] - 100, periods[1][0], 1),
            ("ПЕРЕД «повтори» → сброс (0) → позиция 1 (+100…150)", rs - 300, rs, 0),
            ("ЧТЕНИЕ позиции 1 (0 — начало окна чтения)", rs + 400 - 100, rs + 400, 0),
            ("ЧТЕНИЕ позиции 2", rs + 400 + 1200 - 100, rs + 1600, 1)]
    cols = ["поз Гц", "поз v", "стрЗ Гц", "SNrЗ Гц", "MDЗ Гц", "стрЧ Гц", "стрЧ v", "стрЧ Iвх", "SNrЧ Гц",
            "MDЧ своё", "MDЧ чуж", "MDЧ v", "MDЧ Iкора", "MDЧ Isnr", "яч своё", "отв своё", "отв чуж", "отв Iвх"]
    idx = [Trace.NAMES.index(c) for c in cols]
    for title, a, zero, k in wins:
        print(f"\n[трасса] {title}; канал = позиция {k + 1}; бины {TB:.0f} мс, среднее по испытаниям; Гц, мВ, пА")
        print("   t,мс " + " ".join(f"{c:>9s}" for c in cols))
        for b in range(int(a // TB), int((a + 800) // TB)):
            print(f"  {b * TB - zero:+5.0f} " + " ".join(f"{float(x[b, j, k]):9.1f}" for j in idx))


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
    trace = int(opts.get("trace", 0))
    TB = 50.0
    tr = Trace(int(total // TB) + 1, B, phrase, length) if trace else None
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
        if tr is not None:
            tr.add(int(t // TB), cur, ws, bg_w, bg_r, md_r, md_w, ans)
        if kind == "r" and 0 <= ph < 300:
            ans_read[idx[0], idx[1]] += a
            dbg[idx[0], idx[1], 0] += bg_r.msn.per_channel() / 40
            dbg[idx[0], idx[1], 1] += bg_r.snr.per_channel() / 30
            dbg[idx[0], idx[1], 2] += md_r.spk.view(B, P, P * NR).sum(2) / (P * NR)
            dbg[idx[0], idx[1], 3] += (cur.spk_e @ cur.onehot)[:, :P] / cur.nsub
        if kind == "w" and 0 <= ph < 600:
            ans_write[idx] += a
    if tr is not None:
        trace_report(tr, TB, periods, rec_starts)
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
                if k < LMAX:
                    print("      по испытаниям (поз. Гц / стриатум / SNr / реле своего канала): " + "  ".join(
                        f"{''.join('АБВГД'[x] for x in phrase[b, :length[b]].tolist())}"
                        f" {float(dbg[r, k, 3, b, k]):.0f}/{float(dbg[r, k, 0, b, k]):.0f}/{float(dbg[r, k, 1, b, k]):.0f}/{float(dbg[r, k, 2, b, k]):.0f}"
                        for b in range(B)))
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
