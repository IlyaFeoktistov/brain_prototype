"""
E1, шаг 5б: функциональный тест «сброс → загрузка позиции 1» у счётчика
(tools/e1_counter.py) — отдельно от записи и чтения, числа сети не меняются.

Зачем: при выводе фразы (шаг 5а, SPAUN_TRANSFER) позиция 1, загруженная
через 50 мс после сброса «повтори», держится 17 Гц против 27 при загрузке
из покоя, слабая позиция не проходит сдвиг. Тест разделяет причины:
сам сброс, остаток прежнего состояния (позиция 4 в обеих ячейках, как в
конце записи), зазор сброс → загрузка, повторение циклов.

Сценарии (по trials испытаний каждый; обе ячейки — текущая и следующая,
каналы G1/G2 закрыты всё время, как между сдвигами):
  0 из покоя: загрузка позиции 1 в 1100 мс;
  1 покой → сброс 1000–1050 → загрузка 1100 (зазор 50 мс);
  2 держит позицию 4 (загружена в 100 мс в обе ячейки) → сброс → загрузка 1100;
  3 = 2, но загрузка в 1350 (зазор 300 мс);
  4 = 2, затем ещё два цикла сброс → загрузка: 2000/2100 и 3000/3100.
Сброс — общий залп авторов (800 × 25 Гц на все E и I, 50 мс) в обе ячейки,
как при «повтори» в e1_recall; загрузка — стимул авторов 50 мс в группу 1.

Мера: частоты всех 5 групп текущей, неизбирательных, I и NMDA s группы 1 /
группы 4 по бинам 50 мс; после загрузки — частота позиции 1 в +200…300 и в
окне чтения +300…600 мс; «держит» > 10 Гц; остаток позиции 4.
Критерий: после сброса все позиции < 10 Гц; после загрузки активна только 1,
её частота в окне чтения как при загрузке из покоя.

    PYTHONPATH=tools python tools/e1_counter_reset.py [trials=8] [seed=1]
"""

from __future__ import annotations

import sys
import time

import torch

from e1_counter import relay_input, sources
from e1_gate import Relay
from e1_slot import C_EXT, C_SEL, DT, P, R_RESET, R_SEL, Slot

NAMES = ["0 из покоя", "1 покой → сброс → +50", "2 держит 4 → сброс → +50", "3 держит 4 → сброс → +300",
         "4 держит 4 → 3 цикла"]
TB = 50.0


def main(opts: dict) -> None:
    dev = opts.get("device", "cpu")
    torch.set_num_threads(int(opts.get("threads", 2)))
    torch.manual_seed(int(opts.get("seed", 1)))
    n = int(opts.get("trials", 8))
    S = len(NAMES)
    B = S * n
    sc = torch.arange(S).repeat_interleave(n)
    cur, nxt = Slot(B, dev), Slot(B, dev)
    g1, g2 = Relay(B, dev, 16.0, 1.0, 4.0), Relay(B, dev, 16.0, 1.0, 4.0)
    src_cur, src_nxt = sources(cur, 40), sources(nxt, 40)
    closed = torch.ones(B, dtype=torch.bool)
    rate_e = C_EXT * R_RESET * 1e-3
    group = lambda k: (cur.lab[None, :] == k).float() * (C_SEL * R_SEL * 1e-3)
    # расписание по сценариям: сбросы и загрузки позиции 1 (начала, мс)
    hold4 = (sc >= 2)
    resets = {1: [1000.0], 2: [1000.0], 3: [1000.0], 4: [1000.0, 2000.0, 3000.0]}
    loads = {0: [1100.0], 1: [1100.0], 2: [1100.0], 3: [1350.0], 4: [1100.0, 2100.0, 3100.0]}
    total = 3800.0
    nb = int(total // TB)
    x = torch.zeros(nb, S, 10)        # группы 1–5, неизб., I текущей; NMDA s гр. 1, гр. 4; макс. группа следующей
    win = {s: [torch.zeros(n, P) for _ in loads[s]] for s in range(S)}       # окно чтения +300…600
    early = {s: [torch.zeros(n, P) for _ in loads[s]] for s in range(S)}     # +200…300
    t0 = time.time()
    for step in range(int(round(total / DT))):
        t = step * DT
        de = torch.zeros(B, cur.ne)
        di = torch.zeros(B, cur.ni)
        any_d = False
        if 100 <= t < 150:
            de += hold4[:, None].float() * group(3)
            any_d = True
        for s, rr in resets.items():
            if any(r <= t < r + 50 for r in rr):
                m = (sc == s)[:, None].float()
                de += m * rate_e
                di += m * rate_e
                any_d = True
        for s, ll in loads.items():
            if any(l <= t < l + 50 for l in ll):
                de += (sc == s)[:, None].float() * group(0)
                any_d = True
        # следующая: та же позиция 4 и те же сбросы, без загрузки
        de_n = de - sum((sc == s)[:, None].float() * group(0) * any(l <= t < l + 50 for l in ll) for s, ll in loads.items())
        g1.step_counts(relay_input(cur.spk_e, src_cur, 1), closed)
        g2.step_counts(relay_input(nxt.spk_e, src_nxt, 0), closed)
        nxt.step(de_n if any_d else None, di if any_d else None, add_e=g1.to_slot(nxt.lab))
        cur.step(de if any_d else None, di if any_d else None, add_e=g2.to_slot(cur.lab))
        g = (cur.spk_e @ cur.onehot) / torch.tensor([cur.nsub] * P + [cur.nnon]) * 1e4    # Гц, [B, G]
        gi = cur.spk_i.mean(1, keepdim=True) * 1e4
        s_pool = (cur.s_nmda @ cur.onehot)[:, [0, 3]] / cur.nsub
        gn = ((nxt.spk_e @ nxt.onehot)[:, :P] / nxt.nsub * 1e4).max(1, keepdim=True).values
        row = torch.cat([g, gi, s_pool, gn], 1).view(S, n, 10).mean(1)
        b = int(t // TB)
        if b < nb:
            x[b] += row
        for s in range(S):
            for j, l in enumerate(loads[s]):
                sl = slice(s * n, (s + 1) * n)
                if l + 300 <= t < l + 600:
                    win[s][j] += g[sl, :P] / 3000.0
                if l + 200 <= t < l + 300:
                    early[s][j] += g[sl, :P] / 1000.0
    x /= TB / DT
    print(f"Сброс → загрузка счётчика: {S} сценариев × {n} испытаний, {time.time() - t0:.0f} с")
    cols = ["гр1", "гр2", "гр3", "гр4", "гр5", "неизб", "I", "sNMDA1", "sNMDA4", "след.макс"]
    for s in range(S):
        print(f"\n[{NAMES[s]}] бины {TB:.0f} мс, среднее по испытаниям, Гц (s — доля)")
        print("   t,мс " + " ".join(f"{c:>9s}" for c in cols))
        for b in range(int(800 // TB), nb):
            if s != 4 and b * TB >= 2000:
                break
            print(f"  {b * TB:5.0f} " + " ".join(f"{float(v):9.3f}" if j >= 7 and j < 9 else f"{float(v):9.1f}"
                                            for j, v in enumerate(x[b, s])))
    print("\nИтог (позиция 1 после загрузки; «держит» > 10 Гц):")
    for s in range(S):
        for j, l in enumerate(loads[s]):
            w, e = win[s][j], early[s][j]
            only1 = (w[:, 0] > 10) & ((w[:, 1:] > 10).sum(1) == 0)
            print(f"  {NAMES[s]:28s} загрузка {l:4.0f}: +200…300 {e[:, 0].mean():5.1f} Гц;"
                  f" окно +300…600 {w[:, 0].mean():5.1f} Гц (по испытаниям {' '.join(f'{float(v):.0f}' for v in w[:, 0])});"
                  f" держит одна 1: {int(only1.sum())}/{n}; остаток поз. 4 {w[:, 3].mean():.1f} Гц")


if __name__ == "__main__":
    main(dict(kv.split("=", 1) for kv in sys.argv[1:]))
