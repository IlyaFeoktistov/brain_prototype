"""
E1, шаг 5б/5в: функциональный тест «сброс → позиция 1» у счётчика
(tools/e1_counter.py) — отдельно от записи и чтения, числа сети не меняются.

Зачем: при выводе фразы (шаг 5а, SPAUN_TRANSFER) позиция 1, загруженная
через 50 мс после сброса «повтори», держится 17 Гц против 27 при загрузке
из покоя. Шаг 5б: причина — сам сброс (общий возбуждающий залп авторов
Brunel & Wang 2001 на все E и I). Шаг 5в — формы сброса из исходников:

  burst — залп авторов: 800 × 25 Гц на все E и I ячейки, 50 мс (как было);
  inhib — тот же залп только на тормозные клетки ячейки: E молчат под
          торможением — аналог сброса nengo InputGatedMemory (вход −3 на
          все клетки памяти, nengo/networks/workingmemory.py);
  write — сброс Spaun (spaun2.0 _spaun/_networks/workingmemory.py,
          make_resettable; счётчик — Pos_Inc_Network, reset_key = POS1):
          сброс открывает ворота записи и пишет значение сброса (позицию 1)
          поверх старого, без стирания — «надёжнее, чем просто стереть»
          (комментарий авторов). У нас: стимул позиции 1 в обе ячейки
          (у Spaun reset_mode 3 — обе памяти блока), старую группу вытесняет
          общее торможение ячейки (Brunel–Wang: новое слово заменяет старое).

Сценарии (по trials испытаний каждый; G1/G2 закрыты всё время):
  0 из покоя: позиция 1 в 1100 мс;
  1 burst: держит позицию 4 → сброс 1000–1050 → позиция 1 в 1100;
  2 inhib: держит 4 → сброс 1000–1050 → позиция 1 в 1100;
  3 inhib: держит 4 → сброс 1000–1050 → позиция 1 в 1050 (сразу);
  4 write: держит 4 → позиция 1 в 1000 (это и есть сброс);
  5 write × 3: как 4, затем позиция 4 снова (1600) и сброс-запись (2100),
    ещё раз (2600 / 3100) — не копится ли ущерб.
Позиция 4 загружается в обе ячейки (как в конце записи фразы).

Мера: частоты групп текущей по бинам 50 мс; «вспышка» — макс. частота
групп 2–5 в окне сброс…+100 мс (бин 50 мс): выше 10 Гц — каналы чтения
откроются; позиция 1 в окне чтения +300…600 мс после загрузки; остаток
позиции 4 и «ожил ли» он: позиция 4 > 10 Гц в любом бине +300…+900 мс.
Критерии (шаг 5в): все позиции сброшены; вспышки нет; позиция 1 ≈ 26–27 Гц
как из покоя; старое не оживает; запись фразы — отдельно (e1_phrase).

    PYTHONPATH=tools python tools/e1_counter_reset.py [trials=8] [seed=1]
"""

from __future__ import annotations

import sys
import time

import torch

from e1_counter import relay_input, sources
from e1_gate import Relay
from e1_slot import C_EXT, C_SEL, DT, P, R_RESET, R_SEL, Slot

# (название, держит позицию 4 с 100 мс, сбросы [(начало, вид)], загрузки позиции 1, загрузки позиции 4 потом)
SCEN = [
    ("0 из покоя", False, [], [1100.0], []),
    ("1 burst → +50", True, [(1000.0, "burst")], [1100.0], []),
    ("2 inhib → +50", True, [(1000.0, "inhib")], [1100.0], []),
    ("3 inhib → сразу", True, [(1000.0, "inhib")], [1050.0], []),
    ("4 write (Spaun)", True, [], [1000.0], []),
    ("5 write × 3", True, [], [1000.0, 2100.0, 3100.0], [1600.0, 2600.0]),
]
TB = 50.0


def main(opts: dict) -> None:
    dev = opts.get("device", "cpu")
    torch.set_num_threads(int(opts.get("threads", 2)))
    torch.manual_seed(int(opts.get("seed", 1)))
    n = int(opts.get("trials", 8))
    S = len(SCEN)
    B = S * n
    sc = torch.arange(S).repeat_interleave(n)
    cur, nxt = Slot(B, dev), Slot(B, dev)
    g1, g2 = Relay(B, dev, 16.0, 1.0, 4.0), Relay(B, dev, 16.0, 1.0, 4.0)
    src_cur, src_nxt = sources(cur, 40), sources(nxt, 40)
    closed = torch.ones(B, dtype=torch.bool)
    rate = C_EXT * R_RESET * 1e-3
    group = lambda k: (cur.lab[None, :] == k).float() * (C_SEL * R_SEL * 1e-3)
    mask = [(sc == s)[:, None].float() for s in range(S)]
    total = 3800.0
    nb = int(total // TB)
    x = torch.zeros(nb, S, 8)       # группы 1–5, неизб., I текущей; макс. группа следующей
    xb = torch.zeros(nb, B, P)      # по испытаниям, группы текущей
    t0 = time.time()
    for step in range(int(round(total / DT))):
        t = step * DT
        de_c = torch.zeros(B, cur.ne)
        di_c = torch.zeros(B, cur.ni)
        de_n = torch.zeros(B, cur.ne)
        any_d = False
        for s, (_, hold4, resets, loads, loads4) in enumerate(SCEN):
            m = mask[s]
            if hold4 and 100 <= t < 150 or any(l <= t < l + 50 for l in loads4):
                de_c += m * group(3)
                de_n += m * group(3)
                any_d = True
            for r, kind in resets:
                if r <= t < r + 50:
                    di_c += m * rate
                    if kind == "burst":
                        de_c += m * rate
                        de_n += m * rate
                    any_d = True
            if any(l <= t < l + 50 for l in loads):
                de_c += m * group(0)
                if not resets:          # write: значение сброса — в обе памяти
                    de_n += m * group(0)
                any_d = True
        g1.step_counts(relay_input(cur.spk_e, src_cur, 1), closed)
        g2.step_counts(relay_input(nxt.spk_e, src_nxt, 0), closed)
        nxt.step(de_n if any_d else None, di_c if any_d else None, add_e=g1.to_slot(nxt.lab))
        cur.step(de_c if any_d else None, di_c if any_d else None, add_e=g2.to_slot(cur.lab))
        g = (cur.spk_e @ cur.onehot) / torch.tensor([cur.nsub] * P + [cur.nnon]) * 1e4    # Гц, [B, G]
        gi = cur.spk_i.mean(1, keepdim=True) * 1e4
        gn = ((nxt.spk_e @ nxt.onehot)[:, :P] / nxt.nsub * 1e4).max(1, keepdim=True).values
        b = int(t // TB)
        if b < nb:
            x[b] += torch.cat([g, gi, gn], 1).view(S, n, 8).mean(1)
            xb[b] += g[:, :P]
    x /= TB / DT
    xb /= TB / DT
    print(f"Сброс → позиция 1: {S} сценариев × {n} испытаний, {time.time() - t0:.0f} с")
    cols = ["гр1", "гр2", "гр3", "гр4", "гр5", "неизб", "I", "след.макс"]
    for s, (name, *_rest) in enumerate(SCEN):
        print(f"\n[{name}] бины {TB:.0f} мс, среднее по испытаниям, Гц")
        print("   t,мс " + " ".join(f"{c:>9s}" for c in cols))
        for b in range(int(900 // TB), nb):
            if s != 5 and b * TB >= 2000:
                break
            print(f"  {b * TB:5.0f} " + " ".join(f"{float(v):9.1f}" for v in x[b, s]))
    print("\nИтог («держит» > 10 Гц):")
    for s, (name, hold4, resets, loads, _l4) in enumerate(SCEN):
        sl = slice(s * n, (s + 1) * n)
        for l in loads:
            r0 = resets[0][0] if resets else l
            fl = xb[int(r0 // TB):int((r0 + 100) // TB), sl, 1:].amax((0, 2))       # вспышка групп 2–5
            win = xb[int((l + 300) // TB):int((l + 600) // TB), sl].mean(0)         # [n, P]
            alive4 = (xb[int((l + 300) // TB):int((l + 900) // TB), sl, 3] > 10).any(0)
            only1 = (win[:, 0] > 10) & ((win[:, 1:] > 10).sum(1) == 0)
            print(f"  {name:18s} поз.1 в {l:4.0f}: вспышка групп 2–5 при сбросе {fl.mean():5.1f} Гц"
                  f" (> 10 в {int((fl > 10).sum())}/{n}); поз.1 в окне {win[:, 0].mean():5.1f} Гц"
                  f" (по испытаниям {' '.join(f'{float(v):.0f}' for v in win[:, 0])}); одна поз.1 {int(only1.sum())}/{n};"
                  f" поз.4 {win[:, 3].mean():.1f} Гц, ожила {int(alive4.sum())}/{n}")


if __name__ == "__main__":
    main(dict(kv.split("=", 1) for kv in sys.argv[1:]))
