"""
E1, шаг 3: счётчик позиции — устройство Spaun (Pos_Inc_Network, Eliasmith
et al. 2012, https://doi.org/10.1126/science.1225266; код spaun2.0
`_spaun/modules/working_memory.py`, `info_encoding.py`): две памяти —
«текущая» и «следующая»; следующая грузит текущая ⊛ INC, потом текущая
грузит следующую. У нас способ (Б): обе памяти — ячейки Brunel & Wang 2001
(tools/e1_slot.py, 5 групп = 5 позиций), ворота — реле MD под SNr
(tools/e1_gate.py, фон реле 1 Гц, кора → реле 4 нС, SNr 30 × 60 Гц по 2 нС).

  канал G1: группа k текущей → реле k+1 → группа k+1 следующей (сдвиг на
            единицу — заданная проводка, как INC у Spaun; долг: не выучен);
  канал G2: группа k следующей → реле k → группа k текущей.
  Вход реле — спайки первых nsrc = 40 клеток группы-источника (ведущий
  вход), SNr → реле 16 нС. Подбор по передаточной кривой реле (2026-11-18):
  закрытые ворота должны держать всплеск записи ~170 Гц (при 20 источниках
  и SNr 2 нС реле пробивало → цепная реакция 1→2→…→5 за 100 мс), открытые —
  пропускать удерживаемую позицию 20–25 Гц. 40 / 16 нС: открыто 20 Гц →
  реле 29 Гц, 25 → 53; закрыто 170 Гц → 0.1, 250 → 9.

Сдвиг (каждые period мс, с t0): 0–50 мс сброс следующей (общий залп
авторов) → 100–200 открыт G1 → 400–450 сброс текущей → 500–600 открыт G2.
Расписание ворот и сбросов задаёт опыт (заглушка: правила BG «сдвинь» нет,
ISSUES 28). Счётчик не знает, что писать в позиции, — он только держит
текущую позицию.

Начало: позиция 1 в текущую — стимул авторов 50 мс (заглушка «начало
фразы»). Проверка: после каждого из 4 сдвигов текущая = следующая
позиция, без пропусков и двойных шагов; мера — победитель в окне
period−200…period мс, «держит» > 10 Гц.

    PYTHONPATH=tools python tools/e1_counter.py [trials=8] [seed=1] [period=1000] [gsnr=16] [nsrc=40] [gate=200] [reset=burst|inhib]
"""

from __future__ import annotations

import sys
import time

import torch

from e1_gate import NR, Relay
from e1_slot import C_EXT, C_SEL, DT, P, R_RESET, R_SEL, Slot


def sources(slot: Slot, n: int) -> torch.Tensor:
    """Индексы первых n клеток каждой группы-позиции [P, n]."""

    return torch.stack([torch.nonzero(slot.lab == k).flatten()[:n] for k in range(P)])


def relay_input(spk: torch.Tensor, src: torch.Tensor, shift: int) -> torch.Tensor:
    """Спайки групп-источников → ведущий вход реле [B, P·NR]; реле j слушает группу j − shift."""

    per = spk[:, src].sum(2)                      # [B, P] спайков из NW клеток группы
    out = torch.zeros_like(per)
    if shift:
        out[:, shift:] = per[:, :-shift]
    else:
        out = per
    return out.repeat_interleave(NR, 1)


def main(opts: dict) -> None:
    dev = opts.get("device", "cpu")
    torch.set_num_threads(int(opts.get("threads", 2)))
    torch.manual_seed(int(opts.get("seed", 1)))
    B = int(opts.get("trials", 8))
    period = float(opts.get("period", 1000.0))
    cur, nxt = Slot(B, dev), Slot(B, dev)
    gsnr = float(opts.get("gsnr", 16.0))
    nsrc = int(opts.get("nsrc", 40))
    gate = float(opts.get("gate", 200.0))  # мс открытых ворот (100 мс — копия не всегда зажигает: 2/8 путей)
    g1, g2 = Relay(B, dev, gsnr, 1.0, 4.0), Relay(B, dev, gsnr, 1.0, 4.0)
    src_cur, src_nxt = sources(cur, nsrc).to(dev), sources(nxt, nsrc).to(dev)
    n_adv = 4
    total = period * (n_adv + 1)
    # Форма сброса ячеек счётчика (шаг 5в, tools/e1_counter_reset.py): burst — залп авторов
    # Brunel–Wang на все E и I; inhib — тот же залп только на тормозные (как сброс nengo
    # InputGatedMemory — торможение клеток памяти): без вспышки всех позиций и хвоста NMDA.
    reset_mode = opts.get("reset", "burst")
    reset_e = None if reset_mode == "inhib" else torch.full((B, cur.ne), C_EXT * R_RESET * 1e-3, device=dev)
    reset_i = torch.full((B, cur.ni), C_EXT * R_RESET * 1e-3, device=dev)
    start = (cur.lab[None, :] == 0).float().expand(B, cur.ne) * (C_SEL * R_SEL * 1e-3)
    closed, opened = torch.ones(B, dtype=torch.bool, device=dev), torch.zeros(B, dtype=torch.bool, device=dev)
    win = [(period * (k + 1) - 200.0, period * (k + 1)) for k in range(n_adv + 1)]
    cnt_cur = torch.zeros(n_adv + 1, B, P)
    cnt_nxt = torch.zeros(n_adv + 1, B, P)
    t0 = time.time()
    for step in range(int(round(total / DT))):
        t = step * DT
        ph = t % period
        adv = t >= period
        de_c = di_c = de_n = di_n = None
        if 200 <= t < 250:
            de_c = start
        if adv and ph < 50:
            de_n, di_n = reset_e, reset_i
        if adv and 400 <= ph < 450:
            de_c, di_c = reset_e, reset_i
        g1.step_counts(relay_input(cur.spk_e, src_cur, 1), opened if adv and 100 <= ph < 100 + gate else closed)
        g2.step_counts(relay_input(nxt.spk_e, src_nxt, 0), opened if adv and 500 <= ph < 500 + gate else closed)
        nxt.step(de_n, di_n, add_e=g1.to_slot(nxt.lab))
        cur.step(de_c, di_c, add_e=g2.to_slot(cur.lab))
        for k, (a, b) in enumerate(win):
            if a <= t < b:
                cnt_cur[k] += (cur.spk_e @ cur.onehot)[:, :P].cpu()
                cnt_nxt[k] += (nxt.spk_e @ nxt.onehot)[:, :P].cpu()
    rc = cnt_cur / cur.nsub / 0.2
    rn = cnt_nxt / nxt.nsub / 0.2
    print(f"Счётчик позиции: {B} пар ячеек, SNr → реле {gsnr} нС, источников на реле {nsrc}, ворота {gate:.0f} мс, сдвиг каждые {period:.0f} мс, {time.time() - t0:.0f} с")
    ok_all = torch.ones(B, dtype=torch.bool)
    for k in range(n_adv + 1):
        mc, mn = rc[k].mean(0), rn[k].mean(0)
        win_c = rc[k].argmax(1)
        single = (rc[k] > 10).sum(1) == 1
        ok = single & (win_c == k)
        ok_all &= ok
        print(f"  после {k} сдвигов: текущая {' '.join(f'{float(x):5.1f}' for x in mc)} | следующая "
              f"{' '.join(f'{float(x):5.1f}' for x in mn)} Гц; позиция {k + 1} одна: {int(ok.sum())}/{B}")
    print(f"  весь путь 1→{n_adv + 1} без ошибок: {int(ok_all.sum())}/{B}")


if __name__ == "__main__":
    main(dict(kv.split("=", 1) for kv in sys.argv[1:]))
