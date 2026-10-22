"""
Стенд звена сетчатка → LGN: что даёт каждый параметр клетки реле и их пары.

Мозг целиком (small) пишет прогон: 1.5 с пустой экран, потом 0.5 с слово —
спайки сетчатки (растр стенда со спонтанными 20 Гц) и всех клеток, что
шлют входы клеткам реле (TRN, местные тормозные, V1 L6). Затем в том же мозге
клетки реле делятся на группы, у каждой группы — своё сочетание параметров,
и прогон повторяется с входами, зажатыми по записи (engine.clamp_*): у каждой
клетки её настоящие входы и веса, но вход не отвечает на изменения реле —
причинный тест звена; вывод потом проверять в целом мозге.

Параметры (контроль → вариант):
    vt      V_T реле −50 → −59 (начало спайка −48 как в таламусе in vivo) [гипотеза]
    rest    фон реле +15 мВ (покой −63 в сети, а не без сети) [гипотеза]
    trn     TRN → реле выкл.
    triad   триада выкл.
    local   местные тормозные → реле выкл.
    cortex  V1 L6 → реле выкл.
    t       T-ток выкл.
    noise   случайный фон ядра выкл.
Сочетания: контроль, каждый отдельно, все пары; grid — V_T −50…−61 × TRN
×1/0.5/0.25/0 × фон +0/+15. Клетки, видящие слово,
делятся между сочетаниями поровну.

Меры по сочетанию (сумма зёрен): частота клеток слова на слове и на пустом
экране, частота прочих на пустом (спонтанная), efficacy водителя
(сетчатка i → реле i, 0.5–4.5 мс, минус случайные; макака 49 ± 14%) и
передача одиночного спайка после паузы > 30 мс (кошка ~30%), латентность и
длительность ответа (окна 5 мс выше пустого на 3σ), доля клеток слова, у
которых спайков на слове значимо больше, чем на пустом (различение).
Контроль сверяется с записью (верность зажима).

    PYTHONPATH=. python tools/lgn_link_bench.py [seeds=1,2,3] [word=мама] [combos=pairs|single|grid]
"""

from __future__ import annotations

import itertools
import sys

import torch

from brain.config import BrainConfig, log_path
from brain.mind import Mind
from brain.session import Session

FACTORS = ("vt", "rest", "trn", "triad", "local", "cortex", "t", "noise")
PATHS = {"trn": "trn -> lgn", "triad": "(триада)", "local": "lgn: I->E", "cortex": "visual -> lgn"}
BLANK_MS, WORD_MS, SETTLE_MS = 1500.0, 500.0, 500.0


def run_seed(seed: int, word: str, combos: list[tuple[str, ...]]):
    cfg = BrainConfig()
    cfg.size = "small"
    cfg.seed = seed
    cfg.autosave = False
    mind = Mind(cfg)
    b = mind.brain
    b.plasticity = False
    torch.manual_seed(seed)
    b.run(2500.0)
    b.CHUNK, b._graphs = 1, {}
    lgn = b.areas["lgn"].E
    dt = cfg.dt_ms
    eye_i = [x.name for x in b.sensors].index("eye")
    off, size = b.sensor_offsets[eye_i], b.sensors[eye_i].size
    spont = float(b.spontaneous[off])
    b.spontaneous[off: off + size] = 0.0
    inputs = [p for p in b.projections if p.post is lgn]
    pre = torch.unique(torch.cat([p.pre.long().flatten() for p in inputs if p.source == "net"]))
    pre = pre[(pre < lgn.start) | (pre >= lgn.start + lgn.size)]

    driver = next(p for p in inputs if p.name == "eye -> " + lgn.name)
    drv_col = (driver.pre[:, 0].long() - off).cpu()        # рецептор-водитель каждой клетки реле

    s = Session(mind)
    s.gaze = 0.0
    steps_blank, steps_word = int(BLANK_MS / dt), int(WORD_MS / dt)
    views, pre_rec, lgn_rec = [], [], []

    for t in range(steps_blank + steps_word):
        s.line = (word + " ") if t >= steps_blank else ""
        view = s._view(dt)
        retina = view[:, off: off + size]
        retina.copy_(((retina + (torch.rand_like(retina) < spont).float()) > 0).float())
        b.run(dt, view)
        views.append(view.clone())
        pre_rec.append(b.spikes[pre] > 0)
        lgn_rec.append((b.spikes[lgn.slice] > 0).cpu())

    # Клетки слова: их водитель на слове чаще, чем на пустом, на 20 Гц и больше.
    retina_rec = torch.stack([v[0, off: off + size].cpu() > 0 for v in views])
    rate = lambda a, z: retina_rec[a:z].float().mean(0) * 1000 / dt
    sees = (rate(steps_blank, None) - rate(0, steps_blank))[drv_col] >= 20.0
    # Группы клеток реле: поровну клеток слова и прочих на сочетание.
    g = torch.Generator().manual_seed(seed)
    group = torch.empty(lgn.size, dtype=torch.long)
    for mask in (sees, ~sees):
        idx = mask.nonzero().squeeze(1)
        idx = idx[torch.randperm(len(idx), generator=g)]
        group[idx] = torch.arange(len(idx)) % len(combos)

    for c, combo in enumerate(combos):
        cells = (group == c).nonzero().squeeze(1).to(b.device)
        for item in combo:
            f, _, val = item.partition("=")
            if f == "vt":
                b.v_t[lgn.start + cells] = float(val or -59.0)
            elif f == "rest":
                b.bias[lgn.start + cells] += float(val or 15.0)
            elif f == "trn" and val:
                # trn=k — проводимость TRN → реле ×k (сетка).
                for p in inputs:
                    if PATHS[f] in p.name:
                        p.weight[cells] *= float(val)
            elif f == "t":
                b.t_mask[lgn.start + cells] = False
            elif f == "noise":
                b.noise_gain[lgn.start + cells] = 0.0
            else:
                for p in inputs:
                    if PATHS[f] in p.name:
                        p.weight[cells] = 0.0

    b.clamp_on = True
    b.clamp_mask[pre] = True
    b.recompile()
    replay = []

    for t in range(steps_blank + steps_word):
        b.clamp_now[pre] = pre_rec[t]
        b.run(dt, views[t])
        replay.append((b.spikes[lgn.slice] > 0).cpu())

    return torch.stack(lgn_rec), torch.stack(replay), retina_rec[:, drv_col], sees, group, dt


def measure(L, R, sees, cells, dt):
    """Меры для клеток cells по растрам реле L и их водителей R [шаги × клетки реле]."""

    settle, blank_end = int(SETTLE_MS / dt), int(BLANK_MS / dt)
    word_cells = cells[sees[cells]]
    other = cells[~sees[cells]]
    hz = lambda r: float(r.float().mean()) * 1000 / dt if r.numel() else float("nan")
    out = {
        "слово/слово": hz(L[blank_end:, word_cells]),
        "слово/пусто": hz(L[settle:blank_end, word_cells]),
        "прочие/пусто": hz(L[settle:blank_end, other]),
    }
    pre, post = R[settle:, cells], L[settle:, cells]
    lo, hi = int(round(0.5 / dt)), int(round(4.5 / dt))
    window = torch.zeros_like(post)
    for lag in range(lo, hi + 1):
        window[:-lag] |= post[lag:]
    chance = float(post.float().mean()) * (hi - lo + 1)
    out["efficacy"] = float((pre & window).sum() / pre.sum().clamp(min=1)) - chance
    # Одиночный спайк сетчатки после паузы > 30 мс.
    gap = int(30 / dt)
    quiet = torch.zeros_like(pre)
    for lag in range(1, gap + 1):
        quiet[lag:] |= pre[:-lag]
    lone = pre & ~quiet
    lone[:gap] = False
    out["пауза"] = float((lone & window).sum() / lone.sum().clamp(min=1)) - chance
    # Ответ клеток слова по окнам 5 мс.
    k = int(5 / dt)
    base = L[settle:blank_end, word_cells].float().mean(1)
    base = base[: len(base) // k * k].view(-1, k).mean(1)
    resp = L[blank_end:, word_cells].float().mean(1)
    resp = resp[: len(resp) // k * k].view(-1, k).mean(1)
    above = (resp > base.mean() + 3 * base.std()).nonzero().squeeze(1)
    out["латентность"] = float(above[0]) * 5 if len(above) else float("nan")
    out["длительность"] = float(above[-1] - above[0] + 1) * 5 if len(above) else 0.0
    n_word = L[blank_end:, word_cells].sum(0).float()
    expect = L[settle:blank_end, word_cells].sum(0).float() * (WORD_MS / (BLANK_MS - SETTLE_MS))
    out["различают"] = float((n_word >= expect + 2 * expect.sqrt() + 1).float().mean()) if len(word_cells) else float("nan")
    out["клеток слова"] = len(word_cells)
    return out


def main(seeds: list[int], word: str, mode: str) -> None:
    combos = [()] + [(f,) for f in FACTORS]
    if mode == "pairs":
        combos += list(itertools.combinations(FACTORS, 2))
    if mode == "grid":
        # Соседи лучших пар: V_T × сила TRN × фон.
        combos = [tuple(x for x in (f"vt={vt}" if vt != -50 else "", f"trn={k}" if k != 1 else "",
                                    f"rest={r}" if r else "") if x)
                  for vt in (-50, -55, -57, -59, -61) for k in (1, 0.5, 0.25, 0) for r in (0, 15)]
    rows: dict[tuple, list] = {c: [] for c in combos}
    record = []

    for seed in seeds:
        L0, L, R, sees, group, dt = run_seed(seed, word, combos)
        for c, combo in enumerate(combos):
            rows[combo].append((L, R, sees, (group == c).nonzero().squeeze(1), dt))
        record.append((L0, R, sees, (group == 0).nonzero().squeeze(1), dt))
        print(f"зерно {seed}: записано и повторено", flush=True)
        # Растры — чтобы пересчитать меры без нового прогона.
        torch.save({"record": L0, "replay": L, "driver": R, "sees": sees, "group": group, "dt": dt,
                    "combos": combos, "word": word}, log_path(f"lgn-link-{mode}-{seed}.pt"))

    def pooled(items):
        ms = [measure(*x) for x in items]
        return {k: sum(m[k] for m in ms) / len(ms) for k in ms[0]}

    keys = ("слово/слово", "слово/пусто", "прочие/пусто", "efficacy", "пауза", "латентность", "длительность", "различают")
    head = f"{'сочетание':26s} " + " ".join(f"{k:>12s}" for k in keys)
    print(f"\nслово «{word}», зёрна {seeds}; частоты — Гц, латентность и длительность — мс")
    print(head)
    fmt = lambda name, m: f"{name:26s} " + " ".join(f"{m[k]:12.2f}" for k in keys)
    print(fmt("запись (контроль)", pooled(record)))

    for combo in combos:
        print(fmt("+".join(combo) or "контроль", pooled(rows[combo])), flush=True)
    print("опыт: efficacy 0.49 ± 0.14 (макака), пауза ~0.30 (кошка); спонтанная LGN ~10 Гц, на стимул — десятки")

    # Устойчивость: мин–макс по зёрнам.
    spread = ("слово/слово", "прочие/пусто", "efficacy", "пауза", "различают")
    print(f"\nпо зёрнам, мин–макс\n{'сочетание':26s} " + " ".join(f"{k:>14s}" for k in spread))
    for combo in combos:
        ms = [measure(*x) for x in rows[combo]]
        print(f"{'+'.join(combo) or 'контроль':26s} " + " ".join(
            f"{min(m[k] for m in ms):6.2f}–{max(m[k] for m in ms):<7.2f}" for k in spread), flush=True)


if __name__ == "__main__":
    opts = dict(kv.split("=", 1) for kv in sys.argv[1:])
    main([int(x) for x in opts.get("seeds", "1,2,3").split(",")], opts.get("word", "мама"), opts.get("combos", "pairs"))
