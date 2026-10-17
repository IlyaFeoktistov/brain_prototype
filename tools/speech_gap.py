"""
Пауза после своего слова: сколько спайков у мест слов, сколько клеток рта
стреляет и чьи они (по выученному входу «смысл → рот»), что сказано.
Своя речь фразой включена (own_speech_offset 1, own_speech_memory 1, sAHP
смысла 4 мВ). Окружение: TM — множитель «таламус → рот», MAHP — усталость
рта (мВ).

    TM=0.25 PYTHONPATH=. .venv/bin/python tools/speech_gap.py МОЗГ "Человек пьёт" ...
"""
import sys, math, torch
from brain.mind import Mind
mind = Mind.load(sys.argv[1]); mind.cfg.autosave = False
mind.cfg.own_speech_offset = 1.0; mind.cfg.own_speech_memory = 1.0
lang, brain = mind.language, mind.brain
E = brain.areas["association"].E; M = brain.areas["motor"].E; T = brain.areas["thalamus"].E
brain.slow_ahp_b[E.slice] = 4.0; brain.decay_slow_ahp[E.slice] = math.exp(-brain.cfg.dt_ms / 1500.0)
gap = brain.cfg.turn_gap_ms / 1000
import os
scale = float(os.environ.get("TM", "1"))
tm = brain.projection("thalamus -> motor"); tm.weight.mul_(scale)
if "MAHP" in os.environ:
    brain.slow_ahp_b[M.slice] = float(os.environ["MAHP"])
mouth = brain.projection("association -> motor")
def owner(cells):
    # чьи клетки рта: у кого из мест самый большой выученный вход
    tokens, pc = lang.concepts.table()
    pre = mouth.pre.long() - E.start
    out = {}
    for c in cells.tolist():
        w = torch.zeros(E.size, device=pre.device).index_add_(0, pre[c], mouth.weight[c].float())
        sc = w[pc].sum(1); out.setdefault(tokens[int(sc.argmax())], 0); out[tokens[int(sc.argmax())]] += 1
    return out
orig = lang._articulated
def traced(rates, heard):
    said = orig(rates, heard)
    tokens, cells = lang.concepts.table()
    mean = rates[E.slice][cells].mean(1) * 4 * gap   # спайков места за паузу
    top = mean.topk(4)
    tops = " ".join(f"{tokens[i]}:{float(v):.0f}" for v, i in zip(top.values, top.indices))
    fired = (rates[M.slice] > 0).nonzero().flatten()
    print(f"      клетки рта чьи: {owner(fired)}")
    print(f"    «{heard}» → {said[0] if said else '—'}: моторных спайков {float(rates[M.slice].sum()*gap):.0f} (клеток {int((rates[M.slice]>0).sum())}/{M.size}), таламус {float(rates[T.slice].mean()):.0f} Гц; места по спайкам: {tops}")
    return said
lang._articulated = traced
for q in sys.argv[2:]:
    print(q)
    with brain.frozen():
        print("  →", lang.converse(q).text)
