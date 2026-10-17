import sys; sys.path.insert(0, ".")
import torch
from brain.mind import Mind
m = Mind.load(sys.argv[1]); L = m.language; B = m.brain; cfg = m.cfg
q = sys.argv[2] if len(sys.argv) > 2 else "столица франции?"
words, cells = L.concepts.table()
E = L.meaning.E
with L.isolated():
    toks = L.tokenize(q)
    for i, t in enumerate(toks):
        L.experience([t], learn=False, answer=i == len(toks) - 1)
        nb = max(1, cfg.dendrite_branches)
        d = B.i_dend.view(-1, nb)[E.start:E.stop].amax(1)[cells].mean(1)
        s = (B.i_syn + B.i_slow)[E.slice][cells].mean(1)
        plateau = torch.sigmoid((d - cfg.dendrite_threshold_mv) / cfg.dendrite_slope_mv)
        top = d.topk(8).indices.tolist()
        print(f"«{t}»: дендрит мВ / плато / сома мВ:", ", ".join(f"{words[j]} {d[j]:.1f}/{plateau[j]:.2f}/{s[j]:.1f}" for j in top))
    print("всего мест с дендритом > 1 мВ:", int((d > 1).sum()), "> 5 мВ:", int((d > 5).sum()), "из", len(words))
    for w in ["париж", "москва", "рим", "берлин", "\n"]:
        if w in L.concepts:
            j = words.index(w); print(f"  {w!r}: дендрит {d[j]:.1f} плато {plateau[j]:.2f} сома {s[j]:.1f}")
