"""
Сверка доставки событий шага (brain/csrc/events.cu): новая — ядра идут по
строкам таблицы выстреливших нейронов, без буфера; прежняя — через буфер
событий (compact_events + *_buf), здесь с буфером под все события. Один
процесс, одно состояние: свежий мозг small, шаг не собран (compile_step
выкл.); спайки прошлого шага задаются — доля share нейронов (0, обычная,
и такие, при которых прежний буфер 6% переполнялся). Шаг делается новой
доставкой, состояние возвращается, тот же шаг (то же зерно шума) —
прежней; сравниваются все тензоры мозга, которые шаг меняет. Отличаться
может только порядок атомарных сложений (float) и округление fp16 следа.

    PYTHONPATH=. .venv/bin/python tools/kernel_check.py [shares=0,0.002,0.02,0.1] [plastic=1] [tol=1e-4]
"""

import sys

import torch

import brain.kernels as K
from brain.config import BrainConfig
from brain.mind import Mind

opts = {"shares": "0,0.002,0.02,0.1", "plastic": "1", "tol": "1e-4", "seed": "0"}
opts.update(dict(kv.split("=", 1) for kv in sys.argv[1:]))
cfg = BrainConfig()
cfg.save_path = "/nonexistent/brain.pt"
cfg.autosave = False
cfg.anatomy = True
cfg.hand = True
cfg.dt_ms = 0.5
cfg.compile_step = False
cfg.seed = int(opts["seed"])
torch.manual_seed(cfg.seed)
b = Mind(cfg).brain
plastic = opts["plastic"] == "1"
b.plasticity = plastic
b.run(200.0)
assert b._fused() and K.ext is not None, "быстрые ядра не включены — сверять нечего"
new = (K.deliver, K.ltd, K.ltp)


def compact(fired, n, start, count, perm):
    """Прежний список событий шага — в буфер под все события."""

    fired = fired.long()
    lengths = torch.where(fired < n, count.index_select(0, fired.clamp(max=n - 1)), 0).long()
    offsets = torch.cumsum(lengths, 0) - lengths
    total = lengths.sum().reshape(1)
    size = max(1, int(total))
    synapses = torch.zeros(size, dtype=torch.long, device=fired.device)
    pre = torch.zeros(size, dtype=torch.long, device=fired.device)
    K.ext.compact_events(fired, n, start, count, perm, offsets, synapses, pre)
    return synapses, pre, total


def deliver_buf(fired, n, start, count, perm, *rest):
    s, p, t = compact(fired, n, start, count, perm)
    K.ext.deliver_buf(s, p, t, *rest)


def ltd_buf(fired, n, start, count, perm, *rest):
    s, p, t = compact(fired, n, start, count, perm)
    K.ext.ltd_buf(s, p, t, *rest)


def ltp_buf(fired, n, start, count, perm, *rest):
    s, _, t = compact(fired, n, start, count, perm)
    K.ext.ltp_buf(s, t, *rest)


names = [k for k, v in vars(b).items() if isinstance(v, torch.Tensor) and v.is_cuda]
heard = torch.zeros(b.ear_input.shape[1], device=b.device)
ok = True

for share in (float(x) for x in opts["shares"].split(",")):
    g = torch.Generator(device=b.device).manual_seed(1)
    spikes = (torch.rand(b.n, device=b.device, generator=g) < share).float()
    b.spikes.zero_()
    b.spikes[: b.n] = spikes
    events = int(b.out_all.count[: b.n].long()[spikes > 0].sum())
    before = {k: getattr(b, k).clone() for k in names}
    results = []

    for kernels in (new, (deliver_buf, ltd_buf, ltp_buf)):
        for k in names:
            getattr(b, k).copy_(before[k])

        K.deliver, K.ltd, K.ltp = kernels
        torch.manual_seed(123)
        b._step(heard, plastic)
        torch.cuda.synchronize()
        results.append({k: getattr(b, k).clone() for k in names})

    K.deliver, K.ltd, K.ltp = new
    fresh, old = results
    worst = []

    for k in names:
        a, r = fresh[k].float(), old[k].float()

        if a.shape != r.shape or k == "dropped":
            continue

        changed = (r != before[k].float()) | (a != before[k].float())

        if not bool(changed.any()):
            continue

        diff = (a - r).abs()
        rel = float((diff / r.abs().clamp(min=1.0)).max())
        worst.append((rel, k, float(diff.max()), int(changed.sum())))

    worst.sort(reverse=True)
    bad = [w for w in worst if w[0] > float(opts["tol"])]
    ok &= not bad
    print(f"доля выстреливших {share}: событий {events} (прежний буфер 6% — {int(0.06 * int(b.out_all.perm.numel()))}); "
          f"изменено тензоров {len(worst)}, расхождение > {opts['tol']}: {len(bad)}", flush=True)

    for rel, k, d, n in worst[:5]:
        print(f"    {k:20s} отн. {rel:.2e}  абс. {d:.2e}  изменено элементов {n}")

print("СОВПАДАЕТ" if ok else "РАСХОЖДЕНИЕ")
