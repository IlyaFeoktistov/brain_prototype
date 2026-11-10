"""
Доставка спайка пути ядра с депрессией (cfg.nucleus_stp) — настоящий шаг
движка: ядро CUDA (device=cuda) или шаг на PyTorch (path=torch — та же
ветка, что на процессоре, на любом устройстве; device=cpu — медленно).

Все клетки под зажимом (clamp_mask, engine._step); шум и сенсоры выключены;
стреляют только n клеток LGN по расписанию: серия 50 Гц × 5, затем
одиночный спайк через pause мс (1 с). Ресурс тратит и восстанавливает сам движок;
доставленный ток в V1 L4 E — прирост i_syn за шаг (/ syn_scale). Сила
спайка = ток / Σ w входов от этих клеток; сверка — ресурс до выброса,
посчитанный отдельно (x += (1 − x)·dt/τ каждый шаг; на спайке записать x,
затем x −= U·x). Старый счёт (ресурс после выброса) дал бы ×(1 − U).

    PYTHONPATH=. .venv/bin/python tools/nucleus_delivery.py [device=cuda|cpu] [path=torch] [pause=1000] [warm=200] [n=50] [save=файл]
"""

from __future__ import annotations

import sys

import torch

from brain.config import BrainConfig
from brain.mind import Mind


def main(opts: dict) -> None:
    cfg = BrainConfig()
    cfg.size = "small"
    cfg.autosave = False
    cfg.seed = 1
    cfg.device = opts.get("device", "cuda")
    b = Mind(cfg).brain
    b.plasticity = False
    if opts.get("path") == "torch":
        # Шаг на PyTorch (ветка без ядер, как на процессоре) — на любом устройстве.
        b.compact = False
        b.recompile()
    print(f"устройство {b.device}, ядро CUDA: {b._fused()}", flush=True)
    lgn = b.areas["lgn"].E
    L4 = b.areas["visual"].layers["L4"]["E"]
    tc = next(p for p in b.projections if p.post is L4 and p.name == f"lgn -> {L4.name}")
    n = int(opts.get("n", 50))
    cells = torch.arange(lgn.start, lgn.start + n, device=b.device)
    U = float(b.stp_u_cell[lgn.start])
    tau = next(t for s, d, _, t in cfg.nucleus_stp if (s, d) == ("lgn", "visual"))
    chosen = torch.zeros(b.n_slots, dtype=torch.bool, device=b.device)
    chosen[cells] = True
    W = float((tc.weight.float() * chosen[tc.pre.long()]).sum())

    # Тишина: шум, спонтанные рецепторы, все клетки — по зажиму.
    b.noise_mean.zero_()
    b.noise_std.zero_()
    b.spontaneous = None
    b.clamp_on = True
    b.clamp_mask.fill_(True)
    b.clamp_now.zero_()
    b.stp_x.fill_(1.0)
    b.recompile()
    for _ in range(int(opts.get("warm", 200))):
        b._step(torch.zeros_like(b.ear_input[0]), False)
    b.stp_x.fill_(1.0)

    dt = cfg.dt_ms
    gap = round(20.0 / dt)
    fire = [10 + k * gap for k in range(5)] + [10 + 4 * gap + round(float(opts.get("pause", 1000)) / dt)]
    steps = fire[-1] + 40
    decay = b.decay_syn[L4.slice] if torch.is_tensor(b.decay_syn) else b.decay_syn
    scale = b.syn_scale[L4.slice] if torch.is_tensor(b.syn_scale) else b.syn_scale
    got = []
    x, expect = 1.0, []
    zero = torch.zeros_like(b.ear_input[0])
    for t in range(steps):
        x += (1.0 - x) * dt / tau
        if t in fire:
            expect.append(x)
            x -= U * x
        b.clamp_now.zero_()
        if t in fire:
            b.clamp_now[cells] = True
        before = b.i_syn[L4.slice].clone()
        b._step(zero, False)
        arrived = float(((b.i_syn[L4.slice] - before * decay) / scale).sum())
        if abs(arrived) > 1e-6 * W:
            got.append((t, arrived / W))

    print(f"U {U}, τ {tau:.0f} мс; Σ w входов от {n} клеток в L4 E {W:.2f} мВ")
    print("спайк: шаг выстрела → шаг прихода, сила по движку / ресурс до выброса (отношение)")
    for k, (t_f, e) in enumerate(zip(fire, expect)):
        t_a, g = got[k] if k < len(got) else (None, float("nan"))
        print(f"  {k + 1}: {t_f} → {t_a}, {g:.5f} / {e:.5f} ({g / e:.5f}); старый счёт {e * (1 - U):.5f}")
    if len(got) != len(fire):
        print(f"  приходов {len(got)}, спайков {len(fire)} — лишние/пропущенные: {got[len(fire):]}")
    if "save" in opts:
        torch.save({"fire": fire, "got": got, "expect": expect}, opts["save"])


if __name__ == "__main__":
    main(dict(a.split("=", 1) for a in sys.argv[1:]))
