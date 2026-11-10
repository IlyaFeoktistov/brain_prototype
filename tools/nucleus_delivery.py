"""
Доставка спайка пути ядра с депрессией (cfg.nucleus_stp) на таблицах
настоящего мозга: 50 клеток LGN с ресурсом до выброса x_до 0.2–1.0
(в stp_x — состояние после выброса, как его оставляет конец шага движка,
x_до · (1 − U)) стреляют через ядро CUDA deliver; ток в V1 L4 E (кольцо
задержек + прямой вход) должен быть Σ w · x_до. Старый счёт (ресурс после
выброса) дал бы 1 − U. Нужна GPU (ядра brain/csrc).

    PYTHONPATH=. .venv/bin/python tools/nucleus_delivery.py
"""
import torch
from brain.config import BrainConfig
from brain.mind import Mind
from brain import kernels as K
from brain.engine import KINDS

cfg = BrainConfig(); cfg.size = "small"; cfg.autosave = False; cfg.seed = 1
b = Mind(cfg).brain
b.plasticity = False
lgn = b.areas["lgn"].E
L4 = b.areas["visual"].layers["L4"]["E"]
tc = next(p for p in b.projections if p.post is L4 and p.name == f"lgn -> {L4.name}")
print("fused:", b._fused(), "U:", float(b.stp_u_cell[lgn.start]))
U = float(b.stp_u_cell[lgn.start])
cells = torch.arange(lgn.start, lgn.start + 50, device=b.device)
x_before = torch.linspace(0.2, 1.0, 50, device=b.device)
# Ожидаемый ток в L4 E: Σ по входам от выбранных клеток w · x_до.
pre = tc.pre.long()
w = tc.weight.float()
xb = torch.zeros(b.n_slots, device=b.device); xb[cells] = x_before
expect = (w * xb[pre]).sum(1)

# 1) ядро напрямую
saved = b.stp_x.clone()
b.stp_x.copy_(torch.ones_like(b.stp_x)); b.stp_x[cells] = x_before * (1 - U)
fired = torch.full((b.n,), b.n, dtype=torch.long, device=b.device); fired[:50] = cells
t = b.out_all
drive = torch.zeros(b.n_slots, device=b.device); z = lambda: torch.zeros_like(drive)
inh, ngf = z(), z(); ring = torch.zeros_like(b.ring); dring = torch.zeros_like(b.dend_ring); dend = torch.zeros_like(b.i_dend)
K.deliver(fired, b.n, t.start, t.count, t.perm, b.path_post, b.path_fanin, b.net_weight, b.path_start, b.path_flags,
          b.ach_paths, 0.0, b.acetylcholine, b.dendrite_pre, b.dendrite_post, b.net_branch, max(1, cfg.dendrite_branches),
          b.typed_pre, b.cell_type, len(KINDS), b.stp_u4, b.stp_x4, b.stp_U4, b.is_lamp5, b.ngf_d,
          1.0 / (1.0 - cfg.lamp5_desens_u), KINDS.index("LAMP5"), b.stp_x, b.stp_u_cell, b.inh_gain,
          dend, drive, None, inh, ngf, b.path_delay, b.tick, ring, dring)
got = (ring[:, 0, :].sum(0) + drive)[L4.slice]
m = expect > 0
print(f"1) ядро: клеток L4 со входом {int(m.sum())}, получено/ожидаемо {float(got[m].sum() / expect[m].sum()):.4f}, "
      f"макс. отн. ошибка {float(((got[m] - expect[m]).abs() / expect[m]).max()):.2e}; "
      f"старый счёт дал бы {1 - U:.2f}")
b.stp_x.copy_(saved)
