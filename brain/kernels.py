"""
Ядра шага на C++/CUDA (brain/csrc): собираются при первом запуске
(torch.utils.cpp_extension, кэш ~/.cache/torch_extensions). Не собрались —
шаг идёт прежним путём на PyTorch (compact_events = None).

compact_events — плотный список событий шага: только настоящие синапсы
выстреливших нейронов, в буфер постоянной ёмкости (форма постоянна — шаг
остаётся CUDA-графом и собирается torch.compile).

deliver, ltd, ltp — доставка спайков и STDP одним ядром каждое прямо по
строкам таблицы синапсов выстреливших нейронов: без буфера событий, ничего
не теряется (compact_events — только для пути PyTorch).
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

compact_events = deliver = ltd = ltp = plastic_update = conserve_rows = None
# Само расширение — для сверки ядер (tools/kernel_check.py).
ext = None

if torch.cuda.is_available() and os.environ.get("BRAIN_KERNELS", "1") == "1":
    # ninja и компилятор ищутся в PATH — добавить bin окружения проекта.
    import sys

    os.environ["PATH"] = os.path.join(sys.prefix, "bin") + os.pathsep + os.environ.get("PATH", "")

    try:
        from torch.utils.cpp_extension import load

        _ext = load(
            name="brain_events",
            sources=[str(Path(__file__).parent / "csrc" / "events.cu")],
            extra_cuda_cflags=["-O3"],
            verbose=False,
        )
        ext = _ext

        @torch.library.custom_op("brain::compact_events", mutates_args=("synapses", "pre"))
        def compact_events(
            fired: torch.Tensor, n: int, start: torch.Tensor, count: torch.Tensor, perm: torch.Tensor,
            offsets: torch.Tensor, synapses: torch.Tensor, pre: torch.Tensor,
        ) -> None:
            _ext.compact_events(fired, n, start, count, perm, offsets, synapses, pre)

        @compact_events.register_fake
        def _(fired, n, start, count, perm, offsets, synapses, pre) -> None:
            return None

        @torch.library.custom_op("brain::deliver", mutates_args=("dendrite", "drive", "glutamate", "inhibition", "ngf", "ring", "dend_ring"))
        def deliver(
            fired: torch.Tensor, n: int, start: torch.Tensor, count: torch.Tensor, perm: torch.Tensor,
            path_post: torch.Tensor, path_fanin: torch.Tensor, net_weight: torch.Tensor,
            path_start: torch.Tensor, path_flags: torch.Tensor, ach_paths: bool, ach_suppression: float,
            acetylcholine: torch.Tensor, dendrite_pre: torch.Tensor, dendrite_post: torch.Tensor,
            net_branch: torch.Tensor, branches: int,
            typed_pre: torch.Tensor, cell_type: torch.Tensor, kinds: int,
            stp_u4: torch.Tensor, stp_x4: torch.Tensor, stp_U4: torch.Tensor,
            is_lamp5: torch.Tensor, ngf_d: torch.Tensor, lamp5_scale: float, lamp5_kind: int,
            stp_x: torch.Tensor, inh_gain: torch.Tensor,
            dendrite: torch.Tensor, drive: torch.Tensor, glutamate: torch.Tensor | None,
            inhibition: torch.Tensor, ngf: torch.Tensor,
            path_delay: torch.Tensor, tick: torch.Tensor, ring: torch.Tensor, dend_ring: torch.Tensor,
        ) -> None:
            _ext.deliver(fired, n, start, count, perm, path_post, path_fanin, net_weight, path_start, path_flags, ach_paths,
                         ach_suppression, acetylcholine, dendrite_pre, dendrite_post, net_branch, branches,
                         typed_pre, cell_type, kinds, stp_u4, stp_x4, stp_U4, is_lamp5, ngf_d,
                         lamp5_scale, lamp5_kind, stp_x, inh_gain, dendrite, drive, glutamate, inhibition, ngf,
                         path_delay, tick, ring, dend_ring)

        @deliver.register_fake
        def _(*args) -> None:
            return None

        @torch.library.custom_op("brain::ltd", mutates_args=("eligibility",))
        def ltd(
            fired: torch.Tensor, n: int, start: torch.Tensor, count: torch.Tensor, perm: torch.Tensor,
            path_start: torch.Tensor, path_post: torch.Tensor, path_fanin: torch.Tensor, trace_post: torch.Tensor, ltd_mask: torch.Tensor, symmetric_area: torch.Tensor | None,
            a_plus: float, a_minus: float, net_weight: torch.Tensor, ltd_ref: torch.Tensor | None,
            scale: torch.Tensor, eligibility: torch.Tensor,
        ) -> None:
            _ext.ltd(fired, n, start, count, perm, path_start, path_post, path_fanin, trace_post, ltd_mask, symmetric_area,
                     a_plus, a_minus, net_weight, ltd_ref, scale, eligibility)

        @ltd.register_fake
        def _(*args) -> None:
            return None

        @torch.library.custom_op("brain::ltp", mutates_args=("eligibility",))
        def ltp(
            fired: torch.Tensor, n: int, start: torch.Tensor, count: torch.Tensor, perm: torch.Tensor, net_pre: torch.Tensor, ltp_kernel: torch.Tensor,
            a_plus: float, scale: torch.Tensor, eligibility: torch.Tensor,
        ) -> None:
            _ext.ltp(fired, n, start, count, perm, net_pre, ltp_kernel, a_plus, scale, eligibility)

        @ltp.register_fake
        def _(*args) -> None:
            return None

        def plastic_update(weight, eligibility, early, path_start, step, cap, spend, total: int) -> None:
            _ext.plastic_update(weight, eligibility, early, path_start, step, cap, spend, total)

        def conserve_rows(weight, early, row_start, row_len, row_target, row_cap) -> None:
            _ext.conserve_rows(weight, early, row_start, row_len, row_target, row_cap)
    except Exception as error:  # нет компилятора, другая CUDA — прежний путь
        print(f"ядра C++/CUDA не собрались, шаг на PyTorch: {error}")
        compact_events = deliver = ltd = ltp = plastic_update = conserve_rows = None
