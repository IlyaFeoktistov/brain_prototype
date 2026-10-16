"""
Ядра шага на C++/CUDA (brain/csrc): собираются при первом запуске
(torch.utils.cpp_extension, кэш ~/.cache/torch_extensions). Не собрались —
шаг идёт прежним путём на PyTorch (compact_events = None).

compact_events — плотный список событий шага: только настоящие синапсы
выстреливших нейронов, в буфер постоянной ёмкости (форма постоянна — шаг
остаётся CUDA-графом и собирается torch.compile).

deliver, ltd, ltp — доставка спайков и STDP по этому списку одним ядром
каждое, только до настоящего числа событий (не по всей ёмкости).
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

compact_events = deliver = ltd = ltp = plastic_update = None

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

        @torch.library.custom_op("brain::compact_events", mutates_args=("synapses", "pre"))
        def compact_events(
            fired: torch.Tensor, n: int, start: torch.Tensor, count: torch.Tensor, perm: torch.Tensor,
            offsets: torch.Tensor, synapses: torch.Tensor, pre: torch.Tensor,
        ) -> None:
            _ext.compact_events(fired, n, start, count, perm, offsets, synapses, pre)

        @compact_events.register_fake
        def _(fired, n, start, count, perm, offsets, synapses, pre) -> None:
            return None

        @torch.library.custom_op("brain::deliver", mutates_args=("dendrite", "drive", "glutamate", "inhibition", "ngf"))
        def deliver(
            synapses: torch.Tensor, pre_of: torch.Tensor, total: torch.Tensor,
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
        ) -> None:
            _ext.deliver(synapses, pre_of, total, path_post, path_fanin, net_weight, path_start, path_flags, ach_paths,
                         ach_suppression, acetylcholine, dendrite_pre, dendrite_post, net_branch, branches,
                         typed_pre, cell_type, kinds, stp_u4, stp_x4, stp_U4, is_lamp5, ngf_d,
                         lamp5_scale, lamp5_kind, stp_x, inh_gain, dendrite, drive, glutamate, inhibition, ngf)

        @deliver.register_fake
        def _(*args) -> None:
            return None

        @torch.library.custom_op("brain::ltd", mutates_args=("eligibility",))
        def ltd(
            synapses: torch.Tensor, pre_of: torch.Tensor, total: torch.Tensor,
            path_start: torch.Tensor, path_post: torch.Tensor, path_fanin: torch.Tensor, trace_post: torch.Tensor, ltd_mask: torch.Tensor, symmetric_area: torch.Tensor | None,
            a_plus: float, a_minus: float, scale: torch.Tensor, eligibility: torch.Tensor,
        ) -> None:
            _ext.ltd(synapses, pre_of, total, path_start, path_post, path_fanin, trace_post, ltd_mask, symmetric_area,
                     a_plus, a_minus, scale, eligibility)

        @ltd.register_fake
        def _(*args) -> None:
            return None

        @torch.library.custom_op("brain::ltp", mutates_args=("eligibility",))
        def ltp(
            synapses: torch.Tensor, total: torch.Tensor, net_pre: torch.Tensor, ltp_kernel: torch.Tensor,
            a_plus: float, scale: torch.Tensor, eligibility: torch.Tensor,
        ) -> None:
            _ext.ltp(synapses, total, net_pre, ltp_kernel, a_plus, scale, eligibility)

        @ltp.register_fake
        def _(*args) -> None:
            return None

        def plastic_update(weight, eligibility, early, path_start, step, cap, spend, total: int) -> None:
            _ext.plastic_update(weight, eligibility, early, path_start, step, cap, spend, total)
    except Exception as error:  # нет компилятора, другая CUDA — прежний путь
        print(f"ядра C++/CUDA не собрались, шаг на PyTorch: {error}")
        compact_events = deliver = ltd = ltp = plastic_update = None
