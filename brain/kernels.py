"""
Ядра шага на C++/CUDA (brain/csrc): собираются при первом запуске
(torch.utils.cpp_extension, кэш ~/.cache/torch_extensions). Не собрались —
шаг идёт прежним путём на PyTorch (compact_events = None).

compact_events — плотный список событий шага: только настоящие синапсы
выстреливших нейронов, в буфер постоянной ёмкости (форма постоянна — шаг
остаётся CUDA-графом и собирается torch.compile).
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

compact_events = None

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
    except Exception as error:  # нет компилятора, другая CUDA — прежний путь
        print(f"ядра C++/CUDA не собрались, шаг на PyTorch: {error}")
        compact_events = None
