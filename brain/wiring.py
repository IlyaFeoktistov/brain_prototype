"""
Процедурная проводка: врождённые пути с постоянным весом не хранятся.

Получатели каждого нейрона вычисляются хешем (зерно пути, нейрон,
номер связи) прямо в шаге — как будто связи «записаны в генах», а не в
памяти. Хранить нужно только то, что меняется опытом.

У хранимого пути фиксировано число входов (k на получателя); у
вычисляемого — число выходов: k_in · |post| / |pre| на нейрон (дробная
часть — у части нейронов на одну связь больше). Всего связей столько же.
"""

from __future__ import annotations

import math

import torch

_M32 = 0xFFFFFFFF


def _mix(x: torch.Tensor) -> torch.Tensor:
    """32-битное перемешивание (Wellons): равномерный хеш целых."""

    x = x & _M32
    x = ((x ^ (x >> 16)) * 0x45D9F3B) & _M32
    x = ((x ^ (x >> 16)) * 0x45D9F3B) & _M32
    return x ^ (x >> 16)


def _hash(seed: torch.Tensor, neuron: torch.Tensor, slot: torch.Tensor) -> torch.Tensor:
    return _mix(seed * 0x9E3779B1 + _mix(neuron * 0x85EBCA6B + slot * 0xC2B2AE35))


class ProceduralProjection:
    """Путь pre -> post с постоянным весом; связи — из хеша."""

    source = "net"
    plastic = False
    predictive = True

    def __init__(self, name: str, pre_pop, post, k_in: int, weight: float, seed: int, device):
        self.name = name
        self.post_device = device
        self.pre_pop = pre_pop
        self.post = post
        self.k_in = k_in
        self.value = weight
        self.innate_range = (weight, weight)
        self.innate_max = weight
        self.seed = seed
        self.same = pre_pop is post
        self.mean_out = k_in * post.size / pre_pop.size
        self.n_synapses = k_in * post.size

    def degree(self, local: torch.Tensor) -> torch.Tensor:
        """Число выходов нейронов (локальные индексы в pre_pop)."""

        low = math.floor(self.mean_out)
        frac = self.mean_out - low
        extra = (_hash(torch.full_like(local, self.seed), local, torch.full_like(local, 1 << 20)) % 1024) < int(frac * 1024)
        return low + extra.long()

    def targets(self, local: torch.Tensor, slot: torch.Tensor) -> torch.Tensor:
        """Локальный индекс получателя связи slot нейрона local."""

        seed = torch.full_like(local, self.seed)
        target = _hash(seed, local, slot) % self.post.size

        if self.same:
            # Без автапсов.
            target = torch.where(target == local, (target + 1) % self.post.size, target)

        return target

    def outgoing(self, neuron: int) -> torch.Tensor:
        """Глобальные индексы получателей нейрона neuron."""

        device = self.post_device
        local = torch.tensor([neuron - self.pre_pop.start], device=device)
        slots = torch.arange(int(self.degree(local)[0]), device=device)
        return self.post.start + self.targets(local.expand_as(slots), slots)

    def incoming(self, row: int) -> torch.Tensor:
        """Глобальные индексы нейронов, связанных с получателем row (локальный)."""

        device = self.post_device
        local = torch.arange(self.pre_pop.size, device=device)
        width = int(self.degree(local).max()) if len(local) else 0
        slots = torch.arange(width, device=device)
        targets = self.targets(local[:, None].expand(-1, width), slots[None, :].expand(len(local), -1))
        valid = slots[None, :] < self.degree(local)[:, None]
        hits = (targets == row) & valid
        return self.pre_pop.start + hits.nonzero()[:, 0]

    def drive(self, rates: torch.Tensor) -> torch.Tensor:
        """Σ rate[pre] · вес по каждому получателю: [post.size]."""

        device = rates.device
        local = (rates[self.pre_pop.slice] > 0).nonzero().flatten()
        out = torch.zeros(self.post.size, device=device)

        if not len(local):
            return out

        degree = self.degree(local)
        slots = torch.arange(int(degree.max()), device=device)
        targets = self.targets(local[:, None].expand(-1, len(slots)), slots[None, :].expand(len(local), -1))
        valid = slots[None, :] < degree[:, None]
        rate = rates[self.pre_pop.start + local][:, None].expand_as(targets)
        out.index_add_(0, targets[valid], rate[valid] * self.value)
        return out


class ProceduralRows:
    """
    Вычисляемая строка каждого нейрона для шага: подряд выходы всех его
    процедурных путей. Хранится только [нейрон, путь] -> номер пути и
    накопленные числа выходов (несколько чисел на нейрон).
    """

    def __init__(self, projections: list[ProceduralProjection], n: int, n_slots: int, dummy_neurons: int, device):
        self.projections = projections
        self.n = n
        self.dummy_neurons = dummy_neurons
        count = torch.zeros(n_slots, dtype=torch.long, device=device)

        for proj in projections:
            count[proj.pre_pop.slice] += 1

        self.per_neuron = max(1, int(count.max())) if projections else 1
        # Пустая клетка указывает на фиктивный путь (последний).
        which = torch.full((n_slots, self.per_neuron), len(projections), dtype=torch.long, device=device)
        degree = torch.zeros(n_slots, self.per_neuron, dtype=torch.long, device=device)
        column = torch.zeros(n_slots, dtype=torch.long, device=device)

        for q, proj in enumerate(projections):
            pop = proj.pre_pop.slice
            col = int(column[pop][0]) if proj.pre_pop.size else 0
            which[pop, col] = q
            degree[pop, col] = proj.degree(torch.arange(proj.pre_pop.size, device=device))
            column[pop] += 1

        cum = torch.zeros(n_slots, self.per_neuron + 1, dtype=torch.long, device=device)
        cum[:, 1:] = degree.cumsum(1)
        # int32: несколько чисел на нейрон, а не на синапс.
        self.which = which.to(torch.int32)
        self.cum = cum.to(torch.int32)
        width = int(cum[:, -1].max()) if projections else 1
        self.width = max(64, -(-width // 64) * 64)
        self.cols = torch.arange(self.width, device=device)

        # Параметры путей; последний — фиктивный (ни одного выхода).
        def column_of(values, dummy):
            return torch.tensor(values + [dummy], device=device)

        self.seed = column_of([p.seed for p in projections], 0)
        self.pre_start = column_of([p.pre_pop.start for p in projections], 0)
        self.post_start = column_of([p.post.start for p in projections], n)
        self.post_size = column_of([p.post.size for p in projections], 1)
        self.same = column_of([p.same for p in projections], False)
        self.value = column_of([float(p.value) for p in projections], 0.0).float()

        # Входов у получателя — сколько выпало хешу (28–80 при k = 50).
        # Синаптическое масштабирование: вес × k / входов, суммарный вход
        # каждой клетки — как при фиксированном k. [замер] без этого
        # разброс торможения ломал тесты (инициатива, ругань, вопросы).
        scales, offsets, offset = [], [], 0
        for proj in projections:
            offsets.append(offset)
            local = torch.arange(proj.pre_pop.size, device=device)
            degree_ = proj.degree(local)
            width_ = int(degree_.max()) if len(local) else 0
            slots = torch.arange(width_, device=device)
            targets = proj.targets(local[:, None].expand(-1, width_), slots[None, :].expand(len(local), -1))
            valid = slots[None, :] < degree_[:, None]
            indegree = torch.bincount(targets[valid], minlength=proj.post.size).float()
            scales.append((proj.k_in / indegree.clamp(min=1.0)).to(torch.float16))
            offset += proj.post.size
        offsets.append(offset)
        scales.append(torch.zeros(1, dtype=torch.float16, device=device))
        self.post_scale = torch.cat(scales)
        self.scale_offset = torch.tensor(offsets, device=device)

    def rows(self, neurons: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Получатели и веса вычисляемых строк neurons: [len, width] каждое."""

        neurons = neurons.long()
        cum = self.cum.index_select(0, neurons).long()
        which = self.which.index_select(0, neurons).long()
        cols = self.cols

        segment = (cols[None, :, None] >= cum[:, None, 1:]).sum(-1).clamp(max=self.per_neuron - 1)
        q = which.gather(1, segment)
        slot = cols[None, :] - cum.gather(1, segment)
        valid = cols[None, :] < cum[:, -1:]

        local = neurons[:, None] - self.pre_start[q]
        size = self.post_size[q]
        target = _hash(self.seed[q], local, slot) % size
        target = torch.where(self.same[q] & (target == local), (target + 1) % size, target)

        dummy = self.n + (neurons[:, None] * self.width + cols) % self.dummy_neurons
        post = torch.where(valid, self.post_start[q] + target, dummy)
        scale = self.post_scale.index_select(0, (self.scale_offset[q] + target).flatten()).view_as(target)
        weight = torch.where(valid, self.value[q] * scale.float(), torch.zeros_like(self.value[q]))

        return post, weight

    def nbytes(self) -> int:
        return self.which.numel() * 4 + self.cum.numel() * 4 + self.post_scale.numel() * 2
