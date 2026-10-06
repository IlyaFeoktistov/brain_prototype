from __future__ import annotations

import nest


class LifecycleController:
    """Slow biological timescale: neurogenesis, pruning and homeostasis."""

    def __init__(self, network, cfg):
        self.network = network
        self.cfg = cfg

    def update(self, now_ms: float, rates: dict[int, float]) -> dict:
        self._update_patience(rates)
        births = self._maybe_birth(rates, now_ms) if self.cfg.neurogenesis_enabled else 0
        deaths = self._maybe_death(rates, now_ms) if self.cfg.neuron_death_enabled else 0
        return {"births": births, "deaths": deaths}

    def _update_patience(self, rates):
        for gid, record in self.network.records.items():
            if not record.alive:
                continue
            if rates.get(gid, 0.0) < self.cfg.death_rate_threshold_hz:
                record.low_activity_epochs += 1
            else:
                record.low_activity_epochs = 0

    def _maybe_birth(self, rates, now_ms):
        if len(self.network.alive_ids()) >= self.cfg.max_total_neurons or not rates:
            return 0
        mean_rate = sum(rates.values()) / len(rates)
        if mean_rate >= self.cfg.birth_rate_threshold_hz:
            return 0
        count = min(self.cfg.max_births_per_epoch,
                    max(1, int((self.cfg.birth_rate_threshold_hz - mean_rate) * 2)))
        return len(self.network.add_neurons(count, now_ms))

    def _maybe_death(self, rates, now_ms):
        candidates = []
        for gid, record in self.network.records.items():
            if not record.alive:
                continue
            if now_ms - record.born_ms < self.cfg.min_age_ms_before_death:
                continue
            if record.low_activity_epochs >= self.cfg.death_patience_epochs:
                candidates.append(gid)

        allowed = max(0, len(self.network.alive_ids()) - 100)
        killed = 0
        for gid in candidates[:allowed]:
            node = nest.NodeCollection([gid])
            degree = len(nest.GetConnections(target=node)) + len(nest.GetConnections(source=node))
            if degree <= self.cfg.min_connections_before_death:
                self.network.kill_neuron(gid)
                killed += 1
        return killed
