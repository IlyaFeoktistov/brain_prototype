from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


class Metrics:
    def __init__(self):
        self.history = []

    def record(self, network, epoch: int, now_ms: float, rates: dict, lifecycle: dict):
        ca_e, ca_i = network.mean_calcium()

        self.history.append(
            {
                "epoch": epoch,
                "time_ms": now_ms,
                "alive_neurons": len(network.alive_ids()),
                "total_created_neurons": len(network.records),
                "connections": network.alive_connection_count(),
                "mean_rate_hz": (
                    sum(rates.values()) / len(rates) if rates else 0.0
                ),
                "mean_ca_excitatory": ca_e,
                "mean_ca_inhibitory": ca_i,
                "births": lifecycle["births"],
                "deaths": lifecycle["deaths"],
            }
        )

    def save(self, directory: str = "data"):
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)

        (out / "history.json").write_text(
            json.dumps(self.history, indent=2),
            encoding="utf-8",
        )

        if not self.history:
            return

        t = [x["time_ms"] / 1000 for x in self.history]
        neurons = [x["alive_neurons"] for x in self.history]
        connections = [x["connections"] for x in self.history]
        rates = [x["mean_rate_hz"] for x in self.history]

        plt.figure()
        plt.plot(t, neurons)
        plt.xlabel("time [s]")
        plt.ylabel("living neurons")
        plt.title("Neuron population")
        plt.savefig(out / "population.png", dpi=150)
        plt.close()

        plt.figure()
        plt.plot(t, connections)
        plt.xlabel("time [s]")
        plt.ylabel("connections")
        plt.title("Structural plasticity")
        plt.savefig(out / "connectivity.png", dpi=150)
        plt.close()

        plt.figure()
        plt.plot(t, rates)
        plt.xlabel("time [s]")
        plt.ylabel("mean firing rate [Hz]")
        plt.title("Network activity")
        plt.savefig(out / "activity.png", dpi=150)
        plt.close()
