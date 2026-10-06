from brain.config import BrainConfig
from brain.lifecycle import LifecycleController
from brain.metrics import Metrics
from brain.network import BrainNetwork


def main() -> None:
    cfg = BrainConfig()
    network = BrainNetwork(cfg)
    network.setup()
    lifecycle = LifecycleController(network, cfg)
    metrics = Metrics()

    import nest

    print("brain prototype")
    print("architecture: AdEx/Clopath + dopamine + structural plasticity + homeostasis")
    print(f"initial neurons: {len(network.alive_ids())}")
    print(f"neuron death: {'ON' if cfg.neuron_death_enabled else 'OFF'}")

    now_ms = 0.0
    for epoch in range(cfg.epochs):
        nest.Simulate(cfg.epoch_ms)
        now_ms += cfg.epoch_ms
        rates = network.neuron_rates(cfg.epoch_ms)
        lifecycle_result = lifecycle.update(now_ms, rates)
        metrics.record(network, epoch, now_ms, rates, lifecycle_result)
        mean_rate = sum(rates.values()) / len(rates) if rates else 0.0
        print(
            f"[{now_ms / 1000:6.1f}s] neurons={len(network.alive_ids()):4d} "
            f"connections={network.alive_connection_count():7d} "
            f"rate={mean_rate:.3f}Hz +{lifecycle_result['births']} "
            f"-{lifecycle_result['deaths']}"
        )
    metrics.save()
    print("done; results are in ./data")


if __name__ == "__main__":
    main()
