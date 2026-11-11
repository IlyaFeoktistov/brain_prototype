"""
Депрессия синапса таламус → кора: нынешняя модель против восстановления,
ускоряемого кальцием (этап A — один синапс, мозг не меняется).

Модели (сила спайка — ресурс до выброса, как в движке после b0f0dd4):
  tm  — нынешняя (cfg.nucleus_stp): x −= U·x на спайк, восстановление
        (1 − x)/τ, U 0.45, τ 500 мс (срез, Lee & Sherman 2008);
  ca  — Dittman & Regehr 1998 (https://doi.org/10.1523/JNEUROSCI.18-16-06147.1998,
        climbing fiber → Purkinje, срез) в подгонке Cimenser & Miller 2014
        (https://doi.org/10.1371/journal.pone.0106046, уравнения 5–10) к
        in vivo кошки (Boudreau & Ferster 2005): на спайк N −= p0·N, Ca += Ca0;
        Ca спадает к Ca_rest с τ_Ca; скорость восстановления
        k(Ca) = k_max · Ca / (Ca + Ca0) (K_N = 1), в покое
        k0 = k_max / (1 + Ca0/Ca_rest). Числа подгонки: p0 0.85, k_max 84 с⁻¹,
        k0/k_max 0.03 (τ покоя ~400 мс), τ_Ca 3 мс. Механизм — из мозжечка;
        для таламуса это феноменологическая подгонка (сами авторы: τ_Ca
        намного меньше, чем у Dittman & Regehr).

Счёт по шагам dt как в движке (в конце шага восстановление, затем выброс;
  у ca — точное решение за шаг, τ_Ca 3 мс сравнима с шагом);
у ca — сверка с точным решением между спайками (уравнения 9–10 статьи).

Режимы (доля от силы отдохнувшего синапса, если не сказано иное):
  1. протокол Boudreau & Ferster: 1.75 с пуассоновского фона 11.8 Гц, затем
     серия 20 / 50 / 100 Гц × 6 — спайки относительно первого; снижение фона
     до 4.1 Гц на 5 с — первый спайк серии 50 Гц относительно контроля
     (данные: 20 Гц 5-й 0.80, 50 Гц 6-й 0.71, 100 Гц 6-й 0.60, минимум ко
     2–3-му; первый ВПСП +45 ± 11%);
  2. отдохнувший синапс; средняя сила при пуассоновских 1 Гц (из неё
     определён thalamic_psp_mv: Bruno & Sakmann 2006 — 0.82);
  3. парные импульсы из покоя, интервал 10–1000 мс;
  4. восстановление после 20 Гц × 10: тест через 50–1000 мс (бодрствующий
     кролик: > 750 мс — Stoelzel et al. 2008);
  5. передача на пуассоновском фоне (частота × средняя сила): 8 → 21 Гц
     (реле под буквой в мозге) и потолок.

    PYTHONPATH=. .venv/bin/python tools/tc_calcium.py [trials=2000] [seed=1] [dt=0.5]
"""

from __future__ import annotations

import math
import sys

import torch

TM = {"U": 0.45, "tau": 500.0}
CA = {"p0": 0.85, "kmax": 84.0 / 1000.0, "k0_ratio": 0.03, "tau_ca": 3.0}


class Synapse:
    """Пачка независимых синапсов (по одному на испытание), счёт по шагам dt."""

    def __init__(self, kind: str, n: int, dt: float):
        self.kind, self.dt = kind, dt
        self.x = torch.ones(n, dtype=torch.float64)
        if kind == "ca":
            # Единицы: Ca0 = 1 (прирост на спайк), Ca_rest = 1 / (kmax/k0 − 1).
            self.ca_rest = 1.0 / (1.0 / CA["k0_ratio"] - 1.0)
            self.ca = torch.full((n,), self.ca_rest, dtype=torch.float64)
            self.decay = math.exp(-dt / CA["tau_ca"])

    def step(self, spikes: torch.Tensor) -> torch.Tensor:
        """Один шаг: сила спайков этого шага (ресурс до выброса), затем восстановление и выброс."""

        w = self.x * spikes
        if self.kind == "tm":
            self.x = self.x + (1.0 - self.x) * (1.0 - math.exp(-self.dt / TM["tau"]))
            self.x = self.x - TM["U"] * self.x * spikes
        else:
            # Точное решение за шаг (уравнение 10 статьи): восстановление
            # с k0 плюс прибавка, пока кальций спадает за шаг.
            ca_next = self.ca_rest + (self.ca - self.ca_rest) * self.decay
            k0 = CA["kmax"] * self.ca_rest / (self.ca_rest + 1.0)
            extra = CA["tau_ca"] * (CA["kmax"] - k0) * torch.log((1.0 + self.ca) / (1.0 + ca_next))
            self.x = 1.0 - (1.0 - self.x) * torch.exp(-k0 * self.dt - extra)
            self.ca = ca_next
            self.x = self.x - CA["p0"] * self.x * spikes
            self.ca = self.ca + spikes
        return w

    def run(self, raster: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.step(raster[:, t]) for t in range(raster.shape[1])], 1)


def exact_ca(times: list[float]) -> list[float]:
    """Точное решение модели ca между спайками (Cimenser & Miller, уравнения 9–10)."""

    kmax, tau = CA["kmax"], CA["tau_ca"]
    rest = 1.0 / (1.0 / CA["k0_ratio"] - 1.0)
    k0 = kmax * rest / (rest + 1.0)
    n, ca, last, out = 1.0, rest, None, []
    for t in times:
        if last is not None:
            d = t - last
            ca_d = rest + (ca - rest) * math.exp(-d / tau)
            n = 1.0 - (1.0 - n) * math.exp(-k0 * d) * ((1.0 + ca) / (1.0 + ca_d)) ** (-tau * (kmax - k0))
            ca = ca_d
        out.append(n)
        n *= 1.0 - CA["p0"]
        ca += 1.0
        last = t
    return out


def poisson(n: int, steps: int, hz: float, dt: float, g: torch.Generator) -> torch.Tensor:
    return (torch.rand(n, steps, generator=g, dtype=torch.float64) < hz * dt / 1000.0).double()


def train(n: int, hz: float, count: int, dt: float, tail: float = 0.0) -> torch.Tensor:
    steps = int(round((count / hz * 1000.0 + tail) / dt))
    r = torch.zeros(n, steps, dtype=torch.float64)
    for j in range(count):
        r[:, int(round(j * 1000.0 / hz / dt))] = 1.0
    return r


def main(opts: dict) -> None:
    n = int(opts.get("trials", 2000))
    dt = float(opts.get("dt", 0.5))
    g = torch.Generator().manual_seed(int(opts.get("seed", 1)))
    kinds = ("tm", "ca")
    print(f"tm: U {TM['U']}, τ {TM['tau']:.0f} мс; ca: p0 {CA['p0']}, k_max {CA['kmax'] * 1000:.0f} с⁻¹, "
          f"k0/k_max {CA['k0_ratio']} (τ покоя {1 / (CA['kmax'] * CA['k0_ratio']):.0f} мс), τ_Ca {CA['tau_ca']} мс; "
          f"шаг {dt} мс, испытаний {n}")

    # Сверка счёта по шагам с точным решением (ca): серия 50 Гц × 6 после паузы и тест через 300 мс.
    times = [j * 20.0 for j in range(6)] + [100.0 + 300.0]
    steps = int(round((times[-1] + 1.0) / dt))
    r = torch.zeros(1, steps, dtype=torch.float64)
    for t in times:
        r[0, int(round(t / dt))] = 1.0
    w = Synapse("ca", 1, dt).run(r)[0]
    stepped = w[w > 0].tolist()
    print("сверка ca, шаги против точного: " + " ".join(f"{a:.3f}/{b:.3f}" for a, b in zip(stepped, exact_ca(times))))

    print("\n1. Boudreau & Ferster: фон 11.8 Гц 1.75 с, затем серия × 6 (относительно 1-го серии)")
    pre = int(round(1750.0 / dt))
    data = {20: "5-й 0.80", 50: "6-й 0.71", 100: "6-й 0.60"}
    first = {}
    for kind in kinds:
        for hz in (20, 50, 100):
            r = torch.cat([poisson(n, pre, 11.8, dt, g), train(n, hz, 6, dt, tail=1.0)], 1)
            w = Synapse(kind, n, dt).run(r)[:, pre:]
            amp = torch.stack([w[:, int(round(j * 1000.0 / hz / dt))] for j in range(6)], 1).mean(0)
            first[(kind, hz)] = amp
            print(f"  {kind} {hz:3d} Гц: " + " ".join(f"{float(a / amp[0]):.2f}" for a in amp)
                  + f"   (1-й {float(amp[0]):.2f} от отдохнувшего; данные {data[hz]})")
        low = int(round(5000.0 / dt))
        r = torch.cat([poisson(n, pre, 11.8, dt, g), poisson(n, low, 4.1, dt, g), train(n, 50, 6, dt, tail=1.0)], 1)
        w = Synapse(kind, n, dt).run(r)[:, pre + low:]
        amp = torch.stack([w[:, int(round(j * 20.0 / dt))] for j in range(6)], 1).mean(0)
        ctrl = first[(kind, 50)]
        print(f"  {kind} фон 4.1 Гц: серия 50 Гц к контролю по спайкам " + " ".join(f"×{float(a / c):.2f}" for a, c in zip(amp, ctrl))
              + " (данные: 1-й ×1.45 ± 0.11, следующие — без значимой разницы)")

    print("\n2. отдохнувший — 1.00 по определению; средняя сила на пуассоновских 1 Гц (Bruno & Sakmann: 0.82)")
    for kind in kinds:
        steps = int(round(20000.0 / dt))
        w = Synapse(kind, n // 4, dt).run(poisson(n // 4, steps, 1.0, dt, g))[:, steps // 4:]
        print(f"  {kind}: {float(w.sum() / (w > 0).sum()):.2f}")

    print("\n3. парные импульсы из покоя: 2-й / 1-й")
    gaps = (10, 20, 50, 100, 200, 500, 1000)
    for kind in kinds:
        out = []
        for gap in gaps:
            r = torch.zeros(1, int(round((gap + 2.0) / dt)), dtype=torch.float64)
            r[0, 0] = r[0, int(round(gap / dt))] = 1.0
            w = Synapse(kind, 1, dt).run(r)[0]
            out.append(float(w[int(round(gap / dt))]))
        print(f"  {kind}: " + " ".join(f"{gap} мс {v:.2f}" for gap, v in zip(gaps, out)))

    print("\n4. восстановление после 20 Гц × 10 (из покоя): тестовый спайк через, мс")
    tests = (50, 100, 250, 500, 750, 1000)
    for kind in kinds:
        out = []
        for gap in tests:
            r = train(1, 20, 10, dt, tail=gap + 2.0)
            last = int(round(9 * 50.0 / dt))
            r[0, last + int(round(gap / dt))] = 1.0
            w = Synapse(kind, 1, dt).run(r)[0]
            out.append(float(w[last + int(round(gap / dt))]))
        print(f"  {kind}: " + " ".join(f"{gap} {v:.2f}" for gap, v in zip(tests, out)))

    print("\n5. передача на пуассоновском фоне: средняя сила, частота × сила (ВПСП отдохнувшего в с)")
    for kind in kinds:
        row = {}
        for hz in (4.1, 8, 11.8, 21, 50, 100):
            steps = int(round(6000.0 / dt))
            w = Synapse(kind, n // 4, dt).run(poisson(n // 4, steps, hz, dt, g))[:, steps // 3:]
            mean = float(w.sum() / (w > 0).sum())
            row[hz] = (mean, hz * mean)
        print(f"  {kind}: " + " ".join(f"{hz} Гц {m:.2f}/{t:.2f}" for hz, (m, t) in row.items())
              + f"; 8 → 21 Гц ×{row[21][1] / row[8][1]:.2f}")


if __name__ == "__main__":
    main(dict(kv.split("=", 1) for kv in sys.argv[1:]))
