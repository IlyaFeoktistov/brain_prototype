"""
Передача отдельных спайков по парам на двух ступенях зрения, как в
кросскорреляционных опытах (кошка, in vivo):

    сетчатка → LGN   рецептор i и клетка реле i (водитель; Usrey, Reppas &
                     Reid 1999)
    LGN → V1 L4 E    клетка реле и простая клетка, которой она — вход
                     (Reid & Alonso 1995; Alonso, Usrey & Reid 2001)

efficacy — доля спайков пресинапса, за которыми через 0.5–4.5 мс следует
спайк получателя, минус случайные совпадения (частота получателя × окно);
contribution — доля спайков получателя, которым предшествовал спайк этого
входа, минус случайные. Ещё: задержка пика, ВПСП в L4 после одиночного
спайка LGN (среднее V по спайкам минус V до спайка; с неудачами — как
0.49 ± 0.23 мВ у Sedigh-Sarvestani et al. 2019) и синхронность соседних
клеток LGN (совпадения за ±1 мс сверх случайных; Alonso, Usrey & Reid 1996).

Мозг считается по шагу (кусок в 1 шаг, 0.5 мс); спонтанные спайки
сетчатки (20 Гц) — во входном растре стенда, чтобы знать их время.

    PYTHONPATH=. python tools/relay_pairs.py [seconds=20] [seed=42]
"""

from __future__ import annotations

import sys

import torch

from brain.config import BrainConfig
from brain.mind import Mind
from brain.session import Session


def main(seconds: float, seed: int, text: str) -> None:
    cfg = BrainConfig()
    cfg.size = "small"
    cfg.seed = seed
    cfg.autosave = False
    mind = Mind(cfg)
    b = mind.brain
    b.plasticity = False
    b.run(2500.0)
    b.CHUNK, b._graphs = 1, {}
    s = Session(mind)
    s.line, s.gaze = text + " ", 0.0
    eye = [x.name for x in b.sensors].index("eye")
    offset, size = b.sensor_offsets[eye], b.sensors[eye].size
    spont = float(b.spontaneous[offset]) if b.spontaneous is not None else 0.0
    b.spontaneous[offset: offset + size] = 0.0
    lgn, l4 = b.areas["lgn"].E, b.areas["visual"].layers["L4"]["E"]
    proj = next(p for p in b.projections if p.name == f"lgn -> {l4.name}")
    l4_pre = (proj.pre.long() - lgn.start).cpu()            # [клетки L4, входы] — номер клетки реле
    watched = torch.arange(0, l4.size, max(1, l4.size // 300))  # клетки L4 для ВПСП
    dt = cfg.dt_ms
    steps = int(seconds * 1000 / dt)
    rec, rel, cor, volt = [], [], [], []

    for t in range(steps):
        view = s._view(dt)
        retina = view[:, offset: offset + size]
        retina.copy_(((retina + (torch.rand_like(retina) < spont).float()) > 0).float())
        b.run(dt, view)
        rec.append(retina[0].nonzero().squeeze(1).cpu())
        rel.append(b.spikes[lgn.slice].nonzero().squeeze(1).cpu())
        cor.append(b.spikes[l4.slice].nonzero().squeeze(1).cpu())
        volt.append(b.v[l4.slice][watched.to(b.device)].cpu())

    def raster(events, n) -> torch.Tensor:
        r = torch.zeros(steps, n, dtype=torch.bool)
        for t, idx in enumerate(events):
            r[t, idx] = True
        return r

    R, L, C = raster(rec, size), raster(rel, lgn.size), raster(cor, l4.size)
    V = torch.stack(volt)
    lo, hi = int(round(0.5 / dt)), int(round(4.5 / dt))

    def follow(pre: torch.Tensor, post: torch.Tensor) -> torch.Tensor:
        """Было ли у post хоть одно событие через lo..hi шагов после каждого шага."""

        window = torch.zeros_like(post)
        for lag in range(lo, hi + 1):
            window[:-lag] |= post[lag:]
        return window

    def stats(pre: torch.Tensor, post: torch.Tensor):
        """По парам: (спайков пре, спайков пост, efficacy, contribution, коррелограмма 0–10 мс)."""

        n_pre, n_post = pre.sum(0).float(), post.sum(0).float()
        eff = (pre & follow(pre, post)).sum(0).float() / n_pre.clamp(min=1) - (n_post / steps) * (hi - lo + 1)
        back = torch.zeros_like(pre)
        for lag in range(lo, hi + 1):
            back[lag:] |= pre[:-lag]
        con = (post & back).sum(0).float() / n_post.clamp(min=1) - (n_pre / steps) * (hi - lo + 1)
        lags = torch.tensor([float((pre[:-k] & post[k:]).sum()) if k else float((pre & post).sum()) for k in range(int(10 / dt))])
        return n_pre, n_post, eff, con, lags

    def pairs(pre_cols: torch.Tensor, pre_r: torch.Tensor, post_cols: torch.Tensor, post_r: torch.Tensor, label: str, ref: str) -> None:
        # Пачками по 256 пар: растр [шаги × пары] целиком не помещается.
        parts = [stats(pre_r[:, pre_cols[i:i + 256]], post_r[:, post_cols[i:i + 256]]) for i in range(0, len(pre_cols), 256)]
        n_pre, n_post, eff, con = (torch.cat([x[k] for x in parts]) for k in range(4))
        lags = sum(x[4] for x in parts)
        ok = (n_pre >= 20) & (n_post >= 5)
        peak = int(lags.argmax()) * dt
        q = lambda x: f"медиана {x[ok].median():.3f}, среднее {x[ok].mean():.3f}, макс {x[ok].max():.3f}" if ok.any() else "—"
        print(f"\n{label}: пар {int(ok.sum())} (из {len(pre_cols)}), пресинапс {float(n_pre[ok].mean() / seconds) if ok.any() else 0:.1f} Гц, получатель {float(n_post[ok].mean() / seconds) if ok.any() else 0:.1f} Гц")
        print(f"   efficacy:     {q(eff)}")
        print(f"   contribution: {q(con)}")
        print(f"   пик кросскоррелограммы: {peak:.1f} мс   [{ref}]")

    every = torch.arange(size)
    pairs(every, R, every, L, "сетчатка → LGN (водитель)", "кошка: связь — при перекрытии центров ≥ 50%, окно 2–5 мс; Usrey et al. 1999")
    cols = l4_pre[watched].flatten()
    post_idx = watched.repeat_interleave(l4_pre.shape[1])
    pairs(cols, L, post_idx, C, "LGN → V1 L4 E", "кошка: contribution 3.1% (макс 10.3%), задержка ВПСП 2.6 мс; Reid & Alonso 1995, Sedigh-Sarvestani 2019")

    # ВПСП после одиночного спайка LGN: V клетки L4 через 0–10 мс минус V за
    # 1 мс до спайка; минус то же по случайным моментам (общая корреляция
    # входа, не этот синапс). Спайки клетки L4 обнуляют V — окно с её спайком
    # не берётся.
    span = int(10 / dt)
    trace, ctrl, n = torch.zeros(span), torch.zeros(span), 0
    gen = torch.Generator().manual_seed(0)
    for w, cell in enumerate(watched.tolist()):
        own = C[:, cell]
        for j in l4_pre[cell].tolist():
            times = L[:, j].nonzero().squeeze(1)
            times = times[(times > 4) & (times < steps - span)]
            fake = torch.randint(5, steps - span, (len(times),), generator=gen)
            for t0, f0 in zip(times.tolist(), fake.tolist()):
                if own[t0 - 2: t0 + span].any() or own[f0 - 2: f0 + span].any():
                    continue
                trace += V[t0: t0 + span, w] - V[t0 - 2: t0, w].mean()
                ctrl += V[f0: f0 + span, w] - V[f0 - 2: f0, w].mean()
                n += 1
    trace = (trace - ctrl) / max(1, n)
    k = int(trace.argmax())
    print(f"\nВПСП в L4 после спайка LGN (вместе с неудачами, минус случайные моменты): пик {trace[k]:.2f} мВ через {k * dt:.1f} мс, спайков {n}   [кошка: 0.49 ± 0.23 мВ, задержка 2.6 мс]")

    # Синхронность соседних клеток реле (тот же слой ON/OFF, соседний пиксель).
    cols_n = cfg.retina_cols
    a = torch.arange(size - 1)
    a = a[(a % cols_n) != cols_n - 1]
    both = (L[:, a] & L[:, a + 1]).sum(0).float()
    wide = sum((L[:-k, a] & L[k:, a + 1]).sum(0).float() + (L[k:, a] & L[:-k, a + 1]).sum(0).float() for k in (1, 2))
    rate = L[:, a].sum(0).float() * L[:, a + 1].sum(0).float() / steps
    excess = (both + wide - 5 * rate).sum() / (5 * rate).sum().clamp(min=1)
    print(f"синхронность соседних клеток LGN (±1 мс): совпадений больше случайных на {float(excess):+.0%}   [кошка: часто синхронны до 1 мс]")


if __name__ == "__main__":
    opts = dict(kv.split("=", 1) for kv in sys.argv[1:])
    main(float(opts.get("seconds", 20)), int(opts.get("seed", 42)), opts.get("text", "мама мыла раму"))
