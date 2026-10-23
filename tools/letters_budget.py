"""
Из чего складывается возбуждение клеток букв L4 E и какая часть несёт
различие фраз (DESIGN «Без ветки V1 → PV…»: различие слабеет от входа V1
к возбуждающему току 0.60 → 0.28).

Мозг как в reading_bench, без пластичности, веса и проводка не меняются.
A и B вперемешку по n показов чтением. У каждой клетки за показ, в мВ, как
в engine (до движущей силы): явные синапсы AMPA (i_syn) и их доля по путям
(Σ вес × спайки источника, у i_syn — та же доля), явные NMDA (i_nmda ×
блок Mg), среднее поле недостающих соседей AMPA (mf_exc × усталость ×
частоты — одно на всю популяцию) и NMDA (nmda_r × nmda_mf × блок), фон
(bias, сдвиг покоя). У каждой части: среднее, разность A−B (средняя по
клеткам |A−B|), её отношение к разбросу повторов, надёжность половин
(корреляция по клеткам A−B первой половины показов со второй; у среднего
поля клетки одинаковы — только A−B популяции к разбросу повторов).
Отдельно — главные источники среднего поля.

    PYTHONPATH=. python tools/letters_budget.py [n=20] [seed=1]
"""

from __future__ import annotations

import sys

import torch

from brain.engine import KINDS
from brain.session import Session
from tools.reading_bench import build

WIN = 10.0


def main(opts: dict) -> None:
    opts = {"fit": "lgn-a", "homeo": f"lgn-a-s{opts.get('seed', 1)}", "relay_v_t_mv": "-59",
            "relay_rest_mv": "-48", "trn_scale": "0.5", **opts}
    mind, b = build(opts)
    cfg = b.cfg
    s = Session(mind, window_ms=WIN)
    n_rep, fix, step = int(opts.get("n", 20)), float(opts.get("fix", 225)), float(opts.get("step", 5))
    texts = (opts.get("a", "мама мыла раму"), opts.get("b", "папа пила чай"))
    gen = torch.Generator().manual_seed(int(opts.get("seed", 1)))
    E = b.areas["letters"].layers["L4"]["E"]
    paths = [p for p in b.projections if p.post is E and p.source == "net" and float(p.weight.mean()) > 0]
    # Номер популяции-получателя и источники среднего поля (как в _build_mean_field).
    q = int(b.mf_post_id[E.start])
    pres = sorted({(p.start, p.size): p.name for _, p, *_ in b.mean_field}.items())
    pre_names = [name for _, name in pres]

    s.line, s.gaze = "", 0.0
    b.run(2500.0)

    def mean_field_e() -> torch.Tensor:
        """Возбуждение среднего поля популяции E по источникам (мВ), как в engine._step."""

        exc = b.mf_exc[q] - b.acetylcholine * b.mf_ach[q]
        nxt = b.stp_u4 + b.stp_U4 * (1.0 - b.stp_u4)
        gain = nxt * b.stp_x4 / b.stp_U4
        gain = torch.where(b.is_lamp5[:, None], gain * b.ngf_d[:, None], gain)
        gain = torch.where(b.typed_pre[:, None], gain, 1.0)
        n = len(KINDS)
        g = torch.zeros_like(b.mf_gain)
        g[:, n] = 1.0
        g[:, :n].index_add_(0, b.mf_pre_id, gain)
        g[:, :n].div_(b.mf_size[:, None])
        exc = exc * g[:, int(b.mf_post_type[q])]
        if b.mf_stp_on:
            resource = torch.zeros_like(b.mf_rate).index_add_(0, b.mf_pre_id, b.stp_x) / b.mf_size
            exc = torch.where(b.mf_stp[q], exc * resource, exc)
        return (exc * b.mf_rate)[:-1]

    def gazes(line: str) -> list[float]:
        width = s.screen.width(line) / max(1, len(line))
        n = max(1, int((len(line) + step - 1) // step))
        return [i * step * width for i in range(n)][:3]

    keys = ["явные AMPA", *[f"  {p.name}" for p in paths], "явные NMDA", "среднее поле AMPA",
            "среднее поле NMDA", "фон (bias)", "спайки, Гц"]

    def show(lab: int) -> dict[str, torch.Tensor]:
        s.line = ""
        b.run(500.0, s._blank(500.0))
        line = texts[lab] + " "
        acc = {k: 0 for k in keys}
        acc["_mf"] = 0
        n_win = 0
        for i, g in enumerate(gazes(line)):
            if i:
                s.line, s.gaze = line, g
                b.run(cfg.saccade_ms, s._blank(cfg.saccade_ms))
            s.line, s.gaze = line, g
            for _ in range(int(fix // WIN)):
                r = b.observe(WIN, s._view(WIN)) * (WIN / 1000.0)
                v = b.v[E.slice]
                block = 1.0 / (1.0 + cfg.mg_mm / 3.57 * torch.exp(-0.062 * v))
                acc["явные AMPA"] = acc["явные AMPA"] + b.i_syn[E.slice].float().cpu()
                drives = [(p.weight.float() * r[p.pre.long()]).sum(1) for p in paths]
                total = torch.stack(drives).sum(0).clamp_min(1e-9)
                for p, d in zip(paths, drives):
                    acc[f"  {p.name}"] = acc[f"  {p.name}"] + (b.i_syn[E.slice] * d / total).float().cpu()
                if b.nmda_on:
                    acc["явные NMDA"] = acc["явные NMDA"] + (b.i_nmda[E.slice] * block).float().cpu()
                    acc["среднее поле NMDA"] = acc["среднее поле NMDA"] + (
                        b.nmda_r[E.slice] * b.nmda_mf[E.slice] * block).float().cpu()
                mf = mean_field_e()
                acc["_mf"] = acc["_mf"] + mf.cpu()
                acc["среднее поле AMPA"] = acc["среднее поле AMPA"] + (float(mf.sum()) + b.dc_mv[E.slice]).float().cpu()
                acc["фон (bias)"] = acc["фон (bias)"] + b.bias[E.slice].float().cpu()
                acc["спайки, Гц"] = acc["спайки, Гц"] + r[E.slice].cpu() * (1000.0 / WIN)
                n_win += 1
        return {k: v / n_win if torch.is_tensor(v) else torch.zeros(E.size) for k, v in acc.items()}

    order = [0, 1] * n_rep
    order = [order[i] for i in torch.randperm(len(order), generator=gen)]
    feats = []
    for j, lab in enumerate(order):
        feats.append(show(lab))
        if (j + 1) % 10 == 0:
            print(f"  показов {j + 1}/{len(order)}", flush=True)
    y = torch.tensor(order)
    half = len(order) // 2

    def halves(x: torch.Tensor) -> float:
        d = [x[p][y[p] == 0].mean(0) - x[p][y[p] == 1].mean(0) for p in (slice(0, half), slice(half, None))]
        d = [u - u.mean() for u in d]
        return float((d[0] * d[1]).sum() / (d[0].norm() * d[1].norm()).clamp_min(1e-12))

    print(f"\n=== Возбуждение букв L4 E ({E.size} клеток), мВ до движущей силы; «{texts[0]}» против «{texts[1]}», "
          f"по {n_rep} показов")
    print(f"  {'часть':34s} {'среднее':>8s} {'доля':>6s} {'|A−B|':>7s} {'A−B/шум':>8s} {'надёжн.':>8s}")
    excit = [k for k in keys if not k.startswith("  ") and k not in ("спайки, Гц",)]
    whole = sum(float(torch.stack([f[k] for f in feats]).mean()) for k in excit if k != "фон (bias)")
    for k in keys:
        x = torch.stack([f[k] for f in feats])
        ma, mb = x[y == 0].mean(0), x[y == 1].mean(0)
        noise = torch.cat([x[y == 0] - ma, x[y == 1] - mb]).std(0)
        live = noise > 1e-9
        snr = float(((ma - mb).abs()[live] / noise[live]).mean()) if live.any() else 0.0
        share = float(x.mean()) / whole if k != "фон (bias)" and k != "спайки, Гц" else float("nan")
        same = bool((x.std(1) < 1e-6).all())
        rel = "общее" if same else f"{halves(x):+8.3f}"
        print(f"  {k:34s} {float(x.mean()):8.2f} {share:6.2f} {float((ma - mb).abs().mean()):7.3f} {snr:8.2f} {rel:>8s}",
              flush=True)
    mf = torch.stack([f["_mf"] for f in feats]).mean(0)
    top = mf.topk(min(6, len(mf))).indices
    print("  главные источники среднего поля AMPA: " + ", ".join(
        f"{pre_names[int(i)]} {float(mf[i]):.2f}" for i in top))
    print(f"  (явные синапсы и среднее поле — без фона; доля — от их суммы {whole:.2f} мВ)")


if __name__ == "__main__":
    main(dict(kv.split("=", 1) for kv in sys.argv[1:]))
