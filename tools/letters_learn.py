"""
Учится ли зона букв различать увиденное (звено 1 ROADMAP, DESIGN «Зона букв:
V1 — водитель, различие в токе, не в спайках»).

Мозг как в reading_bench (LGN A, фон, гомеостаз зерна). Три части:
    до        проверка: A и B вперемешку, n повторов, чтение, без пластичности;
    учёба     train показов каждой фразы вперемешку, тем же чтением —
              пластичность мозга включена (plastic=0 — контроль: то же время
              без пластичности);
    после     проверка на новых показах, как «до».
Учёба и проверка — разные показы; классификатор (ближайший центр по
корреляции, выброс одного повтора — l4_decode) видит только показы своей
проверки. Признаки — спайки клеток за каждую фиксацию (V1 L2/3 — для
сравнения, буквы L4 и L2/3) и ток возбуждения букв L4. Случайно —
перемешанные метки. Изменение пути V1 → буквы: средний |Δw| в долях
среднего веса.

Полезны ли новые веса — без шума спайков букв: вход пути V1 → буквы
(Σ вес × спайки V1 L2/3 за фиксацию) на одной и той же записи спайков V1
проверки «после» — с весами при рождении и с выученными: разность A−B
у клетки в долях шума повторов и точность по этому входу.

rule=clopath — правило по напряжению (cfg.clopath_paths, Clopath et al.
2010) на пути V1 → буквы вместо STDP; по ходу учёбы — суммы LTP и LTD
правила, перцентили Δw (в долях среднего веса), V и частота букв L4.

    PYTHONPATH=. python tools/letters_learn.py [train=60] [n=12] [plastic=1] [rule=stdp|clopath] [seed=1] [fix=225] [step=5]
"""

from __future__ import annotations

import sys
import time

import torch

from brain.session import Session
from tools.letters_drive import loo
from tools.reading_bench import build

WIN = 10.0


def main(opts: dict) -> None:
    opts = {"fit": "lgn-a", "homeo": f"lgn-a-s{opts.get('seed', 1)}", "relay_v_t_mv": "-59",
            "relay_rest_mv": "-48", "trn_scale": "0.5", **opts}
    if opts.get("rule") == "clopath":
        opts["clopath"] = "visual -> letters"
    mind, b = build(opts)
    cfg = b.cfg
    s = Session(mind, window_ms=WIN)
    n_test, n_train = int(opts.get("n", 12)), int(opts.get("train", 60))
    fix, step = float(opts.get("fix", 225)), float(opts.get("step", 5))
    texts = (opts.get("a", "мама мыла раму"), opts.get("b", "папа пила чай"))
    gen = torch.Generator().manual_seed(int(opts.get("seed", 1)))
    vis, let = b.areas["visual"].layers, b.areas["letters"].layers
    stages = {"V1 L2/3": vis["L23"]["E"], "буквы L4": let["L4"]["E"], "буквы L2/3": let["L23"]["E"]}
    path = next(p for p in b.projections if p.name == "visual -> letters")
    w0 = path.weight.float().clone()
    v1 = vis["L23"]["E"]
    pre_v1 = path.pre.long() - v1.start
    assert int(pre_v1.min()) >= 0 and int(pre_v1.max()) < v1.size, "вход пути — не V1 L2/3 E"
    train_v = []

    s.line, s.gaze = "", 0.0
    b.run(2500.0)

    def gazes(line: str) -> list[float]:
        width = s.screen.width(line) / max(1, len(line))
        n = max(1, int((len(line) + step - 1) // step))
        return [i * step * width for i in range(n)][:3]

    def show(lab: int) -> dict[str, torch.Tensor]:
        """Один показ фразы чтением; признаки по фиксациям."""

        s.line = ""
        b.run(500.0, s._blank(500.0))
        line = texts[lab] + " "
        rows = {k: [] for k in [*stages, "ток букв L4"]}
        for i, g in enumerate(gazes(line)):
            if i:
                s.line, s.gaze = line, g
                b.run(cfg.saccade_ms, s._blank(cfg.saccade_ms))
            s.line, s.gaze = line, g
            acc = {k: 0 for k in rows}
            n_win = int(fix // WIN)
            for _ in range(n_win):
                r = b.observe(WIN, s._view(WIN)) * (WIN / 1000.0)
                for k, pop in stages.items():
                    acc[k] = acc[k] + r[pop.slice].cpu()
                acc["ток букв L4"] = acc["ток букв L4"] + b.i_syn[stages["буквы L4"].slice].float().cpu() / n_win
                if b.plasticity:
                    train_v.append(float(b.v[stages["буквы L4"].slice].mean()))
            for k in rows:
                rows[k].append(acc[k])
        return {k: torch.cat(v) for k, v in rows.items()}

    def shuffled(n: int) -> list[int]:
        order = [0, 1] * n
        return [order[i] for i in torch.randperm(len(order), generator=gen)]

    def test(name: str):
        b.plasticity = False
        order = shuffled(n_test)
        feats = [show(lab) for lab in order]
        y = torch.tensor(order)
        dw = float((path.weight.float() - w0).abs().mean() / w0.abs().mean())
        print(f"\n=== {name}: по {n_test} показов, |Δw| V1 → буквы {dw:.4f} среднего веса")
        print(f"  {'ступень':12s} {'сп/кл/фикс':>10s} {'точность':>9s} {'случайно':>9s}")
        for k in feats[0]:
            x = torch.stack([f[k] for f in feats])
            ch = float(torch.tensor([loo(x, y[torch.randperm(len(y), generator=gen)]) for _ in range(20)]).mean())
            print(f"  {k:12s} {float(x.mean()):10.3f} {loo(x, y):9.2f} {ch:9.2f}", flush=True)
        # Вход пути V1 → буквы на этой записи V1: веса при рождении и нынешние.
        counts = torch.stack([f["V1 L2/3"] for f in feats]).view(len(feats), -1, v1.size)
        pre = pre_v1.cpu()
        for name_w, w in (("вход V1, веса рождения", w0), ("вход V1, веса сейчас", path.weight.float())):
            w = w.cpu()
            drive = torch.stack([(w[None] * counts[:, f][:, pre]).sum(2) for f in range(counts.shape[1])], 1)
            x = drive.reshape(len(feats), -1)
            ma, mb = x[y == 0].mean(0), x[y == 1].mean(0)
            noise = torch.cat([x[y == 0] - ma, x[y == 1] - mb]).std(0).mean()
            snr = float((ma - mb).abs().mean() / noise.clamp_min(1e-9))
            print(f"  {name_w:22s} A−B / шум {snr:5.2f}, точность {loo(x, y):.2f}", flush=True)
        return feats, y

    test("до учёбы")
    v1_seen = torch.zeros(v1.size)
    plastic = opts.get("plastic", "1") == "1"
    t0 = time.time()
    for i, lab in enumerate(shuffled(n_train)):
        b.plasticity = plastic
        v1_seen += show(lab)["V1 L2/3"].view(-1, v1.size).sum(0)
        if (i + 1) % 20 == 0:
            d = (path.weight.float() - w0) / w0.mean()
            dw = float(d.abs().mean())
            q = torch.quantile(d.flatten()[torch.randperm(d.numel(), device=d.device)[:200000]],
                               torch.tensor([0.01, 0.25, 0.5, 0.75, 0.99], device=d.device))
            extra = ""
            if getattr(b, "cl_ltp", None) is not None and b.clopath:
                extra = (f", LTP {float(b.cl_ltp) / float(w0.sum()):.4f} LTD {float(b.cl_ltd) / float(w0.sum()):.4f}"
                         f" (доли суммы весов)")
                b.cl_ltp.zero_()
                b.cl_ltd.zero_()
            print(f"  учёба {i + 1}/{2 * n_train} показов, |Δw| {dw:.4f}{extra}; Δw перц. 1/25/50/75/99: "
                  + " ".join(f"{float(v):+.3f}" for v in q)
                  + f"; V букв L4 {sum(train_v) / max(1, len(train_v)):.1f} мВ, {time.time() - t0:.0f} с", flush=True)
            train_v.clear()
    feats, y = test(f"после учёбы ({2 * n_train} показов, пластичность {'вкл' if plastic else 'выкл — контроль'})")
    if not plastic:
        return
    # Зависят ли изменения весов от входов: у клетки — корреляция
    # относительного Δw синапса с частотой его входа V1 при учёбе и с
    # избирательностью входа к «своей» фразе клетки (знак A−B ответа клетки
    # на проверке × A−B входа); контроль — знак клетки перемешан.
    rel = ((path.weight.float() - w0) / w0.mean()).cpu()
    pre = pre_v1.cpu()

    def row_corr(a: torch.Tensor, c: torch.Tensor) -> float:
        a = a - a.mean(1, keepdim=True)
        c = c - c.mean(1, keepdim=True)
        r = (a * c).sum(1) / (a.norm(dim=1) * c.norm(dim=1)).clamp_min(1e-12)
        return float(r.mean())

    v1_counts = torch.stack([f["V1 L2/3"] for f in feats]).view(len(feats), -1, v1.size).sum(1)
    d = v1_counts[y == 0].mean(0) - v1_counts[y == 1].mean(0)
    l4_counts = torch.stack([f["буквы L4"] for f in feats]).view(len(feats), 3, -1).sum(1)
    pref = torch.sign(l4_counts[y == 0].mean(0) - l4_counts[y == 1].mean(0))
    shuffled_pref = pref[torch.randperm(len(pref), generator=gen)]
    print(f"\n=== Δw по входам (в долях среднего веса): среднее {float(rel.mean()):+.4f}, "
          f"разброс внутри клетки {float(rel.std(1).mean()):.4f}")
    print(f"  корр. Δw с частотой входа при учёбе      {row_corr(rel, v1_seen[pre]):+.4f}")
    print(f"  корр. Δw с избирательностью к своей фразе {row_corr(rel, pref[:, None] * d[pre]):+.4f} "
          f"(контроль {row_corr(rel, shuffled_pref[:, None] * d[pre]):+.4f}; "
          f"клеток с предпочтением {int((pref != 0).sum())} из {len(pref)})", flush=True)


if __name__ == "__main__":
    main(dict(kv.split("=", 1) for kv in sys.argv[1:]))
