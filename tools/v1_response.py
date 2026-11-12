"""
Ответ V1 L2/3 E мозга на буквы — для сверки с записями V1 бодрствующей
макаки (DESIGN «Обучение пути V1 → буквы»): сильный ли и правдоподобный ли
сигнал V1 передаёт зоне букв. Считается по записи tools/v1_record.py (мозг
не гоняется).

Меры (фон — 150 мс паузы перед буквой, все показы; ответ — 350 мс буквы):
- фон: медиана и среднее частоты клеток, доля клеток без единого спайка
  за все паузы;
- лучшая буква клетки (наибольший средний счёт на показах меток — первые
  2/3 показов каждой буквы), её ответ на остальных показах (без отбора по
  тем же данным): частота и отношение к своему фону, по окнам 0–50, 50–150,
  150–350 мс; перцентили 50 / 90 / 99 по клеткам;
- отвечают вверх / вниз: хотя бы на одну букву |t| > 3 (парная разность
  буква − свой фон по показам этой буквы);
- Fano factor счёта за 350 мс на лучшую букву (клетки со средним ≥ 1
  спайка): по всем показам и внутри одного сдвига взгляда (±1 px — как
  движение глаз при фиксации, Gur, Beylin & Snodderly 1997,
  https://doi.org/10.1523/JNEUROSCI.17-08-02914.1997);
- различение 10 букв по окнам (корреляция с центром буквы; центры — те же
  2/3 показов, проверка — оставшаяся 1/3).

pop=LGN | V1 L4 | V1 L2/3 — популяция из записи v1_record chain=1 (по
умолчанию V1 L2/3); если в записи есть пустой экран — доля клеток без
единого спайка за него и медиана частоты.

    PYTHONPATH=. .venv/bin/python tools/v1_response.py rec=файл.pt [pop="V1 L2/3"]
"""

from __future__ import annotations

import sys

import torch


def main(opts: dict) -> None:
    torch.set_num_threads(1)
    rec = torch.load(opts["rec"])
    pop = opts.get("pop", "V1 L2/3")
    if pop != "V1 L2/3":
        rec["spikes"], rec["n_src"] = rec["extra"][pop], rec["sizes"][pop]
    n, L, dt = rec["n_src"], len(rec["letters"]), rec["dt_ms"]
    blank = rec.get("blank") or {}
    if pop in blank:
        c, sec = blank[pop].float(), blank["seconds"]
        q = torch.quantile(c / sec, torch.tensor([0.1, 0.5, 0.9]))
        print(f"{pop}: пустой экран {sec:.0f} с — без единого спайка {float((c == 0).float().mean()):.1%}, "
              f"частота 10/50/90% {float(q[0]):.2f} / {float(q[1]):.2f} / {float(q[2]):.2f} Гц")
    rest, show = rec["rest_ms"], rec["show_ms"]
    nr = int(round(rest / dt))
    y, shifts = rec["labels"], rec["shifts"]
    windows = {"0–50": (0, 50), "50–150": (50, 150), "150–350": (150, 350), "0–350": (0, 350)}

    def count(lo: float, hi: float) -> torch.Tensor:
        a, b = nr + int(lo / dt), nr + int(hi / dt)
        return torch.stack([torch.bincount(s[(s[:, 0] >= a) & (s[:, 0] < b)][:, 1].long(), minlength=n).float()
                            for s in rec["spikes"]])

    bg = torch.stack([torch.bincount(s[s[:, 0] < nr][:, 1].long(), minlength=n).float() for s in rec["spikes"]])
    bg_hz = bg.mean(0) / (rest / 1000)
    print(f"{pop} {n} клеток, {len(y)} показов; фон: медиана {float(bg_hz.median()):.2f} Гц, среднее "
          f"{float(bg_hz.mean()):.2f}, без спайков за все паузы ({len(y) * rest / 1000:.0f} с) "
          f"{float((bg.sum(0) == 0).float().mean()):.1%}")

    first, rest_idx = [], []
    for k in range(L):
        idx = (y == k).nonzero()[:, 0].tolist()
        first += idx[: len(idx) * 2 // 3]
        rest_idx += idx[len(idx) * 2 // 3:]
    first, rest_idx = torch.tensor(first), torch.tensor(rest_idx)
    full = count(0, show)
    best = torch.stack([full[first][y[first] == k].mean(0) for k in range(L)]).argmax(0)

    z = lambda a: (a - a.mean(-1, keepdim=True)) / a.std(-1, keepdim=True).clamp_min(1e-9)
    for name, (lo, hi) in windows.items():
        c = count(lo, hi)
        hz = c / ((hi - lo) / 1000)
        # Ответ на лучшую букву — на показах, не участвовавших в выборе буквы.
        on_best = torch.stack([hz[rest_idx][y[rest_idx] == int(best[i])][:, i].mean() for i in range(n)]) \
            if name == "0–350" else None
        sel = (y[rest_idx][:, None] == best[None, :]).float()
        on_best = (hz[rest_idx] * sel).sum(0) / sel.sum(0).clamp_min(1)
        ratio = (on_best + 0.1) / (bg_hz + 0.1)
        q = torch.quantile(on_best, torch.tensor([0.5, 0.9, 0.99]))
        qr = torch.quantile(ratio, torch.tensor([0.5, 0.9, 0.99]))
        centre = torch.stack([z(c[first][y[first] == k]).mean(0) for k in range(L)])
        acc = float(((z(c[rest_idx]) @ z(centre).T).argmax(1) == y[rest_idx]).float().mean())
        print(f"  {name:8s} мс: на лучшую букву 50/90/99% {float(q[0]):.1f} / {float(q[1]):.1f} / {float(q[2]):.1f} Гц, "
              f"к своему фону {float(qr[0]):.2f} / {float(qr[1]):.2f} / {float(qr[2]):.2f}; различение {acc:.2f}")

    # Отвечают вверх / вниз: |t| > 3 хотя бы на одну букву.
    d = full / (show / 1000) - bg / (rest / 1000)
    t = torch.stack([d[y == k].mean(0) / (d[y == k].std(0) / (y == k).sum() ** 0.5).clamp_min(1e-9) for k in range(L)])
    up, down = (t > 3).any(0), (t < -3).any(0)
    print(f"отвечают хотя бы на одну букву (|t| > 3): вверх {float(up.float().mean()):.1%}, вниз "
          f"{float(down.float().mean()):.1%}, и так и так {float((up & down).float().mean()):.1%}")

    # Fano factor на лучшую букву (на показах вне выбора).
    ff_all, ff_shift = [], []
    for i in range(n):
        m = rest_idx[y[rest_idx] == best[i]]
        x = full[m, i]
        if x.mean() < 1:
            continue
        ff_all.append(float(x.var() / x.mean()))
        parts = [full[m[shifts[m] == sh], i] for sh in (-1, 0, 1)]
        v = [float(p.var()) for p in parts if len(p) >= 3]
        mu = [float(p.mean()) for p in parts if len(p) >= 3]
        if v and sum(mu) > 0:
            ff_shift.append(sum(v) / sum(mu))
    f1, f2 = torch.tensor(ff_all), torch.tensor(ff_shift)
    print(f"Fano factor на лучшую букву (клеток со средним ≥ 1 спайка: {len(f1)}): медиана {float(f1.median()):.2f} "
          f"по всем показам, {float(f2.median()):.2f} внутри одного сдвига взгляда")


if __name__ == "__main__":
    main(dict(kv.split("=", 1) for kv in sys.argv[1:]))
