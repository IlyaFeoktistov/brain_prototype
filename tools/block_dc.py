"""
Эталон блока «обучение без учителя» — Diehl & Cook 2015
(https://doi.org/10.3389/fncom.2015.00099), уравнения и числа из кода
авторов (github.com/peter-u-diehl/stdp-mnist, Diehl&Cook_spiking_MNIST.py),
на буквах кириллицы вместо цифр MNIST. Не мозг — отдельная сеть на torch:
проверяет, что сам блок учит наши буквы, до переноса в движок (шаг 1 из
DESIGN «Сборка из готовых блоков»).

Сеть (как у авторов):
  вход — пиксель картинки, пуассон, яркость чернил × 63.75 Гц × intensity/2;
  E (n_e) — проводимостный LIF: τm 100 мс, покой −65, сброс −65, порог −52
    + θ − 20 (θ стартует 20 мВ, +0.05 мВ за спайк, спад 10⁷ мс), рефр. 5 мс;
    ge τ 1 мс (E_rev 0), gi τ 2 мс (E_rev −100);
  I (n_i = n_e) — τm 10 мс, покой −60, сброс −45, порог −40, рефр. 2 мс,
    gi E_rev −85;
  вход → E все ко всем, вес U(0,1)·0.3 + 0.01 (у авторов задержки 0–10 мс —
    здесь без задержек); E → своя I вес 10.4; I → все прочие E вес 17;
  STDP вход → E: следы pre τ20, post1 τ20, post2 τ40 (ставятся в 1);
    на спайк входа w −= 0.0001·post1, на спайк E w += 0.01·pre·post2(до);
    w в [0, 1]; перед каждым показом сумма весов на клетку E = 78;
  показ 350 мс + пауза 150 мс; меньше 5 спайков E за показ — intensity + 1
    и показ заново.
Проверка (как у авторов): после учёбы веса и θ заморожены; клетке — класс
с наибольшим средним ответом на учебных показах; ответ сети — класс с
наибольшим средним ответом его клеток.

Буквы: 33 строчные, учёба — на train_fonts шрифтах со сдвигами ±1 px,
проверка — на других шрифтах (не виденных) и на учебных.

    PYTHONPATH=. python tools/block_dc.py [n_e=400] [epochs=3] [px=14] [side=20] [seed=1] [device=cpu] [threads=2]
"""

from __future__ import annotations

import sys
import time

import torch

LETTERS = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
FONTS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSerif.ttf",
    "/usr/share/fonts/truetype/freefont/FreeMono.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerifCondensed.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    # проверка — шрифты, которых не было на учёбе
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf",
]
N_TRAIN_FONTS = 10

DT = 0.5
SHOW_MS, REST_MS = 350.0, 150.0


def glyph(ch: str, font: str, px: int, side: int, dx: int, dy: int) -> torch.Tensor:
    """Чернила буквы 0..1 на поле side × side, буква по центру со сдвигом."""

    from PIL import Image, ImageDraw, ImageFont

    f = ImageFont.truetype(font, px)
    img = Image.new("L", (side, side), 0)
    d = ImageDraw.Draw(img)
    l, t, r, b_ = d.textbbox((0, 0), ch, font=f)
    d.text(((side - (r - l)) // 2 - l + dx, (side - (b_ - t)) // 2 - t + dy), ch, font=f, fill=255)
    return torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8).float().view(-1) / 255.0


def dataset(fonts: list[str], shifts: list[tuple[int, int]], px: int, side: int) -> tuple[torch.Tensor, torch.Tensor]:
    import os

    xs, ys = [], []
    for font in fonts:
        if not os.path.exists(font):
            continue
        for dx, dy in shifts:
            for k, ch in enumerate(LETTERS):
                xs.append(glyph(ch, font, px, side, dx, dy))
                ys.append(k)
    return torch.stack(xs), torch.tensor(ys)


class Net:
    def __init__(self, n_in: int, n_e: int, dev: str, gen: torch.Generator):
        self.n_in, self.n_e, self.dev = n_in, n_e, dev
        self.w = ((torch.rand(n_in, n_e, generator=gen) + 0.01) * 0.3).to(dev)
        self.theta = torch.full((n_e,), 20.0, device=dev)
        self.w_ie = torch.full((n_e, n_e), 17.0, device=dev).fill_diagonal_(0.0)
        self.learn = True

    def normalize(self) -> None:
        self.w *= 78.0 / self.w.sum(0, keepdim=True)

    def run(self, rates: torch.Tensor, ms: float) -> torch.Tensor:
        """Прожить ms при частотах входа rates (Гц); вернуть спайки E за время."""

        dev, n_e = self.dev, self.n_e
        steps = int(ms / DT)
        p = rates.to(dev) * DT / 1000.0
        count = torch.zeros(n_e, device=dev)
        s = self.state
        dec_ge_e, dec_gi_e = torch.exp(torch.tensor(-DT / 1.0)), torch.exp(torch.tensor(-DT / 2.0))
        dec_pre, dec_p1, dec_p2 = [float(torch.exp(torch.tensor(-DT / t))) for t in (20.0, 20.0, 40.0)]
        dec_theta = float(torch.exp(torch.tensor(-DT / 1e7)))
        for _ in range(steps):
            x = (torch.rand(self.n_in, device=dev) < p).float()
            # входные следы и депрессия на спайк входа
            if self.learn:
                s["pre"] *= dec_pre
                s["post1"] *= dec_p1
                s["post2"] *= dec_p2
                if x.any():
                    idx = x.bool()
                    self.w[idx] -= 0.0001 * s["post1"]
                    s["pre"][idx] = 1.0
            # проводимости
            s["ge_e"] = s["ge_e"] * dec_ge_e + x @ self.w
            s["gi_e"] = s["gi_e"] * dec_gi_e + s["spk_i"] @ self.w_ie
            s["ge_i"] = s["ge_i"] * dec_ge_e + s["spk_e"] * 10.4
            s["gi_i"] = s["gi_i"] * dec_gi_e
            # мембраны
            v, vi = s["v_e"], s["v_i"]
            dv = ((-65.0 - v) + s["ge_e"] * (-v) + s["gi_e"] * (-100.0 - v)) / 100.0
            dvi = ((-60.0 - vi) + s["ge_i"] * (-vi) + s["gi_i"] * (-85.0 - vi)) / 10.0
            free_e, free_i = s["ref_e"] <= 0, s["ref_i"] <= 0
            v = torch.where(free_e, v + DT * dv, v)
            vi = torch.where(free_i, vi + DT * dvi, vi)
            spk_e = (v > (self.theta - 20.0 - 52.0)) & free_e
            spk_i = (vi > -40.0) & free_i
            v = torch.where(spk_e, torch.full_like(v, -65.0), v)
            vi = torch.where(spk_i, torch.full_like(vi, -45.0), vi)
            s["ref_e"] = torch.where(spk_e, torch.full_like(v, 5.0), s["ref_e"] - DT)
            s["ref_i"] = torch.where(spk_i, torch.full_like(vi, 2.0), s["ref_i"] - DT)
            s["v_e"], s["v_i"] = v, vi
            s["spk_e"], s["spk_i"] = spk_e.float(), spk_i.float()
            count += s["spk_e"]
            if self.learn:
                self.theta = self.theta * dec_theta + 0.05 * s["spk_e"]
                if spk_e.any():
                    j = spk_e
                    self.w[:, j] += 0.01 * s["pre"][:, None] * s["post2"][j][None, :]
                    s["post1"][j] = 1.0
                    s["post2"][j] = 1.0
                self.w.clamp_(0.0, 1.0)
        return count

    def reset(self) -> None:
        z = lambda n: torch.zeros(n, device=self.dev)
        self.state = {
            "v_e": torch.full((self.n_e,), -65.0, device=self.dev), "v_i": torch.full((self.n_e,), -60.0, device=self.dev),
            "ge_e": z(self.n_e), "gi_e": z(self.n_e), "ge_i": z(self.n_e), "gi_i": z(self.n_e),
            "ref_e": z(self.n_e), "ref_i": z(self.n_e), "spk_e": z(self.n_e), "spk_i": z(self.n_e),
            "pre": z(self.n_in), "post1": z(self.n_e), "post2": z(self.n_e),
        }

    def present(self, img: torch.Tensor) -> torch.Tensor:
        """Показ как у авторов: 350 мс, пауза 150 мс; < 5 спайков — ярче и заново."""

        intensity = 2.0
        while True:
            if self.learn:
                self.normalize()
            count = self.run(img * 63.75 * intensity / 2.0, SHOW_MS)
            self.run(torch.zeros(self.n_in), REST_MS)
            if count.sum() >= 5 or intensity > 20:
                return count
            intensity += 1.0


def main(opts: dict) -> None:
    dev = opts.get("device", "cpu")
    # Сеть маленькая: на CPU быстрее, чем на GPU (запуск ядер на шаг), и
    # с 2 потоками быстрее, чем со всеми — не толкается с прогоном мозга на
    # GPU (замер: 2 потока 0.27 с на показ, 4 — 0.37, 8 — 2.3).
    torch.set_num_threads(int(opts.get("threads", 2)))
    n_e, epochs = int(opts.get("n_e", 400)), int(opts.get("epochs", 3))
    px, side = int(opts.get("px", 14)), int(opts.get("side", 20))
    gen = torch.Generator().manual_seed(int(opts.get("seed", 1)))
    torch.manual_seed(int(opts.get("seed", 1)))
    shifts = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
    x_tr, y_tr = dataset(FONTS[:N_TRAIN_FONTS], shifts, px, side)
    x_new, y_new = dataset(FONTS[N_TRAIN_FONTS:], [(0, 0)], px, side)
    print(f"учёба {len(x_tr)} картинок ({N_TRAIN_FONTS} шрифтов × {len(shifts)} сдвигов × 33), "
          f"проверка на новых шрифтах {len(x_new)}; n_e {n_e}, эпох {epochs}, {side}×{side} px", flush=True)

    net = Net(side * side, n_e, dev, gen)
    net.reset()
    t0 = time.time()
    for ep in range(epochs):
        for j, i in enumerate(torch.randperm(len(x_tr), generator=gen).tolist()):
            net.present(x_tr[i])
            if (j + 1) % 500 == 0:
                print(f"  эпоха {ep + 1}, показов {j + 1}/{len(x_tr)}, {time.time() - t0:.0f} с, "
                      f"θ {float(net.theta.mean() - 20):.2f} мВ", flush=True)

    net.learn = False

    def responses(x: torch.Tensor) -> torch.Tensor:
        return torch.stack([net.present(img) for img in x]).cpu()

    r_tr = responses(x_tr)
    mean_cls = torch.stack([r_tr[y_tr == k].mean(0) for k in range(len(LETTERS))])
    label = mean_cls.argmax(0)

    def accuracy(r: torch.Tensor, y: torch.Tensor) -> float:
        score = torch.stack([r[:, label == k].mean(1) if (label == k).any() else torch.full((len(r),), -1.0)
                             for k in range(len(LETTERS))], 1)
        return float((score.argmax(1) == y).float().mean())

    used = len(set(label.tolist()))
    print(f"\nклеток с классом: букв покрыто {used}/33; ответов на показ (учебные) {float(r_tr.sum(1).mean()):.1f} спайков")
    print(f"точность на учебных картинках: {accuracy(r_tr, y_tr):.3f}  (случайно {1 / 33:.3f})")
    print(f"точность на новых шрифтах:     {accuracy(responses(x_new), y_new):.3f}")
    print(f"всего {time.time() - t0:.0f} с")


if __name__ == "__main__":
    main(dict(kv.split("=", 1) for kv in sys.argv[1:]))
