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

    PYTHONPATH=. python tools/block_dc.py [n_e=400] [epochs=3] [px=14] [side=20] [seeds=1,2,3] [device=cuda|cpu] [threads=2] [save=имя] [theta_plus=0.05] [norm=1] [w_ie=17] [tau_e=100] [rule=dc|ours]
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
    """
    B независимых сетей в одном тензоре (зёрна / варианты): веса [B, вход,
    E], состояние [B, клетки]. На GPU шаги показа записаны в CUDA graph
    (один запуск на STEPS_PER_GRAPH шагов вместо ~20 ядер на шаг); на CPU —
    тот же код без графа. Уравнения — как у авторов (см. начало файла).
    """

    STEPS_PER_GRAPH = 50

    def __init__(self, n_in: int, n_e: int, seeds: list[int], dev: str, compile: bool = True,
                 variant: dict | None = None):
        self.n_in, self.n_e, self.dev, self.B = n_in, n_e, dev, len(seeds)
        self.compile, self.fused = compile, None
        # Замены деталей блока на наши (шаг 2′ сборки), по умолчанию — авторы:
        #   theta_plus — прирост порога за спайк, мВ (0 — без адаптивного порога);
        #   norm — нормировка суммы весов клетки до 78 перед показом;
        #   w_ie — торможение I → прочие E (авторы 17 — «победитель забирает всё»);
        #   tau_e — τm клеток E, мс (авторы 100, у нас ~20);
        #   rule — "dc" (авторы) или "ours" (наше окно: τ+ 13.5, τ− 34.5, A+ 1%
        #     среднего веса, A− 0.5% · w, LTD от веса — config stdp_*).
        v = {"theta_plus": 0.05, "norm": True, "w_ie": 17.0, "tau_e": 100.0, "rule": "dc", **(variant or {})}
        self.variant = v
        self.theta_plus, self.tau_e, self.rule = float(v["theta_plus"]), float(v["tau_e"]), v["rule"]
        ws = []
        for sd in seeds:
            g = torch.Generator().manual_seed(sd)
            ws.append((torch.rand(n_in, n_e, generator=g) + 0.01) * 0.3)
        self.w = torch.stack(ws).to(dev)
        self.theta = torch.full((self.B, n_e), 20.0, device=dev)
        self.w_ie = torch.full((n_e, n_e), float(v["w_ie"]), device=dev).fill_diagonal_(0.0)
        self.w0 = 78.0 / n_in  # средний вес после нормировки
        shape_e, B = (self.B, n_e), self.B
        z = lambda *sh: torch.zeros(*sh, device=dev)
        self.v_e, self.v_i = torch.full(shape_e, -65.0, device=dev), torch.full(shape_e, -60.0, device=dev)
        self.ge_e, self.gi_e, self.ge_i, self.gi_i = z(*shape_e), z(*shape_e), z(*shape_e), z(*shape_e)
        self.ref_e, self.ref_i, self.spk_e, self.spk_i = z(*shape_e), z(*shape_e), z(*shape_e), z(*shape_e)
        self.pre, self.post1, self.post2 = z(B, n_in), z(*shape_e), z(*shape_e)
        self.p = z(B, n_in)          # вероятность спайка входа за шаг
        self.count = z(*shape_e)     # спайки E за показ
        self.learn = True
        self.graphs: dict[bool, torch.cuda.CUDAGraph] = {}
        d = lambda tau: float(torch.exp(torch.tensor(-DT / tau)))
        self.d_ge, self.d_gi = d(1.0), d(2.0)
        self.d_pre, self.d_p1, self.d_p2, self.d_theta = d(20.0), d(20.0), d(40.0), d(1e7)
        if self.rule == "ours":
            # След пресинапса τ+ (LTP на спайк E), след постсинапса τ− (LTD на спайк входа).
            self.d_pre, self.d_p1 = d(13.5), d(34.5)

    def normalize(self) -> None:
        if self.variant["norm"]:
            self.w.mul_(78.0 / self.w.sum(1, keepdim=True))

    def _step(self, learn: bool) -> None:
        x = (torch.rand(self.B, self.n_in, device=self.dev) < self.p).float()
        if learn:
            self.pre.mul_(self.d_pre)
            self.post1.mul_(self.d_p1)
            self.post2.mul_(self.d_p2)
            if self.rule == "dc":
                # спайк входа: w −= 0.0001 · post1, след pre ← 1
                self.w.sub_(0.0001 * x[:, :, None] * self.post1[:, None, :])
                self.pre.copy_(torch.where(x > 0, 1.0, self.pre))
            else:
                # спайк входа: w −= A− · w · след post (τ−), след pre += 1
                self.w.sub_(0.010 * 51.0 / 103.0 * self.w * x[:, :, None] * self.post1[:, None, :])
                self.pre.add_(x)
        self.ge_e.mul_(self.d_ge).add_(torch.bmm(x[:, None, :], self.w)[:, 0])
        self.gi_e.mul_(self.d_gi).add_(self.spk_i @ self.w_ie)
        self.ge_i.mul_(self.d_ge).add_(self.spk_e * 10.4)
        self.gi_i.mul_(self.d_gi)
        # Шаг мембраны — экспоненциальный: за шаг v точно релаксирует к
        # равновесию при текущих проводимостях. Простой Эйлер при
        # торможении 17 × (число выстреливших I) неустойчив — v
        # перескакивает и даёт ложные спайки (θ уходил в тысячи мВ).
        g = 1.0 + self.ge_e + self.gi_e
        v_inf = (-65.0 - 100.0 * self.gi_e) / g
        gi_ = 1.0 + self.ge_i + self.gi_i
        vi_inf = (-60.0 - 85.0 * self.gi_i) / gi_
        free_e, free_i = self.ref_e <= 0, self.ref_i <= 0
        v = torch.where(free_e, v_inf + (self.v_e - v_inf) * torch.exp(-DT * g / self.tau_e), self.v_e)
        vi = torch.where(free_i, vi_inf + (self.v_i - vi_inf) * torch.exp(-DT * gi_ / 10.0), self.v_i)
        spk_e = (v > self.theta - 72.0) & free_e
        spk_i = (vi > -40.0) & free_i
        self.v_e.copy_(torch.where(spk_e, -65.0, v))
        self.v_i.copy_(torch.where(spk_i, -45.0, vi))
        self.ref_e.copy_(torch.where(spk_e, 5.0, self.ref_e - DT))
        self.ref_i.copy_(torch.where(spk_i, 2.0, self.ref_i - DT))
        self.spk_e.copy_(spk_e.float())
        self.spk_i.copy_(spk_i.float())
        self.count.add_(self.spk_e)
        if learn:
            self.theta.mul_(self.d_theta).add_(self.theta_plus * self.spk_e)
            if self.rule == "dc":
                # спайк E: w += 0.01 · pre · post2 (до сброса), следы post ← 1
                self.w.add_(0.01 * self.pre[:, :, None] * (self.post2 * self.spk_e)[:, None, :])
                self.post1.copy_(torch.where(spk_e, 1.0, self.post1))
                self.post2.copy_(torch.where(spk_e, 1.0, self.post2))
            else:
                # спайк E: w += A+ · w0 · след pre (τ+), след post += 1
                self.w.add_(0.010 * self.w0 * self.pre[:, :, None] * self.spk_e[:, None, :])
                self.post1.add_(self.spk_e)
            self.w.clamp_(0.0, 1.0)

    def _steps(self, n: int) -> None:
        learn = self.learn
        if self.dev != "cuda":
            for _ in range(n):
                self._step(learn)
            return
        if self.fused is None:
            # Шаг одним куском (inductor): поэлементные действия над весами
            # (−LTD, +LTP, обрезка) — один проход вместо четырёх ядер.
            self.fused = torch.compile(self._step, fullgraph=True, dynamic=False) if self.compile else self._step
        if learn not in self.graphs:
            # Прогрев на побочном потоке и запись графа (состояние после
            # прогрева — настоящие шаги, не выбрасываются).
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    self.fused(learn)
            torch.cuda.current_stream().wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(self.STEPS_PER_GRAPH):
                    self.fused(learn)
            self.graphs[learn] = graph
            n -= 3
        k, rest = divmod(n, self.STEPS_PER_GRAPH)
        for _ in range(k):
            self.graphs[learn].replay()
        for _ in range(rest):
            self.fused(learn)

    def run(self, rates: torch.Tensor, ms: float) -> torch.Tensor:
        """Прожить ms при частотах входа rates [B, вход] (Гц); спайки E [B, E]."""

        self.p.copy_(rates.to(self.dev) * DT / 1000.0)
        self.count.zero_()
        self._steps(int(ms / DT))
        return self.count.clone()

    def present(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Показ как у авторов: 350 мс, пауза 150 мс; сеть с < 5 спайками E —
        ярче и заново (только она; прочие в это время получают пустой вход).
        imgs [B, вход]; вернуть спайки E [B, E] принятого показа.
        """

        intensity = torch.full((self.B, 1), 2.0)
        pending = torch.ones(self.B, dtype=torch.bool)
        out = torch.zeros(self.B, self.n_e)
        while pending.any():
            if self.learn:
                self.normalize()
            rates = imgs * 63.75 * intensity / 2.0 * pending[:, None]
            count = self.run(rates, SHOW_MS).cpu()
            self.run(torch.zeros_like(imgs), REST_MS)
            ok = pending & ((count.sum(1) >= 5) | (intensity[:, 0] > 20))
            out[ok] = count[ok]
            pending &= ~ok
            intensity[pending] += 1.0
        return out


def main(opts: dict) -> None:
    dev = opts.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    # На CPU — 2 потока: не толкается с прогоном мозга на GPU (замер одной
    # сети: 2 потока 0.27 с на показ, 4 — 0.37, 8 — 2.3).
    torch.set_num_threads(int(opts.get("threads", 2)))
    seeds = [int(x) for x in opts.get("seeds", opts.get("seed", "1")).split(",")]
    n_e, epochs = int(opts.get("n_e", 400)), int(opts.get("epochs", 3))
    px, side = int(opts.get("px", 14)), int(opts.get("side", 20))
    torch.manual_seed(seeds[0])
    shifts = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
    x_tr, y_tr = dataset(FONTS[:N_TRAIN_FONTS], shifts, px, side)
    x_new, y_new = dataset(FONTS[N_TRAIN_FONTS:], [(0, 0)], px, side)
    print(f"учёба {len(x_tr)} картинок ({N_TRAIN_FONTS} шрифтов × {len(shifts)} сдвигов × 33), "
          f"проверка на новых шрифтах {len(x_new)}; n_e {n_e}, эпох {epochs}, {side}×{side} px, "
          f"зёрна {seeds}, {dev}", flush=True)

    variant = {k: (v == "1") if k == "norm" else v if k == "rule" else float(v)
               for k, v in opts.items() if k in ("theta_plus", "norm", "w_ie", "tau_e", "rule")}
    if variant:
        print(f"вариант: {variant}", flush=True)
    net = Net(side * side, n_e, seeds, dev, variant=variant)
    gens = [torch.Generator().manual_seed(sd) for sd in seeds]
    t0 = time.time()
    for ep in range(epochs):
        orders = torch.stack([torch.randperm(len(x_tr), generator=g) for g in gens])
        recent = torch.zeros(len(seeds))
        for j in range(len(x_tr)):
            recent += net.present(x_tr[orders[:, j]]).sum(1)
            if (j + 1) % 100 == 0:
                th = net.theta.cpu() - 20
                print(f"  эпоха {ep + 1}, показов {j + 1}/{len(x_tr)}, {time.time() - t0:.0f} с; по зёрнам "
                      f"θ {', '.join(f'{float(t.mean()):.2f}' for t in th)} мВ, "
                      f"спайков E на показ {', '.join(f'{float(r) / 100:.1f}' for r in recent)}", flush=True)
                recent.zero_()

    net.learn = False

    def responses(x: torch.Tensor) -> torch.Tensor:
        """Ответы [B, показы, E]: каждая сеть видит те же картинки."""

        return torch.stack([net.present(img.expand(len(seeds), -1)) for img in x], 1)

    r_tr, r_new = responses(x_tr), responses(x_new)
    print()
    for b, sd in enumerate(seeds):
        mean_cls = torch.stack([r_tr[b][y_tr == k].mean(0) for k in range(len(LETTERS))])
        label = mean_cls.argmax(0)

        def accuracy(r: torch.Tensor, y: torch.Tensor) -> float:
            score = torch.stack([r[:, label == k].mean(1) if (label == k).any() else torch.full((len(r),), -1.0)
                                 for k in range(len(LETTERS))], 1)
            return float((score.argmax(1) == y).float().mean())

        print(f"зерно {sd}: букв покрыто клетками {len(set(label.tolist()))}/33, спайков на показ "
              f"{float(r_tr[b].sum(1).mean()):.1f}; точность учебные {accuracy(r_tr[b], y_tr):.3f}, "
              f"новые шрифты {accuracy(r_new[b], y_new):.3f}  (случайно {1 / 33:.3f})", flush=True)
    print(f"всего {time.time() - t0:.0f} с")
    if opts.get("save"):
        # Замороженный эталон: веса, θ, картинки и ответы — для сверки блока
        # в движке на тех же входах (шаг 2 сборки).
        from brain.config import BRAIN_HOME
        out = BRAIN_HOME / "fits" / f"block_dc-{opts['save']}.pt"
        torch.save({"seeds": seeds, "w": net.w.cpu(), "theta": net.theta.cpu(), "x_tr": x_tr, "y_tr": y_tr,
                    "x_new": x_new, "y_new": y_new, "r_tr": r_tr, "r_new": r_new, "opts": opts}, out)
        print(f"эталон → {out}")


if __name__ == "__main__":
    main(dict(kv.split("=", 1) for kv in sys.argv[1:]))
