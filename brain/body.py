"""
Тело: органы мозга с общим разъёмом (ROADMAP чат-минимум 7.1).

Орган-датчик переводит сигнал мира в спайки своих рецепторов — свой
участок общего массива (engine.sensor_offsets); куда рецепторы входят в
кору, задаёт его SensorSpec. Новый орган — новый класс здесь и его
SensorSpec в BrainConfig.sensors().
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _gaussian(sigma: float) -> torch.Tensor:
    radius = max(1, int(round(3 * sigma)))
    x = torch.arange(-radius, radius + 1).float()
    g = torch.exp(-(x[:, None] ** 2 + x[None, :] ** 2) / (2 * sigma ** 2))
    return g / g.sum()


class Eye:
    """
    Глаз: картинка яркости → ганглиозные клетки сетчатки → растр спайков.

    Ганглиозная клетка сравнивает центр с окружением (Kuffler 1953): ON —
    центр светлее окружения, OFF — темнее. Ровный фон не даёт ничего,
    отвечают края и штрихи. Отклик — разность гауссиан (центр уже
    окружения); частота — доля от отклика на штрих толщиной в пиксель
    (тонкий тёмный штрих — самый сильный стимул OFF-клеток здесь).
    Спайки — пуассоновские, пока длится фиксация.
    """

    def __init__(self, cfg, offset: int, size: int, device):
        self.cfg = cfg
        self.offset = offset
        self.size = size
        self.device = device
        self.rows, self.cols = cfg.retina_rows, cfg.retina_cols
        self.center = _gaussian(cfg.retina_center_px)
        self.surround = _gaussian(cfg.retina_surround_px)
        line = torch.ones(15, 15)
        line[:, 7] = 0.0
        self.scale = float(self._contrast(line)[7, 7].neg()) or 1.0

    def _contrast(self, picture: torch.Tensor) -> torch.Tensor:
        """Центр минус окружение в каждой точке (фон за краем — как край)."""

        img = picture[None, None]

        def blur(kernel):
            pad = kernel.shape[-1] // 2
            return F.conv2d(F.pad(img, (pad, pad, pad, pad), mode="replicate"), kernel[None, None])

        return (blur(self.center) - blur(self.surround))[0, 0]

    def rates(self, picture: torch.Tensor) -> torch.Tensor:
        """Частота каждой ганглиозной клетки, Гц: слой ON, затем OFF."""

        contrast = self._contrast(picture.cpu()) / self.scale
        on = contrast.clamp(0.0, 1.0)
        off = (-contrast).clamp(0.0, 1.0)
        return torch.cat([on.flatten(), off.flatten()]) * self.cfg.retina_rate_hz

    def look(self, picture: torch.Tensor, ms: float | None = None) -> torch.Tensor:
        """Растр спайков фиксации [steps, offset + size]: участок глаза заполнен."""

        ms = self.cfg.fixation_ms if ms is None else ms
        steps = int(round(ms / self.cfg.dt_ms))
        p = (self.rates(picture) * self.cfg.dt_ms / 1000.0).to(self.device)
        raster = torch.zeros(steps, self.offset + self.size, device=self.device)
        raster[:, self.offset:] = (torch.rand(steps, self.size, device=self.device) < p).float()
        return raster


class Periphery(Eye):
    """
    Боковое зрение: та же сетчатка, но клетки реже и поля крупнее —
    картинка поля огрублена в scale раз (среднее по квадрату), дальше
    центр минус окружение, как у ямки (Curcio & Allen 1990).
    """

    def __init__(self, cfg, offset: int, size: int, device):
        super().__init__(cfg, offset, size, device)
        self.pool = cfg.periphery_scale
        self.rows = cfg.retina_rows // self.pool
        self.cols = cfg.retina_cols * cfg.periphery_fields // self.pool

    def rates(self, picture: torch.Tensor) -> torch.Tensor:
        coarse = F.avg_pool2d(picture[None, None].cpu(), self.pool)[0, 0]
        return super().rates(coarse)


class Hand:
    """
    Рука (орган-исполнитель): у каждой клавиши своё врождённое движение —
    свой набор клеток моторной коры руки подряд (соматотопия пальцев,
    Penfield; движение — популяцией, Georgopoulos 1986). Это тело, а не
    знание: какая буква на клавише, мозг выучивает глазом. Нажата клавиша,
    чей набор выстрелил сильнее всех и хотя бы наполовину.
    """

    def __init__(self, keys: int, cells: int):
        self.keys = keys
        self.cells = cells

    def code(self, key: int) -> torch.Tensor:
        return torch.arange(key * self.cells, (key + 1) * self.cells)

    def shares(self, fired: torch.Tensor) -> torch.Tensor:
        """Доля выстреливших клеток движения каждой клавиши [keys]."""

        return (fired[: self.keys * self.cells] > 0).float().view(self.keys, self.cells).mean(1)

    def pressed(self, fired: torch.Tensor, share: float = 0.5) -> int | None:
        """fired — спайки клеток руки за окно; номер нажатой клавиши или None."""

        blocks = self.shares(fired)
        key = int(blocks.argmax())
        return key if float(blocks[key]) >= share else None


# Волокна слухового нерва в полосе (Liberman 1978, кошка: 61 / 23 / 16%
# по спонтанной частоте): (доля, спонтанная Гц, порог над порогом полосы дБ,
# динамический диапазон дБ). Высокая SR — самые чувствительные, узкий
# диапазон ~20–25 дБ; низкая — порог на ~20–30 дБ выше, диапазон 40–60 дБ
# (Sachs & Abbas 1974; Winter, Robertson & Yates 1990) [сверить по человеку].
FIBERS = ((0.61, 60.0, 0.0, 25.0), (0.23, 8.0, 10.0, 35.0), (0.16, 0.2, 25.0, 50.0))
# Порог слышимости человека по частоте, дБ УЗД (ISO 226:2003, чистый тон,
# поле): ниже 500 Гц и выше 6 кГц слух хуже — порог полосы выше.
HEARING_THRESHOLD = ((100, 26.0), (250, 12.0), (500, 4.5), (1000, 2.4), (2000, -1.3), (4000, -5.4), (8000, 12.6))


class Cochlea:
    """
    Ухо: звуковая волна → улитка → слуховой нерв → растр спайков. Один вход
    для всего, что слышно: файл записи, синтезатор речи, микрофон — мозг
    различает их не больше, чем человек.

    Улитка — полосы гамматон-фильтров 4-го порядка, середины по шкале ERB
    (Glasberg & Moore 1990: ERB = 24.7 · (4.37 f/кГц + 1), ширина фильтра
    1.019 ERB), cochlea_low_hz…cochlea_high_hz. Громкость полосы — мощность
    выхода, сглаженная cochlea_smooth_ms, в дБ УЗД. Запись приводится к
    уровню обычной речи cochlea_speech_db (65 дБ, Pearsons et al. 1977),
    так что громкость микрофона и файлов одна.

    В каждой полосе волокна трёх классов (FIBERS) — частота растёт с
    уровнем над порогом класса до cochlea_max_hz за его диапазон. Начало
    звука — сильнее (кратковременная адаптация: ответ на начало в 2–3 раза
    выше установившегося, спад ~40 мс — Westerman & Smith 1984).
    Спонтанные спайки даёт разъём органа (spontaneous_hz), здесь — только
    прибавка к ним.
    """

    def __init__(self, cfg, offset: int, size: int, device):
        self.cfg = cfg
        self.offset = offset
        self.size = size
        self.device = device
        self.bands = cfg.cochlea_bands
        per = size // self.bands
        counts = [round(per * share) for share, *_ in FIBERS]
        counts[0] += per - sum(counts)
        # Класс каждого волокна полосы (одинаково во всех полосах).
        self.kind = torch.cat([torch.full((n,), i) for i, n in enumerate(counts)])
        erb = lambda f: 21.4 * torch.log10(4.37 * f / 1000.0 + 1.0)
        inv = lambda e: (10 ** (e / 21.4) - 1.0) * 1000.0 / 4.37
        low, high = erb(torch.tensor(cfg.cochlea_low_hz)), erb(torch.tensor(cfg.cochlea_high_hz))
        self.centres = inv(torch.linspace(float(low), float(high), self.bands))
        f, db = (torch.tensor(v, dtype=torch.float32) for v in zip(*HEARING_THRESHOLD))
        x = torch.log(self.centres).clamp(float(torch.log(f[0])), float(torch.log(f[-1])))
        i = torch.searchsorted(torch.log(f), x).clamp(1, len(f) - 1)
        w = (x - torch.log(f[i - 1])) / (torch.log(f[i]) - torch.log(f[i - 1]))
        self.threshold = db[i - 1] + w * (db[i] - db[i - 1])
        self._kernels: dict[int, torch.Tensor] = {}

    @staticmethod
    def spontaneous(size: int, bands: int) -> torch.Tensor:
        """Спонтанная частота каждого волокна, Гц (для разъёма органа)."""

        per = size // bands
        counts = [round(per * share) for share, *_ in FIBERS]
        counts[0] += per - sum(counts)
        band = torch.cat([torch.full((n,), FIBERS[i][1]) for i, n in enumerate(counts)])
        out = torch.zeros(size)
        out[: per * bands] = band.repeat(bands)
        return out

    def _bank(self, rate: int) -> torch.Tensor:
        """Импульсные отклики гамматон-фильтров [полосы, 1, длина] при частоте записи rate."""

        if rate not in self._kernels:
            t = torch.arange(int(0.05 * rate)) / rate
            f = self.centres[:, None]
            b = 1.019 * 24.7 * (4.37 * f / 1000.0 + 1.0)
            h = t ** 3 * torch.exp(-2 * torch.pi * b * t) * torch.cos(2 * torch.pi * f * t)
            # Единичное усиление на своей частоте.
            gain = (h * torch.exp(-2j * torch.pi * f * t)).sum(1, keepdim=True).abs()
            # conv1d — корреляция: отклик разворачивается, чтобы свёртка была
            # причинной (иначе звук приходил на ~длину ядра позже — 30–48 мс;
            # нашла сверка с латентностью IC человека, tools/ic_latency.py).
            self._kernels[rate] = (h / gain).flip(-1)[:, None, :].float()

        return self._kernels[rate]

    def levels(self, wave: torch.Tensor, rate: int) -> torch.Tensor:
        """Уровень каждой полосы, дБ УЗД, по шагам мозга [steps, полосы]."""

        cfg = self.cfg
        wave = wave.float().flatten()
        rms = wave.pow(2).mean().sqrt().clamp(min=1e-9)
        # Па: запись — на уровне обычной речи.
        wave = wave / rms * 20e-6 * 10 ** (cfg.cochlea_speech_db / 20)
        kernels = self._bank(rate)
        out = F.conv1d(F.pad(wave[None, None], (kernels.shape[-1] - 1, 0)), kernels)[0]
        power = out.pow(2)
        # Сглаживание мощности (интеграция волосковой клетки) и шаг мозга.
        tau = cfg.cochlea_smooth_ms * rate / 1000.0
        width = max(1, int(round(3 * tau)))
        decay = torch.exp(-torch.arange(width) / tau)
        power = F.conv1d(F.pad(power[:, None], (width - 1, 0)), (decay / decay.sum()).flip(0)[None, None])[:, 0]
        step = cfg.dt_ms * rate / 1000.0
        index = (torch.arange(int(power.shape[1] / step)) * step).long()
        return (10 * torch.log10(power[:, index] / (20e-6) ** 2 + 1e-12)).t()

    def rates(self, levels: torch.Tensor) -> torch.Tensor:
        """Прибавка частоты каждого волокна к спонтанной, Гц [steps, size]."""

        cfg = self.cfg
        spont, threshold, span = (torch.tensor([f[i] for f in FIBERS]) for i in (1, 2, 3))
        k = self.kind
        over = levels[:, :, None] - self.threshold[None, :, None] - threshold[k][None, None]
        drive = (over / span[k][None, None]).clamp(0.0, 1.0)
        steady = drive * (cfg.cochlea_max_hz - spont[k][None, None])
        # Кратковременная адаптация: начало звука — выше установившегося.
        alpha = cfg.dt_ms / cfg.cochlea_adapt_ms
        slow = torch.zeros_like(steady[0])
        out = torch.empty_like(steady)

        for i in range(steady.shape[0]):
            slow = slow + alpha * (steady[i] - slow)
            out[i] = steady[i] + cfg.cochlea_onset * (steady[i] - slow).clamp(min=0.0)

        return out.reshape(steady.shape[0], -1)

    def hear(self, wave: torch.Tensor, rate: int) -> torch.Tensor:
        """Растр спайков звука [steps, offset + size]: участок уха заполнен."""

        hz = self.rates(self.levels(wave, rate))
        p = (hz * self.cfg.dt_ms / 1000.0).clamp(max=1.0).to(self.device)
        raster = torch.zeros(p.shape[0], self.offset + self.size, device=self.device)
        raster[:, self.offset: self.offset + p.shape[1]] = (torch.rand_like(p) < p).float()
        return raster
