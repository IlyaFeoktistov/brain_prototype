"""
Разговор анатомического мозга с миром (ARCHITECTURE 11): экран, глаз,
рука, клавиатура — как переписка в мессенджере. Здесь только тело и мир:
что видит сетчатка, куда повернулся глаз, какая клавиша нажата. Что
прочитать, куда посмотреть, что напечатать — решает мозг.

Глаз: на экране вся строка, сетчатка видит окно вокруг точки взгляда
(ямка — cfg.fovea_col). Куда и когда прыгнуть — выход мозга: залп
клеток верхнего двухолмия (их тормозят базальные ганглии); генератор
саккад ствола переводит взгляд на средний вектор выстреливших клеток (Lee,
Rohrer & Sparks 1988), ~30 мс саккады сетчатка не передаёт картинку
(саккадическое подавление). Поэтому у новичка взгляд прыгает коротко и
часто, а как читать крупнее — буквами, слогами, словами — мозг учится сам
(Rayner 1986: у детей фиксаций на слово больше, саккады короче).

Рука: клавиша нажата, когда её набор клеток выстрелил хотя бы наполовину
(brain.body.Hand); засчитывается начало нажатия — клавишу надо отпустить,
чтобы нажать снова (свойство клавиши). Напечатанное появляется на экране.
"""

from __future__ import annotations

import torch

from brain.body import Eye, Hand, Periphery
from brain.world import Keyboard, Screen


class Session:
    def __init__(self, mind, window_ms: float = 10.0):
        self.mind = mind
        self.cfg = cfg = mind.cfg
        self.brain = mind.brain
        names = [s.name for s in self.brain.sensors]
        i, j = names.index("eye"), names.index("periphery")
        self.eye = Eye(cfg, self.brain.sensor_offsets[i], self.brain.sensors[i].size, self.brain.device)
        self.side = Periphery(cfg, self.brain.sensor_offsets[j], self.brain.sensors[j].size, self.brain.device)
        self.screen = Screen(cfg.retina_rows, cfg.retina_cols)
        # Поле бокового зрения — periphery_fields окон ямки, взгляд в середине.
        self.field = cfg.retina_cols * cfg.periphery_fields
        self.wide = Screen(cfg.retina_rows, self.field)
        self.keyboard = Keyboard()
        self.hand = Hand(len(self.keyboard), cfg.hand_cells)
        self.H = self.brain.areas["hand"].E
        self.SC = self.brain.areas["colliculus"].E
        # Цель саккады клетки двухолмия — её место в поле (px) минус середина.
        cell = torch.arange(self.SC.size, device=self.brain.device).float()
        self.sc_target = (cell + 0.5) * self.field / self.SC.size - self.field / 2
        self.window_ms = window_ms
        # Строка на экране, точка взгляда на ней (px от начала строки) и
        # указка учителя (px или None) — метка под строкой, как палец под
        # буквой: её видит боковое зрение, взгляд идёт за ней сам.
        self.line = ""
        self.gaze = 0.0
        self.pointer: float | None = None
        self.saccades: list[float] = []
        self._held = None
        # Вкус (награда учителя): участок рецепторов и сколько ещё мс он
        # чувствуется.
        self.taste = {s.name: (o, s.size) for s, o in zip(self.brain.sensors, self.brain.sensor_offsets)
                      if s.name in ("sweet", "bitter")}
        self._tasting: tuple[str, float] | None = None

    def _picture(self, screen: Screen, centre: float) -> torch.Tensor:
        """Строка с указкой, сдвинутая так, что точка взгляда — в centre."""

        shift = int(round(centre - self.gaze))
        picture = screen.picture(self.line, shift).clone()

        if self.pointer is not None:
            x = int(round(self.pointer)) + shift
            picture[-3:, max(0, x - 3):max(0, x + 4)] = 0.0

        return picture

    def _view(self, ms: float) -> torch.Tensor:
        """Растр глаза: ямка (полное разрешение) и боковое зрение (грубо)."""

        raster = self.eye.look(self._picture(self.screen, self.cfg.fovea_col), ms)
        side = self.side.look(self._picture(self.wide, self.field / 2), ms)
        out = torch.zeros(raster.shape[0], self.brain.n_receptors, device=self.brain.device)
        out[:, : raster.shape[1]] = raster
        out[:, self.side.offset: self.side.offset + self.side.size] = side[:, self.side.offset:]

        if self._tasting is not None:
            # Рецепторы вкуса стреляют ~50 Гц, пока вкус чувствуется.
            name, left = self._tasting
            offset, size = self.taste[name]
            p = 50.0 * self.cfg.dt_ms / 1000.0
            out[:, offset: offset + size] = (torch.rand(out.shape[0], size, device=out.device) < p).float()
            left -= ms
            self._tasting = (name, left) if left > 0 else None

        return out

    def praise(self, good: bool = True, ms: float = 300.0) -> None:
        """Учитель: «верно» — сладкое, «неверно» — горькое (дальше живёт мозг)."""

        self._tasting = ("sweet" if good else "bitter", ms)
        self.live(ms)

    def _blank(self, ms: float) -> torch.Tensor:
        steps = int(round(ms / self.cfg.dt_ms))
        return torch.zeros(steps, self.brain.n_receptors, device=self.brain.device)

    def live(self, ms: float) -> str:
        """
        Прожить ms: окно за окном сетчатка видит строку у точки взгляда;
        залп двухолмия — саккада, нажатие клавиши — символ в строку.
        Возвращает напечатанное за это время.
        """

        typed, t = "", 0.0

        while t < ms:
            rates = self.brain.observe(self.window_ms, self._view(self.window_ms))
            t += self.window_ms
            # Залп: клетка двухолмия перед саккадой стреляет 300–800 Гц
            # (Sparks 1978) — 3 и больше спайков за окно.
            fired = rates[self.SC.slice] * self.window_ms / 1000.0 >= 3

            if float(fired.float().mean()) >= self.cfg.saccade_share:
                # Саккада: средний вектор выстреливших клеток двухолмия.
                step = float(self.sc_target[fired].mean())
                self.gaze = max(0.0, min(self.gaze + step, self.screen.width(self.line) + 4))
                self.saccades.append(step)
                self.brain.run(self.cfg.saccade_ms, self._blank(self.cfg.saccade_ms))
                t += self.cfg.saccade_ms

            key = self.hand.pressed(rates[self.H.slice])

            if key is not None and key != self._held:
                char = self.keyboard.type(key)
                self.line += char
                typed += char

            self._held = key

        return typed

    def reply(self, text: str, max_ms: float = 10000.0, silence_ms: float = 2000.0) -> str:
        """
        Собеседник пишет text — строка на экране, взгляд в её начале. Мозг
        живёт, пока не напечатает что-то и не замолчит на silence_ms (или
        max_ms всего). Ответ — что напечатала рука.
        """

        self.line, self.gaze, self.saccades = text + " ", 0.0, []
        answer, quiet, t = "", 0.0, 0.0

        while t < max_ms and not (answer and quiet >= silence_ms):
            typed = self.live(self.window_ms * 10)
            t += self.window_ms * 10
            answer += typed
            quiet = 0.0 if typed else quiet + self.window_ms * 10

        return answer
