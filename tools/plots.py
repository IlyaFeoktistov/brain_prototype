"""
Общие картинки опытов: запись по времени, лента панелей, растр.

Подключение в любой опыт (цикл b.observe кусками):

    import plots
    rec = plots.Recorder(b, groups={"слово А": cells_a, ...})   # зоны пишутся сами
    for ...:
        r = b.observe(ms, raster)
        rec.add(r, ms, extra={"ответ E1": ans})                # свои ряды — по желанию
    rec.save("/tmp/x.pt", windows=[(400, 600, "Б")])
    plots.timeline_from("/tmp/x.pt")                           # → /tmp/x.png

Лента (timeline): панели друг под другом с общей осью времени. Панель —
словарь: label; series {имя: ряд [T] или матрица [T, k]}; по желанию
hlines [(y, подпись)], thr {имя: порог} (пунктир цвета ряда), heat=True
(матрица [T, k] картой log10 Гц, rows — подписи строк), smooth (кусков).
Окна стимулов — [(начало мс, конец мс, подпись)]: серые полосы с подписью.

Цвета — категориальная палитра dataviz (порядок фиксирован, цвет следует за
именем ряда в пределах картинки, не за местом в списке).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
WINDOW, WINDOW_ALT = "#d8d7d2", "#e6eefa"


def smooth(x: np.ndarray, k: int) -> np.ndarray:
    """Скользящее среднее по k точкам вдоль времени (ось 0)."""

    if k <= 1:
        return x
    ker = np.ones(k) / k
    if x.ndim == 1:
        return np.convolve(x, ker, mode="same")
    return np.apply_along_axis(lambda v: np.convolve(v, ker, mode="same"), 0, x)


def _np(v) -> np.ndarray:
    return v.detach().float().cpu().numpy() if torch.is_tensor(v) else np.asarray(v, dtype=float)


class Recorder:
    """
    Запись частот по кускам: средняя частота E каждой зоны, выбранные группы
    клеток (номера клеток мозга) и свои ряды (extra). Время — конец куска, мс.
    """

    def __init__(self, b, groups: dict | None = None, areas: bool = True):
        self.areas = [a for a in b.areas if b.areas[a].E is not None] if areas else []
        self._slices = [b.areas[a].E.slice for a in self.areas]
        self.groups = {k: torch.as_tensor(v, device=b.device) for k, v in (groups or {}).items()}
        self.t, self.rows = [], {"areas": []} if areas else {}
        self.now = 0.0

    def add(self, r: torch.Tensor, ms: float, extra: dict | None = None) -> None:
        self.now += ms
        self.t.append(self.now)
        if self.areas:
            self.rows["areas"].append(torch.stack([r[s].mean() for s in self._slices]).cpu())
        for k, idx in self.groups.items():
            self.rows.setdefault(k, []).append(r[idx].float().mean().cpu())
        for k, v in (extra or {}).items():
            self.rows.setdefault(k, []).append(torch.as_tensor(v).detach().float().cpu().clone())

    def data(self) -> dict:
        return {"t_ms": torch.tensor(self.t), "area_names": self.areas,
                "rows": {k: torch.stack(v) for k, v in self.rows.items() if v}}

    def save(self, path: str, **meta) -> None:
        torch.save({**self.data(), **meta}, path)


def _style(a) -> None:
    a.grid(axis="y", color=GRID, lw=0.6)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)


def timeline(path: str, t_ms, panels: list[dict], windows=None, windows_alt=None,
             title: str = "", width: float = 16.0, dpi: int = 90) -> str:
    """Лента панелей с общей осью времени (с). Возвращает путь картинки."""

    t = _np(t_ms) / 1000.0
    dt = float(t[1] - t[0]) if len(t) > 1 else 0.005
    heights = [p.get("height", 1.6 if p.get("heat") else 1.0) for p in panels]
    fig, axes = plt.subplots(len(panels), 1, figsize=(width, 2.6 * sum(heights) + 1), sharex=True,
                             gridspec_kw={"height_ratios": heights}, squeeze=False)
    axes = axes[:, 0]
    color_of: dict[str, str] = {}

    def color(name: str) -> str:
        if name not in color_of:
            color_of[name] = COLORS[len(color_of) % len(COLORS)] if len(color_of) < len(COLORS) else MUTED
        return color_of[name]

    for a, p in zip(axes, panels):
        _style(a)
        k = int(p.get("smooth", 1))
        if p.get("heat"):
            (name, m), = p["series"].items()
            m = smooth(_np(m), k)
            im = a.imshow(np.log10(m.T + 0.1), aspect="auto", cmap="Blues", interpolation="nearest",
                          extent=[t[0] - dt, t[-1], m.shape[1] - 0.5, -0.5], vmin=-1, vmax=2)
            rows = p.get("rows") or [str(i) for i in range(m.shape[1])]
            a.set_yticks(range(len(rows)), rows, fontsize=max(5, min(9, int(400 / max(len(rows), 1)) // 10)))
            a.grid(False)
            for s, e, w in windows or []:
                a.axvline(s / 1000, color=INK, lw=0.6, ls=":")
            cb = fig.colorbar(im, cax=a.inset_axes([1.005, 0.0, 0.008, 1.0]))   # шкала вне панели — ось времени общая
            cb.set_ticks([-1, 0, 1, 2], labels=["0.1", "1", "10", "100"])
            cb.set_label("Гц (log)", fontsize=8)
        else:
            for s, e, w in windows or []:
                a.axvspan(s / 1000, e / 1000, color=WINDOW, lw=0)
            for s, e, w in windows_alt or []:
                a.axvspan(s / 1000, e / 1000, color=WINDOW_ALT, lw=0)
            for name, v in p["series"].items():
                v = smooth(_np(v), k)
                a.plot(t, v, color=color(name), lw=1.3, label=name)
            for name, y in (p.get("thr") or {}).items():
                a.axhline(y, color=color(name), lw=0.8, ls="--")
            for y, text in p.get("hlines") or []:
                a.axhline(y, color=MUTED, lw=0.8, ls=":")
                a.text(t[-1], y, " " + text, fontsize=8, color=MUTED, va="center")
            if p.get("log"):
                a.set_yscale("symlog", linthresh=p.get("linthresh", 1.0))
            if len(p["series"]) > 1:
                a.legend(ncol=min(8, len(p["series"])), loc="upper right", fontsize=8, frameon=False)
        a.set_ylabel(p.get("label", ""), fontsize=9)
    if windows:
        top = axes[0]
        for s, e, w in windows:
            top.text((s + e) / 2000, 1.0, w, transform=top.get_xaxis_transform(), ha="center", va="bottom",
                     fontsize=9, color=MUTED)
    if title:
        axes[0].set_title(title, fontsize=11, color=INK, loc="left", pad=14)
    axes[-1].set_xlabel("время, с")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def timeline_from(path: str, out: str | None = None, smooth_k: int = 5, title: str = "") -> str:
    """Лента по файлу Recorder.save: панель на каждую группу/ряд и карта зон."""

    d = torch.load(path, weights_only=False)
    rows = d["rows"]
    panels = []
    groups = {k: v for k, v in rows.items() if k != "areas" and v.ndim == 1}
    if groups:
        panels.append({"label": "группы, Гц", "series": groups, "smooth": smooth_k})
    for k, v in rows.items():
        if k != "areas" and v.ndim == 2:
            panels.append({"label": k, "series": {f"{k} {i}": v[:, i] for i in range(v.shape[1])},
                           "smooth": smooth_k})
    if "areas" in rows:
        panels.append({"label": "зоны (E)", "series": {"areas": rows["areas"]}, "heat": True,
                       "rows": d.get("area_names"), "smooth": smooth_k})
    return timeline(out or path.rsplit(".", 1)[0] + ".png", d["t_ms"], panels,
                    windows=d.get("windows"), title=title or d.get("title", ""))


def raster(path: str, rows: list[tuple[str, torch.Tensor]], frac: list[tuple[str, torch.Tensor]] | None = None,
           step_ms: float = 1.0, title: str = "", dpi: int = 110) -> str:
    """
    Растр: rows — [(имя группы, спайки [T, клеток] bool)], группы друг под
    другом; frac — [(имя, ряд [T])] нижней панелью (доля клеток за шаг).
    """

    fig, axes = plt.subplots(2 if frac else 1, 1, figsize=(13, 8 if frac else 6), sharex=True, squeeze=False,
                             gridspec_kw={"height_ratios": [3, 1.3] if frac else [1]})
    ax = axes[0, 0]
    base = 0
    T = 1
    for j, (name, spk) in enumerate(rows):
        T = max(T, spk.shape[0])
        ti, ni = torch.nonzero(spk, as_tuple=True)
        ax.scatter((ti.cpu() * step_ms).numpy(), (ni + base).cpu().numpy(), s=2,
                   color=COLORS[j % len(COLORS)], marker="|")
        ax.text(-T * step_ms * 0.01, base + spk.shape[1] / 2, name, ha="right", va="center", fontsize=9)
        base += spk.shape[1] + 10
    ax.set_yticks([])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if title:
        ax.set_title(title, fontsize=11, color=INK, loc="left")
    if frac:
        a2 = axes[1, 0]
        _style(a2)
        for j, (name, v) in enumerate(frac):
            v = _np(v)
            a2.plot(np.arange(len(v)) * step_ms, v, color=COLORS[j % len(COLORS)], lw=1, label=name)
        a2.set_ylabel("доля клеток за шаг")
        a2.set_yscale("symlog", linthresh=1e-3)
        a2.legend(fontsize=8, ncol=6, loc="upper right", frameon=False)
    axes[-1, 0].set_xlabel("мс")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path
