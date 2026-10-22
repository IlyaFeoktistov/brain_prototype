import { useEffect, useRef, useState } from "react";
import type { Anatomy, Area, Population } from "./brain";
import { KIND_TITLE } from "./brain";

// Флуоресценция каждого нейрона: 1 в момент спайка, затем гаснет, как
// кальциевый индикатор. Спайки приходят снимками раз в ~50 мс (сервер
// прореживает показ, сеть его не ждёт).
export class Fluorescence {
  level: Float32Array;
  private active: number[] = [];
  private isActive: Uint8Array;
  // Клетки, за которыми следят (выбранная и её соседи): время их спайков.
  watched = new Map<number, number[]>();

  constructor(readonly size: number) {
    this.level = new Float32Array(size);
    this.isActive = new Uint8Array(size);
  }

  spike(indices: number[], t: number) {
    for (const i of indices) {
      if (i >= this.size) continue;
      this.level[i] = 1;
      if (!this.isActive[i]) {
        this.isActive[i] = 1;
        this.active.push(i);
      }
      const times = this.watched.get(i);
      if (times) {
        times.push(t);
        if (times.length > 400) times.splice(0, times.length - 400);
      }
    }
  }

  decay(factor: number) {
    const next: number[] = [];
    for (const i of this.active) {
      const value = this.level[i] * factor;
      if (value < 0.03) {
        this.level[i] = 0;
        this.isActive[i] = 0;
      } else {
        this.level[i] = value;
        next.push(i);
      }
    }
    this.active = next;
  }

  activeIn(start: number, end: number): number {
    let count = 0;
    for (const i of this.active) if (i >= start && i < end && this.level[i] > 0.5) count++;
    return count;
  }

  get activeIndices(): readonly number[] {
    return this.active;
  }
}

// Окраска: главные — гематоксилин, тормозные по типам — свои красители.
type RGB = [number, number, number];
const GLASS: RGB = [245, 243, 247];
const DYES: Record<string, [RGB, RGB]> = {
  E: [[222, 216, 236], [44, 30, 120]],
  PV: [[244, 222, 230], [205, 40, 98]],
  I: [[244, 222, 230], [205, 40, 98]],
  SOM: [[226, 236, 222], [40, 120, 60]],
  VIP: [[244, 234, 214], [190, 120, 20]],
  LAMP5: [[222, 236, 240], [20, 120, 150]],
};

const pack = ([r, g, b]: RGB) => (255 << 24) | (b << 16) | (g << 8) | r;
const STEPS = 32;
const RAMPS: Record<string, number[]> = Object.fromEntries(
  Object.entries(DYES).map(([kind, [a, b]]) => [
    kind,
    Array.from({ length: STEPS + 1 }, (_, i) =>
      pack([0, 1, 2].map((c) => Math.round(a[c] + ((b[c] - a[c]) * i) / STEPS)) as RGB),
    ),
  ]),
);

function scatter(count: number, seed: number): Int32Array {
  const order = new Int32Array(count);
  for (let i = 0; i < count; i++) order[i] = i;
  let state = seed || 1;
  for (let i = count - 1; i > 0; i--) {
    state = (state * 1664525 + 1013904223) >>> 0;
    const j = state % (i + 1);
    [order[i], order[j]] = [order[j], order[i]];
  }
  return order;
}

interface Section {
  area: Area;
  cols: number;
  rows: number;
  cellOf: Int32Array; // локальный индекс клетки → пиксель
  neuronAt: Int32Array; // пиксель → локальный индекс (−1 — стекло)
  ramp: number[][]; // локальный индекс → палитра (по типу)
  base: Uint32Array;
  image: ImageData;
  pixels: Uint32Array;
  canvas: HTMLCanvasElement;
}

const layerOf = (pop: Population) => (pop.label.includes(".") ? pop.label.split(".")[0] : "L23");

// Колонка — слоями сверху вниз (L2/3, L4, L5, L6), как на срезе коры;
// внутри слоя главные и тормозные клетки перемешаны. Ядро — одной полосой.
function layout(area: Area, canvas: HTMLCanvasElement, index: number): Section {
  const cols = Math.max(1, Math.ceil(Math.sqrt(area.size)));
  const bands = new Map<string, Population[]>();
  for (const pop of area.populations) {
    const key = area.layered ? layerOf(pop) : "core";
    bands.set(key, [...(bands.get(key) ?? []), pop]);
  }
  const order = ["L23", "L4", "L5", "L6", "core"].filter((k) => bands.has(k));
  const rowsOf = order.map((k) => Math.ceil(bands.get(k)!.reduce((s, p) => s + p.size, 0) / cols));
  const rows = rowsOf.reduce((s, r) => s + r, 0);
  const cellOf = new Int32Array(area.size);
  const neuronAt = new Int32Array(cols * rows).fill(-1);
  const ramp: number[][] = new Array(area.size);
  let row0 = 0;

  order.forEach((key, b) => {
    const pops = bands.get(key)!;
    const cells = pops.flatMap((p) => Array.from({ length: p.size }, (_, k) => [p.start - area.start + k, p.kind] as const));
    const place = scatter(rowsOf[b] * cols, 7919 * (index + 1) + b);
    cells.forEach(([local, kind], k) => {
      const pixel = row0 * cols + place[k];
      cellOf[local] = pixel;
      neuronAt[pixel] = local;
      ramp[local] = RAMPS[kind] ?? RAMPS.I;
    });
    row0 += rowsOf[b];
  });

  const base = new Uint32Array(cols * rows).fill(pack(GLASS));
  for (let j = 0; j < area.size; j++) base[cellOf[j]] = ramp[j][0];
  canvas.width = cols;
  canvas.height = rows;
  const image = new ImageData(cols, rows);
  return { area, cols, rows, cellOf, neuronAt, ramp, base, image, pixels: new Uint32Array(image.data.buffer), canvas };
}

function draw(section: Section, fluo: Fluorescence) {
  const { pixels, base, cellOf, ramp, area } = section;
  pixels.set(base);
  const end = area.start + area.size;
  for (const i of fluo.activeIndices) {
    if (i < area.start || i >= end) continue;
    const j = i - area.start;
    pixels[cellOf[j]] = ramp[j][Math.min(STEPS, Math.round(fluo.level[i] * STEPS))];
  }
  section.canvas.getContext("2d")!.putImageData(section.image, 0, 0);
}

export interface Hover {
  index: number;
  area: Area;
  population: Population;
  x: number;
  y: number;
}

// Связь выбранной клетки: вспыхивает, когда стреляет клетка-источник.
export interface Link {
  from: number;
  to: number;
  inhibitory: boolean;
}

interface Props {
  anatomy: Anatomy;
  fluo: Fluorescence;
  selected: number | null;
  links: Link[];
  onHover: (hover: Hover | null) => void;
  onSelect: (index: number) => void;
  onActivity: (perArea: Record<string, number>) => void;
}

function populationOf(area: Area, index: number): Population {
  return area.populations.find((p) => index >= p.start && index < p.start + p.size) ?? area.populations[0];
}

export function Microscope({ anatomy, fluo, selected, links, onHover, onSelect, onActivity }: Props) {
  const canvases = useRef(new Map<string, HTMLCanvasElement>());
  const sections = useRef<Section[]>([]);
  const slide = useRef<HTMLDivElement>(null);
  const svg = useRef<SVGSVGElement>(null);
  const [marks, setMarks] = useState<{ key: string; x1: number; y1: number; x2: number; y2: number; link: Link }[]>([]);
  const [selectedBox, setSelectedBox] = useState<{ x: number; y: number } | null>(null);

  const point = (index: number): [number, number] | null => {
    const s = sections.current.find((x) => index >= x.area.start && index < x.area.start + x.area.size);
    if (!s || !slide.current) return null;
    const origin = slide.current.getBoundingClientRect();
    const rect = s.canvas.getBoundingClientRect();
    const cell = s.cellOf[index - s.area.start];
    return [
      rect.left - origin.left + ((cell % s.cols) + 0.5) * (rect.width / s.cols),
      rect.top - origin.top + (Math.floor(cell / s.cols) + 0.5) * (rect.height / s.rows),
    ];
  };

  // Линии связей выбранной клетки (пересчёт при смене выбора и размера).
  useEffect(() => {
    const place = () => {
      setSelectedBox(selected === null ? null : (() => {
        const p = point(selected);
        return p && { x: p[0], y: p[1] };
      })());
      setMarks(
        links.flatMap((link, k) => {
          const a = point(link.from);
          const b = point(link.to);
          return a && b ? [{ key: `${k}`, x1: a[0], y1: a[1], x2: b[0], y2: b[1], link }] : [];
        }),
      );
    };
    place();
    window.addEventListener("resize", place);
    return () => window.removeEventListener("resize", place);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [links, selected, anatomy]);

  useEffect(() => {
    const rebuild = () => {
      sections.current = anatomy.areas.flatMap((area, i) => {
        const canvas = canvases.current.get(area.name);
        return canvas ? [layout(area, canvas, i)] : [];
      });
    };
    rebuild();
    let frame = 0;
    let last = performance.now();
    let lastReport = 0;
    const tick = (now: number) => {
      fluo.decay(Math.exp(-(now - last) / 320));
      last = now;
      for (const s of sections.current) draw(s, fluo);
      // Связи выбранной клетки горят, пока горит клетка-источник.
      svg.current?.querySelectorAll<SVGLineElement>("line[data-from]").forEach((line) => {
        const level = fluo.level[Number(line.dataset.from)] ?? 0;
        line.style.opacity = String(0.12 + 0.88 * level);
        line.style.strokeWidth = String(1 + 2.5 * level);
      });
      if (now - lastReport > 200) {
        lastReport = now;
        const perArea: Record<string, number> = {};
        for (const area of anatomy.areas) perArea[area.name] = fluo.activeIn(area.start, area.start + area.size) / area.size;
        onActivity(perArea);
      }
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [anatomy, fluo, onActivity]);

  const locate = (area: Area, event: React.MouseEvent<HTMLCanvasElement>): Hover | null => {
    const s = sections.current.find((x) => x.area.name === area.name);
    if (!s) return null;
    const rect = event.currentTarget.getBoundingClientRect();
    const col = Math.floor(((event.clientX - rect.left) / rect.width) * s.cols);
    const row = Math.floor(((event.clientY - rect.top) / rect.height) * s.rows);
    for (let r = 0; r <= 2; r++)
      for (let dr = -r; dr <= r; dr++)
        for (let dc = -r; dc <= r; dc++) {
          const y = row + dr;
          const x = col + dc;
          if (y < 0 || x < 0 || y >= s.rows || x >= s.cols) continue;
          const j = s.neuronAt[y * s.cols + x];
          if (j >= 0) {
            const index = area.start + j;
            return { index, area, population: populationOf(area, index), x: event.clientX, y: event.clientY };
          }
        }
    return null;
  };

  const systems = anatomy.systems.filter((system) => anatomy.areas.some((a) => a.system === system));

  return (
    <div className="slide" ref={slide}>
      {systems.map((system) => (
        <section key={system} className="system">
          <h3>{system}</h3>
          <div className="system-areas">
            {anatomy.areas
              .filter((a) => a.system === system)
              .map((area) => (
                <figure
                  key={area.name}
                  className={`section${area.layered ? " section-column" : ""}`}
                  style={{ width: `${Math.round(Math.max(56, Math.min(220, Math.sqrt(area.size) * 1.4)))}px` }}
                >
                  <canvas
                    ref={(node) => {
                      if (node) canvases.current.set(area.name, node);
                      else canvases.current.delete(area.name);
                    }}
                    aria-label={`${area.title}: активность клеток`}
                    onPointerMove={(event) => onHover(locate(area, event))}
                    onPointerLeave={() => onHover(null)}
                    onClick={(event) => {
                      const hit = locate(area, event);
                      if (hit) onSelect(hit.index);
                    }}
                  />
                  <figcaption title={area.title}>
                    <span className="section-title">{area.title}</span>
                    <span className="section-count">{area.size.toLocaleString("ru")}</span>
                  </figcaption>
                </figure>
              ))}
          </div>
        </section>
      ))}
      <svg className="wiring" ref={svg} aria-hidden="true">
        {marks.map(({ key, x1, y1, x2, y2, link }) => (
          <line key={key} data-from={link.from} x1={x1} y1={y1} x2={x2} y2={y2}
                className={link.inhibitory ? "wire wire-inh" : "wire"} />
        ))}
        {selectedBox && <circle cx={selectedBox.x} cy={selectedBox.y} r={6} className="selected-cell" />}
      </svg>
    </div>
  );
}

export function HoverTip({ hover }: { hover: Hover }) {
  return (
    <div className="tip" style={{ left: hover.x + 14, top: hover.y + 14 }}>
      <strong>{hover.area.title}</strong>
      <span>
        {hover.population.label} · {KIND_TITLE[hover.population.kind] ?? hover.population.kind}
      </span>
      <span>
        клетка {hover.index - hover.population.start} из {hover.population.size} · №{hover.index}
      </span>
      <em>клик — всё о клетке</em>
    </div>
  );
}
