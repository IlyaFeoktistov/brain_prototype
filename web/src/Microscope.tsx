import { useEffect, useRef } from "react";
import type { Anatomy, Area } from "./brain";

// Флуоресценция каждого нейрона: 1 в момент спайка, затем
// затухает как кальциевый индикатор (~0.3 с реального времени).
export class Fluorescence {
  level: Float32Array;
  private active: number[] = [];
  private isActive: Uint8Array;

  constructor(readonly size: number) {
    this.level = new Float32Array(size);
    this.isActive = new Uint8Array(size);
  }

  spike(indices: number[]) {
    for (const i of indices) {
      if (i >= this.size) continue;
      this.level[i] = 1;
      if (!this.isActive[i]) {
        this.isActive[i] = 1;
        this.active.push(i);
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

// Цвета окраски H&E: гематоксилин (E), эозин (I) на стекле.
const GLASS: [number, number, number] = [245, 243, 247];
const HEMA_REST: [number, number, number] = [222, 216, 236];
const HEMA_FIRE: [number, number, number] = [44, 30, 120];
const EOSIN_REST: [number, number, number] = [244, 222, 230];
const EOSIN_FIRE: [number, number, number] = [205, 40, 98];

function pack([r, g, b]: [number, number, number]): number {
  return (255 << 24) | (b << 16) | (g << 8) | r; // little-endian RGBA
}

function mix(a: [number, number, number], b: [number, number, number], t: number): number {
  return pack([
    Math.round(a[0] + (b[0] - a[0]) * t),
    Math.round(a[1] + (b[1] - a[1]) * t),
    Math.round(a[2] + (b[2] - a[2]) * t),
  ]);
}

// Палитры по уровню флуоресценции, чтобы не смешивать цвета в цикле.
const STEPS = 32;
const HEMA_RAMP = Array.from({ length: STEPS + 1 }, (_, i) => mix(HEMA_REST, HEMA_FIRE, i / STEPS));
const EOSIN_RAMP = Array.from({ length: STEPS + 1 }, (_, i) => mix(EOSIN_REST, EOSIN_FIRE, i / STEPS));

interface SectionLayout {
  area: Area;
  start: number;
  count: number;
  cols: number;
  rows: number;
  cellOf: Int32Array; // локальный индекс нейрона -> клетка сетки
  base: Uint32Array;
  image: ImageData;
  pixels: Uint32Array;
  canvas: HTMLCanvasElement;
}

// Детерминированно перемешать: I-клетки рассыпаны среди E, как в ткани.
function scatter(count: number, seed: number): Int32Array {
  const order = new Int32Array(count);
  for (let i = 0; i < count; i++) order[i] = i;
  let state = seed || 1;
  for (let i = count - 1; i > 0; i--) {
    state = (state * 1664525 + 1013904223) >>> 0;
    const j = state % (i + 1);
    const t = order[i];
    order[i] = order[j];
    order[j] = t;
  }
  return order;
}

function layout(area: Area, canvas: HTMLCanvasElement, index: number): SectionLayout {
  const start = area.E.start;
  const count = area.E.size + area.I.size;
  const aspect = canvas.clientWidth / Math.max(1, canvas.clientHeight);
  const cols = Math.max(1, Math.ceil(Math.sqrt(count * aspect)));
  const rows = Math.max(1, Math.ceil(count / cols));
  const cellOf = scatter(cols * rows, 7919 * (index + 1)).subarray(0, count);

  const base = new Uint32Array(cols * rows).fill(pack(GLASS));
  for (let j = 0; j < count; j++) {
    base[cellOf[j]] = j < area.E.size ? HEMA_RAMP[0] : EOSIN_RAMP[0];
  }

  canvas.width = cols;
  canvas.height = rows;
  const image = new ImageData(cols, rows);

  return {
    area,
    start,
    count,
    cols,
    rows,
    cellOf,
    base,
    image,
    pixels: new Uint32Array(image.data.buffer),
    canvas,
  };
}

function draw(section: SectionLayout, fluo: Fluorescence) {
  const { pixels, base, start, count, cellOf, area } = section;
  pixels.set(base);
  const end = start + count;
  for (const i of fluo.activeIndices) {
    if (i < start || i >= end) continue;
    const j = i - start;
    const step = Math.min(STEPS, Math.round(fluo.level[i] * STEPS));
    pixels[cellOf[j]] = (j < area.E.size ? HEMA_RAMP : EOSIN_RAMP)[step];
  }
  section.canvas.getContext("2d")!.putImageData(section.image, 0, 0);
}

interface Props {
  anatomy: Anatomy;
  fluo: Fluorescence;
  onActivity: (perArea: Record<string, number>) => void;
}

// Порядок срезов повторяет путь сигнала: слух -> смысл -> память / рот.
const PLACES: Record<string, string> = {
  auditory: "auditory",
  association: "association",
  prefrontal: "prefrontal",
  motor: "motor",
  limbic: "limbic",
};

export function Microscope({ anatomy, fluo, onActivity }: Props) {
  const canvases = useRef<Map<string, HTMLCanvasElement>>(new Map());
  const reduceMotion = useRef(
    typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches,
  );

  useEffect(() => {
    let sections: SectionLayout[] = [];
    const rebuild = () => {
      sections = anatomy.areas.flatMap((area, index) => {
        const canvas = canvases.current.get(area.name);
        return canvas ? [layout(area, canvas, index)] : [];
      });
    };
    rebuild();

    const observer = new ResizeObserver(rebuild);
    canvases.current.forEach((canvas) => observer.observe(canvas));

    let frame = 0;
    let last = performance.now();
    let lastReport = 0;
    const tick = (now: number) => {
      const elapsed = now - last;
      last = now;
      // Медленнее гаснет при reduced motion: меньше мелькания.
      const tau = reduceMotion.current ? 900 : 320;
      fluo.decay(Math.exp(-elapsed / tau));
      for (const section of sections) draw(section, fluo);

      if (now - lastReport > 120) {
        lastReport = now;
        const perArea: Record<string, number> = {};
        for (const area of anatomy.areas) {
          perArea[area.name] = fluo.activeIn(area.E.start, area.E.start + area.E.size) / area.E.size;
        }
        onActivity(perArea);
      }
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);

    return () => {
      cancelAnimationFrame(frame);
      observer.disconnect();
    };
  }, [anatomy, fluo, onActivity]);

  return (
    <div className="slide">
      {anatomy.areas.map((area) => (
        <figure key={area.name} className={`section section-${PLACES[area.name] ?? "other"}`}>
          <canvas
            ref={(node) => {
              if (node) canvases.current.set(area.name, node);
              else canvases.current.delete(area.name);
            }}
            role="img"
            aria-label={`${area.title}: активность нейронов`}
          />
          <figcaption>
            <span className="section-title">{area.title}</span>
            <span className="section-count">
              {(area.E.size + area.I.size).toLocaleString("ru")} нейронов
            </span>
          </figcaption>
        </figure>
      ))}
    </div>
  );
}
