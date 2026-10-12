import { type MouseEvent, useCallback, useEffect, useRef, useState } from "react";
import { neuron } from "./api";
import type { Anatomy, Area, NeuronCard as Card } from "./brain";
import { NeuronCard } from "./NeuronCard";

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
  neuronAt: Int32Array; // клетка сетки -> локальный индекс (-1 — стекло)
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

// Зона-карта: срез ровно в grid столбцов, E-клетка j — точно на своём
// месте сетки мозга (места слов компактны); тормозные — в оставшиеся
// клетки нижней полосы.
function mapCells(area: Area, cols: number, rows: number, index: number): Int32Array {
  const cellOf = new Int32Array(area.E.size + area.I.size);
  for (let j = 0; j < area.E.size; j++) cellOf[j] = j;
  const spare = cols * rows - area.E.size;
  const order = scatter(spare, 7919 * (index + 1));
  for (let k = 0; k < area.I.size; k++) cellOf[area.E.size + k] = area.E.size + order[k % spare];
  return cellOf;
}

function layout(area: Area, canvas: HTMLCanvasElement, index: number): SectionLayout {
  const start = area.E.start;
  const count = area.E.size + area.I.size;
  const aspect = canvas.clientWidth / Math.max(1, canvas.clientHeight);
  const cols = area.grid ? area.grid : Math.max(1, Math.ceil(Math.sqrt(count * aspect)));
  const rows = Math.max(1, Math.ceil(count / cols));
  const cellOf = area.grid ? mapCells(area, cols, rows, index) : scatter(cols * rows, 7919 * (index + 1)).subarray(0, count);

  const base = new Uint32Array(cols * rows).fill(pack(GLASS));
  const neuronAt = new Int32Array(cols * rows).fill(-1);
  for (let j = 0; j < count; j++) {
    base[cellOf[j]] = j < area.E.size ? HEMA_RAMP[0] : EOSIN_RAMP[0];
    neuronAt[cellOf[j]] = j;
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
    neuronAt,
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
  // Последний рост синапсов: дуги «место A → место B».
  growth?: { id: number; links: { path: string; pre: number[]; post: number[] }[] } | null;
}

interface Arc {
  key: string;
  d: string;
  from: [number, number];
  to: [number, number];
  path: string;
}

// Сколько дуга роста видна (мс).
const ARC_MS = 2200;

// Порядок срезов повторяет путь сигнала: слух -> смысл -> память / рот.
const PLACES: Record<string, string> = {
  auditory: "auditory",
  association: "association",
  prefrontal: "prefrontal",
  motor: "motor",
  limbic: "limbic",
  vta: "vta",
  lc: "lc",
  amygdala: "amygdala",
  category: "category",
};

interface Hover {
  index: number;
  area: string;
  // Рамка вокруг клетки, в долях канваса.
  box: { left: number; top: number; width: number; height: number };
  x: number;
  y: number;
}

// Клетка под курсором; клетки — по пикселю, поэтому, если попал в
// стекло, ищем ближайший нейрон в окрестности 2 клеток.
function hit(section: SectionLayout, canvas: HTMLCanvasElement, clientX: number, clientY: number) {
  const rect = canvas.getBoundingClientRect();
  const col = Math.floor(((clientX - rect.left) / rect.width) * section.cols);
  const row = Math.floor(((clientY - rect.top) / rect.height) * section.rows);
  for (let radius = 0; radius <= 2; radius++) {
    for (let dr = -radius; dr <= radius; dr++) {
      for (let dc = -radius; dc <= radius; dc++) {
        const r = row + dr;
        const c = col + dc;
        if (r < 0 || c < 0 || r >= section.rows || c >= section.cols) continue;
        const j = section.neuronAt[r * section.cols + c];
        if (j >= 0) {
          return {
            index: section.start + j,
            box: {
              left: c / section.cols,
              top: r / section.rows,
              width: 1 / section.cols,
              height: 1 / section.rows,
            },
          };
        }
      }
    }
  }
  return null;
}

// Карточки кэшируются ненадолго: веса меняются, пока мозг учится.
const CARD_TTL_MS = 3000;

function useNeuronCard(index: number | null) {
  const cache = useRef(new Map<number, { card: Card; at: number }>());
  const [card, setCard] = useState<Card | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (index === null) return;
    const cached = cache.current.get(index);
    if (cached && performance.now() - cached.at < CARD_TTL_MS) {
      setCard(cached.card);
      setError(null);
      return;
    }
    setCard(null);
    setLoading(true);
    let cancelled = false;
    // Не дёргать мозг на каждом пикселе, пока мышь едет.
    const timer = window.setTimeout(() => {
      neuron(index)
        .then((fresh) => {
          cache.current.set(index, { card: fresh, at: performance.now() });
          if (!cancelled) {
            setCard(fresh);
            setError(null);
          }
        })
        .catch(() => !cancelled && setError("мозг занят — попробуй ещё раз"))
        .finally(() => !cancelled && setLoading(false));
    }, 120);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [index]);

  return { card, loading, error };
}

// Где на срезе клетка: центр её пикселя в координатах слайда.
function cellPoint(sections: SectionLayout[], index: number, origin: DOMRect): [number, number] | null {
  const section = sections.find((s) => index >= s.start && index < s.start + s.count);
  if (!section) return null;
  const cell = section.cellOf[index - section.start];
  const rect = section.canvas.getBoundingClientRect();
  const col = cell % section.cols;
  const row = Math.floor(cell / section.cols);
  return [
    rect.left - origin.left + ((col + 0.5) / section.cols) * rect.width,
    rect.top - origin.top + ((row + 0.5) / section.rows) * rect.height,
  ];
}

function centroid(points: ([number, number] | null)[]): [number, number] | null {
  const real = points.filter((p): p is [number, number] => p !== null);
  if (!real.length) return null;
  return [real.reduce((s, p) => s + p[0], 0) / real.length, real.reduce((s, p) => s + p[1], 0) / real.length];
}

export function Microscope({ anatomy, fluo, onActivity, growth }: Props) {
  const canvases = useRef<Map<string, HTMLCanvasElement>>(new Map());
  const sectionsRef = useRef<SectionLayout[]>([]);
  const slide = useRef<HTMLDivElement>(null);
  const [arcs, setArcs] = useState<Arc[]>([]);

  // Рост синапсов: дуга от места-источника к месту-цели, гаснет сама.
  useEffect(() => {
    if (!growth || !slide.current) return;
    const origin = slide.current.getBoundingClientRect();
    const fresh: Arc[] = [];
    growth.links.forEach((link, i) => {
      const from = centroid(link.pre.map((n) => cellPoint(sectionsRef.current, n, origin)));
      const to = centroid(link.post.map((n) => cellPoint(sectionsRef.current, n, origin)));
      if (!from || !to) return;
      const bend = Math.hypot(to[0] - from[0], to[1] - from[1]) * 0.25 + 12;
      const mx = (from[0] + to[0]) / 2;
      const my = (from[1] + to[1]) / 2 - bend;
      fresh.push({ key: `${growth.id}-${i}`, d: `M${from[0]},${from[1]} Q${mx},${my} ${to[0]},${to[1]}`, from, to, path: link.path });
    });
    if (!fresh.length) return;
    setArcs((list) => [...list, ...fresh].slice(-60));
    // Без отмены: следующий рост не должен оставлять прошлые дуги навсегда.
    window.setTimeout(() => setArcs((list) => list.filter((arc) => !fresh.includes(arc))), ARC_MS);
  }, [growth]);
  const [hover, setHover] = useState<Hover | null>(null);
  const [pinned, setPinned] = useState<Hover | null>(null);
  const shown = pinned ?? hover;
  const { card, loading, error } = useNeuronCard(shown?.index ?? null);

  const locate = useCallback((area: Area, event: MouseEvent<HTMLCanvasElement>): Hover | null => {
    const section = sectionsRef.current.find((s) => s.area.name === area.name);
    if (!section) return null;
    const found = hit(section, event.currentTarget, event.clientX, event.clientY);
    return found && { ...found, area: area.name, x: event.clientX, y: event.clientY };
  }, []);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => event.key === "Escape" && setPinned(null);
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);
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
      sectionsRef.current = sections;
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

  const section = (area: Area) => (
    <figure key={area.name} className={`section section-${PLACES[area.name] ?? "other"}`}>
      <div className="section-glass">
        <canvas
          ref={(node) => {
            if (node) canvases.current.set(area.name, node);
            else canvases.current.delete(area.name);
          }}
          role="img"
          aria-label={`${area.title}: активность нейронов; наведи на клетку — что она знает`}
          onPointerMove={(event) => setHover(locate(area, event))}
          onPointerLeave={() => setHover(null)}
          onClick={(event) => setPinned(locate(area, event))}
        />
        {[hover, pinned].map(
          (mark, i) =>
            mark?.area === area.name && (
              <span
                key={i}
                className={`neuron-mark${i ? " neuron-mark-pinned" : ""}`}
                style={{
                  left: `${mark.box.left * 100}%`,
                  top: `${mark.box.top * 100}%`,
                  width: `${mark.box.width * 100}%`,
                  height: `${mark.box.height * 100}%`,
                }}
              />
            ),
        )}
      </div>
      <figcaption>
        <span className="section-title">{area.title}</span>
        <span className="section-count">
          {(area.E.size + area.I.size).toLocaleString("ru")} нейронов
        </span>
      </figcaption>
    </figure>
  );
  // Старые зоны — на своих местах сетки; остальные ядра (цель, сон,
  // базальные ганглии) — своей сеткой ниже, иначе они вставали в неявные
  // строки, а заданные схлопывались и подписи наезжали друг на друга.
  const placed = anatomy.areas.filter((area) => PLACES[area.name]);
  const rest = anatomy.areas.filter((area) => !PLACES[area.name]);

  return (
    <div className="slide" ref={slide}>
      <div className="slide-main">{placed.map(section)}</div>
      {rest.length > 0 && <div className="slide-extra">{rest.map(section)}</div>}
      {arcs.length > 0 && (
        <svg className="growth" aria-hidden="true">
          {arcs.map((arc) => (
            <g key={arc.key} className={`growth-arc growth-${arc.path.startsWith("prefrontal") ? "context" : arc.path.endsWith("amygdala") ? "value" : "word"}`}>
              <path d={arc.d} />
              <circle cx={arc.from[0]} cy={arc.from[1]} r={3} />
              <circle cx={arc.to[0]} cy={arc.to[1]} r={4} />
            </g>
          ))}
        </svg>
      )}
      {shown && (
        <NeuronCard
          anatomy={anatomy}
          card={card}
          loading={loading}
          error={error}
          x={shown.x}
          y={shown.y}
          pinned={pinned !== null}
          onClose={() => setPinned(null)}
        />
      )}
    </div>
  );
}
