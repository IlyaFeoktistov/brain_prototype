import { useEffect, useRef, useState } from "react";
import * as api from "./api";
import type { Anatomy, Area, Population } from "./brain";
import { type Fluorescence, GLASS, type Hover, pack, RAMPS, STEPS } from "./Microscope";

// Карта нейронов: весь мозг одной сплошной тканью с зумом, «от центра»: в
// середине глубинные зоны (таламус, базальные ганглии, гиппокамп, миндалина,
// модуляторы), вокруг — кора по органам: сверху зрение, слева слух, снизу рот
// и движения (моторная, речевая, рука, мозжечок), справа — кора смысла и
// решений (смысловая с портом слов, категории, лимбическая, лобная, «цель»).
// Это схема по смыслу, не координаты мозга. Каждая сторона заполнена зонами
// сплошь (плиткой), площадь ∝ клеткам. Внутри зоны клетка лежит на своём
// месте, только если оно задано анатомией модели (brain/places.py): лист коры
// — место на листе, слои друг под другом (L2/3 сверху, как на срезе); зубчатая
// — по длине, типы полосами. У ядер места нет — клетки по номеру, строками.
// Линии между зонами (по желанию) — только существующие пути.

type Side = "center" | "top" | "left" | "right" | "bottom";
const SIDES: Record<Exclude<Side, "center">, string[]> = {
  top: ["lgn", "visual", "letters", "colliculus"],
  left: ["mgn", "trn_mgn", "cn", "vnll", "ic", "auditory"],
  bottom: ["premotor", "motor", "hand", "thalamus", "trn_motor", "vl_cb", "granule", "purkinje", "dcn", "io"],
  right: ["association", "category", "limbic", "prefrontal", "goal", "md", "trn_md"],
};
// Всё, чего нет в сторонах, — в центре (гиппокамп, базальные ганглии, ...).
const sideOf = (name: string): Side =>
  (Object.entries(SIDES).find(([, names]) => names.includes(name))?.[0] as Side) ?? "center";

// Пикселей карты на клетку: чуть больше 1 — у клеток листа со случайными
// местами меньше совпадений пикселя.
const DENSITY = 1.3;
// Фон вне зон (углы креста).
const BACKDROP = pack([28, 26, 34]);

interface PlaceArea {
  name: string;
  mode: "sheet" | "line" | "none";
  tiles: string[];
  tile: string;
  xy: string;
}

interface Rect {
  x: number;
  y: number;
  w: number;
  h: number;
}

interface Block extends Rect {
  area: Area;
  mode: PlaceArea["mode"];
  tiles: (Rect & { label: string })[];
}

interface World {
  width: number;
  height: number;
  blocks: Block[];
  pixelOf: Int32Array; // клетка → пиксель мира (−1 — нет)
  neuronAt: Int32Array; // пиксель → клетка (−1 — стекло)
  ramp: (number[] | null)[];
  base: Uint32Array;
}

const decode = (b64: string) => Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));

function kindOf(pop: Population, tileLabel: string | undefined): string {
  if (tileLabel && /Granule|Mossy/.test(tileLabel)) return "E";
  if (tileLabel && /Basket|HIPP/.test(tileLabel)) return "PV";
  return pop.kind;
}

// Делит длину total на части ∝ weights (целые, сумма — total, каждая ≥ 1 при total ≥ числу частей).
function split(total: number, weights: number[]): number[] {
  const sum = weights.reduce((a, b) => a + b, 0) || 1;
  const out = weights.map((w) => Math.max(1, Math.floor((total * w) / sum)));
  let rest = total - out.reduce((a, b) => a + b, 0);
  for (let i = 0; rest !== 0 && out.length; i = (i + 1) % out.length) {
    if (rest > 0) {
      out[i]++;
      rest--;
    } else if (out[i] > 1) {
      out[i]--;
      rest++;
    }
  }
  return out;
}

// Плитка зон в прямоугольнике: полосами (slice-and-dice по длинной стороне,
// рекурсивно пополам по клеткам) — сплошь, площадь ∝ клеткам, целые пиксели.
function treemap(list: Area[], r: Rect): [Area, Rect][] {
  if (list.length === 1) return [[list[0], r]];
  const total = list.reduce((s, a) => s + a.size, 0);
  let acc = 0;
  let cut = 0;
  while (cut < list.length - 1 && acc + list[cut].size <= total / 2) acc += list[cut++].size;
  if (cut === 0) {
    acc = list[0].size;
    cut = 1;
  }
  const [a, b] = [list.slice(0, cut), list.slice(cut)];
  if (r.w >= r.h) {
    const w = Math.min(r.w - 1, Math.max(1, Math.round((r.w * acc) / total)));
    return [...treemap(a, { x: r.x, y: r.y, w, h: r.h }), ...treemap(b, { x: r.x + w, y: r.y, w: r.w - w, h: r.h })];
  }
  const h = Math.min(r.h - 1, Math.max(1, Math.round((r.h * acc) / total)));
  return [...treemap(a, { x: r.x, y: r.y, w: r.w, h }), ...treemap(b, { x: r.x, y: r.y + h, w: r.w, h: r.h - h })];
}

function build(anatomy: Anatomy, places: PlaceArea[]): World {
  const placeOf = new Map(places.map((p) => [p.name, p]));
  const tileOf = new Map(places.map((p) => [p.name, decode(p.tile)]));
  const groups: Record<Side, Area[]> = { center: [], top: [], left: [], right: [], bottom: [] };
  for (const area of anatomy.areas) groups[sideOf(area.name)].push(area);
  const cellsOf = (list: Area[]) => list.reduce((s, a) => s + a.size, 0) * DENSITY;
  // Крест: центр — квадрат; верх и низ — шириной центра, бока — его высотой.
  const side = Math.max(40, Math.ceil(Math.sqrt(cellsOf(groups.center))));
  const extent = (list: Area[]) => (list.length ? Math.max(4, Math.ceil(cellsOf(list) / side)) : 0);
  const top = extent(groups.top);
  const bottom = extent(groups.bottom);
  const left = extent(groups.left);
  const right = extent(groups.right);
  const width = left + side + right;
  const height = top + side + bottom;
  const regions: Record<Side, Rect> = {
    center: { x: left, y: top, w: side, h: side },
    top: { x: left, y: 0, w: side, h: top },
    bottom: { x: left, y: top + side, w: side, h: bottom },
    left: { x: 0, y: top, w: left, h: side },
    right: { x: left + side, y: top, w: right, h: side },
  };
  const blocks: Block[] = [];
  for (const key of Object.keys(regions) as Side[]) {
    const list = groups[key];
    if (!list.length) continue;
    // Порядок внутри стороны — как в списке (от органа к коре); центр — как в анатомии.
    const order = key === "center" ? list : SIDES[key as Exclude<Side, "center">].map((n) => list.find((a) => a.name === n)).filter((a): a is Area => !!a);
    for (const [area, r] of treemap(order, regions[key])) {
      const place = placeOf.get(area.name);
      const mode = place?.mode ?? "none";
      const tiles: Block["tiles"] = [];
      if (mode === "none") {
        tiles.push({ label: "", ...r });
      } else {
        const ids = tileOf.get(area.name)!;
        const counts = place!.tiles.map((_, t) => ids.reduce((s, v) => s + (v === t ? 1 : 0), 0));
        let ty = r.y;
        split(r.h, counts).forEach((th, t) => {
          tiles.push({ label: place!.tiles[t], x: r.x, y: ty, w: r.w, h: th });
          ty += th;
        });
      }
      blocks.push({ area, mode, tiles, ...r });
    }
  }

  const pixelOf = new Int32Array(anatomy.neurons).fill(-1);
  const neuronAt = new Int32Array(width * height).fill(-1);
  const ramp: (number[] | null)[] = new Array(anatomy.neurons).fill(null);
  const base = new Uint32Array(width * height).fill(BACKDROP);
  const glass = pack(GLASS);
  for (const b of blocks) for (let yy = b.y; yy < b.y + b.h; yy++) base.fill(glass, yy * width + b.x, yy * width + b.x + b.w);

  for (const block of blocks) {
    const area = block.area;
    const place = placeOf.get(area.name);
    const ids = place ? tileOf.get(area.name)! : null;
    const xy = place && place.mode !== "none" ? new Uint16Array(decode(place.xy).buffer) : null;
    const grid = block.tiles[0];
    // Ядро: клетки по номеру строками, равномерно по всей площади зоны.
    const step = (grid.w * grid.h) / Math.max(1, area.size);
    for (const pop of area.populations) {
      for (let k = 0; k < pop.size; k++) {
        const index = pop.start + k;
        const local = index - area.start;
        let px: number;
        let py: number;
        let label: string | undefined;
        if (xy && ids) {
          const t = block.tiles[ids[local]];
          label = t.label;
          px = t.x + Math.min(t.w - 1, Math.floor((xy[2 * local] / 65535) * t.w));
          py = t.y + Math.min(t.h - 1, Math.floor((xy[2 * local + 1] / 65535) * t.h));
        } else {
          const slot = Math.min(grid.w * grid.h - 1, Math.floor(local * step));
          px = grid.x + (slot % grid.w);
          py = grid.y + Math.floor(slot / grid.w);
        }
        const pixel = py * width + px;
        pixelOf[index] = pixel;
        neuronAt[pixel] = index;
        ramp[index] = RAMPS[kindOf(pop, label)] ?? RAMPS.I;
        base[pixel] = ramp[index]![0];
      }
    }
  }
  return { width, height, blocks, pixelOf, neuronAt, ramp, base };
}

// Мини-карта: правый нижний угол холста, ширина ≤ 200 px (экран).
function miniRect(cw: number, ch: number, world: { width: number; height: number }) {
  const w = Math.min(200, cw * 0.3);
  const h = (w * world.height) / world.width;
  return { x: cw - w - 10, y: ch - h - 10, w, h };
}

interface Props {
  anatomy: Anatomy;
  fluo: Fluorescence;
  selected: number | null;
  onHover: (hover: Hover | null) => void;
  onSelect: (index: number) => void;
  onActivity: (perArea: Record<string, number>) => void;
}

export function NeuronMap({ anatomy, fluo, selected, onHover, onSelect, onActivity }: Props) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const [world, setWorld] = useState<World | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [paths, setPaths] = useState(false);
  const view = useRef({ scale: 1, tx: 0, ty: 0, fitted: false });
  const pathsRef = useRef(paths);
  pathsRef.current = paths;
  const selectedRef = useRef(selected);
  selectedRef.current = selected;
  const worldRef = useRef<World | null>(null);
  worldRef.current = world;

  useEffect(() => {
    let alive = true;
    api.places()
      .then((p) => alive && setWorld(build(anatomy, p.areas as PlaceArea[])))
      .catch((e) => alive && setError(String(e.message ?? e)));
    return () => {
      alive = false;
    };
  }, [anatomy]);

  const fit = () => {
    const c = canvas.current;
    const w = worldRef.current;
    if (!c || !w) return;
    const scale = Math.min(c.clientWidth / w.width, c.clientHeight / w.height);
    view.current = { scale, tx: (c.clientWidth - w.width * scale) / 2, ty: (c.clientHeight - w.height * scale) / 2, fitted: true };
  };

  const toWorld = (clientX: number, clientY: number) => {
    const rect = canvas.current!.getBoundingClientRect();
    const { scale, tx, ty } = view.current;
    return [(clientX - rect.left - tx) / scale, (clientY - rect.top - ty) / scale];
  };

  // Колесо — зум к курсору; слушатель не пассивный, чтобы страница не листалась.
  useEffect(() => {
    const c = canvas.current;
    if (!c) return;
    const wheel = (event: WheelEvent) => {
      event.preventDefault();
      const [wx, wy] = toWorld(event.clientX, event.clientY);
      const v = view.current;
      const scale = Math.min(60, Math.max(0.05, v.scale * Math.exp(-event.deltaY * 0.0015)));
      const rect = c.getBoundingClientRect();
      view.current = { scale, tx: event.clientX - rect.left - wx * scale, ty: event.clientY - rect.top - wy * scale, fitted: true };
    };
    c.addEventListener("wheel", wheel, { passive: false });
    return () => c.removeEventListener("wheel", wheel);
  }, []);

  useEffect(() => {
    if (!world) return;
    const c = canvas.current!;
    const off = document.createElement("canvas");
    off.width = world.width;
    off.height = world.height;
    const offCtx = off.getContext("2d")!;
    const image = new ImageData(world.width, world.height);
    const pixels = new Uint32Array(image.data.buffer);
    const byName = new Map(world.blocks.map((b) => [b.area.name, b]));
    let frame = 0;
    let last = performance.now();
    let lastReport = 0;

    const tick = (now: number) => {
      const dpr = window.devicePixelRatio || 1;
      if (c.width !== Math.round(c.clientWidth * dpr) || c.height !== Math.round(c.clientHeight * dpr)) {
        c.width = Math.round(c.clientWidth * dpr);
        c.height = Math.round(c.clientHeight * dpr);
      }
      if (!view.current.fitted) fit();
      fluo.decay(Math.exp(-(now - last) / 320));
      last = now;
      pixels.set(world.base);
      for (const i of fluo.activeIndices) {
        const p = world.pixelOf[i];
        const r = world.ramp[i];
        if (p >= 0 && r) pixels[p] = r[Math.min(STEPS, Math.round(fluo.level[i] * STEPS))];
      }
      offCtx.putImageData(image, 0, 0);

      const ctx = c.getContext("2d")!;
      const { scale, tx, ty } = view.current;
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.fillStyle = "#1c1a22";
      ctx.fillRect(0, 0, c.width, c.height);
      ctx.setTransform(scale * dpr, 0, 0, scale * dpr, tx * dpr, ty * dpr);
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(off, 0, 0);

      // Границы зон (и слоёв при приближении) — тонкие линии.
      ctx.lineWidth = 1 / scale;
      ctx.strokeStyle = "rgba(40, 30, 90, 0.55)";
      for (const b of world.blocks) ctx.strokeRect(b.x, b.y, b.w, b.h);
      if (scale > 2) {
        ctx.strokeStyle = "rgba(40, 30, 90, 0.2)";
        for (const b of world.blocks) for (const t of b.tiles) if (t.label) ctx.strokeRect(t.x, t.y, t.w, t.h);
      }

      if (pathsRef.current) {
        // Пути — между серединами зон; яркость — доля горящих клеток источника.
        for (const p of anatomy.pathways) {
          const a = byName.get(p.src);
          const b = byName.get(p.dst);
          if (!a || !b || a === b) continue;
          const lit = fluo.activeIn(a.area.start, a.area.start + a.area.size) / a.area.size;
          ctx.strokeStyle = `rgba(255, 170, 60, ${Math.min(0.9, 0.15 + lit * 12)})`;
          const x1 = a.x + a.w / 2;
          const y1 = a.y + a.h / 2;
          const x2 = b.x + b.w / 2;
          const y2 = b.y + b.h / 2;
          ctx.beginPath();
          ctx.moveTo(x1, y1);
          ctx.bezierCurveTo((x1 + x2) / 2, y1, (x1 + x2) / 2, y2, x2, y2);
          ctx.stroke();
        }
      }

      // Подписи — в экранном размере, только если помещаются в зону.
      const fontPx = 11 / scale;
      ctx.font = `${fontPx}px system-ui, sans-serif`;
      ctx.textBaseline = "top";
      const fitted = (text: string, room: number) => {
        if (ctx.measureText(text).width <= room) return text;
        let lo = 0;
        let hi = text.length;
        while (lo < hi) {
          const mid = (lo + hi + 1) >> 1;
          if (ctx.measureText(text.slice(0, mid) + "…").width <= room) lo = mid;
          else hi = mid - 1;
        }
        return lo > 3 ? text.slice(0, lo) + "…" : "";
      };
      for (const b of world.blocks) {
        if (b.h < fontPx * 1.3) continue;
        const note = b.mode === "none" ? " · без места" : b.mode === "line" ? " · по длине" : " · лист";
        const text = fitted(`${b.area.title}${note}`, b.w - 4 / scale);
        if (!text) continue;
        const tw = ctx.measureText(text).width;
        ctx.fillStyle = "rgba(28, 26, 34, 0.72)";
        ctx.fillRect(b.x + 1 / scale, b.y + 1 / scale, tw + 4 / scale, fontPx * 1.25);
        ctx.fillStyle = "#f3eefc";
        ctx.fillText(text, b.x + 3 / scale, b.y + 2 / scale);
        if (scale > 2)
          for (const t of b.tiles) {
            if (!t.label || t === b.tiles[0] || t.h < fontPx * 1.3) continue;
            const lt = fitted(t.label, t.w - 4 / scale);
            if (!lt) continue;
            ctx.fillStyle = "rgba(28, 26, 34, 0.55)";
            ctx.fillRect(t.x + 1 / scale, t.y + 1 / scale, ctx.measureText(lt).width + 4 / scale, fontPx * 1.25);
            ctx.fillStyle = "#f3eefc";
            ctx.fillText(lt, t.x + 3 / scale, t.y + 2 / scale);
          }
      }

      // Мини-карта в углу: весь мозг и рамка видимой части.
      const mini = miniRect(c.clientWidth, c.clientHeight, world);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.fillStyle = "rgba(20, 18, 26, 0.85)";
      ctx.fillRect(mini.x - 3, mini.y - 3, mini.w + 6, mini.h + 6);
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(off, mini.x, mini.y, mini.w, mini.h);
      const k = mini.w / world.width;
      const vx = (-tx / scale) * k;
      const vy = (-ty / scale) * k;
      const vw = (c.clientWidth / scale) * k;
      const vh = (c.clientHeight / scale) * k;
      ctx.save();
      ctx.beginPath();
      ctx.rect(mini.x, mini.y, mini.w, mini.h);
      ctx.clip();
      ctx.strokeStyle = "#ffb000";
      ctx.lineWidth = 1.5;
      ctx.strokeRect(mini.x + vx, mini.y + vy, Math.max(2, vw), Math.max(2, vh));
      ctx.restore();
      ctx.setTransform(scale * dpr, 0, 0, scale * dpr, tx * dpr, ty * dpr);

      const sel = selectedRef.current;
      if (sel !== null && world.pixelOf[sel] >= 0) {
        const p = world.pixelOf[sel];
        ctx.strokeStyle = "#ffb000";
        ctx.lineWidth = 2 / scale;
        ctx.beginPath();
        ctx.arc((p % world.width) + 0.5, Math.floor(p / world.width) + 0.5, 6 / scale, 0, 2 * Math.PI);
        ctx.stroke();
      }

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
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [world, fluo]);

  const locate = (event: React.PointerEvent): Hover | null => {
    if (!world) return null;
    const [wx, wy] = toWorld(event.clientX, event.clientY);
    const reach = Math.max(1, Math.ceil(3 / view.current.scale));
    for (let r = 0; r <= reach; r++)
      for (let dy = -r; dy <= r; dy++)
        for (let dx = -r; dx <= r; dx++) {
          const x = Math.floor(wx) + dx;
          const y = Math.floor(wy) + dy;
          if (x < 0 || y < 0 || x >= world.width || y >= world.height) continue;
          const index = world.neuronAt[y * world.width + x];
          if (index < 0) continue;
          const area = anatomy.areas.find((a) => index >= a.start && index < a.start + a.size)!;
          const population = area.populations.find((p) => index >= p.start && index < p.start + p.size) ?? area.populations[0];
          return { index, area, population, x: event.clientX, y: event.clientY };
        }
    return null;
  };

  const drag = useRef<{ x: number; y: number; moved: boolean } | null>(null);

  return (
    <div className="neuron-map">
      <div className="map-tools">
        <button type="button" onClick={fit} title="весь мозг на экране">весь мозг</button>
        <label title="линии существующих путей между зонами; яркость — доля горящих клеток источника">
          <input type="checkbox" checked={paths} onChange={(e) => setPaths(e.target.checked)} /> пути
        </label>
        <span className="map-hint">центр — глубинные зоны, сверху зрение, слева слух, снизу рот и движения, справа кора смысла; колесо — зум, перетаскивание — сдвиг, клик — клетка</span>
      </div>
      {error && <p className="map-error">карта: {error}</p>}
      <canvas
        ref={canvas}
        className="map-canvas"
        aria-label="карта нейронов мозга"
        onPointerDown={(event) => {
          // Клик по мини-карте — перейти туда (середина экрана — на это место).
          const c = canvas.current!;
          if (world) {
            const rect = c.getBoundingClientRect();
            const mini = miniRect(c.clientWidth, c.clientHeight, world);
            const mx = event.clientX - rect.left - mini.x;
            const my = event.clientY - rect.top - mini.y;
            if (mx >= 0 && my >= 0 && mx <= mini.w && my <= mini.h) {
              const wx = (mx / mini.w) * world.width;
              const wy = (my / mini.h) * world.height;
              const { scale } = view.current;
              view.current = { scale, tx: c.clientWidth / 2 - wx * scale, ty: c.clientHeight / 2 - wy * scale, fitted: true };
              return;
            }
          }
          drag.current = { x: event.clientX, y: event.clientY, moved: false };
          (event.target as HTMLElement).setPointerCapture(event.pointerId);
        }}
        onPointerMove={(event) => {
          const d = drag.current;
          if (d) {
            const dx = event.clientX - d.x;
            const dy = event.clientY - d.y;
            if (Math.abs(dx) + Math.abs(dy) > 2) d.moved = true;
            if (d.moved) {
              view.current = { ...view.current, tx: view.current.tx + dx, ty: view.current.ty + dy };
              d.x = event.clientX;
              d.y = event.clientY;
              onHover(null);
              return;
            }
          }
          onHover(locate(event));
        }}
        onPointerUp={(event) => {
          const d = drag.current;
          drag.current = null;
          if (d && !d.moved) {
            const hit = locate(event);
            if (hit) onSelect(hit.index);
          }
        }}
        onPointerLeave={() => onHover(null)}
      />
    </div>
  );
}
