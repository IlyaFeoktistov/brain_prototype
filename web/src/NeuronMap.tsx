import { useEffect, useRef, useState } from "react";
import * as api from "./api";
import type { Anatomy, Area, Population } from "./brain";
import { type Fluorescence, GLASS, type Hover, pack, RAMPS, STEPS } from "./Microscope";

// Карта нейронов: весь мозг одним холстом с зумом. Зоны — по ходу сигнала
// слева направо (вход → выход), модуляторы — полосой снизу. Внутри зоны
// клетка лежит на своём месте, только если оно задано анатомией модели
// (brain/places.py): лист коры — по слоям, зубчатая — по длине. У ядер места
// нет — плотная группа по номеру с подписью «без места». Линии между зонами —
// только существующие пути.

// Колонки карты по ходу сигнала; зоны, которых нет в мозге, пропускаются.
const STAGES: string[][] = [
  ["lgn", "mgn", "cn", "vnll", "ic", "visual", "letters", "auditory", "colliculus"],
  ["association"],
  ["category", "limbic", "amygdala"],
  ["entorhinal", "dentate", "ca3", "ca1", "entorhinal_vb", "entorhinal_va"],
  ["prefrontal", "goal", "md", "trn_md", "trn", "trn_mgn"],
  ["striatum_d1", "striatum_d2", "fsi", "gpe", "stn", "gpi", "thalamus", "trn_motor"],
  ["premotor", "motor", "hand", "vl_cb", "granule", "purkinje", "dcn", "io"],
];
const STAGE_TITLES = ["вход (органы)", "смысловая кора — порт слов", "категории, лимбическая", "гиппокамп",
  "лобная кора", "базальные ганглии, таламус", "выход (моторная)"];

const COLUMN_GAP = 60;
const BLOCK_GAP = 34;
const TILE_GAP = 3;
const LABEL = 24; // место под подпись зоны (мир)
const BACKDROP = pack([28, 26, 34]);

interface PlaceArea {
  name: string;
  mode: "sheet" | "line" | "none";
  tiles: string[];
  tile: string;
  xy: string;
}

interface Block {
  area: Area;
  x: number;
  y: number;
  w: number;
  h: number;
  mode: PlaceArea["mode"];
  tiles: { label: string; x: number; y: number; w: number; h: number }[];
}

interface World {
  width: number;
  height: number;
  blocks: Block[];
  pixelOf: Int32Array; // клетка → пиксель мира (−1 — нет)
  neuronAt: Int32Array; // пиксель → клетка (−1 — стекло)
  ramp: (number[] | null)[];
  base: Uint32Array;
  titles: { text: string; x: number; y: number; room: number }[];
}

const decode = (b64: string) => Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));

function tileSize(n: number, mode: PlaceArea["mode"]): [number, number] {
  if (mode === "line") {
    // Длина зубчатой — вширь.
    const w = Math.max(100, Math.ceil(Math.sqrt(n * 8)));
    return [w, Math.max(4, Math.ceil((n / w) * 1.6))];
  }
  const side = Math.max(4, Math.ceil(Math.sqrt(n) * (mode === "sheet" ? 1.35 : 1)));
  return [side, mode === "sheet" ? side : Math.ceil(n / side)];
}

function kindOf(pop: Population, tileLabel: string | undefined): string {
  if (tileLabel && /Granule|Mossy/.test(tileLabel)) return "E";
  if (tileLabel && /Basket|HIPP/.test(tileLabel)) return "PV";
  return pop.kind;
}

function build(anatomy: Anatomy, places: PlaceArea[]): World {
  const byName = new Map(anatomy.areas.map((a) => [a.name, a]));
  const placeOf = new Map(places.map((p) => [p.name, p]));
  const tileOf = new Map(places.map((p) => [p.name, decode(p.tile)]));
  const blocks: Block[] = [];
  const titles: World["titles"] = [];
  const used = new Set<string>();

  const makeBlock = (area: Area, x: number, y: number): Block => {
    const place = placeOf.get(area.name);
    const mode = place?.mode ?? "none";
    const tiles: Block["tiles"] = [];
    if (mode === "none") {
      const [w, h] = tileSize(area.size, "none");
      tiles.push({ label: "", x, y: y + LABEL, w, h });
    } else {
      const ids = tileOf.get(area.name)!;
      const counts = place!.tiles.map((_, k) => ids.reduce((s, t) => s + (t === k ? 1 : 0), 0));
      let tx = x;
      let ty = y + LABEL;
      place!.tiles.forEach((label, k) => {
        const [w, h] = tileSize(counts[k], mode);
        if (mode === "line") {
          tiles.push({ label, x, y: ty, w, h });
          ty += h + TILE_GAP + 8;
        } else {
          tiles.push({ label, x: tx, y: ty, w, h });
          tx += w + TILE_GAP;
        }
      });
    }
    const right = Math.max(...tiles.map((t) => t.x + t.w));
    const bottom = Math.max(...tiles.map((t) => t.y + t.h));
    return { area, x, y, w: right - x, h: bottom - y, mode, tiles };
  };

  let x = 0;
  let tallest = 0;
  STAGES.forEach((names, s) => {
    const present = names.map((n) => byName.get(n)).filter((a): a is Area => !!a);
    if (!present.length) return;
    let y = 30;
    let wide = 0;
    const title = { text: STAGE_TITLES[s], x, y: 6, room: 0 };
    titles.push(title);
    for (const area of present) {
      const block = makeBlock(area, x, y);
      blocks.push(block);
      used.add(area.name);
      y += block.h + BLOCK_GAP;
      wide = Math.max(wide, block.w);
    }
    tallest = Math.max(tallest, y);
    title.room = wide + COLUMN_GAP * 0.8;
    x += wide + COLUMN_GAP;
  });
  const width = Math.max(x - COLUMN_GAP, 200);

  // Остальное (модуляторы, сон и прочее) — полосой снизу.
  let rx = 0;
  let ry = tallest + 30;
  titles.push({ text: "модуляторы, сон и прочее (без места)", x: 0, y: ry - 14, room: width });
  let rowH = 0;
  for (const area of anatomy.areas) {
    if (used.has(area.name)) continue;
    const probe = makeBlock(area, 0, 0);
    if (rx > 0 && rx + probe.w > width) {
      rx = 0;
      ry += rowH + BLOCK_GAP;
      rowH = 0;
    }
    const block = makeBlock(area, rx, ry);
    blocks.push(block);
    rx += block.w + BLOCK_GAP;
    rowH = Math.max(rowH, block.h);
  }
  const height = ry + rowH + 4;

  const pixelOf = new Int32Array(anatomy.neurons).fill(-1);
  const neuronAt = new Int32Array(width * height).fill(-1);
  const ramp: (number[] | null)[] = new Array(anatomy.neurons).fill(null);
  const base = new Uint32Array(width * height).fill(BACKDROP);
  const glass = pack(GLASS);

  for (const block of blocks) {
    for (const t of block.tiles)
      for (let yy = t.y; yy < t.y + t.h; yy++) base.fill(glass, yy * width + t.x, yy * width + t.x + t.w);
    const area = block.area;
    const place = placeOf.get(area.name);
    const ids = place ? tileOf.get(area.name)! : null;
    const xy = place && place.mode !== "none" ? new Uint16Array(decode(place.xy).buffer) : null;
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
          const t = block.tiles[0];
          px = t.x + (local % t.w);
          py = t.y + Math.floor(local / t.w);
        }
        const pixel = py * width + px;
        pixelOf[index] = pixel;
        neuronAt[pixel] = index;
        ramp[index] = RAMPS[kindOf(pop, label)] ?? RAMPS.I;
        base[pixel] = ramp[index]![0];
      }
    }
  }
  return { width, height, blocks, pixelOf, neuronAt, ramp, base, titles };
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
  const [paths, setPaths] = useState(true);
  const view = useRef({ scale: 1, tx: 0, ty: 0, fitted: false });
  const pathsRef = useRef(paths);
  pathsRef.current = paths;
  const selectedRef = useRef(selected);
  selectedRef.current = selected;

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
    if (!c || !world) return;
    const scale = Math.min(c.clientWidth / world.width, c.clientHeight / world.height) * 0.97;
    view.current = { scale, tx: (c.clientWidth - world.width * scale) / 2, ty: 8, fitted: true };
  };

  useEffect(() => {
    if (!world) return;
    const c = canvas.current!;
    const off = document.createElement("canvas");
    off.width = world.width;
    off.height = world.height;
    const offCtx = off.getContext("2d")!;
    const image = new ImageData(world.width, world.height);
    const pixels = new Uint32Array(image.data.buffer);
    const centres = new Map(world.blocks.map((b) => [b.area.name, b]));
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

      if (pathsRef.current) {
        // Пути — от правого края источника к левому краю получателя; яркость — доля
        // горящих клеток источника.
        ctx.lineWidth = 1 / scale;
        for (const p of anatomy.pathways) {
          const a = centres.get(p.src);
          const b = centres.get(p.dst);
          if (!a || !b || a === b) continue;
          const lit = fluo.activeIn(a.area.start, a.area.start + a.area.size) / a.area.size;
          ctx.strokeStyle = `rgba(150, 190, 255, ${Math.min(0.85, 0.12 + lit * 12)})`;
          const x1 = a.x + a.w;
          const y1 = a.y + a.h / 2;
          const x2 = b.x;
          const y2 = b.y + b.h / 2;
          ctx.beginPath();
          ctx.moveTo(x1, y1);
          ctx.bezierCurveTo(x1 + 30, y1, x2 - 30, y2, x2, y2);
          ctx.stroke();
        }
      }

      // Подписи — в экранном размере.
      const fontPx = 11 / scale;
      ctx.font = `${fontPx}px system-ui, sans-serif`;
      ctx.textBaseline = "top";
      // Подпись не шире своей зоны (полностью — при приближении).
      const fitted = (text: string, room: number) => {
        if (ctx.measureText(text).width <= room) return text;
        let lo = 0;
        let hi = text.length;
        while (lo < hi) {
          const mid = (lo + hi + 1) >> 1;
          if (ctx.measureText(text.slice(0, mid) + "…").width <= room) lo = mid;
          else hi = mid - 1;
        }
        return lo > 0 ? text.slice(0, lo) + "…" : "";
      };
      ctx.fillStyle = "#e9e4f2";
      for (const b of world.blocks) {
        const note = b.mode === "none" ? " · без места" : b.mode === "line" ? " · по длине" : " · лист";
        const room = Math.max(b.w, 40) + (b.mode === "none" ? BLOCK_GAP * 0.8 : 0);
        ctx.fillText(fitted(`${b.area.title}${note}`, room), b.x, b.y);
        if (scale > 1.5)
          for (const t of b.tiles)
            if (t.label) ctx.fillText(fitted(t.label, t.w), t.x, t.y - fontPx * 0.05 - (b.mode === "line" ? fontPx : 0));
      }
      ctx.fillStyle = "#9f97b4";
      for (const t of world.titles) ctx.fillText(fitted(t.text, t.room), t.x, t.y);

      const sel = selectedRef.current;
      if (sel !== null && world.pixelOf[sel] >= 0) {
        const p = world.pixelOf[sel];
        ctx.strokeStyle = "#ffd75e";
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

  const toWorld = (event: { clientX: number; clientY: number }) => {
    const rect = canvas.current!.getBoundingClientRect();
    const { scale, tx, ty } = view.current;
    return [(event.clientX - rect.left - tx) / scale, (event.clientY - rect.top - ty) / scale];
  };

  const locate = (event: React.PointerEvent | React.MouseEvent): Hover | null => {
    if (!world) return null;
    const [wx, wy] = toWorld(event);
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
        <span className="map-hint">колесо — зум, перетаскивание — сдвиг, клик — клетка</span>
      </div>
      {error && <p className="map-error">карта: {error}</p>}
      <canvas
        ref={canvas}
        className="map-canvas"
        aria-label="карта нейронов мозга"
        onWheel={(event) => {
          const [wx, wy] = toWorld(event);
          const v = view.current;
          const scale = Math.min(40, Math.max(0.05, v.scale * Math.exp(-event.deltaY * 0.0015)));
          const rect = canvas.current!.getBoundingClientRect();
          view.current = { scale, tx: event.clientX - rect.left - wx * scale, ty: event.clientY - rect.top - wy * scale, fitted: true };
        }}
        onPointerDown={(event) => {
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
