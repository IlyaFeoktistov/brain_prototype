import { useCallback, useEffect, useState } from "react";
import * as api from "./api";
import type { Anatomy, Summary } from "./brain";
import { SUBSTANCE } from "./brain";

// Схема проводящих путей: глаз и ухо → таламус → кора → смысл → рука →
// клавиатура; ниже — таламус движений, базальные ганглии, гиппокамп,
// мозжечок, миндалина и ядра веществ. Толщина пути — число синапсов,
// цветной — учится (доля изменившихся синапсов — в подсказке, из сводки),
// серый — врождённый. Ореол зоны — доля её клеток, горящих в микроскопе.

const POSITION: Record<string, [number, number]> = {
  // зрение
  lgn: [90, 50], trn: [90, 100], visual: [170, 50], letters: [250, 50], colliculus: [170, 110],
  // слух
  cn: [70, 190], vnll: [110, 230], ic: [130, 180], mgn: [190, 180], trn_mgn: [190, 230], auditory: [250, 180],
  // смысл и лобная кора
  association: [340, 115], category: [340, 40], limbic: [340, 190], prefrontal: [430, 70], goal: [430, 20],
  md: [430, 135], trn_md: [480, 160],
  // движение
  premotor: [530, 80], motor: [530, 170], hand: [620, 120], thalamus: [620, 200], trn_motor: [680, 200], vl_cb: [570, 235],
  // базальные ганглии
  fsi: [330, 300], striatum_d1: [390, 285], striatum_d2: [390, 325], gpe: [460, 325], stn: [520, 325], gpi: [520, 280],
  // гиппокамп
  entorhinal: [200, 290], dentate: [150, 320], ca3: [200, 345], ca1: [250, 320],
  // мозжечок
  granule: [620, 285], purkinje: [690, 285], dcn: [690, 335], io: [620, 335],
  // миндалина и награда, модуляторы и сон
  amygdala: [430, 230], cea: [470, 255],
  vta: [70, 420], lhb: [140, 420], lc: [220, 420], raphe: [290, 420], nbm: [360, 420],
  orexin: [430, 420], vlpo: [500, 420], scn: [570, 420], dmh: [640, 420],
};

// Концы: органы и клавиатура (вне мозга).
const ENDS: { name: string; at: [number, number]; to: string[]; reverse?: boolean }[] = [
  { name: "глаз", at: [20, 75], to: ["lgn", "colliculus"] },
  { name: "ухо", at: [20, 190], to: ["cn"] },
  { name: "вкус", at: [20, 380], to: ["vta", "lhb"] },
  { name: "клавиатура", at: [680, 120], to: ["hand"], reverse: true },
];

const SHORT: Record<string, string> = {
  lgn: "LGN", trn: "TRN", visual: "V1", letters: "буквы", colliculus: "двухолмие",
  cn: "улит. ядро", vnll: "VNLL", ic: "IC", mgn: "MGN", trn_mgn: "TRN слуха", auditory: "A1",
  association: "смысл", category: "категории", limbic: "чувства", prefrontal: "раб. память", goal: "цель",
  md: "MD", trn_md: "TRN MD", premotor: "премотор", motor: "речь", hand: "рука", thalamus: "VL",
  trn_motor: "TRN дв.", vl_cb: "VL мозж.", fsi: "FSI", striatum_d1: "D1 «иди»", striatum_d2: "D2 «стой»",
  gpe: "GPe", stn: "STN", gpi: "GPi", entorhinal: "энторин.", dentate: "зубчатая", ca3: "CA3", ca1: "CA1",
  granule: "зернистые", purkinje: "Пуркинье", dcn: "ядра мозж.", io: "олива", amygdala: "миндалина", cea: "CeA",
  vta: "VTA дофамин", lhb: "LHb", lc: "LC норадр.", raphe: "шов серот.", nbm: "Мейнерт ACh", orexin: "орексин",
  vlpo: "VLPO сон", scn: "часы SCN", dmh: "DMH",
};

// Зона без места на схеме — справа внизу, в кадре.
const place = (name: string, i: number): [number, number] => POSITION[name] ?? [740, 260 + (i % 6) * 25];

const curve = (a: [number, number], b: [number, number], bend: number) => {
  const mx = (a[0] + b[0]) / 2;
  const my = (a[1] + b[1]) / 2 + bend;
  return `M${a[0]},${a[1]} Q${mx},${my} ${b[0]},${b[1]}`;
};

export function Pathways({ anatomy, activity }: { anatomy: Anatomy; activity: Record<string, number> }) {
  const [summary, setSummary] = useState<Summary | null>(null);
  const [error, setError] = useState<string | null>(null);
  const refresh = useCallback(() => api.summary().then(setSummary).then(() => setError(null)).catch((e) => setError((e as Error).message)), []);
  useEffect(() => { refresh(); }, [refresh, anatomy]);

  const index = new Map(anatomy.areas.map((a, i) => [a.name, i]));
  // Доля выученного по паре зон: изменившиеся синапсы / все синапсы путей.
  const learned = new Map<string, { changed: number; total: number }>();
  for (const p of summary?.paths ?? []) {
    const key = `${p.src}>${p.dst}`;
    const old = learned.get(key) ?? { changed: 0, total: 0 };
    learned.set(key, { changed: old.changed + p.stronger + p.weaker, total: old.total + p.synapses });
  }
  const self = new Set(anatomy.pathways.filter((p) => p.src === p.dst && p.plastic).map((p) => p.src));

  return (
    <figure className="pathways-figure">
      <svg className="pathways" viewBox="0 0 770 450" role="img" aria-label="Схема проводящих путей между зонами">
        <defs>
          <marker id="tip" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto">
            <path d="M0,0 L10,5 L0,10 z" className="arrow" />
          </marker>
          <marker id="tip-plastic" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto">
            <path d="M0,0 L10,5 L0,10 z" className="arrow-plastic" />
          </marker>
        </defs>

        {ENDS.map((end) =>
          end.to.filter((name) => index.has(name)).map((name) => {
            const at = place(name, index.get(name)!);
            return <path key={end.name + name} d={end.reverse ? curve(at, end.at, 0) : curve(end.at, at, 0)} className="path path-organ" markerEnd="url(#tip)" />;
          }),
        )}

        {anatomy.pathways.filter((p) => p.src !== p.dst && index.has(p.src) && index.has(p.dst)).map((p) => {
          const from = place(p.src, index.get(p.src)!);
          const to = place(p.dst, index.get(p.dst)!);
          const width = Math.max(0.6, Math.log10(Math.max(10, p.synapses)) - 3);
          const stats = learned.get(`${p.src}>${p.dst}`);
          const share = stats && stats.total ? stats.changed / stats.total : null;
          return (
            <path key={`${p.src}-${p.dst}`} d={curve(from, to, from[0] > to[0] ? 18 : -10)}
                  className={p.plastic ? "path path-plastic" : "path"} strokeWidth={width}
                  markerEnd={p.plastic ? "url(#tip-plastic)" : "url(#tip)"}>
              <title>
                {`${p.src} → ${p.dst}: ${p.synapses.toLocaleString("ru")} синапсов` +
                  (p.plastic ? `, учится (третий фактор — ${SUBSTANCE[p.modulator ?? "dopamine"] ?? p.modulator})` + (share === null ? "" : `, изменилось ${(share * 100).toFixed(1)}%`) : ", врождённый")}
              </title>
            </path>
          );
        })}

        {anatomy.areas.map((area, i) => {
          const [x, y] = place(area.name, i);
          const level = Math.min(1, (activity[area.name] ?? 0) * 12);
          const big = area.layered;
          return (
            <g key={area.name} transform={`translate(${x},${y})`}>
              <title>{`${area.title} (${area.name}), ${area.size.toLocaleString("ru")} клеток`}</title>
              <circle r={(big ? 12 : 7) + level * 7} className="node-halo" style={{ opacity: level * 0.45 }} />
              <circle r={big ? 12 : 7} className={big ? "node node-cortex" : "node"} />
              {self.has(area.name) && <circle r={big ? 15 : 10} className="node-plastic" />}
              <text y={big ? 25 : 18} className="node-label">{SHORT[area.name] ?? area.name}</text>
            </g>
          );
        })}

        {ENDS.map((end) => (
          <text key={end.name} x={end.at[0]} y={end.at[1] + 4} className="endpoint" textAnchor={end.reverse ? "start" : "middle"}>
            {end.name}
          </text>
        ))}
      </svg>
      <figcaption className="muted">
        Цветные пути учатся, серые — врождённые; толщина — число синапсов; кольцо — зона учится сама на себя.
        Ореол — доля горящих клеток. {summary ? `Выученное — по сводке на ${(summary.timeMs / 1000).toFixed(1)} с жизни.` : error ? `Сводка: ${error}.` : "Сводка грузится…"}{" "}
        <button type="button" className="link" onClick={refresh}>обновить</button>
      </figcaption>
    </figure>
  );
}
