import type { Anatomy, EmotionState, Status } from "./brain";

interface Props {
  anatomy: Anatomy;
  status: Status | null;
  activity: Record<string, number>;
  emotion: EmotionState | null;
}

// Схема проводящих путей: ухо -> зоны -> рот.
const POSITION: Record<string, [number, number]> = {
  ear: [30, 70],
  auditory: [120, 70],
  association: [250, 70],
  prefrontal: [360, 28],
  motor: [360, 112],
  mouth: [470, 136],
  limbic: [250, 136],
  // Ядра нейромедиаторов: действуют химически, путей у них нет.
  vta: [36, 136],
  lc: [128, 136],
  amygdala: [420, 70],
};

// Подписи сбоку: справа от схемы и у нижнего ряда.
const SIDE = new Set(["limbic", "vta", "lc", "amygdala"]);

const SHORT: Record<string, string> = {
  auditory: "слух",
  association: "смысл",
  prefrontal: "раб. память",
  motor: "речь",
  limbic: "чувства",
  vta: "дофамин",
  lc: "норадр.",
  amygdala: "миндалина",
};

function place(name: string, index: number): [number, number] {
  return POSITION[name] ?? [80 + index * 90, 150];
}

export function Pathways({ anatomy, status, activity, emotion }: Props) {
  const index = new Map(anatomy.areas.map((a, i) => [a.name, i]));
  const learned = new Map(status?.plastic.map((p) => [p.name, p]) ?? []);

  const edge = (from: [number, number], to: [number, number], bend = 0) => {
    const mx = (from[0] + to[0]) / 2;
    const my = (from[1] + to[1]) / 2 + bend;
    return `M${from[0]},${from[1]} Q${mx},${my} ${to[0]},${to[1]}`;
  };

  return (
    <svg className="pathways" viewBox="0 0 500 160" role="img" aria-label="Схема проводящих путей между зонами">
      <defs>
        <marker id="tip" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto">
          <path d="M0,0 L10,5 L0,10 z" className="tip" />
        </marker>
        <marker id="tip-plastic" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto">
          <path d="M0,0 L10,5 L0,10 z" className="tip-plastic" />
        </marker>
      </defs>

      <path d={edge(POSITION.ear, place("auditory", 0))} className="path" markerEnd="url(#tip)" />
      {index.has("motor") && (
        <path d={edge(place("motor", 0), POSITION.mouth)} className="path" markerEnd="url(#tip)" />
      )}

      {anatomy.pathways.map((p) => {
        const from = place(p.src, index.get(p.src) ?? 0);
        const to = place(p.dst, index.get(p.dst) ?? 0);
        const backwards = from[0] > to[0];
        const width = Math.max(1, Math.log10(p.synapses) - 3);
        const stats = learned.get(`${p.src} -> ${p.dst}`);
        const share = stats ? stats.learned / stats.synapses : 0;
        return (
          <g key={`${p.src}-${p.dst}`}>
            <path
              d={edge(from, to, backwards ? 26 : -8)}
              className={p.plastic ? "path path-plastic" : "path"}
              strokeWidth={width}
              markerEnd={p.plastic ? "url(#tip-plastic)" : "url(#tip)"}
            >
              <title>
                {`${p.src} → ${p.dst}: ${p.synapses.toLocaleString("ru")} синапсов` +
                  (p.plastic ? `, учится, выучено ${(share * 100).toFixed(1)}%` : ", врождённый")}
              </title>
            </path>
          </g>
        );
      })}

      {anatomy.areas.map((area, i) => {
        const [x, y] = place(area.name, i);
        const level = Math.min(1, (activity[area.name] ?? 0) * 12);
        const local = learned.get(`${area.name}: E->E`);
        return (
          <g key={area.name} transform={`translate(${x},${y})`}>
            <circle r={13 + level * 6} className="node-halo" style={{ opacity: level * 0.3 }} />
            <circle
              r={area.name === "vta" || area.name === "lc" || area.name === "amygdala" ? 8 : 13}
              className={area.kind === "memory" ? "node node-memory" : "node"}
            />
            {local && <circle r={16} className="node-plastic" />}
            <text
              x={x >= 350 || SIDE.has(area.name) ? 20 : 0}
              y={x >= 350 || SIDE.has(area.name) ? 4 : 30}
              className={x >= 350 || SIDE.has(area.name) ? "node-label node-label-side" : "node-label"}
            >
              {area.name === "limbic" && emotion ? `чувства: ${emotion.name}` : SHORT[area.name] ?? area.name}
            </text>
          </g>
        );
      })}

      {/* Старые мозги без лимбической зоны: эмоции — модель, пунктир. */}
      {!index.has("limbic") && (
        <>
          <path d={edge(POSITION.limbic, place("association", 0), 0)} className="path path-limbic" />
          <g transform={`translate(${POSITION.limbic[0]},${POSITION.limbic[1]})`}>
            <circle r={10} className={emotion ? "node node-limbic" : "node node-limbic node-off"} />
            <text x={18} y={4} className="node-label node-label-side">
              {emotion ? `эмоции: ${emotion.name}` : "эмоции выкл."}
            </text>
          </g>
        </>
      )}

      <text x={POSITION.ear[0]} y={POSITION.ear[1] + 4} className="endpoint">
        ухо
      </text>
      {index.has("motor") && (
        <text x={POSITION.mouth[0]} y={POSITION.mouth[1] + 4} className="endpoint">
          рот
        </text>
      )}
    </svg>
  );
}
