// Протокол сервера (brain/server.py): WebSocket — события мозга, HTTP — действия.

export interface Population {
  label: string; // «L4.E», «L23.PV»… у колонок; «E», «I» у ядер
  start: number;
  size: number;
  kind: string; // E, PV, SOM, VIP, LAMP5, I
}

export interface Area {
  name: string;
  title: string;
  system: string;
  layered: boolean;
  start: number;
  size: number;
  populations: Population[];
}

export interface Pathway {
  src: string;
  dst: string;
  plastic: boolean;
  modulator: string | null;
  synapses: number;
}

export interface Anatomy {
  type: "anatomy";
  device: string;
  neurons: number;
  synapses: number;
  systems: string[];
  keyboard: string[]; // ряды клавиш мира; клавиши по порядку — как у руки
  areas: Area[];
  pathways: Pathway[];
}

export interface Substance {
  name: string;
  level: number;
  occupancy: number;
  // Занятость наяву в покое (частота ядра по записям); у аденозина нет.
  rest: number | null;
}

export interface Status {
  type: "status";
  loaded: boolean;
  busy: string | null;
  live: boolean;
  brain?: string;
  timeMs?: number;
  hour?: number;
  asleep?: boolean;
  chemistry?: Substance[];
}

export interface SavedBrain {
  name: string;
  megabytes: number;
  modified: number;
  size?: string;
  neurons?: number;
  synapses?: number;
  timeMs?: number;
}

export interface Brains {
  type: "brains";
  current: string | null;
  items: SavedBrain[];
  minNeurons: number;
}

export interface Spikes {
  type: "spikes";
  t: number;
  chunks: number[][];
}

export interface Screen {
  type: "screen";
  line: string;
  gaze: number;
  pointer: number | null;
}

export interface Typed {
  type: "typed";
  char: string;
}

export interface Info {
  type: "info";
  text: string;
}

// Фиксация глаза (слово под взглядом) или буква, напечатанная рукой; пик
// дофамина за это время — насколько занятость выше покоя.
export interface Fixation {
  type: "fixation";
  t: number;
  word: string;
  letter: number;
  typed: boolean;
  dopamine: number;
}

// Нажатие руки: из чего выбирала — доля клеток движения у четырёх лучших клавиш.
export interface Press {
  type: "press";
  t: number;
  char: string;
  reply: boolean;
  choices: [string, number][];
}

// Глаз и рука (brain/view.py), ~5 Гц: картинки rows × cols по строкам.
export interface View {
  type: "view";
  rows: number;
  cols: number;
  sideCols: number;
  picture: number[]; // яркость 0–1, 1 — фон
  side: number[]; // боковое поле, rows × sideCols
  retina: number[]; // ON − OFF, −1…1
  lgn: number[]; // спайки реле по пикселям: + ON, − OFF
  v1: number[]; // спайки L4 по пикселям своих подполей
  keys: number[]; // доля выстреливших клеток движения клавиши
  held: number | null;
}

export type ServerMessage = Anatomy | Status | Brains | Spikes | Screen | Typed | Info | Fixation | Press | View;

// ---- карточки (GET /api/neuron, /api/synapse) ----

export interface SynapseInfo {
  id: number;
  path: string;
  pre: number;
  preName: string;
  post: number;
  postName: string;
  weight: number;
  psp: number;
  plastic: boolean;
  modulator: string | null;
  achSuppressed: boolean;
  dendrite: boolean;
  branch: number | null;
  delayMs: number;
  birthPsp?: number;
  learnedPsp?: number;
  earlyPsp?: number;
  eligibility?: number;
  capPsp?: number;
}

export interface SynapseGroup {
  path: string;
  population: string | null;
  count: number;
  pspMean: number;
  pspMin?: number;
  pspMax?: number;
  plastic: boolean;
  modulator?: string | null;
  dendrite?: boolean;
  achSuppressed?: boolean;
  learnedPsp?: number;
  stronger?: number;
  weaker?: number;
  eligibility?: number;
  sensory?: boolean;
  procedural?: boolean;
}

export interface NeuronInfo {
  index: number;
  area: string;
  areaTitle: string;
  population: string;
  layer: string | null;
  kind: string;
  inhibitory: boolean;
  local: number;
  model: {
    type: string;
    tauM: number;
    tauSyn: number;
    deltaT: number;
    adaptB: number;
    vRest: number;
    vThreshold: number;
    vReset: number;
    gapMv: number;
    refractoryMs: number;
    tCurrent: boolean;
    stp?: { to: string; U: number; u: number; x: number }[];
  };
  state: {
    v: number;
    iSyn: number;
    iInh: number;
    iNgf: number;
    iNmda: number | null;
    adapt: number;
    refractory: number;
    bias: number;
    tracePre: number;
    tracePost: number;
    resource: number;
    inhGain: number;
    tGate: number | null;
    spikesInWindow: number;
  };
  receptors: { substance: string; occupancy: number; mvFull: number; mvNow: number }[];
  background: { inputsPerStep: number; gain: number; mvMean: number };
  meanField: { source: string; mv: number; slow: boolean; ach: boolean }[];
  dendrite: {
    branches: { current: number; plateau: number }[];
    thresholdMv: number;
    slopeMv: number;
    gain: number;
    somaMv: number;
  };
  incoming: SynapseGroup[];
  outgoing: SynapseGroup[];
  topIncoming: SynapseInfo[];
  topOutgoing: SynapseInfo[];
}

export interface Summary {
  neurons: number;
  synapses: number;
  plastic: number;
  timeMs: number;
  areas: { name: string; title: string; cells: number; spikes: number }[];
  paths: { path: string; src: string; dst: string; synapses: number; modulator: string; stronger: number; weaker: number; learnedPsp: number }[];
}

export interface Health {
  ms: number;
  nuclei: { name: string; title: string; hz: number; target: number | null }[];
  columns: { name: string; title: string; layers: { layer: string; kind: string; hz: number; target: number | null }[] }[];
}

// ---- соединение ----

export function connect(onMessage: (message: ServerMessage) => void, onState: (open: boolean) => void): () => void {
  let socket: WebSocket | null = null;
  let closed = false;
  let retry: number | undefined;

  const open = () => {
    const scheme = location.protocol === "https:" ? "wss" : "ws";
    socket = new WebSocket(`${scheme}://${location.host}/ws`);
    socket.onopen = () => onState(true);
    socket.onmessage = (event) => onMessage(JSON.parse(event.data) as ServerMessage);
    socket.onclose = () => {
      onState(false);
      if (!closed) retry = window.setTimeout(open, 1500);
    };
  };

  open();

  return () => {
    closed = true;
    window.clearTimeout(retry);
    socket?.close();
  };
}

// Названия веществ для людей.
export const SUBSTANCE: Record<string, string> = {
  dopamine: "дофамин",
  noradrenaline: "норадреналин",
  adenosine: "аденозин",
  galanin: "галанин (VLPO, сон)",
  serotonin: "серотонин",
  acetylcholine: "ацетилхолин",
  orexin: "орексин",
};

// Что вещество делает в модели и откуда оно.
export const SUBSTANCE_HINT: Record<string, string> = {
  dopamine: "VTA. Залп — «лучше, чем ждал», провал — «хуже»; третий фактор обучения: превращает след STDP в изменение веса (D1 — залп, D2 — провал).",
  noradrenaline: "Голубое пятно (LC). Бодрствование и настороженность; держит мозг наяву против VLPO.",
  adenosine: "Обмен веществ всех нейронов. Копится наяву (давление сна), вымывается во сне.",
  galanin: "VLPO гипоталамуса. Ядро сна: когда его выброс больше норадреналина — мозг спит.",
  serotonin: "Дорсальное ядро шва. Наяву выше, во сне ниже; меняет возбудимость коры.",
  acetylcholine: "Ядро Мейнерта. Внимание и запись нового: третий фактор в гиппокампе.",
  orexin: "Латеральный гипоталамус. Стабилизирует бодрствование, поддерживает LC.",
};

export const KIND_TITLE: Record<string, string> = {
  E: "главная (возбуждающая)",
  PV: "PV — быстрая тормозная",
  SOM: "SOM — тормозная дендритов",
  VIP: "VIP — тормозит тормозных",
  LAMP5: "LAMP5 — медленное объёмное торможение",
  I: "местная тормозная",
};

export function bioTime(ms: number): string {
  const s = ms / 1000;
  return s < 120 ? `${s.toFixed(1)} с` : s < 7200 ? `${(s / 60).toFixed(1)} мин` : `${(s / 3600).toFixed(1)} ч`;
}

export function hourText(hour: number): string {
  const h = Math.floor(hour);
  const m = Math.floor((hour - h) * 60);
  return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}`;
}
