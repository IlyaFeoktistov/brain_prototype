// Протокол WebSocket — см. brain/server.py.

export interface Population {
  start: number;
  size: number;
}

export interface Area {
  name: string;
  title: string;
  kind: "cortex" | "memory";
  E: Population;
  I: Population;
}

export interface Pathway {
  src: string;
  dst: string;
  plastic: boolean;
  synapses: number;
}

export interface Anatomy {
  type: "anatomy";
  device: string;
  neurons: number;
  synapses: number;
  areas: Area[];
  pathways: Pathway[];
}

export interface PlasticStatus {
  name: string;
  synapses: number;
  learned: number;
  mean: number;
}

export interface EmotionState {
  name: string;
  title: string;
  face: string;
  valence: number;
  arousal: number;
  confusion: number;
  sleeping: boolean;
}

export interface EmotionEvent extends EmotionState {
  type: "emotion";
}

export interface Status {
  type: "status";
  emotionsEnabled: boolean;
  emotion: EmotionState;
  sleeping: boolean;
  sleepPressure: number;
  episodes: number;
  thinking: boolean;
  initiative: boolean;
  live: boolean;
  // Очередь фонового чтения.
  reading: { name: string; done: number; total: number }[];
  busy: boolean;
  words: number;
  tokens: number;
  dialogTurns: number;
  timeMs: number;
  grown: number;
  plastic: PlasticStatus[];
}

export interface Spikes {
  type: "spikes";
  t: number;
  chunks: number[][];
}

export interface StepEvent {
  type: "step";
  token: string;
  expected: string | null;
  error: number | null;
  dopamine: number;
  learn: boolean;
  // Слово проигрывается во сне (гиппокамп -> кора).
  dream: boolean;
  // Мысль в тишине (живой режим) — не запоминается.
  musing: boolean;
}

export interface Out {
  type: "out";
  kind: "brain" | "info" | "progress";
  text: string;
}

export interface ReplyEvent {
  type: "reply";
  text: string;
  promptTokens: number;
  promptSeconds: number;
  generated: number;
  generateSeconds: number;
  hits: number;
  predicted: number;
  dopamine: number;
  // Уверенность каждого слова ответа (0..1), выученная мозгом из опыта.
  confidence: number[];
}

export interface SpeakEvent {
  type: "speak";
  token: string;
  confidence: number;
}

export interface BrainFile {
  name: string;
  megabytes: number;
  modified: number;
}

export interface Brains {
  type: "brains";
  current: string;
  items: BrainFile[];
}

export interface MusingEvent {
  type: "musing";
  text: string;
}

// Шаг размышления: слово-кандидат и вердикт проверяющего.
export interface ThoughtEvent {
  type: "thought";
  thought: string;
  verdict: "ok" | "no" | "stop";
  score: number;
}

export interface InitiativeEvent {
  type: "initiative";
  reason: string;
  text: string;
}

export type ServerMessage =
  | Anatomy
  | Status
  | Spikes
  | StepEvent
  | Out
  | ReplyEvent
  | SpeakEvent
  | Brains
  | EmotionEvent
  | InitiativeEvent
  | MusingEvent
  | ThoughtEvent;

export type ClientMessage =
  | { type: "line"; text: string }
  | { type: "read_text"; name: string; text: string }
  | { type: "stop" };

export interface Connection {
  send: (message: ClientMessage) => void;
  close: () => void;
}

export function connect(
  onMessage: (message: ServerMessage) => void,
  onState: (open: boolean) => void,
): Connection {
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

  return {
    send: (message) => {
      if (socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify(message));
    },
    close: () => {
      closed = true;
      window.clearTimeout(retry);
      socket?.close();
    },
  };
}

// GET /api/neuron/{index} — карточка нейрона (brain/neuron.py).
export interface SynapseGroup {
  population: string; // "association.E"
  count: number;
  mean: number;
  plastic: number;
  stronger: number;
  weaker: number;
}

export interface NeuronCard {
  index: number;
  area: string;
  title: string;
  type: "E" | "I";
  local: number;
  state: { v: number; bias: number; adapt: number; rate: number | null };
  tuning?: { word: string; share: number }[];
  tuningWords?: number;
  feeling?: string | null;
  predicts: { word: string; score: number }[];
  wokenBy: { word: string; score: number }[];
  incoming: SynapseGroup[];
  outgoing: SynapseGroup[];
}
