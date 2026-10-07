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

export interface Status {
  type: "status";
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
}

export type ServerMessage = Anatomy | Status | Spikes | StepEvent | Out | ReplyEvent;

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
