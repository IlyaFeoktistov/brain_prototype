// Действия с мозгом — HTTP (brain/server.py).

import type { FlagState, Health, NeuronInfo, Summary, SynapseInfo } from "./brain";

async function request<T>(method: string, path: string, body?: unknown): Promise<T> {
  const response = await fetch(path, {
    method,
    headers: body === undefined ? undefined : { "content-type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!response.ok) {
    const detail = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(detail.detail ?? response.statusText);
  }
  return response.json() as Promise<T>;
}

const post = <T>(path: string, body: unknown = {}) => request<T>("POST", path, body);
const get = <T>(path: string) => request<T>("GET", path);

export const say = (text: string) => post<{ text: string }>("/api/say", { text });
export const praise = (good: boolean) => post("/api/praise", { good });
export const showLetter = (letter: string) =>
  post<{ letter: string; seen: string; spikes: Record<string, number> }>("/api/letter", { letter });
export const correctLetter = (shown: string, right: string) =>
  post<{ shown: string; right: string; changed: number }>("/api/letter/correct", { shown, right });
export const repeat = (phrase: string) =>
  post<{ phrases: string[]; said: string[][]; leak: number; leak_windows: number }>("/api/repeat", { phrase });
export const stop = () => post("/api/stop");
export const setLive = (enabled: boolean) => post("/api/live", { enabled });
export const lesson = (kind: string, n: number, extra: { text?: string; first?: number; last?: number } = {}) =>
  post<{ text: string }>("/api/lesson", { kind, n, ...extra });

export const load = (name: string) => post<{ brain: string }>("/api/load", { name });
export const newBrain = (name: string, neurons: number, preset: string | null, seed = 1, flags: Record<string, boolean> = {},
                         organs: string[] | null = null, develop = 0) =>
  post<{ brain: string }>("/api/new", { name, neurons, preset, seed, flags, organs, develop });
export const flagState = () => get<FlagState>("/api/flags");
export const setFlag = (key: string, enabled: boolean) => post<FlagState>("/api/flag", { key, enabled });
export const save = () => post<{ brain: string }>("/api/save");
export const sizes = (neurons: number, small = false) =>
  get<{ neurons: number; total: number; min: number }>(`/api/sizes?neurons=${neurons}&small=${small}`);

export const neuron = (index: number) => get<NeuronInfo>(`/api/neuron/${index}`);
export const synapse = (id: number) => get<SynapseInfo>(`/api/synapse/${id}`);
export const summary = () => get<Summary>("/api/summary");
export const health = (ms = 500) => get<Health>(`/api/health?ms=${ms}`);
export const places = () => get<{ areas: unknown[] }>("/api/places");
