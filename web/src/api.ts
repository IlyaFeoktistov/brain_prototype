// Клиент HTTP API мозга (brain/api.py). Любой другой фронт может
// говорить с мозгом так же — это обычные OpenAI/llama.cpp-ручки.

import type { ReplyEvent } from "./brain";

async function post<T>(path: string, body: unknown): Promise<T> {
  const response = await fetch(path, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    const detail = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(detail.detail ?? response.statusText);
  }
  return response.json() as Promise<T>;
}

interface Chunk {
  choices: { delta: { content?: string }; finish_reason: string | null }[];
  usage?: { prompt_tokens: number; completion_tokens: number };
  brain: {
    confidence: number | number[];
    text?: string;
    dopamine?: number;
    guessed?: number;
    predicted?: number;
    prompt_seconds?: number;
    generate_seconds?: number;
  };
}

// POST /v1/chat/completions со stream: слова приходят по мере выбора.
export async function chat(
  text: string,
  onToken: (token: string, confidence: number) => void,
): Promise<{ text: string; reply: ReplyEvent }> {
  const response = await fetch("/v1/chat/completions", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ stream: true, messages: [{ role: "user", content: text }] }),
  });
  if (!response.ok || !response.body) throw new Error(response.statusText);

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let final: Chunk | null = null;

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let cut;
    while ((cut = buffer.indexOf("\n\n")) >= 0) {
      const event = buffer.slice(0, cut).replace(/^data: /, "");
      buffer = buffer.slice(cut + 2);
      if (event === "[DONE]") continue;
      const chunk = JSON.parse(event) as Chunk;
      const choice = chunk.choices[0];
      if (choice.finish_reason) final = chunk;
      else if (choice.delta.content) onToken(choice.delta.content.trim(), chunk.brain.confidence as number);
    }
  }

  if (!final) throw new Error("Мозг не закончил ответ");
  const brain = final.brain;
  return {
    text: brain.text ?? "",
    reply: {
      type: "reply",
      text: brain.text ?? "",
      promptTokens: final.usage?.prompt_tokens ?? 0,
      promptSeconds: brain.prompt_seconds ?? 0,
      generated: final.usage?.completion_tokens ?? 0,
      generateSeconds: brain.generate_seconds ?? 0,
      hits: brain.guessed ?? 0,
      predicted: brain.predicted ?? 0,
      dopamine: brain.dopamine ?? 0,
      confidence: (brain.confidence as number[]) ?? [],
    },
  };
}

export const command = (text: string) =>
  post<{ output: { kind: string; text: string }[] }>("/api/command", { text });

// В фон: мозг читает порциями между сообщениями, разговаривать можно.
export const read = (name: string, text: string) =>
  post<{ queued: boolean; sentences: number }>("/api/read", { name, text, background: true });

export const feedback = (reward: number) => post<{ text: string }>("/api/feedback", { reward });

export const load = (name: string) => post<{ brain: string; neurons: number }>("/api/load", { name });

export const setEmotions = (enabled: boolean) => post("/api/emotions", { enabled });

export const setLive = (enabled: boolean) => post("/api/live", { enabled });

export const stop = () => post("/api/stop", {});
