import { useEffect, useRef, useState, type ReactNode } from "react";
import type { ReplyEvent } from "./brain";

export interface ChatEntry {
  id: number;
  who: "you" | "brain" | "info";
  text: string;
  reply?: ReplyEvent;
  // Ответ ещё произносится: слова приходят по одному.
  streaming?: boolean;
  // Слова ответа с уверенностью каждого.
  words?: { token: string; confidence: number }[];
  // Человек уже похвалил или поругал этот ответ.
  rated?: boolean;
}

interface Props {
  header?: ReactNode;
  entries: ChatEntry[];
  progress: string | null;
  busy: boolean;
  connected: boolean;
  actions: ChatActions;
}

// Что чат умеет делать с мозгом (реализация — HTTP API, см. api.ts).
export interface ChatActions {
  say: (text: string) => void;
  command: (text: string) => void;
  upload: (name: string, text: string, size: number) => void;
  feedback: (reward: number) => void;
  stop: () => void;
}

function speed(reply: ReplyEvent): string {
  const prompt = reply.promptTokens / Math.max(reply.promptSeconds, 1e-6);
  let text = `прочитал ${reply.promptTokens} ток за ${reply.promptSeconds.toFixed(2)} с (${prompt.toFixed(0)} ток/с)`;
  if (reply.generated) {
    const gen = reply.generated / Math.max(reply.generateSeconds, 1e-6);
    text += `, ответил ${reply.generated} ток за ${reply.generateSeconds.toFixed(2)} с (${gen.toFixed(0)} ток/с)`;
  }
  return text + `, угадал ${reply.hits} из ${reply.predicted}`;
}

function percent(value: number): string {
  return `${Math.round(value * 100)}%`;
}

function averageConfidence(words: { confidence: number }[]): number {
  return words.reduce((sum, w) => sum + w.confidence, 0) / words.length;
}

export function Chat({ header, entries, progress, busy, connected, actions }: Props) {
  const [draft, setDraft] = useState("");
  const log = useRef<HTMLDivElement>(null);
  const files = useRef<HTMLInputElement>(null);

  useEffect(() => {
    log.current?.scrollTo({ top: log.current.scrollHeight });
  }, [entries, progress]);

  const submit = (text: string) => {
    const line = text.trim();
    if (!line) return;
    if (line.startsWith("/")) actions.command(line);
    else actions.say(line);
    setDraft("");
  };

  const upload = async (list: FileList | null) => {
    if (!list) return;
    for (const file of Array.from(list)) {
      actions.upload(file.name, await file.text(), file.size);
    }
    if (files.current) files.current.value = "";
  };

  const lastBrain = [...entries].reverse().find((e) => e.who === "brain");

  return (
    <section className="chat" aria-label="Разговор с мозгом">
      {header}
      <div className="chat-log" ref={log} aria-live="polite">
        {entries.length === 0 && (
          <p className="chat-empty">
            Напиши фразу целиком, например «Столица Франции Париж». Мозг учится на каждой
            реплике и отвечает продолжением мысли. Похвали удачный ответ кнопкой «Верно».
            Бледные слова — те, в которых мозг не уверен; если не знает, так и скажет.
          </p>
        )}
        {entries.map((entry) => (
          <div key={entry.id} className={`line line-${entry.who}`}>
            {entry.who === "info" ? (
              <pre>{entry.text}</pre>
            ) : (
              <p>
                {entry.words?.length
                  ? entry.words.map((w, i) => (
                      <span
                        key={i}
                        className="spoken"
                        style={{ opacity: 0.45 + 0.55 * w.confidence }}
                        title={`уверенность ${percent(w.confidence)}`}
                      >
                        {i ? " " : ""}
                        {w.token}
                      </span>
                    ))
                  : entry.text}
                {entry.streaming && <span className="caret" aria-hidden="true" />}
              </p>
            )}
            {entry.who === "brain" && !entry.streaming && entry.words?.length ? (
              <small>уверен на {percent(averageConfidence(entry.words))}</small>
            ) : null}
            {entry.reply && entry.reply.promptTokens > 0 && <small>{speed(entry.reply)}</small>}
            {entry === lastBrain && !entry.streaming && !entry.rated && entry.text !== "не знаю" && !busy && (
              <div className="feedback">
                <button type="button" onClick={() => actions.feedback(1)}>
                  Верно
                </button>
                <button type="button" onClick={() => actions.feedback(-1)}>
                  Неверно
                </button>
              </div>
            )}
          </div>
        ))}
        {progress && <p className="line line-progress">{progress}</p>}
      </div>

      <form
        className="chat-input"
        onSubmit={(event) => {
          event.preventDefault();
          submit(draft);
        }}
      >
        <textarea
          value={draft}
          rows={2}
          placeholder={connected ? "Фраза или команда, /help — список команд" : "Нет связи с мозгом…"}
          onChange={(event) => setDraft(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              submit(draft);
            }
          }}
          aria-label="Сообщение мозгу"
        />
        <div className="chat-actions">
          <button type="submit" disabled={!connected || !draft.trim()}>
            Сказать
          </button>
          <button type="button" onClick={() => files.current?.click()} disabled={!connected}>
            Дать прочитать файл
          </button>
          {busy && (
            <button type="button" className="stop" onClick={actions.stop}>
              Остановить чтение
            </button>
          )}
          <input
            ref={files}
            type="file"
            multiple
            hidden
            accept=".txt,.md,.rst,.csv,.json,.html,.xml,.py,.ts,.js,text/*"
            onChange={(event) => upload(event.target.files)}
          />
        </div>
      </form>
    </section>
  );
}
