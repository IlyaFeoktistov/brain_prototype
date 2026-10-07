import { useEffect, useRef, useState } from "react";
import type { ClientMessage, ReplyEvent } from "./brain";

export interface ChatEntry {
  id: number;
  who: "you" | "brain" | "info";
  text: string;
  reply?: ReplyEvent;
}

interface Props {
  entries: ChatEntry[];
  progress: string | null;
  busy: boolean;
  connected: boolean;
  send: (message: ClientMessage) => void;
  onLocal: (entry: Omit<ChatEntry, "id">) => void;
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

export function Chat({ entries, progress, busy, connected, send, onLocal }: Props) {
  const [draft, setDraft] = useState("");
  const log = useRef<HTMLDivElement>(null);
  const files = useRef<HTMLInputElement>(null);

  useEffect(() => {
    log.current?.scrollTo({ top: log.current.scrollHeight });
  }, [entries, progress]);

  const submit = (text: string) => {
    const line = text.trim();
    if (!line) return;
    onLocal({ who: "you", text: line });
    send({ type: "line", text: line });
    setDraft("");
  };

  const upload = async (list: FileList | null) => {
    if (!list) return;
    for (const file of Array.from(list)) {
      const text = await file.text();
      onLocal({ who: "you", text: `Загружаю «${file.name}» (${(file.size / 1024).toFixed(0)} КБ)` });
      send({ type: "read_text", name: file.name, text });
    }
    if (files.current) files.current.value = "";
  };

  const lastBrain = [...entries].reverse().find((e) => e.who === "brain");

  return (
    <section className="chat" aria-label="Разговор с мозгом">
      <div className="chat-log" ref={log} aria-live="polite">
        {entries.length === 0 && (
          <p className="chat-empty">
            Напиши фразу целиком, например «Столица Франции Париж». Мозг учится на каждой
            реплике и отвечает продолжением мысли. Похвали удачный ответ кнопкой «Верно».
          </p>
        )}
        {entries.map((entry) => (
          <div key={entry.id} className={`line line-${entry.who}`}>
            {entry.who === "info" ? <pre>{entry.text}</pre> : <p>{entry.text}</p>}
            {entry.reply && entry.reply.promptTokens > 0 && <small>{speed(entry.reply)}</small>}
            {entry === lastBrain && entry.text !== "..." && !busy && (
              <div className="feedback">
                <button type="button" onClick={() => submit("+")}>
                  Верно
                </button>
                <button type="button" onClick={() => submit("-")}>
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
            <button type="button" className="stop" onClick={() => send({ type: "stop" })}>
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
