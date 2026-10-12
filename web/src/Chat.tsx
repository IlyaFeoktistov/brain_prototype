import { useEffect, useRef, useState, type ReactNode } from "react";
import type { ReplyEvent } from "./brain";
import { Splitter, useStoredSize } from "./Splitter";

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
  // Мозг заговорил сам; повод — эмоция («любопытство», «скука»…).
  initiative?: string;
}

interface Props {
  header?: ReactNode;
  // Окошко «Мысли»: ход размышления, отдельно от разговора.
  thoughts?: { node: ReactNode; count: number; active: boolean; clear: () => void };
  entries: ChatEntry[];
  progress: string | null;
  busy: boolean;
  reading: boolean;
  sleeping: boolean;
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
  sleep: () => void;
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

const THOUGHTS_MIN = 60;
const thoughtsMax = () => Math.max(THOUGHTS_MIN, window.innerHeight * 0.6);

export function Chat({ header, thoughts, entries, progress, busy, reading, sleeping, connected, actions }: Props) {
  const [draft, setDraft] = useState("");
  const [thoughtsHeight, setThoughtsHeight] = useStoredSize("brain.thoughtsHeight", 160);
  const [thoughtsOpen, setThoughtsOpen] = useState(() => {
    try {
      return localStorage.getItem("brain.thoughtsOpen") !== "0";
    } catch {
      return true;
    }
  });
  const toggleThoughts = () => {
    setThoughtsOpen((open) => {
      try {
        localStorage.setItem("brain.thoughtsOpen", open ? "0" : "1");
      } catch {
        // не запомнится — не страшно
      }
      return !open;
    });
  };
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
      {thoughts && (
        <div className={`thoughts${thoughtsOpen ? "" : " thoughts-closed"}`}>
          <div className="thoughts-bar">
            <button type="button" className="thoughts-toggle" onClick={toggleThoughts} aria-expanded={thoughtsOpen}>
              <span aria-hidden="true">{thoughtsOpen ? "▾" : "▸"}</span> Мысли
              {thoughts.count > 0 && <span className="thoughts-count">{thoughts.count}</span>}
              {thoughts.active && <span className="thoughts-live" title="думает: живой режим или /think on" />}
            </button>
            {thoughtsOpen && thoughts.count > 0 && (
              <button type="button" className="thoughts-clear" onClick={thoughts.clear}>
                очистить
              </button>
            )}
          </div>
          {thoughtsOpen && (
            <>
              <div className="thoughts-body" style={{ height: thoughtsHeight }}>
                {thoughts.node}
              </div>
              <Splitter
                axis="y"
                size={thoughtsHeight}
                min={THOUGHTS_MIN}
                max={thoughtsMax}
                direction={1}
                onResize={setThoughtsHeight}
                label="Высота окошка мыслей"
              />
            </>
          )}
        </div>
      )}
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
            {entry.initiative && <small>заговорил сам: {entry.initiative}</small>}
            {entry.reply?.praised ? (
              <small title="слова сами стали наградой — выучено миндалиной на кнопках">
                {entry.reply.praised > 0 ? "понял похвалу — закрепил прошлый ответ" : "понял, что ругают — ослабил прошлый ответ"}
              </small>
            ) : null}
            {entry === lastBrain && !entry.streaming && !entry.rated && !entry.initiative && entry.text !== "*Промолчал*" && !busy && (
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
          <button type="button" onClick={actions.sleep} disabled={!connected || busy || sleeping}>
            Уснуть
          </button>
          {(busy || reading || sleeping) && (
            <button type="button" className="stop" onClick={actions.stop}>
              {sleeping ? "Разбудить" : reading ? "Остановить чтение" : "Остановить"}
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
