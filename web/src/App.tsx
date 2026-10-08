import { type CSSProperties, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { connect, type GrowEvent, type Anatomy, type Brains, type EmotionState, type Connection, type ServerMessage, type Status, type StepEvent } from "./brain";
import { Chat, type ChatActions, type ChatEntry } from "./Chat";
import * as api from "./api";
import { Fluorescence, Microscope } from "./Microscope";
import { Guide } from "./Guide";
import { Face } from "./Face";
import { Pathways } from "./Pathways";
import { Timeline } from "./Timeline";
import { Splitter, useStoredSize } from "./Splitter";
import { Thoughts, type ThoughtEntry } from "./Thoughts";

const TIMELINE_LENGTH = 60;
const THOUGHTS_LENGTH = 400;
// Чат: не уже 300 px и не шире 70% окна — срезу мозга нужно место.
const CHAT_MIN = 300;
const chatMax = () => Math.max(CHAT_MIN, window.innerWidth * 0.7);

function bioTime(ms: number): string {
  const s = ms / 1000;
  return s < 120 ? `${s.toFixed(1)} с` : `${(s / 60).toFixed(1)} мин`;
}

export function App() {
  const [anatomy, setAnatomy] = useState<Anatomy | null>(null);
  const [status, setStatus] = useState<Status | null>(null);
  const [steps, setSteps] = useState<StepEvent[]>([]);
  const [entries, setEntries] = useState<ChatEntry[]>([]);
  const [progress, setProgress] = useState<string | null>(null);
  const [activity, setActivity] = useState<Record<string, number>>({});
  const [connected, setConnected] = useState(false);
  const [brains, setBrains] = useState<Brains | null>(null);
  const [emotion, setEmotion] = useState<EmotionState | null>(null);
  const [thoughts, setThoughts] = useState<ThoughtEntry[]>([]);
  // Последний рост синапсов — микроскоп рисует дуги.
  const [growth, setGrowth] = useState<{ id: number; links: GrowEvent["links"] } | null>(null);
  const [chatWidth, setChatWidth] = useStoredSize("brain.chatWidth", 380);

  const connection = useRef<Connection | null>(null);
  const nextId = useRef(1);
  const fluoRef = useRef<Fluorescence | null>(null);

  const fluo = useMemo(() => {
    const created = new Fluorescence(anatomy?.neurons ?? 1);
    fluoRef.current = created;
    return created;
  }, [anatomy]);

  const addThought = useCallback((entry: Omit<ThoughtEntry, "id">) => {
    setThoughts((list) => [...list, { ...entry, id: nextId.current++ }].slice(-THOUGHTS_LENGTH));
  }, []);

  const addEntry = useCallback((entry: Omit<ChatEntry, "id">) => {
    setEntries((list) => [...list, { ...entry, id: nextId.current++ }].slice(-300));
  }, []);

  useEffect(() => {
    const handle = (message: ServerMessage) => {
      switch (message.type) {
        case "anatomy":
          setAnatomy(message);
          break;
        case "status":
          setStatus(message);
          setEmotion(message.emotionsEnabled ? message.emotion : null);
          if (!message.busy) setProgress(null);
          break;
        case "spikes":
          for (const chunk of message.chunks) fluoRef.current?.spike(chunk);
          break;
        case "step":
          setSteps((list) => [...list, message].slice(-TIMELINE_LENGTH));
          break;
        case "out":
          // Чат живёт на HTTP API; по WebSocket — только прогресс чтения.
          if (message.kind === "progress") setProgress(message.text);
          break;
        case "brains":
          setBrains(message);
          break;
        case "emotion":
          setEmotion(message);
          break;
        case "initiative":
          addEntry({ who: "brain", text: message.text, initiative: message.reason });
          break;
        case "musing":
          addThought({ kind: "musing", text: message.text });
          break;
        case "choice":
          addThought({ kind: "choice", text: message.token, candidates: message.candidates });
          break;
        case "grow":
          setGrowth({ id: nextId.current++, links: message.links });
          break;
        case "thought":
          addThought({ kind: "step", text: message.thought, verdict: message.verdict, score: message.score });
          break;
      }
    };
    connection.current = connect(handle, setConnected);
    return () => connection.current?.close();
  }, [addEntry, addThought]);

  const report = useCallback(
    (error: unknown) => addEntry({ who: "info", text: `Ошибка: ${(error as Error).message}` }),
    [addEntry],
  );

  const actions: ChatActions = useMemo<ChatActions>(
    (): ChatActions => ({
      say: (text) => {
        addEntry({ who: "you", text });
        const id = nextId.current++;
        setEntries((list) => [...list, { id, who: "brain", text: "", streaming: true, words: [] }]);
        const update = (patch: (entry: ChatEntry) => ChatEntry) =>
          setEntries((list) => list.map((entry) => (entry.id === id ? patch(entry) : entry)));
        api
          .chat(text, (token, confidence) =>
            update((entry) => ({
              ...entry,
              text: `${entry.text} ${token}`.trim(),
              words: [...(entry.words ?? []), { token, confidence }],
            })),
          )
          .then(({ text: final, reply }) => {
            update((entry) => ({ ...entry, text: final, streaming: false, reply }));
            // Обдумывал ответ (/think on) — цепочку мыслей закрывает сам ответ.
            setThoughts((list) =>
              list.length && list[list.length - 1].kind !== "musing"
                ? [...list, { id: nextId.current++, kind: "musing", text: final }]
                : list,
            );
          })
          .catch((error) => {
            update((entry) => ({ ...entry, text: "…", streaming: false }));
            report(error);
          });
      },
      command: (text) => {
        addEntry({ who: "you", text });
        api
          .command(text)
          .then(({ output }) =>
            output.forEach((line) => addEntry({ who: line.kind === "brain" ? "brain" : "info", text: line.text })),
          )
          .catch(report);
      },
      upload: (name, text, size) => {
        addEntry({ who: "you", text: `Загружаю «${name}» (${(size / 1024).toFixed(0)} КБ)` });
        api
          .read(name, text)
          .then((r) =>
            addEntry({ who: "info", text: `Читаю «${name}» в фоне: ${r.sentences} фраз. Можно разговаривать.` }),
          )
          .catch(report);
      },
      feedback: (reward) => {
        setEntries((list) => list.map((entry) => (entry.who === "brain" ? { ...entry, rated: true } : entry)));
        addEntry({ who: "you", text: reward > 0 ? "Верно" : "Неверно" });
        api.feedback(reward).then(({ text }) => addEntry({ who: "info", text })).catch(report);
      },
      stop: () => {
        api.stop().catch(report);
      },
      sleep: () => actions.command("/sleep"),
    }),
    [addEntry, report],
  );

  const thinking = Boolean(status?.thinking);
  const live = Boolean(status?.live);

  return (
    <div className="app">
      <header className="masthead">
        <h1>Срез мозга</h1>
        {brains && brains.items.length > 0 && (
          <label className="brain-picker">
            <span>Мозг</span>
            <select
              value={brains.current}
              disabled={!connected || status?.busy}
              onChange={(event) => {
                const name = event.target.value;
                addEntry({ who: "you", text: `Переключаюсь на ${name}` });
                api
                  .load(name)
                  .then((r) => addEntry({ who: "info", text: `Загружено: ${r.brain} — ${r.neurons.toLocaleString("ru")} нейронов` }))
                  .catch(report);
              }}
            >
              {brains.items.map((brain) => (
                <option key={brain.name} value={brain.name}>
                  {brain.name} · {brain.megabytes} МБ
                </option>
              ))}
            </select>
          </label>
        )}
        {status?.features && (
          <details className="features">
            <summary>Функции</summary>
            <div className="features-list">
              {status.features.map((feature) => (
                <label key={feature.name} className="toggle" title={feature.name}>
                  <input
                    type="checkbox"
                    checked={feature.enabled}
                    disabled={!connected}
                    onChange={(event) => api.setFeature(feature.name, event.target.checked).catch(report)}
                  />
                  {feature.title}
                </label>
              ))}
            </div>
          </details>
        )}
        {anatomy ? (
          <p className="vitals">
            <span>{anatomy.neurons.toLocaleString("ru")} нейронов</span>
            <span>{anatomy.synapses.toLocaleString("ru")} синапсов</span>
            {status && (
              <>
                <span>{status.words.toLocaleString("ru")} слов знает</span>
                <span>{status.tokens.toLocaleString("ru")} прожито</span>
                <span>{bioTime(status.timeMs)} жизни</span>
                <span>{status.grown.toLocaleString("ru")} новых синапсов выросло</span>
                {status.reading.length > 0 && (
                  <span>
                    читает «{status.reading[0].name}» {status.reading[0].done}/{status.reading[0].total}
                    {status.reading.length > 1 ? ` и ещё ${status.reading.length - 1}` : ""}
                  </span>
                )}
                <span>
                  {status.sleeping
                    ? "спит"
                    : `${status.sleepPressure.toLocaleString("ru")} новых впечатлений`}
                </span>
              </>
            )}
            <span className="device">{anatomy.device === "cpu" ? "процессор" : "видеокарта"}</span>
          </p>
        ) : (
          <p className="vitals">{connected ? "Мозг просыпается…" : "Запусти python -m brain.server"}</p>
        )}
      </header>

      <main className="bench" style={{ "--chat-width": `${chatWidth}px` } as CSSProperties}>
        <div className="stage">
          {anatomy ? (
            <>
              <Microscope anatomy={anatomy} fluo={fluo} onActivity={setActivity} growth={growth} />
              <div className="legend">
                <Pathways anatomy={anatomy} status={status} activity={activity} emotion={emotion} />
                <dl className="key">
                  <div>
                    <dt className="swatch swatch-e" />
                    <dd>возбуждающий нейрон</dd>
                  </div>
                  <div>
                    <dt className="swatch swatch-i" />
                    <dd>тормозный нейрон</dd>
                  </div>
                  <div>
                    <dt className="swatch swatch-dopamine" />
                    <dd>путь учится от дофамина</dd>
                  </div>
                </dl>
              </div>
              <Timeline steps={steps} />
              <Guide />
            </>
          ) : (
            <div className="slide-placeholder" />
          )}
        </div>

        <Splitter
          axis="x"
          size={chatWidth}
          min={CHAT_MIN}
          max={chatMax}
          direction={-1}
          onResize={setChatWidth}
          label="Ширина чата"
        />

        <Chat
          thoughts={{
            count: thoughts.filter((t) => t.kind === "musing").length,
            active: live || thinking,
            node: <Thoughts entries={thoughts} live={live} thinking={thinking} />,
            clear: () => setThoughts([]),
          }}
          header={
            <div className="mood">
              <Face emotion={emotion} />
              <label className="toggle" title="Думает в тишине, иногда заговаривает сам">
                <input
                  type="checkbox"
                  checked={status?.live ?? false}
                  disabled={!connected}
                  onChange={(event) => {
                    api.setLive(event.target.checked).catch(report);
                  }}
                />
                Живой режим
              </label>
              <label className="toggle">
                <input
                  type="checkbox"
                  checked={status?.emotionsEnabled ?? true}
                  disabled={!connected}
                  onChange={(event) => api.setEmotions(event.target.checked).catch(report)}
                />
                Эмоции
              </label>
            </div>
          }
          entries={entries}
          progress={progress}
          busy={status?.busy ?? false}
          reading={(status?.reading.length ?? 0) > 0}
          connected={connected}
          actions={actions}
        />
      </main>
    </div>
  );
}
