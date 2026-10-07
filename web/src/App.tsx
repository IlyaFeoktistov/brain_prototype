import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { connect, type Anatomy, type ClientMessage, type Connection, type ServerMessage, type Status, type StepEvent } from "./brain";
import { Chat, type ChatEntry } from "./Chat";
import { Fluorescence, Microscope } from "./Microscope";
import { Guide } from "./Guide";
import { Pathways } from "./Pathways";
import { Timeline } from "./Timeline";

const TIMELINE_LENGTH = 60;

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

  const connection = useRef<Connection | null>(null);
  const nextId = useRef(1);
  const fluoRef = useRef<Fluorescence | null>(null);

  const fluo = useMemo(() => {
    const created = new Fluorescence(anatomy?.neurons ?? 1);
    fluoRef.current = created;
    return created;
  }, [anatomy]);

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
          if (!message.busy) setProgress(null);
          break;
        case "spikes":
          for (const chunk of message.chunks) fluoRef.current?.spike(chunk);
          break;
        case "step":
          setSteps((list) => [...list, message].slice(-TIMELINE_LENGTH));
          break;
        case "out":
          if (message.kind === "progress") setProgress(message.text);
          else if (message.kind === "brain") addEntry({ who: "brain", text: message.text });
          // Строку скорости интерфейс рисует сам из события reply.
          else if (!message.text.startsWith("[промпт:")) addEntry({ who: "info", text: message.text });
          break;
        case "reply":
          setEntries((list) => {
            const copy = [...list];
            for (let i = copy.length - 1; i >= 0; i--) {
              if (copy[i].who === "brain") {
                copy[i] = { ...copy[i], reply: message };
                break;
              }
            }
            return copy;
          });
          break;
      }
    };
    connection.current = connect(handle, setConnected);
    return () => connection.current?.close();
  }, [addEntry]);

  const send = useCallback((message: ClientMessage) => connection.current?.send(message), []);

  return (
    <div className="app">
      <header className="masthead">
        <h1>Срез мозга</h1>
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
              </>
            )}
            <span className="device">{anatomy.device === "cpu" ? "процессор" : "видеокарта"}</span>
          </p>
        ) : (
          <p className="vitals">{connected ? "Мозг просыпается…" : "Запусти python -m brain.server"}</p>
        )}
      </header>

      <main className="bench">
        <div className="stage">
          {anatomy ? (
            <>
              <Microscope anatomy={anatomy} fluo={fluo} onActivity={setActivity} />
              <div className="legend">
                <Pathways anatomy={anatomy} status={status} activity={activity} />
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

        <Chat
          entries={entries}
          progress={progress}
          busy={status?.busy ?? false}
          connected={connected}
          send={send}
          onLocal={addEntry}
        />
      </main>
    </div>
  );
}
