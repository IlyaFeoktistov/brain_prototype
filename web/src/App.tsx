import { type CSSProperties, type ReactNode, useCallback, useEffect, useMemo, useRef, useState } from "react";
import * as api from "./api";
import type { Anatomy, Brains, Fixation, NeuronInfo, Press, ServerMessage, Status, SynapseInfo, View } from "./brain";
import { bioTime, connect, hourText } from "./brain";
import { Face } from "./Face";
import { Guide } from "./Guide";
import { NeuronCard, SynapseCard } from "./Inspector";
import { Fluorescence, type Hover, HoverTip, type Link, Microscope } from "./Microscope";
import { Chemistry, Chooser, Conversation, type Entry, Menu, type WorldScreen } from "./Panels";
import { Pathways } from "./Pathways";
import { Sight } from "./Sight";
import { Splitter, useStoredSize } from "./Splitter";
import { Strip } from "./Strip";
import { Thoughts } from "./Thoughts";

type Tab = "talk" | "sight" | "cell" | "chemistry" | "menu" | "report" | "guide";
const SIDE_MIN = 340;
// Сколько последних фиксаций и нажатий держать.
const FIXATIONS = 80;
const PRESSES = 300;

export function App() {
  const [connected, setConnected] = useState(false);
  const [status, setStatus] = useState<Status | null>(null);
  const [anatomy, setAnatomy] = useState<Anatomy | null>(null);
  const [brains, setBrains] = useState<Brains | null>(null);
  const [entries, setEntries] = useState<Entry[]>([]);
  const [screen, setScreen] = useState<WorldScreen>({ line: "", gaze: 0, pointer: null });
  const [typing, setTyping] = useState("");
  const [tab, setTab] = useState<Tab>("talk");
  const [hover, setHover] = useState<Hover | null>(null);
  const [selected, setSelected] = useState<number | null>(null);
  const [synapse, setSynapse] = useState<SynapseInfo | null>(null);
  const [links, setLinks] = useState<Link[]>([]);
  const [report, setReport] = useState<{ title: string; node: ReactNode } | null>(null);
  const [choosing, setChoosing] = useState(false);
  const [now, setNow] = useState(0);
  const [activity, setActivity] = useState<Record<string, number>>({});
  const [fixations, setFixations] = useState<Fixation[]>([]);
  const [presses, setPresses] = useState<Press[]>([]);
  const [view, setView] = useState<View | null>(null);
  const [side, setSide] = useStoredSize("brain.side", 460);
  const nextId = useRef(1);
  const fluoRef = useRef<Fluorescence | null>(null);

  const fluo = useMemo(() => {
    const created = new Fluorescence(anatomy?.neurons ?? 1);
    fluoRef.current = created;
    return created;
  }, [anatomy]);

  const add = useCallback((who: Entry["who"], text: string) => {
    setEntries((list) => [...list, { id: nextId.current++, who, text }].slice(-400));
  }, []);

  useEffect(
    () =>
      connect((message: ServerMessage) => {
        switch (message.type) {
          case "status":
            setStatus(message);
            break;
          case "anatomy":
            setAnatomy(message);
            setSelected(null);
            setChoosing(false);
            setFixations([]);
            setPresses([]);
            setView(null);
            break;
          case "brains":
            setBrains(message);
            break;
          case "spikes":
            for (const chunk of message.chunks) fluoRef.current?.spike(chunk, message.t);
            setNow(message.t);
            break;
          case "screen":
            setScreen(message);
            break;
          case "typed":
            setTyping((text) => text + message.char);
            break;
          case "info":
            add("info", message.text);
            break;
          case "fixation":
            setFixations((list) => [...list, message].slice(-FIXATIONS));
            break;
          case "press":
            setPresses((list) => [...list, message].slice(-PRESSES));
            break;
          case "view":
            setView(message);
            break;
        }
      }, setConnected),
    [add],
  );

  const say = useCallback(
    (text: string) => {
      add("you", text);
      setTyping("");
      api
        .say(text)
        .then((r) => {
          add("brain", r.text || "(молчит)");
          setTyping("");
        })
        .catch((e) => add("info", `Ошибка: ${(e as Error).message}`));
    },
    [add],
  );

  const select = useCallback((index: number) => {
    setSelected(index);
    setSynapse(null);
    setTab("cell");
  }, []);

  // Карточка пришла: следить за клеткой и её самыми сильными связями.
  const onLoaded = useCallback((info: NeuronInfo) => {
    const f = fluoRef.current;
    if (!f) return;
    const partners = [...info.topIncoming.map((s) => s.pre), ...info.topOutgoing.map((s) => s.post), info.index];
    for (const key of [...f.watched.keys()]) if (!partners.includes(key)) f.watched.delete(key);
    for (const p of partners) if (!f.watched.has(p)) f.watched.set(p, []);
    setLinks((old) => {
      const fresh = [
        ...info.topIncoming.map((s) => ({ from: s.pre, to: s.post, inhibitory: s.psp < 0 })),
        ...info.topOutgoing.map((s) => ({ from: s.pre, to: s.post, inhibitory: s.psp < 0 })),
      ];
      const same = old.length === fresh.length && old.every((l, i) => l.from === fresh[i].from && l.to === fresh[i].to);
      return same ? old : fresh;
    });
  }, []);

  if (!status || !brains) {
    return <div className="boot">{connected ? "Связь есть, жду мозг…" : "Запусти: python -m brain.server"}</div>;
  }

  if (!status.loaded || !anatomy || choosing) {
    return (
      <div className="boot">
        <Chooser brains={brains} info={(text) => add("info", text)} onClose={status.loaded ? () => setChoosing(false) : undefined} />
        <div className="boot-log">{entries.slice(-6).map((e) => <p key={e.id} className="muted">{e.text}</p>)}</div>
      </div>
    );
  }

  return (
    <div className="app">
      <header className="masthead">
        <h1>Срез мозга</h1>
        <p className="vitals">
          <span>{status.brain}</span>
          <span>{anatomy.neurons.toLocaleString("ru")} нейронов</span>
          <span>{anatomy.synapses.toLocaleString("ru")} синапсов</span>
          <span>{status.asleep ? "спит" : "бодрствует"}</span>
          <span>{hourText(status.hour ?? 0)}</span>
          <span>прожито {bioTime(status.timeMs ?? 0)}</span>
          {status.busy && <span className="busy">{status.busy}…</span>}
          {status.live && <span>живёт в тишине</span>}
          <span className="device">{anatomy.device === "cpu" ? "процессор" : "видеокарта"}</span>
          {!connected && <span className="busy">нет связи</span>}
        </p>
      </header>

      <main className="bench" style={{ "--side": `${side}px` } as CSSProperties}>
        <div className="stage">
          <Microscope
            anatomy={anatomy}
            fluo={fluo}
            selected={selected}
            links={selected === null ? [] : links}
            onHover={setHover}
            onSelect={select}
            onActivity={setActivity}
          />
          <dl className="legend">
            <div><dt className="swatch swatch-E" /><dd>главная клетка</dd></div>
            <div><dt className="swatch swatch-PV" /><dd>PV / местная тормозная</dd></div>
            <div><dt className="swatch swatch-SOM" /><dd>SOM</dd></div>
            <div><dt className="swatch swatch-VIP" /><dd>VIP</dd></div>
            <div><dt className="swatch swatch-LAMP5" /><dd>LAMP5</dd></div>
            <div><dd>колонки — слоями сверху вниз: L2/3, L4, L5, L6; яркость — спайк (показ с задержкой ~50 мс)</dd></div>
          </dl>
          <Strip fixations={fixations} />
          <Pathways anatomy={anatomy} activity={activity} />
        </div>

        <Splitter axis="x" size={side} min={SIDE_MIN} max={() => Math.max(SIDE_MIN, window.innerWidth * 0.7)} direction={-1} onResize={setSide} label="Ширина панели" />

        <aside className="side">
          <nav className="tabs">
            {([
              ["talk", "Разговор", "написать мозгу: реплика появится на экране мира, мозг прочтёт её глазом и ответит рукой; лицо, мысли руки"],
              ["sight", "Глаз и рука", "что видит глаз на каждом этапе (сетчатка, таламус, кора) и куда тянется рука на клавиатуре"],
              ["cell", "Клетка", "всё о клетке, выбранной в микроскопе: модель, состояние, вещества, дендрит, синапсы"],
              ["chemistry", "Химия", "вещества мозга: сколько рецепторов занято сейчас и в покое, что каждое делает"],
              ["menu", "Меню", "уроки, похвала, сводка и здоровье, жизнь в тишине, сохранение и выбор мозга"],
              ["report", "Отчёт", "результат «Сводки» или «Здоровья» из меню"],
              ["guide", "Справка", "как устроен мозг и как пользоваться интерфейсом"],
            ] as [Tab, string, string][]).map(([key, title, hint]) => (
              <button key={key} type="button" className={tab === key ? "on" : ""} onClick={() => setTab(key)} title={hint}>{title}</button>
            ))}
          </nav>
          {tab === "talk" && (
            <Conversation face={<Face status={status} />} entries={entries} screen={screen} typing={typing} busy={status.busy} onSay={say}
                          thoughts={<Thoughts presses={presses} live={status.live} />} />
          )}
          {tab === "sight" && <Sight view={view} keyboard={anatomy.keyboard} />}
          {tab === "cell" && (
            <div className="panel">
              {synapse && <SynapseCard synapse={synapse} onNeuron={select} onClose={() => setSynapse(null)} />}
              {selected !== null ? (
                <NeuronCard index={selected} fluo={fluo} now={now} onNeuron={select} onSynapse={setSynapse}
                            onClose={() => { setSelected(null); setLinks([]); }} onLoaded={onLoaded} />
              ) : (
                <p className="muted">Кликни клетку в микроскопе: её модель, состояние, вещества, ветки дендрита и все синапсы.</p>
              )}
            </div>
          )}
          {tab === "chemistry" && <Chemistry status={status} />}
          {tab === "menu" && (
            <Menu
              status={status}
              actions={{
                info: (text) => add("info", text),
                report: (title, node) => { setReport({ title, node }); setTab("report"); },
                choose: () => setChoosing(true),
              }}
            />
          )}
          {tab === "guide" && <Guide />}
          {tab === "report" && (
            <div className="panel">
              {report ? <><h3>{report.title}</h3>{report.node}</> : <p className="muted">Сводка и «Здоровье» из меню появятся здесь.</p>}
            </div>
          )}
        </aside>
      </main>
      {hover && <HoverTip hover={hover} />}
    </div>
  );
}
