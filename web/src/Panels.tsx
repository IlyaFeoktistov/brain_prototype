import { useEffect, useRef, useState } from "react";
import * as api from "./api";
import type { Brains, FlagInfo, FlagState, Health, Status, Summary } from "./brain";
import { SUBSTANCE, SUBSTANCE_HINT, bioTime, hourText } from "./brain";

// ---- химия и состояние: то, что наведением не увидишь ----

export function Chemistry({ status }: { status: Status }) {
  return (
    <div className="panel">
      <dl className="state">
        <div className="kv"><dt>состояние</dt><dd>{status.asleep ? "спит (VLPO > LC)" : "бодрствует"}</dd></div>
        <div className="kv"><dt>время суток</dt><dd>{hourText(status.hour ?? 0)}</dd></div>
        <div className="kv"><dt>прожито</dt><dd>{bioTime(status.timeMs ?? 0)}</dd></div>
      </dl>
      <p className="muted">
        Уровень — спайков на клетку ядра за время выведения; занятость рецепторов — уровень / (уровень + полунасыщение).
        Аденозин копится наяву и вымывается во сне.
      </p>
      <table className="grid">
        <thead><tr><th>вещество</th><th title="доля занятых рецепторов: 0 — пусто, 1 — всё занято">занятость рецепторов</th><th title="занятость наяву в покое — от частоты ядра по записям">в покое</th><th title="спайков на клетку ядра за время выведения">уровень</th></tr></thead>
        <tbody>
          {(status.chemistry ?? []).map((c) => (
            <tr key={c.name}>
              <td><span className="hint" title={SUBSTANCE_HINT[c.name]}>{SUBSTANCE[c.name] ?? c.name}</span></td>
              <td><span className="bar"><span style={{ width: `${c.occupancy * 100}%` }} /></span> {c.occupancy.toFixed(2)}</td>
              <td>{c.rest === null ? "—" : c.rest.toFixed(2)}</td>
              <td>{c.level.toPrecision(3)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <dl className="explain">
        {(status.chemistry ?? []).map((c) => (
          <div key={c.name}><dt>{SUBSTANCE[c.name] ?? c.name}</dt><dd>{SUBSTANCE_HINT[c.name] ?? ""}</dd></div>
        ))}
      </dl>
    </div>
  );
}

// ---- разговор через мир ----

export interface Entry {
  id: number;
  who: "you" | "brain" | "info";
  text: string;
}

export interface WorldScreen {
  line: string;
  gaze: number;
  pointer: number | null;
}

export function Conversation({ face, entries, screen, typing, busy, onSay, thoughts }: {
  face: React.ReactNode;
  thoughts: React.ReactNode;
  entries: Entry[];
  screen: WorldScreen;
  typing: string;
  busy: string | null;
  onSay: (text: string) => void;
}) {
  const [text, setText] = useState("");
  const log = useRef<HTMLDivElement>(null);
  useEffect(() => {
    log.current?.scrollTo({ top: log.current.scrollHeight });
  }, [entries, typing]);
  // Экран мира: ~7 px на букву (шрифт 12 px) — взгляд и указка в буквах.
  const PX = 7;

  return (
    <div className="panel conversation">
      {face}
      <div className="world" title="Экран мира: что видит глаз; ▲ — точка взгляда, ● — указка учителя">
        <div className="world-line">{screen.line || " "}</div>
        <div className="world-marks">
          <span className="gaze" style={{ left: `${(screen.gaze / PX) * 0.6}em` }}>▲</span>
          {screen.pointer !== null && <span className="pointer" style={{ left: `${(screen.pointer / PX) * 0.6}em` }}>●</span>}
        </div>
      </div>
      <div className="log" ref={log}>
        {entries.length === 0 && <>
          <p className="muted">Напиши реплику — она появится на экране мира, мозг посмотрит на неё глазом и ответит рукой.</p>
          <p className="muted"><strong>Разговора пока нет.</strong> Рука печатает шум: залпы моторного таламуса (VA) жмут клавиши и на
            пустом экране. Глаз не ведёт взгляд по строке — видит только первые буквы. Флаг «выключить ток VA» (Меню → Флаги)
            убирает шум, но мозг тогда молчит. Что уже работает — в README («Состояние»).</p>
        </>}
        {entries.map((e) => <p key={e.id} className={`line line-${e.who}`}>{e.text}</p>)}
        {typing && <p className="line line-brain typing">{typing}</p>}
      </div>
      <details className="thoughts-box">
        <summary>Мысли — что печатает рука и из чего выбирает</summary>
        {thoughts}
      </details>
      <form
        className="say"
        onSubmit={(event) => {
          event.preventDefault();
          if (!text.trim()) return;
          onSay(text.trim());
          setText("");
        }}
      >
        <input value={text} onChange={(e) => setText(e.target.value)} placeholder={busy ? `мозг занят: ${busy}` : "реплика"} />
        <button type="submit" disabled={!text.trim()} title="показать реплику на экране мира; мозг читает её и печатает ответ, пока не замолчит на 2 с (не дольше 10 с своей жизни)">Сказать</button>
        {/* Прервать ответ или урок: мозг дочитывает текущее окно (~0.1 с жизни мозга) и останавливается. */}
        {busy && <button type="button" className="stop" onClick={() => api.stop()} title="прервать ответ или урок">Стоп</button>}
      </form>
    </div>
  );
}

// ---- меню команд ----

export interface MenuActions {
  info: (text: string) => void;
  report: (title: string, node: React.ReactNode) => void;
  choose: () => void;
}

function Lessons({ actions }: { actions: MenuActions }) {
  const [n, setN] = useState(50);
  const [first, setFirst] = useState(1);
  const [last, setLast] = useState(7);
  const [text, setText] = useState("");
  const run = (kind: string, extra: { text?: string; first?: number; last?: number } = {}) => {
    actions.info(`урок «${kind}» начат`);
    api.lesson(kind, n, extra).then((r) => actions.info(r.text)).catch((e) => actions.info(`Ошибка: ${(e as Error).message}`));
  };
  return (
    <div className="menu-group">
      <h4>Уроки (учитель мира)</h4>
      <label className="inline">повторов <input type="number" min={1} value={n} onChange={(e) => setN(Number(e.target.value))} /></label>
      <div className="buttons">
        <button type="button" onClick={() => run("letters")} title="буква на экране, указка под ней">Азбука</button>
        <button type="button" onClick={() => run("babble")} title="пустой экран, рука нажимает сама, нажатое видно">Лепет руки</button>
        <button type="button" onClick={() => run("copy")} title="учитель пишет букву, мозг печатает, верно — сладкое">Копирование букв</button>
      </div>
      <div className="buttons">
        <label className="inline">курс с <input type="number" min={1} max={7} value={first} onChange={(e) => setFirst(Number(e.target.value))} /></label>
        <label className="inline">по <input type="number" min={1} max={7} value={last} onChange={(e) => setLast(Number(e.target.value))} /></label>
        <button type="button" onClick={() => run("course", { first, last })} title="data/course: азбука → слоги → слова → фразы → факты → вопросы; повторов — кругов">Курс</button>
      </div>
      <textarea rows={3} value={text} placeholder="текст: учитель покажет его по словам с указкой" onChange={(e) => setText(e.target.value)} />
      <div className="buttons">
        <button type="button" disabled={!text.trim()} onClick={() => run("read", { text })} title="учитель показывает текст по словам, указка ведёт взгляд под каждой буквой">Прочитать текст</button>
        <label className="file">
          из файла
          <input type="file" accept=".txt,.md" onChange={(e) => e.target.files?.[0]?.text().then(setText)} />
        </label>
      </div>
    </div>
  );
}

function SummaryView({ data }: { data: Summary }) {
  return (
    <>
      <p>{data.neurons.toLocaleString("ru")} нейронов · {data.synapses.toLocaleString("ru")} синапсов · учатся {data.plastic.toLocaleString("ru")} · прожито {bioTime(data.timeMs)}</p>
      <h4>Что выучено по путям</h4>
      <table className="grid">
        <thead><tr><th>путь</th><th>синапсов</th><th>правило</th><th>сильнее</th><th>слабее</th><th>|выучено| ср., мВ</th></tr></thead>
        <tbody>{data.paths.map((p) => <tr key={p.path}><td>{p.path}</td><td>{p.synapses.toLocaleString("ru")}</td><td>{p.modulator}</td><td>{p.stronger}</td><td>{p.weaker}</td><td>{p.learnedPsp.toFixed(4)}</td></tr>)}</tbody>
      </table>
    </>
  );
}

const mark = (hz: number, target: number | null) => {
  if (target === null || target <= 0) return "";
  const r = (hz + 0.5) / (target + 0.5);
  return r < 0.5 ? "мало" : r > 2 ? "много" : "ок";
};

function HealthView({ data }: { data: Health }) {
  return (
    <>
      <p className="muted">Покой {data.ms} мс без входа и без пластичности; цель — записи у бодрствующих (anatomy.RATES_HZ, column_target_hz).</p>
      <h4>Ядра (главные клетки), Гц</h4>
      <table className="grid">
        <thead><tr><th>ядро</th><th>у нас</th><th>запись</th><th /></tr></thead>
        <tbody>{data.nuclei.map((n) => <tr key={n.name} className={`health-${mark(n.hz, n.target)}`}><td title={n.title}>{n.name}</td><td>{n.hz.toFixed(1)}</td><td>{n.target ?? "—"}</td><td>{mark(n.hz, n.target)}</td></tr>)}</tbody>
      </table>
      <h4>Колонки, Гц</h4>
      <table className="grid">
        <thead><tr><th>зона</th><th>слой.тип: у нас / цель</th></tr></thead>
        <tbody>
          {data.columns.map((c) => (
            <tr key={c.name}>
              <td title={c.title}>{c.name}</td>
              <td>{c.layers.map((l) => <span key={l.layer + l.kind} className={`chip health-${mark(l.hz, l.target)}`}>{l.layer}.{l.kind} {l.hz.toFixed(1)}/{l.target ?? "—"}</span>)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </>
  );
}

// ---- флаги мозга (brain/flags.py) ----

const FLAG_STATUS: Record<FlagInfo["status"], string> = {
  debugged: "отлажен",
  partial: "отлажен частично",
  not_debugged: "не отлажен",
  not_data: "не по данным",
};

function FlagRow({ flag, checked, disabled, onChange }: {
  flag: FlagInfo; checked: boolean; disabled?: boolean; onChange?: (on: boolean) => void;
}) {
  return (
    <label className="toggle flag" title={`${flag.note}\nГде разобран: ${flag.where}`}>
      <input type="checkbox" checked={checked} disabled={disabled || !onChange} onChange={(e) => onChange?.(e.target.checked)} />
      <span>
        {flag.title} <span className={`flag-status ${flag.status}`}>{FLAG_STATUS[flag.status]}</span>
        <small className="muted">{flag.note}</small>
      </span>
    </label>
  );
}

function BrainFlags({ info }: { info: (text: string) => void }) {
  const [state, setState] = useState<FlagState | null>(null);
  const [busy, setBusy] = useState(false);
  useEffect(() => { api.flagState().then(setState).catch(() => setState(null)); }, []);
  if (!state) return null;
  const toggle = (key: string, on: boolean) => {
    setBusy(true);
    api.setFlag(key, on).then(setState).catch((e) => info(`Ошибка: ${(e as Error).message}`)).finally(() => setBusy(false));
  };
  const wiring = state.flags.filter((f) => f.kind === "wiring");
  const runtime = state.flags.filter((f) => f.kind === "runtime");
  return (
    <div className="menu-group">
      <h4>Флаги</h4>
      <p className="muted">
        Собран: {state.preset ? <>пресет «{state.preset}»</> : "голый мозг"}, зерно {state.seed},{" "}
        {state.neurons.toLocaleString("ru")} нейронов. Проводка — с чем мозг создан (меняется только новым мозгом):
      </p>
      {wiring.map((f) => <FlagRow key={f.key} flag={f} checked={!!f.value} />)}
      <p className="muted">На ходу (сохраняются с мозгом):</p>
      {runtime.map((f) => <FlagRow key={f.key} flag={f} checked={!!f.value} disabled={busy} onChange={(on) => toggle(f.key, on)} />)}
    </div>
  );
}

export function Menu({ status, actions }: { status: Status; actions: MenuActions }) {
  const fail = (e: unknown) => actions.info(`Ошибка: ${(e as Error).message}`);
  return (
    <div className="panel menu">
      <div className="menu-group">
        <h4>Учитель</h4>
        <div className="buttons">
          <button type="button" onClick={() => api.praise(true).then(() => actions.info("сладкое (верно)")).catch(fail)}
                  title="награда: 0.3 с вкуса сладкого — дофамин закрепляет то, что мозг только что делал">Верно — сладкое</button>
          <button type="button" onClick={() => api.praise(false).then(() => actions.info("горькое (неверно)")).catch(fail)}
                  title="наказание: 0.3 с вкуса горького — провал дофамина ослабляет то, что мозг только что делал">Неверно — горькое</button>
          <button type="button" onClick={() => api.stop().then(() => actions.info("остановлено")).catch(fail)} title="прервать ответ или урок (жизнь в тишине выключается отдельно, флажком ниже)">Стоп</button>
        </div>
      </div>
      <Lessons actions={actions} />
      <div className="menu-group">
        <h4>Наблюдение</h4>
        <div className="buttons">
          <button type="button" onClick={() => api.summary().then((d) => actions.report("Сводка", <SummaryView data={d} />)).catch(fail)} title="сколько клеток и синапсов, сколько их учится и что выучено на каждом пути (вкладка «Отчёт»)">Сводка</button>
          <button type="button" onClick={() => { actions.info("здоровье: мозг в покое 500 мс…"); api.health(500).then((d) => actions.report("Здоровье", <HealthView data={d} />)).catch(fail); }} title="полсекунды покоя без входа и без обучения: частоты ядер и слоёв коры против записей у людей и животных (вкладка «Отчёт»)">Здоровье</button>
        </div>
        <label className="toggle">
          <input type="checkbox" checked={status.live} onChange={(e) => api.setLive(e.target.checked).catch(fail)} />
          жить в тишине (мозг работает и без разговора)
        </label>
      </div>
      <div className="menu-group">
        <h4>Мозг</h4>
        <div className="buttons">
          <button type="button" onClick={() => api.save().then((r) => actions.info(`сохранено: ${r.brain}`)).catch(fail)} title="записать мозг на диск (~/.brain/brains) со всем выученным">Сохранить</button>
          <button type="button" onClick={actions.choose} title="открыть сохранённый мозг или создать новый (текущий сначала сохраняется)">Другой мозг…</button>
        </div>
      </div>
      <BrainFlags info={actions.info} />
    </div>
  );
}

// ---- выбор мозга ----

export function Chooser({ brains, onClose, info }: { brains: Brains; onClose?: () => void; info: (text: string) => void }) {
  const [name, setName] = useState("");
  const [neurons, setNeurons] = useState(brains.minNeurons);
  // По умолчанию — рабочий мозг (тот, с которым идут опыты), если его файлы на месте.
  const [preset, setPreset] = useState<string>(brains.presets.find((p) => p.ready)?.name ?? "");
  const [preview, setPreview] = useState<{ total: number; min: number } | null>(null);
  const [busy, setBusy] = useState(false);
  const [seed, setSeed] = useState(1);
  const [chosen, setChosen] = useState<Record<string, boolean>>({});
  const small = !preset && !!chosen.small_columns;
  const flagsShown = brains.flags.filter((f) => !(preset && f.key === "small_columns"));

  useEffect(() => {
    const timer = window.setTimeout(() => api.sizes(neurons, small).then(setPreview).catch(() => setPreview(null)), 250);
    return () => window.clearTimeout(timer);
  }, [neurons, small]);

  const act = (promise: Promise<{ brain: string }>, what: string) => {
    setBusy(true);
    info(`${what}…`);
    promise.then((r) => { info(`мозг: ${r.brain}`); onClose?.(); }).catch((e) => info(`Ошибка: ${(e as Error).message}`)).finally(() => setBusy(false));
  };

  return (
    <div className="chooser">
      <h2>Мозг</h2>
      {onClose && <button type="button" className="close" onClick={onClose} aria-label="закрыть" title="вернуться к текущему мозгу">×</button>}
      <section>
        <h3>Открыть сохранённый</h3>
        {brains.items.length === 0 && <p className="muted">Сохранённых мозгов нового формата нет.</p>}
        <ul className="brains">
          {brains.items.map((b) => (
            <li key={b.name}>
              <button type="button" disabled={busy} onClick={() => act(api.load(b.name), `загружаю ${b.name}`)} title="загрузить этот мозг (текущий сначала сохраняется)">
                <strong>{b.name}</strong>
                <span>{(b.neurons ?? 0).toLocaleString("ru")} нейронов · {b.megabytes} МБ · прожито {bioTime(b.timeMs ?? 0)} · {new Date(b.modified * 1000).toLocaleString("ru")}</span>
              </button>
            </li>
          ))}
        </ul>
      </section>
      <section>
        <h3>Создать новый</h3>
        <form onSubmit={(e) => {
          e.preventDefault();
          const fl = Object.fromEntries(Object.entries(chosen).filter(([k, v]) => v && !(preset && k === "small_columns")));
          act(api.newBrain(name.trim(), neurons, preset || null, seed, fl), `создаю ${name}`);
        }}>
          <label>имя <input value={name} onChange={(e) => setName(e.target.value)} placeholder="например, первый" /></label>
          <label>сборка <select value={preset} onChange={(e) => setPreset(e.target.value)}>
            {brains.presets.map((p) => (
              <option key={p.name} value={p.name} disabled={!p.ready}>{p.title}{p.ready ? "" : " — нет файлов"}</option>
            ))}
            <option value="">голый мозг заданного размера</option>
          </select></label>
          {!preset && <label>нейронов <input type="number" min={1} step={10000} value={neurons} onChange={(e) => setNeurons(Number(e.target.value))} /></label>}
          <label>зерно <input type="number" min={1} step={1} value={seed} onChange={(e) => setSeed(Number(e.target.value))} /></label>
          {preset ? <p className="muted">Размер — наименьший мозг ({brains.minNeurons.toLocaleString("ru")} нейронов): подгонки посчитаны на нём.
            Файлы пресета в репозитории — для зерна 1.</p> : <p className="muted">
            Ядра получают клетки по данным, остальное делится между колонками коры по их долям — мозг растёт числом клеток в
            колонках. Колонка не меньше {small ? "1 тыс." : "20 тыс."} клеток, минимум — {(small ? brains.minNeuronsSmall : brains.minNeurons).toLocaleString("ru")}.
            {preview && <> Получится <strong>{preview.total.toLocaleString("ru")}</strong> нейронов.</>}
          </p>}
          <fieldset className="flags">
            <legend>Флаги (опыты; по умолчанию всё выключено — как в пресете)</legend>
            {flagsShown.map((f) => (
              <FlagRow key={f.key} flag={f} checked={!!chosen[f.key]} onChange={(on) => setChosen({ ...chosen, [f.key]: on })} />
            ))}
          </fieldset>
          <button type="submit" disabled={busy || !name.trim()} title="собрать новый мозг (по сборке или голый) и сразу сохранить его">Создать</button>
        </form>
      </section>
    </div>
  );
}
