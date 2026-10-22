import { useEffect, useState } from "react";
import * as api from "./api";
import type { NeuronInfo, SynapseGroup, SynapseInfo } from "./brain";
import { KIND_TITLE, SUBSTANCE, SUBSTANCE_HINT } from "./brain";
import type { Fluorescence } from "./Microscope";

const f = (x: number | null | undefined, digits = 2) => (x === null || x === undefined ? "—" : x.toFixed(digits));
const MODULATOR: Record<string, string> = {
  hebbian: "STDP сразу в вес (кора)",
  acetylcholine: "ацетилхолин (гиппокамп)",
  dopamine: "дофамин (D1)",
  dopamine_d2: "провал дофамина (D2)",
  climbing: "лазающее волокно (мозжечок)",
};

function Row({ name, value, hint }: { name: string; value: string; hint?: string }) {
  return (
    <div className="kv" title={hint}>
      <dt className={hint ? "hint" : undefined}>{name}</dt>
      <dd>{value}</dd>
    </div>
  );
}

function Groups({ groups, direction }: { groups: SynapseGroup[]; direction: "in" | "out" }) {
  if (!groups.length) return <p className="muted">нет</p>;
  return (
    <table className="grid">
      <thead>
        <tr>
          <th>путь</th>
          <th>{direction === "in" ? "откуда" : "куда"}</th>
          <th title="сколько синапсов этого пути у клетки">синапсов</th>
          <th title="пик ВПСП/ТПСП в покое">мВ ср. (мин…макс)</th>
          <th title="врождённый или по какому правилу учится">учится</th>
          <th title="выученное в среднем, мВ; сильнее / слабее рождения">выучено</th>
        </tr>
      </thead>
      <tbody>
        {groups.map((g, i) => (
          <tr key={i}>
            <td>{g.path}{g.dendrite ? " · на дендрит" : ""}{g.achSuppressed ? " · ACh ⊣" : ""}</td>
            <td>{g.sensory ? "рецепторы" : g.population ?? "—"}</td>
            <td>{g.count.toLocaleString("ru")}</td>
            <td>{f(g.pspMean, 3)}{g.pspMin !== undefined ? ` (${f(g.pspMin, 2)}…${f(g.pspMax, 2)})` : ""}</td>
            <td>{g.plastic ? MODULATOR[g.modulator ?? ""] ?? g.modulator : "врождённый"}</td>
            <td>{g.plastic ? `${f(g.learnedPsp, 4)} · ↑${g.stronger} ↓${g.weaker}` : ""}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function Synapses({ list, onSynapse, onNeuron, direction }: {
  list: SynapseInfo[];
  onSynapse: (s: SynapseInfo) => void;
  onNeuron: (i: number) => void;
  direction: "in" | "out";
}) {
  return (
    <table className="grid">
      <thead>
        <tr>
          <th>{direction === "in" ? "от клетки" : "к клетке"}</th>
          <th>путь</th>
          <th title="сила сейчас: пик ВПСП (+) или ТПСП (−) в покое">мВ</th>
          <th title="сила при рождении">врожд.</th>
          <th title="след STDP: ждёт третьего фактора">след</th>
          <th title="куда садится: ветка дендрита или сома">ветка</th>
          <th />
        </tr>
      </thead>
      <tbody>
        {list.map((s) => {
          const other = direction === "in" ? s.pre : s.post;
          return (
            <tr key={s.id}>
              <td>
                <button type="button" className="link" onClick={() => onNeuron(other)}>
                  {direction === "in" ? s.preName : s.postName} №{other}
                </button>
              </td>
              <td>{s.path}</td>
              <td>{f(s.psp, 3)}</td>
              <td>{s.plastic ? f(s.birthPsp, 3) : "—"}</td>
              <td>{s.plastic ? f(s.eligibility, 4) : "—"}</td>
              <td>{s.dendrite ? s.branch : "сома"}</td>
              <td>
                <button type="button" className="link" onClick={() => onSynapse(s)}>синапс</button>
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

function Raster({ times, now }: { times: number[]; now: number }) {
  // Последние 2 с времени мозга: каждая черта — спайк.
  const span = 2000;
  return (
    <svg className="raster" viewBox={`0 0 ${span} 20`} preserveAspectRatio="none">
      {times.filter((t) => now - t < span).map((t, i) => (
        <line key={i} x1={span - (now - t)} x2={span - (now - t)} y1={2} y2={18} />
      ))}
    </svg>
  );
}

export function SynapseCard({ synapse, onNeuron, onClose }: {
  synapse: SynapseInfo;
  onNeuron: (i: number) => void;
  onClose: () => void;
}) {
  return (
    <div className="card">
      <header>
        <h3>Синапс №{synapse.id}</h3>
        <button type="button" className="close" onClick={onClose} aria-label="закрыть">×</button>
      </header>
      <dl>
        <Row name="путь" value={synapse.path} hint="анатомический путь, к которому принадлежит синапс (откуда → куда)" />
        <div className="kv">
          <dt>от</dt>
          <dd><button type="button" className="link" onClick={() => onNeuron(synapse.pre)}>{synapse.preName} №{synapse.pre}</button></dd>
        </div>
        <div className="kv">
          <dt>к</dt>
          <dd><button type="button" className="link" onClick={() => onNeuron(synapse.post)}>{synapse.postName} №{synapse.post}</button></dd>
        </div>
        <Row name="вес" value={f(synapse.weight, 4)} hint="в единицах движка; в мВ — строка «сила»" />
        <Row name="сила" value={`${f(synapse.psp, 3)} мВ`} hint="пик ВПСП/ТПСП у получателя в покое" />
        <Row name="задержка" value={`${f(synapse.delayMs, 1)} мс`} hint="время пробега спайка по аксону и через синапс" />
        <Row name="куда" value={synapse.dendrite ? `ветка дендрита ${synapse.branch}` : "сома"} hint="на ветку — участвует в её плато; на сому — действует напрямую" />
        <Row name="ацетилхолин" value={synapse.achSuppressed ? "подавляет передачу" : "не действует"} hint="ацетилхолин глушит внутренние (рекуррентные) пути, чтобы новое записывалось, а не вспоминалось старое (Hasselmo)" />
        <Row name="учится" hint="правило: STDP даёт след, третий фактор (дофамин, ацетилхолин, лазающее волокно) превращает его в изменение веса" value={synapse.plastic ? MODULATOR[synapse.modulator ?? ""] ?? String(synapse.modulator) : "врождённый, не учится"} />
        {synapse.plastic && (
          <>
            <Row name="при рождении" value={`${f(synapse.birthPsp, 3)} мВ`} hint="сила, с которой синапс вырос по анатомии" />
            <Row name="выучено" value={`${f(synapse.learnedPsp, 4)} мВ`} hint="насколько обучение сдвинуло силу от рождения: плюс — усилился, минус — ослаб" />
            <Row name="из него свежее" value={`${f(synapse.earlyPsp, 4)} мВ`} hint="ещё не закреплено сном" />
            <Row name="след STDP" value={f(synapse.eligibility, 5)} hint="ждёт третьего фактора" />
            <Row name="потолок" value={`${f(synapse.capPsp, 2)} мВ`} hint="сильнее синапс не станет — предел размера шипика" />
          </>
        )}
      </dl>
    </div>
  );
}

type Tab = "cell" | "dendrite" | "in" | "out";

export function NeuronCard({ index, fluo, now, onNeuron, onSynapse, onClose, onLoaded }: {
  index: number;
  fluo: Fluorescence;
  now: number;
  onNeuron: (i: number) => void;
  onSynapse: (s: SynapseInfo) => void;
  onClose: () => void;
  onLoaded: (info: NeuronInfo) => void;
}) {
  const [info, setInfo] = useState<NeuronInfo | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<Tab>("cell");

  // Карточка обновляется раз в секунду, пока открыта (чтение — рядом с
  // потоком мозга, сеть не ждёт).
  useEffect(() => {
    let alive = true;
    const load = () =>
      api.neuron(index).then((fresh) => {
        if (!alive) return;
        setInfo(fresh);
        setError(null);
        onLoaded(fresh);
      }).catch((e) => alive && setError((e as Error).message));
    load();
    const timer = window.setInterval(load, 1000);
    return () => {
      alive = false;
      window.clearInterval(timer);
    };
  }, [index, onLoaded]);

  if (!info) return <div className="card">{error ?? "читаю клетку…"}</div>;
  const m = info.model;
  const s = info.state;

  return (
    <div className="card">
      <header>
        <h3>
          {info.areaTitle} · {info.population} №{info.local}
        </h3>
        <button type="button" className="close" onClick={onClose} aria-label="закрыть">×</button>
      </header>
      <p className="muted">
        {KIND_TITLE[info.kind] ?? info.kind} · {m.type} · клетка №{info.index}
      </p>
      <Raster times={fluo.watched.get(info.index) ?? []} now={now} />
      <nav className="tabs">
        {([["cell", "клетка"], ["dendrite", "дендрит"], ["in", `входы (${info.incoming.reduce((a, g) => a + g.count, 0)})`],
           ["out", `выходы (${info.outgoing.reduce((a, g) => a + g.count, 0)})`]] as [Tab, string][]).map(([key, title]) => (
          <button key={key} type="button" className={tab === key ? "on" : ""} onClick={() => setTab(key)}>{title}</button>
        ))}
      </nav>

      {tab === "cell" && (
        <>
          <h4>Сейчас</h4>
          <dl>
            <Row name="мембрана" value={`${f(s.v, 1)} мВ`} hint="потенциал сомы сейчас; дошёл до порога — спайк, затем сброс" />
            <Row name="до порога в покое" value={`${f(m.gapMv, 1)} мВ`} hint="порог − покой − ток покоя (фон, согласование)" />
            <Row name="возбуждающий ток" value={`${f(s.iSyn, 2)} мВ`} hint="сумма входящих ВПСП (AMPA), затухает с τ синапса" />
            <Row name="тормозной ток (ГАМК-А)" value={`${f(s.iInh, 2)} мВ`} hint="проводимость: к соме × (v − E_ГАМК)" />
            <Row name="медленное торможение (LAMP5)" value={`${f(s.iNgf, 2)} мВ`} hint="объёмное ГАМК от нейроглиаформных клеток: медленное, на все ветки сразу" />
            {s.iNmda !== null && <Row name="NMDA" value={`${f(s.iNmda, 2)} мВ`} hint="медленный ток NMDA: открывается, только когда клетка уже деполяризована (магниевый блок); нужен для обучения" />}
            <Row name="адаптация" value={`${f(s.adapt, 2)} мВ`} hint="каждый спайк копит ток, тянущий вниз: частая стрельба сама себя тормозит" />
            {s.tGate !== null && <Row name="T-ток, ворота h" value={f(s.tGate, 2)} hint="кальциевый T-ток таламуса: после торможения даёт пачку спайков (сон, веретёна)" />}
            <Row name="рефрактерность" value={`${f(s.refractory, 1)} мс`} hint="сколько ещё клетка не может выстрелить после спайка" />
            <Row name="ресурс синапсов" value={f(s.resource, 2)} hint="доля готовых пузырьков медиатора; частая стрельба его истощает (депрессия путей ядер)" />
            <Row name="усиление торможения" value={f(s.inhGain, 3)} hint="торможение на клетку медленно подстраивается, чтобы она стреляла с частотой по записям (гомеостаз)" />
            <Row name="след STDP пре / пост" value={`${f(s.tracePre, 3)} / ${f(s.tracePost, 3)}`} hint="память о недавних спайках входа и самой клетки: по их порядку синапс решает, усилиться или ослабнуть" />
          </dl>
          <h4>Модель</h4>
          <dl>
            <Row name="τ мембраны" value={`${f(m.tauM, 1)} мс`} hint="как быстро мембрана забывает вход: больше τ — дольше копит" />
            <Row name="τ синапса" value={`${f(m.tauSyn, 1)} мс`} hint="сколько длится ток одного синаптического события" />
            <Row name="покой / порог / сброс" value={`${m.vRest} / ${m.vThreshold} / ${m.vReset} мВ`} hint="куда мембрана стремится без входа / где спайк / куда падает после спайка" />
            {m.type === "AdEx" && <Row name="ΔT" value={`${f(m.deltaT, 1)} мВ`} hint="крутизна разгона к спайку у AdEx: натриевые каналы открываются лавиной" />}
            <Row name="адаптация за спайк" value={`${f(m.adaptB, 3)} мВ`} hint="на сколько каждый спайк добавляет тормозящий ток адаптации" />
            <Row name="рефрактерность" value={`${m.refractoryMs} мс`} hint="мёртвое время после спайка" />
          </dl>
          {m.stp && (
            <>
              <h4>Кратковременная пластичность выходов</h4>
              <table className="grid">
                <thead><tr><th>к типу</th><th title="доля ресурса, которую тратит один спайк в покое">U</th><th title="текущая доля трат: растёт при облегчении">u</th><th title="запас медиатора: падает при депрессии">ресурс x</th></tr></thead>
                <tbody>{m.stp.map((r) => <tr key={r.to}><td>{r.to}</td><td>{f(r.U)}</td><td>{f(r.u)}</td><td>{f(r.x)}</td></tr>)}</tbody>
              </table>
            </>
          )}
          <h4>Вещества</h4>
          {info.receptors.length ? (
            <table className="grid">
              <thead><tr><th>вещество</th><th title="доля занятых рецепторов этого вещества">занятость</th><th title="сдвиг мембраны, если все рецепторы заняты">при полной, мВ</th><th title="сдвиг мембраны сейчас = при полной × занятость">сейчас, мВ</th></tr></thead>
              <tbody>
                {info.receptors.map((r) => (
                  <tr key={r.substance}><td><span className="hint" title={SUBSTANCE_HINT[r.substance]}>{SUBSTANCE[r.substance] ?? r.substance}</span></td><td>{f(r.occupancy)}</td><td>{f(r.mvFull, 1)}</td><td>{f(r.mvNow, 2)}</td></tr>
                ))}
              </tbody>
            </table>
          ) : <p className="muted">рецепторов в модели нет</p>}
          <h4>Вход извне модели</h4>
          <dl>
            <Row name="фон" value={`${f(info.background.mvMean, 2)} мВ в среднем`} hint="Пуассон: внешний вход колонки или фон ядра" />
          </dl>
          {info.meanField.length > 0 && (
            <table className="grid">
              <thead><tr><th>недостающие соседи из</th><th>мВ сейчас</th></tr></thead>
              <tbody>{info.meanField.map((mf, i) => <tr key={i}><td>{mf.source}{mf.slow ? " (медленно)" : ""}</td><td>{f(mf.mv, 2)}</td></tr>)}</tbody>
            </table>
          )}
        </>
      )}

      {tab === "dendrite" && (
        <>
          <p className="muted">
            Плато ветки — сигмоида (ток − {info.dendrite.thresholdMv} мВ) / {info.dendrite.slopeMv}; горячая ветка
            усиливает вход ×{info.dendrite.gain} и поднимает сому на {f(info.dendrite.somaMv, 1)} мВ.
          </p>
          <table className="grid">
            <thead><tr><th>ветка</th><th title="вход, пришедший на эту ветку">ток, мВ</th><th title="0 — ветка спокойна, 1 — NMDA-плато: ветка «горит» и усиливает свои входы">плато</th></tr></thead>
            <tbody>
              {info.dendrite.branches.map((b, i) => (
                <tr key={i}>
                  <td>{i}</td>
                  <td>{f(b.current, 2)}</td>
                  <td><span className="bar"><span style={{ width: `${b.plateau * 100}%` }} /></span> {f(b.plateau)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <h4>Что садится на ветки</h4>
          <Groups groups={info.incoming.filter((g) => g.dendrite)} direction="in" />
        </>
      )}

      {tab === "in" && (
        <>
          <Groups groups={info.incoming} direction="in" />
          <h4>Самые сильные входы</h4>
          <Synapses list={info.topIncoming} onSynapse={onSynapse} onNeuron={onNeuron} direction="in" />
        </>
      )}
      {tab === "out" && (
        <>
          <Groups groups={info.outgoing} direction="out" />
          <h4>Самые сильные выходы</h4>
          <Synapses list={info.topOutgoing} onSynapse={onSynapse} onNeuron={onNeuron} direction="out" />
        </>
      )}
    </div>
  );
}
