import { useLayoutEffect, useRef, useState } from "react";
import type { Anatomy, NeuronCard as Card, SynapseGroup } from "./brain";

// Карточка нейрона под курсором: что знает одна клетка
// (на какие слова отвечает, что предсказывает, кто её будит).

interface Props {
  anatomy: Anatomy;
  card: Card | null;
  loading: boolean;
  error: string | null;
  x: number;
  y: number;
  pinned: boolean;
  onClose: () => void;
}

const SHORT: Record<string, string> = {
  auditory: "слух",
  association: "смысл",
  prefrontal: "раб. память",
  motor: "рот",
  limbic: "чувства",
};

function population(name: string): string {
  const [area, kind] = name.split(".");
  return `${SHORT[area] ?? area} ${kind === "I" ? "(торм.)" : ""}`.trim();
}

// Слова слабее 5% от сильнейшего — шум, не знание клетки.
const VISIBLE_SHARE = 0.05;

function Bars({ items, empty }: { items: { word: string; value: number }[]; empty: string }) {
  const max = Math.max(...items.map((item) => item.value), 1e-9);
  items = items.filter((item) => item.value >= max * VISIBLE_SHARE);
  if (!items.length) return <p className="neuron-empty">{empty}</p>;
  return (
    <ol className="neuron-bars">
      {items.map((item) => (
        <li key={item.word}>
          <span className="neuron-word">{item.word}</span>
          <span className="neuron-bar" style={{ width: `${(item.value / max) * 100}%` }} />
          <span className="neuron-value">{item.value < 0.1 ? item.value.toFixed(3) : item.value.toFixed(2)}</span>
        </li>
      ))}
    </ol>
  );
}

function Synapses({ groups, title }: { groups: SynapseGroup[]; title: string }) {
  return (
    <table className="neuron-table">
      <caption>{title}</caption>
      <thead>
        <tr>
          <th>откуда / куда</th>
          <th>шт.</th>
          <th>ср. вес</th>
          <th title="выученные: сильнее / слабее, чем при рождении">↑ / ↓</th>
        </tr>
      </thead>
      <tbody>
        {groups.map((group) => (
          <tr key={group.population}>
            <td>{population(group.population)}</td>
            <td>{group.count}</td>
            <td>{group.mean.toFixed(2)}</td>
            <td>{group.plastic ? `${group.stronger} / ${group.weaker}` : "врожд."}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function NeuronCard({ anatomy, card, loading, error, x, y, pinned, onClose }: Props) {
  const ref = useRef<HTMLDivElement>(null);
  const [place, setPlace] = useState({ left: x + 16, top: y + 16 });

  // Не вылезать за край окна.
  useLayoutEffect(() => {
    const box = ref.current?.getBoundingClientRect();
    if (!box) return;
    const left = x + 16 + box.width > window.innerWidth - 8 ? Math.max(8, x - 16 - box.width) : x + 16;
    const top = Math.max(8, Math.min(y + 16, window.innerHeight - 8 - box.height));
    setPlace({ left, top });
  }, [x, y, card, loading]);

  const area = anatomy.areas.find((a) => a.name === card?.area);

  return (
    <div
      ref={ref}
      className={`neuron-card${pinned ? " neuron-card-pinned" : ""}`}
      style={{ left: place.left, top: place.top }}
      role="dialog"
      aria-label="нейрон"
    >
      {error ? (
        <p className="neuron-empty">{error}</p>
      ) : !card ? (
        <p className="neuron-empty">{loading ? "смотрю в микроскоп…" : ""}</p>
      ) : (
        <>
          <header className="neuron-head">
            <div>
              <strong>нейрон №{card.index.toLocaleString("ru")}</strong>
              <span className={`neuron-type neuron-type-${card.type}`}>
                {card.type === "E" ? "возбуждающий" : "тормозный"}
              </span>
            </div>
            <div className="neuron-sub">
              {area?.title ?? card.title} · №{card.local.toLocaleString("ru")} в зоне
            </div>
            {pinned && (
              <button className="neuron-close" onClick={onClose} aria-label="закрыть">
                ×
              </button>
            )}
          </header>

          {card.feeling && (
            <p className="neuron-feeling">
              клетка чувства: <strong>{card.feeling}</strong>
            </p>
          )}

          <dl className="neuron-state">
            <div>
              <dt>потенциал</dt>
              <dd>{card.state.v.toFixed(1)} мВ</dd>
            </div>
            <div>
              <dt title="сдвиг возбудимости от гомеостаза">возбудимость</dt>
              <dd>{card.state.bias >= 0 ? "+" : ""}{card.state.bias.toFixed(1)} мВ</dd>
            </div>
            <div>
              <dt title="на последнем слове">частота</dt>
              <dd>{card.state.rate === null ? "—" : `${card.state.rate.toFixed(0)} Гц`}</dd>
            </div>
          </dl>

          {card.tuning !== undefined && (
            <section>
              <h4>
                отвечает на слова
                {card.tuningWords ? <span> · в ансамблях {card.tuningWords}</span> : null}
              </h4>
              <Bars
                items={card.tuning.map((t) => ({ word: t.word, value: t.share }))}
                empty="ни в одном слове пока нет"
              />
            </section>
          )}

          <section>
            <h4 title="куда толкают выученные исходящие синапсы — какие слова он будит следующими">
              предсказывает
            </h4>
            <Bars
              items={card.predicts.map((t) => ({ word: t.word, value: t.score }))}
              empty={card.type === "I" ? "тормозный — не предсказывает" : "ещё ничего не выучил"}
            />
          </section>

          <section>
            <h4 title="от каких слов выучены входящие связи">будят его</h4>
            <Bars
              items={card.wokenBy.map((t) => ({ word: t.word, value: t.score }))}
              empty="выученных входов от слов нет"
            />
          </section>

          <Synapses groups={card.incoming} title="входящие синапсы" />
          <Synapses groups={card.outgoing} title="исходящие синапсы" />

          {!pinned && <p className="neuron-hint">клик — закрепить</p>}
        </>
      )}
    </div>
  );
}
