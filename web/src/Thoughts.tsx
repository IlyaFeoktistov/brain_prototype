import { useEffect, useRef, useState } from "react";
import type { Press } from "./brain";

// «Мысли»: что рука печатала — в ответ собеседнику и сама, в тишине.
// Нажатия идут цепочками: новая цепочка — после паузы или когда ответ
// сменился тишиной. У каждого нажатия видно, из чего рука выбирала: доля
// выстреливших клеток движения у четырёх лучших клавиш (нажата та, у
// которой больше и не меньше половины, brain/body.Hand).

// Пауза (мс жизни мозга), после которой нажатия — новая цепочка.
const GAP_MS = 1000;

export function Thoughts({ presses, live }: { presses: Press[]; live: boolean }) {
  const log = useRef<HTMLDivElement>(null);
  const [picked, setPicked] = useState<Press | null>(null);

  useEffect(() => {
    log.current?.scrollTo({ top: log.current.scrollHeight });
  }, [presses]);

  const chains: Press[][] = [];
  for (const p of presses) {
    const chain = chains[chains.length - 1];
    const last = chain?.[chain.length - 1];
    if (last && last.reply === p.reply && p.t - last.t < GAP_MS) chain.push(p);
    else chains.push([p]);
  }
  const shown = picked && presses.includes(picked) ? picked : presses[presses.length - 1];

  return (
    <div className="thoughts">
      <div className="thoughts-log" ref={log} aria-live="polite" aria-label="Нажатия руки">
        {chains.length === 0 ? (
          <p className="muted">
            {live ? "Пока рука ничего не нажала." : "Здесь видно, что печатает рука — в ответ и сама, в тишине (флаг «жить в тишине» в меню)."}
          </p>
        ) : (
          chains.map((chain) => (
            <p key={chain[0].t} className={chain[0].reply ? "chain" : "chain chain-silent"}>
              <span className="chain-label">{chain[0].reply ? "в ответ" : "в тишине"}</span>
              {chain.map((p, i) => (
                <button key={i} type="button" className={p === shown ? "press on" : "press"} onClick={() => setPicked(p)}
                        title={p.choices.map(([c, v]) => `«${c}» ${Math.round(v * 100)}%`).join(", ")}>
                  {p.char === " " ? "␣" : p.char}
                </button>
              ))}
            </p>
          ))
        )}
      </div>
      {shown && (
        <p className="choices" title="доля выстреливших клеток движения клавиши">
          <strong>«{shown.char === " " ? "␣" : shown.char}» выбрано из:</strong>
          {shown.choices.map(([c, v], i) => (
            <span key={i} className={c === shown.char ? "choice-best" : ""}>
              {c === " " ? "␣" : c} {Math.round(v * 100)}%
            </span>
          ))}
        </p>
      )}
    </div>
  );
}
