import { useEffect, useRef } from "react";

// Окошко «Мысли» в чате: ход размышления виден по шагам.
// Шаг — слово-кандидат и вердикт проверяющего; «мысль» — итог
// блуждания в тишине (живой режим) или ответа с /think on.

export interface ThoughtEntry {
  id: number;
  kind: "step" | "musing";
  text: string;
  verdict?: "ok" | "no" | "stop";
  score?: number;
}

const VERDICT: Record<string, { mark: string; title: string }> = {
  ok: { mark: "✓", title: "проверяющий принял" },
  no: { mark: "✗", title: "проверяющий отверг" },
  stop: { mark: "■", title: "не уверен — остановился" },
};

interface Props {
  entries: ThoughtEntry[];
  live: boolean;
  thinking: boolean;
}

export function Thoughts({ entries, live, thinking }: Props) {
  const log = useRef<HTMLDivElement>(null);

  useEffect(() => {
    log.current?.scrollTo({ top: log.current.scrollHeight });
  }, [entries]);

  // Шаги между итогами — одна цепочка мысли.
  const chains: { steps: ThoughtEntry[]; result?: ThoughtEntry }[] = [];
  let current: ThoughtEntry[] = [];
  for (const entry of entries) {
    if (entry.kind === "step") current.push(entry);
    else {
      chains.push({ steps: current, result: entry });
      current = [];
    }
  }
  if (current.length) chains.push({ steps: current });

  return (
    <div className="thoughts-log" ref={log} aria-live="polite" aria-label="Ход мыслей">
      {chains.length === 0 ? (
        <p className="thoughts-empty">
          {live || thinking
            ? "Пока не думал."
            : "Здесь виден ход мыслей: включи «Живой режим» (думает в тишине) или /think on (обдумывает ответ)."}
        </p>
      ) : (
        chains.map((chain, i) => (
          <div key={chain.result?.id ?? `open-${i}`} className="thought-chain">
            {chain.steps.map((step) => {
              const verdict = VERDICT[step.verdict ?? "ok"];
              return (
                <span
                  key={step.id}
                  className={`thought thought-${step.verdict}`}
                  title={`${verdict.title}, оценка ${step.score?.toFixed(2)}`}
                >
                  {step.text}
                  <span className="thought-mark" aria-label={verdict.title}>
                    {verdict.mark}
                  </span>
                </span>
              );
            })}
            {chain.result && <p className="thought-result">→ {chain.result.text || "ничего не придумал"}</p>}
          </div>
        ))
      )}
    </div>
  );
}
