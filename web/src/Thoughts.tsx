import { useEffect, useRef } from "react";

// Окошко «Мысли» в чате: ход размышления виден по шагам.
// Шаг — слово-кандидат и вердикт проверяющего; «мысль» — итог
// блуждания в тишине (живой режим) или ответа с /think on.

export interface ThoughtEntry {
  id: number;
  kind: "step" | "musing" | "choice";
  text: string;
  verdict?: "ok" | "no" | "stop";
  score?: number;
  // Выбор слова: кандидаты с вероятностями.
  candidates?: [string, number][];
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
  const chains: { steps: ThoughtEntry[]; result?: ThoughtEntry; choices?: ThoughtEntry[] }[] = [];
  let current: ThoughtEntry[] = [];
  let choices: ThoughtEntry[] = [];
  for (const entry of entries) {
    if (entry.kind === "step") current.push(entry);
    else if (entry.kind === "choice") choices.push(entry);
    else {
      chains.push({ steps: current, result: entry, choices });
      current = [];
      choices = [];
    }
  }
  if (current.length || choices.length) chains.push({ steps: current, choices });

  return (
    <div className="thoughts-log" ref={log} aria-live="polite" aria-label="Ход мыслей">
      {chains.length === 0 ? (
        <p className="thoughts-empty">
          {live || thinking
            ? "Пока не думал."
            : "Здесь виден ход мыслей и выбор каждого слова ответа. Живой режим — думает в тишине, /think on — обдумывает ответ."}
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
            {chain.choices?.map((choice) => (
              <p key={choice.id} className="thought-choice" title="из чего выбирал слово (вероятности)">
                <strong>{choice.text}</strong>
                {choice.candidates!.map(([word, chance], i) => (
                  <span key={word} className={i === 0 ? "choice-best" : ""}>
                    {word} {Math.round(chance * 100)}%
                  </span>
                ))}
              </p>
            ))}
            {chain.result && <p className="thought-result">→ {chain.result.text || "ничего не придумал"}</p>}
          </div>
        ))
      )}
    </div>
  );
}
