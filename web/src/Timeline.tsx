import type { StepEvent } from "./brain";

interface Props {
  steps: StepEvent[];
}

// Лента услышанных слов. Под словом — выброс dopamine:
// чем неожиданнее слово, тем выше столбик.
export function Timeline({ steps }: Props) {
  if (!steps.length) {
    return (
      <p className="timeline-empty">
        Здесь побегут слова, которые слышит мозг, и выбросы дофамина на неожиданных.
      </p>
    );
  }

  return (
    <ol className="timeline" aria-label="Последние услышанные слова">
      {steps.map((step, i) => {
        const guessed = step.expected === step.token;
        const height = Math.min(1, step.dopamine / 6);
        const title =
          step.expected === null
            ? "ожидания не было"
            : `ожидал «${step.expected}»` + (step.error !== null ? `, ошибка ${step.error.toFixed(2)}` : "");
        return (
          <li
            key={i}
            className={[
              "word",
              guessed ? "word-guessed" : "",
              step.learn ? "" : "word-own",
            ].join(" ")}
            title={title}
          >
            <span className="word-text">{step.token}</span>
            <span className="word-dopamine" style={{ transform: `scaleY(${height})` }} />
          </li>
        );
      })}
    </ol>
  );
}
