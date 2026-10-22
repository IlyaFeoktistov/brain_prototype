import type { Fixation } from "./brain";

// Лента фиксаций: слово, на котором остановился взгляд (буква под точкой
// взгляда выделена), и под ним — пик дофамина за фиксацию относительно
// покоя: столбик вверх — выброс («лучше, чем ждал»), вниз — провал.
// Буквы, напечатанные рукой, — курсивом.

// Отклонение занятости, при котором столбик в полную высоту.
const FULL = 0.3;

export function Strip({ fixations }: { fixations: Fixation[] }) {
  if (!fixations.length) {
    return <p className="strip-empty muted">Здесь побегут слова, на которых останавливается взгляд, и выбросы дофамина на каждом.</p>;
  }

  return (
    <ol className="strip" aria-label="Последние фиксации взгляда и нажатия">
      {fixations.map((f, i) => {
        const level = Math.max(-1, Math.min(1, f.dopamine / FULL));
        const title =
          (f.typed ? `рука напечатала «${f.word}»` : f.word ? `взгляд на «${f.word}», буква ${f.letter + 1}` : "взгляд за строкой") +
          `; дофамин ${f.dopamine >= 0 ? "+" : ""}${f.dopamine.toFixed(3)} к покою`;
        return (
          <li key={i} className={f.typed ? "fix fix-typed" : "fix"} title={title}>
            <span className="fix-word">
              {f.typed || !f.word ? (
                f.word || "·"
              ) : (
                <>
                  {f.word.slice(0, f.letter)}
                  <mark>{f.word[f.letter]}</mark>
                  {f.word.slice(f.letter + 1)}
                </>
              )}
            </span>
            <span className="fix-dopamine">
              <span className={level >= 0 ? "up" : "down"} style={{ height: `${Math.abs(level) * 50}%` }} />
            </span>
          </li>
        );
      })}
    </ol>
  );
}
