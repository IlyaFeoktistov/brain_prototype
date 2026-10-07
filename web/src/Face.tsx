import { useEffect, useRef, useState } from "react";
import type { EmotionState } from "./brain";

interface Props {
  emotion: EmotionState | null;
}

interface Pose {
  valence: number;
  arousal: number;
  confusion: number;
}

// Лицо плавно тянется к текущей эмоции, а не прыгает.
function useEased(target: Pose): Pose {
  const [pose, setPose] = useState(target);
  const current = useRef(target);

  useEffect(() => {
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (reduce) {
      current.current = target;
      setPose(target);
      return;
    }
    let frame = 0;
    const tick = () => {
      const c = current.current;
      const next = {
        valence: c.valence + (target.valence - c.valence) * 0.12,
        arousal: c.arousal + (target.arousal - c.arousal) * 0.12,
        confusion: c.confusion + (target.confusion - c.confusion) * 0.12,
      };
      current.current = next;
      setPose(next);
      const done =
        Math.abs(next.valence - target.valence) +
          Math.abs(next.arousal - target.arousal) +
          Math.abs(next.confusion - target.confusion) <
        0.005;
      if (!done) frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [target.valence, target.arousal, target.confusion]);

  return pose;
}

// Лицо из трёх чисел: настроение — изгиб рта, возбуждение —
// раскрытие глаз, растерянность — разъехавшиеся брови.
export function Face({ emotion }: Props) {
  const pose = useEased({
    valence: emotion?.valence ?? 0,
    arousal: emotion?.arousal ?? 0.2,
    confusion: emotion?.confusion ?? 0,
  });

  const { valence: v, arousal: a, confusion: c } = pose;
  const eyeOpen = 1.5 + a * 5.5;
  const surprised = emotion?.name === "удивление";
  const mouthCurve = v * 14;
  // Злость: внутренние концы бровей вниз; грусть: вверх.
  const browInner = v < 0 ? (a > 0.4 ? 4 : -4) * Math.min(1, -v * 2) : 0;

  return (
    <figure className="face" aria-label={emotion ? `Мозг ${emotion.title}` : "Эмоции выключены"}>
      <svg viewBox="0 0 100 100" role="img">
        <circle cx="50" cy="52" r="40" className="face-head" />
        {/* брови */}
        <line x1="26" y1={30 - c * 6} x2="42" y2={30 + browInner + c * 4} className="face-line" />
        <line x1="58" y1={30 + browInner - c * 2} x2="74" y2={30 + c * 6} className="face-line" />
        {/* глаза */}
        <ellipse cx="34" cy="42" rx="5" ry={eyeOpen} className="face-eye" />
        <ellipse cx="66" cy="42" rx={5 + c * 1.5} ry={eyeOpen + c * 2} className="face-eye" />
        {/* рот */}
        {surprised ? (
          <ellipse cx="50" cy="72" rx="6" ry={4 + a * 4} className="face-eye" />
        ) : (
          <path
            d={`M32,${70 - c * 2} Q50,${70 + mouthCurve} 68,${70 + c * 3}`}
            className="face-line"
          />
        )}
      </svg>
      <figcaption>{emotion ? emotion.title : "эмоции выключены"}</figcaption>
    </figure>
  );
}
