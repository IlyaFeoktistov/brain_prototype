import { useEffect, useRef, useState } from "react";
import type { Status, Substance } from "./brain";

// Лицо — показ химии мозга, не модель эмоций: каждая черта читает одно
// вещество и сравнивает его с покоем наяву (rest от сервера).
//   рот — дофамин: выше покоя — ошибка предсказания награды «лучше, чем
//     ждал», ниже — «хуже» (Schultz, Dayan & Montague 1997,
//     https://doi.org/10.1126/science.275.5306.1593; обезьяна);
//   зрачок и раскрытие глаз — норадреналин: диаметр зрачка идёт за
//     частотой голубого пятна (Joshi et al. 2016,
//     https://doi.org/10.1016/j.neuron.2015.11.028; обезьяна);
//   веки — аденозин: давление сна копится наяву (Porkka-Heiskanen et al.
//     1997, https://doi.org/10.1126/science.276.5316.1265; кошка);
//   брови — ацетилхолин: внимание к новому (Hasselmo 2006,
//     https://doi.org/10.1016/j.conb.2006.09.002, обзор);
//   глаза закрыты — сон: VLPO (галанин) перебил LC.

interface Pose {
  mood: number;
  arousal: number;
  pressure: number;
  attention: number;
}

// Отклонение от покоя в [-1, 1]: +1 — рецепторы заняты целиком, -1 — пусты.
function deviation(c: Substance | undefined): number {
  if (!c || c.rest === null || c.rest === undefined) return 0;
  const d = c.occupancy - c.rest;
  return Math.max(-1, Math.min(1, d > 0 ? d / (1 - c.rest) : d / c.rest));
}

// Лицо тянется к химии плавно — статус приходит раз в ~0.5 с.
function useEased(target: Pose): Pose {
  const [pose, setPose] = useState(target);
  const current = useRef(target);

  useEffect(() => {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      current.current = target;
      setPose(target);
      return;
    }
    let frame = 0;
    const tick = () => {
      const c = current.current;
      const next = {
        mood: c.mood + (target.mood - c.mood) * 0.12,
        arousal: c.arousal + (target.arousal - c.arousal) * 0.12,
        pressure: c.pressure + (target.pressure - c.pressure) * 0.12,
        attention: c.attention + (target.attention - c.attention) * 0.12,
      };
      current.current = next;
      setPose(next);
      const left = Math.abs(next.mood - target.mood) + Math.abs(next.arousal - target.arousal)
        + Math.abs(next.pressure - target.pressure) + Math.abs(next.attention - target.attention);
      if (left > 0.005) frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [target.mood, target.arousal, target.pressure, target.attention]);

  return pose;
}

const sign = (x: number) => (x > 0.15 ? "выше покоя" : x < -0.15 ? "ниже покоя" : "как в покое");

export function Face({ status }: { status: Status }) {
  const chem = new Map((status.chemistry ?? []).map((c) => [c.name, c]));
  const target: Pose = {
    mood: deviation(chem.get("dopamine")),
    arousal: deviation(chem.get("noradrenaline")),
    pressure: chem.get("adenosine")?.occupancy ?? 0,
    attention: deviation(chem.get("acetylcholine")),
  };
  const { mood, arousal, pressure, attention } = useEased(target);
  const asleep = status.asleep ?? false;

  // Глаз раскрыт от норадреналина, веко прикрывает его по аденозину.
  const eyeOpen = Math.max(1, (4 + arousal * 3) * (1 - pressure * 0.6));
  const pupil = Math.max(0.8, Math.min(eyeOpen, 2 + arousal * 1.5));
  const brow = 30 - attention * 4;
  const mouth = mood * 12;

  return (
    <figure className="face">
      <svg viewBox="0 0 100 100" role="img" aria-label={asleep ? "спит" : "лицо из химии мозга"}>
        <circle cx="50" cy="52" r="40" className="face-head" />
        <line x1="26" y1={brow} x2="42" y2={brow + 1 - mood * 2} className="face-line" />
        <line x1="58" y1={brow + 1 - mood * 2} x2="74" y2={brow} className="face-line" />
        {asleep ? (
          <>
            <path d="M28,43 Q34,47 40,43" className="face-line" />
            <path d="M60,43 Q66,47 72,43" className="face-line" />
          </>
        ) : (
          <>
            <ellipse cx="34" cy="42" rx="6" ry={eyeOpen} className="face-white" />
            <ellipse cx="66" cy="42" rx="6" ry={eyeOpen} className="face-white" />
            <circle cx="34" cy="42" r={pupil} className="face-eye" />
            <circle cx="66" cy="42" r={pupil} className="face-eye" />
          </>
        )}
        <path d={`M32,70 Q50,${70 + mouth} 68,70`} className="face-line" />
      </svg>
      {asleep && (
        <span className="zzz" aria-hidden="true"><span>z</span><span>z</span><span>Z</span></span>
      )}
      <figcaption>
        <span title="рот: выше покоя — «лучше, чем ждал», ниже — «хуже» (ошибка предсказания награды)">дофамин {sign(mood)}</span>
        <span title="зрачок и раскрытие глаз: зрачок идёт за частотой голубого пятна">норадреналин {sign(arousal)}</span>
        <span title="веки: давление сна, копится наяву, вымывается во сне">аденозин {(pressure * 100).toFixed(0)}%</span>
        <span title="брови: внимание к новому, ядро Мейнерта">ацетилхолин {sign(attention)}</span>
      </figcaption>
    </figure>
  );
}
