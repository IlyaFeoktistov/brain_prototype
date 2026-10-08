import { useCallback, useEffect, useState } from "react";

// Перетаскиваемая граница: меняет размер (ширину или высоту) и
// запоминает его в браузере. Стрелки на клавиатуре — по 24 px.

export function useStoredSize(key: string, initial: number): [number, (value: number) => void] {
  const [size, setSize] = useState(() => {
    try {
      const saved = Number(localStorage.getItem(key));
      return saved > 0 ? saved : initial;
    } catch {
      return initial;
    }
  });

  const update = useCallback(
    (value: number) => {
      setSize(value);
      try {
        localStorage.setItem(key, String(Math.round(value)));
      } catch {
        // Без хранилища размер просто не запомнится.
      }
    },
    [key],
  );

  return [size, update];
}

interface Props {
  // "x": граница вертикальная, тянем влево-вправо.
  axis: "x" | "y";
  size: number;
  min: number;
  max: () => number;
  // +1: размер растёт, когда тянем вправо/вниз; -1 — влево/вверх.
  direction: 1 | -1;
  onResize: (size: number) => void;
  label: string;
}

const STEP = 24;

export function Splitter({ axis, size, min, max, direction, onResize, label }: Props) {
  const [drag, setDrag] = useState<{ from: number; size: number } | null>(null);
  const clamp = useCallback((value: number) => Math.max(min, Math.min(max(), value)), [min, max]);

  useEffect(() => {
    if (!drag) return;
    const move = (event: PointerEvent) => {
      const at = axis === "x" ? event.clientX : event.clientY;
      onResize(clamp(drag.size + direction * (at - drag.from)));
    };
    const up = () => setDrag(null);
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
    document.body.classList.add(axis === "x" ? "resizing-x" : "resizing-y");
    return () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
      document.body.classList.remove("resizing-x", "resizing-y");
    };
  }, [drag, axis, direction, clamp, onResize]);

  const grow = axis === "x" ? (direction > 0 ? "ArrowRight" : "ArrowLeft") : direction > 0 ? "ArrowDown" : "ArrowUp";
  const shrink = axis === "x" ? (direction > 0 ? "ArrowLeft" : "ArrowRight") : direction > 0 ? "ArrowUp" : "ArrowDown";

  return (
    <div
      className={`splitter splitter-${axis}${drag ? " splitter-active" : ""}`}
      role="separator"
      aria-orientation={axis === "x" ? "vertical" : "horizontal"}
      aria-label={label}
      aria-valuenow={Math.round(size)}
      aria-valuemin={min}
      tabIndex={0}
      onPointerDown={(event) => {
        event.preventDefault();
        setDrag({ from: axis === "x" ? event.clientX : event.clientY, size });
      }}
      onKeyDown={(event) => {
        if (event.key === grow) onResize(clamp(size + STEP));
        else if (event.key === shrink) onResize(clamp(size - STEP));
        else return;
        event.preventDefault();
      }}
    />
  );
}
