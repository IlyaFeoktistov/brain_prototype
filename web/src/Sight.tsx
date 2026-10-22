import { useEffect, useRef } from "react";
import type { View } from "./brain";

// Глаз и рука: что видит мозг на каждом этапе зрительного пути и куда
// тянется рука. Картинки — обратные преобразования по проводке самого мозга
// (brain/view.py): спайк клетки рисуется в пикселе, откуда она получает вход.

// Пикселей экрана на пиксель картинки.
const SCALE = 6;

type Paint = (value: number) => [number, number, number];

// Яркость: 1 — фон (светлый), 0 — чернила.
const gray: Paint = (v) => {
  const c = Math.round(255 * Math.max(0, Math.min(1, v)));
  return [c, c, c];
};

// Со знаком: + светлое (ON) — бирюзовое, − тёмное (OFF) — пурпурное.
const signed = (scale: number): Paint => (v) => {
  const a = Math.max(-1, Math.min(1, v / scale));
  return a >= 0
    ? [Math.round(250 - 230 * a), Math.round(250 - 70 * a), Math.round(250 - 100 * a)]
    : [Math.round(250 + 60 * a), Math.round(250 + 220 * a), Math.round(250 + 130 * a)];
};

function Picture({ values, rows, cols, paint, scale = SCALE, frame }: {
  values: number[];
  rows: number;
  cols: number;
  paint: Paint;
  scale?: number;
  frame?: [number, number]; // столбцы рамки ямки на боковом поле
}) {
  const canvas = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const ctx = canvas.current?.getContext("2d");
    if (!ctx) return;
    const image = ctx.createImageData(cols, rows);
    values.forEach((v, i) => {
      const [r, g, b] = paint(v);
      image.data.set([r, g, b, 255], i * 4);
    });
    ctx.putImageData(image, 0, 0);
  }, [values, rows, cols, paint]);

  return (
    <div className="picture" style={{ width: cols * scale, height: rows * scale }}>
      <canvas ref={canvas} width={cols} height={rows} />
      {frame && <span className="fovea-frame" style={{ left: frame[0] * scale, width: (frame[1] - frame[0]) * scale }} />}
    </div>
  );
}

const peak = (values: number[]) => Math.max(1, ...values.map(Math.abs));

export function Sight({ view, keyboard }: { view: View | null; keyboard: string[] }) {
  if (!view) return <div className="panel"><p className="muted">Жду первый снимок глаза…</p></div>;
  const { rows, cols, sideCols } = view;
  const side = (sideCols - cols) / 2;
  let key = 0;

  return (
    <div className="panel sight">
      <h4>Боковое зрение</h4>
      <p className="muted">Поле вокруг точки взгляда, в рамке — ямка. Двухолмие видит его грубо и выбирает, куда прыгнуть.</p>
      <Picture values={view.side} rows={rows} cols={sideCols} paint={gray} scale={1.5} frame={[side, side + cols]} />

      <h4>Ямка — картинка</h4>
      <p className="muted">Что сейчас на сетчатке в полном разрешении.</p>
      <Picture values={view.picture} rows={rows} cols={cols} paint={gray} />

      <h4>Сетчатка</h4>
      <p className="muted">Ганглиозные клетки: центр против окружения (Kuffler 1953). Бирюзовое — ON (светлее окружения), пурпурное — OFF (темнее). Ровный фон молчит, отвечают края штрихов.</p>
      <Picture values={view.retina} rows={rows} cols={cols} paint={signed(1)} />

      <h4>Таламус (LGN)</h4>
      <p className="muted">Спайки клеток реле за последние ~0.4 с, каждая — в пикселе своего рецептора. Реле стреляют и без картинки (~10 Гц), слово — там, где чаще.</p>
      <Picture values={view.lgn} rows={rows} cols={cols} paint={signed(peak(view.lgn))} />

      <h4>Зрительная кора, L4</h4>
      <p className="muted">Спайки простых клеток, нарисованные по их полям: «+» подполе OFF — тёмный штрих, ON — светлые бока (Hubel & Wiesel 1962). Здесь видно, какие штрихи кора нашла.</p>
      <Picture values={view.v1} rows={rows} cols={cols} paint={signed(peak(view.v1))} />

      <h4>Рука</h4>
      <p className="muted">На клавише — доля клеток её движения, стрелявших за последние ~0.2 с. Нажатие решает окно в 10 мс: клавиша, где выстрелило больше всех и не меньше половины; обведена клавиша, нажатая сейчас.</p>
      <div className="keyboard">
        {keyboard.map((row) => (
          <div key={row} className="keys">
            {[...row].map((char) => {
              const i = key++;
              const share = view.keys[i] ?? 0;
              return (
                <span key={i} className={i === view.held ? "key held" : "key"} title={`«${char}»: ${Math.round(share * 100)}% клеток`}
                      style={{ "--share": share } as React.CSSProperties}>
                  {char === " " ? "пробел" : char}
                </span>
              );
            })}
          </div>
        ))}
      </div>
    </div>
  );
}
