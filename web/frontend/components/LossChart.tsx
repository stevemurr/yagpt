'use client';

interface LossChartProps {
  data: { step: number; loss: number }[];
  height?: number;
  color?: string;
}

export function LossChart({ data, height = 70, color = '#333' }: LossChartProps) {
  if (!data.length) return null;

  // EMA smoothing
  const smoothed: number[] = [];
  let ema = data[0].loss;
  for (const d of data) {
    ema = 0.95 * ema + 0.05 * d.loss;
    smoothed.push(ema);
  }

  const mx = Math.max(...smoothed);
  const mn = Math.min(...smoothed);
  const rng = mx - mn || 1;

  const toY = (l: number) => ((mx - l) / rng) * 90 + 5;
  const toX = (i: number) => (i / Math.max(smoothed.length - 1, 1)) * 100;

  return (
    <svg
      viewBox="0 0 100 100"
      preserveAspectRatio="none"
      style={{ width: '100%', height, display: 'block' }}
    >
      {data.length < 400 &&
        data.map((d, i) => (
          <circle
            key={i}
            cx={toX(i)}
            cy={toY(Math.max(mn, Math.min(mx, d.loss)))}
            r="0.3"
            fill="#e8e8e8"
            vectorEffect="non-scaling-stroke"
          />
        ))}
      <polyline
        fill="none"
        stroke={color}
        strokeWidth="1.2"
        points={smoothed.map((v, i) => `${toX(i)},${toY(v)}`).join(' ')}
        vectorEffect="non-scaling-stroke"
      />
      {smoothed.length > 1 && (
        <circle
          cx={toX(smoothed.length - 1)}
          cy={toY(smoothed[smoothed.length - 1])}
          r="2"
          fill={color}
          vectorEffect="non-scaling-stroke"
        />
      )}
      <text x="1" y="8" fontSize="5" fill="#ccc" vectorEffect="non-scaling-stroke">
        {mx.toFixed(2)}
      </text>
      <text x="1" y="98" fontSize="5" fill="#ccc" vectorEffect="non-scaling-stroke">
        {mn.toFixed(2)}
      </text>
    </svg>
  );
}
