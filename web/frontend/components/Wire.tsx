'use client';

import type { NodePosition, StageStatus } from '@/lib/types';

interface WireProps {
  positions: Record<string, NodePosition>;
  wires: [string, string][];
  stageStatuses: Record<string, { status: StageStatus }>;
  tick: number;
}

function getWireColor(from: string, to: string, statuses: Record<string, { status: StageStatus }>): string {
  const fromStatus = statuses[from]?.status;
  const toStatus = statuses[to]?.status;
  if (toStatus === 'done' || toStatus === 'running') return '#3b82f6';
  if (fromStatus === 'done') return '#94a3b8';
  return '#d4d4d4';
}

export function Wires({ positions, wires, stageStatuses, tick }: WireProps) {
  return (
    <svg
      style={{
        position: 'absolute',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        pointerEvents: 'none',
        zIndex: 5,
      }}
    >
      {wires.map(([from, to], i) => {
        const a = positions[from];
        const b = positions[to];
        if (!a || !b) return null;

        const aCx = a.x + a.w / 2;
        const aCy = a.y + a.h / 2;
        const bCx = b.x + b.w / 2;
        const bCy = b.y + b.h / 2;

        const dx = bCx - aCx;
        const dy = bCy - aCy;

        let x1: number, y1: number, x2: number, y2: number;

        if (Math.abs(dx) > Math.abs(dy)) {
          if (dx > 0) {
            x1 = a.x + a.w; y1 = aCy; x2 = b.x; y2 = bCy;
          } else {
            x1 = a.x; y1 = aCy; x2 = b.x + b.w; y2 = bCy;
          }
        } else {
          if (dy > 0) {
            x1 = aCx; y1 = a.y + a.h; x2 = bCx; y2 = b.y;
          } else {
            x1 = aCx; y1 = a.y; x2 = bCx; y2 = b.y + b.h;
          }
        }

        const mx = (x1 + x2) / 2;
        const my = (y1 + y2) / 2;
        const d = Math.abs(dx) > Math.abs(dy)
          ? `M ${x1} ${y1} C ${mx} ${y1}, ${mx} ${y2}, ${x2} ${y2}`
          : `M ${x1} ${y1} C ${x1} ${my}, ${x2} ${my}, ${x2} ${y2}`;

        const color = getWireColor(from, to, stageStatuses);

        return (
          <g key={i}>
            <path d={d} fill="none" stroke={color} strokeWidth="1.5" strokeDasharray="6,4" />
            <circle cx={x2} cy={y2} r="3" fill={color} />
            <circle cx={x1} cy={y1} r="2.5" fill="none" stroke={color} strokeWidth="1" />
          </g>
        );
      })}
    </svg>
  );
}
