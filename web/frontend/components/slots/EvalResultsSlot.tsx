'use client';

import { useStore } from '@/lib/store';
import type { SlotProps } from '@/lib/module-schema';

export function EvalResultsSlot(_props: SlotProps) {
  const evalResults = useStore((s) => s.evalResults);
  if (!evalResults) return null;

  return (
    <div style={{ borderTop: '1px solid #f0f0f0', paddingTop: '6px' }}>
      {Object.entries(evalResults).map(([task, metrics]) => (
        <div key={task} style={{ marginBottom: '6px' }}>
          <div style={{ fontSize: '9px', fontWeight: 600, color: '#555', letterSpacing: '1px', marginBottom: '2px' }}>
            {task.toUpperCase()}
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2px' }}>
            {Object.entries(metrics).map(([key, val]) => (
              <div key={key} style={{ fontSize: '9px', color: '#888' }}>
                {key.replace(/,none$/, '')}: <span style={{ fontWeight: 700, color: '#333' }}>
                  {typeof val === 'number' ? (val * 100).toFixed(1) + '%' : String(val)}
                </span>
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}
