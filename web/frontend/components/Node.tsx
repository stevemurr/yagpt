'use client';

import { ReactNode } from 'react';
import type { StageStatus } from '@/lib/types';

interface NodeProps {
  id: string;
  title: string;
  status?: StageStatus;
  accent?: string;
  onRemove?: () => void;
  children: ReactNode;
}

const STATUS_COLORS: Record<StageStatus, string> = {
  idle: '#ccc',
  running: '#3b82f6',
  done: '#22c55e',
  error: '#ef4444',
  stopping: '#f59e0b',
  stopped: '#f59e0b',
};

export function Node({ id, title, status = 'idle', accent, onRemove, children }: NodeProps) {
  const active = status !== 'idle';
  const color = accent || STATUS_COLORS[status] || '#333';

  return (
    <div
      style={{
        width: '100%',
        background: '#fff',
        borderRadius: '3px',
        border: `1px solid ${active ? color + '40' : '#e0e0e0'}`,
        boxShadow: active
          ? `0 2px 12px ${color}15`
          : '0 1px 3px rgba(0,0,0,0.04)',
        transition: 'border-color 0.3s, box-shadow 0.3s',
      }}
    >
      <div
        style={{
          padding: '8px 12px',
          borderBottom: `1px solid ${active ? color + '20' : '#f0f0f0'}`,
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
        }}
      >
        <span
          style={{
            width: 6,
            height: 6,
            borderRadius: '50%',
            background: STATUS_COLORS[status],
            flexShrink: 0,
          }}
        />
        <span
          style={{
            fontSize: '10px',
            fontWeight: 600,
            letterSpacing: '1.5px',
            color: active ? '#555' : '#bbb',
            fontFamily: "'JetBrains Mono', monospace",
            textTransform: 'uppercase',
            flex: 1,
          }}
        >
          {title}
        </span>
        {status === 'running' && (
          <span style={{ fontSize: '8px', color: '#3b82f6', letterSpacing: '1px' }}>RUNNING</span>
        )}
        {status === 'done' && (
          <span style={{ fontSize: '8px', color: '#22c55e', letterSpacing: '1px' }}>DONE</span>
        )}
        {status === 'error' && (
          <span style={{ fontSize: '8px', color: '#ef4444', letterSpacing: '1px' }}>ERROR</span>
        )}
        {onRemove && status === 'idle' && (
          <button
            onClick={(e) => { e.stopPropagation(); onRemove(); }}
            style={{
              background: 'none',
              border: 'none',
              cursor: 'pointer',
              fontSize: '12px',
              color: '#ccc',
              padding: '0 2px',
              lineHeight: 1,
              fontFamily: "'JetBrains Mono', monospace",
            }}
            title="Remove module"
          >
            ×
          </button>
        )}
      </div>
      <div style={{ padding: '12px', fontFamily: "'JetBrains Mono', monospace", fontSize: '11px' }}>
        {children}
      </div>
    </div>
  );
}
