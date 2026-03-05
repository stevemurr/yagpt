'use client';

import { useStore } from '@/lib/store';
import { ExperimentPicker } from './ExperimentPicker';

export function Header() {
  const wsConnected = useStore((s) => s.wsConnected);
  const setDataModalOpen = useStore((s) => s.setDataModalOpen);

  return (
    <div
      style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        padding: '12px 20px',
        borderBottom: '1px solid #e8e8e8',
        background: '#fff',
        fontFamily: "'JetBrains Mono', monospace",
      }}
    >
      {/* Left: logo + WS status */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '10px',
          fontSize: '10px',
          fontWeight: 700,
          letterSpacing: '3px',
          color: '#ccc',
        }}
      >
        YAGPT
        <span
          style={{
            width: 6,
            height: 6,
            borderRadius: '50%',
            background: wsConnected ? '#22c55e' : '#ef4444',
          }}
          title={wsConnected ? 'Connected' : 'Disconnected'}
        />
      </div>

      {/* Center: experiment picker */}
      <ExperimentPicker />

      {/* Right: Data button */}
      <button
        onClick={() => setDataModalOpen(true)}
        style={{
          padding: '4px 12px',
          fontSize: '9px',
          fontWeight: 600,
          letterSpacing: '1px',
          fontFamily: "'JetBrains Mono', monospace",
          background: '#fafafa',
          color: '#888',
          border: '1px solid #e0e0e0',
          borderRadius: '3px',
          cursor: 'pointer',
        }}
      >
        DATA
      </button>
    </div>
  );
}
