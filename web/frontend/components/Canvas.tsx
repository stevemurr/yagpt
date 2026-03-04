'use client';

import { useStore } from '@/lib/store';
import { useWebSocket } from '@/hooks/useWebSocket';
import { Wires } from './Wire';
import { ModuleNode } from './ModuleNode';
import { moduleRegistry } from '@/lib/module-registry';

const WIRES: [string, string][] = [
  ['data', 'pretrain'],
  ['pretrain', 'sft'],
  ['sft', 'lora'],
  ['sft', 'align'],
  ['lora', 'align'],
  ['align', 'eval'],
  ['eval', 'generate'],
  ['pretrain', 'eval'],
  ['pretrain', 'generate'],
];

export function Canvas() {
  useWebSocket();

  const nodePositions = useStore((s) => s.nodePositions);
  const positionTick = useStore((s) => s.positionTick);
  const stages = useStore((s) => s.stages);
  const updateNodePosition = useStore((s) => s.updateNodePosition);
  const wsConnected = useStore((s) => s.wsConnected);

  return (
    <div
      style={{
        width: '100vw',
        height: '100vh',
        overflow: 'auto',
        position: 'relative',
        fontFamily: "'JetBrains Mono', monospace",
        background: '#fafafa',
        backgroundImage: 'radial-gradient(circle, #e0e0e0 0.8px, transparent 0.8px)',
        backgroundSize: '20px 20px',
      }}
    >
      {/* Header */}
      <div
        style={{
          position: 'fixed',
          top: 16,
          left: 20,
          zIndex: 100,
          fontSize: '10px',
          fontWeight: 700,
          letterSpacing: '3px',
          color: '#ccc',
          display: 'flex',
          alignItems: 'center',
          gap: '10px',
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

      <div
        style={{
          position: 'fixed',
          top: 16,
          right: 20,
          zIndex: 100,
          fontSize: '9px',
          letterSpacing: '1px',
          color: '#ccc',
        }}
      >
        node-graph pipeline control
      </div>

      {/* Wires */}
      <Wires positions={nodePositions} wires={WIRES} stageStatuses={stages} tick={positionTick} />

      {/* Nodes */}
      {moduleRegistry.map((def) => (
        <ModuleNode key={def.id} definition={def} onPosChange={updateNodePosition} />
      ))}

      {/* Footer */}
      <div
        style={{
          position: 'fixed',
          bottom: 16,
          left: 20,
          zIndex: 100,
          fontSize: '8px',
          color: '#ccc',
          letterSpacing: '0.5px',
        }}
      >
        server-side PyTorch · real-time WebSocket metrics
      </div>

      {/* Global styles */}
      <style
        dangerouslySetInnerHTML={{
          __html: `
            * { box-sizing: border-box; }
            input[type=range] { -webkit-appearance: none; background: #e8e8e8; border-radius: 2px; outline: none; }
            input[type=range]::-webkit-slider-thumb { -webkit-appearance: none; width: 10px; height: 10px; border-radius: 50%; background: #999; cursor: pointer; }
            ::selection { background: #dbeafe; }
          `,
        }}
      />
    </div>
  );
}
