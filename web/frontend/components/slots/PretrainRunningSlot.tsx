'use client';

import { LossChart } from '../LossChart';
import { stopBtnStyle } from '../fields/styles';
import { useStore } from '@/lib/store';
import type { SlotProps } from '@/lib/module-schema';

export function PretrainRunningSlot({ config, stageState, onAction }: SlotProps) {
  const metrics = useStore((s) => s.metrics.pretrain || []);
  const currentStep = stageState.currentStep;
  const maxSteps = config.max_steps as number;
  const lastMetric = metrics.length > 0 ? metrics[metrics.length - 1] : null;
  const lossData = metrics.map((m) => ({ step: m.step, loss: m.loss }));

  const modelInfo = [
    `${config.n_layers}L`,
    `${config.n_heads}H`,
    `${config.dim}D`,
    `seq ${config.max_seq_len}`,
    `bs ${config.batch_size}`,
    `${config.optimizer}`,
  ].join(' · ');

  return (
    <>
      <div style={{ fontSize: '9px', color: '#999', letterSpacing: '0.3px' }}>
        {modelInfo}
      </div>

      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10px', color: '#999' }}>
        <span>step {currentStep || 0}/{maxSteps}</span>
        <span>loss {lastMetric?.loss?.toFixed(4) || '—'}</span>
      </div>

      <div style={{ height: '2px', background: '#f0f0f0', borderRadius: '1px', overflow: 'hidden' }}>
        <div style={{
          height: '100%',
          width: `${((currentStep || 0) / maxSteps) * 100}%`,
          background: '#333',
          transition: 'width 0.1s',
        }} />
      </div>

      <LossChart data={lossData} height={60} />

      {lastMetric && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '4px', fontSize: '10px' }}>
          {[
            { label: 'LR', value: lastMetric.lr?.toExponential(2) || '—' },
            { label: 'MFU', value: lastMetric.mfu ? `${(lastMetric.mfu * 100).toFixed(1)}%` : '—' },
            { label: 'TOK/S', value: lastMetric.tokens_per_sec ? `${Math.round(lastMetric.tokens_per_sec).toLocaleString()}` : '—' },
            { label: 'GRAD', value: lastMetric.grad_norm?.toFixed(2) || '—' },
          ].map(({ label, value }) => (
            <div key={label} style={{ background: '#f8f8f8', borderRadius: '3px', padding: '4px 6px' }}>
              <div style={{ fontSize: '7px', color: '#bbb', letterSpacing: '1px' }}>{label}</div>
              <div style={{ fontWeight: 700, color: '#333' }}>{value}</div>
            </div>
          ))}
        </div>
      )}

      <button
        onClick={onAction}
        disabled={stageState.status === 'stopping'}
        style={stopBtnStyle}
      >
        {'■'} {stageState.status === 'stopping' ? 'STOPPING...' : 'STOP'}
      </button>
    </>
  );
}
