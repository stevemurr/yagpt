'use client';

import { LossChart } from '../LossChart';
import { stopBtnStyle } from '../fields/styles';
import { useStore } from '@/lib/store';
import type { SlotProps } from '@/lib/module-schema';

export function PretrainDoneSlot({ config, stageState, onAction }: SlotProps) {
  const metrics = useStore((s) => s.metrics.pretrain || []);
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
    <div style={{ textAlign: 'center' }}>
      <div style={{
        fontSize: '10px', fontWeight: 600,
        color: stageState.status === 'done' ? '#22c55e' : '#f59e0b',
        letterSpacing: '2px', marginBottom: '4px',
      }}>
        {stageState.status === 'done' ? '✓ TRAINING COMPLETE' : '■ STOPPED'}
      </div>
      <div style={{ fontSize: '9px', color: '#999', letterSpacing: '0.3px', marginBottom: '4px' }}>
        {modelInfo}
      </div>
      {lastMetric && (
        <div style={{ fontSize: '9px', color: '#bbb', marginBottom: '8px' }}>
          loss {lastMetric.loss.toFixed(4)} · step {lastMetric.step}
        </div>
      )}
      <LossChart data={lossData} height={50} />
      <button onClick={onAction} style={{ ...stopBtnStyle, marginTop: '8px' }}>RETRAIN</button>
    </div>
  );
}
