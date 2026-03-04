'use client';

import { useStore } from '@/lib/store';
import type { SlotProps } from '@/lib/module-schema';

export function LoRAInfoSlot(_props: SlotProps) {
  const loraInfo = useStore((s) => s.loraInfo);
  if (!loraInfo) return null;

  return (
    <div style={{ borderTop: '1px solid #f0f0f0', paddingTop: '6px', fontSize: '9px', color: '#888', lineHeight: '16px' }}>
      <div>LoRA params: <span style={{ fontWeight: 700, color: '#333' }}>{loraInfo.lora_params.toLocaleString()}</span></div>
      <div>Total params: <span style={{ fontWeight: 700, color: '#333' }}>{loraInfo.total_params.toLocaleString()}</span></div>
      <div>Ratio: <span style={{ fontWeight: 700, color: '#333' }}>{loraInfo.ratio}%</span></div>
      {loraInfo.qlora && <div style={{ color: '#ec4899' }}>QLoRA enabled (NF4)</div>}
    </div>
  );
}
