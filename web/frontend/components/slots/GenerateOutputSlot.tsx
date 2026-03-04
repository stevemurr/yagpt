'use client';

import type { SlotProps } from '@/lib/module-schema';

export function GenerateOutputSlot({ store }: SlotProps) {
  const output = store.generateOutput as string | null;
  const tokensGenerated = store.tokensGenerated as number;
  const prompt = store.generatePrompt as string;

  if (!output) return null;

  return (
    <div style={{ borderTop: '1px solid #f0f0f0', paddingTop: '6px' }}>
      <div style={{ fontSize: '8px', color: '#bbb', letterSpacing: '1px', marginBottom: '4px' }}>
        OUTPUT ({tokensGenerated} tokens)
      </div>
      <div style={{
        padding: '8px', fontSize: '10px', lineHeight: '16px', color: '#333',
        background: '#f8f8f8', borderRadius: '3px', maxHeight: '200px',
        overflow: 'auto', whiteSpace: 'pre-wrap', wordBreak: 'break-word',
      }}>
        {prompt && <span style={{ color: '#3b82f6', fontWeight: 600 }}>{prompt}</span>}
        <span>{output.startsWith(prompt || '') ? output.slice((prompt || '').length) : output}</span>
      </div>
    </div>
  );
}
