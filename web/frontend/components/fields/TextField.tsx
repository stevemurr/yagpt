'use client';

import type { TextFieldDef } from '@/lib/module-schema';
import { labelStyle, inputStyle } from './styles';

interface Props {
  field: TextFieldDef;
  value: string;
  onChange: (value: string) => void;
}

export function TextField({ field, value, onChange }: Props) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
      <span style={{ ...labelStyle, minWidth: field.minWidth || '45px' }}>{field.label}</span>
      <input
        type="text"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={field.placeholder}
        style={inputStyle}
      />
    </div>
  );
}
