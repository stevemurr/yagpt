'use client';

import type { SelectFieldDef } from '@/lib/module-schema';
import { labelStyle, selectStyle } from './styles';

interface Props {
  field: SelectFieldDef;
  value: string;
  onChange: (value: string) => void;
}

export function SelectField({ field, value, onChange }: Props) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
      <span style={{ ...labelStyle, minWidth: field.minWidth || '45px' }}>{field.label}</span>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        style={selectStyle}
      >
        {field.options.map((opt) => (
          <option key={opt.value} value={opt.value}>{opt.label}</option>
        ))}
      </select>
    </div>
  );
}
