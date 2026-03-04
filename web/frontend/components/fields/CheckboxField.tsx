'use client';

import type { CheckboxFieldDef } from '@/lib/module-schema';

interface Props {
  field: CheckboxFieldDef;
  value: boolean;
  onChange: (value: boolean) => void;
}

export function CheckboxField({ field, value, onChange }: Props) {
  return (
    <label style={{
      display: 'flex', alignItems: 'center', gap: '6px',
      fontSize: '9px', color: '#999', cursor: 'pointer',
    }}>
      <input
        type="checkbox"
        checked={value}
        onChange={(e) => onChange(e.target.checked)}
        style={{ accentColor: '#999' }}
      />
      {field.label}
    </label>
  );
}
