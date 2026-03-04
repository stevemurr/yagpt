'use client';

import type { MultiToggleFieldDef } from '@/lib/module-schema';
import { toggleBtnStyle } from './styles';

interface Props {
  field: MultiToggleFieldDef;
  value: string[];
  onChange: (value: string[]) => void;
}

export function MultiToggleField({ field, value, onChange }: Props) {
  const toggle = (opt: string) => {
    onChange(value.includes(opt) ? value.filter((v) => v !== opt) : [...value, opt]);
  };

  return (
    <div>
      <span style={{ fontSize: '9px', color: '#999', display: 'block', marginBottom: '4px' }}>
        {field.label}
      </span>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '3px' }}>
        {field.options.map((opt) => (
          <button
            key={opt}
            onClick={() => toggle(opt)}
            style={toggleBtnStyle(value.includes(opt))}
          >
            {opt}
          </button>
        ))}
      </div>
    </div>
  );
}
