'use client';

import type { SliderFieldDef } from '@/lib/module-schema';
import { labelStyle, sliderStyle, valueStyle } from './styles';

interface Props {
  field: SliderFieldDef;
  value: number;
  onChange: (value: number) => void;
}

export function SliderField({ field, value, onChange }: Props) {
  const displayValue = field.format === 'exponential'
    ? value.toExponential(1)
    : field.parse === 'int' ? Math.round(value) : value;

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
      <span style={{ ...labelStyle, minWidth: field.minWidth || '45px' }}>{field.label}</span>
      <input
        type="range"
        min={field.min}
        max={field.max}
        step={field.step}
        value={value}
        onChange={(e) => {
          const v = parseFloat(e.target.value);
          onChange(field.parse === 'int' ? Math.round(v) : v);
        }}
        style={sliderStyle}
      />
      <span style={{ ...valueStyle, minWidth: field.valueMinWidth || '30px' }}>
        {displayValue}
      </span>
    </div>
  );
}
