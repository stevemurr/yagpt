'use client';

import type { TextareaFieldDef } from '@/lib/module-schema';

interface Props {
  field: TextareaFieldDef;
  value: string;
  onChange: (value: string) => void;
  onSubmit?: () => void;
}

export function TextareaField({ field, value, onChange, onSubmit }: Props) {
  return (
    <textarea
      value={value}
      onChange={(e) => onChange(e.target.value)}
      onKeyDown={(e) => {
        if (field.onSubmit && e.key === 'Enter' && !e.shiftKey && onSubmit) {
          e.preventDefault();
          onSubmit();
        }
      }}
      placeholder={field.placeholder}
      rows={field.rows || 2}
      style={{
        flex: 1, padding: '6px 8px', fontSize: '11px', fontFamily: 'inherit',
        border: '1px solid #e0e0e0', borderRadius: '3px', outline: 'none', color: '#333',
        resize: 'vertical',
      }}
    />
  );
}
