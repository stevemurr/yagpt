'use client';

import type { FieldDefWithVisibility } from '@/lib/module-schema';
import { SliderField } from './SliderField';
import { SelectField } from './SelectField';
import { TextField } from './TextField';
import { CheckboxField } from './CheckboxField';
import { CheckpointPicker } from './CheckpointPicker';
import { MultiToggleField } from './MultiToggleField';
import { TextareaField } from './TextareaField';
import { DataDirPicker } from './DataDirPicker';

interface Props {
  field: FieldDefWithVisibility;
  value: unknown;
  onChange: (value: unknown) => void;
  onSubmit?: () => void;
}

export function FieldRenderer({ field, value, onChange, onSubmit }: Props) {
  switch (field.type) {
    case 'slider':
      return <SliderField field={field} value={value as number} onChange={onChange} />;
    case 'select':
      return <SelectField field={field} value={value as string} onChange={onChange} />;
    case 'text':
      return <TextField field={field} value={value as string} onChange={onChange} />;
    case 'checkbox':
      return <CheckboxField field={field} value={value as boolean} onChange={onChange} />;
    case 'checkpoint':
      return <CheckpointPicker field={field} value={value as string} onChange={onChange} />;
    case 'multi-toggle':
      return <MultiToggleField field={field} value={value as string[]} onChange={onChange} />;
    case 'textarea':
      return <TextareaField field={field} value={value as string} onChange={onChange} onSubmit={onSubmit} />;
    case 'datadir':
      return <DataDirPicker field={field} value={value as string} onChange={onChange} />;
    default:
      return null;
  }
}
