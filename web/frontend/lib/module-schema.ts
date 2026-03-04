import type { ComponentType, CSSProperties } from 'react';
import type { StageId, StageStatus } from './types';

// --- Field definitions ---

export interface SliderFieldDef {
  type: 'slider';
  key: string;
  label: string;
  min: number;
  max: number;
  step: number;
  default: number;
  format?: 'exponential';
  parse?: 'int';
  minWidth?: string;
  valueMinWidth?: string;
}

export interface SelectFieldDef {
  type: 'select';
  key: string;
  label: string;
  options: { value: string; label: string }[];
  default: string;
  minWidth?: string;
}

export interface TextFieldDef {
  type: 'text';
  key: string;
  label: string;
  default: string;
  placeholder?: string;
  minWidth?: string;
}

export interface CheckboxFieldDef {
  type: 'checkbox';
  key: string;
  label: string;
  default: boolean;
}

export interface CheckpointFieldDef {
  type: 'checkpoint';
  key: string;
  label: string;
  default: string;
  placeholder?: string;
  minWidth?: string;
}

export interface MultiToggleFieldDef {
  type: 'multi-toggle';
  key: string;
  label: string;
  options: string[];
  default: string[];
}

export interface TextareaFieldDef {
  type: 'textarea';
  key: string;
  label: string;
  default: string;
  placeholder?: string;
  rows?: number;
  onSubmit?: boolean;
}

export interface DataDirFieldDef {
  type: 'datadir';
  key: string;
  label: string;
  default: string;
  placeholder?: string;
  minWidth?: string;
}

export type FieldDef =
  | SliderFieldDef
  | SelectFieldDef
  | TextFieldDef
  | CheckboxFieldDef
  | CheckpointFieldDef
  | MultiToggleFieldDef
  | TextareaFieldDef
  | DataDirFieldDef;

// --- Conditional visibility ---

export interface VisibleWhen {
  field: string;
  value: string | string[];
}

export type FieldDefWithVisibility = FieldDef & { visibleWhen?: VisibleWhen };

// --- Field effects (reactive cross-field updates) ---

export interface FieldEffect {
  watch: string;
  update: (value: unknown, config: Record<string, unknown>) => Record<string, unknown>;
}

// --- Action result context ---

export interface ActionResultContext {
  setLocalState: (key: string, value: unknown) => void;
  config: Record<string, unknown>;
}

// --- Actions ---

export interface ActionDef {
  label: string | ((config: Record<string, unknown>) => string);
  apiCall: string;
  buildPayload: (config: Record<string, unknown>, store: Record<string, unknown>) => Record<string, unknown>;
  onResult?: (result: unknown, ctx: ActionResultContext) => void;
  validate?: (config: Record<string, unknown>) => string | null;
}

export interface StartStopActions {
  type: 'start-stop';
  start: ActionDef;
  stop: { apiCall: string };
}

export interface FireActions {
  type: 'fire';
  fire: ActionDef;
  runningLabel?: string;
  inlineField?: string;
  inlineButtonStyle?: CSSProperties;
}

export interface MultiActions {
  type: 'multi';
  buttons: ActionDef[];
}

export type ModuleActions = StartStopActions | FireActions | MultiActions;

// --- Slots (custom UI escape hatches) ---

export interface SlotProps {
  config: Record<string, unknown>;
  stageState: { status: StageStatus; currentStep?: number; maxSteps?: number; error?: string };
  store: Record<string, unknown>;
  onAction?: () => void;
}

export interface ModuleSlots {
  belowFieldsIdle?: ComponentType<SlotProps>;
  runningView?: ComponentType<SlotProps>;
  doneView?: ComponentType<SlotProps>;
  belowActions?: ComponentType<SlotProps>;
}

// --- Module definition ---

export interface ModuleDefinition {
  id: StageId;
  title: string;
  accent: string;
  position: { x: number; y: number };
  width: number;
  fields: FieldDefWithVisibility[];
  hiddenDefaults?: Record<string, unknown>;
  fieldEffects?: FieldEffect[];
  actions: ModuleActions;
  slots?: ModuleSlots;
  metricsKey?: string;
  lossChartColor?: string;
  hasProgressBar?: boolean;
  maxStepsField?: string;
  runningPrefix?: (config: Record<string, unknown>) => string;
  doneLabel?: (config: Record<string, unknown>, status: StageStatus) => string;
  onMount?: (store: Record<string, unknown>, setLocalState: (key: string, value: unknown) => void) => void;
}
