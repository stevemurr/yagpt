export type StageId = 'data' | 'pretrain' | 'sft' | 'lora' | 'align' | 'eval' | 'generate';
export type StageStatus = 'idle' | 'running' | 'done' | 'error' | 'stopping' | 'stopped';

export interface MetricPoint {
  stage: string;
  step: number;
  loss: number;
  lr: number;
  grad_norm: number | null;
  tokens_per_sec: number | null;
  mfu: number | null;
}

export interface StageState {
  status: StageStatus;
  currentStep?: number;
  maxSteps?: number;
  error?: string;
}

export interface CheckpointInfo {
  path: string;
  filename: string;
  step: number | null;
  loss: number | null;
  val_loss: number | null;
  n_layers: number | null;
  n_heads: number | null;
  dim: number | null;
  max_seq_len: number | null;
  size_mb: number | null;
}

export interface EvalTaskResult {
  [metric: string]: number;
}

export interface GenerateResponse {
  text: string;
  tokens_generated: number;
}

export interface LoRAInfo {
  lora_params: number;
  total_params: number;
  ratio: number;
  rank: number;
  qlora: boolean;
}

export interface WsMessage {
  type: string;
  data: Record<string, unknown>;
}

export interface NodePosition {
  x: number;
  y: number;
  w: number;
  h: number;
}
