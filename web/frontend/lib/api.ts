const BASE = typeof window !== 'undefined'
  ? `http://${window.location.hostname}:8000/api`
  : 'http://localhost:8000/api';

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

// Health
export const healthCheck = () => request<{ status: string; has_model: boolean }>('/health');

// Checkpoints
export const listCheckpoints = (dir?: string) =>
  request<import('./types').CheckpointInfo[]>(`/checkpoints${dir ? `?directory=${encodeURIComponent(dir)}` : ''}`);

export const inspectCheckpoint = (path: string) =>
  request<import('./types').CheckpointInfo>(`/checkpoints/inspect?path=${encodeURIComponent(path)}`);

// Generate
export const generate = (body: {
  checkpoint?: string; prompt?: string; max_tokens?: number; temperature?: number; top_k?: number;
}) => request<import('./types').GenerateResponse>('/generate', { method: 'POST', body: JSON.stringify(body) });

// Pretrain
export const startPretrain = (config: Record<string, unknown>) => {
  const { experiment_id, ...rest } = config;
  return request<{ status: string }>('/pretrain/start', {
    method: 'POST',
    body: JSON.stringify({ config: rest, ...(experiment_id != null ? { experiment_id } : {}) }),
  });
};

export const stopPretrain = () =>
  request<{ status: string }>('/pretrain/stop', { method: 'POST' });

export const pretrainStatus = () =>
  request<{ status: string; step: number | null }>('/pretrain/status');

export const pretrainMetrics = (lastN = 100) =>
  request<Record<string, unknown>[]>(`/pretrain/metrics?last_n=${lastN}`);

export const pretrainConfigSchema = () =>
  request<Record<string, { type: string; default: unknown }>>('/pretrain/config/schema');

// SFT
export const startSFT = (checkpoint: string, config: Record<string, unknown>) => {
  const { experiment_id, ...rest } = config;
  return request<{ status: string }>('/sft/start', {
    method: 'POST',
    body: JSON.stringify({ checkpoint, config: rest, ...(experiment_id != null ? { experiment_id } : {}) }),
  });
};

export const stopSFT = () =>
  request<{ status: string }>('/sft/stop', { method: 'POST' });

// LoRA
export const applyLoRA = (body: {
  checkpoint?: string; rank?: number; alpha?: number; target_modules?: string[];
  dropout?: number; qlora?: boolean; block_size?: number;
}) => request<import('./types').LoRAInfo>('/lora/apply', { method: 'POST', body: JSON.stringify(body) });

export const loraInfo = () =>
  request<{ has_lora: boolean; lora_params: number; total_params: number; ratio: number }>('/lora/info');

// Alignment
export const startAlignment = (body: {
  checkpoint?: string; data_path: string; method?: string; beta?: number;
  lr?: number; max_steps?: number; batch_size?: number; gamma?: number;
  group_size?: number; output_dir?: string; experiment_id?: number;
}) => request<{ status: string; method: string }>('/alignment/start', { method: 'POST', body: JSON.stringify(body) });

export const stopAlignment = () =>
  request<{ status: string }>('/alignment/stop', { method: 'POST' });

// Eval
export const runEval = (body: {
  checkpoint?: string; tasks?: string; batch_size?: number; num_fewshot?: number; experiment_id?: number;
}) => request<{ status: string }>('/eval/run', { method: 'POST', body: JSON.stringify(body) });

export const evalResults = () =>
  request<{ status: string; results: Record<string, Record<string, number>> | null }>('/eval/results');

// Data
export const dataStatus = () =>
  request<Record<string, { raw_shards: number; tokenized_shards: number }>>('/data/status');

export const dataDirs = () =>
  request<{ path: string; name: string; shards: number }[]>('/data/dirs');

export const downloadData = (body: {
  output_dir?: string; subset?: string; num_shards?: number; max_rows?: number;
}) => request<{ status: string }>('/data/download', { method: 'POST', body: JSON.stringify(body) });

export const tokenizeData = (body: {
  input_dir?: string; output_dir?: string; encoding?: string; max_seq_len?: number;
}) => request<{ status: string }>('/data/tokenize', { method: 'POST', body: JSON.stringify(body) });

export const validateData = (body: { data_dir?: string }) =>
  request<{ status: string }>('/data/validate', { method: 'POST', body: JSON.stringify(body) });

// SFT Data
export const downloadSFTData = (body: {
  output_dir?: string; dataset?: string; subset?: string; max_rows?: number;
}) => request<{ status: string }>('/data/sft/download', { method: 'POST', body: JSON.stringify(body) });

export const validateSFTData = (body: { data_path?: string }) =>
  request<{ status: string }>('/data/sft/validate', { method: 'POST', body: JSON.stringify(body) });

export const listSFTDatasets = () =>
  request<{ available: Record<string, { repo: string; desc: string }>; downloaded: { name: string; path: string; rows: number }[] }>('/data/sft/datasets');
