const BASE = typeof window !== 'undefined'
  ? `http://${window.location.hostname}:8000/api/experiments`
  : 'http://localhost:8000/api/experiments';

export interface ExperimentSummary {
  id: number;
  name: string;
  created_at: string;
  updated_at: string;
  active_modules: string[];
}

export interface Experiment {
  id: number;
  name: string;
  created_at: string;
  updated_at: string;
  data_config: Record<string, unknown>;
  active_modules: string[];
  module_configs: Record<string, Record<string, unknown>>;
}

export interface Run {
  id: number;
  experiment_id: number;
  module_id: string;
  started_at: string;
  completed_at: string | null;
  status: string;
  config_snapshot: Record<string, unknown>;
  metrics_summary: Record<string, unknown>;
  checkpoint_path: string | null;
  eval_results: Record<string, unknown> | null;
}

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${res.status}: ${text}`);
  }
  return res.json();
}

export async function listExperiments(): Promise<ExperimentSummary[]> {
  return fetchJson<ExperimentSummary[]>(BASE + '/');
}

export async function createExperiment(data: {
  name: string;
  active_modules?: string[];
  module_configs?: Record<string, unknown>;
}): Promise<Experiment> {
  return fetchJson<Experiment>(BASE + '/', {
    method: 'POST',
    body: JSON.stringify(data),
  });
}

export async function getExperiment(id: number): Promise<Experiment> {
  return fetchJson<Experiment>(`${BASE}/${id}`);
}

export async function updateExperiment(
  id: number,
  data: Partial<Pick<Experiment, 'name' | 'data_config' | 'active_modules' | 'module_configs'>>,
): Promise<Experiment> {
  return fetchJson<Experiment>(`${BASE}/${id}`, {
    method: 'PUT',
    body: JSON.stringify(data),
  });
}

export async function deleteExperiment(id: number): Promise<void> {
  await fetchJson<{ status: string }>(`${BASE}/${id}`, { method: 'DELETE' });
}

export async function listRuns(experimentId: number): Promise<Run[]> {
  return fetchJson<Run[]>(`${BASE}/${experimentId}/runs`);
}
