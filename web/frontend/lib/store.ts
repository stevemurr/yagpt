import { create } from 'zustand';
import type { StageId, StageStatus, MetricPoint, EvalTaskResult } from './types';

interface PipelineStore {
  // Active modules in the grid
  activeModuleIds: string[];
  addModule: (id: string) => void;
  removeModule: (id: string) => void;

  // Stage statuses
  stages: Record<StageId, { status: StageStatus; currentStep?: number; maxSteps?: number; error?: string; [key: string]: unknown }>;
  setStageStatus: (stage: StageId, status: StageStatus, extra?: Record<string, unknown>) => void;

  // Training metrics (appended from WebSocket)
  metrics: Record<string, MetricPoint[]>;
  appendMetric: (stage: string, point: MetricPoint) => void;
  clearMetrics: (stage: string) => void;

  // Eval results
  evalResults: Record<string, EvalTaskResult> | null;
  setEvalResults: (results: Record<string, EvalTaskResult> | null) => void;

  // Generate output
  generateOutput: string | null;
  setGenerateOutput: (text: string | null) => void;

  // Active checkpoint
  activeCheckpoint: string | null;
  setActiveCheckpoint: (path: string | null) => void;

  // LoRA info
  loraInfo: { lora_params: number; total_params: number; ratio: number; rank: number; qlora: boolean } | null;
  setLoraInfo: (info: PipelineStore['loraInfo']) => void;

  // WebSocket connected
  wsConnected: boolean;
  setWsConnected: (connected: boolean) => void;

  // Data modal
  dataModalOpen: boolean;
  setDataModalOpen: (open: boolean) => void;

  // Current experiment
  currentExperimentId: number | null;
  setCurrentExperimentId: (id: number | null) => void;

  // Module configs (synced from ModuleNode local state for persistence)
  moduleConfigs: Record<string, Record<string, unknown>>;
  setModuleConfig: (moduleId: string, config: Record<string, unknown>) => void;
  setAllModuleConfigs: (configs: Record<string, Record<string, unknown>>) => void;

  // Bumped when an experiment is loaded to signal ModuleNodes to re-read configs
  configLoadTick: number;
}

const defaultStages: Record<string, { status: StageStatus }> = {
  data: { status: 'idle' },
  pretrain: { status: 'idle' },
  sft: { status: 'idle' },
  lora: { status: 'idle' },
  align: { status: 'idle' },
  eval: { status: 'idle' },
  generate: { status: 'idle' },
};

export const useStore = create<PipelineStore>((set) => ({
  activeModuleIds: ['pretrain'],
  addModule: (id) =>
    set((s) => ({
      activeModuleIds: s.activeModuleIds.includes(id) ? s.activeModuleIds : [...s.activeModuleIds, id],
    })),
  removeModule: (id) =>
    set((s) => ({
      activeModuleIds: s.activeModuleIds.filter((m) => m !== id),
    })),

  stages: defaultStages,
  setStageStatus: (stage, status, extra) =>
    set((s) => ({
      stages: { ...s.stages, [stage]: { ...s.stages[stage], status, ...extra } },
    })),

  metrics: { pretrain: [], sft: [], align: [] },
  appendMetric: (stage, point) =>
    set((s) => ({
      metrics: {
        ...s.metrics,
        [stage]: [...(s.metrics[stage] || []), point].slice(-10000),
      },
    })),
  clearMetrics: (stage) =>
    set((s) => ({ metrics: { ...s.metrics, [stage]: [] } })),

  evalResults: null,
  setEvalResults: (results) => set({ evalResults: results }),

  generateOutput: null,
  setGenerateOutput: (text) => set({ generateOutput: text }),

  activeCheckpoint: null,
  setActiveCheckpoint: (path) => set({ activeCheckpoint: path }),

  loraInfo: null,
  setLoraInfo: (info) => set({ loraInfo: info }),

  wsConnected: false,
  setWsConnected: (connected) => set({ wsConnected: connected }),

  dataModalOpen: false,
  setDataModalOpen: (open) => set({ dataModalOpen: open }),

  currentExperimentId: null,
  setCurrentExperimentId: (id) => set({ currentExperimentId: id }),

  moduleConfigs: {},
  setModuleConfig: (moduleId, config) =>
    set((s) => ({
      moduleConfigs: { ...s.moduleConfigs, [moduleId]: config },
    })),
  setAllModuleConfigs: (configs) => set({ moduleConfigs: configs, configLoadTick: Date.now() }),

  configLoadTick: 0,
}));
