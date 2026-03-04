import { create } from 'zustand';
import type { StageId, StageStatus, MetricPoint, NodePosition, EvalTaskResult } from './types';

interface PipelineStore {
  // Node positions (for dragging)
  nodePositions: Record<string, NodePosition>;
  updateNodePosition: (id: string, x: number, y: number, w: number, h: number) => void;
  positionTick: number;

  // Stage statuses
  stages: Record<StageId, { status: StageStatus; currentStep?: number; maxSteps?: number; error?: string }>;
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
}

const defaultStages: Record<StageId, { status: StageStatus }> = {
  data: { status: 'idle' },
  pretrain: { status: 'idle' },
  sft: { status: 'idle' },
  lora: { status: 'idle' },
  align: { status: 'idle' },
  eval: { status: 'idle' },
  generate: { status: 'idle' },
};

export const useStore = create<PipelineStore>((set) => ({
  nodePositions: {},
  positionTick: 0,
  updateNodePosition: (id, x, y, w, h) =>
    set((s) => ({
      nodePositions: { ...s.nodePositions, [id]: { x, y, w, h } },
      positionTick: s.positionTick + 1,
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
}));
