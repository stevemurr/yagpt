import type { ModuleDefinition } from './module-schema';
import type { GenerateResponse, LoRAInfo } from './types';
import { useStore } from './store';
import { DataPrepStatusSlot } from '@/components/slots/DataPrepStatusSlot';
import { PretrainMemorySlot } from '@/components/slots/PretrainMemorySlot';
import { PretrainRunningSlot } from '@/components/slots/PretrainRunningSlot';
import { PretrainDoneSlot } from '@/components/slots/PretrainDoneSlot';
import { LoRAInfoSlot } from '@/components/slots/LoRAInfoSlot';
import { EvalResultsSlot } from '@/components/slots/EvalResultsSlot';
import { GenerateOutputSlot } from '@/components/slots/GenerateOutputSlot';

// --- DataPrep ---

const dataPrepModule: ModuleDefinition = {
  id: 'data',
  title: 'Data Prep',
  accent: '#8b5cf6',
  position: { x: 60, y: 80 },
  width: 260,
  fields: [
    {
      type: 'select', key: 'subset', label: 'subset',
      options: [
        { value: 'sample-10BT', label: 'sample-10BT' },
        { value: 'sample-100BT', label: 'sample-100BT' },
        { value: 'CC-MAIN-2024-10', label: 'CC-MAIN-2024-10' },
      ],
      default: 'sample-10BT', minWidth: '55px',
    },
    { type: 'text', key: 'output_dir', label: 'out dir', default: './data/raw/sample-10BT', minWidth: '55px' },
    { type: 'slider', key: 'num_shards', label: 'shards', min: 10, max: 500, step: 10, default: 100, parse: 'int', minWidth: '55px', valueMinWidth: '30px' },
  ],
  fieldEffects: [
    {
      watch: 'subset',
      update: (value) => ({
        output_dir: `./data/raw/${value}`,
        tokenized_dir: `./data/tokenized/${value}`,
      }),
    },
  ],
  hiddenDefaults: {
    tokenized_dir: './data/tokenized/sample-10BT',
  },
  actions: {
    type: 'multi',
    buttons: [
      {
        label: 'Download',
        apiCall: 'downloadData',
        buildPayload: (config) => ({
          output_dir: config.output_dir,
          subset: config.subset,
          num_shards: config.num_shards,
        }),
      },
      {
        label: 'Tokenize',
        apiCall: 'tokenizeData',
        buildPayload: (config) => ({
          input_dir: config.output_dir,
          output_dir: config.tokenized_dir,
        }),
      },
      {
        label: 'Validate',
        apiCall: 'validateData',
        buildPayload: (config) => ({
          data_dir: config.tokenized_dir,
        }),
      },
    ],
  },
  slots: {
    belowFieldsIdle: DataPrepStatusSlot,
  },
};

// --- Pretrain ---

const pretrainModule: ModuleDefinition = {
  id: 'pretrain',
  title: 'Pretrain',
  accent: '#22c55e',
  position: { x: 350, y: 80 },
  width: 310,
  fields: [
    { type: 'slider', key: 'n_layers', label: 'layers', min: 1, max: 96, step: 1, default: 6, parse: 'int', minWidth: '50px', valueMinWidth: '50px' },
    { type: 'slider', key: 'n_heads', label: 'heads', min: 1, max: 96, step: 1, default: 6, parse: 'int', minWidth: '50px', valueMinWidth: '50px' },
    { type: 'slider', key: 'dim', label: 'dim', min: 64, max: 8192, step: 64, default: 384, parse: 'int', minWidth: '50px', valueMinWidth: '50px' },
    { type: 'slider', key: 'max_seq_len', label: 'seq len', min: 256, max: 32768, step: 256, default: 1024, parse: 'int', minWidth: '50px', valueMinWidth: '50px' },
    { type: 'slider', key: 'max_steps', label: 'steps', min: 1000, max: 100000, step: 1000, default: 10000, parse: 'int', minWidth: '50px', valueMinWidth: '50px' },
    { type: 'slider', key: 'learning_rate', label: 'lr', min: 1e-5, max: 1e-2, step: 1e-5, default: 3e-4, format: 'exponential', minWidth: '50px', valueMinWidth: '50px' },
    {
      type: 'select', key: 'optimizer', label: 'optim',
      options: [
        { value: 'dual', label: 'Dual (Muon+AdamW)' },
        { value: 'adamw', label: 'AdamW' },
        { value: 'muon', label: 'Muon' },
      ],
      default: 'dual', minWidth: '50px',
    },
    { type: 'slider', key: 'batch_size', label: 'batch', min: 1, max: 128, step: 1, default: 32, parse: 'int', minWidth: '50px', valueMinWidth: '50px' },
    { type: 'datadir', key: 'train_data_dir', label: 'data dir', default: './data/tokenized', placeholder: './data/tokenized', minWidth: '50px' },
    { type: 'checkpoint', key: 'resume_from', label: 'resume from', default: '', placeholder: 'none (train from scratch)', minWidth: '50px' },
    { type: 'text', key: 'checkpoint_dir', label: 'save to', default: './checkpoints', placeholder: './checkpoints', minWidth: '50px' },
    { type: 'slider', key: 'checkpoint_interval', label: 'ckpt every', min: 100, max: 10000, step: 100, default: 1000, parse: 'int', minWidth: '50px', valueMinWidth: '50px' },
    { type: 'slider', key: 'keep_checkpoints', label: 'keep last', min: 1, max: 20, step: 1, default: 5, parse: 'int', minWidth: '50px', valueMinWidth: '50px' },
  ],
  hiddenDefaults: {
    total_batch_size: 524288,
    muon_lr: 0.02,
    lr_schedule: 'warmup_cosine',
    dtype: 'bfloat16',
    compile: true,
  },
  actions: {
    type: 'start-stop',
    start: {
      label: (config) => (config.resume_from as string) ? '▶ RESUME' : '▶ TRAIN',
      apiCall: 'startPretrain',
      buildPayload: (config) => {
        const payload = { ...config };
        // Clean up empty resume_from so backend treats it as fresh training
        if (!payload.resume_from) delete payload.resume_from;
        return payload;
      },
    },
    stop: { apiCall: 'stopPretrain' },
  },
  slots: {
    belowFieldsIdle: PretrainMemorySlot,
    runningView: PretrainRunningSlot,
    doneView: PretrainDoneSlot,
  },
  metricsKey: 'pretrain',
  maxStepsField: 'max_steps',
  hasProgressBar: true,
};

// --- SFT ---

const sftModule: ModuleDefinition = {
  id: 'sft',
  title: 'SFT',
  accent: '#f59e0b',
  position: { x: 690, y: 80 },
  width: 280,
  fields: [
    { type: 'checkpoint', key: 'checkpoint', label: 'ckpt', default: '', placeholder: './checkpoints/final.pt', minWidth: '60px' },
    { type: 'text', key: 'data_path', label: 'data', default: './data/sft', minWidth: '60px' },
    { type: 'slider', key: 'epochs', label: 'epochs', min: 1, max: 10, step: 1, default: 3, parse: 'int', minWidth: '60px', valueMinWidth: '45px' },
    { type: 'slider', key: 'batch_size', label: 'batch', min: 1, max: 32, step: 1, default: 4, parse: 'int', minWidth: '60px', valueMinWidth: '45px' },
    { type: 'slider', key: 'learning_rate', label: 'lr', min: 1e-6, max: 1e-3, step: 1e-6, default: 2e-5, format: 'exponential', minWidth: '60px', valueMinWidth: '45px' },
  ],
  hiddenDefaults: {
    max_seq_len: 2048,
  },
  actions: {
    type: 'start-stop',
    start: {
      label: '▶ START SFT',
      apiCall: 'startSFT',
      buildPayload: (config, store) => {
        const ckpt = (config.checkpoint as string) || (store.activeCheckpoint as string) || './checkpoints/final.pt';
        const { checkpoint: _, ...rest } = config;
        return { checkpoint: ckpt, config: rest };
      },
    },
    stop: { apiCall: 'stopSFT' },
  },
  metricsKey: 'sft',
  lossChartColor: '#f59e0b',
};

// --- LoRA ---

const loraModule: ModuleDefinition = {
  id: 'lora',
  title: 'LoRA',
  accent: '#ec4899',
  position: { x: 1000, y: 80 },
  width: 260,
  fields: [
    { type: 'checkpoint', key: 'checkpoint', label: 'ckpt', default: '', placeholder: 'checkpoint path', minWidth: '45px' },
    { type: 'slider', key: 'rank', label: 'rank', min: 1, max: 128, step: 1, default: 16, parse: 'int', minWidth: '45px', valueMinWidth: '30px' },
    { type: 'slider', key: 'alpha', label: 'alpha', min: 1, max: 256, step: 1, default: 32, parse: 'int', minWidth: '45px', valueMinWidth: '30px' },
    {
      type: 'multi-toggle', key: 'target_modules', label: 'targets',
      options: ['q_proj', 'k_proj', 'v_proj', 'out_proj', 'gate_up', 'down'],
      default: ['q_proj', 'k_proj', 'v_proj', 'out_proj'],
    },
    { type: 'checkbox', key: 'qlora', label: 'QLoRA (NF4 quantize)', default: false },
  ],
  actions: {
    type: 'fire',
    fire: {
      label: 'APPLY LORA',
      apiCall: 'applyLoRA',
      buildPayload: (config, store) => ({
        checkpoint: (config.checkpoint as string) || (store.activeCheckpoint as string) || undefined,
        rank: config.rank,
        alpha: config.alpha,
        target_modules: config.target_modules,
        qlora: config.qlora,
      }),
      onResult: (result) => {
        useStore.getState().setLoraInfo(result as LoRAInfo);
      },
    },
  },
  slots: {
    belowActions: LoRAInfoSlot,
  },
};

// --- Align ---

const alignModule: ModuleDefinition = {
  id: 'align',
  title: 'Alignment',
  accent: '#8b5cf6',
  position: { x: 690, y: 420 },
  width: 280,
  fields: [
    {
      type: 'select', key: 'method', label: 'method',
      options: [
        { value: 'dpo', label: 'DPO' },
        { value: 'simpo', label: 'SimPO' },
        { value: 'grpo', label: 'GRPO' },
      ],
      default: 'dpo', minWidth: '55px',
    },
    { type: 'checkpoint', key: 'checkpoint', label: 'ckpt', default: '', placeholder: 'checkpoint', minWidth: '55px' },
    { type: 'text', key: 'data_path', label: 'data', default: './data/prefs.jsonl', minWidth: '55px' },
    { type: 'slider', key: 'beta', label: 'beta', min: 0.01, max: 5.0, step: 0.01, default: 0.1, minWidth: '55px', valueMinWidth: '50px' },
    { type: 'slider', key: 'lr', label: 'lr', min: 1e-7, max: 1e-4, step: 1e-7, default: 1e-6, format: 'exponential', minWidth: '55px', valueMinWidth: '50px' },
    { type: 'slider', key: 'max_steps', label: 'steps', min: 100, max: 10000, step: 100, default: 1000, parse: 'int', minWidth: '55px', valueMinWidth: '50px' },
    {
      type: 'slider', key: 'gamma', label: 'gamma', min: 0.0, max: 2.0, step: 0.1, default: 0.5, minWidth: '55px', valueMinWidth: '30px',
      visibleWhen: { field: 'method', value: 'simpo' },
    },
    {
      type: 'slider', key: 'group_size', label: 'group', min: 2, max: 16, step: 1, default: 4, parse: 'int', minWidth: '55px', valueMinWidth: '30px',
      visibleWhen: { field: 'method', value: 'grpo' },
    },
  ],
  hiddenDefaults: {
    batch_size: 4,
  },
  actions: {
    type: 'start-stop',
    start: {
      label: (config) => `▶ START ${(config.method as string).toUpperCase()}`,
      apiCall: 'startAlignment',
      buildPayload: (config, store) => ({
        checkpoint: (config.checkpoint as string) || (store.activeCheckpoint as string) || undefined,
        data_path: config.data_path,
        method: config.method,
        beta: config.beta,
        lr: config.lr,
        max_steps: config.max_steps,
        batch_size: config.batch_size,
        gamma: config.gamma,
        group_size: config.group_size,
      }),
    },
    stop: { apiCall: 'stopAlignment' },
  },
  metricsKey: 'align',
  lossChartColor: '#8b5cf6',
  hasProgressBar: true,
  maxStepsField: 'max_steps',
  runningPrefix: (config) => `${(config.method as string).toUpperCase()} · `,
  doneLabel: (config, status) =>
    status === 'done'
      ? `✓ ${(config.method as string).toUpperCase()} COMPLETE`
      : '■ STOPPED',
};

// --- Eval ---

const evalModule: ModuleDefinition = {
  id: 'eval',
  title: 'Eval',
  accent: '#06b6d4',
  position: { x: 1000, y: 420 },
  width: 280,
  fields: [
    { type: 'checkpoint', key: 'checkpoint', label: 'ckpt', default: '', placeholder: 'checkpoint', minWidth: '50px' },
    {
      type: 'multi-toggle', key: 'tasks', label: 'tasks',
      options: ['hellaswag', 'arc_easy', 'arc_challenge', 'piqa', 'winogrande', 'lambada_openai'],
      default: ['hellaswag'],
    },
    { type: 'slider', key: 'batch_size', label: 'batch', min: 1, max: 64, step: 1, default: 8, parse: 'int', minWidth: '50px', valueMinWidth: '25px' },
    { type: 'slider', key: 'num_fewshot', label: 'fewshot', min: 0, max: 10, step: 1, default: 0, parse: 'int', minWidth: '50px', valueMinWidth: '25px' },
  ],
  actions: {
    type: 'fire',
    fire: {
      label: '▶ RUN EVAL',
      apiCall: 'runEval',
      buildPayload: (config, store) => ({
        checkpoint: (config.checkpoint as string) || (store.activeCheckpoint as string) || undefined,
        tasks: (config.tasks as string[]).join(','),
        batch_size: config.batch_size,
        num_fewshot: config.num_fewshot,
      }),
      validate: (config) =>
        (config.tasks as string[]).length === 0 ? 'Select at least one task' : null,
    },
    runningLabel: 'EVALUATING...',
  },
  slots: {
    belowActions: EvalResultsSlot,
  },
};

// --- Generate ---

const generateModule: ModuleDefinition = {
  id: 'generate',
  title: 'Generate',
  accent: '#06b6d4',
  position: { x: 1310, y: 420 },
  width: 300,
  fields: [
    { type: 'checkpoint', key: 'checkpoint', label: 'ckpt', default: '', placeholder: 'checkpoint path', minWidth: '45px' },
    { type: 'textarea', key: 'prompt', label: 'prompt', default: '', placeholder: 'type a prompt...', rows: 2, onSubmit: true },
    { type: 'slider', key: 'temperature', label: 'temp', min: 0.1, max: 2.0, step: 0.1, default: 0.8, minWidth: '45px', valueMinWidth: '25px' },
    { type: 'slider', key: 'top_k', label: 'top_k', min: 1, max: 200, step: 1, default: 50, parse: 'int', minWidth: '45px', valueMinWidth: '25px' },
    { type: 'slider', key: 'max_tokens', label: 'tokens', min: 10, max: 1000, step: 10, default: 200, parse: 'int', minWidth: '45px', valueMinWidth: '35px' },
  ],
  actions: {
    type: 'fire',
    fire: {
      label: '→',
      apiCall: 'generate',
      buildPayload: (config, store) => ({
        checkpoint: (config.checkpoint as string) || (store.activeCheckpoint as string) || undefined,
        prompt: (config.prompt as string) || 'Once upon a time',
        max_tokens: config.max_tokens,
        temperature: config.temperature,
        top_k: config.top_k,
      }),
      onResult: (result, ctx) => {
        const r = result as GenerateResponse;
        ctx.setLocalState('generateOutput', r.text);
        ctx.setLocalState('tokensGenerated', r.tokens_generated);
        ctx.setLocalState('generatePrompt', (ctx.config.prompt as string) || 'Once upon a time');
      },
    },
    inlineField: 'prompt',
  },
  slots: {
    belowActions: GenerateOutputSlot,
  },
};

export const moduleRegistry: ModuleDefinition[] = [
  dataPrepModule,
  pretrainModule,
  sftModule,
  loraModule,
  alignModule,
  evalModule,
  generateModule,
];
