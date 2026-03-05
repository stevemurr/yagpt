'use client';

import { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { Node } from './Node';
import { LossChart } from './LossChart';
import { FieldRenderer } from './fields/FieldRenderer';
import { TextareaField } from './fields/TextareaField';
import { btnStyle, stopBtnStyle } from './fields/styles';
import { useStore } from '@/lib/store';
import * as api from '@/lib/api';
import type { ModuleDefinition, FieldDefWithVisibility, SlotProps } from '@/lib/module-schema';
import type { StageStatus } from '@/lib/types';

interface Props {
  definition: ModuleDefinition;
  onRemove?: () => void;
}

// Map api function names to actual api functions
const apiMap: Record<string, (body: unknown) => Promise<unknown>> = {
  generate: (body) => api.generate(body as Parameters<typeof api.generate>[0]),
  startPretrain: (body) => api.startPretrain(body as Record<string, unknown>),
  stopPretrain: () => api.stopPretrain(),
  startSFT: (body) => {
    const b = body as { checkpoint: string; config: Record<string, unknown>; experiment_id?: number };
    return api.startSFT(b.checkpoint, { ...b.config, ...(b.experiment_id != null ? { experiment_id: b.experiment_id } : {}) });
  },
  stopSFT: () => api.stopSFT(),
  applyLoRA: (body) => api.applyLoRA(body as Parameters<typeof api.applyLoRA>[0]),
  startAlignment: (body) => api.startAlignment(body as Parameters<typeof api.startAlignment>[0]),
  stopAlignment: () => api.stopAlignment(),
  runEval: (body) => api.runEval(body as Parameters<typeof api.runEval>[0]),
  downloadData: (body) => api.downloadData(body as Parameters<typeof api.downloadData>[0]),
  tokenizeData: (body) => api.tokenizeData(body as Parameters<typeof api.tokenizeData>[0]),
  validateData: (body) => api.validateData(body as Parameters<typeof api.validateData>[0]),
};

function buildDefaults(fields: FieldDefWithVisibility[], hidden?: Record<string, unknown>): Record<string, unknown> {
  const defaults: Record<string, unknown> = { ...(hidden || {}) };
  for (const f of fields) {
    defaults[f.key] = f.default;
  }
  return defaults;
}

function isFieldVisible(field: FieldDefWithVisibility, config: Record<string, unknown>): boolean {
  if (!field.visibleWhen) return true;
  const val = config[field.visibleWhen.field];
  if (Array.isArray(field.visibleWhen.value)) {
    return field.visibleWhen.value.includes(val as string);
  }
  return val === field.visibleWhen.value;
}

export function ModuleNode({ definition: def, onRemove }: Props) {
  const stageState = useStore((s) => s.stages[def.id]);
  const status = stageState.status;
  const currentStep = stageState.currentStep;
  const metrics = useStore((s) => s.metrics[def.metricsKey || ''] || []);
  const activeCheckpoint = useStore((s) => s.activeCheckpoint);
  const activeModuleIds = useStore((s) => s.activeModuleIds);
  const setStageStatus = useStore((s) => s.setStageStatus);
  const evalResults = useStore((s) => s.evalResults);
  const loraInfo = useStore((s) => s.loraInfo);

  // When pretrain is in the grid and this module has a 'checkpoint' field,
  // lock it to the pretrain output — user cannot override.
  const pretrainInChain = activeModuleIds.includes('pretrain') && def.id !== 'pretrain';
  const hasCheckpointField = def.fields.some((f) => f.key === 'checkpoint');
  const checkpointLocked = pretrainInChain && hasCheckpointField;

  const setModuleConfig = useStore((s) => s.setModuleConfig);
  const configLoadTick = useStore((s) => s.configLoadTick);

  const initialConfig = useMemo(() => buildDefaults(def.fields, def.hiddenDefaults), [def]);
  const [config, setConfig] = useState<Record<string, unknown>>(initialConfig);
  const [error, setError] = useState<string | null>(null);
  const [localState, setLocalStateRaw] = useState<Record<string, unknown>>({});
  const [generating, setGenerating] = useState(false);

  // When an experiment is loaded (configLoadTick changes), pull saved config from store
  const prevTickRef = useRef(configLoadTick);
  useEffect(() => {
    if (configLoadTick !== prevTickRef.current) {
      prevTickRef.current = configLoadTick;
      const saved = useStore.getState().moduleConfigs[def.id];
      if (saved) {
        setConfig((prev) => ({ ...prev, ...saved }));
      }
    }
  }, [configLoadTick, def.id]);

  const setLocalState = useCallback((key: string, value: unknown) => {
    setLocalStateRaw((s) => ({ ...s, [key]: value }));
  }, []);

  // Mount hook
  useEffect(() => {
    if (def.onMount) {
      const storeSnapshot = { activeCheckpoint, evalResults, loraInfo };
      def.onMount(storeSnapshot, setLocalState);
    }
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Sync config to store whenever it changes (for experiment persistence)
  useEffect(() => {
    setModuleConfig(def.id, config);
  }, [config, def.id, setModuleConfig]);

  const updateConfig = useCallback((key: string, value: unknown) => {
    setConfig((prev) => {
      let next = { ...prev, [key]: value };
      if (def.fieldEffects) {
        for (const effect of def.fieldEffects) {
          if (effect.watch === key) {
            next = { ...next, ...effect.update(value, next) };
          }
        }
      }
      return next;
    });
  }, [def.fieldEffects]);

  const storeSnapshot = useMemo(() => ({
    activeCheckpoint, evalResults, loraInfo,
  }), [activeCheckpoint, evalResults, loraInfo]);

  const slotProps: SlotProps = useMemo(() => ({
    config,
    stageState,
    store: { ...storeSnapshot, ...localState },
  }), [config, stageState, storeSnapshot, localState]);

  // --- Action handlers ---

  const currentExperimentId = useStore((s) => s.currentExperimentId);

  const executeApi = useCallback(async (apiCallKey: string, payload: Record<string, unknown>) => {
    const fn = apiMap[apiCallKey];
    if (!fn) throw new Error(`Unknown API: ${apiCallKey}`);
    // Force checkpoint from pretrain when locked
    if (checkpointLocked && activeCheckpoint) {
      payload = { ...payload, checkpoint: activeCheckpoint };
    }
    // Inject experiment_id for run tracking
    if (currentExperimentId != null) {
      payload = { ...payload, experiment_id: currentExperimentId };
    }
    return await fn(payload);
  }, [currentExperimentId, checkpointLocked, activeCheckpoint]);

  const handleFireAction = useCallback(async () => {
    if (def.actions.type !== 'fire') return;
    const action = def.actions.fire;

    // Validate
    if (action.validate) {
      const validationError = action.validate(config);
      if (validationError) { setError(validationError); return; }
    }

    setError(null);
    const payload = action.buildPayload(config, { ...storeSnapshot, ...localState });

    // For modules that manage their own status (generate)
    if (def.id === 'generate') {
      setGenerating(true);
      setStageStatus('generate', 'running');
      try {
        const result = await executeApi(action.apiCall, payload);
        if (action.onResult) {
          action.onResult(result, { setLocalState, config });
        }
        setStageStatus('generate', 'done');
      } catch (e: unknown) {
        const msg = e instanceof Error ? e.message : String(e);
        setError(msg);
        setStageStatus('generate', 'error');
      } finally {
        setGenerating(false);
      }
      return;
    }

    try {
      const result = await executeApi(action.apiCall, payload);
      if (action.onResult) {
        action.onResult(result, { setLocalState, config });
      }
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e);
      setError(msg);
    }
  }, [def, config, storeSnapshot, localState, setStageStatus, executeApi, setLocalState]);

  const handleStartAction = useCallback(async () => {
    if (def.actions.type !== 'start-stop') return;
    const action = def.actions.start;
    setError(null);
    const payload = action.buildPayload(config, { ...storeSnapshot, ...localState });
    try {
      await executeApi(action.apiCall, payload);
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e);
      setError(msg);
    }
  }, [def, config, storeSnapshot, localState, executeApi]);

  const handleResetToIdle = useCallback(() => {
    setStageStatus(def.id, 'idle');
    if (def.metricsKey) {
      useStore.getState().clearMetrics(def.metricsKey);
    }
    setError(null);
  }, [def, setStageStatus]);

  const handleStopAction = useCallback(async () => {
    if (def.actions.type !== 'start-stop') return;
    try {
      await executeApi(def.actions.stop.apiCall, {});
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e);
      setError(msg);
    }
  }, [def, executeApi]);

  const handleMultiAction = useCallback(async (index: number) => {
    if (def.actions.type !== 'multi') return;
    const action = def.actions.buttons[index];
    setError(null);
    const payload = action.buildPayload(config, { ...storeSnapshot, ...localState });
    try {
      await executeApi(action.apiCall, payload);
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e);
      setError(msg);
    }
  }, [def, config, storeSnapshot, localState, executeApi]);

  // --- Computed ---

  const lastMetric = metrics.length > 0 ? metrics[metrics.length - 1] : null;
  const lossData = metrics.map((m: { step: number; loss: number }) => ({ step: m.step, loss: m.loss }));
  const maxSteps = def.maxStepsField ? (config[def.maxStepsField] as number) : undefined;
  const stageError = stageState.error;

  // For fire/multi actions, fields are always visible
  // For start-stop actions, fields are only visible in idle
  const isStartStop = def.actions.type === 'start-stop';
  const showFields = isStartStop ? status === 'idle' : true;

  // For Generate: use local output state for Node status
  const nodeStatus: StageStatus = def.id === 'generate'
    ? (localState.generateOutput ? 'done' : 'idle')
    : status;

  // Inline fire button setup
  const inlineFieldKey = def.actions.type === 'fire' ? def.actions.inlineField : undefined;

  // --- Render fields ---

  const renderFields = () => {
    const visibleFields = def.fields.filter((f) => isFieldVisible(f, config));

    return visibleFields.map((field) => {
      // Locked checkpoint: show static display linked to pretrain
      if (checkpointLocked && field.key === 'checkpoint') {
        const display = activeCheckpoint
          ? activeCheckpoint.split('/').pop() || activeCheckpoint
          : 'waiting for pretrain...';
        return (
          <div key={field.key} style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <span style={{ fontSize: '9px', color: '#999', minWidth: field.type === 'checkpoint' ? (field as { minWidth?: string }).minWidth || '45px' : '45px' }}>ckpt</span>
            <span style={{
              flex: 1, padding: '4px 6px', fontSize: '10px', fontFamily: 'inherit',
              color: activeCheckpoint ? '#22c55e' : '#bbb',
              background: '#f8fdf8', border: '1px solid #e0e0e0', borderRadius: '3px',
              overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
            }}>
              {activeCheckpoint ? `↳ ${display}` : display}
            </span>
          </div>
        );
      }

      // Handle inline textarea + fire button
      if (inlineFieldKey && field.key === inlineFieldKey && field.type === 'textarea') {
        return (
          <div key={field.key} style={{ display: 'flex', gap: '4px' }}>
            <TextareaField
              field={field}
              value={config[field.key] as string}
              onChange={(v) => updateConfig(field.key, v)}
              onSubmit={handleFireAction}
            />
            <button
              onClick={handleFireAction}
              disabled={generating}
              style={def.actions.type === 'fire' && def.actions.inlineButtonStyle
                ? def.actions.inlineButtonStyle
                : {
                  padding: '6px 10px', fontSize: '10px', fontWeight: 600, fontFamily: 'inherit',
                  background: '#333', color: '#fff', border: 'none', borderRadius: '3px',
                  cursor: 'pointer', alignSelf: 'stretch',
                }}
            >
              {generating ? '...' : (typeof def.actions.type === 'string' && def.actions.type === 'fire'
                ? (typeof def.actions.fire.label === 'function' ? def.actions.fire.label(config) : def.actions.fire.label)
                : '→')}
            </button>
          </div>
        );
      }

      return (
        <FieldRenderer
          key={field.key}
          field={field}
          value={config[field.key]}
          onChange={(v) => updateConfig(field.key, v)}
          onSubmit={def.actions.type === 'fire' ? handleFireAction : handleStartAction}
        />
      );
    });
  };

  // --- Render action buttons (when not using inlineField) ---

  const renderActionButtons = () => {
    // For fire actions with inlineField, the button is rendered inline with the field
    if (def.actions.type === 'fire' && def.actions.inlineField) return null;

    if (def.actions.type === 'start-stop' && status === 'idle') {
      return (
        <button onClick={handleStartAction} style={{
          ...btnStyle,
          padding: def.id === 'pretrain' ? '10px' : '8px',
          fontSize: def.id === 'pretrain' ? '11px' : '10px',
        }}>
          {typeof def.actions.start.label === 'function'
            ? def.actions.start.label(config)
            : def.actions.start.label}
        </button>
      );
    }

    if (def.actions.type === 'fire') {
      const isRunning = status === 'running';
      const label = isRunning && def.actions.runningLabel
        ? def.actions.runningLabel
        : (typeof def.actions.fire.label === 'function'
          ? def.actions.fire.label(config)
          : def.actions.fire.label);

      return (
        <button
          onClick={handleFireAction}
          disabled={isRunning}
          style={{
            ...btnStyle,
            background: isRunning ? '#e8e8e8' : '#333',
            color: isRunning ? '#bbb' : '#fff',
            cursor: isRunning ? 'not-allowed' : 'pointer',
          }}
        >
          {label}
        </button>
      );
    }

    if (def.actions.type === 'multi') {
      return (
        <div style={{ display: 'flex', gap: '4px' }}>
          {def.actions.buttons.map((btn, i) => (
            <button
              key={i}
              onClick={() => handleMultiAction(i)}
              disabled={status === 'running'}
              style={{
                flex: 1, padding: '6px', fontSize: '9px', fontWeight: 600,
                fontFamily: 'inherit', letterSpacing: '0.5px', background: '#333',
                color: '#fff', border: 'none', borderRadius: '3px', cursor: 'pointer',
              }}
            >
              {typeof btn.label === 'function' ? btn.label(config) : btn.label}
            </button>
          ))}
        </div>
      );
    }

    return null;
  };

  // --- Render ---

  return (
    <Node
      id={def.id}
      title={def.title}
      status={nodeStatus}
      accent={def.accent}
      onRemove={onRemove}
    >
      <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
        {/* Fields (always visible for fire/multi, idle-only for start-stop) */}
        {showFields && (
          <>
            {renderFields()}
            {status === 'idle' && def.slots?.belowFieldsIdle && <def.slots.belowFieldsIdle {...slotProps} />}
            {renderActionButtons()}
          </>
        )}

        {/* Running state for start-stop actions */}
        {isStartStop && (status === 'running' || status === 'stopping') && (
          <>
            {def.slots?.runningView ? (
              <def.slots.runningView {...slotProps} onAction={handleStopAction} />
            ) : (
              <>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10px', color: '#999' }}>
                  <span>
                    {def.runningPrefix ? def.runningPrefix(config) : ''}
                    step {currentStep || 0}{maxSteps ? `/${maxSteps}` : ''}
                  </span>
                  <span>loss {lastMetric?.loss?.toFixed(4) || '—'}</span>
                </div>

                {def.hasProgressBar && maxSteps && (
                  <div style={{ height: '2px', background: '#f0f0f0', borderRadius: '1px', overflow: 'hidden' }}>
                    <div style={{
                      height: '100%',
                      width: `${((currentStep || 0) / maxSteps) * 100}%`,
                      background: def.lossChartColor || '#333',
                      transition: 'width 0.1s',
                    }} />
                  </div>
                )}

                {def.metricsKey && <LossChart data={lossData} height={50} color={def.lossChartColor} />}

                <button onClick={handleStopAction} disabled={status === 'stopping'} style={stopBtnStyle}>
                  {'■ STOP'}
                </button>
              </>
            )}
          </>
        )}

        {/* Done/Stopped state for start-stop actions */}
        {isStartStop && (status === 'done' || status === 'stopped') && (
          <>
            {def.slots?.doneView ? (
              <def.slots.doneView {...slotProps} onAction={handleResetToIdle} />
            ) : (
              <>
                {def.metricsKey && (
                  <div style={{ textAlign: 'center' }}>
                    <div style={{
                      fontSize: '10px', fontWeight: 600,
                      color: status === 'done' ? '#22c55e' : '#f59e0b',
                      letterSpacing: '2px', marginBottom: '4px',
                    }}>
                      {def.doneLabel
                        ? def.doneLabel(config, status)
                        : (status === 'done' ? `✓ ${def.title.toUpperCase()} COMPLETE` : '■ STOPPED')}
                    </div>
                    <LossChart data={lossData} height={40} color={def.lossChartColor} />
                    <button onClick={handleResetToIdle} style={{ ...stopBtnStyle, marginTop: '8px' }}>RETRAIN</button>
                  </div>
                )}
              </>
            )}
          </>
        )}

        {/* DataPrep running state (multi-action with progress) */}
        {def.actions.type === 'multi' && status === 'running' && (
          <DataPrepRunningView stageState={stageState} accent={def.accent} />
        )}

        {/* Always-visible slots (below actions) */}
        {def.slots?.belowActions && <def.slots.belowActions {...slotProps} />}

        {/* Error display */}
        {(error || stageError) && (
          <div style={{ fontSize: '9px', color: '#ef4444' }}>{error || stageError}</div>
        )}
      </div>
    </Node>
  );
}

function DataPrepRunningView({ stageState, accent }: { stageState: Record<string, unknown>; accent: string }) {
  const progress = stageState.progress as number | undefined;
  const total = stageState.total as number | undefined;
  const message = stageState.message as string | undefined;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
      {progress != null && total != null && (
        <div style={{ width: '100%', height: '3px', background: '#e8e8e8', borderRadius: '2px', overflow: 'hidden' }}>
          <div style={{ width: `${(progress / total) * 100}%`, height: '100%', background: accent, borderRadius: '2px', transition: 'width 0.3s ease' }} />
        </div>
      )}
      {message && <div style={{ fontSize: '9px', color: '#888' }}>{message}</div>}
    </div>
  );
}
