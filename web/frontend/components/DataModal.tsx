'use client';

import { useState, useEffect, useCallback, useMemo } from 'react';
import { useStore } from '@/lib/store';
import { dataPrepModule } from '@/lib/module-registry';
import { FieldRenderer } from './fields/FieldRenderer';
import * as api from '@/lib/api';
import type { FieldDefWithVisibility, SlotProps } from '@/lib/module-schema';

const apiMap: Record<string, (body: unknown) => Promise<unknown>> = {
  downloadData: (body) => api.downloadData(body as Parameters<typeof api.downloadData>[0]),
  tokenizeData: (body) => api.tokenizeData(body as Parameters<typeof api.tokenizeData>[0]),
  validateData: (body) => api.validateData(body as Parameters<typeof api.validateData>[0]),
  downloadSFTData: (body) => api.downloadSFTData(body as Parameters<typeof api.downloadSFTData>[0]),
  validateSFTData: (body) => api.validateSFTData(body as Parameters<typeof api.validateSFTData>[0]),
};

function buildDefaults(fields: FieldDefWithVisibility[], hidden?: Record<string, unknown>): Record<string, unknown> {
  const defaults: Record<string, unknown> = { ...(hidden || {}) };
  for (const f of fields) {
    defaults[f.key] = f.default;
  }
  return defaults;
}

interface DataDir {
  path: string;
  name: string;
  shards: number;
}

type Tab = 'pretrain' | 'sft';

interface SFTDatasetInfo {
  repo: string;
  desc: string;
}

interface SFTDownloaded {
  name: string;
  path: string;
  rows: number;
}

export function DataModal() {
  const open = useStore((s) => s.dataModalOpen);
  const setOpen = useStore((s) => s.setDataModalOpen);
  const stageState = useStore((s) => s.stages['data']);
  const status = stageState.status;
  const def = dataPrepModule;

  const initialConfig = useMemo(() => buildDefaults(def.fields, def.hiddenDefaults), [def]);
  const [config, setConfig] = useState<Record<string, unknown>>(initialConfig);
  const [error, setError] = useState<string | null>(null);
  const [dataDirs, setDataDirs] = useState<DataDir[]>([]);
  const [tab, setTab] = useState<Tab>('pretrain');

  // SFT state
  const [sftDataset, setSftDataset] = useState('OpenOrca');
  const [sftCustomRepo, setSftCustomRepo] = useState('');
  const [sftSubset, setSftSubset] = useState('');
  const [sftMaxRows, setSftMaxRows] = useState<number | null>(null);
  const [sftAvailable, setSftAvailable] = useState<Record<string, SFTDatasetInfo>>({});
  const [sftDownloaded, setSftDownloaded] = useState<SFTDownloaded[]>([]);

  const sftDatasetNames = useMemo(() => [...Object.keys(sftAvailable), 'Custom'], [sftAvailable]);
  const sftDesc = sftDataset !== 'Custom' ? sftAvailable[sftDataset]?.desc : undefined;

  // Fetch available data dirs (pretrain)
  useEffect(() => {
    if (!open) return;
    fetch(`http://${window.location.hostname}:8000/api/data/dirs`)
      .then((r) => r.json())
      .then((dirs) => setDataDirs(dirs))
      .catch(() => {});
  }, [open]);

  // Fetch SFT datasets
  useEffect(() => {
    if (!open || tab !== 'sft') return;
    api.listSFTDatasets()
      .then((data) => {
        setSftAvailable(data.available);
        setSftDownloaded(data.downloaded);
      })
      .catch(() => {});
  }, [open, tab]);

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

  const handleAction = useCallback(async (index: number) => {
    if (def.actions.type !== 'multi') return;
    const action = def.actions.buttons[index];
    setError(null);
    const payload = action.buildPayload(config, {});
    try {
      const fn = apiMap[action.apiCall];
      if (!fn) throw new Error(`Unknown API: ${action.apiCall}`);
      await fn(payload);
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e);
      setError(msg);
    }
  }, [def, config]);

  const handleSFTAction = useCallback(async (action: 'download' | 'validate') => {
    setError(null);
    const dataset = sftDataset === 'Custom' ? sftCustomRepo : sftDataset;
    if (!dataset) {
      setError('Please enter a dataset name');
      return;
    }
    try {
      if (action === 'download') {
        await apiMap.downloadSFTData({
          dataset,
          subset: sftSubset || undefined,
          max_rows: sftMaxRows || undefined,
        });
      } else {
        const path = `./data/sft/${dataset}.jsonl`;
        await apiMap.validateSFTData({ data_path: path });
      }
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e);
      setError(msg);
    }
  }, [sftDataset, sftCustomRepo, sftSubset, sftMaxRows]);

  const slotProps: SlotProps = useMemo(() => ({
    config,
    stageState,
    store: {},
  }), [config, stageState]);

  if (!open) return null;

  return (
    <div
      onClick={() => setOpen(false)}
      style={{
        position: 'fixed',
        inset: 0,
        zIndex: 1000,
        background: 'rgba(0,0,0,0.3)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        fontFamily: "'JetBrains Mono', monospace",
      }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          background: '#fff',
          borderRadius: '6px',
          border: '1px solid #e0e0e0',
          boxShadow: '0 8px 32px rgba(0,0,0,0.1)',
          width: '420px',
          maxHeight: '80vh',
          overflow: 'auto',
        }}
      >
        {/* Modal header */}
        <div
          style={{
            padding: '12px 16px',
            borderBottom: '1px solid #f0f0f0',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span style={{ width: 6, height: 6, borderRadius: '50%', background: def.accent }} />
            <span style={{ fontSize: '10px', fontWeight: 600, letterSpacing: '1.5px', color: '#555', textTransform: 'uppercase' }}>
              {def.title}
            </span>
          </div>
          <button
            onClick={() => setOpen(false)}
            style={{
              background: 'none',
              border: 'none',
              cursor: 'pointer',
              fontSize: '14px',
              color: '#ccc',
              fontFamily: "'JetBrains Mono', monospace",
            }}
          >
            ×
          </button>
        </div>

        {/* Tab switcher */}
        <div style={{ display: 'flex', gap: '0', borderBottom: '1px solid #f0f0f0' }}>
          {(['pretrain', 'sft'] as Tab[]).map((t) => (
            <button
              key={t}
              onClick={() => { setTab(t); setError(null); }}
              style={{
                flex: 1,
                padding: '8px 12px',
                fontSize: '9px',
                fontWeight: 600,
                fontFamily: 'inherit',
                letterSpacing: '1.5px',
                textTransform: 'uppercase',
                background: 'none',
                border: 'none',
                borderBottom: tab === t ? `2px solid ${def.accent}` : '2px solid transparent',
                color: tab === t ? '#333' : '#aaa',
                cursor: 'pointer',
                transition: 'color 0.15s, border-color 0.15s',
              }}
            >
              {t}
            </button>
          ))}
        </div>

        {/* Modal body */}
        <div style={{ padding: '16px', display: 'flex', flexDirection: 'column', gap: '8px', fontSize: '11px' }}>

          {/* ===== PRETRAIN TAB ===== */}
          {tab === 'pretrain' && (
            <>
              {/* Fields */}
              {def.fields.map((field) => (
                <FieldRenderer
                  key={field.key}
                  field={field}
                  value={config[field.key]}
                  onChange={(v) => updateConfig(field.key, v)}
                />
              ))}

              {/* Status slot */}
              {def.slots?.belowFieldsIdle && status === 'idle' && (
                <def.slots.belowFieldsIdle {...slotProps} />
              )}

              {/* Action buttons */}
              {def.actions.type === 'multi' && (
                <div style={{ display: 'flex', gap: '4px' }}>
                  {def.actions.buttons.map((btn, i) => (
                    <button
                      key={i}
                      onClick={() => handleAction(i)}
                      disabled={status === 'running'}
                      style={{
                        flex: 1, padding: '8px', fontSize: '9px', fontWeight: 600,
                        fontFamily: 'inherit', letterSpacing: '0.5px', background: '#333',
                        color: '#fff', border: 'none', borderRadius: '3px', cursor: 'pointer',
                      }}
                    >
                      {typeof btn.label === 'function' ? btn.label(config) : btn.label}
                    </button>
                  ))}
                </div>
              )}

              {/* Available datasets */}
              {dataDirs.length > 0 && (
                <div style={{ marginTop: '8px' }}>
                  <div style={{ fontSize: '9px', color: '#999', letterSpacing: '1px', marginBottom: '4px', textTransform: 'uppercase' }}>
                    Available Datasets
                  </div>
                  {dataDirs.map((dir) => (
                    <div
                      key={dir.path}
                      style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        padding: '4px 0',
                        fontSize: '10px',
                        color: '#666',
                        borderBottom: '1px solid #f5f5f5',
                      }}
                    >
                      <span>{dir.name}</span>
                      <span style={{ color: '#999' }}>{dir.shards} shards</span>
                    </div>
                  ))}
                </div>
              )}
            </>
          )}

          {/* ===== SFT TAB ===== */}
          {tab === 'sft' && (
            <>
              {/* Dataset select */}
              <div>
                <label style={{ fontSize: '9px', color: '#999', letterSpacing: '1px', textTransform: 'uppercase', display: 'block', marginBottom: '4px' }}>
                  Dataset
                </label>
                <select
                  value={sftDataset}
                  onChange={(e) => setSftDataset(e.target.value)}
                  style={{
                    width: '100%', padding: '6px 8px', fontSize: '11px',
                    fontFamily: 'inherit', border: '1px solid #ddd', borderRadius: '3px',
                    background: '#fafafa', outline: 'none',
                  }}
                >
                  {sftDatasetNames.map((d) => (
                    <option key={d} value={d}>{d}</option>
                  ))}
                </select>
                {sftDesc && (
                  <div style={{ fontSize: '9px', color: '#aaa', marginTop: '3px' }}>{sftDesc}</div>
                )}
              </div>

              {/* Custom repo field */}
              {sftDataset === 'Custom' && (
                <div>
                  <label style={{ fontSize: '9px', color: '#999', letterSpacing: '1px', textTransform: 'uppercase', display: 'block', marginBottom: '4px' }}>
                    HuggingFace Repo
                  </label>
                  <input
                    type="text"
                    value={sftCustomRepo}
                    onChange={(e) => setSftCustomRepo(e.target.value)}
                    placeholder="user/repo"
                    style={{
                      width: '100%', padding: '6px 8px', fontSize: '11px',
                      fontFamily: 'inherit', border: '1px solid #ddd', borderRadius: '3px',
                      background: '#fafafa', outline: 'none', boxSizing: 'border-box',
                    }}
                  />
                </div>
              )}

              {/* Subset field */}
              <div>
                <label style={{ fontSize: '9px', color: '#999', letterSpacing: '1px', textTransform: 'uppercase', display: 'block', marginBottom: '4px' }}>
                  Subset / Config <span style={{ color: '#ccc' }}>(optional)</span>
                </label>
                <input
                  type="text"
                  value={sftSubset}
                  onChange={(e) => setSftSubset(e.target.value)}
                  placeholder="e.g. default, en, ..."
                  style={{
                    width: '100%', padding: '6px 8px', fontSize: '11px',
                    fontFamily: 'inherit', border: '1px solid #ddd', borderRadius: '3px',
                    background: '#fafafa', outline: 'none', boxSizing: 'border-box',
                  }}
                />
              </div>

              {/* Max rows */}
              <div>
                <label style={{ fontSize: '9px', color: '#999', letterSpacing: '1px', textTransform: 'uppercase', display: 'block', marginBottom: '4px' }}>
                  Max Rows <span style={{ color: '#ccc' }}>(optional, blank = all)</span>
                </label>
                <input
                  type="number"
                  value={sftMaxRows ?? ''}
                  onChange={(e) => setSftMaxRows(e.target.value ? parseInt(e.target.value, 10) : null)}
                  placeholder="e.g. 10000"
                  min={1}
                  style={{
                    width: '100%', padding: '6px 8px', fontSize: '11px',
                    fontFamily: 'inherit', border: '1px solid #ddd', borderRadius: '3px',
                    background: '#fafafa', outline: 'none', boxSizing: 'border-box',
                  }}
                />
              </div>

              {/* SFT action buttons */}
              <div style={{ display: 'flex', gap: '4px' }}>
                <button
                  onClick={() => handleSFTAction('download')}
                  disabled={status === 'running'}
                  style={{
                    flex: 1, padding: '8px', fontSize: '9px', fontWeight: 600,
                    fontFamily: 'inherit', letterSpacing: '0.5px', background: '#333',
                    color: '#fff', border: 'none', borderRadius: '3px', cursor: 'pointer',
                  }}
                >
                  DOWNLOAD
                </button>
                <button
                  onClick={() => handleSFTAction('validate')}
                  disabled={status === 'running'}
                  style={{
                    flex: 1, padding: '8px', fontSize: '9px', fontWeight: 600,
                    fontFamily: 'inherit', letterSpacing: '0.5px', background: '#333',
                    color: '#fff', border: 'none', borderRadius: '3px', cursor: 'pointer',
                  }}
                >
                  VALIDATE
                </button>
              </div>

              {/* Downloaded SFT datasets */}
              {sftDownloaded.length > 0 && (
                <div style={{ marginTop: '8px' }}>
                  <div style={{ fontSize: '9px', color: '#999', letterSpacing: '1px', marginBottom: '4px', textTransform: 'uppercase' }}>
                    Downloaded Datasets
                  </div>
                  {sftDownloaded.map((ds) => (
                    <div
                      key={ds.path}
                      style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        padding: '4px 0',
                        fontSize: '10px',
                        color: '#666',
                        borderBottom: '1px solid #f5f5f5',
                      }}
                    >
                      <span>{ds.name}</span>
                      <span style={{ color: '#999' }}>{ds.rows.toLocaleString()} rows</span>
                    </div>
                  ))}
                </div>
              )}
            </>
          )}

          {/* Running progress (shared) */}
          {status === 'running' && (
            <DataPrepProgress stageState={stageState} accent={def.accent} />
          )}

          {/* Error */}
          {error && (
            <div style={{ fontSize: '9px', color: '#ef4444' }}>{error}</div>
          )}
        </div>
      </div>
    </div>
  );
}

function DataPrepProgress({ stageState, accent }: { stageState: Record<string, unknown>; accent: string }) {
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
