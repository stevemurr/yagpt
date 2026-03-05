'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import { useStore } from '@/lib/store';
import * as expApi from '@/lib/experiment-api';
import type { ExperimentSummary, Experiment } from '@/lib/experiment-api';

export function ExperimentPicker() {
  const [experiments, setExperiments] = useState<ExperimentSummary[]>([]);
  const [current, setCurrent] = useState<Experiment | null>(null);
  const [open, setOpen] = useState(false);
  const [creating, setCreating] = useState(false);
  const [newName, setNewName] = useState('');
  const dropdownRef = useRef<HTMLDivElement>(null);
  const saveTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const activeModuleIds = useStore((s) => s.activeModuleIds);
  const addModule = useStore((s) => s.addModule);
  const removeModule = useStore((s) => s.removeModule);
  const setCurrentExperimentId = useStore((s) => s.setCurrentExperimentId);
  const moduleConfigs = useStore((s) => s.moduleConfigs);
  const setAllModuleConfigs = useStore((s) => s.setAllModuleConfigs);

  // Fetch experiments on mount and restore last selected
  useEffect(() => {
    expApi.listExperiments().then((exps) => {
      setExperiments(exps);
      // Restore last selected experiment from localStorage
      const savedId = localStorage.getItem('yagpt_experiment_id');
      if (savedId) {
        const id = parseInt(savedId, 10);
        if (exps.some((e) => e.id === id)) {
          loadExperiment(id);
        }
      }
    }).catch(() => {});
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Close dropdown on outside click
  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setOpen(false);
        setCreating(false);
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [open]);

  // Track latest values for beforeunload flush
  const latestRef = useRef({ current, activeModuleIds, moduleConfigs });
  latestRef.current = { current, activeModuleIds, moduleConfigs };

  // Auto-save: debounced save when activeModuleIds or moduleConfigs change
  useEffect(() => {
    if (!current) return;
    if (saveTimerRef.current) clearTimeout(saveTimerRef.current);
    saveTimerRef.current = setTimeout(() => {
      expApi.updateExperiment(current.id, {
        active_modules: activeModuleIds,
        module_configs: moduleConfigs,
      }).catch(() => {});
    }, 2000);
    return () => {
      if (saveTimerRef.current) clearTimeout(saveTimerRef.current);
    };
  }, [activeModuleIds, moduleConfigs, current]);

  // Flush pending save on page unload
  useEffect(() => {
    const handleUnload = () => {
      const { current: exp, activeModuleIds: mods, moduleConfigs: cfgs } = latestRef.current;
      if (!exp) return;
      const body = JSON.stringify({ active_modules: mods, module_configs: cfgs });
      const url = `http://${window.location.hostname}:8000/api/experiments/${exp.id}`;
      fetch(url, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body,
        keepalive: true,
      });
    };
    window.addEventListener('beforeunload', handleUnload);
    return () => window.removeEventListener('beforeunload', handleUnload);
  }, []);

  const loadExperiment = useCallback(async (id: number) => {
    try {
      const exp = await expApi.getExperiment(id);
      setCurrent(exp);
      setCurrentExperimentId(exp.id);
      localStorage.setItem('yagpt_experiment_id', String(exp.id));
      setOpen(false);

      // Restore module configs first so ModuleNodes pick them up when mounted
      if (exp.module_configs && Object.keys(exp.module_configs).length > 0) {
        setAllModuleConfigs(exp.module_configs);
      }

      // Set active modules from experiment
      const currentIds = useStore.getState().activeModuleIds;
      // Remove modules not in experiment
      for (const mid of currentIds) {
        if (!exp.active_modules.includes(mid)) {
          removeModule(mid);
        }
      }
      // Add modules from experiment
      for (const mid of exp.active_modules) {
        addModule(mid);
      }
    } catch {
      // ignore
    }
  }, [addModule, removeModule, setCurrentExperimentId, setAllModuleConfigs]);

  const handleCreate = useCallback(async () => {
    if (!newName.trim()) return;
    try {
      const exp = await expApi.createExperiment({
        name: newName.trim(),
        active_modules: activeModuleIds,
        module_configs: moduleConfigs,
      });
      setCurrent(exp);
      setCurrentExperimentId(exp.id);
      localStorage.setItem('yagpt_experiment_id', String(exp.id));
      setExperiments((prev) => [exp, ...prev]);
      setCreating(false);
      setNewName('');
      setOpen(false);
    } catch {
      // ignore
    }
  }, [newName, activeModuleIds, moduleConfigs]);

  const handleDelete = useCallback(async (id: number, e: React.MouseEvent) => {
    e.stopPropagation();
    try {
      await expApi.deleteExperiment(id);
      setExperiments((prev) => prev.filter((exp) => exp.id !== id));
      if (current?.id === id) {
        setCurrent(null);
        setCurrentExperimentId(null);
        localStorage.removeItem('yagpt_experiment_id');
      }
    } catch {
      // ignore
    }
  }, [current, setCurrentExperimentId]);

  return (
    <div style={{ position: 'relative' }} ref={dropdownRef}>
      <button
        onClick={() => setOpen(!open)}
        style={{
          padding: '4px 12px',
          fontSize: '9px',
          fontWeight: 600,
          letterSpacing: '0.5px',
          fontFamily: "'JetBrains Mono', monospace",
          background: current ? '#f0f9ff' : '#fafafa',
          color: current ? '#3b82f6' : '#888',
          border: `1px solid ${current ? '#bfdbfe' : '#e0e0e0'}`,
          borderRadius: '3px',
          cursor: 'pointer',
          maxWidth: '200px',
          overflow: 'hidden',
          textOverflow: 'ellipsis',
          whiteSpace: 'nowrap',
        }}
      >
        {current ? current.name : 'No Experiment'}
      </button>

      {open && (
        <div
          style={{
            position: 'absolute',
            top: '100%',
            left: '50%',
            transform: 'translateX(-50%)',
            marginTop: '4px',
            background: '#fff',
            border: '1px solid #e0e0e0',
            borderRadius: '3px',
            boxShadow: '0 4px 16px rgba(0,0,0,0.08)',
            zIndex: 100,
            minWidth: '200px',
            overflow: 'hidden',
          }}
        >
          {/* Experiment list */}
          {experiments.map((exp) => (
            <div
              key={exp.id}
              onClick={() => loadExperiment(exp.id)}
              style={{
                padding: '8px 12px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                gap: '8px',
                cursor: 'pointer',
                borderBottom: '1px solid #f5f5f5',
                fontSize: '10px',
                fontFamily: "'JetBrains Mono', monospace",
                color: exp.id === current?.id ? '#3b82f6' : '#666',
                background: exp.id === current?.id ? '#f0f9ff' : '#fff',
              }}
              onMouseEnter={(e) => {
                if (exp.id !== current?.id) (e.currentTarget as HTMLElement).style.background = '#fafafa';
              }}
              onMouseLeave={(e) => {
                if (exp.id !== current?.id) (e.currentTarget as HTMLElement).style.background = '#fff';
              }}
            >
              <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                {exp.name}
              </span>
              <button
                onClick={(e) => handleDelete(exp.id, e)}
                style={{
                  background: 'none',
                  border: 'none',
                  cursor: 'pointer',
                  fontSize: '11px',
                  color: '#ccc',
                  padding: '0 2px',
                  fontFamily: "'JetBrains Mono', monospace",
                  flexShrink: 0,
                }}
              >
                ×
              </button>
            </div>
          ))}

          {/* Create new */}
          {creating ? (
            <div style={{ padding: '8px 12px', display: 'flex', gap: '4px' }}>
              <input
                autoFocus
                value={newName}
                onChange={(e) => setNewName(e.target.value)}
                onKeyDown={(e) => { if (e.key === 'Enter') handleCreate(); if (e.key === 'Escape') setCreating(false); }}
                placeholder="experiment name"
                style={{
                  flex: 1,
                  padding: '4px 6px',
                  fontSize: '10px',
                  fontFamily: "'JetBrains Mono', monospace",
                  border: '1px solid #e0e0e0',
                  borderRadius: '3px',
                  outline: 'none',
                }}
              />
              <button
                onClick={handleCreate}
                style={{
                  padding: '4px 8px',
                  fontSize: '9px',
                  fontWeight: 600,
                  fontFamily: "'JetBrains Mono', monospace",
                  background: '#333',
                  color: '#fff',
                  border: 'none',
                  borderRadius: '3px',
                  cursor: 'pointer',
                }}
              >
                OK
              </button>
            </div>
          ) : (
            <button
              onClick={() => setCreating(true)}
              style={{
                width: '100%',
                padding: '8px 12px',
                fontSize: '9px',
                fontWeight: 600,
                letterSpacing: '0.5px',
                fontFamily: "'JetBrains Mono', monospace",
                background: 'none',
                color: '#3b82f6',
                border: 'none',
                cursor: 'pointer',
                textAlign: 'left',
              }}
            >
              + New Experiment
            </button>
          )}
        </div>
      )}
    </div>
  );
}
