'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import * as api from '@/lib/api';
import type { SlotProps } from '@/lib/module-schema';

interface SubsetStatus {
  raw_shards: number;
  tokenized_shards: number;
}

export function DataPrepStatusSlot({ config, stageState }: SlotProps) {
  const [subsetStatuses, setSubsetStatuses] = useState<Record<string, SubsetStatus>>({});
  const prevStatus = useRef(stageState.status);
  const subset = config.subset as string;

  const fetchStatus = useCallback(async () => {
    try {
      const s = await api.dataStatus();
      setSubsetStatuses(s);
    } catch { /* ignore */ }
  }, []);

  // Fetch on mount
  useEffect(() => { fetchStatus(); }, [fetchStatus]);

  // Re-fetch when a job completes
  useEffect(() => {
    if (prevStatus.current === 'running' && stageState.status === 'done') {
      fetchStatus();
    }
    prevStatus.current = stageState.status;
  }, [stageState.status, fetchStatus]);

  const currentStatus = subsetStatuses[subset];
  if (!currentStatus) return null;

  return (
    <div style={{ fontSize: '9px', display: 'flex', gap: '8px' }}>
      <span style={{ color: currentStatus.raw_shards > 0 ? '#22c55e' : '#bbb' }}>
        {currentStatus.raw_shards > 0 ? `downloaded ${currentStatus.raw_shards} shards` : 'not downloaded'}
      </span>
      <span style={{ color: '#ddd' }}>|</span>
      <span style={{ color: currentStatus.tokenized_shards > 0 ? '#22c55e' : '#bbb' }}>
        {currentStatus.tokenized_shards > 0 ? `tokenized ${currentStatus.tokenized_shards} shards` : 'not tokenized'}
      </span>
    </div>
  );
}
