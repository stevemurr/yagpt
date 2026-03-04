'use client';

import { useState, useRef, useEffect } from 'react';
import type { DataDirFieldDef } from '@/lib/module-schema';
import * as api from '@/lib/api';
import { labelStyle, inputStyle } from './styles';

interface DataDirInfo {
  path: string;
  name: string;
  shards: number;
}

interface Props {
  field: DataDirFieldDef;
  value: string;
  onChange: (value: string) => void;
}

export function DataDirPicker({ field, value, onChange }: Props) {
  const [open, setOpen] = useState(false);
  const [dirs, setDirs] = useState<DataDirInfo[]>([]);
  const [loading, setLoading] = useState(false);
  const cacheRef = useRef<{ data: DataDirInfo[]; ts: number } | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const fetchDirs = async () => {
    const now = Date.now();
    if (cacheRef.current && now - cacheRef.current.ts < 10000) {
      setDirs(cacheRef.current.data);
      return;
    }
    setLoading(true);
    try {
      const data = await api.dataDirs();
      cacheRef.current = { data, ts: now };
      setDirs(data);
    } catch {
      setDirs([]);
    } finally {
      setLoading(false);
    }
  };

  const handleToggle = () => {
    if (!open) fetchDirs();
    setOpen(!open);
  };

  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(e.target as HTMLElement)) {
        setOpen(false);
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [open]);

  return (
    <div ref={containerRef} style={{ position: 'relative' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
        <span style={{ ...labelStyle, minWidth: field.minWidth || '45px' }}>{field.label}</span>
        <input
          type="text"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder={field.placeholder}
          style={inputStyle}
        />
        <button
          onClick={handleToggle}
          style={{
            padding: '3px 5px', fontSize: '10px', fontFamily: 'inherit',
            background: open ? '#333' : '#f5f5f5', color: open ? '#fff' : '#888',
            border: `1px solid ${open ? '#333' : '#e0e0e0'}`, borderRadius: '3px',
            cursor: 'pointer', lineHeight: 1, flexShrink: 0,
          }}
          title="Browse data directories"
        >
          {'[ ]'}
        </button>
      </div>

      {open && (
        <div style={{
          position: 'absolute', top: '100%', left: 0, right: 0,
          marginTop: '4px', background: '#fff', border: '1px solid #e0e0e0',
          borderRadius: '3px', boxShadow: '0 4px 12px rgba(0,0,0,0.08)',
          zIndex: 50, maxHeight: '200px', overflow: 'auto',
        }}>
          {loading && (
            <div style={{ padding: '8px', fontSize: '9px', color: '#999', textAlign: 'center' }}>
              loading...
            </div>
          )}
          {!loading && dirs.length === 0 && (
            <div style={{ padding: '8px', fontSize: '9px', color: '#999', textAlign: 'center' }}>
              no tokenized data found
            </div>
          )}
          {!loading && dirs.map((dir) => (
            <button
              key={dir.path}
              onClick={() => { onChange(dir.path); setOpen(false); }}
              style={{
                display: 'block', width: '100%', padding: '6px 8px',
                fontSize: '9px', fontFamily: 'inherit', textAlign: 'left',
                background: 'transparent', border: 'none', borderBottom: '1px solid #f0f0f0',
                cursor: 'pointer', color: '#555',
              }}
              onMouseEnter={(e) => { (e.target as HTMLElement).style.background = '#f8f8f8'; }}
              onMouseLeave={(e) => { (e.target as HTMLElement).style.background = 'transparent'; }}
            >
              <span style={{ fontWeight: 600, color: '#333' }}>{dir.name}</span>
              <span style={{ color: '#bbb', marginLeft: '6px' }}>
                {dir.shards} shards
              </span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
