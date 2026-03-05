'use client';

import { useState, useRef, useEffect } from 'react';
import { useStore } from '@/lib/store';
import { moduleRegistry } from '@/lib/module-registry';

export function AddModuleCard() {
  const [open, setOpen] = useState(false);
  const activeModuleIds = useStore((s) => s.activeModuleIds);
  const addModule = useStore((s) => s.addModule);
  const popoverRef = useRef<HTMLDivElement>(null);

  const availableModules = moduleRegistry.filter((m) => !activeModuleIds.includes(m.id));

  // Close popover on outside click
  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      if (popoverRef.current && !popoverRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [open]);

  if (availableModules.length === 0) return null;

  return (
    <div style={{ position: 'relative' }} ref={popoverRef}>
      <button
        onClick={() => setOpen(!open)}
        style={{
          width: '100%',
          minHeight: '80px',
          background: 'transparent',
          border: '2px dashed #e0e0e0',
          borderRadius: '3px',
          cursor: 'pointer',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          gap: '4px',
          fontFamily: "'JetBrains Mono', monospace",
          transition: 'border-color 0.2s',
        }}
        onMouseEnter={(e) => { (e.target as HTMLElement).style.borderColor = '#bbb'; }}
        onMouseLeave={(e) => { (e.target as HTMLElement).style.borderColor = '#e0e0e0'; }}
      >
        <span style={{ fontSize: '16px', color: '#ccc', lineHeight: 1 }}>+</span>
        <span style={{ fontSize: '9px', color: '#bbb', letterSpacing: '1px' }}>ADD MODULE</span>
      </button>

      {/* Popover */}
      {open && (
        <div
          style={{
            position: 'absolute',
            top: '50%',
            left: '50%',
            transform: 'translate(-50%, -50%)',
            background: '#fff',
            border: '1px solid #e0e0e0',
            borderRadius: '3px',
            boxShadow: '0 4px 16px rgba(0,0,0,0.08)',
            zIndex: 50,
            minWidth: '160px',
            overflow: 'hidden',
          }}
        >
          {availableModules.map((m) => (
            <button
              key={m.id}
              onClick={() => {
                addModule(m.id);
                setOpen(false);
              }}
              style={{
                width: '100%',
                padding: '8px 12px',
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                background: 'none',
                border: 'none',
                borderBottom: '1px solid #f5f5f5',
                cursor: 'pointer',
                fontFamily: "'JetBrains Mono', monospace",
                fontSize: '10px',
                color: '#666',
                letterSpacing: '0.5px',
                textAlign: 'left',
              }}
              onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.background = '#fafafa'; }}
              onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.background = 'none'; }}
            >
              <span
                style={{
                  width: 6,
                  height: 6,
                  borderRadius: '50%',
                  background: m.accent,
                  flexShrink: 0,
                }}
              />
              {m.title}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
