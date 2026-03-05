'use client';

import { useStore } from '@/lib/store';
import { useWebSocket } from '@/hooks/useWebSocket';
import { ModuleNode } from './ModuleNode';
import { Header } from './Header';
import { DataModal } from './DataModal';
import { AddModuleCard } from './AddModuleCard';
import { moduleRegistry } from '@/lib/module-registry';

export function Dashboard() {
  useWebSocket();

  const activeModuleIds = useStore((s) => s.activeModuleIds);
  const removeModule = useStore((s) => s.removeModule);

  const activeModules = moduleRegistry.filter((m) => activeModuleIds.includes(m.id));

  return (
    <div
      style={{
        width: '100vw',
        minHeight: '100vh',
        fontFamily: "'JetBrains Mono', monospace",
        background: '#fafafa',
      }}
    >
      <Header />

      {/* Grid */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))',
          gap: '16px',
          padding: '24px',
        }}
      >
        {activeModules.map((def) => (
          <ModuleNode
            key={def.id}
            definition={def}
            onRemove={() => removeModule(def.id)}
          />
        ))}
        <AddModuleCard />
      </div>

      {/* Data Modal */}
      <DataModal />

      {/* Global styles */}
      <style
        dangerouslySetInnerHTML={{
          __html: `
            * { box-sizing: border-box; }
            input[type=range] { -webkit-appearance: none; background: #e8e8e8; border-radius: 2px; outline: none; }
            input[type=range]::-webkit-slider-thumb { -webkit-appearance: none; width: 10px; height: 10px; border-radius: 50%; background: #999; cursor: pointer; }
            ::selection { background: #dbeafe; }
          `,
        }}
      />
    </div>
  );
}
