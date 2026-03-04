import { useEffect, useRef } from 'react';
import { useStore } from '@/lib/store';
import type { MetricPoint, StageId } from '@/lib/types';

const WS_URL = typeof window !== 'undefined'
  ? `ws://${window.location.hostname}:8000/ws`
  : 'ws://localhost:8000/ws';
const RECONNECT_DELAY = 3000;

export function useWebSocket() {
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout>>();
  const {
    setStageStatus,
    appendMetric,
    setEvalResults,
    setWsConnected,
    setActiveCheckpoint,
  } = useStore();

  useEffect(() => {
    function connect() {
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;

      ws.onopen = () => {
        setWsConnected(true);
      };

      ws.onclose = () => {
        setWsConnected(false);
        reconnectTimer.current = setTimeout(connect, RECONNECT_DELAY);
      };

      ws.onerror = () => {
        ws.close();
      };

      ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data);
          handleMessage(msg);
        } catch {
          // ignore malformed messages
        }
      };
    }

    function handleMessage(msg: { type: string; data: Record<string, unknown> }) {
      switch (msg.type) {
        case 'init': {
          const stages = msg.data.stages as Record<string, string>;
          if (stages) {
            Object.entries(stages).forEach(([stage, status]) => {
              setStageStatus(stage as StageId, status as any);
            });
          }
          if (msg.data.checkpoint) {
            setActiveCheckpoint(msg.data.checkpoint as string);
          }
          break;
        }
        case 'train_step': {
          const d = msg.data as unknown as MetricPoint;
          appendMetric(d.stage, d);
          setStageStatus(d.stage as StageId, 'running', { currentStep: d.step });
          break;
        }
        case 'status_update': {
          const stage = msg.data.stage as StageId;
          const status = msg.data.status as string;
          const { stage: _, status: __, ...extra } = msg.data;
          setStageStatus(stage, status as any, extra);
          break;
        }
        case 'train_complete': {
          const stage = msg.data.stage as StageId;
          setStageStatus(stage, 'done');
          break;
        }
        case 'eval_complete': {
          const results = msg.data.results as Record<string, Record<string, number>>;
          setEvalResults(results);
          break;
        }
        case 'eval_result':
          break;
        default:
          break;
      }
    }

    connect();

    return () => {
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
      wsRef.current?.close();
    };
  }, [setStageStatus, appendMetric, setEvalResults, setWsConnected, setActiveCheckpoint]);
}
