import { useState, useEffect, useRef, useCallback } from 'react';

export function useDraggable(
  initialX: number,
  initialY: number,
  id: string,
  onPosChange: (id: string, x: number, y: number, w: number, h: number) => void,
) {
  const [pos, setPos] = useState({ x: initialX, y: initialY });
  const dragging = useRef(false);
  const offset = useRef({ x: 0, y: 0 });
  const ref = useRef<HTMLDivElement>(null);

  const onMouseDown = useCallback(
    (e: React.MouseEvent) => {
      const tag = (e.target as HTMLElement).tagName;
      if (['INPUT', 'TEXTAREA', 'SELECT', 'BUTTON'].includes(tag)) return;
      dragging.current = true;
      offset.current = { x: e.clientX - pos.x, y: e.clientY - pos.y };
      e.preventDefault();
    },
    [pos],
  );

  useEffect(() => {
    const onMove = (e: MouseEvent) => {
      if (dragging.current) {
        setPos({ x: e.clientX - offset.current.x, y: e.clientY - offset.current.y });
      }
    };
    const onUp = () => {
      dragging.current = false;
    };
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
    return () => {
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
    };
  }, []);

  useEffect(() => {
    if (ref.current) {
      const r = ref.current.getBoundingClientRect();
      onPosChange(id, pos.x, pos.y, r.width, r.height);
    }
  }, [pos, id, onPosChange]);

  return { pos, onMouseDown, ref };
}
