// Board wrapper for touch and mouse: tap picks the NEAREST target within a
// generous radius (finger-sized hit areas without changing Board), long-press
// removes the piece under the finger, pinch or the +/- buttons zoom, and
// one-finger drag pans while zoomed.
import { useRef, useState } from "react";
import type { Action, Topology, View } from "../api";
import Board, { Targets } from "../board/Board";

type Pt = { x: number; y: number };
const LONG_MS = 550, SLOP = 10;
const d2 = (a: Pt, b: Pt) => (a.x - b.x) ** 2 + (a.y - b.y) ** 2;

export function nearestTarget(topo: Topology, t: Targets | undefined, p: Pt, ports: number[]): Action | null {
  if (!t) return null;
  let best: { a: Action; d: number } | null = null;
  const consider = (a: Action, at: Pt, r: number) => { const d = d2(at, p) / (r * r); if (d <= 1 && (!best || d < best.d)) best = { a, d }; };
  t.vertices?.forEach((a, v) => consider(a, topo.vertices[v], 0.48));
  t.edges?.forEach((a, e) => {
    const [u, w] = topo.edges[e].v.map((i) => topo.vertices[i]);
    consider(a, { x: (u.x + w.x) / 2, y: (u.y + w.y) / 2 }, ports.includes(e) && !t.vertices ? 1.05 : 0.42);
  });
  t.hexes?.forEach((a, h) => consider(a, topo.hexes[h], 0.92));
  return (best as { a: Action } | null)?.a ?? null;
}

export default function TouchBoard({ topo, view, targets, onPick, onLongPress, children }: {
  topo: Topology; view: View; targets?: Targets; onPick: (a: Action) => void; onLongPress: (p: Pt) => void; children?: React.ReactNode;
}) {
  const wrap = useRef<HTMLDivElement>(null);
  const [z, setZ] = useState({ s: 1, x: 0, y: 0 });
  const ptrs = useRef(new Map<number, Pt>());
  const g = useRef<{ start: Pt; moved: boolean; long: boolean; timer?: number; pinch?: { d: number; s: number; mid: Pt; x: number; y: number }; pan?: { x: number; y: number } }>({ start: { x: 0, y: 0 }, moved: false, long: false });

  const toBoard = (c: Pt): Pt | null => {
    const svg = wrap.current?.querySelector("svg.board") as SVGSVGElement | null;
    const m = svg?.getScreenCTM();
    if (!m) return null;
    const q = new DOMPoint(c.x, c.y).matrixTransform(m.inverse());
    return { x: q.x, y: q.y };
  };
  const clampZ = (s: number, x: number, y: number) => {
    const el = wrap.current, w = el?.clientWidth ?? 0, h = el?.clientHeight ?? 0;
    s = Math.min(4, Math.max(1, s));
    const mx = ((s - 1) * w) / 2, my = ((s - 1) * h) / 2;
    return { s, x: Math.max(-mx, Math.min(mx, x)), y: Math.max(-my, Math.min(my, y)) };
  };

  const down = (e: React.PointerEvent) => {
    if ((e.target as HTMLElement).closest(".an-zoom")) return;
    ptrs.current.set(e.pointerId, { x: e.clientX, y: e.clientY });
    (e.currentTarget as HTMLElement).setPointerCapture?.(e.pointerId);
    const st = g.current;
    clearTimeout(st.timer);
    if (ptrs.current.size === 1) {
      Object.assign(st, { start: { x: e.clientX, y: e.clientY }, moved: false, long: false, pinch: undefined, pan: { x: z.x, y: z.y } });
      st.timer = window.setTimeout(() => {
        if (st.moved || ptrs.current.size !== 1) return;
        st.long = true;
        const p = toBoard(st.start);
        if (p) { navigator.vibrate?.(25); onLongPress(p); }
      }, LONG_MS);
    } else if (ptrs.current.size === 2) {
      const [a, b] = [...ptrs.current.values()];
      st.moved = true;
      st.pinch = { d: Math.sqrt(d2(a, b)), s: z.s, mid: { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 }, x: z.x, y: z.y };
    }
  };
  const move = (e: React.PointerEvent) => {
    if (!ptrs.current.has(e.pointerId)) return;
    ptrs.current.set(e.pointerId, { x: e.clientX, y: e.clientY });
    const st = g.current;
    if (st.pinch && ptrs.current.size >= 2) {
      const [a, b] = [...ptrs.current.values()];
      const mid = { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
      setZ(clampZ(st.pinch.s * (Math.sqrt(d2(a, b)) / st.pinch.d), st.pinch.x + mid.x - st.pinch.mid.x, st.pinch.y + mid.y - st.pinch.mid.y));
      return;
    }
    if (d2(st.start, { x: e.clientX, y: e.clientY }) > SLOP * SLOP) { st.moved = true; clearTimeout(st.timer); }
    if (st.moved && z.s > 1 && st.pan) setZ(clampZ(z.s, st.pan.x + e.clientX - st.start.x, st.pan.y + e.clientY - st.start.y));
  };
  const up = (e: React.PointerEvent) => {
    if (!ptrs.current.has(e.pointerId)) return;
    ptrs.current.delete(e.pointerId);
    const st = g.current;
    clearTimeout(st.timer);
    if (ptrs.current.size === 0 && !st.moved && !st.long) {
      const p = toBoard({ x: e.clientX, y: e.clientY });
      const a = p && nearestTarget(topo, targets, p, view.layout.port_edges);
      if (a) onPick(a);
    }
    if (ptrs.current.size === 0) st.pinch = undefined;
  };

  return (
    <div className="an-touch" ref={wrap} onPointerDown={down} onPointerMove={move} onPointerUp={up} onPointerCancel={up}
      onContextMenu={(e) => e.preventDefault()}
      data-zoom={z.s.toFixed(2)}>
      <div className="an-zoomable" style={{ transform: `translate(${z.x}px, ${z.y}px) scale(${z.s})` }}>
        <Board topo={topo} view={view} targets={targets}>{children}</Board>
      </div>
      <div className="an-zoom" role="group" aria-label="Zoom">
        <button className="btn" onClick={() => setZ(clampZ(z.s * 1.4, z.x, z.y))} aria-label="Zoom in">+</button>
        <button className="btn" onClick={() => setZ(clampZ(z.s / 1.4, z.x / 1.4, z.y / 1.4))} aria-label="Zoom out" disabled={z.s <= 1}>−</button>
        {z.s > 1 && <button className="btn" onClick={() => setZ({ s: 1, x: 0, y: 0 })} aria-label="Reset zoom">1×</button>}
      </div>
    </div>
  );
}
