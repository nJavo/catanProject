import React, { useMemo, useState } from "react";
import type { Action, Topology, View } from "../api";
import { PLAYER_COLORS, PLAYER_INK, CREST, RES, RES_LABEL, tileName, tilesName } from "../rules";
import TerrainDefs, { hexPoints, ResourceGlyph, TERRAIN_SYMBOL } from "./Terrain";

export interface Targets {
  vertices?: Map<number, Action>;
  edges?: Map<number, Action>;
  hexes?: Map<number, Action>;
  ghost?: "settlement" | "city" | "road" | "robber";
  seat?: number;
}

// Click area around an edge: a thin quad (a stroked line has a zero-width
// bounding box when vertical, which hit-testing tools treat as not visible).
function edgeHit(a: { x: number; y: number }, b: { x: number; y: number }, w = 0.16) {
  const dx = b.x - a.x, dy = b.y - a.y, l = Math.hypot(dx, dy);
  const nx = (-dy / l) * w, ny = (dx / l) * w;
  const s = 0.2, ax = a.x + dx * s, ay = a.y + dy * s, bx = b.x - dx * s, by = b.y - dy * s;
  return `${ax + nx},${ay + ny} ${bx + nx},${by + ny} ${bx - nx},${by - ny} ${ax - nx},${ay - ny}`;
}

const PIPS: Record<number, number> = { 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1 };

export function Crest({ shape, x = 0, y = 0, s = 0.07, fill }: { shape: string; x?: number; y?: number; s?: number; fill: string }) {
  switch (shape) {
    case "diamond": return <path d={`M${x} ${y - s} L${x + s} ${y} L${x} ${y + s} L${x - s} ${y} Z`} fill={fill} />;
    case "triangle": return <path d={`M${x} ${y - s} L${x + s * 1.05} ${y + s * 0.8} L${x - s * 1.05} ${y + s * 0.8} Z`} fill={fill} />;
    case "square": return <rect x={x - s * 0.8} y={y - s * 0.8} width={s * 1.6} height={s * 1.6} fill={fill} />;
    default: return <circle cx={x} cy={y} r={s * 0.85} fill={fill} />;
  }
}

function Settlement({ x, y, p, ghost }: { x: number; y: number; p: number; ghost?: boolean }) {
  return (
    <g transform={`translate(${x} ${y})`} className={ghost ? "ghost" : "piece"}>
      <ellipse cx={0} cy={0.15} rx={0.2} ry={0.05} fill="#000" opacity={0.25} />
      <path d="M-0.15 0.13 V-0.04 L0 -0.19 L0.15 -0.04 V0.13 Z" fill={PLAYER_COLORS[p]} stroke={PLAYER_INK[p]} strokeWidth={0.035} strokeLinejoin="round" />
      <path d="M-0.15 -0.04 L0 -0.19 L0.15 -0.04" fill="none" stroke="#fff" strokeOpacity={0.35} strokeWidth={0.02} />
      <Crest shape={CREST[p]} y={0.03} s={0.055} fill={PLAYER_INK[p]} />
    </g>
  );
}

function City({ x, y, p, ghost }: { x: number; y: number; p: number; ghost?: boolean }) {
  return (
    <g transform={`translate(${x} ${y})`} className={ghost ? "ghost" : "piece"}>
      <ellipse cx={0} cy={0.17} rx={0.27} ry={0.06} fill="#000" opacity={0.25} />
      <path d="M-0.24 0.15 V-0.02 L-0.12 -0.12 L0 -0.02 V-0.1 H0.06 V-0.26 L0.15 -0.32 L0.24 -0.26 V0.15 Z"
        fill={PLAYER_COLORS[p]} stroke={PLAYER_INK[p]} strokeWidth={0.035} strokeLinejoin="round" />
      <rect x={0.115} y={-0.2} width={0.07} height={0.08} fill={PLAYER_INK[p]} opacity={0.7} />
      <Crest shape={CREST[p]} x={-0.12} y={0.05} s={0.06} fill={PLAYER_INK[p]} />
    </g>
  );
}

function Robber({ x, y, ghost }: { x: number; y: number; ghost?: boolean }) {
  return (
    <g transform={`translate(${x} ${y})`} className={ghost ? "ghost" : "robber"}>
      <ellipse cx={0} cy={0.2} rx={0.16} ry={0.05} fill="#000" opacity={0.35} />
      <path d="M-0.13 0.2 Q-0.14 0 -0.07 -0.06 Q-0.13 -0.12 -0.08 -0.2 Q0 -0.28 0.08 -0.2 Q0.13 -0.12 0.07 -0.06 Q0.14 0 0.13 0.2 Z"
        fill="#2b2a30" stroke="#0d0c10" strokeWidth={0.02} />
      <path d="M-0.04 -0.2 Q0 -0.24 0.04 -0.21" stroke="#fff" strokeOpacity={0.3} strokeWidth={0.015} fill="none" />
    </g>
  );
}

export default function Board({ topo, view, targets, onPick, compact, children }: {
  topo: Topology; view: View; targets?: Targets; onPick?: (a: Action) => void; compact?: boolean;
  children?: React.ReactNode; // overlay drawn on top, in board coordinates (analysis highlights)
}) {
  const [hover, setHover] = useState<string | null>(null);
  const { layout } = view;

  const box = useMemo(() => {
    const xs = topo.vertices.map((v) => v.x), ys = topo.vertices.map((v) => v.y);
    const m = 1.05;
    return [Math.min(...xs) - m, Math.min(...ys) - m, Math.max(...xs) - Math.min(...xs) + 2 * m, Math.max(...ys) - Math.min(...ys) + 2 * m];
  }, [topo]);

  const ports = useMemo(() => layout.port_edges.map((e, i) => {
    const [a, b] = topo.edges[e].v.map((id) => topo.vertices[id]);
    const mx = (a.x + b.x) / 2, my = (a.y + b.y) / 2;
    const len = Math.hypot(mx, my);
    const ox = mx / len, oy = my / len;
    return { a, b, x: mx + ox * 0.62, y: my + oy * 0.62, type: layout.port_types[i] };
  }), [layout, topo]);

  const rolled = view.rolled && view.last_roll ? view.last_roll[0] + view.last_roll[1] : null;
  const pick = (a?: Action) => a && onPick?.(a);
  const vpos = (id: number) => topo.vertices[id];
  // Targets are keyboard buttons too: Tab focuses one (showing the same ghost as hover), Enter or Space picks it.
  const keyTarget = (key: string, a: Action) => ({
    tabIndex: 0, onFocus: () => setHover(key), onBlur: () => setHover(null),
    // no focus on mouse press: focusing scrolls the clipped board-wrap mid-click and the click is lost
    onMouseDown: (e: React.MouseEvent) => e.preventDefault(),
    onKeyDown: (e: React.KeyboardEvent) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pick(a); } },
  });
  const tile = (h: number) => tileName(layout, h);
  const touching = (hs: number[]) => tilesName(layout, hs);

  return (
    <svg className={"board" + (compact ? " compact" : "")} viewBox={box.join(" ")} role="group" aria-label="Game board">
      <TerrainDefs />
      <rect x={box[0] - 5} y={box[1] - 5} width={box[2] + 10} height={box[3] + 10} fill="url(#g-sea)" />
      <rect x={box[0] - 5} y={box[1] - 5} width={box[2] + 10} height={box[3] + 10} fill="url(#p-waves)" />
      {/* beach: oversized hexes behind the tiles read as a coastline */}
      <g>
        {topo.hexes.map((h) => <polygon key={h.id} points={hexPoints(1.2)} transform={`translate(${h.x} ${h.y})`} fill="#e9d6a4" />)}
        {topo.hexes.map((h) => <polygon key={"s" + h.id} points={hexPoints(1.08)} transform={`translate(${h.x} ${h.y})`} fill="#d4bc82" />)}
      </g>

      {/* harbors */}
      {ports.map((pt, i) => {
        const res = RES.indexOf(pt.type as any);
        return (
          <g key={i} className="port">
            {[pt.a, pt.b].map((v, k) => (
              <line key={k} x1={v.x + (pt.x - v.x) * 0.15} y1={v.y + (pt.y - v.y) * 0.15} x2={pt.x - (pt.x - v.x) * 0.3} y2={pt.y - (pt.y - v.y) * 0.3}
                stroke="#7a5a36" strokeWidth={0.07} strokeLinecap="round" />
            ))}
            <circle cx={pt.x} cy={pt.y} r={0.3} fill="#f4ecd8" stroke="#7a5a36" strokeWidth={0.04} />
            {res >= 0 ? (
              <>
                <g transform={`translate(${pt.x} ${pt.y - 0.08})`}><ResourceGlyph r={res} size={0.9} /></g>
                <text x={pt.x} y={pt.y + 0.19} className="port-text">2:1</text>
                <title>{`Harbor: 2 ${RES_LABEL[res]} for 1 of anything`}</title>
              </>
            ) : (
              <>
                <text x={pt.x} y={pt.y + 0.07} className="port-text big">3:1</text>
                <title>Harbor: 3 of one kind for 1 of anything</title>
              </>
            )}
          </g>
        );
      })}

      {/* tiles */}
      {topo.hexes.map((h) => {
        const terr = layout.hex_terrain[h.id];
        const num = layout.hex_number[h.id];
        const t = targets?.hexes?.get(h.id);
        const hot = rolled !== null && num === rolled && view.robber !== h.id;
        return (
          <g key={h.id} transform={`translate(${h.x} ${h.y})`}>
            <use href={TERRAIN_SYMBOL[terr]} />
            <polygon points={hexPoints(0.97)} fill="none" stroke="#3b2f1e" strokeOpacity={0.35} strokeWidth={0.03} />
            {hot && <polygon points={hexPoints(0.9)} className="hex-hot" />}
            {num && (
              <g className={"token" + (num === 6 || num === 8 ? " hot" : "")}>
                <circle r={0.31} cy={0.015} fill="#000" opacity={0.25} />
                <circle r={0.3} fill="#f6efdc" stroke="#b49a6a" strokeWidth={0.025} />
                <text y={0.06} className="token-num">{num}</text>
                <g>
                  {Array.from({ length: PIPS[num] }, (_, i) => (
                    <circle key={i} cx={(i - (PIPS[num] - 1) / 2) * 0.065} cy={0.19} r={0.022} />
                  ))}
                </g>
              </g>
            )}
            {t && (
              <polygon points={hexPoints(0.92)} className={"hex-target" + (hover === "h" + h.id ? " on" : "")}
                onMouseEnter={() => setHover("h" + h.id)} onMouseLeave={() => setHover(null)}
                onClick={() => pick(t)} {...keyTarget("h" + h.id, t)} data-hex={h.id} role="button" aria-label={`Move robber to ${tile(h.id)}`}>
              </polygon>
            )}
            {view.robber === h.id && <Robber x={num ? -0.5 : 0} y={num ? 0.05 : 0} />}
            {t && hover === "h" + h.id && targets?.ghost === "robber" && <Robber x={num ? -0.5 : 0} y={num ? 0.05 : 0} ghost />}
          </g>
        );
      })}

      {/* roads */}
      {view.edge_owner.map((p, e) => {
        if (p < 0) return null;
        const [a, b] = topo.edges[e].v.map(vpos);
        const sx = a.x + (b.x - a.x) * 0.16, sy = a.y + (b.y - a.y) * 0.16, ex = b.x - (b.x - a.x) * 0.16, ey = b.y - (b.y - a.y) * 0.16;
        return (
          <g key={e} className="piece road" data-edge-owner={p}>
            <line x1={sx} y1={sy} x2={ex} y2={ey} stroke={PLAYER_INK[p]} strokeWidth={0.17} strokeLinecap="round" />
            <line x1={sx} y1={sy} x2={ex} y2={ey} stroke={PLAYER_COLORS[p]} strokeWidth={0.1} strokeLinecap="round" />
          </g>
        );
      })}
      {targets?.edges && [...targets.edges.entries()].map(([e, a]) => {
        const [va, vb] = topo.edges[e].v.map(vpos);
        const on = hover === "e" + e;
        return (
          <g key={"te" + e} onMouseEnter={() => setHover("e" + e)} onMouseLeave={() => setHover(null)} onClick={() => pick(a)} {...keyTarget("e" + e, a)}
            className={"edge-target" + (on ? " on" : "")} data-edge={e} role="button"
            aria-label={`Road along ${touching(va.hexes.filter((h) => vb.hexes.includes(h)))}`}>
            <polygon points={edgeHit(va, vb)} fill="transparent" />
            {on && targets.seat !== undefined ? (
              <line x1={va.x + (vb.x - va.x) * 0.16} y1={va.y + (vb.y - va.y) * 0.16} x2={vb.x - (vb.x - va.x) * 0.16} y2={vb.y - (vb.y - va.y) * 0.16}
                stroke={PLAYER_COLORS[targets.seat]} strokeWidth={0.12} strokeLinecap="round" opacity={0.85} />
            ) : (
              // DS28-01: a dark casing under the cream dashes, like .vertex-hint's stroke; on the sand coast they vanished
              [<line key="c" className="edge-hint-casing" x1={va.x + (vb.x - va.x) * 0.22} y1={va.y + (vb.y - va.y) * 0.22} x2={vb.x - (vb.x - va.x) * 0.22} y2={vb.y - (vb.y - va.y) * 0.22} />,
               <line key="h" className="edge-hint" x1={va.x + (vb.x - va.x) * 0.22} y1={va.y + (vb.y - va.y) * 0.22} x2={vb.x - (vb.x - va.x) * 0.22} y2={vb.y - (vb.y - va.y) * 0.22} />]
            )}
          </g>
        );
      })}

      {/* buildings */}
      {view.vertex_owner.map((p, v) => {
        if (p < 0) return null;
        const { x, y } = vpos(v);
        const isTarget = targets?.vertices?.has(v);
        if (isTarget && hover === "v" + v && targets?.ghost === "city") return <City key={v} x={x} y={y} p={p} ghost />;
        return view.vertex_level[v] === 2 ? <City key={v} x={x} y={y} p={p} /> : <Settlement key={v} x={x} y={y} p={p} />;
      })}
      {targets?.vertices && [...targets.vertices.entries()].map(([v, a]) => {
        const { x, y } = vpos(v);
        const on = hover === "v" + v;
        const city = targets.ghost === "city";
        return (
          <g key={"tv" + v} onMouseEnter={() => setHover("v" + v)} onMouseLeave={() => setHover(null)} onClick={() => pick(a)} {...keyTarget("v" + v, a)}
            className={"vertex-target" + (on ? " on" : "")} data-vertex={v} role="button" aria-label={`${city ? "City" : "Settlement"} on ${touching(vpos(v).hexes)}`}>
            <circle cx={x} cy={y} r={0.26} fill="transparent" />
            {!city && (on && targets.seat !== undefined ? <Settlement x={x} y={y} p={targets.seat} ghost /> : <circle className="vertex-hint" cx={x} cy={y} r={0.11} />)}
            {city && !on && <circle className="vertex-hint" cx={x} cy={y} r={0.25} fill="none" />}
          </g>
        );
      })}
      {children}
    </svg>
  );
}
