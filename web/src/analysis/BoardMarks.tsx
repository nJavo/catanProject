// Highlights drawn inside <Board> (board coordinates): candidate targets,
// played/best moves and validation errors. Never intercepts clicks.
import type { Topology } from "../api";
import { hexPoints } from "../board/Terrain";

export type Tone = "cand" | "hot" | "best" | "played" | "error" | "low";
export interface Mark { kind: "v" | "e" | "h"; id: number; tone: Tone; rank?: number }

export default function BoardMarks({ topo, marks }: { topo: Topology; marks: Mark[] }) {
  return (
    <g className="an-marks" pointerEvents="none">
      {marks.map((m, i) => {
        const key = `${m.kind}${m.id}-${m.tone}-${i}`;
        if (m.kind === "h") {
          const h = topo.hexes[m.id];
          if (!h) return null;
          return (
            <g key={key} transform={`translate(${h.x} ${h.y})`} className={`mk mk-${m.tone}`}>
              <polygon points={hexPoints(0.86)} className="mk-shape" />
              {m.rank && <Badge x={0.45} y={-0.55} n={m.rank} />}
            </g>
          );
        }
        if (m.kind === "v") {
          const v = topo.vertices[m.id];
          if (!v) return null;
          return (
            <g key={key} className={`mk mk-${m.tone}`}>
              <circle cx={v.x} cy={v.y} r={0.3} className="mk-shape" />
              {m.rank && <Badge x={v.x + 0.26} y={v.y - 0.26} n={m.rank} />}
            </g>
          );
        }
        const e = topo.edges[m.id];
        if (!e) return null;
        const [a, b] = e.v.map((x) => topo.vertices[x]);
        return (
          <g key={key} className={`mk mk-${m.tone}`}>
            <line x1={a.x + (b.x - a.x) * 0.12} y1={a.y + (b.y - a.y) * 0.12} x2={b.x - (b.x - a.x) * 0.12} y2={b.y - (b.y - a.y) * 0.12} className="mk-shape" />
            {m.rank && <Badge x={(a.x + b.x) / 2 + 0.18} y={(a.y + b.y) / 2 - 0.18} n={m.rank} />}
          </g>
        );
      })}
    </g>
  );
}

function Badge({ x, y, n }: { x: number; y: number; n: number }) {
  return (
    <g className="mk-badge" transform={`translate(${x} ${y})`}>
      <circle r={0.14} />
      <text y={0.05}>{n}</text>
    </g>
  );
}
