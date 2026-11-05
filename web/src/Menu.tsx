import { useEffect, useMemo, useState } from "react";
import { api, SaveInfo, Topology, View } from "./api";
import Board from "./board/Board";
import { go } from "./App";

// A fixed decorative island for the title screen (not a real game).
function demoView(topo: Topology): View {
  const terrain = ["ore", "sheep", "wood", "wheat", "brick", "sheep", "brick", "wheat", "wood", "desert", "wood", "ore", "wood", "ore", "wheat", "sheep", "brick", "wheat", "sheep"];
  const numbers = [10, 2, 9, 12, 6, 4, 10, 9, 11, null, 3, 8, 8, 3, 4, 5, 5, 6, 11];
  const vo = Array(topo.vertices.length).fill(-1), vl = Array(topo.vertices.length).fill(0), eo = Array(topo.edges.length).fill(-1);
  const place = (v: number, p: number, lvl = 1) => { vo[v] = p; vl[v] = lvl; };
  place(12, 0); place(20, 1, 2); place(31, 2); place(40, 3); place(24, 0, 2); place(43, 1);
  const road = (v: number, p: number) => { const e = topo.edges.find((x) => x.v.includes(v) && eo[x.id] < 0); if (e) eo[e.id] = p; };
  road(12, 0); road(20, 1); road(31, 2); road(40, 3); road(24, 0); road(43, 1);
  return {
    layout: { hex_terrain: terrain, hex_number: numbers, port_edges: [], port_types: [] },
    vertex_owner: vo, vertex_level: vl, edge_owner: eo, robber: 9, rolled: false, last_roll: null,
  } as unknown as View;
}

function SavedGames() {
  const [saves, setSaves] = useState<SaveInfo[]>([]);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => { api.saves().then((r) => setSaves(r.saves)).catch(() => {}); }, []);
  if (!saves.length) return null;
  const resume = (name: string) => api.resume(name).then((r) => go(`/game/${r.id}`)).catch((e) => setErr(e.message));
  return (
    <section className="saves" aria-label="Saved games" data-testid="saves">
      <h2>Resume a saved game</h2>
      <ul>
        {saves.map((s) => (
          <li key={s.name}>
            {/* auto names ("20261104-122453-a4f8…") told several saves apart only by a hash; show when it was saved */}
            <span title={s.name}>{/^\d{8}-\d{6}-[0-9a-f]+$/.test(s.name) && s.saved ? new Date(s.saved * 1000).toLocaleString("sv-SE", { dateStyle: "short", timeStyle: "short" }) : s.name}</span>
            {/* outside the truncating name span, so long auto-generated names never hide it */}
            <small className="muted">turn {(s.turn ?? 0) + 1} · {s.seats?.filter((x) => x.type === "human").length ?? 0} human, {s.seats?.filter((x) => x.type === "agent").length ?? 0} AI</small>
            <button className="btn small" onClick={() => resume(s.name)} data-resume={s.name} aria-label={`Resume ${s.name}`}>Resume</button>
          </li>
        ))}
      </ul>
      {err && <p className="error">{err}</p>}
    </section>
  );
}

export default function Menu({ topo }: { topo: Topology }) {
  const view = useMemo(() => demoView(topo), [topo]);
  return (
    <div className="menu">
      <div className="menu-board" aria-hidden><Board topo={topo} view={view} /></div>
      <main className="menu-panel">
        <h1 className="title">Catan<span> Lab</span></h1>
        <p className="tagline">Settle a hand-drawn island against friends or a table of AI rivals, then study how the machines play.</p>
        <nav className="menu-actions">
          <button className="btn primary big" onClick={() => go("/new")}>New game</button>
          <button className="btn big" onClick={() => go("/analysis")} data-testid="menu-analysis">Analysis board</button>
          <button className="btn big" onClick={() => go("/lab")}>AI laboratory</button>
          <button className="btn big ghost" onClick={() => go("/lab/replays")}>Replays</button>
        </nav>
        <SavedGames />
        <p className="fineprint">Base game rules, 5th edition. Fan-made, not affiliated with the publisher.</p>
      </main>
    </div>
  );
}
