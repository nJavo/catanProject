import { useEffect, useState } from "react";
import { api, setRoomToken, Topology } from "./api";
import Menu from "./Menu";
import Setup from "./Setup";
import GameScreen from "./game/GameScreen";
import Lab from "./lab/Lab";
import MatchViewer from "./lab/MatchViewer";
import ReplayViewer from "./lab/ReplayViewer";
import AnalysisPage from "./analysis/AnalysisPage";
import Room from "./Room";
import { setChampion } from "./rules";

// Hash routes keep the static server trivial: #/, #/new, #/game/:id,
// #/lab[/:tab], #/lab/match/:id, #/replay/:source/:ref, #/analysis[/:source/:ref[/:index]], #/analysis/import/:payload,
// #/room/:id/:token[/play] (online rooms, docs/ONLINE.md)
export function go(path: string) {
  window.location.hash = path;
}

function useRoute() {
  const [h, setH] = useState(window.location.hash.slice(1) || "/");
  useEffect(() => {
    const on = () => setH(window.location.hash.slice(1) || "/");
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  return h.split("/").filter(Boolean).map(decodeURIComponent);
}

export default function App() {
  const parts = useRoute();
  const [topo, setTopo] = useState<Topology | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const page = ({ new: "New game", game: "Game", lab: parts[1] === "match" ? "AI match" : "AI laboratory", replay: "Replay", analysis: "Analysis", room: "Online room" } as Record<string, string>)[parts[0]];
  // every route had the same tab title, so screen readers and the tab bar couldn't tell pages apart
  useEffect(() => { document.title = page ? `${page} · Hexhaven` : "Hexhaven: settle the island"; }, [page]);
  useEffect(() => {
    // The champion's model name is only a label: a failed registry fetch must not block the app.
    const champ = fetch("/api/registry").then((r) => r.json()).then((r) => setChampion(r.champion)).catch(() => {});
    Promise.all([api.topology(), champ]).then(([t]) => setTopo(t)).catch((e) => setErr(`Can't reach the game server: ${e.message}. Start it with uv run python -m catan.server.`));
  }, []);
  if (err) return <div className="fatal"><h1>Server unavailable</h1><p>{err}</p></div>;
  if (!topo) return <div className="fatal"><p>Laying out the island…</p></div>;
  const [a, b, c] = parts;
  setRoomToken(a === "room" && c ? c : null); // before children render, so their first fetch carries it
  if (a === "room" && b && c) return parts[3] === "play" ? <GameScreen key={b} id={b} topo={topo} lobby={`/room/${b}/${c}`} /> : <Room key={b + c} id={b} token={c} />;
  if (a === "new") return <Setup />;
  if (a === "game" && b) return <GameScreen key={b} id={b} topo={topo} />;
  if (a === "lab" && b === "match" && c) return <MatchViewer key={c} id={c} topo={topo} />;
  if (a === "replay" && (b === "game" || b === "file") && c) return <ReplayViewer key={b + c} source={b} refId={c} topo={topo} />;
  if (a === "lab") return <Lab tab={b} />;
  if (a === "analysis") return <AnalysisPage key={parts.slice(1, 3).join("/")} topo={topo} route={parts.slice(1)} />;
  return <Menu topo={topo} />;
}
