import { useCallback, useEffect, useState } from "react";
import { Action, api, GameEvent, Topology, View } from "../api";
import Board, { Targets } from "../board/Board";
import { go } from "../App";
import { PHASE_LABEL, PLAYER_NAMES, seatName, turnNumber } from "../rules";
import { CrestBadge } from "../Setup";
import { Bank, Dice, Log, OpponentPanel } from "../game/Hud";
import DecisionInspector, { Highlight } from "./DecisionInspector";
import { DevChips, HandDetail, PrevHand } from "./HandInspector";

export const LAB_SPEEDS = [{ label: "0.5×", ms: 900 }, { label: "1×", ms: 400 }, { label: "4×", ms: 100 }, { label: "Max", ms: 0 }];

/** Board highlight for an inspector candidate (read-only: the viewer passes no onPick). */
function highlight(h: Highlight): Targets | undefined {
  if (h && "refs" in h) {  // provenance item: every vertex/edge/hex it mentions
    const m = (ids?: number[]) => ids?.length ? new Map(ids.map((i) => [i, { player: 0, type: "HIGHLIGHT", value: i } as Action])) : undefined;
    return { vertices: m(h.refs.vertices), edges: m(h.refs.edges), hexes: m(h.refs.hexes) };
  }
  const a = h;
  if (!a || typeof a.value !== "number") return undefined;
  const one = new Map([[a.value, a]]);
  switch (a.type) {
    case "BUILD_SETTLEMENT": return { vertices: one, ghost: "settlement", seat: a.player };
    case "BUILD_CITY": return { vertices: one, ghost: "city", seat: a.player };
    case "BUILD_ROAD": return { edges: one, ghost: "road", seat: a.player };
    case "MOVE_ROBBER": return { hexes: one, ghost: "robber", seat: a.player };
  }
  return undefined;
}

/** Read-only table layout shared by the live match viewer and replays. */
export function SpectatorTable({ topo, view: full, controls, side, prev, moveEvents = [], span = 1, spanTitle }: {
  topo: Topology; view: View; controls: React.ReactNode; side?: React.ReactNode;
  prev?: PrevHand[] | null; moveEvents?: GameEvent[]; span?: number; spanTitle?: string;
}) {
  const [showAll, setShowAll] = useState(true);
  const [pinned, setPinned] = useState<number | null>(null);
  const [hovered, setHovered] = useState<Highlight>(null);
  useEffect(() => setHovered(null), [full.action_count]);
  // "Show all hands" off: drop the revealed fields so panels fall back to public info.
  const view = showAll ? full
    : { ...full, players: full.players.map(({ hand, dev_cards, dev_new, vp, ...pub }) => pub) };
  const rollSeq = [...view.events].reverse().find((e) => e.type === "dice_rolled")?.seq ?? view.action_count;
  const who = view.actor ?? view.winner;
  return (
    <div className="game spectator">
      <header className="topbar">
        <button className="btn ghost small" onClick={() => go("/lab")}>Lab</button>
        <div className="banner">
          {who !== null && who >= 0 && <CrestBadge p={who} size={30} />}
          <div className="banner-text">
            <span className="banner-phase">{PHASE_LABEL[view.phase]} · Turn {turnNumber(view.turn, view.phase === "GAME_OVER", (view as any).truncated)} · Action {view.action_count}</span>
            <strong className="banner-instr" data-testid="instruction">
              {view.phase === "GAME_OVER"
                ? view.winner >= 0 ? `${seatName(view.winner, view.seats)} wins` : "Stopped at the turn limit"
                : `${seatName(view.actor!, view.seats)} to act`}
            </strong>
          </div>
        </div>
        <Dice roll={view.last_roll} rollSeq={rollSeq} />
        <div className="lab-controls">{controls}</div>
      </header>
      <aside className="left">
        <div className="hi-bar">
          <span className="field-label">Players</span>
          <button className={"chip" + (showAll ? " on" : "")} onClick={() => setShowAll((s) => !s)} aria-pressed={showAll} data-testid="show-hands">Show all hands</button>
        </div>
        {view.players.map((p) => (
          <div key={p.id} className={"hi-panel" + (pinned === p.id ? " pinned" : "")} role="button" tabIndex={0} aria-expanded={pinned === p.id}
            title="Click to inspect" onClick={() => setPinned((x) => (x === p.id ? null : p.id))}
            onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); setPinned((x) => (x === p.id ? null : p.id)); } }}>
            <OpponentPanel p={p} view={view} active={view.actor === p.id} seats={view.seats}>
              <DevChips p={p} />
              {pinned === p.id && <HandDetail p={p} view={view} topo={topo} prev={showAll ? prev?.[p.id] : undefined} events={moveEvents} span={span} title={spanTitle} />}
            </OpponentPanel>
          </div>
        ))}
        <Bank view={view} />
      </aside>
      <main className="board-wrap"><Board topo={topo} view={view} targets={highlight(hovered)} /></main>
      <aside className="right">
        {side}
        <DecisionInspector decision={view.last_decision} onHover={setHovered} where={{ topo, layout: view.layout }} />
        <Log events={view.events} seats={view.seats} layout={view.layout} />
      </aside>
    </div>
  );
}

export default function MatchViewer({ id, topo }: { id: string; topo: Topology }) {
  const [{ view, prev }, setPair] = useState<{ view: View | null; prev: View | null }>({ view: null, prev: null });
  const setView = (v: View) => setPair((o) => ({ view: v, prev: o.view }));
  // ?paused opens the match without auto-play (deterministic e2e, linking to a position)
  const [playing, setPlaying] = useState(() => !new URLSearchParams(window.location.search).has("paused"));
  const [speed, setSpeed] = useState(1);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => { api.view(id, "all").then(setView).catch((e) => setErr(e.message)); }, [id]);

  const step = useCallback((n = 1) => api.advance(id, "all", n).then((r) => { setJump(null); setView(r.view); }).catch((e) => { setErr(e.message); setPlaying(false); }), [id]);
  const [jump, setJump] = useState<string | null>(null);   // label of the turn just played by "Next turn"
  const [busy, setBusy] = useState(false);
  // Advance one action at a time until the turn (or setup placement pair) ends; same boundaries as turn_segments().
  const nextTurn = async () => {
    if (!view || busy || view.phase === "GAME_OVER") return;
    setPlaying(false); setBusy(true);
    const setup = view.phase.startsWith("SETUP");
    const label = `${setup ? "Setup" : "Turn"} ${setup ? view.players.reduce((k, p) => k + p.settlements.length, view.phase === "SETUP_ROAD" ? 0 : 1) : view.turn + 1} · ${PLAYER_NAMES[view.current]}`;
    try {
      let last: View | null = null;
      for (let k = 0; k < 500; k++) {
        const r = await api.advance(id, "all", 1);
        last = r.view;
        const ev = r.steps[0]?.events ?? [];
        if (!r.steps.length || r.view.phase === "GAME_OVER" || r.view.phase === "SETUP_SETTLEMENT" || ev.some((e) => e.type === "turn_started")) break;
      }
      if (last) { setJump(`Summary of ${label}`); setView(last); }
    } catch (e: any) { setErr(e.message); } finally { setBusy(false); }
  };

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement;
      if (e.key !== "ArrowRight" || t instanceof HTMLInputElement || t instanceof HTMLTextAreaElement || t instanceof HTMLSelectElement) return;
      e.preventDefault();
      if (e.shiftKey) nextTurn(); else if (!playing) step(1);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  useEffect(() => {
    if (!view || !playing || view.phase === "GAME_OVER") return;
    const ms = LAB_SPEEDS[speed].ms;
    const t = setTimeout(() => step(ms === 0 ? 25 : 1), ms);
    return () => clearTimeout(t);
  }, [view, playing, speed, step]);

  if (err && !view) return <div className="fatal"><h1>Match unavailable</h1><p>{err}. Matches live in server memory: a restart clears them, and old finished matches are dropped. Saved replays are kept.</p><button className="btn" onClick={() => go("/lab")}>AI laboratory</button></div>;
  if (!view) return <div className="fatal"><p>Loading match…</p></div>;
  const over = view.phase === "GAME_OVER";
  const since = prev && prev.action_count < view.action_count ? prev.action_count : null;
  return (
    <SpectatorTable topo={topo} view={view} prev={since === null ? null : prev!.players as PrevHand[]}
      moveEvents={since === null ? [] : view.events.filter((e) => e.i >= since)} span={since === null ? 1 : view.action_count - since} spanTitle={jump ?? undefined} controls={
      <>
        <button className="chip" onClick={() => setPlaying((p) => !p)} disabled={over} data-testid="play-pause">{playing && !over ? "Pause" : "Play"}</button>
        <button className="chip" onClick={() => go(`/replay/game/${id}`)} aria-label="Previous turn" title="Live matches only run forward: opens the replay to step back">⏮</button>
        <button className="chip" onClick={() => step(1)} disabled={over || playing || busy} title="Next move (→)">Step</button>
        <button className="chip" onClick={nextTurn} disabled={over || busy} aria-label="Next turn" title="Next turn (Shift+→)" data-testid="next-turn">⏭</button>
        {LAB_SPEEDS.map((s, i) => <button key={s.label} className={"chip" + (speed === i ? " on" : "")} onClick={() => setSpeed(i)} data-speed={s.label}>{s.label}</button>)}
        {over && <button className="chip on" onClick={() => go(`/replay/game/${id}`)} data-testid="open-replay">Open replay</button>}
      </>
    } side={over ? (
      <section className="result-card" data-testid="match-result">
        <h3>Final scores</h3>
        <ol>{[...view.players].sort((a, b) => (b.vp ?? 0) - (a.vp ?? 0)).map((p) => <li key={p.id}><CrestBadge p={p.id} size={18} /> {seatName(p.id, view.seats, true)} <strong>{p.vp}</strong></li>)}</ol>
      </section>
    ) : null} />
  );
}
