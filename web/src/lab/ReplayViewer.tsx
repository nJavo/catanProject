import { useEffect, useRef, useState } from "react";
import { api, Action, Topology, TurnSegment, View } from "../api";
import { describeAction, PLAYER_COLORS, PLAYER_NAMES } from "../rules";
import { LAB_SPEEDS, SpectatorTable } from "./MatchViewer";
import { go } from "../App";

export default function ReplayViewer({ source, refId, topo }: { source: "game" | "file"; refId: string; topo: Topology }) {
  const [actions, setActions] = useState<Action[] | null>(null);
  const [index, setIndex] = useState(0);
  const [view, setView] = useState<View | null>(null);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [err, setErr] = useState<string | null>(null);
  const [since, setSince] = useState<number | undefined>(undefined);   // set after a turn jump: summarize [since, index)
  const listRef = useRef<HTMLOListElement>(null);

  useEffect(() => {
    (source === "game" ? api.replay(refId) : api.replayFile(refId))
      .then((r) => { setActions(r.actions); setIndex(r.actions.length); })
      .catch((e) => setErr(e.message));
  }, [source, refId]);

  useEffect(() => {
    if (!actions) return;
    let live = true;
    api.replayState(source, refId, index, since).then((v) => live && setView(v)).catch((e) => setErr(e.message));
    return () => { live = false; };
  }, [actions, index, since, source, refId]);

  const segs = view?.turns ?? [];
  const n = actions?.length ?? 0;
  const ti = Math.max(0, segs.reduce((k, s, i) => (s.start <= index ? i : k), -1));
  const move = (k: number) => { setSince(undefined); setIndex(Math.max(0, Math.min(n, k))); };
  // Jump to a turn start; the hand inspector then summarizes the turn that just ended there.
  const toTurn = (k: number) => { setPlaying(false); setIndex(k); setSince(segs.find((s) => s.end === k && s.start < k)?.start); };
  const nextTurn = () => toTurn(segs[ti + 1]?.start ?? n);
  const prevTurn = () => toTurn(segs.length && index > segs[ti].start ? segs[ti].start : segs[ti - 1]?.start ?? 0);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement;
      if ((t instanceof HTMLInputElement && t.type !== "range") || t instanceof HTMLTextAreaElement || t instanceof HTMLSelectElement || t.isContentEditable) return;
      const edge = e.key === "Home" || e.key === "End";
      if (e.key !== "ArrowLeft" && e.key !== "ArrowRight" && !edge) return;
      if (edge && t instanceof HTMLInputElement) return; // a focused slider keeps its own Home/End
      e.preventDefault();
      setPlaying(false);
      if (edge) move(e.key === "Home" ? 0 : n);
      else if (e.shiftKey) (e.key === "ArrowRight" ? nextTurn : prevTurn)();
      else move(index + (e.key === "ArrowRight" ? 1 : -1));
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  useEffect(() => {
    if (!playing || !actions) return;
    if (index >= actions.length) { setPlaying(false); return; }
    const t = setTimeout(() => move(index + 1), Math.max(LAB_SPEEDS[speed].ms, 60));
    return () => clearTimeout(t);
  }, [playing, index, speed, actions]);

  useEffect(() => {
    listRef.current?.querySelector(".on")?.scrollIntoView({ block: "nearest" });
  }, [index, !!view]); // !!view: the list mounts after the first state loads

  if (err) return <div className="fatal"><h1>Replay unavailable</h1><p>{err}</p><button className="btn" onClick={() => go("/lab/replays")}>Back to replays</button></div>;
  if (!actions || !view) return <div className="fatal"><p>Re-simulating replay…</p></div>;
  const seg: TurnSegment | undefined = segs[ti];
  const summary = since !== undefined && since < index ? segs.find((s) => s.start === since) : undefined;
  return (
    <SpectatorTable topo={topo} view={view} prev={view.prev_players} moveEvents={view.span_events ?? view.last_events}
      span={summary ? index - summary.start : 1} spanTitle={summary && `Summary of ${turnLabel(summary)}`} controls={
      <>
        <button className="chip" onClick={() => move(0)} aria-label="First move" title="First move (Home)">⇤</button>
        <button className="chip" onClick={prevTurn} aria-label="Previous turn" title="Previous turn (Shift+←)" data-testid="prev-turn">⏮</button>
        <button className="chip" onClick={() => move(index - 1)} aria-label="Previous move" title="Previous move (←)">◀</button>
        <button className="chip" onClick={() => { if (index >= n) move(0); setPlaying((p) => !p); }}>{playing ? "Pause" : "Play"}</button>
        <button className="chip" onClick={() => move(index + 1)} aria-label="Next move" title="Next move (→)">▶</button>
        <button className="chip" onClick={nextTurn} aria-label="Next turn" title="Next turn (Shift+→)" data-testid="next-turn">⏭</button>
        <button className="chip" onClick={() => move(n)} aria-label="Last move" title="Last move (End)">⇥</button>
        <button className="chip on" onClick={() => go(`/analysis/${source}/${encodeURIComponent(refId)}/${Math.min(index, n)}`)} data-testid="analyze-replay">Analyze game</button>
        {LAB_SPEEDS.slice(0, 3).map((s, i) => <button key={s.label} className={"chip" + (speed === i ? " on" : "")} onClick={() => setSpeed(i)}>{s.label}</button>)}
      </>
    } side={
      <section className="movelist">
        <label className="scrub">
          <span className="field-label">Move {index} of {n}</span>
          <input type="range" min={0} max={n} value={index} onChange={(e) => { setPlaying(false); move(Number(e.target.value)); }} aria-label="Scrub replay" />
        </label>
        {seg && <>
          <label className="scrub turn-scrub">
            <span className="field-label" data-testid="turn-label"><strong>{turnLabel(seg)}</strong>{" "}
              {/* "· 90 of 90" counted the 8 setup placements as turns, next to "Turn 82" */}
              <span className="muted">(of {segs.filter((x) => x.setup === seg.setup).length})</span></span>
            <input type="range" min={0} max={segs.length - 1} value={ti} onChange={(e) => toTurn(segs[Number(e.target.value)].start)} aria-label="Scrub turns" />
          </label>
          <label className="scrub turn-scrub">
            <span className="field-label">Move {index - seg.start} of {seg.end - seg.start} in this turn</span>
            <input type="range" min={seg.start} max={seg.end} value={index} onChange={(e) => { setPlaying(false); move(Number(e.target.value)); }} aria-label="Scrub moves in turn" />
          </label>
        </>}
        <ol ref={listRef}>
          <li className={index === 0 ? "on" : ""} onClick={() => move(0)}>Start</li>
          {actions.map((a, i) => (
            <li key={i} className={index === i + 1 ? "on" : ""} onClick={() => move(i + 1)} style={{ ["--pc" as any]: PLAYER_COLORS[a.player] }}>
              <span className="mv-i">{i + 1}</span>{describeAction(a, { topo, layout: view.layout })}
            </li>
          ))}
        </ol>
      </section>
    } />
  );
}

export const turnLabel = (s: TurnSegment) => `${s.setup ? "Setup" : "Turn"} ${s.turn} · ${PLAYER_NAMES[s.player] ?? "?"}`;
