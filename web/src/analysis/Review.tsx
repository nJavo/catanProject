// Post-game review: eval timeline over the viewer's decisions, human-readable
// move list with mistakes marked, and played-vs-best for the selected move.
import { useEffect, useLayoutEffect, useRef, useState } from "react";
import type { Action, Depth, GameAnalysis, TimelineEntry } from "../api";
import { describeAction, PLAYER_COLORS, PLAYER_NAMES, Where } from "../rules";
import { CrestBadge } from "../Setup";
import { analysis } from "./client";
import { evalNum } from "./spec";
import { pct, VerdictChip } from "./AnalysisPanel";
import { resText } from "../lab/DecisionInspector";

const BAD = new Set(["inaccuracy", "mistake", "blunder"]);
const METRIC_LABEL: Record<string, string> = { rollout_win_rate: "Rollout win rate", value_estimate: "Value estimate", search_score: "Bot score" };
const fmtEval = (x: unknown, metric?: string) => { const v = evalNum(x, metric); return metric === "search_score" ? (v === null ? "—" : v.toFixed(2)) : pct(v); };

export function EvalGraph({ timeline, total, index, onJump }: { timeline: TimelineEntry[]; total: number; index: number; onJump: (i: number) => void }) {
  const ref = useRef<HTMLDivElement>(null);
  const [w, setW] = useState(800);
  const [hov, setHov] = useState<TimelineEntry | null>(null);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setW(el.clientWidth));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  const H = 96, pad = 6, X = (i: number) => pad + (i / Math.max(1, total)) * (w - 2 * pad), Y = (v: number) => pad + (1 - v) * (H - 2 * pad);
  const metric = timeline.find((t) => t.metric)?.metric;
  const raw = timeline.map((t) => ({ t, v: evalNum(t.eval_after, t.metric) })).filter((p): p is { t: TimelineEntry; v: number } => p.v !== null);
  // Bot scores are on their own scale: plot them min..max, never as a win rate.
  const lo = Math.min(...raw.map((p) => p.v)), hi = Math.max(...raw.map((p) => p.v));
  const scaled = metric === "search_score";
  const pts = scaled ? raw.map((p) => ({ ...p, v: hi > lo ? (p.v - lo) / (hi - lo) : 0.5 })) : raw;
  const path = pts.map((p, i) => `${i ? "L" : "M"}${X(p.t.index + 1).toFixed(1)},${Y(p.v).toFixed(1)}`).join(" ");
  const area = pts.length ? `${path} L${X(pts[pts.length - 1].t.index + 1).toFixed(1)},${Y(0)} L${X(pts[0].t.index + 1).toFixed(1)},${Y(0)} Z` : "";
  const pick = (e: React.MouseEvent) => {
    const r = (e.currentTarget as SVGElement).getBoundingClientRect();
    const i = Math.round(((e.clientX - r.left - pad) / (w - 2 * pad)) * total);
    const near = timeline.reduce<TimelineEntry | null>((b, t) => (!b || Math.abs(t.index - i) < Math.abs(b.index - i) ? t : b), null);
    return near;
  };
  return (
    <div className="an-graph" ref={ref}>
      <svg width={w} height={H} role="img" aria-label="Evaluation over the game" data-testid="eval-graph"
        onMouseMove={(e) => setHov(pick(e))} onMouseLeave={() => setHov(null)} onClick={(e) => { const t = pick(e); if (t) onJump(t.index); }}>
        <rect x={0} y={0} width={w} height={H} className="g-bg" />
        <line x1={0} x2={w} y1={Y(0.5)} y2={Y(0.5)} className="g-mid" />
        {!scaled && <text x={4} y={Y(0.5) - 3} className="g-tick">50%</text>}
        {area && <path d={area} className="g-area" />}
        {path && <path d={path} className="g-line" />}
        <line x1={X(index)} x2={X(index)} y1={0} y2={H} className="g-now" />
        {pts.filter((p) => BAD.has(p.t.verdict)).map((p) => (
          <circle key={p.t.index} cx={X(p.t.index + 1)} cy={Y(p.v)} r={p.t.verdict === "blunder" ? 5 : 4} className={`g-mark v-${p.t.verdict}`} />
        ))}
        {hov && <line x1={X(hov.index + 1)} x2={X(hov.index + 1)} y1={0} y2={H} className="g-hov" />}
      </svg>
      <div className="g-legend small">
        <span>{METRIC_LABEL[metric ?? ""] ?? "Win estimate"} of each played move{scaled ? " (scaled min to max)" : " (0 to 100%)"}
          {scaled && <span className="g-note" data-testid="eval-caption-quick">Quick depth runs no rollouts, and each phase scores moves on its own scale, so the line is mostly flat with spikes. Standard depth plots rollout win rates.</span>}</span>
        {hov && <span className="g-tip">Move {hov.index + 1}: {resText(hov.played_label)} · {fmtEval(hov.eval_after, hov.metric)} <VerdictChip v={hov.verdict} /></span>}
      </div>
    </div>
  );
}

export default function Review({ kind, refId, actions, index, onJump, viewer, setViewer, numPlayers, policy, game, setGame, where }: {
  kind: "game" | "replay"; refId: string; actions: Action[]; index: number; onJump: (i: number) => void;
  viewer: number; setViewer: (v: number) => void; numPlayers: number; policy: string;
  game: GameAnalysis | null; setGame: (g: GameAnalysis | null) => void; where?: Where;
}) {
  const [busy, setBusy] = useState<AbortController | null>(null);
  const [depth, setDepth] = useState<Depth>("quick"); // a whole game: quick by default (the server's default too)
  const [err, setErr] = useState<string | null>(null);
  const listRef = useRef<HTMLOListElement>(null);
  const byIndex = new Map((game?.timeline ?? []).map((t) => [t.index, t]));
  const cur = byIndex.get(index);

  useEffect(() => { listRef.current?.querySelector(".on")?.scrollIntoView({ block: "nearest" }); }, [index]);
  useEffect(() => () => busy?.abort(), [busy]);

  const run = async () => {
    const ctl = new AbortController();
    setBusy(ctl); setErr(null);
    try {
      setGame(await analysis.game({ source: { kind, ref: refId }, viewer, policy, depth }, ctl.signal));
    } catch (e: any) {
      if (e?.name !== "AbortError") setErr(e.message);
    } finally {
      setBusy(null);
    }
  };

  return (
    <section className="an-card an-review" aria-label="Game review">
      <div className="an-row">
        <h3>Game review</h3>
        <select value={viewer} onChange={(e) => { setViewer(Number(e.target.value)); setGame(null); }} aria-label="Reviewed seat">
          {Array.from({ length: numPlayers }, (_, p) => <option key={p} value={p}>{PLAYER_NAMES[p]}'s moves</option>)}
        </select>
        {busy ? <button className="btn small" onClick={() => busy.abort()}>Cancel</button>
          : <button className="btn small primary" onClick={run} data-testid="analyze-game">{game ? "Re-run" : "Analyze game"}</button>}
        <span className="segmented small" role="radiogroup" aria-label="Review depth">
          {(["quick", "standard"] as Depth[]).map((d) => <button key={d} className={depth === d ? "on" : ""} onClick={() => setDepth(d)} data-review-depth={d}>{d === "quick" ? "Quick" : "Standard"}</button>)}
        </span>
      </div>
      {busy && <p className="muted small" aria-live="polite">Evaluating every {PLAYER_NAMES[viewer]} decision with {policy} ({depth})…</p>}
      {err && <p className="error small">Game analysis failed: {err}</p>}
      {game?.mock && <p className="an-mock-note">MOCK TIMELINE: invented numbers from the dev mock.</p>}
      {game && (
        <p className="small an-summary" data-testid="review-summary">
          {typeof game.summary.accuracy === "number" && <><span title={(game.summary as any).accuracy_definition}>Accuracy</span> <strong>{pct(game.summary.accuracy, 0)}</strong>
            <span className="muted"> (played move was {game.policy ?? policy}'s top pick)</span> · </>}
          {game.depth === "quick"
            ? <span className="muted">Quick depth runs no rollouts, so no move is rated a mistake. Re-run at Standard for verdicts.</span>
            : <><span className="v-mistake-t">{game.summary.mistakes} mistakes</span> · <span className="v-blunder-t">{game.summary.blunders} blunders</span>
              <span className="muted"> · verdicts fire only when the gap exceeds the rollout CI</span></>}
        </p>
      )}
      {cur && (
        <div className="an-pvb" data-testid="played-vs-best">
          <div><span className="field-label">Played</span><strong>{resText(cur.played_label)}</strong> <VerdictChip v={cur.verdict} /></div>
          <div><span className="field-label">Best</span><strong>{cur.best_label ? resText(cur.best_label) : "—"}</strong></div>
          <div className="muted small">{METRIC_LABEL[cur.metric ?? ""] ?? "Win estimate"}: best {fmtEval(cur.eval_before, cur.metric)}, played {fmtEval(cur.eval_after, cur.metric)}
            {cur.delta !== null && ` (${cur.delta >= 0 ? "+" : ""}${cur.metric === "search_score" ? cur.delta.toFixed(2) : `${(cur.delta * 100).toFixed(1)} pts`})`}</div>
        </div>
      )}
      <ol className="an-moves" ref={listRef} data-testid="move-list">
        <li className={index === 0 ? "on" : ""} onClick={() => onJump(0)}>Start</li>
        {actions.map((a, i) => {
          const t = byIndex.get(i);
          return (
            <li key={i} className={(index === i ? "on" : "") + (t && BAD.has(t.verdict) ? ` v-${t.verdict}` : "")} onClick={() => onJump(i)} style={{ ["--pc" as any]: PLAYER_COLORS[a.player] }}
              title={`Position before move ${i + 1}`} data-move={i}>
              <span className="mv-i">{i + 1}</span>
              <CrestBadge p={a.player} size={12} /> {describeAction(a, where)}
              {t && BAD.has(t.verdict) && <span className={`an-glyph v-${t.verdict}`}>{t.verdict === "blunder" ? "??" : t.verdict === "mistake" ? "?" : "?!"}</span>}
            </li>
          );
        })}
        <li className={index === actions.length ? "on" : ""} onClick={() => onJump(actions.length)}>Final position</li>
      </ol>
      <p className="muted small">← → step through moves. Click a move to see the position before it.</p>
    </section>
  );
}
