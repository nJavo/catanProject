// Right column: policy/depth settings, "Best move?", ranked candidates.
// Honesty: policy probability, value estimate, search score and rollout win
// rate are separate columns with their own labels; null is shown as "—".
import { useEffect, useState } from "react";
import type { AnalysisResult, AnalysisSource, BeliefSummary, Candidate, Depth, PositionSpec, Topology, Verdict } from "../api";
import { RateBar } from "../lab/Charts";
import { AgentSelect } from "../Setup";
import { championKind, championName, describeAction, PLAYER_NAMES, RES_LABEL } from "../rules";
import { EntityAttribution, Refs, resText } from "../lab/DecisionInspector";
import { analysis } from "./client";
import { policySpec } from "./spec";

// Search agents search on every move: far too slow to play rollouts (the server refuses them too).
const SLOW_ROLLOUT = new Set(["search", "search_trader", "jointb_gated"]);

export interface Settings { agent: string; checkpoint: string; rollout: string; depth: Depth; compare: boolean; agentB: string; checkpointB: string; belief?: boolean }
export const DEFAULT_SETTINGS: Settings = { agent: "intermediate", checkpoint: "", rollout: "", depth: "standard", compare: false, agentB: "search", checkpointB: "", belief: false };

export const DEPTHS: { id: Depth; label: string; target: string; ms: number }[] = [
  { id: "quick", label: "Quick", target: "< 0.3 s", ms: 300 },
  { id: "standard", label: "Standard", target: "< 5 s", ms: 5000 },
  { id: "deep", label: "Deep", target: "< 60 s", ms: 60000 },
];
const MEASURED = "catan.analysis.latency.v1";

export const pct = (x: number | null | undefined, d = 1) => (x === null || x === undefined ? "—" : `${(x * 100).toFixed(d)}%`);
// DS27-04: the server label says "Build settlement at vertex 35 (...)"; name the tiles like Review and the inspector do.
// The server's port note ("3:1 port") and "(free)" are kept; the robber keeps its "touches" note with the hex id dropped.
const BOARD = new Set(["BUILD_SETTLEMENT", "BUILD_CITY", "BUILD_ROAD"]);
function candName(c: Candidate, res: AnalysisResult, topo?: Topology): string {
  if (!topo || !BOARD.has(c.action.type)) return resText(c.label.replace(/ to hex \d+ \((.*)\)$/, " to $1"));
  const port = c.label.match(/, ([^,()]*port)\)/)?.[1];
  return resText(describeAction(c.action, { topo, layout: res.position.layout }) + (port ? `, ${port}` : "") + (c.label.endsWith("(free)") ? " (free)" : ""));
}
const secs = (ms: number) => (ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(1)} s`);

export function VerdictChip({ v, rolled }: { v: Verdict | string | null | undefined; rolled?: boolean }) {
  if (!v) return null;
  // DS18: with rollouts, "unrated" means worse on average but the CIs overlap, not "not analysed"
  if (v === "unrated" && rolled) return <span className={`verdict v-${v}`} title="Lower rollout win rate than the best move, but the 95% intervals overlap. More rollouts (Deep) can separate them.">unclear</span>;
  return <span className={`verdict v-${v}`}>{v}</span>;
}

export type Results = { a?: AnalysisResult; b?: AnalysisResult };

const AgentChoice = () => <span className="verdict" data-testid="agent-choice" title="The move the policy itself plays here, decided on your view of the position">agent's choice</span>;

/** Why the policy p / value est. columns are hidden, by policy kind (the server's policy_kind). */
function kindNote(res: AnalysisResult, topo?: Topology): string {
  const info = res.agent_info;
  const replaced = info?.replaced && topo ? ` Its search picked ${describeAction(info.replaced, { topo, layout: res.position.layout })}; the trade layer replaced that with an offer.` : "";
  switch (res.policy_kind) {
    case "heuristic": return `${res.policy} is a rule-based bot: it has no policy probability or value estimate, so those columns are hidden. Pick a trained policy to see them.`;
    case "search": return `${res.policy} is a search agent: it ran ${info?.sims ?? "its"} simulations from your view of this position and marks its own pick as "agent's choice". Search value is its value net's win estimate for each move it searched (a model output, not calibrated).${replaced}`;
    case "agent": return `${res.policy} is a learned agent. "Agent's choice" is the move it plays here${info?.source === "forced" || info?.source === "nego_offer" || info?.source === "nego" ? "; no search ran for this decision" : ""}, and the other candidates come from the intermediate bot's pruning.${replaced}`;
    default: return `${res.policy} reports no policy probability or value estimate here, so those columns are hidden.`;
  }
}

function Candidates({ res, hover, setHover, onExplore, compact, busy, topo }: {
  res: AnalysisResult; hover: number | null; setHover: (r: number | null) => void;
  onExplore?: (c: Candidate) => void; compact?: boolean; busy?: boolean; topo?: Topology;
}) {
  const hasSearch = res.candidates.some((c) => c.search_score !== null);
  // Heuristic policies have no policy probability or value net: drop those all-"—" columns.
  const hasModel = res.candidates.some((c) => c.policy_prob != null || c.value_estimate != null);
  const hasSV = res.candidates.some((c) => c.search_value != null);
  const rollN = res.candidates.find((c) => c.rollout)?.rollout?.n;
  if (!res.candidates.length) return <p className="muted">No legal moves to rank in this position.</p>;
  if (compact) {
    return (
      <ol className="an-cands-compact">
        {res.candidates.map((c) => (
          <li key={c.rank} className={hover === c.rank ? "on" : ""} onMouseEnter={() => setHover(c.rank)} onMouseLeave={() => setHover(null)}>
            <div><span className="an-rank">{c.rank}</span> <strong>{candName(c, res, topo)}</strong> <VerdictChip v={c.verdict} rolled={!!c.rollout} />{c.agent_choice && <AgentChoice />}</div>
            <div className="an-nums">
              {hasModel && <><span title="Policy probability">p {pct(c.policy_prob)}</span>
              <span title="Value estimate">v {pct(c.value_estimate)}</span></>}
              {hasSV && <span title="Search value: the search's own win estimate, not calibrated">s {pct(c.search_value)}</span>}
              <span title="Rollout win rate">roll {pct(c.rollout?.win_rate)}{c.rollout ? ` n=${c.rollout.n}` : ""}</span>
            </div>
          </li>
        ))}
      </ol>
    );
  }
  return (
    <>
    {!hasModel && <p className="muted small" data-testid="no-model-cols">{kindNote(res, topo)}</p>}
    {/* DS18-10: rank 4 could have the top bot score with no word on why */}
    {res.ranking && <p className="muted small" data-testid="ranking-note">Ranked by {res.ranking}.{(hasSearch || hasSV) && res.ranking.startsWith("rollout") ? ` ${hasSV ? "Search value" : "Bot score"} is the agent's own estimate, so it can disagree with the rank.` : ""}</p>}
    {rollN !== undefined && res.depth !== "deep" && <p className="muted small" data-testid="noise-note">Rollouts are played by {res.rollout_policy ?? "the rollout policy"}, not by {res.policy}. At n={rollN} a win rate is only good to about ±15 points, so close moves are noise; Deep runs more rollouts and separates them.</p>}
    <table className="an-cands" data-testid="candidates">
      <thead>
        <tr>
          <th>#</th><th>Move · verdict</th>
          {hasModel && <th title="The policy's probability of choosing this move. Not a win rate.">Policy p</th>}
          {hasModel && <th title="Value network's win-probability estimate after this move. Model output, no CI.">Value est.</th>}
          {hasSearch && <th title={`Score from the heuristic/search bot, on its own scale${res.search_score_kind ? `: ${res.search_score_kind}` : ""}`}>Bot score</th>}
          {hasSV && <th title={`The search's own estimate for this move${res.search_value_kind ? `: ${res.search_value_kind}` : ""}. Model output, not calibrated, no CI. Sims = simulations the search spent on it.`}>Search value</th>}
          <th title="Empirical win rate over n simulated continuations, with 95% CI">Rollout win rate</th>
          {onExplore && <th />}
        </tr>
      </thead>
      <tbody>
        {res.candidates.map((c) => (
          <tr key={c.rank} className={hover === c.rank ? "on" : ""} onMouseEnter={() => setHover(c.rank)} onMouseLeave={() => setHover(null)} data-rank={c.rank}>
            <td className="an-rank">{c.rank}</td>
            <td className="an-move">{candName(c, res, topo)}<br /><VerdictChip v={c.verdict} rolled={!!c.rollout} />{c.agent_choice && <AgentChoice />}</td>
            {hasModel && <td className="num">{pct(c.policy_prob)}</td>}
            {hasModel && <td className="num">{pct(c.value_estimate)}</td>}
            {hasSearch && <td className="num">{c.search_score === null ? "—" : c.search_score.toFixed(2)}</td>}
            {hasSV && <td className="num">{pct(c.search_value)}{c.search_visits ? <span className="muted small"> {c.search_visits} sim{c.search_visits === 1 ? "" : "s"}</span> : null}</td>}
            <td className="an-roll">
              {c.rollout ? (
                <>
                  <span className="num">{pct(c.rollout.win_rate)}</span>
                  <RateBar rate={c.rollout.win_rate} ci={c.rollout.ci95} />
                  <span className="muted small" title={`95% CI ${pct(c.rollout.ci95[0])} to ${pct(c.rollout.ci95[1])}`}>n={c.rollout.n}</span>
                </>
              ) : <span className="num">—</span>}
            </td>
            {onExplore && <td><button className="chip" onClick={() => onExplore(c)} disabled={busy} title="Play this move and let the policy continue to the end of the turn" aria-label={`Explore ${candName(c, res, topo)}`} data-testid="explore">Try</button></td>}
          </tr>
        ))}
      </tbody>
    </table>
    </>
  );
}

function HowComputed({ res, clientMs }: { res: AnalysisResult; clientMs?: number }) {
  return (
    <details className="an-how">
      <summary>How this was computed</summary>
      <dl>
        <dt>Policy</dt><dd className="mono-ish">{res.policy}</dd>
        <dt>Rollout policy</dt><dd className="mono-ish">{res.rollout_policy ?? "— (no rollouts)"}</dd>
        <dt>Depth</dt><dd>{res.depth}</dd>
        <dt>Acting seat</dt><dd>{PLAYER_NAMES[res.actor] ?? res.actor} · {res.phase} · {res.legal_count} legal moves</dd>
        {res.ranking && <><dt>Ranking</dt><dd>{res.ranking}</dd></>}
        {res.candidate_source && <><dt>Candidates from</dt><dd>{res.candidate_source}</dd></>}
        {res.search_score_kind && <><dt>Bot score scale</dt><dd>{res.search_score_kind}</dd></>}
        {res.search_value_kind && <><dt>Search value</dt><dd>{res.search_value_kind}; not calibrated</dd></>}
        {res.verdict_thresholds !== undefined && <><dt>Verdict thresholds</dt><dd className="mono-ish">{JSON.stringify(res.verdict_thresholds)}</dd></>}
        <dt>Determinizations</dt><dd>{res.determinization?.samples ?? 0} samples of hidden information</dd>
        <dt>Timing</dt><dd>{Object.entries(res.timing_ms ?? {}).map(([k, v]) => `${k} ${secs(v)}`).join(" · ")}{clientMs !== undefined ? ` · round trip ${secs(clientMs)}` : ""}</dd>
      </dl>
      <span className="field-label">Assumptions</span>
      <ul>{(res.determinization?.assumptions ?? []).map((a, i) => <li key={i}>{a}</li>)}</ul>
      <p className="muted small">Policy p is what the model would play. Value est. is a model guess at the win chance. Rollout win rate is measured over n simulated games, with a 95% interval. They answer different questions, so they are never combined.</p>
    </details>
  );
}

/** Posterior over opponents' hands from the viewer's public events (catan/bots/belief.py). */
function BeliefPanel({ b }: { b: BeliefSummary }) {
  const dist = (d: Record<string, number>) => Object.entries(d).map(([k, p]) => `${k}:${(p * 100).toFixed(0)}%`).join(" ");
  return (
    <div className="an-card an-belief" data-testid="belief-panel">
      <h4>Opponent hands <span className="muted small">({b.label})</span></h4>
      <table className="an-belief-t">
        <thead><tr><th></th>{RES_LABEL.map((r) => <th key={r}>{r}</th>)}</tr></thead>
        <tbody>
          {Object.entries(b.opponents).map(([q, o]) => (
            <tr key={q} data-seat={q}>
              <th>{PLAYER_NAMES[+q]} <span className={"an-bk" + (o.known ? " known" : "")}>{o.known ? "known" : "estimated"}</span></th>
              {o.expected.map((e, r) => {
                const sure = Object.keys(o.dist[r]).length === 1;
                return <td key={r} className={sure ? "" : "unsure"} title={sure ? "Certain from public events" : `P(count): ${dist(o.dist[r])}`}>
                  {sure ? e.toFixed(0) : e.toFixed(1)}{!sure && <small>{dist(o.dist[r])}</small>}
                </td>;
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <p className="muted small">
        Exact posterior from the cards everyone saw move: {b.states} possible hand combination{b.states === 1 ? "" : "s"}
        {b.unseen_steals ? `, after ${b.unseen_steals} steal${b.unseen_steals === 1 ? "" : "s"} you didn't see` : ""}
        {b.approx ? " (least likely combinations pruned)" : ""}. Determinizations draw hands from it.
      </p>
    </div>
  );
}

/** Integrated-gradient attribution of an RL policy's top move, by block and board entity. */
function AttributionCard({ at, onRefs }: { at: NonNullable<AnalysisResult["attribution"]>; onRefs?: (r: Refs | null) => void }) {
  const [head, setHead] = useState<"logit" | "value">("logit");
  const h = at[head];
  return (
    <div className="an-card an-attr" data-testid="analysis-attribution">
      <h4>{at.label} <span className="muted small">for the top move</span></h4>
      <div className="segmented small" role="group" aria-label="Attribution target">
        <button className={head === "logit" ? "on" : ""} onClick={() => setHead("logit")}>Top move's logit</button>
        <button className={head === "value" ? "on" : ""} onClick={() => setHead("value")}>Value</button>
      </div>
      <EntityAttribution entities={at.entities} head={head} onRefs={onRefs} />
      <p className="muted small">By input block: {at.blocks.slice(0, 4).map((b) => `${b.block} ${b[head] >= 0 ? "+" : "−"}${Math.abs(b[head]).toFixed(2)}`).join(" · ")}</p>
      <p className="muted small">Baseline: {at.baseline}. {head === "logit" ? "Logit" : "Value"} {h.at_baseline.toFixed(2)} → {h.at_obs.toFixed(2)}; attributions sum to {h.sum.toFixed(2)} (gap {h.completeness_gap.toFixed(3)}, {at.steps} steps).</p>
      <ul className="di-caveats">{at.caveats.map((c) => <li key={c}>{c}</li>)}</ul>
    </div>
  );
}

// Module-level, not per mount: the panel remounts when the layout turns compact, and a per-mount
// cache wiped the results on screen (DS18-08). shortcut: never evicted, fine for one session's positions.
const resultCache = { current: new Map<string, Results>() };

export default function AnalysisPanel({ spec, source, blocked, actor, settings, setSettings, agents, checkpoints, results, setResults, hover, setHover, onExplore, exploring, onAttr, topo }: {
  spec: PositionSpec; source: AnalysisSource | null; blocked: string | null; actor: number;
  settings: Settings; setSettings: (s: Settings) => void; agents: string[]; checkpoints: string[];
  results: Results; setResults: (r: Results) => void; hover: number | null; setHover: (r: number | null) => void;
  onExplore: (c: Candidate, policy: string) => void; exploring: boolean; onAttr?: (r: Refs | null) => void; topo?: Topology;
}) {
  const [busy, setBusy] = useState<{ t0: number; ctl: AbortController } | null>(null);
  const [now, setNow] = useState(0);
  const [err, setErr] = useState<string | null>(null);
  const [clientMs, setClientMs] = useState<Record<string, number>>({});
  const [measured, setMeasured] = useState<Record<string, number>>(() => { try { return JSON.parse(localStorage.getItem(MEASURED) ?? "{}"); } catch { return {}; } });
  const cache = resultCache;
  const S = settings;
  const pa = policySpec(S.agent, S.checkpoint), pb = policySpec(S.agentB, S.checkpointB);
  const useBelief = !!S.belief && !!source;
  const key = JSON.stringify([spec, source, pa, S.compare ? pb : null, S.rollout, S.depth, useBelief]);

  // A new position (or settings) shows its cached result, or nothing. Never a stale one.
  useEffect(() => { setResults(cache.current.get(key) ?? {}); setErr(null); }, [key]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => () => busy?.ctl.abort(), [key]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (!busy) return;
    const t = setInterval(() => setNow(performance.now()), 100);
    return () => clearInterval(t);
  }, [busy]);

  const run = async () => {
    if (busy) return;
    const ctl = new AbortController(), t0 = performance.now();
    setBusy({ t0, ctl }); setNow(t0); setErr(null);
    const body = (policy: string) => ({ position: spec, source: source ?? undefined, viewer: spec.viewer, policy, rollout_policy: S.rollout || undefined, depth: S.depth, top_k: 5, belief: useBelief || undefined });
    try {
      const [a, b] = await Promise.all([analysis.evaluate(body(pa), ctl.signal), S.compare ? analysis.evaluate(body(pb), ctl.signal) : Promise.resolve(undefined)]);
      const r = { a, b };
      cache.current.set(key, r);
      setResults(r);
      const ms = performance.now() - t0;
      setClientMs({ a: ms });
      const m = { ...measured, [S.depth]: a.timing_ms?.total ?? ms };
      setMeasured(m);
      localStorage.setItem(MEASURED, JSON.stringify(m));
    } catch (e: any) {
      if (e?.name !== "AbortError") setErr(navigator.onLine ? e.message ?? String(e) : "you are offline. Analysis needs the server; editing still works.");
    } finally {
      setBusy(null);
    }
  };
  const cancel = () => busy?.ctl.abort();

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") return;
      if (e.key === "Enter" && !e.metaKey && !e.ctrlKey && !blocked) { e.preventDefault(); run(); }
      if (e.key === "Escape" && busy) cancel();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  const res = results.a;
  return (
    <section className="an-panel" aria-label="Analysis">
      <div className="an-card an-settings">
        <div className="an-row an-policy">
          <span className="field-label">Policy</span>
          <AgentSelect value={S.agent} agents={agents} onChange={(a) => setSettings({ ...S, agent: a })} checkpoints={checkpoints} checkpoint={S.checkpoint} onCheckpoint={(c) => setSettings({ ...S, checkpoint: c })} />
          {agents.includes("champion") && (
            <button className={"chip" + (S.agent === "champion" ? " on" : "")} onClick={() => setSettings({ ...S, agent: "champion", checkpoint: "" })}
              title="The model most recently promoted by the gate" data-testid="champion-pick">Champion</button>
          )}
        </div>
        {((S.agent === "champion" && championKind() === "nego") || S.agent.startsWith("nego")) && (
          <p className="note small" data-testid="nego-note">This policy is the intermediate bot with a learned trading model. Only trade offers and replies are learned; every other move is the intermediate bot's, with no policy probability or value estimate.</p>
        )}
        <div className="an-row">
          <span className="field-label">Rollouts by</span>
          <select value={S.rollout} onChange={(e) => setSettings({ ...S, rollout: e.target.value })} aria-label="Rollout policy">
            {/* DS27-05: the result then said "basic picks the move" with no hint that the default was basic */}
            <option value="">Server default{!S.rollout && res?.rollout_policy ? ` (${res.rollout_policy})` : ""}</option>
            {agents.filter((a) => a !== "rl" && !SLOW_ROLLOUT.has(a === "champion" ? championName() ?? "" : a)).map((a) => <option key={a} value={a}>{a}</option>)}
          </select>
        </div>
        <div className="an-row">
          <span className="field-label">Depth</span>
          <span className="segmented small" role="radiogroup" aria-label="Depth">
            {DEPTHS.map((d) => (
              <button key={d.id} className={S.depth === d.id ? "on" : ""} onClick={() => setSettings({ ...S, depth: d.id })} role="radio" aria-checked={S.depth === d.id} data-depth={d.id}
                title={`Target ${d.target}${measured[d.id] ? `, last measured ${secs(measured[d.id])}` : ""}`}>{d.label}</button>
            ))}
          </span>
        </div>
        <p className="muted small an-latency">
          Target {DEPTHS.find((d) => d.id === S.depth)!.target}{measured[S.depth] !== undefined && <> · last run {secs(measured[S.depth])}</>}
          {S.depth === "quick" && " · no rollouts, model outputs only"}
        </p>
        <label className="an-check"><input type="checkbox" checked={S.compare} onChange={(e) => setSettings({ ...S, compare: e.target.checked })} /> Compare with a second policy</label>
        <label className="an-check" title={source ? "Sample opponents' hands from what public events imply, instead of uniformly" : "Needs a game or replay: a hand-built position has no event history"}>
          <input type="checkbox" checked={useBelief} disabled={!source} onChange={(e) => setSettings({ ...S, belief: e.target.checked })} data-testid="belief-toggle" />
          {" "}Hands from public events{!source && <span className="muted small"> (needs a game or replay)</span>}
        </label>
        {S.compare && (
          <div className="an-row">
            <span className="field-label">Policy B</span>
            <AgentSelect value={S.agentB} agents={agents} onChange={(a) => setSettings({ ...S, agentB: a })} checkpoints={checkpoints} checkpoint={S.checkpointB} onCheckpoint={(c) => setSettings({ ...S, checkpointB: c })} />
          </div>
        )}
        <div className="an-go">
          {busy ? (
            <>
              <button className="btn big" onClick={cancel} data-testid="cancel-analysis">Cancel <kbd>Esc</kbd></button>
              <span className="an-spin" aria-live="polite">Analyzing… {((now - busy.t0) / 1000).toFixed(1)} s</span>
            </>
          ) : (
            <button className="btn primary big" onClick={run} disabled={!!blocked} data-testid="best-move" title={blocked ?? "Enter"}>Best move? <kbd>Enter</kbd></button>
          )}
        </div>
        {blocked && <p className="note small" data-testid="blocked">{blocked}</p>}
        {!blocked && actor !== spec.viewer && <p className="note small">{PLAYER_NAMES[actor]} acts next, but you are analyzing as {PLAYER_NAMES[spec.viewer]}.</p>}
        {err && <p className="error small" role="alert">Analysis failed: {err}</p>}
      </div>

      {res && !S.compare && (
        <div className={"an-card an-result" + (busy ? " stale" : "")} data-testid="result">
          {res.mock && <p className="an-mock-note">MOCK DATA: generated by the dev mock, not by any policy.</p>}
          {res.timed_out && <p className="note small" data-testid="timed-out">Time limit reached: rollouts stopped early, so n is partial.</p>}
          {res.depth === "quick" && <p className="muted small">Quick depth runs no rollouts, so every verdict is unrated.</p>}
          <div className="an-eval">
            {/* rule-based policies have no value net (DS8-09) */}
            {res.position_eval?.value_estimate != null && <div data-testid="position-value"><span className="field-label">Value est. (position)</span><strong>{pct(res.position_eval.value_estimate)}</strong></div>}
            {/* this rollout lets the rollout policy pick the move too, so it can sit far below the best candidate's */}
            <div data-testid="position-roll" title={res.rollout_policy ? `Every seat, you included, is played by ${res.rollout_policy} from this position, so ${res.rollout_policy} also picks this move. Compare it with the candidates below to see what a better move is worth.` : undefined}>
              <span className="field-label">Rollout win rate (position{res.rollout_policy ? `, ${res.rollout_policy} picks the move` : ""})</span>
              {res.position_eval?.rollout ? (
                <span className="an-eval-roll"><strong>{pct(res.position_eval.rollout.win_rate)}</strong><RateBar rate={res.position_eval.rollout.win_rate} ci={res.position_eval.rollout.ci95} /><span className="muted small">n={res.position_eval.rollout.n}</span></span>
              ) : <strong>—</strong>}
            </div>
          </div>
          <Candidates res={res} hover={hover} setHover={setHover} onExplore={(c) => onExplore(c, pa)} busy={exploring} topo={topo} />
          <p className="muted small" data-testid="hand-sampling">Opponent hands sampled {res.determinization?.hand_sampling === "belief" ? "from the public-event posterior" : "uniformly from the unseen cards"}.</p>
          <HowComputed res={res} clientMs={clientMs.a} />
        </div>
      )}
      {res?.belief && <BeliefPanel b={res.belief} />}
      {res?.attribution && !S.compare && <AttributionCard at={res.attribution} onRefs={onAttr} />}
      {res && S.compare && results.b && (
        <div className={"an-card an-result" + (busy ? " stale" : "")} data-testid="compare">
          {(res.mock || results.b.mock) && <p className="an-mock-note">MOCK DATA: generated by the dev mock, not by any policy.</p>}
          <div className="an-compare">
            {[res, results.b].map((r, i) => (
              <div key={i}>
                <h4 className="mono-ish" title={r.policy}>{i ? "B" : "A"}: {r.policy.replace(/^rl:checkpoint=runs\//, "rl:")}</h4>
                <Candidates res={r} hover={i === 0 ? hover : null} setHover={i === 0 ? setHover : () => {}} compact topo={topo} />
              </div>
            ))}
          </div>
          <p className="muted small">p = policy probability, v = value estimate, roll = rollout win rate. Board highlights follow policy A.</p>
          <HowComputed res={res} />
        </div>
      )}
    </section>
  );
}
