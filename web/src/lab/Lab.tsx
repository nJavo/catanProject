import { useEffect, useLayoutEffect, useState } from "react";
import { api } from "../api";
import { go } from "../App";
import { PLAYER_COLORS, PLAYER_NAMES, shortSpec, turnNumber } from "../rules";
import { agentHint, AgentSelect, CrestBadge, SeatChoice, seatPayload, useAgents } from "../Setup";
// DS34-03: the game-setup hint "A fair first game" is advice for a human player; every lab seat is an AI
const LAB_HINT: Record<string, string> = { intermediate: "Stronger rule-based bot. The usual benchmark opponent." };
import { LineChart } from "./Charts";
import Tournaments from "./Tournaments";
import Behavior from "./Behavior";
import Leaderboard from "./Leaderboard";

const NO_RESULT = "This replay file doesn't record the result. Watch it to see how the game ended.";

const TABS = [
  { id: "match", label: "AI match", about: "Seat two to four AI agents and watch them play, or simulate the whole game at once." },
  { id: "replays", label: "Replays", about: "Saved games. Step through them move by move and see each player's hand." },
  { id: "runs", label: "Training runs", about: "Learning curves and evaluations of training runs: reinforcement learning, behaviour cloning and distillation." },
  { id: "tournaments", label: "Tournaments", about: "Many-game matches between agents: win rates with confidence intervals, head to head and per seat." },
  { id: "behavior", label: "Behavior", about: "How each agent plays: openings, expansion, robber use, trading and dev cards." },
  { id: "leaderboard", label: "Leaderboard", about: "Every agent rated on one scale, and the model registry. The Champion is the model the gate promoted last." },
];

function MatchSetup() {
  const { agents, checkpoints, loading } = useAgents();
  const [n, setN] = useState(4);
  const [seats, setSeats] = useState<SeatChoice[]>(Array.from({ length: 4 }, () => ({ type: "agent", agent: "random" })));
  const [seed, setSeed] = useState("");
  const [maxTurns, setMaxTurns] = useState(500);
  // DS23-05: same Points to win field as New game (Setup.tsx), clamped on blur
  const [vpText, setVpText] = useState("10");
  const vp = Math.max(3, Math.min(20, Number(vpText) || 10));
  const [err, setErr] = useState<string | null>(null);
  const [batch, setBatch] = useState<string | null>(null);
  // Same default as New game: four random bots make a meaningless first match.
  const best = agents.includes("intermediate") ? "intermediate" : agents.includes("basic") ? "basic" : agents[0];
  // layout effect: no painted frame pairs the loaded agent list with the random placeholder (DS17-10)
  useLayoutEffect(() => {
    setSeats((s) => s.map((x) => (x.agent === "random" && best && best !== "random" ? { ...x, agent: best } : x)));
  }, [best]);
  const body = () => ({
    agents: seats.slice(0, n).map((s) => ({ agent: s.agent, kwargs: seatPayload(s).kwargs as any })),
    seed: seed ? Number(seed) : undefined, max_turns: maxTurns, vp_to_win: vp,
  });
  const watch = () => api.labMatch(body()).then((r) => go(`/lab/match/${r.id}`)).catch((e) => setErr(e.message));
  const instant = async () => {
    setBatch("Simulating…");
    try {
      const r = await api.labMatch({ ...body(), run_to_end: true });
      const res = r.result;
      await api.saveReplay(r.id);
      setBatch(`${res.winner >= 0 ? PLAYER_NAMES[res.winner] + " won" : "Truncated"} on turn ${turnNumber(res.turns, true, res.truncated)} (VP ${res.vp.join(" / ")}). Saved to Replays.`);
    } catch (e: any) { setBatch(null); setErr(e.message); }
  };
  return (
    <section className="lab-card match-setup" data-testid="match-setup">
      <h2>Set up an AI match</h2>
      <p className="muted">Every seat is an agent from the registry. Watch it move by move, or simulate it instantly and save the replay.</p>
      {/* same seat-count control and seat rows as New game (Setup.tsx), DS1-14 */}
      <div className="field">
        <span className="field-label">Players</span>
        <div className="segmented" role="radiogroup" aria-label="Number of players">
          {[2, 3, 4].map((k) => <button key={k} role="radio" aria-checked={n === k} className={n === k ? "on" : ""} onClick={() => setN(k)}>{k}</button>)}
        </div>
        {n === 2 && <p className="note">Two players is an unofficial variant: standard rules, two seats.</p>}
      </div>
      {agents.includes("champion") && (
        <p className="note" data-testid="champion-quick">
          The Champion is the AI model that last passed the promotion gate.{" "}
          <button className="btn small" onClick={() => setSeats((x) => x.map((y) => ({ ...y, agent: "champion" })))} data-testid="all-champion">Make every seat the Champion</button>
        </p>
      )}
      <ol className="seat-list">
        {seats.slice(0, n).map((s, i) => (
          <li key={i} className="seat-row" style={{ ["--pc" as any]: PLAYER_COLORS[i] }}>
            <CrestBadge p={i} size={30} /><strong className="seat-name">{PLAYER_NAMES[i]}</strong>
            <AgentSelect value={s.agent} agents={agents} checkpoints={checkpoints} checkpoint={s.checkpoint}
              onChange={(a) => setSeats((x) => x.map((y, k) => (k === i ? { ...y, agent: a } : y)))}
              onCheckpoint={(c) => setSeats((x) => x.map((y, k) => (k === i ? { ...y, checkpoint: c } : y)))} />
            {!loading && <span className="muted small seat-hint" data-testid="agent-hint">{LAB_HINT[s.agent] ?? agentHint(s.agent)}</span>}
          </li>
        ))}
      </ol>
      <div className="field-row">
        <label className="field"><span className="field-label">Seed</span><input placeholder="Random" value={seed} onChange={(e) => setSeed(e.target.value.replace(/\D/g, ""))} /></label>
        <label className="field"><span className="field-label">Turn limit</span><input type="number" min={10} max={5000} value={maxTurns} onChange={(e) => setMaxTurns(Number(e.target.value) || 500)} /></label>
        <label className="field"><span className="field-label">Points to win</span><input type="number" min={3} max={20} value={vpText} onChange={(e) => setVpText(e.target.value)} onBlur={() => setVpText(String(vp))} /></label>
      </div>
      {err && <p className="error">{err}</p>}
      {batch && <p className="note" data-testid="batch-result">{batch}</p>}
      <div className="setup-go">
        <button className="btn" onClick={instant} disabled={loading}>Simulate instantly</button>
        <button className="btn primary" onClick={watch} disabled={loading} data-testid="watch-match">Watch match</button>
      </div>
    </section>
  );
}

// "2026-11-04 05:12", the format the Tournaments list uses
const fmtWhen = (t?: number) => (t ? new Date(t * 1000).toLocaleString("sv-SE", { dateStyle: "short", timeStyle: "short" }) : "—");

function Replays() {
  const [list, setList] = useState<any[] | null>(null);
  const [q, setQ] = useState("");
  useEffect(() => { api.replays().then((r) => setList(r.replays)).catch(() => setList([])); }, []);
  if (!list) return <p className="muted">Loading…</p>;
  // DS36-04 (DS23-07): filter by file name or agent, as Training runs does above 8 items
  const ql = q.trim().toLowerCase();
  const shown = list.filter((r) => [r.name, ...(r.agents ?? [])].some((x) => String(x).toLowerCase().includes(ql)));
  if (!list.length) return <section className="lab-card"><h2>No replays yet</h2><p className="muted">Finish a game or simulate an AI match, then save its replay. Files live in results/replays/.</p></section>;
  return (
    <section className="lab-card">
      <h2>Saved replays</h2>
      {list.length > 8 && <input type="search" className="run-filter" placeholder={`Filter ${list.length} replays by file or agent`} aria-label="Filter replays" style={{ maxWidth: 360 }} value={q} onChange={(e) => setQ(e.target.value)} />}
      <table className="table">
        <thead><tr><th>File</th><th title="When the game was played (file time if the replay does not say)">Played</th><th>Agents</th><th>Seed</th><th>Moves</th><th title="The turn the game ended on, as the banner shows it" style={{ whiteSpace: "nowrap" }}>Last turn</th><th>Winner</th><th /></tr></thead>
        <tbody>
          {shown.map((r) => (
            <tr key={r.name}>
              <td className="mono-ish replay-name">{String(r.name).split("_").flatMap((part: string, i: number) => (i ? ["_", <wbr key={i} />, part] : [part]))}</td>
              <td style={{ whiteSpace: "nowrap" }}>{r.created ? fmtWhen(r.created) : <span className="muted" title={`The replay does not say when it was played. File time ${fmtWhen(r.modified)}, which changes when the file is copied.`} data-testid="file-time">—</span>}</td>
              <td title={(r.agents ?? []).join(", ")}>{/* DS17-03: bold the winner's agent so "Chalk" in Winner maps to an agent without counting seats */}
                {(r.agents ?? []).flatMap((a: string, i: number) => [i ? ", " : "", i === r.winner ? <strong key={i} data-testid="winner-agent">{shortSpec(a)}</strong> : shortSpec(a)])}</td>
              <td>{r.seed}</td><td>{r.actions}</td><td data-col="last-turn">{r.turns == null ? <span className="muted" title={NO_RESULT}>—</span> : turnNumber(r.turns, true, !!r.truncated)}</td>
              <td data-col="winner" style={{ whiteSpace: "nowrap" }}>{r.winner == null ? <span className="muted" title={NO_RESULT}>not recorded</span>
                : r.winner >= 0 ? <span className="t-seat"><CrestBadge p={r.winner} size={16} /> {PLAYER_NAMES[r.winner]}</span> : "none"}</td>
              <td><button className="btn small" onClick={() => go(`/replay/file/${r.name}`)} aria-label={`Watch ${r.name.replace(/\.json$/, "")}`}>Watch</button></td>
            </tr>
          ))}
        </tbody>
      </table>
      {!shown.length && <p className="muted">No replay matches “{q}”.</p>}
    </section>
  );
}

function Runs() {
  const [runs, setRuns] = useState<any[] | null>(null);
  const [sel, setSel] = useState<string | null>(null); // run id, so a poll that adds runs keeps the selection
  const [q, setQ] = useState("");
  const [detail, setDetail] = useState<any | null>(null);
  useEffect(() => {
    // Poll so a run in progress can be watched live. The list is light (no metric rows).
    const load = () => api.runs().then((r) => setRuns(sortRuns(r.runs))).catch(() => setRuns((x) => x ?? []));
    load();
    const t = setInterval(load, 15000);
    return () => clearInterval(t);
  }, []);
  const selId = runs && runs.length ? (runs.find((r) => r.id === sel) ?? runs[0]).id : null;
  useEffect(() => {
    // Detail of the selected run: full once, then only rows appended after the cursor.
    if (!selId) return;
    let live = true, cursor = 0;
    const merge = (d: any) => {
      if (!live) return;
      setDetail((prev: any) => (prev?.id === d.id && d.since > 0 && !d.reset
        ? { ...prev, ...d, config: prev.config, metrics: prev.metrics.concat(d.metrics) } : d));
      cursor = d.cursor;
    };
    setDetail(null);
    api.run(selId, 0).then(merge).catch(() => {});
    const t = setInterval(() => api.run(selId, cursor).then(merge).catch(() => {}), 15000);
    return () => { live = false; clearInterval(t); };
  }, [selId]);
  if (!runs) return <p className="muted">Loading…</p>;
  if (!runs.length) return <section className="lab-card"><h2>No training runs found</h2><p className="muted">Runs appear here when runs/&lt;run_id&gt;/metrics.jsonl exists. Start one with the RL trainer.</p></section>;
  const item = runs.find((r) => r.id === selId)!;
  const run = detail?.id === selId ? detail : { ...item, metrics: [], eval: [], checkpoints: [], loading: true };
  const shown = runs.filter((r) => r.id.toLowerCase().includes(q.trim().toLowerCase()));
  // value-net runs (runs/value_*) nest held-out scores: {val: {all: {brier, logloss, ece, n}, <subset>: ...}} -> val_brier, ...
  // DS36-03: other nested scores ({rank_val: {tau, top1}}, rank_* runs) -> "rank_val.tau", charted under "Evaluation metrics"
  const nested = (r: any) => Object.fromEntries(Object.entries(r).flatMap(([k, o]: [string, any]) => k === "val" || !o || typeof o !== "object" || Array.isArray(o) ? []
    : Object.entries(o).filter(([sk, v]) => typeof v === "number" && sk !== "n" && sk !== "roots").map(([sk, v]) => [`${k}.${sk}`, v])));
  const rows: any[] = run.metrics.map((r: any) => ({ ...r, ...nested(r), ...(r.val?.all && typeof r.val.all === "object"
    ? Object.fromEntries(Object.entries(r.val.all).filter(([k, v]) => k !== "n" && typeof v === "number").map(([k, v]) => [`val_${k}`, v])) : {}) }));
  const keys = Array.from(new Set(rows.flatMap((r) => Object.keys(r)))).filter((k) => k !== "step" && k !== "time" && rows.some((r) => typeof r[k] === "number"));
  // DS37-05: "<k>_se" is the standard error of <k>: draw it as a 95% band on <k>, not as a chart of its own
  const seOf = (k: string) => (rows.some((r) => typeof r[k + "_se"] === "number") ? k + "_se" : null);
  keys.splice(0, keys.length, ...keys.filter((k) => !(k.endsWith("_se") && keys.includes(k.slice(0, -3)))));
  const pref = ["win_rate", "reward", "mean_reward", "policy_loss", "value_loss", "loss", "entropy"];
  keys.sort((a, b) => (pref.indexOf(a) + 1 || 99) - (pref.indexOf(b) + 1 || 99) || a.localeCompare(b));
  const evalRows: any[] = run.eval ?? [];
  // Eval rows nest results per opponent: {per_opponent: {random: {win_rate, ci95, avg_vp, ...}}}.
  const opponents = Array.from(new Set(evalRows.flatMap((r) => Object.keys(r.per_opponent ?? {}))));
  const evalXs = evalRows.map((r, i) => r.step ?? i);
  const oppSeries = (opp: string, key: string) => evalRows.map((r) => r.per_opponent?.[opp]?.[key] ?? NaN);
  const last = evalRows.length ? evalRows[evalRows.length - 1] : null;
  // eval/* metrics duplicate the evaluation section above, so they are dropped here.
  const OPT = ["policy_loss", "value_loss", "entropy", "approx_kl", "clipfrac", "explained_variance", "grad_norm", "value_mean", "reward_mean", "kl_early_stop"];
  // BC, DAgger, ExIt and KL-regularized runs log their own losses and accuracies (train_ce, heldout_top1, kl_bc, ...)
  const isOpt = (k: string) => OPT.includes(k) || (!k.includes("/") && /loss|_acc$|_ce$|_mse$|_bce$|brier|_ece$|top1|^kl_/.test(k));
  const isNested = (k: string) => k.includes(".");
  // DS39-05: dmc logs train/<opp>/{games,vp,win_rate}. "games" is the size of the rolling window (bookkeeping), and
  // train/self/win_rate is always 100% (a learner seat wins every self-play game), so it is dropped. Win rates read first.
  const counter = (k: string) => /^train\/[^/]+\/games$/.test(k);
  const trainKeys = keys.filter((k) => k.startsWith("train/") && !counter(k) && k !== "train/self/win_rate")
    .sort((a, b) => (a.endsWith("/win_rate") ? 0 : 1) - (b.endsWith("/win_rate") ? 0 : 1) || a.split("/").pop()!.localeCompare(b.split("/").pop()!) || a.localeCompare(b));
  const groups: [string, string[], boolean][] = [
    ["Evaluation metrics", keys.filter(isNested), true],
    ["Training progress (games played during training)", trainKeys, true],
    ["Optimization", keys.filter((k) => isOpt(k) && !isNested(k)), true],
    ["Throughput and bookkeeping", keys.filter((k) => (counter(k) || !k.startsWith("train/")) && !k.startsWith("eval/") && !isOpt(k) && !isNested(k)), false],
  ];
  return (
    <div className="runs">
      <nav className="run-list t-list" aria-label="Training runs">
        {runs.length > 8 && <input type="search" className="run-filter" placeholder={`Filter ${runs.length} runs`} aria-label="Filter runs" value={q} onChange={(e) => setQ(e.target.value)} />}
        {shown.map((r) => (
          <button key={r.id} className={"run-item" + (r.id === selId ? " on" : "")} onClick={() => setSel(r.id)} aria-current={r.id === selId}>
            <strong>{r.id}</strong>
            <small>{hasData(r) ? [startedAt(r) ? startedAt(r).replace("T", " ") : r.updated && `updated ${localIso(r.updated).replace("T", " ")}`,
              r.kind === "bc" ? `behaviour cloning${r.bc?.heldout_acc != null ? `, held-out acc ${pct(r.bc.heldout_acc)}` : ""}` : `${r.metrics_total} log steps`,
              r.eval_total > 0 && plural(r.eval_total, "eval"), plural(r.checkpoints, "checkpoint")].filter(Boolean)
              /* DS17-02: each part stays whole, so "0 checkpoints" never breaks into "0" and an orphaned word */
              .flatMap((s, i) => [i ? " · " : "", <span key={i} style={{ whiteSpace: "nowrap" }}>{s}</span>]) : "no metrics or eval yet"}</small>
          </button>
        ))}
        {!shown.length && <p className="muted">No run matches “{q}”.</p>}
      </nav>
      <section className="lab-card">
        <h2>{run.id}</h2>
        {run.meta && <p className="muted" data-testid="run-meta">{runMeta({ ...run.meta, started: startedAt(run) })}</p>}
        {run.loading && <p className="muted">Loading…</p>}
        {run.bc_summary && <BcSummary b={run.bc_summary} />}
        {opponents.length > 0 && <>
          <h3>Evaluation vs fixed opponents</h3>
          <p className="muted">Held-out seeds{last?.seeds ? ` ${last.seeds[0]}–${last.seeds[1]}` : ""}, seats rotated, {last?.num_players ?? "?"} players. Shaded band = 95% Wilson CI. Chance level = {last?.num_players ? pct(1 / last.num_players) : "1/N"}.</p>
          <div className="charts">
            {opponents.map((o) => <LineChart key={"wr" + o} title={`Win rate vs ${o}`} xs={evalXs} ys={oppSeries(o, "win_rate")}
              band={evalRows.map((r) => r.per_opponent?.[o]?.ci95 ?? null)} yRange={[0, 1]} pct chance={last?.num_players ? 1 / last.num_players : undefined} />)}
            {opponents.map((o) => <LineChart key={"vp" + o} title={`Avg VP vs ${o}`} xs={evalXs} ys={oppSeries(o, "avg_vp")} />)}
            {opponents.map((o) => <LineChart key={"t" + o} title={`Avg game length (turns) vs ${o}`} xs={evalXs} ys={oppSeries(o, "avg_turns")} />)}
          </div>
          <table className="eval-table"><thead><tr><th>checkpoint</th><th>step</th>{opponents.map((o) => <th key={o}>vs {o} (n)</th>)}</tr></thead>
            <tbody>{evalRows.map((r, i) => { const ck = String(r.checkpoint ?? "").split("/").pop()!; return <tr key={i}><td>{ck}
              {ck && !run.loading && !run.checkpoints.includes(ck) && <span className="muted" data-testid="ckpt-missing"
                title="This checkpoint was evaluated, but its file is no longer in the run's checkpoints folder"> (file missing)</span>}</td><td>{r.step?.toLocaleString("en-US")}</td>
              {opponents.map((o) => { const s = r.per_opponent?.[o]; return <td key={o}>{s ? <>{pct(s.win_rate)}{s.ci95 && <span className="muted"> [{pct(s.ci95[0])}, {pct(s.ci95[1])}]</span>} <span className="muted">({s.games})</span></> : "—"}</td>; })}</tr>; })}</tbody></table>
        </>}
        {!keys.length && !run.loading && !run.bc_summary && <p className="muted">metrics.jsonl has no numeric series yet.</p>}
        {groups.map(([title, ks, open]) => ks.length > 0 && (
          <details key={title} open={open} className="metric-group">
            <summary><h3>{title}</h3></summary>
            <div className="charts">{ks.map((k) => <LineChart key={k} title={(METRIC_LABEL[k] ? `${METRIC_LABEL[k]} (${k})` : k) + (seOf(k) ? " ± 95% band" : "")} xs={rows.map((r) => r.step)} ys={rows.map((r) => r[k])}
              band={seOf(k) ? rows.map((r) => (Number.isFinite(r[k]) && Number.isFinite(r[seOf(k)!]) ? [r[k] - 1.96 * r[seOf(k)!], r[k] + 1.96 * r[seOf(k)!]] as [number, number] : null)) : undefined}
              yRange={/win_rate|truncated_rate|_acc$|top1/.test(k) ? [0, 1] : undefined} pct={/win_rate|truncated_rate|_acc$|top1/.test(k)} />)}</div>
          </details>))}
        {run.checkpoints.length > 0 && <p className="muted">Checkpoints: {run.checkpoints.join(", ")}</p>}
        {run.config && <details><summary>config.yaml</summary><pre>{run.config}</pre></details>}
      </section>
    </div>
  );
}

// {"arch":"gnn","obs":2,"d":64} reads "gnn (obs 2, d 64)", not raw JSON (DS19-12)
const archText = (a: any) => a == null ? null : typeof a !== "object" ? String(a)
  : `${a.arch ?? "net"} (${Object.entries(a).filter(([k]) => k !== "arch").map(([k, v]) => `${k} ${v}`).join(", ")})`;

// Behaviour-cloning folders (bc_summary.json, no metrics.jsonl): fit summary and per-epoch curve.
function BcSummary({ b }: { b: any }) {
  const h: any[] = b.history ?? [];
  const facts: [string, any][] = [["teacher", b.teacher], ["arch", archText(b.arch)], ["decisions", b.decisions?.toLocaleString("en-US")],
    ["games", b.games], ["params", b.params?.toLocaleString("en-US")], ["fit time", (b.fit_s ?? b.total_s) != null ? `${Math.round(b.fit_s ?? b.total_s)} s` : null]];
  return (
    <div data-testid="bc-summary">
      <h3>Behaviour cloning</h3>
      <p className="muted">{facts.filter(([, v]) => v != null).map(([k, v]) => `${k}: ${v}`).join("  ·  ")}</p>
      {h.length > 0 && <div className="charts">
        <LineChart title="held-out accuracy" xs={h.map((r) => r.epoch)} ys={h.map((r) => r.heldout_acc ?? NaN)} yRange={[0, 1]} pct />
        <LineChart title="train loss" xs={h.map((r) => r.epoch)} ys={h.map((r) => r.train_loss ?? NaN)} />
      </div>}
    </div>
  );
}

// "Started 2026-11-04 03:11 · commit fd21bf9 (uncommitted changes) · macOS arm64, 16 CPUs, torch 2.15.0, MPS"
function runMeta(m: any): string {
  const h = m.hardware && typeof m.hardware === "object" ? m.hardware : null;
  return [m.started && `Started ${String(m.started).slice(0, 16).replace("T", " ")}`,
    m.git_commit && `commit ${String(m.git_commit).slice(0, 7)}${m.git_dirty ? " (uncommitted changes)" : ""}`,
    h ? [h.platform && String(h.platform).split("-")[0], h.machine, h.cpu_count && `${h.cpu_count} CPUs`, h.torch && `torch ${h.torch}`, h.mps && "MPS", h.cuda && "CUDA"].filter(Boolean).join(", ")
      : m.hardware && String(m.hardware)].filter(Boolean).join("  ·  ");
}

// DS18: plain names for the common logged keys; the raw key stays in brackets so it can be grepped in metrics.jsonl
const METRIC_LABEL: Record<string, string> = {
  win_rate: "Win rate", mean_reward: "Mean episode reward", reward: "Reward", policy_loss: "Policy loss", value_loss: "Value loss",
  loss: "Loss", entropy: "Policy entropy", approx_kl: "Approx. KL to old policy", clipfrac: "Clipped fraction",
  explained_variance: "Value explained variance", grad_norm: "Gradient norm", train_acc: "Train top-1 accuracy",
  train_ce: "Train cross-entropy", train_value_mse: "Train value MSE", heldout_top1: "Held-out top-1 accuracy",
  kl_bc: "KL to the BC policy", kl_ref: "KL to the reference policy", kl_prior: "KL to the prior",
};
const pct = (x: number) => `${(x * 100).toFixed(1)}%`;
const plural = (n: number, w: string) => `${n} ${w}${n === 1 ? "" : "s"}`;
const hasData = (r: any) => r.metrics_total > 0 || r.eval_total > 0 || r.kind === "bc";
// Newest first by meta.started; folders with no metrics or eval (logs, BC summaries) sink to the end.
// DS36-02: half the runs have no meta.json; fall back to the folder's last write so new ones don't sink below old ones
const localIso = (t: number) => { const d = new Date(t * 1000); return new Date(d.getTime() - d.getTimezoneOffset() * 60000).toISOString().slice(0, 16); };
const startedAt = (r: any): string => (r.meta?.started ? String(r.meta.started).slice(0, 16) : "");
const runWhen = (r: any) => startedAt(r) || (r.updated ? localIso(r.updated) : "");
const sortRuns = (rs: any[]) => [...rs].sort((a, b) => Number(hasData(b)) - Number(hasData(a))
  || runWhen(b).localeCompare(runWhen(a)) || a.id.localeCompare(b.id));
export default function Lab({ tab = "match" }: { tab?: string }) {
  // Each tab treats a failed fetch as "no data yet"; this tells the two apart.
  const [down, setDown] = useState(false);
  useEffect(() => { fetch("/api/health").then((r) => setDown(!r.ok)).catch(() => setDown(true)); }, [tab]);
  // DS16-04: on a phone the tab row scrolls; centre the current tab so it isn't clipped ("Tourname")
  useEffect(() => {
    const on = document.querySelector<HTMLElement>(".lab .page-head .tabs .on"), nav = on?.parentElement;
    if (on && nav && nav.scrollWidth > nav.clientWidth)
      nav.scrollLeft += on.getBoundingClientRect().left - nav.getBoundingClientRect().left - (nav.clientWidth - on.offsetWidth) / 2;
  }, [tab]);
  return (
    <main className="page lab">
      <header className="page-head">
        <button className="btn ghost" onClick={() => go("/")}>Menu</button>
        <h1>AI laboratory</h1>
        <nav className="tabs big" role="tablist">
          {TABS.map((t) => <button key={t.id} role="tab" aria-selected={tab === t.id} className={tab === t.id ? "on" : ""} onClick={() => go(`/lab/${t.id}`)}>{t.label}</button>)}
        </nav>
      </header>
      {/* outside .lab-body, whose first child is the tab's content (tests wait on it) */}
      <p className="muted lab-about" data-testid="lab-about">{(TABS.find((t) => t.id === tab) ?? TABS[0]).about}</p>
      <div className="lab-body">
        {/* replaces the tab's "No … yet" empty state, which a failed fetch would otherwise show (DS6-09) */}
        {down && <section className="lab-card" role="alert" data-testid="server-down"><h2>Can't reach the server</h2>
          <p className="muted">Results may exist, but they can't be loaded. Start the server with uv run python -m catan.server, then retry.</p>
          <div><button className="btn" onClick={() => location.reload()}>Retry</button></div></section>}
        {!down && <>
          {tab === "match" && <MatchSetup />}
          {tab === "replays" && <Replays />}
          {tab === "runs" && <Runs />}
          {tab === "tournaments" && <Tournaments />}
          {tab === "behavior" && <Behavior />}
          {tab === "leaderboard" && <Leaderboard />}
        </>}
      </div>
    </main>
  );
}
