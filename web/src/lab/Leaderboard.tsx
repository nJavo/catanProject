import { useEffect, useState } from "react";
import { api } from "../api";
import { go } from "../App";
import { LineChart } from "./Charts";

// Model zoo view: global ratings, registry entries and per-run checkpoint ladders.
const get = (u: string) => fetch(u).then((r) => (r.ok ? r.json() : Promise.reject(new Error(`${u}: ${r.status}`))));
const pct = (x: number) => `${(x * 100).toFixed(1)}%`;

function agentOf(e: any) {
  // any learned entry ("rl:...", "arch_rl:...", "nego:checkpoint=..."): the agent is the part before ":",
  // and its registered checkpoint is passed explicitly; heuristic anchors are a bare agent name.
  // deterministic=true is part of what the gate measured, so play that policy, not the sampled one.
  return e.checkpoint ? { agent: e.spec.split(":")[0], kwargs: { checkpoint: e.checkpoint, ...(greedy(e) && { deterministic: true }) } } : { agent: e.spec, kwargs: {} };
}
const flag = (e: any, k: string) => (e?.spec ?? "").match(new RegExp(`[:,]${k}=(true|false)\\b`))?.[1];
const greedy = (e: any) => flag(e, "deterministic") === "true";
// shortcut: defaults mirrored from the agent constructors (rl/arch_rl sample, combo and nego take the top move); update if they change
function policyMode(e: any): "deterministic" | "sampled" | null {
  if (!e?.checkpoint) return null;
  const kind = e.spec.split(":")[0];
  if (kind === "rl" || kind === "arch_rl") return greedy(e) ? "deterministic" : "sampled";
  if (kind === "combo") return flag(e, "deterministic") === "false" ? "sampled" : "deterministic";
  if (kind === "nego") return flag(e, "sample") === "true" ? "sampled" : "deterministic";
  return null;
}
// Entries are gated with different protocols: name the policy mode and the champion test each one faced.
const Mode = ({ e }: { e: any }) => { const m = policyMode(e); return m && <span className="muted small" data-testid="policy-mode"
  title={m === "deterministic" ? "Gated and played with its top-scored move every turn" : "Gated and played by sampling moves from the policy"}> · {m}</span>; };
function champTest(t: any) {
  if (!t?.candidate) return null;
  const champ = t.lineup?.[1] ?? "the champion";
  // DS15-04: "51.0% vs 20.0%" didn't say whose rate the second number was
  if (t.kind === "same_seat") {
    const d = t.discordant ?? {}, s = t.secondary;
    return `champion test: same seat at a table with ${champ} and 2×intermediate, won ${pct(t.candidate.win_rate)} vs ${pct(t.copy?.win_rate ?? 0)} for a ${champ} copy on the same games`
      + ` (pairs ${d.candidate_only ?? 0}–${d.copy_only ?? 0}, McNemar p=${Number(t.mcnemar_p).toPrecision(2)})`
      + (s ? `; within the table (secondary) ${s.cand_wins} vs ${s.champ_wins} wins for ${champ}` : "");
  }
  return t.kind === "mixed"
    ? `champion test: one table with ${champ} and 2×intermediate, won ${pct(t.candidate.win_rate)} vs ${pct(t.champion.win_rate)} for ${champ}`
    : `champion test: vs 3×intermediate, ${pct(t.candidate.win_rate)} vs ${pct(t.null ?? 0.25)} chance`;
}

// The gate lists one "significantly worse ... vs 3xX" reason per benchmark; say it once.
const WORSE = /^significantly worse than the champion vs (.+)$/;
function gateReasons(rs: string[], g?: any) {
  // DS18-09: the failing benchmark's numbers were hover-only, while the won table test showed in full
  const nums = (b: string) => { const t = g?.heuristic_tests?.[b.replace(/^3x/, "")];
    return t?.champion ? ` (${pct(t.candidate.win_rate)} vs ${pct(t.champion.win_rate)} for ${g.champion ?? "the champion"})` : ""; };
  const worse = rs.map((r) => r.match(WORSE)?.[1]).filter(Boolean) as string[];
  const out = [...rs.filter((r) => !WORSE.test(r)), ...(worse.length ? [`significantly worse than the champion vs ${worse.map((b) => b + nums(b)).join(", ")}`] : [])];
  return out.join("; ").replace(/\b3x(?=\w)/g, "3×");
}

function PlayButton({ entry }: { entry: any }) {
  const [err, setErr] = useState<string | null>(null);
  const play = () => {
    const a = agentOf(entry);
    // You in seat 0, three copies of the model.
    api.createGame({ players: [{ type: "human" }, ...[1, 2, 3].map(() => ({ type: "agent", ...a }))] })
      .then((r) => go(`/game/${r.id}`)).catch((e) => setErr(e.message));
  };
  if (entry.available === false)
    return <span className="muted small" data-testid="checkpoint-missing" title={`${entry.checkpoint} is not on this machine, so the model can't be loaded.`}>checkpoint missing</span>;
  return <><button className="btn small" style={{ whiteSpace: "nowrap" }} onClick={play} aria-label={`Play vs this model (${entry.id})`}>Play vs this model</button>{err && <span className="error"> {err}</span>}</>;
}

const TOP = 10;
const REG_TOP = 6;
// Benchmarks are fixed heuristic opponents, retired entries are history: both are hidden from the registry until asked for.
const isBackground = (e: any) => e.anchor || e.kind === "heuristic" || e.status === "retired";

/** Rating with its 95% CI drawn on one scale shared by every row, so overlap reads at a glance; exact bounds on hover. */
function RatingCell({ e, lo, hi }: { e: any; lo: number; hi: number }) {
  if (e.elo === null) return <span className="muted">unrated (no games yet)</span>;
  const [a, b] = e.ci95 ?? [e.elo, e.elo];
  // DS16-01: percent coordinates so the bar fills the column instead of a fixed 160 px
  const x = (v: number) => `${((v - lo) / (hi - lo || 1)) * 100}%`, w = (v: number) => `${Math.max(0.5, ((v - lo) / (hi - lo || 1)) * 100)}%`;
  return (
    <span className="lb-rating" title={`95% CI [${Math.round(a)}, ${Math.round(b)}] · ${e.games} games`}>
      <strong>{Math.round(e.elo)}</strong> <span className="muted small">±{Math.round((b - a) / 2)}</span>
      <svg height={10} aria-hidden="true"><rect x={0} y={4} width="100%" height={2} fill="var(--hull-3)" />
        <rect x={x(a)} y={1} width={w(lo + b - a)} height={8} rx={2} fill="var(--mist)" opacity={0.55} />
        <line x1={x(e.elo)} x2={x(e.elo)} y1={0} y2={10} stroke="var(--sand)" strokeWidth={2} /></svg>
    </span>
  );
}

export default function Leaderboard() {
  const [lb, setLb] = useState<any>(null);
  const [reg, setReg] = useState<any>(null);
  const [lads, setLads] = useState<any[] | null>(null);
  const [allRated, setAllRated] = useState(false);
  const [allReg, setAllReg] = useState(false);
  useEffect(() => {
    get("/api/leaderboard").then(setLb).catch(() => setLb({ entries: [] }));
    get("/api/registry").then(setReg).catch(() => setReg({ entries: [] }));
    get("/api/ladders").then((r) => setLads(r.ladders)).catch(() => setLads([]));
  }, []);
  if (!lb || !reg || !lads) return <p className="muted">Loading…</p>;
  const rated: any[] = lb.entries ?? [];
  // Same order as the ratings above; unrated entries keep their registry order at the end.
  const rank = new Map<string, number>(rated.map((e: any, i: number) => [e.id, i]));
  const entries: any[] = [...(reg.entries ?? [])].sort((a, b) => (rank.get(a.id) ?? 1e9) - (rank.get(b.id) ?? 1e9));
  // default: the best-rated non-benchmark entries (the champion always among them); "show all" brings the rest
  const shownReg = allReg ? entries : entries.filter((e) => !isBackground(e)).filter((e, i) => i < REG_TOP || e.id === reg.champion?.id);
  const byId = new Map<string, any>((reg.entries ?? []).map((e: any) => [e.id, e]));
  const champId = reg.champion?.id;
  const champ = (id: string) => id === champId && <span className="lb-tag champ" title="Promoted by the gate: this is the Champion agent in every agent list" data-testid="champion-tag">Champion</span>;
  const bench = (e: any) => (e?.benchmark ?? e?.anchor) && <span className="lb-tag" title="Fixed heuristic opponent used by the promotion gate">benchmark</span>;
  // DS34-01: entries that load the same checkpoint (sha256) are one network behind different play-time wrappers
  const twins = (id: string) => { const sha = byId.get(id)?.sha256; return sha ? (reg.entries ?? []).filter((o: any) => o.sha256 === sha && o.id !== id).map((o: any) => o.id) : []; };
  const same = (id: string) => { const t = twins(id); return t.length > 0 && <span className="muted small" data-testid="same-weights"
    title={`Loads the same checkpoint file (same sha256) as ${t.join(", ")}. Only the play-time wrapper in the spec differs, so a small rating gap between them is likely noise.`}> · same weights as {t.join(", ")}</span>; };
  const shownRated = allRated ? rated : rated.filter((e, i) => i < TOP || e.id === champId);
  // the scale spans the rows on screen: random (~800) would otherwise squash the top ten's CIs to a few pixels
  const withCi = shownRated.filter((e) => e.elo !== null && e.ci95);
  const lo = Math.min(...withCi.map((e) => e.ci95[0])), hi = Math.max(...withCi.map((e) => e.ci95[1]));
  // status comes from the registry (the source of truth); leaderboard.json can lag a promotion
  const statusOf = (e: any) => byId.get(e.id)?.status ?? e.status;
  const cr = rated.find((e) => e.id === champId);
  const top = rated[0];
  // DS33-03: say whether #1 is clearly ahead of #2, the first question a reader has
  const second = rated[1];
  const gap = top?.ci95 && second?.ci95 && second.elo !== null
    ? `${Math.round(top.elo - second.elo)} ahead of #2 ${second.id}; ${top.ci95[0] <= second.ci95[1] ? "the 95% CIs overlap, so the order is not settled" : "the 95% CIs do not overlap"}`
    : null;
  const fmtR = (e: any) => <><strong>{Math.round(e.elo)}</strong> <span className="muted">[{Math.round(e.ci95[0])}, {Math.round(e.ci95[1])}]</span></>;
  return (
    <>
      <section className="lab-card">
        <h2>Leaderboard</h2>
        {rated.length > 0 && <p className="lb-headline" data-testid="lb-headline">
          {champId ? <>Champion <strong>{champId}</strong>{cr?.elo != null && cr.ci95 && <>: rating {fmtR(cr)}, rank {rated.indexOf(cr) + 1} of {rated.length}</>}</> : "No champion promoted yet"}
          {top && top.id !== champId && top.elo !== null && top.ci95 && <span className="muted"> · top rated {top.id} {fmtR(top)}</span>}
          {gap && <span className="muted" data-testid="lb-gap"> · {gap}</span>}
        </p>}
        <p className="muted small" title={lb.method ?? undefined}>{rated.length ? "Ratings are relative: no entry is fixed, mean 1500. ± and bar = 95% bootstrap CI on one shared scale; hover for the bounds." : (lb.method ?? "No leaderboard yet: run python -m scripts.ladder.")}{lb.games ? ` ${lb.games.toLocaleString("en-US")} games` : ""}{lb.updated ? ` · updated ${String(lb.updated).replace("T", " ")}` : ""}</p>
        {rated.length > 0 && (
          <table className="table" data-testid="ratings">
            <thead><tr><th>#</th><th>Entry</th><th style={{ width: "50%" }}>Rating · 95% CI{withCi.length > 1 && <span className="muted small" data-testid="lb-scale"> · bar spans {Math.round(lo)} to {Math.round(hi)}</span>}</th><th className="num" title="Plackett-Luce over the full finishing order (D34, docs/research/RATING.md); same bootstrap and mean">PL · 95% CI</th><th className="num">Games</th></tr></thead>
            <tbody>{shownRated.map((e: any) => (
              <tr key={e.id} data-entry={e.id} className={e.id === champId ? "lb-champ" : undefined}><td>{e.elo === null ? "—" : rated.indexOf(e) + 1}</td>
                <td title={`${e.kind} · ${statusOf(e)}`}><strong>{e.id}</strong>{champ(e.id)}{bench(e)}{statusOf(e) === "retired" && <span className="lb-tag">retired</span>}<Mode e={byId.get(e.id)} />{same(e.id)}</td>
                <td><RatingCell e={e} lo={lo} hi={hi} /></td>
                <td className="num" data-testid="lb-pl">{e.pl_elo == null ? "—" : <span title={e.pl_ci95 ? `PL 95% CI [${Math.round(e.pl_ci95[0])}, ${Math.round(e.pl_ci95[1])}]` : undefined}>{Math.round(e.pl_elo)}{e.pl_ci95 && <span className="muted small"> ±{Math.round((e.pl_ci95[1] - e.pl_ci95[0]) / 2)}</span>}</span>}</td>
                <td className="num">{e.games}</td></tr>
            ))}</tbody>
          </table>
        )}
        {rated.length > TOP && <div className="lab-toggles"><button className="chip" data-testid="ratings-all" aria-expanded={allRated} onClick={() => setAllRated(!allRated)}>
          {allRated ? `Show top ${TOP}` : `Show all ${rated.length}`}</button></div>}
      </section>
      <section className="lab-card">
        <h2>Model registry</h2>
        {!entries.length ? <p className="muted">models/registry.json is empty. Register with python -m scripts.promote.</p> : <>
          <table className="table">
            <thead><tr><th>Id</th><th>Status</th><th>Run / step</th><th>Gate</th><th /></tr></thead>
            <tbody>{shownReg.map((e) => {
              const g = e.eval;
              return (
                <tr key={e.id}>
                  <td><strong className="nowrap">{e.id}</strong>{champ(e.id)}{bench(e)}<Mode e={e} />{same(e.id)}</td><td>{e.status}</td>
                  <td title={e.sha256 ? `sha256 ${e.sha256}` : undefined}>{e.source_run ? [e.source_run, e.update != null && `update ${e.update}`, e.step != null && `${e.step.toLocaleString("en-US")} steps`].filter(Boolean).join(" · ") : "—"}</td>
                  <td className="lb-gate">{g ? <span title={Object.entries(g.heuristic_tests).map(([h, r]: [string, any]) => `vs 3×${h} ${pct(r.candidate.win_rate)} n=${r.candidate.games}`).join("\n")}>
                    <span className={g.promote ? "lb-pass" : "lb-fail"}>{g.promote ? "passed" : "failed"}</span>{!g.promote && <span className="small">: {gateReasons(g.reasons, g)}</span>}
                    {champTest(g.champion_test) && <span className="muted small" data-testid="champion-test"><br />{champTest(g.champion_test)}</span>}</span> : "—"}</td>
                  <td>{e.id !== "random" && <PlayButton entry={e} />}</td>
                </tr>
              );
            })}</tbody>
          </table>
          <div className="lab-toggles">
            {(entries.length > shownReg.length || allReg) && <button className="chip" data-testid="registry-all" aria-expanded={allReg} onClick={() => setAllReg(!allReg)}>
              {allReg ? `Show top ${REG_TOP}` : `Show all ${entries.length} (benchmarks, retired and lower-rated)`}</button>}
            <span className="muted small">Hover a gate result for its per-benchmark win rates, and a run for the checkpoint's sha256.</span>
          </div>
        </>}
      </section>
      {lads.length > 0 && <section className="lab-card">
        <h2>Checkpoint ladders</h2>
        <p className="muted small">One per training run: each checkpoint vs 3× a fixed opponent on held-out seeds, seats rotated. Band = 95% Wilson CI; hover a table cell for its CI.</p>
        {lads.map((l) => {
          const cks: any[] = l.checkpoints ?? [];
          const xs = cks.map((c) => c.step ?? c.update);
          const opps: string[] = l.opponents ?? [];
          // the summary names each opponent's best checkpoint rate, the question a ladder answers
          const best = opps.map((o) => { const b = cks.reduce((m: any, c) => ((c.results?.[o]?.win_rate ?? -1) > (m?.results?.[o]?.win_rate ?? -1) ? c : m), null);
            return b?.results?.[o] ? `vs 3×${o} ${pct(b.results[o].win_rate)}` : null; }).filter(Boolean);
          return (
            <details key={l.run} className="metric-group" data-ladder={l.run}>
              <summary><h3>{l.run}</h3><span className="muted small">{cks.length} checkpoints · best {best.join(", ")}</span></summary>
              <p className="muted small">Held-out seed block {l.seed_block}, {l.num_players} players. Chance = {pct(1 / (l.num_players || 4))}.{l.updated ? ` Updated ${l.updated}.` : ""}</p>
              <div className="charts">
                {opps.map((o) => (
                  <LineChart key={o} title={`Win rate vs 3×${o}`} xs={xs} ys={cks.map((c) => c.results?.[o]?.win_rate ?? NaN)}
                    band={cks.map((c) => c.results?.[o]?.ci95 ?? null)} yRange={[0, 1]} pct chance={1 / (l.num_players || 4)} />
                ))}
              </div>
              <table className="eval-table"><thead><tr><th>checkpoint</th><th>step</th>{opps.map((o) => <th key={o}>vs 3×{o} (n)</th>)}</tr></thead>
                <tbody>{cks.map((c) => <tr key={c.checkpoint}><td>{c.checkpoint.split("/").pop()}</td><td>{c.step}</td>
                  {opps.map((o) => { const r = c.results?.[o]; return <td key={o}>{r ? <span title={`95% CI [${pct(r.ci95[0])}, ${pct(r.ci95[1])}]`}>{pct(r.win_rate)} <span className="muted">({r.games})</span></span> : "—"}</td>; })}</tr>)}</tbody></table>
            </details>
          );
        })}
      </section>}
    </>
  );
}
