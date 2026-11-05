import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { LineChart, RateBar } from "./Charts";

// Behavior profiles from results/behavior/*.json (scripts/behavior_report.py).
// Each cell shows mean [95% CI] and n; definitions live in catan/analytics/behavior.py.

type Stat = { mean?: number; rate?: number | null; ci95?: [number, number]; n?: number; den?: number };
type Row = { key: string; label: string; kind: "m" | "r"; pct?: boolean; better?: "hi" | "lo" };

const SECTIONS: [string, Row[]][] = [
  ["Outcome", [
    { key: "vp", label: "Final VP", kind: "m", better: "hi" },
    { key: "round_vp10", label: "Rounds to 10 VP (winners only)", kind: "m", better: "lo" },
    { key: "rounds", label: "Game length (rounds)", kind: "m" },
  ]],
  ["Opening", [
    { key: "open_pips_total", label: "Opening pips (2 settlements)", kind: "m", better: "hi" },
    { key: "open_pick_vs_best", label: "Opening pips / best free spot", kind: "m", better: "hi" },
    { key: "open_resource_count", label: "Resources covered by opening", kind: "m", better: "hi" },
    { key: "open_port", label: "Opening on a port", kind: "m" },
  ]],
  ["Expansion and production", [
    { key: "settlements_built", label: "Settlements built after setup / game", kind: "m", better: "hi" },
    { key: "cities_built", label: "Cities built / game", kind: "m", better: "hi" },
    { key: "final_settlements", label: "Settlements standing at end (incl. setup, minus upgrades)", kind: "m" },
    { key: "final_pips", label: "Production pips at end", kind: "m", better: "hi" },
    { key: "production_cards", label: "Production cards / game", kind: "m", better: "hi" },
    { key: "roads_built", label: "Roads built (paid) / game", kind: "m" },
    { key: "dangling_roads", label: "Roads to nowhere at end", kind: "m", better: "lo" },
    { key: "end_turn_has_spot_rate", label: "Turn ends with a reachable settlement spot", kind: "r", pct: true, better: "hi" },
  ]],
  ["Robber", [
    { key: "robber_self_block_rate", label: "Self-block rate", kind: "r", pct: true, better: "lo" },
    { key: "robber_block_eff_mean", label: "Block efficiency (vs best hex)", kind: "m", better: "hi" },
    { key: "robber_no_opponent_rate", label: "Robber hex with no opponent", kind: "r", pct: true, better: "lo" },
    { key: "robber_hits_leader_rate", label: "Hits a public-VP leader", kind: "r", pct: true },
  ]],
  ["Trading", [
    { key: "offers", label: "Domestic offers / game", kind: "m" },
    { key: "offer_success_rate", label: "Offer success rate", kind: "r", pct: true, better: "hi" },
    { key: "domestic_net_cards", label: "Net cards from domestic trades", kind: "m", better: "hi" },
    { key: "maritime", label: "Bank/port trades / game", kind: "m" },
    { key: "maritime_net_cards", label: "Net cards from bank/port trades", kind: "m" },
    { key: "ports_owned", label: "Ports owned at end", kind: "m", better: "hi" },
  ]],
  ["Development cards", [
    { key: "dev_bought", label: "Dev cards bought / game", kind: "m" },
    { key: "dev_first_buy_round", label: "First dev buy (round)", kind: "m" },
    { key: "dev_hold_rounds_mean", label: "Rounds held before playing", kind: "m" },
    { key: "knight_preroll_rate", label: "Knights played before rolling", kind: "r", pct: true, better: "hi" },
    { key: "dev_unplayed_end", label: "Unplayed action cards at end", kind: "m" },
  ]],
  ["Hand management", [
    { key: "discarded_cards", label: "Cards discarded on 7s / game", kind: "m", better: "lo" },
    { key: "hand_at_7_mean", label: "Hand size when a 7 is rolled", kind: "m" },
  ]],
];

const VP_SOURCES: [string, string, string][] = [
  ["vp_settlements", "Settlements", "var(--sand)"],
  ["vp_cities", "Cities", "var(--rope)"],
  ["vp_longest_road", "Longest Road", "var(--mist)"],
  ["vp_largest_army", "Largest Army", "var(--kelp)"],
  ["vp_cards", "VP cards", "var(--foam)"],
];

const TREND: string[] = ["vp", "settlements_built", "cities_built", "final_pips", "open_pips_total", "dangling_roads",
  "robber_self_block_rate", "robber_block_eff_mean", "offer_success_rate", "domestic_net_cards", "knight_preroll_rate"];

const ALL_ROWS = SECTIONS.flatMap(([, rows]) => rows);
// Default view: the metrics that separate strong from weak play; the rest sit behind "All metrics".
const KEY = new Set(["vp", "round_vp10", "open_pips_total", "settlements_built", "cities_built", "final_pips",
  "robber_self_block_rate", "offer_success_rate", "domestic_net_cards", "dev_bought", "discarded_cards"]);
const CHIPS = 6;
const ciText = (s: Stat | undefined, pct?: boolean) => {
  const n = s?.n ?? s?.den;
  return [s?.ci95 && `95% CI [${fmt(s.ci95[0], pct)}, ${fmt(s.ci95[1], pct)}]`, n !== undefined && `n=${n}`].filter(Boolean).join(" ");
};

async function getJson(url: string) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json();
}

// Only net-card balances can go below zero; a normal-approx CI on a count or rate must not (DS12-11).
const SIGNED = new Set(["domestic_net_cards", "maritime_net_cards"]);
function stat(p: any, row: Row): Stat | undefined {
  const s: Stat | undefined = row.kind === "m" ? p?.behavior?.metrics?.[row.key] : p?.behavior?.ratios?.[row.key];
  return s?.ci95 && !SIGNED.has(row.key) && s.ci95[0] < 0 ? { ...s, ci95: [0, s.ci95[1]] } : s;
}
const value = (s?: Stat) => (s ? (s.mean ?? s.rate ?? null) : null);
const fmt = (v: number | null | undefined, pct?: boolean) =>
  v === null || v === undefined || !Number.isFinite(v) ? "—" : pct ? `${(v * 100).toFixed(1)}%` : v.toFixed(2);

function Cell({ s, row, ci }: { s?: Stat; row: Row; ci: boolean }) {
  const v = value(s);
  if (v === null) return <td className="muted">—</td>;
  const t = ciText(s, row.pct);
  return (
    <td style={{ whiteSpace: "nowrap" }} title={t || undefined}>
      <strong>{fmt(v, row.pct)}</strong>
      {ci && t && <span className="muted small" data-testid="ci"> {t.replace("95% CI ", "")}</span>}
    </td>
  );
}

function VpBars({ agents }: { agents: [string, any][] }) {
  // DS29-02: scale to the largest bar (RL checkpoints average ~4 VP, so a fixed 0-10 axis left most of the chart empty)
  const top = Math.max(0, ...agents.map(([, p]) => VP_SOURCES.reduce((t, [k]) => t + (p.behavior?.metrics?.[k]?.mean ?? 0), 0)));
  const max = Math.max(2, Math.ceil(top / 2) * 2), tick = max <= 6 ? 1 : 2;
  // DS24-04 (DS17-07): the bars fill the card's width (measured, so text stays 1:1) instead of a fixed 600 px
  const ref = useRef<HTMLElement>(null);
  const [fw, setFw] = useState(750);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setFw(el.clientWidth));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  const L = 150, W = Math.max(240, fw - L - 24), H = 22, rowsH = agents.length * (H + 8);
  return (
    <figure className="chart" ref={ref} data-testid="vp-sources">
      <figcaption>VP sources at game end <span className="muted">· mean per game{max < 10 && ` · axis to ${max} VP; 10 VP wins`}</span></figcaption>
      <svg viewBox={`0 0 ${W + L} ${rowsH + 16}`} style={{ width: W + L, maxWidth: "100%" }} role="img" aria-label="Stacked VP sources per agent">
        {/* VP axis: without it the bar lengths have no scale */}
        {Array.from({ length: max / tick + 1 }, (_, i) => i * tick).map((v) => (
          <g key={"ax" + v}>
            <line x1={L + (v / max) * W} x2={L + (v / max) * W} y1={0} y2={rowsH - 8} className="grid" />
            <text x={L + (v / max) * W} y={rowsH + 6} className="tick" textAnchor={v === 0 ? "start" : v === max ? "end" : "middle"}>{v === max ? `${v} VP` : v}</text>
          </g>
        ))}
        {agents.map(([name, p], i) => {
          let x = L;
          const y = i * (H + 8);
          return (
            <g key={name}>
              <text x={L - 8} y={y + H / 2 + 4} className="tick" textAnchor="end">{name.length > 24 ? name.slice(0, 23) + "…" : name}<title>{name}</title></text>
              {VP_SOURCES.map(([k, label, color]) => {
                const v = p.behavior?.metrics?.[k]?.mean ?? 0;
                const w = (v / max) * W;
                const r = <rect key={k} x={x} y={y} width={w} height={H} fill={color} stroke="var(--hull)" strokeWidth={1}>
                  <title>{`${name}: ${label} ${v.toFixed(2)} VP`}</title></rect>;
                x += w;
                return r;
              })}
            </g>
          );
        })}
      </svg>
      {/* DS26-01: an HTML legend wraps on phones; the in-SVG row ran past the viewBox and cut off the last two sources */}
      <div className="small muted" data-testid="vp-legend" style={{ display: "flex", flexWrap: "wrap", gap: "4px 14px", paddingLeft: fw > 500 ? L : 0 }}>
        {VP_SOURCES.map(([k, label, color]) => (
          <span key={k} style={{ display: "inline-flex", alignItems: "center", gap: 4 }}><span style={{ width: 10, height: 10, background: color, display: "inline-block" }} />{label}</span>
        ))}
      </div>
    </figure>
  );
}

// DS24-02: a fixed 0-100% axis drew 0-2% win rates as a flat line; start at 0 and stop just above the data (and its CI).
const pctRange = (v: number[]): [number, number] => [0, Math.min(1, Math.max(0.05, ...v.filter(Number.isFinite)))];

function Trends({ agents }: { agents: [string, any][] }) {
  const ck = agents.filter(([, p]) => typeof p.checkpoint_index === "number")
    .sort((a, b) => a[1].checkpoint_index - b[1].checkpoint_index);
  // one series per run: a deterministic variant of the same checkpoint would double up
  const seen = new Set<number>();
  const series = ck.filter(([, p]) => !seen.has(p.checkpoint_index) && seen.add(p.checkpoint_index));
  if (series.length < 2) return null;
  const xs = series.map(([, p]) => p.checkpoint_index);
  const win = series.map(([, p]) => p.outcome ?? {}); // DS36-07 (DS35-08): a profile may have no outcome block
  return (
    <>
      <h3>Across checkpoints</h3>
      <p className="muted">One point per checkpoint (stochastic policy). Shaded band = 95% CI.</p>
      <div className="charts">
        <LineChart title="Win rate" xs={xs} ys={win.map((o) => o.win_rate)} xLabel="ckpt"
          band={win.map((o) => o.ci95 ?? null)} yRange={pctRange(win.flatMap((o) => [o.win_rate, ...(o.ci95 ?? [])]))} pct />
        {TREND.slice(0, 3).map((k) => trend(k))}
      </div>
      <details className="metric-group" data-testid="more-trends">
        <summary><h3>{TREND.length - 3} more trend charts</h3></summary>
        <div className="charts">{TREND.slice(3).map((k) => trend(k))}</div>
      </details>
    </>
  );
  function trend(k: string) {
    const row = ALL_ROWS.find((r) => r.key === k)!;
    const st = series.map(([, p]) => stat(p, row));
    // y-range spans the CI band so the band never spills over the plot
    const ext = st.flatMap((s) => [...(s?.ci95 ?? []), value(s) ?? NaN]).filter(Number.isFinite);
    const yr: [number, number] | undefined = row.pct ? pctRange(ext) : ext.length ? [Math.min(...ext), Math.max(...ext)] : undefined;
    return <LineChart key={k} title={row.label} xs={xs} ys={st.map((s) => value(s) ?? NaN)} xLabel="ckpt"
      band={st.map((s) => s?.ci95 ?? null)} yRange={yr} pct={row.pct} />;
  }
}

export default function Behavior() {
  const [list, setList] = useState<any[] | null>(null);
  const [file, setFile] = useState<string | null>(null);
  const [prof, setProf] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);
  const [allChips, setAllChips] = useState(false);
  const [allRows, setAllRows] = useState(false);
  const [showCi, setShowCi] = useState(false);
  useEffect(() => {
    getJson("/api/behavior").then((r) => { setList(r.profiles); if (r.profiles.length) setFile(r.profiles[0].file); })
      .catch((e) => { setErr(e.message); setList([]); });
  }, []);
  useEffect(() => {
    if (!file) return;
    setProf(null);
    getJson(`/api/behavior/${encodeURIComponent(file)}`).then(setProf).catch((e) => setErr(e.message));
  }, [file]);
  if (!list) return <p className="muted">Loading…</p>;
  if (!list.length) return <section className="lab-card"><h2>No behavior profiles yet</h2>
    <p className="muted">Run <code>python -m scripts.behavior_report --agents basic … --vs basic</code>. Results appear from results/behavior/*.json.</p>
    {err && <p className="error">{err}</p>}</section>;
  const agents = prof ? (Object.entries(prof.agents) as [string, any][]) : [];
  return (
    <section className="lab-card">
      <h2>Behavior profiles</h2>
      <div className="field-row" role="radiogroup" aria-label="Profile file">
        {/* newest first (server order); the open profile stays visible even when folded */}
        {list.filter((p, i) => allChips || i < CHIPS || p.file === file).map((p) => <button key={p.file} role="radio" aria-checked={file === p.file}
          className={"chip" + (file === p.file ? " on" : "")} onClick={() => setFile(p.file)}
          title={`${p.file}${p.created ? ` · ${String(p.created).replace("T", " ")}` : ""}`}>{p.title ?? p.id}</button>)}
        {list.length > CHIPS && <button className="chip" data-testid="profiles-all" aria-expanded={allChips} onClick={() => setAllChips(!allChips)}>
          {allChips ? "Fewer" : `${list.length - CHIPS} older…`}</button>}
      </div>
      {err && <p className="error">{err}</p>}
      {!prof ? <p className="muted">Loading…</p> : <>
        <p className="muted">Each agent seated against 3× {prof.vs}, seats rotated, {prof.games_per_agent} games per agent.
          Cells: mean per seat-game, or a rate over events; hover a cell for its 95% CI and n. ↑/↓ marks the better direction.</p>
        <div className="lab-toggles">
          <button className={"chip" + (allRows ? " on" : "")} aria-pressed={allRows} data-testid="all-metrics" onClick={() => setAllRows(!allRows)}>All {ALL_ROWS.length} metrics</button>
          <button className={"chip" + (showCi ? " on" : "")} aria-pressed={showCi} data-testid="show-ci" onClick={() => setShowCi(!showCi)}>Show CI and n</button>
          {!allRows && <span className="muted small">Showing {KEY.size} key metrics.</span>}
        </div>
        <div className="t-scroll">
          <table className="table sticky-first" data-testid="behavior-table">
            <thead><tr><th>Metric</th>{agents.map(([name]) => <th key={name} title={name} data-testid="behavior-col">{prof.agent_names?.[name] ?? name}</th>)}</tr></thead>
            <tbody>
              <tr><th>Win rate</th>{agents.map(([name, p]) => <td key={name}>
                {/* DS27-02: older files have no outcome.ci95; ciText skips it instead of crashing the tab */}
                <strong title={ciText({ ci95: p.outcome?.ci95, n: p.outcome?.seat_games ?? p.games }, true)}>{fmt(p.outcome?.win_rate, true)}</strong>
                {showCi && <span className="muted small"> {ciText({ ci95: p.outcome?.ci95, n: p.outcome?.seat_games ?? p.games }, true).replace("95% CI ", "")}</span>}
                {p.outcome && <div><RateBar rate={p.outcome.win_rate} ci={p.outcome.ci95} width={110} /></div>}</td>)}</tr>
              {SECTIONS.map(([title, all]) => [all.filter((r) => allRows || KEY.has(r.key)), title] as const).filter(([rows]) => rows.length).map(([rows, title]) => [
                <tr key={title}><th colSpan={agents.length + 1}><h3>{title}</h3></th></tr>,
                ...rows.map((row) => <tr key={row.key}><th title={row.key}>{row.label}
                  {row.better && <span className="muted" title={`${row.better === "hi" ? "higher" : "lower"} is better`}
                    aria-label={`${row.better === "hi" ? "higher" : "lower"} is better`}> {row.better === "hi" ? "↑" : "↓"}</span>}</th>
                  {agents.map(([name, p]) => <Cell key={name} s={stat(p, row)} row={row} ci={showCi} />)}</tr>),
              ])}
            </tbody>
          </table>
        </div>
        <VpBars agents={agents} />
        <Trends agents={agents} />
      </>}
    </section>
  );
}
