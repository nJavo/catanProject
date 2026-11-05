import { createContext, useContext, useEffect, useMemo, useState } from "react";
import type { Action, Decision } from "../api";
import { describeAction, PLAYER_NAMES, RES_LABEL, seatName, Where } from "../rules";

// Shows exactly what the agent returned from act_with_info. Nothing here is
// inferred: if the agent reports nothing, we say so. Heuristic explanations
// (info.explain) are computed by catan/bots/explain.py from the bot's own
// scoring terms; RL ones (kind "policy_model") by catan/rl/explain.py from the
// network.

const TERM_LABEL: Record<string, string> = {
  vp: "VP", production: "Production × value", diversity: "Diversity", ports: "Ports",
  spot0: "Buildable spot", spot1: "Spot 1 road away", spot2: "Spot 2 roads away", hand: "Hand progress",
  dev_held: "Dev cards held", knights: "Knights played", road_len: "Road length", overflow: "Overflow penalty",
  dev_ev: "Expected dev card value", knight_bonus: "Knight play bonus", monopoly_gain: "Monopoly gain",
  dev_spent: "Card used up", road_building: "Road Building bonus", road_reach: "Best road reach",
  offer_rule: "Offer-first rule", pips_x_scarcity: "Pips × scarcity", spot_diversity: "Types at spot",
  combined_diversity: "Types with 1st settlement", wood_brick: "Wood + brick", wheat_ore: "Wheat + ore",
  port: "Port synergy", expansion: "Expansion room", pips: "Pips", tiebreak: "Tie-break", next_spot: "Best next spot",
  rule: "Rule score", blocked: "Blocked production", victims_bonus: "Victim has cards", own_hex: "Own building",
  desert: "Desert", victim_vp: "Victim VP × 10", victim_cards: "Victim cards", hand_after: "Hand score after",
  surplus: "Surplus over next build", hand_delta: "Hand gain", confirm_cost: "Confirm cost", reach: "Road reach",
  tier: "Priority tier", detail: "Within-tier score", game_over: "Final result", unattributed: "Unattributed",
};
// trade replies already name the seat ("Ember declines the trade"); the headline would say it twice
const REPLY: Record<string, string> = { ACCEPT_TRADE: "accepted the trade", REJECT_TRADE: "declined the trade" };
const WhereCtx = createContext<Where | undefined>(undefined);
// PPO critics predict return; only unshaped runs (win 1, loss 0) put that on a win-probability scale
const CRITIC = "(critic's expected return, not a calibrated win probability)";
const termLabel = (k: string) => TERM_LABEL[k] ?? k.replace(/_/g, " ");
// DS23-04: "Decision rule" keys (catan/bots/explain.py _context). TERM_LABEL's "rule" is a score; here it is prose.
const CTX_LABEL: Record<string, string> = {
  rule: "Rule", hand_delta: "Hand gain", margin: "Required margin", partner_ok: "Proposer not near winning",
  completes_build: "Completes a build", fair_count: "Not a loss in card count", robber_on_me: "Robber on my hex",
  claims_army: "Claims Largest Army", leader_near_win: "Leader within 3 VP of winning",
};
// The bots name resources wood/brick/sheep/wheat/ore; the UI says Lumber/Brick/Wool/Grain/Ore.
const UI_RES: Record<string, string> = { wood: "lumber", sheep: "wool", wheat: "grain" };
export const resText = (t: string) => t.replace(/\b(wood|sheep|wheat)\b/gi, (w) => {
  const m = UI_RES[w.toLowerCase()];
  return w[0] === w[0].toUpperCase() ? m[0].toUpperCase() + m.slice(1) : m;
  // engine text says "seat 2" / "player 1, 3"; people know the seats by name (DS10-05)
}).replace(/\b(?:seat|player) ([0-3](?:, [0-3])*)\b/gi, (_, ids: string) => ids.split(", ").map((i) => PLAYER_NAMES[Number(i)]).join(", "));
const TEXT_KEYS = new Set(["text", "summary", "label", "skip_reason", "rule", "note"]);
/** Copy of an explanation with display text mapped to UI resource names; keys and ids untouched. */
function forDisplay(x: any, key = "", places?: (t: string) => string): any {
  if (typeof x === "string") return TEXT_KEYS.has(key) ? resText(places ? places(x) : x) : x;
  if (Array.isArray(x)) return x.map((v) => forDisplay(v, key, places));
  if (x && typeof x === "object") return Object.fromEntries(Object.entries(x).map(([k, v]) => [k, forDisplay(v, k, places)]));
  return x;
}
// DS25: the Python sentences name board moves by engine id ("Build road on edge 28") while the candidate
// table says "Road along Grain 10, Wool 11"; swap each candidate's engine label for the table's name.
const BOARD_MOVES = new Set(["BUILD_ROAD", "BUILD_SETTLEMENT", "BUILD_CITY", "MOVE_ROBBER"]);
function placeNamer(cands: any, where?: Where): ((t: string) => string) | undefined {
  if (!where || !Array.isArray(cands)) return undefined;
  const m = new Map<string, string>(cands.filter((c: any) => c?.label && BOARD_MOVES.has(c.action?.type))
    .map((c: any) => [c.label, describeAction(c.action, where)]));
  if (!m.size) return undefined;
  // longest first, and no digit after, so "edge 2" never matches inside "edge 28"
  const re = new RegExp(`(${[...m.keys()].sort((a, b) => b.length - a.length).map((l) => l.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|")})(?!\\d)`, "g");
  return (t) => t.replace(re, (l) => m.get(l) ?? l);
}
const RES_KEYS = ["wood", "brick", "sheep", "wheat", "ore"];
const resName = (k: string) => RES_LABEL[RES_KEYS.indexOf(k)] ?? k;
const num = (v: number, d = 2) => (v < 0 ? "−" : "") + Math.abs(v).toFixed(d);
const signed = (v: number, d = 2) => (v > 0 ? "+" : v < 0 ? "−" : "±") + Math.abs(v).toFixed(d);
const cardList = (c: number[]) => c.flatMap((n, i) => Array(n).fill(RES_LABEL[i])).join(" + ") || "nothing";

interface Cand {
  action: Action; label: string; score: number; heuristic?: number; delta?: number;
  terms: Record<string, number>; deltas?: Record<string, number>; cost?: number[];
  affordable_after?: string[]; detail?: Record<string, any>; text?: string; refs?: Refs; why?: Record<string, any>;
  rollout?: { mean: number; sd: number; n: number; ci95: number; diff_vs_chosen: number; diff_sd: number };
}

// nego agents (the Champion): a trade decision came from the learned head, not from the heuristic rule in info.explain.
function NegoHead({ n }: { n: any }) {
  const where = useContext(WhereCtx);
  const teacher = n.options?.[n.teacher];
  return (
    <section className="di-why" data-testid="nego-head">
      <span className="field-label">{n.learned ? "Decided by the learned negotiation model" : "Negotiation option (no checkpoint: the heuristic teacher picks)"}</span>
      {n.learned && <p className="small">{n.chosen === n.teacher ? "The heuristic rule below agrees with this choice."
        : <>The heuristic rule below would have chosen <strong>{teacher ? describeAction(teacher.action, where) : `option ${n.teacher}`}</strong>. It is shown for reference only.</>}</p>}
      {n.learned && <Bars pct entries={n.options.filter((o: any, i: number) => o.p >= 0.001 || i === n.teacher).map((o: any) => [describeAction(o.action, where), o.p] as [string, number])} label="Model output: probability of each option" />}
    </section>
  );
}

function Bars({ entries, label, pct = false }: { entries: [string, number][]; label: string; pct?: boolean }) {
  const top = entries.sort((a, b) => b[1] - a[1]).slice(0, 10);
  const max = Math.max(...top.map((e) => Math.abs(e[1])), 1e-9);
  return (
    <div className="bars">
      <span className="field-label">{label}</span>
      {top.map(([k, v]) => (
        <div key={k} className="bar-row" title={`${k}: ${v}`}>
          <span className="bar-key">{k}</span>
          <span className="bar-track"><span className="bar-fill" style={{ width: `${(Math.abs(v) / max) * 100}%` }} /></span>
          <span className="bar-val">{pct ? `${(v * 100).toFixed(1)}%` : Math.abs(v) < 1 ? v.toFixed(3) : v.toFixed(2)}</span>
        </div>
      ))}
    </div>
  );
}

const asEntries = (x: any): [string, number][] | null => {
  if (x && typeof x === "object" && !Array.isArray(x)) {
    const e = Object.entries(x).filter(([, v]) => typeof v === "number") as [string, number][];
    return e.length ? e : null;
  }
  if (Array.isArray(x) && x.every((v) => typeof v === "number")) return x.map((v, i) => [String(i), v]);
  return null;
};

/** Diverging bars, one row per term: the selected candidate (brass) over the comparison (mist). */
function TermBars({ a, b, mode, base }: { a: Cand; b?: Cand; mode: "delta" | "abs"; base?: Record<string, number> }) {
  const pick = (c: Cand) => (mode === "delta" && c.deltas ? c.deltas : c.terms);
  const ta = pick(a), tb = b ? pick(b) : {};
  const keys = [...new Set([...Object.keys(ta), ...Object.keys(tb)])]
    .sort((x, y) => Math.max(Math.abs(ta[y] ?? 0), Math.abs(tb[y] ?? 0)) - Math.max(Math.abs(ta[x] ?? 0), Math.abs(tb[x] ?? 0)));
  const max = Math.max(1e-9, ...keys.flatMap((k) => [Math.abs(ta[k] ?? 0), Math.abs(tb[k] ?? 0)]));
  const bar = (v: number, cls: string) => (
    <span className={"di-bar " + cls} style={{ left: v < 0 ? `${50 - (50 * -v) / max}%` : "50%", width: `${(50 * Math.abs(v)) / max}%` }} />
  );
  if (!keys.length) return <p className="muted small">No term changes versus the current position.</p>;
  return (
    <div className="di-terms" data-testid="term-bars">
      {keys.map((k) => (
        <div key={k} className="di-term" data-term={k} title={base && k in base ? `Current position: ${num(base[k], 3)}` : undefined}>
          <span className="di-term-key">{termLabel(k)}</span>
          <span className="di-track">
            <span className="di-axis" />
            {bar(ta[k] ?? 0, "a")}
            {b && bar(tb[k] ?? 0, "b")}
          </span>
          <span className="di-term-val">
            <span className="a">{signed(ta[k] ?? 0)}</span>
            {b && <span className="b">{signed(tb[k] ?? 0)}</span>}
          </span>
        </div>
      ))}
    </div>
  );
}

const TERM_HELP: Record<string, string> = {
  vp: "Victory points you hold, including hidden VP cards. One point is worth 1.0 in this model.",
  production: "Expected pips you collect per roll for each resource (cities count twice, the robbed hex counts zero), weighted by how valuable that resource is to you, × 0.045.",
  diversity: "0.12 for every resource type you produce at all.",
  ports: "3:1 port: flat 0.12. 2:1 port: 0.012 per pip you produce of that resource.",
  spot0: "Best free settlement spot you can build on right now, valued by its pips × resource values, × 0.08.",
  spot1: "Best free spot one road away, × 0.04.",
  spot2: "Best free spot two roads away, × 0.015.",
  hand: "How close your hand is to the builds you are working toward: best goal weight / (1 + 1.3 × value of missing cards), plus 0.4 × the second goal, plus a small per-card value; × 0.3.",
  dev_held: "0.3 per unplayed dev card that is not a VP card.",
  knights: "0.08 per knight played, up to 4.",
  road_len: "0.03 per road segment of your longest road, up to 6, unless you already hold Longest Road.",
  overflow: "Penalty per card above the discard limit (weight 0 by default).",
  dev_ev: "A dev card you buy is unknown, so it is valued at a flat expected 0.42.",
  knight_bonus: "Flat 0.4 for playing a knight when the robber is on you, it claims Largest Army, or the leader is close to winning.",
  monopoly_gain: "Hand progress gained from taking every opponent's cards of one resource (totals are public).",
  dev_spent: "A played card no longer counts as a held dev card.",
  road_building: "Flat 0.25 for two free roads.", road_reach: "0.01 × the best road's reach score.",
  offer_rule: "Rule: try a domestic trade before ending the turn or trading with the bank.",
  game_over: "A finished game: 100 if you win, else your VP.",
};

export type Refs = { vertices?: number[]; edges?: number[]; hexes?: number[] };
export type Highlight = Action | { refs: Refs } | null;

function Term({ k }: { k: string }) {
  return <span className="di-termname" title={TERM_HELP[k]}>{termLabel(k)}</span>;
}

function Explanation({ ex, onHover }: { ex: any; onHover?: (h: Highlight) => void }) {
  const where = useContext(WhereCtx);
  const cands: Cand[] = ex.candidates ?? [];
  const [sel, setSel] = useState(0);
  const [mode, setMode] = useState<"delta" | "abs">("delta");
  useEffect(() => setSel(0), [ex]);
  const value = ex.kind === "value_model" && ex.phase === "MAIN";
  const search = ex.search;
  const ran = search?.ran;
  const a: any = cands[Math.min(sel, cands.length - 1)];
  const b: any = sel === 0 ? cands[1] : cands[0];
  const lit = (refs?: Refs) => ({ onMouseEnter: () => refs && onHover?.({ refs }), onMouseLeave: () => onHover?.(null) });
  const why = (c: any, k: string) => c?.why?.[k];
  const cf = ex.counterfactuals;
  return (
    <div className="di" data-testid="explanation">
      <section className="di-why" data-testid="why">
        <span className="field-label">Why this move</span>
        <p className="di-summary" data-testid="explain-summary">{ex.summary}</p>
        <span className="di-source">{ex.summary_source}; every sentence here is filled in from the numbers the bot used</span>
        {ex.decisive?.length > 0 && (
          <>
            <span className="di-sub">Why {describeAction(cands[0].action, where)} over {describeAction(cands[1].action, where)}</span>
            <ol className="di-decisive" data-testid="decisive">
              {ex.decisive.map((d: any) => {
                const p = d.side === "a" ? why(cands[0], d.term) : d.side === "b" ? why(cands[1], d.term) : null;
                return (
                  <li key={d.term} {...lit(p?.refs)} data-term={d.term}>
                    <span className="di-dhead"><Term k={d.term} /> <strong className={d.diff >= 0 ? "pos" : "neg"}>{signed(d.diff)}</strong>
                      <span className="muted"> ({signed(d.a)} vs {signed(d.b)})</span></span>
                    {p && <span className="di-ptext">{p.text}</span>}
                    {d.side === "a" && why(cands[1], d.term) && (
                      <span className="di-ptext muted" {...lit(why(cands[1], d.term).refs)}>Runner-up: {why(cands[1], d.term).text}</span>
                    )}
                  </li>
                );
              })}
            </ol>
          </>
        )}
        {!value && cands[0]?.text && <p className="di-ptext" {...lit(cands[0].refs)}>{cands[0].text}</p>}
        {ex.context?.text && <p className="di-ptext">{ex.context.text}</p>}
        {search && <p className="small muted">{ran ? `Search ran ${a?.rollout?.n ?? search.rollouts} rollouts per candidate (details below).` : `Search skipped: ${search.skip_reason}; the heuristic ranking decided.`}</p>}
        {cf?.items?.length > 0 && (
          <div className="di-cf" data-testid="counterfactuals">
            <span className="di-sub">Counterfactuals <span className="muted">(re-scored on a modified copy; heuristic ranking, no rollouts)</span></span>
            <ul>{cf.items.map((it: any, i: number) => <li key={i} {...lit(it.vertex !== undefined ? { vertices: [it.vertex] } : undefined)}>{it.text}</li>)}</ul>
          </div>
        )}
      </section>

      <table className="di-cands" data-testid="candidates">
        <thead>
          <tr><th>#</th><th>Candidate</th><th title="Heuristic score">Score</th>{value && <th title="Heuristic score minus ending the turn now">Δ end</th>}{ran && <th title="Rollout mean ± 95% CI">Rollout</th>}</tr>
        </thead>
        <tbody onMouseLeave={() => onHover?.(null)}>
          {cands.map((c, i) => (
            <tr key={i} className={(i === 0 ? "chosen " : "") + (i === sel ? "on" : "")} onClick={() => setSel(i)}
              onMouseEnter={() => onHover?.(c.action)} tabIndex={0} aria-selected={i === sel}
              onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); setSel(i); } }}>
              <td>{i === 0 ? "✓" : i + 1}</td>
              <td className="di-label" title={c.label}>{describeAction(c.action, where)}</td>
              <td>{num(c.heuristic ?? c.score)}</td>
              {value && <td className={(c.delta ?? 0) >= 0 ? "pos" : "neg"}>{signed(c.delta ?? 0)}</td>}
              {ran && <td>{c.rollout ? <>{num(c.rollout.mean)}<small> ±{c.rollout.ci95.toFixed(2)}</small></> : <span title="Not rolled out: search only simulates the heuristic's top candidates" data-testid="no-rollout">–</span>}</td>}
            </tr>
          ))}
        </tbody>
      </table>

      {value && a && (
        <section className="di-block" data-testid="provenance">
          <span className="field-label">How each number was computed: {describeAction(a.action, where)}{sel === 0 ? " (chosen)" : ""}</span>
          {a.why ? (
            Object.keys(a.why).length ? Object.entries(a.why).map(([k, p]: [string, any]) => (
              <details key={k} className="di-prov" {...lit(p.refs)} data-term={k}>
                <summary><Term k={k} /> <span className={(a.deltas?.[k] ?? 0) >= 0 ? "pos" : "neg"}>{signed(a.deltas?.[k] ?? 0)}</span>
                  <span className="muted"> → {num(a.terms[k] ?? 0)}</span></summary>
                <p>{p.text}</p>
              </details>
            )) : <p className="small muted">Nothing changes versus the current position.</p>
          ) : <p className="small muted">Provenance is included for the chosen action and the runner-up; select one of them.</p>}
          {ex.baseline_why && (
            <details className="di-prov di-base">
              <summary>Current position, term by term <span className="muted">(total {num(ex.baseline)})</span></summary>
              {Object.entries(ex.baseline_why).map(([k, p]: [string, any]) => (
                <p key={k} {...lit(p.refs)}><Term k={k} /> <strong>{num(ex.baseline_terms[k] ?? 0)}</strong>: {p.text}</p>
              ))}
            </details>
          )}
        </section>
      )}
      {!value && a?.text && sel > 0 && <p className="di-ptext" {...lit(a.refs)}>{a.text}</p>}

      <details className="di-compare" open={!value}>
        <summary className="di-compare-head">
          <span className="di-key a">{describeAction(a.action, where)}{sel === 0 && " (chosen)"}</span>
          {b && <span className="di-key b">vs {describeAction(b.action, where)}{sel !== 0 && " (chosen)"}</span>}
        </summary>
        {value && (
          <div className="segmented small" role="group" aria-label="Term view">
            <button className={mode === "delta" ? "on" : ""} onClick={() => setMode("delta")}>Change vs now</button>
            <button className={mode === "abs" ? "on" : ""} onClick={() => setMode("abs")}>Total terms</button>
          </div>
        )}
        <TermBars a={a} b={b} mode={value ? mode : "abs"} base={ex.baseline_terms} />
        {a.cost && <p className="small muted">Costs {cardList(a.cost)}</p>}
        {a.affordable_after && <p className="small muted">Affordable afterwards: {a.affordable_after.join(", ") || "nothing"}</p>}
        {a.rollout && b?.rollout && sel === 0 && (
          <p className="small muted">Paired rollout gap vs {describeAction(b.action, where)}: {signed(-b.rollout.diff_vs_chosen)} (sd {b.rollout.diff_sd.toFixed(2)}, n={b.rollout.n})</p>
        )}
        {typeof a.detail?.hand_gain_if_accepted === "number" && (
          <p className="small muted">Offer rule: ranked just above the best other action ({num(a.detail.best_other)}). Hand gain if accepted {signed(a.detail.hand_gain_if_accepted)}.</p>
        )}
      </details>

      {search && (
        <div className="di-block" data-testid="search-info">
          <span className="field-label">Search</span>
          {ran
            ? <p className="small">{a?.rollout?.n ?? search.rollouts} rollouts per candidate. Leaf: <code>{search.leaf}</code>. Policy: {search.policy}.</p>
            : <p className="small">Skipped: {search.skip_reason}. The heuristic ranking decided.</p>}
        </div>
      )}

      {ex.context && (
        <div className="di-block">
          <span className="field-label">Decision rule</span>
          {/* DS29-04: the rule is the block's subject; a "Rule" label straight under "Decision rule" said it twice */}
          {typeof ex.context.rule === "string" && <p className="small" data-testid="di-rule">{ex.context.rule}</p>}
          <dl className="di-kv">
            {Object.entries(ex.context).filter(([k]) => k !== "text" && !(k === "rule" && typeof ex.context.rule === "string")).map(([k, v]) => (
              // a sentence-long value right-aligned in a narrow column was hard to read; raw numbers lost the − used above
              <div key={k} className={typeof v === "string" && v.length > 24 ? "prose" : undefined}><dt>{CTX_LABEL[k] ?? k.replace(/_/g, " ")}</dt>
                <dd>{typeof v === "boolean" ? (v ? "yes" : "no") : typeof v === "number" ? num(v, Number.isInteger(v) ? 0 : 2) : String(v)}</dd></div>
            ))}
          </dl>
        </div>
      )}

      {ex.goals && (
        <div className="di-block">
          <span className="field-label">Goals (weight, missing cards)</span>
          <ul className="di-goals">
            {ex.goals.map((g: any) => (
              <li key={g.name} className={ex.affordable_now?.includes(g.name) ? "ok" : ""}>
                <strong>{g.name}</strong> ×{g.weight} · {g.missing.some((n: number) => n) ? `missing ${cardList(g.missing)}` : "affordable now"}
              </li>
            ))}
          </ul>
        </div>
      )}

      {ex.resource_values && (
        <div className="di-block">
          <span className="field-label">Resource values = scarcity × phase × own production</span>
          <table className="di-rv">
            <tbody>
              {ex.resource_values.map((r: any) => (
                <tr key={r.res} title={r.text}><th>{resName(r.res)}</th><td><strong>{r.value.toFixed(2)}</strong></td>
                  <td className="muted">{r.scarcity.toFixed(2)} × {r.phase.toFixed(2)} × {r.own_factor.toFixed(2)}</td></tr>
              ))}
            </tbody>
          </table>
          {ex.resource_values[0]?.text && (
            <details className="di-prov"><summary>Why these values</summary>{ex.resource_values.map((r: any) => <p key={r.res}>{r.text}</p>)}</details>
          )}
        </div>
      )}
    </div>
  );
}

type Ent = { id: number; logit: number; value: number };
const ENT_KINDS: [keyof Refs, string, string][] = [["hexes", "Hexes", "Hex"], ["vertices", "Vertices", "Vertex"], ["edges", "Edges", "Edge"]];

/** Top board entities by integrated-gradient attribution; hovering highlights them on the board. */
export function EntityAttribution({ entities, head, onRefs }: {
  entities: Record<string, Ent[]>; head: "logit" | "value"; onRefs?: (r: Refs | null) => void;
}) {
  const all: Refs = Object.fromEntries(ENT_KINDS.map(([k]) => [k, (entities[k] ?? []).map((e) => e.id)]));
  const max = Math.max(1e-9, ...ENT_KINDS.flatMap(([k]) => (entities[k] ?? []).map((e) => Math.abs(e[head]))));
  if (!ENT_KINDS.some(([k]) => entities[k]?.length)) return <p className="muted small">No board entity carries attribution here.</p>;
  return (
    <div className="di-ents" data-testid="entity-attribution" onMouseEnter={() => onRefs?.(all)} onMouseLeave={() => onRefs?.(null)}>
      <span className="di-sub">By board entity <span className="muted">(hover to highlight on the board)</span></span>
      {ENT_KINDS.filter(([k]) => entities[k]?.length).map(([k, label, one]) => (
        <div key={k} className="di-ent-group">
          <span className="field-label">{label}</span>
          {entities[k].map((e) => {
            const v = e[head];
            return (
              <div key={e.id} className="di-term" data-entity={`${k}:${e.id}`}
                onMouseEnter={(ev) => { ev.stopPropagation(); onRefs?.({ [k]: [e.id] }); }} onMouseLeave={() => onRefs?.(all)}>
                <span className="di-term-key">{one} {e.id}</span>
                <span className="di-track"><span className="di-axis" />
                  <span className="di-bar a" style={{ top: 4, left: v < 0 ? `${50 - (50 * -v) / max}%` : "50%", width: `${(50 * Math.abs(v)) / max}%` }} /></span>
                <span className="di-term-val"><span className="a">{signed(v)}</span></span>
              </div>
            );
          })}
        </div>
      ))}
    </div>
  );
}

/** RL policy view: every number is a network output or derived from one (catan/rl/explain.py). */
function RlExplanation({ ex, onHover }: { ex: any; onHover?: (h: Highlight) => void }) {
  const where = useContext(WhereCtx);
  const [head, setHead] = useState<"logit" | "value">("logit");
  const at = ex.attribution;
  const bars: Cand = { action: ex.candidates[0].action, label: "", score: 0,
    terms: Object.fromEntries(at.blocks.map((b: any) => [b.block, b[head]])) };
  const h = at[head];
  return (
    <div className="di" data-testid="explanation">
      <p className="di-summary" data-testid="explain-summary">{ex.summary}</p>
      <span className="di-source">{ex.summary_source}</span>
      <span className="field-label">Model outputs: policy probability and value-network estimate after each action</span>
      <table className="di-cands" data-testid="candidates">
        <thead><tr><th>#</th><th>Candidate</th><th title="Policy probability of this action">Prob</th><th title={`Value head after the action, mean ± sd over ${ex.value_probe.determinizations} determinizations`}>Value after</th><th title="Value after minus value now">Δ value</th></tr></thead>
        <tbody onMouseLeave={() => onHover?.(null)}>
          {ex.candidates.map((c: any, i: number) => (
            <tr key={i} className={i === 0 ? "chosen on" : ""} onMouseEnter={() => onHover?.(c.action)}>
              <td>{i === 0 ? "✓" : i + 1}</td>
              <td className="di-label" title={c.label}>{describeAction(c.action, where)}</td>
              <td>{(c.prob * 100).toFixed(1)}%</td>
              <td>{num(c.value_after)}<small> ±{c.value_sd.toFixed(2)}</small></td>
              <td className={c.delta >= 0 ? "pos" : "neg"}>{signed(c.delta)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="small">Model output: value estimate now <strong>{num(ex.value, 3)}</strong> <span className="muted">{CRITIC}</span></p>
      <div className="di-compare" data-testid="attribution">
        <span className="field-label">{at.label}</span>
        <div className="segmented small" role="group" aria-label="Attribution target">
          <button className={head === "logit" ? "on" : ""} onClick={() => setHead("logit")}>Chosen-action logit</button>
          <button className={head === "value" ? "on" : ""} onClick={() => setHead("value")}>Value</button>
        </div>
        <TermBars a={bars} mode="abs" />
        {at.entities && <EntityAttribution entities={at.entities} head={head} onRefs={(r) => onHover?.(r ? { refs: r } : null)} />}
        <p className="small muted">
          Baseline: {at.baseline}. {head === "logit" ? "Logit" : "Value"} {num(h.at_baseline)} → {num(h.at_obs)}; attributions sum to {signed(h.sum)} (gap {signed(h.completeness_gap, 3)}, {at.steps} steps).
        </p>
      </div>
      <ul className="di-caveats">{ex.caveats.map((c: string) => <li key={c}>{c}</li>)}</ul>
    </div>
  );
}

export default function DecisionInspector({ decision, onHover, where }: { decision: Decision | null | undefined; onHover?: (h: Highlight) => void; where?: Where }) {
  const raw = decision?.info?.explain;
  const ex = useMemo(() => raw && forDisplay(raw, "", placeNamer(raw.candidates, where)), [raw, where?.topo, where?.layout]); // not `where`: a new object every render
  if (!decision) return <section className="inspector"><h3>Decision inspector</h3><p className="muted">No AI decision at this point.</p></section>;
  const { info } = decision;
  const known = new Set(["probs", "probabilities", "policy", "scores", "value", "values", "explain", "top", "score_kind", "nego"]);
  const probs = asEntries(info.probs ?? info.probabilities ?? info.policy);
  const scores = asEntries(info.scores);
  const value = typeof info.value === "number" ? info.value : null;
  const searched = info.source === "exit_search" || info.source === "value_search";
  const rest = Object.keys(info).filter((k) => !known.has(k) && k !== "agent_error" && k !== "fallback");
  const any = probs || scores || value !== null || info.explain || info.top || rest.length > 0;
  return (
    <WhereCtx.Provider value={where}>
    <section className="inspector" data-testid="decision-inspector">
      <h3>Decision inspector</h3>
      <p><strong>{seatName(decision.seat)}</strong> ({decision.agent}) {REPLY[decision.action.type] ?? <>chose <strong>{describeAction(decision.action, where)}</strong></>}</p>
      {info.agent_error && <p className="error">Agent failed: {String(info.agent_error)}. The server played a random legal move instead.</p>}
      {info.explain_error && <p className="error">Explanation failed: {String(info.explain_error)}. The decision itself is unaffected.</p>}
      {info.nego && <NegoHead n={info.nego} />}
      {ex && (ex.kind === "policy_model"
        ? <RlExplanation ex={ex} onHover={onHover} />
        : <Explanation ex={ex} onHover={onHover} />)}
      {!info.explain && info.top && <Bars entries={info.top.map((t: any) => [describeAction(t.action, where), t.score])} label="Heuristic scores (reported by the bot)" />}
      {probs && !info.explain && <Bars entries={probs} label="Model output: policy probabilities (top-k)" />}
      {/* value_trade relabels its core's "forced" as "value_search", so a search source without sims or value is forced too */}
      {(info.source === "forced" || (searched && info.sims == null && value === null)) && <p className="muted" data-testid="search-summary">Only one move to choose from, so the agent didn't search.</p>}
      {searched && info.sims != null && (
        <p data-testid="search-summary">Picked by search: <strong>{info.sims}</strong> simulations spread over <strong>{info.candidates}</strong> candidate moves sampled from the policy{info.source === "value_search" ? ", with leaf positions scored by the learned value net" : ""}. <span className="muted small">The agent doesn't report per-candidate values.</span></p>
      )}
      {info.source === "nego_offer" && (
        <p data-testid="nego-offer">Offered by the negotiation model. The core policy chose <strong>{info.replaced ? describeAction(info.replaced, where) : "another move"}</strong>, and the trade head replaced it with this offer{typeof info.nego_probs?.[info.chosen] === "number" ? ` (model output: ${(info.nego_probs[info.chosen] * 100).toFixed(1)}% for this offer)` : ""}.</p>
      )}
      {value !== null && !info.explain && (searched
        // search agents report the policy prior's raw value head; with reward shaping it is not on a win-probability scale
        ? <p>Model output: {info.source === "value_search" ? "policy prior's " : ""}value head <strong>{value.toFixed(3)}</strong> <span className="muted small">(raw network output, not a calibrated win probability)</span></p>
        : <p>Model output: value estimate <strong>{value.toFixed(3)}</strong> <span className="muted small">{CRITIC}</span></p>)}
      {typeof info.chosen_prob === "number" && !info.explain && <p className="small muted">Model output: probability of the chosen action {(info.chosen_prob * 100).toFixed(1)}%</p>}
      {scores && <Bars entries={scores} label="Heuristic scores (reported by the bot)" />}
      {!any && <p className="muted">This agent reports no diagnostics.</p>}
      {Object.keys(info).length > 0 && (
        <details className="di-raw">
          <summary data-testid="raw-toggle">Raw diagnostics</summary>
          <pre>{JSON.stringify(info, null, 1)}</pre>
        </details>
      )}
    </section>
    </WhereCtx.Provider>
  );
}
