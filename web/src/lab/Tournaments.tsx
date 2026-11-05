import { useEffect, useMemo, useState } from "react";
import { api } from "../api";
import { PLAYER_NAMES, shortSpec } from "../rules";
import { CrestBadge } from "../Setup";
import { Histogram, LineChart, RateBar } from "./Charts";
import "./tournaments.css";

// Renders results/tournaments/*.json (schema: catan/eval/tournament.py).
// Older files may lack any field, so every accessor tolerates missing data.

type Ci = [number, number];
type Pair = { rate: number; ci: Ci | null; n: number | null; ahead: number | null; ties: number | null; p: number | null };

const isNum = (x: any): x is number => typeof x === "number" && Number.isFinite(x);
const ci2 = (c: any): Ci | null => (Array.isArray(c) && c.length === 2 && isNum(c[0]) && isNum(c[1]) ? [c[0], c[1]] : null);
const DASH = "—";
export const pct = (x: any, d = 1) => (isNum(x) ? `${(x * 100).toFixed(d)}%` : DASH);
const pctCi = (c: Ci | null) => (c ? `[${(c[0] * 100).toFixed(1)}, ${(c[1] * 100).toFixed(1)}]` : "");
const fix = (x: any, d = 2) => (isNum(x) ? x.toFixed(d) : DASH);
const int = (x: any) => (isNum(x) ? String(Math.round(x)) : DASH);
/** Never a percentage: "p=0.031", "p=3.2e-5", "p<1e-6". */
export const fmtP = (p: any) => (!isNum(p) ? DASH : p < 1e-6 ? "p<1e-6" : p < 1e-3 ? `p=${p.toExponential(1)}` : `p=${+p.toPrecision(2)}`);
const fmtMs = (sec: any) => {
  if (!isNum(sec)) return DASH;
  const ms = sec * 1000;
  return ms < 0.01 ? "<0.01" : ms < 10 ? ms.toFixed(2) : ms.toFixed(0);
};
const fmtSec = (s: any) => (!isNum(s) ? DASH : s < 90 ? `${s.toFixed(1)} s` : s < 5400 ? `${(s / 60).toFixed(1)} min` : `${(s / 3600).toFixed(1)} h`);

const ckptStep = (spec: string) => {
  const m = spec.match(/(?:ckpt|step|checkpoint)_0*(\d+)\.\w+/);
  return m ? Number(m[1]) : null;
};

const avgVp = (s: any): { mean: number; ci: Ci | null } | null =>
  isNum(s?.avg_vp) ? { mean: s.avg_vp, ci: null } : isNum(s?.avg_vp?.mean) ? { mean: s.avg_vp.mean, ci: ci2(s.avg_vp.ci95) } : null;
const winCi = (s: any): Ci | null => {
  const c = ci2(s?.ci95);
  if (c) return c;
  if (isNum(s?.ci95) && isNum(s?.win_rate)) return [Math.max(0, s.win_rate - s.ci95), Math.min(1, s.win_rate + s.ci95)];
  return null;
};
const labelsOf = (t: any): string[] => (Array.isArray(t.agents) && t.agents.length ? t.agents : Object.keys(t.per_agent ?? {}));
/** Seats each entrant holds in one game (3 for "basic" in rl + 3x basic). */
const copiesOf = (t: any): Record<string, number> => {
  const lu: string[] | undefined = Array.isArray(t.lineups?.[0]) ? t.lineups[0] : undefined;
  const out: Record<string, number> = {};
  labelsOf(t).forEach((l) => { out[l] = lu ? lu.filter((x) => x === l).length || 1 : 1; });
  return out;
};
const nPlayers = (t: any) => t.config?.num_players ?? t.lineups?.[0]?.length ?? null;
const lineupDesc = (t: any, except?: string) => {
  const cp = copiesOf(t);
  // a 28-agent league listed every spec over nine lines (DS33-01); the table below names them
  if (!except && labelsOf(t).length > 6) return `${labelsOf(t).length} agents`;
  return labelsOf(t).filter((l) => l !== except).map((l) => (cp[l] > 1 ? `${cp[l]}×${shortSpec(l)}` : shortSpec(l))).join(" vs ");
};
const gamesOf = (t: any) => (isNum(t.games) ? t.games : Array.isArray(t.games) ? t.games.length : isNum(t.global?.games) ? t.global.games : null);
const dateOf = (t: any) => (typeof t.created === "string" ? t.created.replace("T", " ").slice(0, 16) : "");

/** Pairwise lookup for both the current ("A vs B": {...}) and the legacy nested ({A: {B: 0.7}}) layouts. */
function pairOf(pw: any, a: string, b: string): Pair | null {
  if (!pw || typeof pw !== "object") return null;
  const x = pw[`${a} vs ${b}`];
  if (x && isNum(x.ahead_rate)) return { rate: x.ahead_rate, ci: ci2(x.ci95), n: isNum(x.games) ? x.games : null, ahead: isNum(x.ahead) ? x.ahead : null, ties: isNum(x.ties) ? x.ties : null, p: isNum(x.sign_test_p) ? x.sign_test_p : null };
  const y = pw[a]?.[b];
  const r = isNum(y) ? y : isNum(y?.win_rate) ? y.win_rate : isNum(y?.ahead_rate) ? y.ahead_rate : null;
  return r === null ? null : { rate: r, ci: ci2(y?.ci95), n: isNum(y?.games) ? y.games : null, ahead: null, ties: null, p: isNum(y?.sign_test_p) ? y.sign_test_p : null };
}

// ------------------------------------------------------------------ pieces

// <wbr> after commas: long specs have no spaces, so they pushed BT-Elo off screen at 1280
// DS24-03: also after ":" and "/", since "arch_rl:gnnrl_cont_shaped/ckpt_00017," alone still pushed it off
function breakable(spec: string) {
  return spec.split(/(?<=[,:/])/).flatMap((part, i) => (i ? [<wbr key={i} />, part] : [part]));
}

function AgentName({ label, copies }: { label: string; copies?: number }) {
  return (
    <span className="t-agent" title={label}>
      <strong>{breakable(shortSpec(label))}</strong>
      {copies && copies > 1 ? <span className="t-badge" title={`${copies} copies per game, aggregated under one name`}>{copies} seats</span> : null}
    </span>
  );
}

/** Detail behind a click: the standings and pairwise table are the headline, the rest waits here. */
function More({ title, note, children }: { title: string; note?: string; children: React.ReactNode }) {
  return (
    <details className="metric-group t-more" data-section={title}>
      <summary><h3>{title}</h3>{note && <span className="muted small">{note}</span>}</summary>
      <div className="t-section">{children}</div>
    </details>
  );
}

/** One line of what matters to read the numbers (games, boards, rules); the full record sits under "Run details". */
function MetaLine({ t }: { t: any }) {
  const seeds: number[] = Array.isArray(t.seeds) ? t.seeds : [];
  const failed = Array.isArray(t.failed_games) ? t.failed_games.length : 0;
  const bits = [isNum(gamesOf(t)) && `${gamesOf(t)} games`, seeds.length > 0 && `${seeds.length} board${seeds.length > 1 ? "s" : ""}`,
    t.rotation && `${t.rotation} seats`, t.config && `${t.config.num_players ?? "?"} players to ${t.config.vp_to_win ?? "?"} VP`,
    isNum(t.elapsed_sec) && fmtSec(t.elapsed_sec)].filter(Boolean);
  return <p className="muted t-meta" data-testid="t-meta">{bits.join(" · ")}{failed > 0 && <span className="error"> · {failed} failed</span>}</p>;
}

function Summary({ t }: { t: any }) {
  const g = t.global ?? {};
  const seeds: number[] = Array.isArray(t.seeds) ? t.seeds : [];
  const lus = Array.isArray(t.lineups) ? t.lineups.length : null;
  const items: [string, React.ReactNode][] = [
    ["Games", <>{int(gamesOf(t))}{Array.isArray(t.failed_games) && t.failed_games.length ? <span className="error"> · {t.failed_games.length} failed</span> : null}</>],
    ["Seed set", seeds.length ? <>{isNum(t.seed) ? `--seed ${t.seed} · ` : ""}{seeds.length > 1 ? `${seeds[0]}–${seeds[seeds.length - 1]}` : seeds[0]} <span className="muted">({seeds.length} boards)</span></> : DASH],
    ["Rotation", t.rotation ? <>{t.rotation}{lus ? <span className="muted"> · {lus} seating{lus > 1 ? "s" : ""}</span> : null}</> : DASH],
    ["Truncated", pct(g.truncation_rate)],
    ["Avg turns", isNum(g.turns?.mean) ? <>{g.turns.mean.toFixed(1)}{isNum(g.turns.sd) ? <span className="muted"> sd {g.turns.sd.toFixed(1)}</span> : null}{isNum(g.median_turns) ? <span className="muted"> · median {g.median_turns}</span> : null}</> : DASH],
    ["Elapsed", <>{fmtSec(t.elapsed_sec)}{isNum(t.workers) ? <span className="muted"> · {t.workers} workers</span> : null}</>],
    ["Git commit", typeof t.git_commit === "string" ? <code title={t.git_commit}>{t.git_commit.slice(0, 8)}</code> : DASH],
    ["Rules", t.config ? `${t.config.num_players ?? "?"} players · ${t.config.vp_to_win ?? "?"} VP · max ${t.config.max_turns ?? "?"} turns` : DASH],
  ];
  return (
    <dl className="t-summary">
      {items.map(([k, v]) => <div key={k}><dt>{k}</dt><dd>{v}</dd></div>)}
    </dl>
  );
}

function Standings({ t, order }: { t: any; order: string[] }) {
  const per = t.per_agent ?? {};
  const cp = copiesOf(t);
  const anyMs = order.some((l) => isNum(per[l]?.sec_per_move));
  // DS34-04: say which column sets the row order (same key as `order` below: BT-Elo when rated, else win rate)
  const byElo = order.some((l) => isNum(t.ratings?.[l]?.elo));
  const sortedBy = (on: boolean) => on ? { "aria-sort": "descending" as const, title: "Rows are sorted by this column, highest first", "data-testid": "sort-key" } : {};
  const anyBehaviour = order.some((l) => per[l]?.trade || isNum(per[l]?.longest_road_freq) || isNum(per[l]?.dev_bought_per_game));
  return (
    <>
      <div className="t-scroll">
      <table className="table t-standings" data-testid="standings">
        <thead><tr>
          <th>Agent</th><th {...sortedBy(!byElo)}>Win rate · 95% Wilson CI{!byElo && " ↓"}</th><th /><th className="num">Wins / games</th>
          <th className="num">Avg VP ± 95% CI</th><th className="num" {...sortedBy(byElo)}>BT-Elo [95% CI]{byElo && " ↓"}</th>
        </tr></thead>
        <tbody>
          {order.map((l) => {
            const s = per[l] ?? {};
            const ci = winCi(s);
            const vp = avgVp(s);
            const r = t.ratings?.[l];
            const rci = ci2(r?.ci95);
            const exp = isNum(s.expected_win_rate_if_equal) ? s.expected_win_rate_if_equal : null;
            return (
              <tr key={l}>
                <td><AgentName label={l} copies={cp[l]} /></td>
                <td>{isNum(s.win_rate) && <RateBar rate={s.win_rate} ci={ci} expected={exp} width={180} />}</td>
                <td className="num t-rate"><strong>{pct(s.win_rate)}</strong> <span className="muted">{pctCi(ci)}</span>
                  {exp !== null && <div className="muted small">expected if equal {pct(exp, 0)}</div>}</td>
                <td className="num count" data-col="wins">{int(s.wins)} / {int(s.games)}</td>
                <td className="num" data-col="avgvp">{vp ? <>{vp.mean.toFixed(2)}{vp.ci && <span className="muted"> ± {((vp.ci[1] - vp.ci[0]) / 2).toFixed(2)}</span>}</> : DASH}</td>
                <td className="num">{isNum(r?.elo) ? <>{r.elo.toFixed(0)}{rci && <span className="muted"> [{rci[0].toFixed(0)}, {rci[1].toFixed(0)}]</span>}</> : DASH}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      </div>
      <p className="muted small t-legend">
        <span className="t-key fill" /> win rate <span className="t-key whisker" /> 95% CI <span className="t-key exp" /> expected if all agents were equal (the agent's share of seats).
        {Object.values(cp).some((c) => c > 1) && " An entrant with several seats wins when any of its copies wins, so its baseline is higher."}
        {" "}Elo is a Bradley-Terry fit on game rankings (winner, then VP), mean 1500.
      </p>
      {anyBehaviour && <More title="Play style" note="per seat-game: awards, dev cards, trading, timing">
        <div className="t-scroll">
          <table className="table t-behaviour">
            <thead>
              <tr><th rowSpan={2}>Agent</th><th colSpan={2} className="grp">Awards held at end</th><th colSpan={2} className="grp">Dev cards</th>
                <th colSpan={6} className="grp">Trading</th><th rowSpan={2} className="num">Illegal</th>
                {anyMs && <th colSpan={2} className="grp" title="Measured with many workers; may be inflated under load">ms / move<sup className="t-warn">†</sup></th>}</tr>
              <tr><th className="num">Longest Road</th><th className="num">Largest Army</th><th className="num">Bought</th><th className="num">Played</th>
                <th className="num">Offers</th><th className="num">Executed</th><th className="num">Success</th><th className="num">As partner</th><th className="num">Counters</th><th className="num">Bank / port</th>
                {anyMs && <><th className="num">Mean</th><th className="num">Max</th></>}</tr>
            </thead>
            <tbody>
              {order.map((l) => {
                const s = per[l] ?? {};
                const tr = s.trade ?? {};
                const played = s.dev_played_per_game && typeof s.dev_played_per_game === "object" ? s.dev_played_per_game : null;
                const playedSum = played ? Object.values(played).reduce((a: number, b: any) => a + (isNum(b) ? b : 0), 0) : null;
                return (
                  <tr key={l}>
                    <td><AgentName label={l} copies={cp[l]} /></td>
                    <td className="num">{pct(s.longest_road_freq, 0)}</td><td className="num">{pct(s.largest_army_freq, 0)}</td>
                    <td className="num">{fix(s.dev_bought_per_game)}</td>
                    <td className="num" title={played ? Object.entries(played).map(([k, v]) => `${k.replace(/_/g, " ")}: ${fix(v)}`).join("\n") : undefined}>{fix(playedSum)}</td>
                    <td className="num">{fix(tr.offers_per_game, 1)}</td><td className="num">{fix(tr.offers_executed_per_game, 1)}</td>
                    <td className="num">{pct(tr.offer_success_rate, 0)}</td><td className="num">{fix(tr.accepted_as_partner_per_game, 1)}</td>
                    <td className="num">{fix(tr.counters_per_game, 1)}</td><td className="num">{fix(tr.maritime_per_game, 1)}</td>
                    <td className={"num count" + (s.illegal_actions > 0 ? " error" : "")}>{int(s.illegal_actions)}</td>
                    {anyMs && <><td className="num">{fmtMs(s.sec_per_move)}</td><td className="num">{fmtMs(s.max_sec_per_move)}</td></>}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <p className="muted small">Played dev cards: hover a cell for the per-type split. Success = executed offers / offers made. Bank / port = maritime trades. Illegal = actions the runner rejected and replaced.
          {anyMs && <><br /><span className="t-warn">†</span> ms/move may be inflated under load: tournaments run many workers in parallel, so compare timings only between runs.</>}</p>
      </More>}
    </>
  );
}

function Pairwise({ t, order }: { t: any; order: string[] }) {
  const cp = copiesOf(t);
  const cells = order.flatMap((a) => order.filter((b) => b !== a).map((b) => pairOf(t.pairwise, a, b)));
  if (order.length < 2 || !cells.some(Boolean)) return null;
  const multi = order.filter((l) => cp[l] > 1);
  // Under equal strength, the best of k copies finishes ahead of the best of m copies k/(k+m) of the time.
  const expOf = (a: string, b: string) => cp[a] / (cp[a] + cp[b]);
  const bg = (r: number, e: number) => {
    const d = r - e, span = d >= 0 ? 1 - e : e;
    const a = Math.min(1, Math.abs(d) / (span || 1));
    return d >= 0 ? `rgba(217,164,65,${(0.1 + 0.75 * a).toFixed(2)})` : `rgba(44,127,194,${(0.1 + 0.7 * a).toFixed(2)})`;
  };
  return (
    <section className="t-section">
      <h3>Pairwise: how often the row finished ahead of the column</h3>
      <p className="muted small">
        "Ahead" means the row agent won the game, or neither won and it ended with more VP. Ties count half. Only games where both
        played are counted, with paired boards and rotated seats. <span className="t-sig">*</span> = sign test on decisive games, p&lt;0.05. Hover a cell for the CI and counts.
      </p>
      {multi.length > 0 && (
        <p className="note small t-callout" data-testid="copies-note">
          {multi.map((l) => `${shortSpec(l)} holds ${cp[l]} seats`).join(", ")}. Its copies are aggregated under one name and its <em>best</em> seat is
          used, so "{shortSpec(order.find((l) => cp[l] === 1) ?? order[0])} vs {shortSpec(multi[0])}" means beating all {cp[multi[0]]} copies.
          The equal-strength rate is then not 50%; each cell shows its own baseline, and colour is relative to it.
        </p>
      )}
      <div className="t-scroll">
        <table className="t-matrix" data-testid="pairwise">
          <thead><tr><th className="corner">row ↓ vs column →</th>{order.map((b) => <th key={b} scope="col" title={b}>{breakable(shortSpec(b))}{cp[b] > 1 && <div className="muted small">best of {cp[b]}</div>}</th>)}</tr></thead>
          <tbody>
            {order.map((a) => (
              <tr key={a}>
                <th scope="row" title={a}>{breakable(shortSpec(a))}{cp[a] > 1 && <div className="muted small">best of {cp[a]}</div>}</th>
                {order.map((b) => {
                  if (a === b) return <td key={b} className="diag">{DASH}</td>;
                  const c = pairOf(t.pairwise, a, b);
                  if (!c) return <td key={b} className="empty">{DASH}</td>;
                  const e = expOf(a, b);
                  const sig = c.p !== null && c.p < 0.05;
                  const tip = [
                    `${shortSpec(a)} ahead of ${shortSpec(b)}: ${pct(c.rate)}`,
                    c.ci && `95% CI ${pctCi(c.ci)}%`,
                    c.ahead !== null && c.n !== null && `ahead in ${c.ahead} of ${c.n} games${c.ties ? `, ${c.ties} ties` : ""}`,
                    `expected if equal: ${pct(e, 0)}`,
                    c.p !== null && `sign test ${fmtP(c.p)}`,
                  ].filter(Boolean).join("\n");
                  return (
                    <td key={b} style={{ background: bg(c.rate, e) }} className={c.rate > e && (c.rate - e) / (1 - e) > 0.6 ? "hot" : ""} title={tip} data-pair={`${a} vs ${b}`}>
                      <div className="rate">{pct(c.rate)}{sig && <span className="t-sig" aria-label="significant">*</span>}</div>
                      <div className="meta">
                        {c.n !== null && <span className="count" data-col="n">n={c.n}</span>}
                        {c.p !== null && <span className="p">{fmtP(c.p)}</span>}
                      </div>
                      {e !== 0.5 && <div className="meta">equal: {pct(e, 0)}</div>}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="muted small t-legend"><span className="t-swatch lo" /> below equal-strength baseline <span className="t-swatch hi" /> above it. Stronger colour = further from the baseline.</p>
    </section>
  );
}

function Seats({ t, order }: { t: any; order: string[] }) {
  const per = t.per_agent ?? {};
  const has = order.filter((l) => per[l]?.seats && Object.keys(per[l].seats).length);
  const glob: any[] = Array.isArray(t.global?.seat_win_rate) ? t.global.seat_win_rate : [];
  if (!has.length && !glob.length) return null;
  const n = nPlayers(t) ?? 4;
  const row = (seat: number, games: any, wins: any, rate: any, ci: any) => (
    <tr key={seat}>
      <td><span className="t-seat"><CrestBadge p={seat} size={16} /> {seat + 1} <span className="muted">{PLAYER_NAMES[seat]}</span></span></td>
      <td className="num count">{int(wins)} / {int(games)}</td>
      <td className="num">{pct(rate)} <span className="muted small">{pctCi(ci2(ci))}</span></td>
    </tr>
  );
  return (
    <More title="Seat breakdown" note="win rate by seat">
      <p className="muted small">Seat 1 moves first. Per agent, a "win" is that seat winning. A one-seat entrant's equal-strength rate is {pct(1 / n, 0)} in every seat.</p>
      <div className="t-seats">
        {glob.length > 0 && (
          <table className="table t-mini">
            <caption>All games, by seat</caption>
            <thead><tr><th>Seat</th><th className="num">Wins / games</th><th className="num">Win rate</th></tr></thead>
            <tbody>{glob.map((s) => row(s.seat, gamesOf(t), s.wins, s.rate, s.ci95))}</tbody>
          </table>
        )}
        {has.map((l) => (
          <table key={l} className="table t-mini">
            <caption title={l}>{shortSpec(l)}</caption>
            <thead><tr><th>Seat</th><th className="num">Wins / games</th><th className="num">Win rate</th></tr></thead>
            <tbody>{Object.entries(per[l].seats as Record<string, any>).sort((a, b) => +a[0] - +b[0]).map(([k, s]) => row(+k, s.games, s.wins, s.win_rate, s.ci95))}</tbody>
          </table>
        ))}
      </div>
    </More>
  );
}

function Charts({ t, order }: { t: any; order: string[] }) {
  const gs: any[] = Array.isArray(t.game_summaries) ? t.game_summaries : [];
  if (!gs.length) return null;
  const turns = gs.map((g) => g.turns).filter(isNum);
  const trunc = gs.filter((g) => g.truncated).length;
  // Final VP of each entrant's best seat, one value per game.
  const vpBy = order.map((l) => [l, gs.map((g) => {
    if (!Array.isArray(g.lineup) || !Array.isArray(g.vp)) return NaN;
    const v = g.vp.filter((_: any, q: number) => g.lineup[q] === l);
    return v.length ? Math.max(...v) : NaN;
  })] as const);
  return (
    <More title="Games" note="game length and final VP histograms">
      <div className="charts t-hists">
        <Histogram title="Game length" values={turns} xLabel="turns" />
        {vpBy.map(([l, vs]) => <Histogram key={l} title={`Final VP · ${shortSpec(l)}${copiesOf(t)[l] > 1 ? " (best seat)" : ""}`} values={vs} xLabel="VP" bin={1} />)}
      </div>
      <p className="muted small">One turn = one END_TURN by any seat. {trunc ? `${trunc} game${trunc > 1 ? "s" : ""} hit the turn limit.` : "No game hit the turn limit."}</p>
    </More>
  );
}

function Detail({ t }: { t: any }) {
  const per = t.per_agent ?? {};
  const order = useMemo(() => {
    const ls = labelsOf(t).filter((l) => per[l]);
    const key = (l: string) => (isNum(t.ratings?.[l]?.elo) ? t.ratings[l].elo : 0) * 10 + (per[l].win_rate ?? 0);
    return ls.sort((a, b) => key(b) - key(a));
  }, [t]);
  return (
    <section className="lab-card t-detail" data-testid="tournament-detail">
      <header className="t-head">
        <div>
          <h2>{t.id}</h2>
          <p className="t-lineup" title={labelsOf(t).join(" vs ")}>{lineupDesc(t)}</p>
        </div>
        <span className="muted">{dateOf(t)}</span>
      </header>
      <MetaLine t={t} />
      {order.length ? <Standings t={t} order={order} /> : <p className="muted">This file has no per_agent results.</p>}
      <Pairwise t={t} order={order} />
      <Seats t={t} order={order} />
      <Charts t={t} order={order} />
      <More title="Run details" note="seeds, turns, commit"><Summary t={t} /></More>
      {Array.isArray(t.failed_games) && t.failed_games.length > 0 && (
        <details><summary className="error">{t.failed_games.length} failed games</summary><pre>{t.failed_games.map((f: any) => `seed ${f.seed} rot ${f.rotation}\n${f.error}`).join("\n\n")}</pre></details>
      )}
      {Array.isArray(t.illegal_log) && t.illegal_log.length > 0 && (
        <details><summary>Illegal action log ({t.illegal_log.length})</summary><pre>{JSON.stringify(t.illegal_log, null, 1)}</pre></details>
      )}
    </section>
  );
}

/** RL checkpoints of one run against the same opponents, across tournaments. */
function Across({ list, onOpen }: { list: any[]; onOpen: (i: number) => void }) {
  const series = new Map<string, { x: number; t: any; i: number; s: any }[]>();
  list.forEach((t, i) => labelsOf(t).forEach((l) => {
    const x = ckptStep(l), s = t.per_agent?.[l];
    if (x === null || !isNum(s?.win_rate)) return;
    // Same seed set only, so every point is played on the same boards.
    const key = `${shortSpec(l).replace(/\/[^/,]*$/, "")} vs ${lineupDesc(t, l)}${isNum(t.seed) ? ` · --seed ${t.seed}` : ""}`;
    if (!series.has(key)) series.set(key, []);
    series.get(key)!.push({ x, t, i, s });
  }));
  const rows = [...series.entries()].filter(([, ps]) => ps.length > 1);
  return (
    <section className="lab-card t-detail" data-testid="across">
      <h2>Across tournaments</h2>
      <p className="muted">Win rate of each RL run's checkpoints against the same opponents, one point per tournament. Shaded band = 95% Wilson CI. Points are grouped by seed set, so every checkpoint in a series played the same boards.</p>
      {!rows.length && <p className="muted">No RL run has checkpoints in two or more tournaments with the same opponents yet.</p>}
      {rows.map(([key, ps]) => {
        ps.sort((a, b) => a.x - b.x || String(a.t.created).localeCompare(String(b.t.created)));
        const exp = ps[0].s.expected_win_rate_if_equal;
        return (
          <div key={key} className="t-across">
            <h3>{key}</h3>
            <div className="charts">
              <LineChart title="Win rate" xs={ps.map((p) => p.x)} ys={ps.map((p) => p.s.win_rate)} band={ps.map((p) => winCi(p.s))} yRange={[0, 1]} pct xLabel="checkpoint" chance={exp ?? undefined} />
              <LineChart title="Avg VP" xs={ps.map((p) => p.x)} ys={ps.map((p) => avgVp(p.s)?.mean ?? NaN)} xLabel="checkpoint" />
            </div>
            <table className="table">
              <thead><tr><th>Tournament</th><th className="num">Checkpoint</th><th>Win rate · 95% CI</th><th className="num">Wins / games</th><th className="num">Avg VP</th><th>Seeds</th></tr></thead>
              <tbody>{ps.map((p) => (
                <tr key={p.t.file ?? p.t.id}>
                  <td><button className="linkish" onClick={() => onOpen(p.i)}>{p.t.id}</button></td>
                  <td className="num count">{p.x}</td>
                  <td><RateBar rate={p.s.win_rate} ci={winCi(p.s)} expected={p.s.expected_win_rate_if_equal} width={140} /> {pct(p.s.win_rate)} <span className="muted">{pctCi(winCi(p.s))}</span></td>
                  <td className="num count">{int(p.s.wins)} / {int(p.s.games)}</td>
                  <td className="num">{fix(avgVp(p.s)?.mean)}</td>
                  <td className="muted">{isNum(p.t.seed) ? `--seed ${p.t.seed}` : DASH}</td>
                </tr>))}</tbody>
            </table>
            {isNum(exp) && <p className="muted small">Expected if equal: {pct(exp, 0)}.</p>}
          </div>
        );
      })}
    </section>
  );
}

// Sidebar groups, by file-name convention (the promotion gate writes gate_*, benchmarks bench-*/bots_*/zoo_*); first match wins.
const KINDS: [string, RegExp][] = [["Promotion gates", /^gate[_-]/], ["Benchmarks", /^(bench|bots|zoo)[_-]/],
  ["Checkpoint screens", /^(screen|ladder|select|rl\d+)[_-]/], ["Experiments and ad hoc", /^/]];
const kindOf = (t: any) => KINDS.findIndex(([, re]) => re.test(String(t.id ?? "")));

export default function Tournaments() {
  const [list, setList] = useState<any[] | null>(null);
  const [q, setQ] = useState("");
  const [sel, setSel] = useState<number>(0); // -1 = cross-tournament view
  useEffect(() => {
    api.tournaments().then((r) => {
      const ts = [...r.tournaments];
      // Newest first by "created"; files without it keep the server's mtime order at the end.
      ts.sort((a, b) => String(b.created ?? "").localeCompare(String(a.created ?? "")));
      setList(ts);
    }).catch(() => setList([]));
  }, []);
  if (!list) return <p className="muted">Loading…</p>;
  if (!list.length) return <section className="lab-card"><h2>No tournaments found</h2><p className="muted">Results appear here from results/tournaments/*.json. Run scripts/tournament.py to make one.</p></section>;
  const t = sel >= 0 ? list[Math.min(sel, list.length - 1)] : null;
  const ql = q.trim().toLowerCase();
  return (
    <div className="runs tournaments">
      <nav className="run-list t-list" aria-label="Tournaments">
        <button className={"run-item t-across-btn" + (sel === -1 ? " on" : "")} onClick={() => setSel(-1)}>
          <strong>Across tournaments</strong><small>RL checkpoints over time</small>
        </button>
        <input type="search" className="run-filter" placeholder={`Filter ${list.length} tournaments`} aria-label="Filter tournaments" value={q} onChange={(e) => setQ(e.target.value)} />
        {KINDS.map(([name], k) => {
          const items = list.map((x, i) => [x, i] as const).filter(([x]) => kindOf(x) === k && (!ql || `${x.id} ${labelsOf(x).join(" ")}`.toLowerCase().includes(ql)));
          if (!items.length) return null;
          // Only the newest file's group starts open (a fixed choice, so React never closes a group the user opened); a filter opens all.
          return (
            <details key={name + (ql ? ":q" : "")} className="metric-group t-group" open={!!ql || k === kindOf(list[0])} data-kind={name}>
              <summary><strong>{name}</strong><span className="muted small">{items.length}</span></summary>
              {items.map(([x, i]) => (
                <button key={x.file ?? x.id} className={"run-item" + (i === sel ? " on" : "")} onClick={() => setSel(i)} aria-current={i === sel} title={lineupDesc(x)}>
                  <strong>{x.id}</strong>
                  <small>{[dateOf(x), isNum(gamesOf(x)) && `${gamesOf(x)} games`].filter(Boolean).join(" · ")}</small>
                </button>
              ))}
            </details>
          );
        })}
        {ql && !list.some((x) => `${x.id} ${labelsOf(x).join(" ")}`.toLowerCase().includes(ql)) && <p className="muted">No tournament matches “{q}”.</p>}
      </nav>
      {t ? <Detail key={t.file ?? t.id} t={t} /> : <Across list={list} onOpen={setSel} />}
    </div>
  );
}
