import type { GameEvent, PlayerInfo, Topology, View } from "../api";
import { describeEvent, DEV_HELP, DEV_LABEL, RES, RES_LABEL, seatShort } from "../rules";
import { ResIcon } from "../board/Terrain";

// Full-information player inspector for replays and AI-only matches. Only
// rendered from data the server chose to reveal (viewer=all / replay-state).

export interface PrevHand { hand: number[]; dev_cards: number[] }

const DEV_SHORT = ["Knight", "VP", "Roads", "Plenty", "Monopoly"];

/** Compact dev-card chips for the player panel. */
export function DevChips({ p }: { p: PlayerInfo }) {
  if (!p.dev_cards) return null;
  const fresh = p.dev_new ?? [0, 0, 0, 0, 0];
  const held = p.dev_cards.map((n, t) => ({ n, t, f: fresh[t] })).filter((c) => c.n > 0);
  return (
    <div className="hi-devs" data-testid="dev-chips">
      {held.length ? held.map(({ n, t, f }) => (
        <span key={t} className={"hi-dev" + (t === 1 ? " vp" : "") + (f ? " fresh" : "")} data-dev={t} data-n={n}
          title={`${n} ${DEV_LABEL[t]}${f ? ` (${f} bought this turn)` : ""}: ${DEV_HELP[t]}`}>
          {DEV_SHORT[t]}{n > 1 && <b>×{n}</b>}{f > 0 && <i aria-label="bought this turn">new</i>}
        </span>
      )) : <span className="muted">No dev cards</span>}
    </div>
  );
}

const pips = (n: number | null) => (n ? 6 - Math.abs(7 - n) : 0);

function production(p: PlayerInfo, view: View, topo: Topology) {
  const out = [0, 0, 0, 0, 0];
  const add = (v: number, w: number) => {
    for (const h of topo.vertices[v].hexes) {
      const r = RES.indexOf(view.layout.hex_terrain[h] as any);
      if (r >= 0 && h !== view.robber) out[r] += w * pips(view.layout.hex_number[h]);
    }
  };
  p.settlements.forEach((v) => add(v, 1));
  p.cities.forEach((v) => add(v, 2));
  return out;
}

/** Why `id`'s hand changed in this move, per resource (first matching event wins). */
function reason(e: GameEvent, id: number, r: number): string | null {
  const t = e.type;
  if (t === "resources_gained" && e.player === id && e.resources[r] > 0)
    return e.reason === "production" ? "from roll" : e.reason === "setup" ? "starting hand" : "Year of Plenty";
  if (t === "discarded" && e.player === id && e.resource === r) return "discarded to robber";
  if (t === "stolen_detail" && e.resource === r) {
    if (e.thief === id) return `stolen from ${seatShort(e.victim)}`;
    if (e.victim === id) return `stolen by ${seatShort(e.thief)}`;
  }
  if (t === "monopoly" && e.resource === r) return e.player === id ? "Monopoly" : `taken by ${seatShort(e.player)}'s Monopoly`;
  if (t === "maritime_trade" && e.player === id && (e.give === r || e.get === r)) return "bank trade";
  if (t === "trade_executed" && (e.player === id || e.partner === id) && (e.give[r] || e.get[r]))
    return `trade with ${seatShort(e.player === id ? e.partner : e.player)}`;
  if (e.player === id && !e.free) {
    if (t === "settlement_built") return "built a settlement";
    if (t === "city_built") return "built a city";
    if (t === "road_built") return "built a road";
    if (t === "dev_bought") return "bought a dev card";
  }
  return null;
}

/** Builds, trades, steals and dev plays involving `id` over a multi-move span. */
function TurnEvents({ id, events }: { id: number; events: GameEvent[] }) {
  const n = seatShort;
  const lines = events.flatMap((e): string[] => {
    const mine = e.player === id || e.partner === id || e.thief === id || e.victim === id;
    if (!mine) return [];
    switch (e.type) {
      case "settlement_built": case "city_built": case "road_built": case "dev_bought": case "dev_played":
      case "maritime_trade": case "trade_executed": case "monopoly":
        return [describeEvent(e) ?? ""];
      case "stolen_detail": return [`${n(e.thief)} stole 1 ${RES_LABEL[e.resource]} from ${n(e.victim)}`];
    }
    return [];
  }).filter(Boolean);
  if (!lines.length) return null;
  return <ul className="hd-turn-events" data-testid="turn-events">{lines.map((t, i) => <li key={i}>{t}</li>)}</ul>;
}

function changes(p: PlayerInfo, prev: PrevHand | undefined, events: GameEvent[]) {
  if (!prev || !p.hand || !p.dev_cards) return null;
  const lines: { key: string; d: number; what: React.ReactNode; why: string | null }[] = [];
  p.hand.forEach((n, r) => {
    const d = n - prev.hand[r];
    if (d) lines.push({ key: "r" + r, d, what: <><ResIcon r={r} size={14} />{RES_LABEL[r]}</>,
                        why: [...new Set(events.map((e) => reason(e, p.id, r)).filter(Boolean))].join(", ") || null });
  });
  p.dev_cards.forEach((n, t) => {
    const d = n - prev.dev_cards[t];
    if (d) lines.push({ key: "d" + t, d, what: <>{DEV_LABEL[t]}</>, why: d > 0 ? "bought" : "played" });
  });
  return lines;
}

export function HandDetail({ p, view, topo, prev, events, span, title }:
  { p: PlayerInfo; view: View; topo: Topology; prev?: PrevHand; events: GameEvent[]; span: number; title?: string }) {
  const revealed = !!p.hand;
  const lr = view.longest_road_owner === p.id ? 2 : 0;
  const la = view.largest_army_owner === p.id ? 2 : 0;
  const vpCards = p.dev_cards?.[1] ?? 0;
  const vp: [string, number][] = [["Settlements", p.settlements.length], ["Cities", p.cities.length * 2],
    ["Longest Road", lr], ["Largest Army", la], ["VP cards", vpCards]];
  const prod = production(p, view, topo);
  const maxProd = Math.max(...prod, 1);
  const ch = changes(p, prev, events);
  return (
    <section className="hand-detail" data-testid="hand-detail" data-seat={p.id} onClick={(e) => e.stopPropagation()}>
      {revealed ? (
        <>
          <h4>Hand <span className="muted">{p.resource_count} cards</span></h4>
          <div className="hd-hand">
            {p.hand!.map((n, r) => (
              <span key={r} className={"hd-res" + (n ? "" : " zero")} data-res={r} data-n={n} title={`${n} ${RES_LABEL[r]}`}>
                <ResIcon r={r} size={20} /><b>{n}</b>
              </span>
            ))}
          </div>
          <h4>Development cards <span className="muted">{p.knights_played} knights played</span></h4>
          <ul className="hd-devs">
            {p.dev_cards!.map((n, t) => n > 0 && (
              <li key={t} data-dev={t}>
                <span>{DEV_LABEL[t]}</span><b>×{n}</b>
                {(p.dev_new?.[t] ?? 0) > 0 && <i className="hd-new">{p.dev_new![t]} bought this turn</i>}
                {t === 1 && <i className="hd-hidden">hidden</i>}
              </li>
            ))}
            {!p.dev_cards!.some(Boolean) && <li className="muted">None held</li>}
          </ul>
        </>
      ) : <p className="muted">Hands are hidden. Turn on "Show all hands" to see cards.</p>}

      <h4>Victory points <span className="hd-total">{p.vp ?? p.public_vp}</span></h4>
      <dl className="hd-vp">
        {vp.filter(([k, n]) => n || k === "Settlements" || (k === "VP cards" && revealed)).map(([k, n]) => (
          <div key={k}><dt>{k}</dt><dd>{k === "VP cards" && !revealed ? "?" : n}</dd></div>
        ))}
      </dl>

      <h4>Production <span className="muted">pips per roll, robber excluded</span></h4>
      <div className="hd-prod">
        {prod.map((n, r) => (
          <div key={r} className="hd-prod-row" title={`${n} pips of ${RES_LABEL[r]} (${(n / 36 * 100).toFixed(0)}% per roll)`}>
            <ResIcon r={r} size={14} />
            <span className="bar-track"><span className="bar-fill" style={{ width: `${(n / maxProd) * 100}%` }} /></span>
            <b>{n}</b>
          </div>
        ))}
      </div>

      <h4>Harbors</h4>
      <div className="hd-ports">
        {p.ports.length ? p.ports.map((t) => (
          <span key={t} className="hd-port" title={t === "3:1" ? "Trade any 3 for 1" : `Trade 2 ${RES_LABEL[RES.indexOf(t as any)]} for 1`}>
            {t === "3:1" ? "3:1" : <><ResIcon r={RES.indexOf(t as any)} size={14} />2:1</>}
          </span>
        )) : <span className="muted">None</span>}
      </div>

      <h4>{title ?? (span > 1 ? `Changes over the last ${span} moves` : "Changes at this move")}</h4>
      {ch === null ? <p className="muted">{revealed ? "Start of game." : "Hidden."}</p>
        : ch.length === 0 ? <p className="muted">No card changes.</p>
        : (
          <ul className="hd-changes" data-testid="hand-changes">
            {ch.map((c) => (
              <li key={c.key} className={c.d > 0 ? "up" : "down"}>
                <b>{c.d > 0 ? "+" : "−"}{Math.abs(c.d)}</b>{c.what}{c.why && <span className="muted">{c.why}</span>}
              </li>
            ))}
          </ul>
        )}
      {span > 1 && ch !== null && <TurnEvents id={p.id} events={events} />}
    </section>
  );
}
