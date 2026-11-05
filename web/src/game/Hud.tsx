import { useEffect, useRef, useState } from "react";
import type { GameEvent, Layout, PlayerInfo, Seat, View } from "../api";
import { DEV_HELP, DEV_LABEL, PLAYER_COLORS, RES_LABEL, describeEvent, groupDiscards, seatName } from "../rules";
import { ResIcon } from "../board/Terrain";
import { CrestBadge } from "../Setup";

const PIP_POS: Record<number, [number, number][]> = {
  1: [[0, 0]], 2: [[-1, -1], [1, 1]], 3: [[-1, -1], [0, 0], [1, 1]],
  4: [[-1, -1], [1, -1], [-1, 1], [1, 1]], 5: [[-1, -1], [1, -1], [0, 0], [-1, 1], [1, 1]],
  6: [[-1, -1], [1, -1], [-1, 0], [1, 0], [-1, 1], [1, 1]],
};

export function Die({ n, red }: { n: number; red?: boolean }) {
  return (
    <svg className="die" viewBox="-1.5 -1.5 3 3" aria-hidden="true">
      <rect x={-1.35} y={-1.35} width={2.7} height={2.7} rx={0.55} fill={red ? "#b3261e" : "#f6efdc"} stroke="#3b2f1e" strokeWidth={0.12} />
      {(PIP_POS[n] ?? []).map(([x, y], i) => <circle key={i} cx={x * 0.68} cy={y * 0.68} r={0.26} fill={red ? "#f6efdc" : "#2b2a30"} />)}
    </svg>
  );
}

export function Dice({ roll, rollSeq }: { roll: number[] | null; rollSeq: number }) {
  const [spin, setSpin] = useState(false);
  const [faces, setFaces] = useState<number[]>(roll ?? [6, 6]);
  const first = useRef(true);
  useEffect(() => {
    if (!roll) return;
    if (first.current) { first.current = false; setFaces(roll); return; }
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (reduce) { setFaces(roll); return; }
    setSpin(true);
    let k = 0;
    const t = setInterval(() => {
      k++;
      setFaces([1 + Math.floor(Math.random() * 6), 1 + Math.floor(Math.random() * 6)]);
      if (k >= 7) { clearInterval(t); setFaces(roll); setSpin(false); }
    }, 60);
    return () => clearInterval(t);
  }, [rollSeq]); // eslint-disable-line
  return (
    <div className={"dice" + (spin ? " spin" : "") + (roll ? "" : " idle")} aria-live="polite">
      <Die n={faces[0]} />
      <Die n={faces[1]} red />
      {roll && !spin && <span className="dice-total" aria-hidden="true">{roll[0] + roll[1]}</span>}
      {/* one announcement per roll, not one per tumble frame; nothing before the first roll */}
      <span className="sr-only">{roll && !spin ? `Rolled ${roll[0]} and ${roll[1]}: ${roll[0] + roll[1]}` : ""}</span>
    </div>
  );
}

export function ResCard({ r, n, selected, onClick, disabled }: { r: number; n: number; selected?: number; onClick?: () => void; disabled?: boolean }) {
  return (
    <button className={`res-card res-${r}` + (n === 0 ? " empty" : "")} onClick={onClick} disabled={disabled || !onClick}
      aria-label={`${n} ${RES_LABEL[r]}`} data-res={r}>
      <ResIcon r={r} size={34} />
      <span className="res-count">{n}</span>
      <span className="res-name">{RES_LABEL[r]}</span>
      {selected ? <span className="res-sel">−{selected}</span> : null}
    </button>
  );
}

export function Cost({ cost }: { cost: number[] }) {
  return (
    <span className="cost">
      {cost.flatMap((c, r) => Array.from({ length: c }, (_, i) => <ResIcon key={`${r}-${i}`} r={r} size={14} />))}
    </span>
  );
}

export function Award({ kind, holder }: { kind: "road" | "army"; holder: boolean }) {
  return (
    <span className={"award " + kind + (holder ? " held" : "")} title={kind === "road" ? "Longest Road: +2 VP" : "Largest Army: +2 VP"}>
      <svg viewBox="-1 -1 2 2" width={16} height={16} aria-hidden>
        {kind === "road"
          ? <path d="M-0.8 0.6 Q0 -0.9 0.8 0.6" fill="none" stroke="currentColor" strokeWidth={0.28} strokeLinecap="round" />
          : <path d="M0 -0.85 L0.7 -0.5 V0.1 Q0.7 0.7 0 0.9 Q-0.7 0.7 -0.7 0.1 V-0.5 Z" fill="currentColor" />}
      </svg>
      {kind === "road" ? "Longest Road" : "Largest Army"}
    </span>
  );
}

export function OpponentPanel({ p, view, active, seats, children }: { p: PlayerInfo; view: View; active: boolean; seats: Seat[]; children?: React.ReactNode }) {
  const tradeResp = view.trade?.responses?.[String(p.id)];
  const pending = view.trade?.pending?.includes(p.id);
  return (
    <div className={"opp" + (active ? " active" : "")} style={{ ["--pc" as any]: PLAYER_COLORS[p.id] }} data-seat={p.id}>
      <div className="opp-head">
        <CrestBadge p={p.id} size={26} />
        <span className="opp-name" title={seatName(p.id, seats, true)}>{seatName(p.id, seats)}</span>
        <span className="opp-vp" title="Public victory points">{p.vp ?? p.public_vp}<small>VP</small></span>
      </div>
      <div className="opp-stats">
        <span title="Resource cards"><i className="ico cards" role="img" aria-label="Resource cards" />{p.resource_count}</span>
        <span title="Development cards"><i className="ico dev" role="img" aria-label="Development cards" />{p.dev_count}</span>
        <span title="Knights played"><i className="ico knight" role="img" aria-label="Knights played" />{p.knights_played}</span>
        <span title="Longest road segment"><i className="ico road" role="img" aria-label="Longest road segment" />{p.road_length}</span>
      </div>
      {p.hand && view.viewer !== p.id && (
        <div className="opp-hand">{p.hand.map((n, r) => <span key={r} title={RES_LABEL[r]} data-res={r} data-n={n}><ResIcon r={r} size={14} />{n}</span>)}</div>
      )}
      <div className="opp-awards">
        {view.longest_road_owner === p.id && <Award kind="road" holder />}
        {view.largest_army_owner === p.id && <Award kind="army" holder />}
        {view.trade && view.trade.proposer !== p.id && (
          <span className={"trade-tag " + (tradeResp ? "yes" : tradeResp === null ? "no" : "")}>
            {tradeResp ? "Will trade" : tradeResp === null ? "Declined" : pending ? "Yet to answer" : ""}
          </span>
        )}
      </div>
      {children}
    </div>
  );
}

export function Bank({ view }: { view: View }) {
  return (
    <div className="bank" aria-label="Bank">
      <span className="bank-label">Bank</span>
      {view.bank.map((n, r) => <span key={r} className="bank-item" title={`${RES_LABEL[r]} in bank`}><ResIcon r={r} size={16} />{n}</span>)}
      <span className="bank-item" title="Development cards left"><i className="ico dev" role="img" aria-label="Development cards left" />{view.dev_deck_count}</span>
    </div>
  );
}

export function DevCards({ me, legalTypes, onPlay }: { me: PlayerInfo; legalTypes: Set<string>; onPlay: (t: number) => void }) {
  const dev = me.dev_cards ?? [0, 0, 0, 0, 0];
  const fresh = me.dev_new ?? [0, 0, 0, 0, 0];
  const playType = ["PLAY_KNIGHT", "", "PLAY_ROAD_BUILDING", "PLAY_YEAR_OF_PLENTY", "PLAY_MONOPOLY"];
  const cards = dev.map((n, t) => ({ t, n, fresh: fresh[t] })).filter((c) => c.n > 0);
  if (!cards.length) return <div className="dev-row empty"><span className="muted">No development cards</span></div>;
  return (
    <div className="dev-row">
      {cards.map(({ t, n, fresh }) => {
        const playable = t !== 1 && legalTypes.has(playType[t]);
        return (
          <button key={t} className={`dev-card dev-${t}` + (playable ? " playable" : "")} disabled={!playable} onClick={() => onPlay(t)}
            title={DEV_HELP[t] + (fresh ? ` (${fresh} bought this turn, playable next turn)` : "")} data-dev={t}>
            <span className="dev-name">{DEV_LABEL[t]}</span>
            <span className="dev-n">×{n}</span>
            {t === 1 ? <span className="dev-note">+{n} hidden VP</span> : fresh ? <span className="dev-note">{fresh} new</span> : playable ? <span className="dev-note">Play</span> : null}
          </button>
        );
      })}
    </div>
  );
}

export function Log({ events, seats, layout }: { events: GameEvent[]; seats: Seat[]; layout?: Layout }) {
  const [tab, setTab] = useState<"all" | "trades">("all");
  const ref = useRef<HTMLOListElement>(null);
  const shown = groupDiscards(events.filter((e) => tab === "all" || e.type.startsWith("trade") || e.type === "maritime_trade"));
  const lastSeq = events.length ? events[events.length - 1].seq : 0;
  useEffect(() => { ref.current?.scrollTo({ top: ref.current.scrollHeight }); }, [lastSeq, tab]);
  return (
    <section className="log">
      <div className="tabs" role="tablist">
        <button role="tab" aria-selected={tab === "all"} className={tab === "all" ? "on" : ""} onClick={() => setTab("all")}>Game log</button>
        <button role="tab" aria-selected={tab === "trades"} className={tab === "trades" ? "on" : ""} onClick={() => setTab("trades")}>Trades</button>
      </div>
      <ol ref={ref} className="log-list">
        {shown.map((e) => {
          const text = describeEvent(e, seats, layout); // layout names the robber's tile
          if (!text) return null;
          const who = e.player ?? e.thief ?? (e.type === "award_changed" && e.to >= 0 ? e.to : undefined); // DS30: award lines had no seat stripe
          return (
            <li key={e.seq} className={"ev ev-" + e.type + (e.type === "trade_response" ? " resp-" + e.response : "") + (e.visible_to ? " private" : "")} style={{ ["--pc" as any]: who !== undefined ? PLAYER_COLORS[who] : "transparent" }}>
              {text}{e.visible_to && <span className="private-tag">private</span>}
            </li>
          );
        })}
        {!shown.length && <li className="muted">Nothing yet.</li>}
      </ol>
    </section>
  );
}
