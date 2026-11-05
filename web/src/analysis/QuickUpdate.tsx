// "Live" tab: record a real turn in a few taps. Applies client-side rule
// mirrors from ./quick.ts (labeled as an estimate); live validation still runs.
import { useState } from "react";
import type { PositionSpec, Topology } from "../api";
import { ResIcon } from "../board/Terrain";
import { CrestBadge } from "../Setup";
import { DEV_LABEL, PLAYER_COLORS, PLAYER_NAMES, RES_LABEL } from "../rules";
import { PHASE_NAME, Stepper } from "./EditorPanel";
import * as Q from "./quick";
import { actorOf } from "./spec";

export type Armed = { kind: "road" | "settlement" | "city" | "robber"; pay: boolean } | null;
type Set = (f: (s: PositionSpec) => PositionSpec, key?: string) => void;
const Z = () => [0, 0, 0, 0, 0];
const sum = (a: number[]) => a.reduce((x, y) => x + y, 0);

function Bundle({ v, onChange, label, max }: { v: number[]; onChange: (v: number[]) => void; label: string; max?: number[] }) {
  return (
    <div className="an-res">
      {v.map((n, r) => (
        <div key={r} className={"an-res-cell" + (n ? "" : " zero")}>
          <ResIcon r={r} size={20} />
          <Stepper value={n} max={max?.[r] ?? 19} label={`${RES_LABEL[r]} ${label}`} onChange={(x) => onChange(v.map((y, i) => (i === r ? x : y)))} />
        </div>
      ))}
    </div>
  );
}

export default function QuickUpdate({ spec, set, topo, arm, armed }: {
  spec: PositionSpec; set: Set; topo: Topology; arm: (a: Armed) => void; armed: Armed;
}) {
  const [last, setLast] = useState<{ total: number; gained: number[][]; notes: string[] } | null>(null);
  const [pay, setPay] = useState(true);
  const [devPick, setDevPick] = useState(false);
  const [victim, setVictim] = useState<number | null>(null);
  const [partner, setPartner] = useState(-1);
  const [give, setGive] = useState(Z());
  const [get, setGet] = useState(Z());
  const [disc, setDisc] = useState(Z());
  const p = spec.current, n = spec.num_players, me = spec.viewer;
  // Compute on a copy of the shown position so results are available right away.
  const apply = <T,>(f: (s: PositionSpec) => T): T => { const c = structuredClone(spec); const r = f(c); set(() => c); return r; };
  const name = (q: number) => (q === me ? `${PLAYER_NAMES[q]} (you)` : PLAYER_NAMES[q]);
  const setup = spec.phase.startsWith("SETUP");
  const over = spec.phase === "GAME_OVER";
  // DS31-02: mid-trade the seat answering is not the seat whose turn it is ("Heather's turn" while Tide must reply)
  const who = setup || over ? p : actorOf(spec);

  const roll = (total: number) => {
    if (total === 7) { apply((s) => Q.rollSeven(s)); setLast({ total, gained: [], notes: [] }); return; }
    setLast({ total, ...apply((s) => Q.produce(s, total, topo)) });
  };

  const knightOk = !spec.dev_played && (p === me ? (spec.players[me].dev_cards?.[0] ?? 0) - (spec.players[me].dev_new?.[0] ?? 0) > 0 : (spec.players[p].dev_count ?? 0) > 0);
  const victims = spec.phase === "STEAL" ? Q.stealVictims(spec, topo) : [];
  const mustDiscard = (spec.discard_remaining ?? []).map((x, q) => (q < n ? x : 0));

  return (
    <section className="an-live" aria-label="Quick update" style={{ ["--pc" as any]: PLAYER_COLORS[p] }}>
      <div className="an-card an-live-head">
        <div className="an-row">
          <CrestBadge p={p} size={26} />
          <span><strong className="an-live-who">{over ? `Game over (${PLAYER_NAMES[p]} moved last)` : p === me ? `Your turn (${PLAYER_NAMES[p]})` : `${PLAYER_NAMES[p]}'s turn`}</strong><br /><span className="muted small">Turn {spec.turn + 1} · {PHASE_NAME[spec.phase] ?? spec.phase}{who !== p && <span data-testid="live-actor"> · {name(who)} to act</span>}</span></span>
          {!over && <button className={setup ? "btn" : "btn primary"} style={{ marginLeft: "auto" }} onClick={() => { apply(Q.nextPlayer); setLast(null); setVictim(null); arm(null); }} data-testid="next-player">Next player ▸</button>}
        </div>
        {/* DS30-04: at game over "Chalk's turn" and a primary Next player read as if the game went on */}
        {over ? <p className="small" data-testid="live-game-over">The game is over, so there are no turn tools. Step back in Game review, or change the phase in Edit, to explore another line.</p>
          : <p className="an-estimate small">Moves here use client-side rules, so each result is an estimate until the validator checks it.</p>}
      </div>

      {setup && (
        <div className="an-card">
          <span className="field-label">Setup: {name(p)} places a {spec.phase === "SETUP_ROAD" ? "road" : "settlement"} (free)</span>
          <button className="btn big primary" aria-pressed={!!armed} data-build={spec.phase === "SETUP_ROAD" ? "road" : "settlement"}
            onClick={() => arm(armed ? null : { kind: spec.phase === "SETUP_ROAD" ? "road" : "settlement", pay: false })}>
            {armed ? "Tap the board…" : spec.phase === "SETUP_ROAD" ? "Place road" : "Place settlement"}</button>
          <p className="muted small">The order snakes automatically. The second settlement collects its starting cards.</p>
        </div>
      )}

      {spec.phase.startsWith("TRADE_") && (
        // the panel was empty here apart from Next player, which read as broken
        <p className="an-card small" data-testid="trade-step-note">This position is mid-trade. Live turn has no tools for trade replies: use Best move? in the analysis panel to rate this step, or Next player to drop the trade and pass the turn.</p>
      )}

      {!spec.rolled && !setup && (
        <div className="an-card">
          <span className="field-label">Dice total</span>
          <div className="an-dice" role="group" aria-label="Dice total">
            {[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12].map((t) => (
              <button key={t} className={"an-die" + (t === 7 ? " seven" : t === 6 || t === 8 ? " red" : "")} onClick={() => roll(t)} data-roll={t}>{t}</button>
            ))}
          </div>
          {knightOk && <button className="btn" onClick={() => apply((s) => Q.playKnight(s, p))}>Knight before rolling</button>}
        </div>
      )}

      {last && (
        <div className="an-card an-roll-result" data-testid="roll-result">
          <strong>Rolled {last.total}</strong>
          {last.total === 7 ? <span className="small" data-testid="seven-status">No production. {mustDiscard.some((x) => x) ? "Discards first."
            : spec.phase === "MOVE_ROBBER" ? "Move the robber."
            : spec.phase === "STEAL" ? `Robber moved to hex ${spec.robber}. Pick whom to rob.`
            : `Robber moved to hex ${spec.robber}.`}</span> : (
            <ul className="small">
              {last.gained.map((g, q) => sum(g) ? <li key={q}><CrestBadge p={q} size={14} /> {name(q)} +{g.map((x, r) => (x ? `${x} ${RES_LABEL[r]}` : "")).filter(Boolean).join(", ")}</li> : null)}
              {!last.gained.some((g) => sum(g)) && <li className="muted">Nobody produced.</li>}
              {last.notes.map((x, i) => <li key={"n" + i} className="muted">{x}</li>)}
            </ul>
          )}
        </div>
      )}

      {spec.phase === "DISCARD" && (
        <div className="an-card">
          <span className="field-label">Discard half</span>
          {mustDiscard.map((k, q) => !k ? null : q === me ? (
            <div key={q} className="an-col">
              <span>{name(q)}: choose {k} ({sum(disc)} picked)</span>
              <Bundle v={disc} onChange={setDisc} label="to discard" max={spec.players[me].hand} />
              <button className="btn primary" disabled={sum(disc) !== k} onClick={() => { apply((s) => Q.discard(s, q, disc)); setDisc(Z()); }}>Discard these</button>
            </div>
          ) : (
            <button key={q} className="btn" onClick={() => apply((s) => Q.discard(s, q, k))} data-discard={q}>{name(q)} discarded {k}</button>
          ))}
        </div>
      )}

      {spec.phase === "MOVE_ROBBER" && (
        <div className="an-card">
          <button className={"btn big" + (armed?.kind === "robber" ? " primary" : "")} onClick={() => arm({ kind: "robber", pay: false })} data-testid="arm-robber">
            {armed?.kind === "robber" ? "Tap the robber's new tile…" : "Move the robber"}</button>
        </div>
      )}

      {spec.phase === "STEAL" && (
        <div className="an-card">
          <span className="field-label">{name(p)} steals from</span>
          <div className="an-row">
            {victims.map((v) => <button key={v} className={"chip" + (victim === v ? " on" : "")} onClick={() => {
              if (p !== me && v !== me) { apply((s) => Q.steal(s, p, v, null)); setVictim(null); } else setVictim(v);
            }} data-victim={v}>{name(v)}</button>)}
            <button className="chip" onClick={() => apply((s) => { s.phase = "MAIN"; })}>No steal</button>
          </div>
          {victim !== null && (
            <div className="an-row">
              <span className="small">Which card?</span>
              {RES_LABEL.map((l, r) => (
                <button key={r} className="btn small" disabled={victim === me && !(spec.players[me].hand?.[r])} onClick={() => { apply((s) => Q.steal(s, p, victim, r)); setVictim(null); }} data-steal-res={r}>
                  <ResIcon r={r} size={16} /> {l}</button>
              ))}
            </div>
          )}
        </div>
      )}

      {spec.rolled && spec.phase === "MAIN" && (
        <>
          <div className="an-card">
            <span className="field-label">{name(p)} builds (tap, then tap the board)</span>
            <div className="an-build">
              {(["road", "settlement", "city"] as const).map((k) => (
                <button key={k} className={"btn" + (armed?.kind === k ? " primary" : "")} onClick={() => arm(armed?.kind === k ? null : { kind: k, pay })} data-build={k}>
                  {k[0].toUpperCase() + k.slice(1)}</button>
              ))}
              <button className="btn" onClick={() => (p === me ? setDevPick(!devPick) : apply((s) => Q.buyDev(s, p, null)))} data-build="dev">Dev card</button>
              {knightOk && <button className="btn" onClick={() => apply((s) => Q.playKnight(s, p))} data-testid="play-knight">Play Knight</button>}
            </div>
            {devPick && p === me && (
              <div className="an-row">{DEV_LABEL.map((d, k) => <button key={d} className="btn small" onClick={() => { apply((s) => Q.buyDev(s, p, k)); setDevPick(false); }}>{d}</button>)}</div>
            )}
            <label className="an-check"><input type="checkbox" checked={pay} onChange={(e) => setPay(e.target.checked)} /> Pay the build cost from {name(p)}'s cards</label>
          </div>
          <div className="an-card">
            <span className="field-label">Trade</span>
            <div className="an-row">
              <span className="small">{name(p)} with</span>
              <button className={"chip" + (partner === -1 ? " on" : "")} onClick={() => setPartner(-1)}>Bank / harbor</button>
              {Array.from({ length: n }, (_, q) => q).filter((q) => q !== p).map((q) => (
                <button key={q} className={"chip" + (partner === q ? " on" : "")} onClick={() => setPartner(q)} data-partner={q}>{name(q)}</button>
              ))}
            </div>
            <span className="small">{name(p)} gives</span>
            <Bundle v={give} onChange={setGive} label="given" />
            <span className="small">{name(p)} gets</span>
            <Bundle v={get} onChange={setGet} label="received" />
            <button className="btn primary" disabled={!sum(give) && !sum(get)} onClick={() => { apply((s) => Q.trade(s, p, partner, give, get)); setGive(Z()); setGet(Z()); }} data-testid="apply-trade">Record trade</button>
          </div>
        </>
      )}
    </section>
  );
}
