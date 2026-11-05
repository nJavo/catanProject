import { useState } from "react";
import type { Action, View } from "../api";
import { RES_LABEL, seatName } from "../rules";
import { ResIcon } from "../board/Terrain";
import { Modal } from "./Dialogs";
import { CrestBadge } from "../Setup";

const zero = () => [0, 0, 0, 0, 0];

function Bundle({ v, size = 24, named = false }: { v: number[]; size?: number; named?: boolean }) {
  // 24 px icons: the glyph inside is ~16 px, the smallest that reads at a glance (DS3-09)
  // named (DS23-03): the response card's two columns have room to print the resource, so Brick vs Ore isn't a guess
  const parts = v.flatMap((n, r) => (n ? [<span key={r} className="bundle-item" title={`${n} ${RES_LABEL[r]}`}><ResIcon r={r} size={size} />{n}<span className={named ? "bundle-name" : "sr-only"}> {RES_LABEL[r]}</span></span>] : []));
  return <span className="bundle">{parts.length ? parts : <span className="muted">nothing</span>}</span>;
}

function Steppers({ label, value, max, onChange, testid, have }: { label: string; value: number[]; max: number[]; onChange: (v: number[]) => void; testid: string; have?: number[] }) {
  return (
    <div className="steppers" data-testid={testid}>
      <span className="field-label">{label}</span>
      <div className="stepper-row">
        {value.map((n, r) => (
          <div key={r} className="stepper-cell">
            <ResIcon r={r} size={28} />
            <span>{RES_LABEL[r]}</span>
            <div className="stepper">
              <button aria-label={`Less ${RES_LABEL[r]}`} disabled={n === 0} onClick={() => onChange(value.map((x, k) => (k === r ? x - 1 : x)))}>−</button>
              <output>{n}</output>
              <button aria-label={`More ${RES_LABEL[r]}`} disabled={n >= max[r]} onClick={() => onChange(value.map((x, k) => (k === r ? x + 1 : x)))} data-plus={r}>+</button>
            </div>
            {have && <small className="muted" data-have={r}>have {have[r]}</small>}
          </div>
        ))}
      </div>
    </div>
  );
}

function offerProblem(give: number[], get: number[]): string | null {
  if (!give.some(Boolean) || !get.some(Boolean)) return "Both sides need at least one card. Gifts aren't allowed.";
  if (give.some((n, r) => n && get[r])) return "You can't trade a resource for the same resource.";
  return null;
}

/** Compose an offer (proposer) or a counter-offer (responder). Values are
 * always sent in the proposer's perspective, as the engine expects. */
export function TradeComposer({ view, mode, onSubmit, onClose }: {
  view: View; mode: "offer" | "counter"; onSubmit: (a: Action) => void; onClose: () => void;
}) {
  const me = view.viewer!;
  const hand = view.players[me].hand ?? zero();
  const tr = view.trade;
  // In counter mode "mine" = what I (responder) give = proposer's get.
  const [mine, setMine] = useState<number[]>(mode === "counter" && tr ? [...tr.get] : zero());
  const [theirs, setTheirs] = useState<number[]>(mode === "counter" && tr ? [...tr.give] : zero());
  const problem = offerProblem(mine, theirs);
  const left = view.max_trade_offers_per_turn === null ? null : view.max_trade_offers_per_turn - view.trades_this_turn;
  const propHand = mode === "counter" && tr ? view.players[tr.proposer] : null;
  const submit = () => onSubmit(mode === "offer"
    ? { player: me, type: "OFFER_TRADE", value: [mine, theirs] }
    : { player: me, type: "COUNTER_OFFER", value: [theirs, mine] });
  return (
    <Modal title={mode === "offer" ? "Offer a trade to the table" : `Counter ${seatName(tr!.proposer, view.seats)}'s offer`} onClose={onClose} wide>
      <Steppers label="You give" value={mine} max={hand} onChange={setMine} testid="give" have={hand} />
      <Steppers label="You get" value={theirs} max={[19, 19, 19, 19, 19]} onChange={setTheirs} testid="get" />
      {propHand && <p className="muted">{seatName(tr!.proposer, view.seats)} holds {propHand.resource_count} cards. The server checks they can pay.</p>}
      {mode === "offer" && <p className="muted">Every other player answers in turn: accept, decline or counter. You then choose a partner.{left !== null ? ` ${left} offer${left === 1 ? "" : "s"} left this turn.` : ""}</p>}
      <div className="modal-actions">
        {/* inside the pinned action row, so the reason stays next to the disabled button on a phone */}
        {problem && <p className="note">{problem}</p>}
        <button className="btn ghost" onClick={onClose}>Cancel</button>
        <button className="btn primary" disabled={!!problem} onClick={submit}>{mode === "offer" ? "Make offer" : "Send counter-offer"}</button>
      </div>
    </Modal>
  );
}

export function MaritimeTrade({ view, legal, onSubmit, onClose }: { view: View; legal: Action[]; onSubmit: (a: Action) => void; onClose: () => void }) {
  const rates = view.trade_rates ?? [4, 4, 4, 4, 4];
  const hand = view.players[view.viewer!].hand ?? zero();
  const [give, setGive] = useState<number | null>(null);
  const [get, setGet] = useState<number | null>(null);
  const act = legal.find((a) => a.type === "MARITIME_TRADE" && (a.value as number[])[0] === give && (a.value as number[])[1] === get);
  const canGive = (r: number) => legal.some((a) => a.type === "MARITIME_TRADE" && (a.value as number[])[0] === r);
  return (
    <Modal title="Trade with the bank" onClose={onClose} wide>
      <p className="muted">Your rates come from your harbors: 4:1 by default, 3:1 at a generic harbor, 2:1 at a matching harbor.</p>
      <div className="maritime">
        <div>
          <span className="field-label" id="mt-give">Give</span>
          <div className="res-picker" role="group" aria-labelledby="mt-give">
            {RES_LABEL.map((l, r) => (
              <button key={r} className={"pick" + (give === r ? " on" : "")} disabled={!canGive(r)} onClick={() => { setGive(r); if (get === r) setGet(null); }} data-give={r}
                aria-pressed={give === r} aria-label={`Give ${rates[r]} ${l} (${rates[r]}:1, you have ${hand[r]})`}>
                <ResIcon r={r} size={30} /><span>{l}</span><b className={"rate r" + rates[r]}>{rates[r]}:1</b><small>have {hand[r]}</small>
              </button>
            ))}
          </div>
        </div>
        <div>
          <span className="field-label" id="mt-get">Get</span>
          <div className="res-picker" role="group" aria-labelledby="mt-get">
            {RES_LABEL.map((l, r) => (
              <button key={r} className={"pick" + (get === r ? " on" : "")} disabled={give === r || view.bank[r] === 0} onClick={() => setGet(r)} data-get={r}
                aria-pressed={get === r} aria-label={`Get 1 ${l} (bank has ${view.bank[r]})`}>
                <ResIcon r={r} size={30} /><span>{l}</span><small>bank {view.bank[r]}</small>
              </button>
            ))}
          </div>
        </div>
      </div>
      <div className="modal-actions">
        <span className="muted" aria-live="polite">{give !== null && get !== null ? `${rates[give]} ${RES_LABEL[give]} for 1 ${RES_LABEL[get]}` : "Pick what to give and what to get"}</span>
        <button className="btn primary" disabled={!act} onClick={() => act && onSubmit(act)}>Trade</button>
      </div>
    </Modal>
  );
}

/** Shown to a responder during TRADE_RESPONSE. */
export function TradeResponse({ view, legal, onSubmit, onCounter, busy }: { view: View; legal: Action[]; onSubmit: (a: Action) => void; onCounter: () => void; busy?: boolean }) {
  const tr = view.trade!;
  const accept = legal.find((a) => a.type === "ACCEPT_TRADE");
  const reject = legal.find((a) => a.type === "REJECT_TRADE");
  return (
    <div className="trade-card" role="region" aria-label="Trade offer">
      <div className="trade-head"><CrestBadge p={tr.proposer} size={24} /> <strong>{seatName(tr.proposer, view.seats)}</strong> offers you a trade</div>
      <div className="trade-sides">
        <div><span className="field-label">You get</span><Bundle v={tr.give} named /></div>
        <div><span className="field-label">You give</span><Bundle v={tr.get} named /></div>
      </div>
      {!accept && <p className="note">You don't hold enough to accept as offered.</p>}
      <div className="trade-actions">
        <button className="btn" disabled={busy} onClick={() => reject && onSubmit(reject)}>Decline</button>
        <button className="btn" disabled={busy} onClick={onCounter}>Counter</button>
        <button className="btn primary" disabled={!accept || busy} onClick={() => accept && onSubmit(accept)}>Accept</button>
      </div>
    </div>
  );
}

/** Shown to the proposer during TRADE_CONFIRM. */
export function TradeConfirm({ view, legal, onSubmit, busy }: { view: View; legal: Action[]; onSubmit: (a: Action) => void; busy?: boolean }) {
  const tr = view.trade!;
  const cancel = legal.find((a) => a.type === "CANCEL_TRADE");
  const entries = Object.entries(tr.responses);
  return (
    <div className="trade-card" role="region" aria-label="Trade responses">
      <div className="trade-head">Responses to your offer: give <Bundle v={tr.give} /> get <Bundle v={tr.get} /></div>
      <ul className="responses">
        {entries.map(([q, r]) => {
          const seat = Number(q);
          const confirm = legal.find((a) => a.type === "CONFIRM_TRADE" && a.value === seat);
          const counter = r && (r.give.join() !== tr.give.join() || r.get.join() !== tr.get.join());
          return (
            // two lines (DS3-09): who and their answer, then the bundle and the Trade button
            <li key={q} className={r ? "yes" : "no"} data-response={seat}>
              <div className="resp-head"><CrestBadge p={seat} size={22} /> <span>{seatName(seat, view.seats)}</span>
                {r === null ? <em>declined</em> : counter ? <em>counters</em> : <em>accepts</em>}</div>
              {r && (
                <div className="resp-deal">
                  <span className="resp-bundle">you give <Bundle v={r.give} /> get <Bundle v={r.get} /></span>
                  {confirm ? <button className="btn primary small" disabled={busy} onClick={() => onSubmit(confirm)} data-confirm={seat}>Trade</button>
                    : <small className="muted" data-cant-pay={seat}>you can't pay this</small>}
                </div>
              )}
            </li>
          );
        })}
      </ul>
      <div className="trade-actions"><button className="btn" disabled={busy} onClick={() => cancel && onSubmit(cancel)}>Cancel offer</button></div>
    </div>
  );
}

export function TradeWaiting({ view }: { view: View }) {
  const tr = view.trade!;
  return (
    <div className="trade-card waiting">
      <div className="trade-head"><CrestBadge p={tr.proposer} size={22} /> <strong>{seatName(tr.proposer, view.seats)}</strong> offers a trade</div>
      {/* "offers 1 for" then "2" on the next line read as a broken sentence (DS32-01) */}
      <div className="trade-sides">
        <div><span className="field-label">Gives</span><Bundle v={tr.give} named /></div>
        <div><span className="field-label">Wants</span><Bundle v={tr.get} named /></div>
      </div>
      <p className="muted">Waiting for {tr.pending.map((p) => seatName(p, view.seats)).join(", ") || "the proposer"}…</p>
    </div>
  );
}
