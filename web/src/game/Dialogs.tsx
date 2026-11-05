import { useEffect, useRef, useState } from "react";
import type { Action, View } from "../api";
import { RES_LABEL, PLAYER_COLORS, describeEvent, seatName, turnNumber } from "../rules";
import { ResIcon } from "../board/Terrain";
import { CrestBadge } from "../Setup";

const FOCUSABLE = 'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';

/** Keep Tab / Shift+Tab inside ``ref`` while it is open (and pull focus back if it escaped). */
export function useFocusTrap(ref: React.RefObject<HTMLElement>) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const box = ref.current;
      if (e.key !== "Tab" || !box) return;
      const items = [...box.querySelectorAll<HTMLElement>(FOCUSABLE)].filter((el) => el.offsetParent !== null || el === document.activeElement);
      if (!items.length) { e.preventDefault(); box.focus(); return; }
      const first = items[0], last = items[items.length - 1], at = document.activeElement as HTMLElement | null;
      const inside = !!at && box.contains(at);
      if (e.shiftKey && (!inside || at === first || at === box)) { e.preventDefault(); last.focus(); }
      else if (!e.shiftKey && (!inside || at === last)) { e.preventDefault(); first.focus(); }
    };
    document.addEventListener("keydown", onKey, true);
    return () => document.removeEventListener("keydown", onKey, true);
  }, [ref]);
}

export function Modal({ title, children, onClose, wide }: { title: string; children: React.ReactNode; onClose?: () => void; wide?: boolean }) {
  useEffect(() => {
    const k = (e: KeyboardEvent) => e.key === "Escape" && onClose?.();
    window.addEventListener("keydown", k);
    return () => window.removeEventListener("keydown", k);
  }, [onClose]);
  // Keyboard users: move focus into the dialog on open, and give it back to the opener on close.
  const box = useRef<HTMLDivElement>(null);
  useFocusTrap(box);
  useEffect(() => {
    const opener = document.activeElement as HTMLElement | null;
    box.current?.focus();
    return () => { if (opener?.isConnected) opener.focus(); };
  }, []);
  return (
    <div className="scrim" onClick={(e) => e.target === e.currentTarget && onClose?.()}>
      <div ref={box} tabIndex={-1} className={"modal" + (wide ? " wide" : "")} role="dialog" aria-modal="true" aria-label={title}>
        <h2>{title}</h2>
        {children}
        {onClose && <button className="modal-x" aria-label="Close" onClick={onClose}>×</button>}
      </div>
    </div>
  );
}

function ResPicker({ value, onPick, allowed }: { value?: number; onPick: (r: number) => void; allowed?: (r: number) => boolean }) {
  return (
    <div className="res-picker">
      {RES_LABEL.map((l, r) => (
        <button key={r} className={"pick" + (value === r ? " on" : "")} disabled={allowed ? !allowed(r) : false} onClick={() => onPick(r)} data-res={r}>
          <ResIcon r={r} size={34} /><span>{l}</span>
        </button>
      ))}
    </div>
  );
}

export function YearOfPlenty({ legal, onSubmit, onClose }: { legal: Action[]; onSubmit: (a: Action) => void; onClose: () => void }) {
  const [picks, setPicks] = useState<number[]>([]);
  const pair = [...picks].sort((a, b) => a - b);
  const act = picks.length === 2 ? legal.find((a) => a.type === "PLAY_YEAR_OF_PLENTY" && (a.value as number[])[0] === pair[0] && (a.value as number[])[1] === pair[1]) : undefined;
  return (
    <Modal title="Year of Plenty: take 2 cards" onClose={onClose}>
      <p className="muted">Pick two resources from the bank. The same one twice is fine.</p>
      <ResPicker onPick={(r) => setPicks((p) => (p.length >= 2 ? [r] : [...p, r]))} />
      <div className="picked">{picks.length ? picks.map((r, i) => <span key={i}><ResIcon r={r} size={22} />{RES_LABEL[r]}</span>) : <span className="muted">Nothing picked</span>}</div>
      {picks.length === 2 && !act && <p className="error">The bank can't supply that pair.</p>}
      <div className="modal-actions">
        <button className="btn ghost" onClick={() => setPicks([])}>Clear</button>
        <button className="btn primary" disabled={!act} onClick={() => act && onSubmit(act)}>Take cards</button>
      </div>
    </Modal>
  );
}

export function Monopoly({ player, onSubmit, onClose }: { player: number; onSubmit: (a: Action) => void; onClose: () => void }) {
  const [r, setR] = useState<number | undefined>();
  return (
    <Modal title="Monopoly: name a resource" onClose={onClose}>
      <p className="muted">Every opponent hands you all of that resource.</p>
      <ResPicker value={r} onPick={setR} />
      <div className="modal-actions">
        <button className="btn primary" disabled={r === undefined} onClick={() => r !== undefined && onSubmit({ player, type: "PLAY_MONOPOLY", value: r })}>
          Claim {r !== undefined ? RES_LABEL[r] : ""}
        </button>
      </div>
    </Modal>
  );
}

export function Discard({ view, onSubmitAll }: { view: View; onSubmitAll: (cards: number[]) => void }) {
  const me = view.players[view.viewer!];
  const need = view.discard_remaining[view.viewer!];
  const [sel, setSel] = useState([0, 0, 0, 0, 0]);
  const total = sel.reduce((a, b) => a + b, 0);
  const hand = me.hand ?? [0, 0, 0, 0, 0];
  return (
    <Modal title={`Discard ${need} card${need === 1 ? "" : "s"}`} wide>
      <p className="muted">The robber strikes: you hold more than 7 cards. Choose which cards to give up.</p>
      <div className="discard-grid">
        {hand.map((n, r) => (
          <div key={r} className="discard-col">
            <ResIcon r={r} size={36} />
            <span>{RES_LABEL[r]}</span>
            <div className="stepper">
              <button aria-label={`Keep one more ${RES_LABEL[r]}`} disabled={sel[r] === 0} onClick={() => setSel((s) => s.map((x, k) => (k === r ? x - 1 : x)))}>−</button>
              <output data-res={r}>{sel[r]}</output>
              <button aria-label={`Discard one more ${RES_LABEL[r]}`} disabled={sel[r] >= n || total >= need} onClick={() => setSel((s) => s.map((x, k) => (k === r ? x + 1 : x)))} data-discard-plus={r}>+</button>
            </div>
            <small className="muted">of {n}</small>
          </div>
        ))}
      </div>
      <div className="modal-actions">
        <span className="muted">{total} / {need} chosen</span>
        <button className="btn primary" disabled={total !== need} onClick={() => onSubmitAll(sel.flatMap((n, r) => Array(n).fill(r)))}>Discard</button>
      </div>
    </Modal>
  );
}

/** Sidebar card, not a modal: the board stays visible, so you can see where the robber went and whose
 *  buildings touch it (DS3-10). Rendered in the right column like the trade cards. */
export function Steal({ view, legal, onSubmit }: { view: View; legal: Action[]; onSubmit: (a: Action) => void }) {
  return (
    <div className="trade-card steal-card" role="region" aria-label="Steal one card">
      <h3>Steal one card</h3>
      <p className="muted small">The robber stays visible on the board. Choose whose card to take; it is drawn at random from their hand.</p>
      <div className="steal-list">
        {legal.filter((a) => a.type === "STEAL").map((a) => {
          const p = a.value as number;
          return (
            <button key={p} className="steal" onClick={() => onSubmit(a)} style={{ ["--pc" as any]: PLAYER_COLORS[p] }} data-steal={p}>
              <CrestBadge p={p} size={34} />
              <span>{seatName(p, view.seats)}</span>
              <small>{view.players[p].resource_count} cards</small>
            </button>
          );
        })}
      </div>
    </div>
  );
}

export function PassDevice({ seat, view, onReady }: { seat: number; view: View; onReady: () => void }) {
  const box = useRef<HTMLDivElement>(null);
  useFocusTrap(box);  // the cover hides hands; Tab must not reach the controls behind it
  return (
    <div ref={box} className="pass-screen" style={{ ["--pc" as any]: PLAYER_COLORS[seat] }} role="dialog" aria-modal="true"
      aria-label={`Pass the device to ${seatName(seat, view.seats)}`} tabIndex={-1}>
      <CrestBadge p={seat} size={84} />
      <h2>Pass the device to {seatName(seat, view.seats)}</h2>
      <p>Everyone else, look away. Hands stay hidden until {seatName(seat, view.seats)} is ready.</p>
      <button className="btn primary big" onClick={onReady} autoFocus>I'm {seatName(seat, view.seats)}, show my hand</button>
    </div>
  );
}

const plural = (n: number, one: string, many = one + "s") => `${n} ${n === 1 ? one : many}`;

export function Victory({ view, onReplay, onNew, onMenu, onClose, onAnalyze }: { view: View; onReplay: () => void; onNew: () => void; onMenu: () => void; onClose: () => void; onAnalyze?: () => void }) {
  const ranked = [...view.players].sort((a, b) => (b.vp ?? b.public_vp) - (a.vp ?? a.public_vp));
  const w = view.winner;
  const vpCards = (p: typeof ranked[number]) => p.dev_cards?.[1] ?? 0;
  return (
    <Modal title={w >= 0 ? `${seatName(w, view.seats)} settles the island` : "The game stopped at the turn limit"} wide onClose={onClose}>
      <table className="scores">
        <thead><tr><th>Player</th><th>Buildings</th><th>Awards</th><th>Hidden VP</th><th>Total</th></tr></thead>
        <tbody>
          {ranked.map((p) => (
            <tr key={p.id} className={p.id === w ? "winner" : ""}>
              <td><span className="who"><CrestBadge p={p.id} size={22} /> {seatName(p.id, view.seats, true)}</span></td>
              <td>{plural(p.settlements.length, "settlement")}, {plural(p.cities.length, "city", "cities")}</td>
              <td>{[view.longest_road_owner === p.id && "Longest Road", view.largest_army_owner === p.id && "Largest Army"].filter(Boolean).join(", ") || "—"}</td>
              <td>{vpCards(p) ? `+${vpCards(p)} revealed` : "—"}</td>
              <td className="total">{p.vp ?? p.public_vp}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="muted">Turn {turnNumber(view.turn, true, (view as any).truncated)}. {view.action_count} actions.</p>
      <div className="modal-actions">
        <button className="btn ghost" onClick={onMenu}>Main menu</button>
        <button className="btn" onClick={onReplay}>Watch replay</button>
        {onAnalyze && <button className="btn" onClick={onAnalyze} data-testid="analyze-this-game">Analyze this game</button>}
        <button className="btn primary" onClick={onNew}>Play again</button>
      </div>
    </Modal>
  );
}

export function eventToast(view: View) {
  const last = view.events[view.events.length - 1];
  return last ? describeEvent(last, view.seats) : null;
}
