import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, Action, Topology, View } from "../api";
import Board, { Targets } from "../board/Board";
import { go } from "../App";
import { COST, PHASE_LABEL, PLAYER_COLORS, seatName, turnNumber } from "../rules";
import { CrestBadge } from "../Setup";
import { Award, Bank, Cost, Dice, DevCards, Log, OpponentPanel, ResCard } from "./Hud";
import { Discard, Monopoly, PassDevice, Steal, Victory, YearOfPlenty } from "./Dialogs";
import { MaritimeTrade, TradeComposer, TradeConfirm, TradeResponse, TradeWaiting } from "./Trade";

type Mode = null | "road" | "settlement" | "city";
type Dialog = null | "yop" | "monopoly" | "offer" | "counter" | "maritime";

const POLL_MS = 1500;
const SPEEDS = [{ label: "Slow", ms: 1100 }, { label: "Normal", ms: 550 }, { label: "Fast", ms: 180 }, { label: "Instant", ms: 0 }];

function instruction(view: View, mine: boolean, mode: Mode, online = false): string {
  const who = view.actor !== null ? seatName(view.actor, view.seats) : "";
  if (view.phase === "GAME_OVER") return "Game over";
  if (!mine && online) return `Waiting for ${who}`;
  if (!mine) {
    if (view.phase === "TRADE_RESPONSE") return `${who} is considering a trade`;
    if (view.phase === "DISCARD") return `${who} is discarding`;
    return `${who} is taking their turn`;
  }
  const setupRound = view.vertex_owner.filter((o) => o === view.actor).length;
  switch (view.phase) {
    case "SETUP_SETTLEMENT": return setupRound === 0 ? "Place your first settlement" : "Place your second settlement. It earns your starting hand";
    case "SETUP_ROAD": return "Place a road next to that settlement";
    // DS27-03: only mention dev cards when one is playable; with none in hand the hint pointed at nothing
    case "PRE_ROLL": return view.legal_actions.some((a) => a.type.startsWith("PLAY_")) ? "Roll the dice, or play a development card first" : "Roll the dice";
    case "DISCARD": { const n = view.discard_remaining[view.actor!]; return `Discard ${n} card${n === 1 ? "" : "s"}`; }
    case "MOVE_ROBBER": return "Move the robber to a new tile";
    case "STEAL": return "Choose a player to steal from";
    case "ROAD_BUILDING": return `Place ${view.free_roads} free road${view.free_roads === 1 ? "" : "s"}`;
    case "TRADE_RESPONSE": return "Answer the trade offer";
    case "TRADE_CONFIRM": return "Pick a trading partner, or cancel";
    case "MAIN":
      if (mode === "road") return "Pick where to build a road";
      if (mode === "settlement") return "Pick where to build a settlement";
      if (mode === "city") return "Pick a settlement to upgrade";
      return "Build, trade, or end your turn";
  }
  return "";
}

/** ``lobby``: an online room's lobby route. The server then fixes the viewer to this link's seat, so there is no hot-seat hand-off. */
export default function GameScreen({ id, topo, lobby }: { id: string; topo: Topology; lobby?: string }) {
  const online = lobby !== undefined;
  const [view, setView] = useState<View | null>(null);
  const [viewer, setViewer] = useState<number | "all" | null>(null);
  const [gate, setGate] = useState<number | null>(null); // hot-seat: seat waiting to take the device
  const [mode, setMode] = useState<Mode>(null);
  const [dialog, setDialog] = useState<Dialog>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [paused, setPaused] = useState(false);
  const [hideVictory, setHideVictory] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const humansRef = useRef<number[]>([]);
  // Synchronous guards: React state lags a fast double-click, refs do not.
  const inflight = useRef(false);
  const shownCount = useRef(-1);
  const shownKey = useRef("");
  const viewerRef = useRef<number | "all" | null>(null);
  viewerRef.current = viewer;

  const apply = useCallback((v: View) => {
    // A slow response must never roll the table back to an older state.
    if (v.action_count < shownCount.current) return;
    const key = `${v.viewer}:${v.action_count}:${v.event_seq}`;
    if (key === shownKey.current) return; // unchanged (typical poll): skip the re-render
    shownCount.current = v.action_count;
    shownKey.current = key;
    setView(v);
    const humans = v.seats.map((s, i) => (s.type === "human" ? i : -1)).filter((i) => i >= 0);
    humansRef.current = humans;
    if (!online && v.actor !== null && v.seats[v.actor].type === "human" && v.viewer !== v.actor) {
      if (humans.length > 1) setGate(v.actor);
      else setViewer(v.actor);
    }
  }, [online]);

  // initial load: pick the first human seat (or spectate)
  useEffect(() => {
    api.view(id, null).then((pub) => {
      if (online) { setViewer(pub.viewer); return; } // the server answered with this link's seat
      const humans = pub.seats.map((s, i) => (s.type === "human" ? i : -1)).filter((i) => i >= 0);
      const first = humans.length === 0 ? "all" : pub.actor !== null && humans.includes(pub.actor) ? pub.actor : humans[0];
      setViewer(first);
      if (humans.length > 1 && typeof first === "number") setGate(first);
    }).catch((e) => setErr(e.message));
  }, [id, online]);

  useEffect(() => {
    if (viewer === null) return;
    api.view(id, viewer).then(apply).catch((e) => setErr(e.message));
  }, [id, viewer, apply]);

  // Live updates: another tab or device may move this game. Polls are
  // dropped if an action is in flight or the viewer changed meanwhile
  // (hot-seat hand-off), so they can never undo a newer local state.
  const over = view?.phase === "GAME_OVER";
  useEffect(() => {
    if (viewer === null || over) return;
    const t = setInterval(() => {
      if (document.hidden || inflight.current) return;
      const asked = viewer;
      api.view(id, asked).then((v) => {
        if (viewerRef.current === asked && !inflight.current) apply(v);
      }).catch(() => {});
    }, POLL_MS);
    return () => clearInterval(t);
  }, [id, viewer, over, apply]);

  // AI seats: step one action at a time so the table animates
  useEffect(() => {
    if (!view || view.awaiting !== "agent" || paused || busy) return;
    const t = setTimeout(() => {
      if (inflight.current) return;
      inflight.current = true;
      api.advance(id, viewer, SPEEDS[speed].ms === 0 ? 50 : 1).then((r) => apply(r.view)).catch((e) => setErr(e.message))
        .finally(() => { inflight.current = false; });
    }, SPEEDS[speed].ms);
    return () => clearTimeout(t);
  }, [view, paused, speed, busy, id, viewer, apply]);

  const submit = useCallback(async (a: Action) => {
    if (inflight.current) return;
    inflight.current = true;
    setBusy(true);
    setErr(null);
    try {
      const r = await api.act(id, a);
      setMode(null);
      setDialog(null);
      apply(r.view);
    } catch (e: any) {
      setErr(e.message);
    } finally {
      inflight.current = false;
      setBusy(false);
    }
  }, [id, apply]);

  const submitMany = useCallback(async (acts: Action[]) => {
    if (inflight.current) return;
    inflight.current = true;
    setBusy(true);
    try {
      for (const a of acts) {
        const r = await api.act(id, a);
        apply(r.view);
      }
    } catch (e: any) {
      setErr(e.message);
    } finally {
      inflight.current = false;
      setBusy(false);
    }
  }, [id, apply]);

  const save = useCallback(async () => {
    try {
      const r = await api.saveGame(id);
      setNotice(`Saved as "${r.name}". Resume it from the main menu.`);
      setTimeout(() => setNotice(null), 4000);
    } catch (e: any) { setErr(e.message); }
  }, [id]);

  const legal = view?.legal_actions ?? [];
  const types = useMemo(() => new Set(legal.map((a) => a.type)), [legal]);
  const mine = !!view && view.actor !== null && view.actor === view.viewer && gate === null;

  // escape leaves build mode; modes that become illegal reset
  useEffect(() => {
    const k = (e: KeyboardEvent) => e.key === "Escape" && setMode(null);
    window.addEventListener("keydown", k);
    return () => window.removeEventListener("keydown", k);
  }, []);
  useEffect(() => {
    if (mode && !types.has({ road: "BUILD_ROAD", settlement: "BUILD_SETTLEMENT", city: "BUILD_CITY" }[mode])) setMode(null);
  }, [types, mode]);
  // Build mode entered from the keyboard: focus the first legal spot (the board sits ~15 Tabs before the dock),
  // and on leaving it (build, Esc) hand focus back to the build button, or End turn once that is disabled (DS10-06).
  const kbMode = useRef<Mode>(null);
  useEffect(() => {
    const m = kbMode.current;
    if (!m) return;
    if (mode === m) { document.querySelector<SVGElement>(".board .edge-target, .board .vertex-target")?.focus({ preventScroll: true }); return; }
    kbMode.current = null;
    if (document.activeElement !== document.body && document.activeElement !== null) return;
    const btn = [`[data-testid="build-${m}"]`, '[data-testid="end-turn"]'].map((q) => document.querySelector<HTMLButtonElement>(q)).find((b) => b && !b.disabled);
    btn?.focus({ preventScroll: true });
  }, [mode]);

  const targets: Targets | undefined = useMemo(() => {
    if (!view || !mine) return undefined;
    const by = (t: string) => new Map(legal.filter((a) => a.type === t).map((a) => [a.value as number, a]));
    const seat = view.viewer!;
    switch (view.phase) {
      case "SETUP_SETTLEMENT": return { vertices: by("BUILD_SETTLEMENT"), ghost: "settlement", seat };
      case "SETUP_ROAD": case "ROAD_BUILDING": return { edges: by("BUILD_ROAD"), ghost: "road", seat };
      case "MOVE_ROBBER": return { hexes: by("MOVE_ROBBER"), ghost: "robber", seat };
      case "MAIN":
        if (mode === "road") return { edges: by("BUILD_ROAD"), ghost: "road", seat };
        if (mode === "settlement") return { vertices: by("BUILD_SETTLEMENT"), ghost: "settlement", seat };
        if (mode === "city") return { vertices: by("BUILD_CITY"), ghost: "city", seat };
    }
    return undefined;
  }, [view, mine, legal, mode]);

  if (err && !view) return <div className="fatal"><h1>Game unavailable</h1><p>{err}</p><button className="btn" onClick={() => go("/")}>Main menu</button></div>;
  if (!view) return <div className="fatal"><p>Loading game…</p></div>;

  const meSeat = typeof view.viewer === "number" ? view.viewer : null;
  const me = meSeat !== null ? view.players[meSeat] : null;
  const actorColor = view.actor !== null ? PLAYER_COLORS[view.actor] : "#999";
  const rollSeq = [...view.events].reverse().find((e) => e.type === "dice_rolled")?.seq ?? 0;
  const humanCount = view.seats.filter((s) => s.type === "human").length;
  const aiCount = view.seats.length - humanCount;
  const playDev = (t: number) => {
    if (t === 0) submit({ player: meSeat!, type: "PLAY_KNIGHT" });
    else if (t === 2) submit({ player: meSeat!, type: "PLAY_ROAD_BUILDING" });
    else if (t === 3) setDialog("yop");
    else if (t === 4) setDialog("monopoly");
  };
  // detail 0: activated by Enter or Space, not a pointer
  const toggle = (m: Mode, e: React.MouseEvent) => { kbMode.current = e.detail === 0 && mode !== m ? m : null; setMode((x) => (x === m ? null : m)); };
  const inMain = mine && view.phase === "MAIN";
  const vpHidden = me?.dev_cards?.[1] ?? 0;

  return (
    <div className="game" style={{ ["--actor" as any]: actorColor }}>
      <header className="topbar">
        <h1 className="sr-only">Game</h1>
        <button className="btn ghost small" onClick={() => go(lobby ?? "/")} aria-label={online ? "Back to the room" : "Leave game"}>{online ? "Room" : "Menu"}</button>
        {view.phase !== "GAME_OVER" && !online && <button className="btn ghost small" onClick={save} data-testid="save-game" title="Save this game to resume later">Save</button>}
        <div className="banner" aria-live="polite">
          {view.actor !== null && <CrestBadge p={view.actor} size={30} />}
          <div className="banner-text">
            <span className="banner-phase">{PHASE_LABEL[view.phase]} · Turn {turnNumber(view.turn, view.phase === "GAME_OVER", (view as any).truncated)}</span>
            <strong className="banner-instr" data-testid="instruction">{instruction(view, mine, mode, online)}</strong>
          </div>
        </div>
        <Dice roll={view.last_roll} rollSeq={rollSeq} />
        {aiCount > 0 && !over && !online && (
          <div className="ai-speed" role="group" aria-label="AI speed">
            {SPEEDS.map((s, i) => <button key={s.label} className={"chip" + (speed === i ? " on" : "")} onClick={() => setSpeed(i)} data-speed={s.label}>{s.label}</button>)}
            <button className="chip" onClick={() => setPaused((p) => !p)}>{paused ? "Resume AI" : "Pause AI"}</button>
          </div>
        )}
      </header>

      <aside className="left">
        <h2 className="sr-only">Opponents and bank</h2>
        {view.players.filter((p) => p.id !== meSeat).map((p) => (
          <OpponentPanel key={p.id} p={p} view={view} active={view.actor === p.id} seats={view.seats} />
        ))}
        <Bank view={view} />
      </aside>

      <main className="board-wrap">
        <h2 className="sr-only">Board</h2>
        <Board topo={topo} view={view} targets={targets} onPick={busy ? undefined : submit} />
        {err && <div className="toast error" role="alert" onClick={() => setErr(null)}>{err}</div>}
        {notice && <div className="toast notice" role="status" onClick={() => setNotice(null)}>{notice}</div>}
      </main>

      <aside className="right">
        <h2 className="sr-only">Trades and game log</h2>
        {mine && view.phase === "STEAL" && <Steal view={view} legal={legal} onSubmit={submit} />}
        {view.trade && mine && view.phase === "TRADE_RESPONSE" && dialog !== "counter" && (
          <TradeResponse view={view} legal={legal} onSubmit={submit} onCounter={() => setDialog("counter")} busy={busy} />
        )}
        {view.trade && mine && view.phase === "TRADE_CONFIRM" && <TradeConfirm view={view} legal={legal} onSubmit={submit} busy={busy} />}
        {view.trade && !mine && <TradeWaiting view={view} />}
        <Log events={view.events} seats={view.seats} layout={view.layout} />
      </aside>

      <footer className="dock">
        <h2 className="sr-only">{me ? "Your hand and actions" : "Spectating"}</h2>
        {me ? (
          <>
            <div className="me" style={{ ["--pc" as any]: PLAYER_COLORS[meSeat!] }}>
              <div className="me-head">
                <CrestBadge p={meSeat!} size={34} />
                <div>
                  <strong>{seatName(meSeat!, view.seats)}</strong>
                  <span className="me-vp" data-testid="my-vp">{me.vp ?? me.public_vp}{"\u00a0"}VP<small>{"\u00a0"}/ {view.vp_to_win}</small>{vpHidden ? <small> ({me.public_vp} public + {vpHidden} hidden)</small> : null}</span>
                </div>
              </div>
              <div className="supply" title="Pieces left in your supply">
                <span>Left:</span><span>{me.supply.road} road{me.supply.road === 1 ? "" : "s"}</span><span>{me.supply.settlement} settlement{me.supply.settlement === 1 ? "" : "s"}</span><span>{me.supply.city} {me.supply.city === 1 ? "city" : "cities"}</span>
              </div>
              <div className="me-awards">
                {view.longest_road_owner === meSeat && <Award kind="road" holder />}
                {view.largest_army_owner === meSeat && <Award kind="army" holder />}
                <span className="muted small">Road {me.road_length} · Knights {me.knights_played}</span>
              </div>
            </div>
            <div className="hand" aria-label="Your hand">
              {(me.hand ?? [0, 0, 0, 0, 0]).map((n, r) => <ResCard key={r} r={r} n={n} />)}
            </div>
            <DevCards me={me} legalTypes={mine ? types : new Set()} onPlay={playDev} />
            <div className="actions">
              {mine && view.phase === "PRE_ROLL" && (
                <button className="btn primary big roll" onClick={() => submit({ player: meSeat!, type: "ROLL" })} disabled={busy} data-testid="roll"
                  // after End turn and the bots' turns focus is on <body>: put it on Roll (DS9-03), never take it from elsewhere
                  ref={(el) => { if (el && document.activeElement === document.body) el.focus({ preventScroll: true }); }}>Roll dice</button>
              )}
              {view.phase === "GAME_OVER" ? hideVictory && (
                // in the dock, not floating: a fixed button covered Offer trade / End turn at 1024px
                <button className="btn primary big" onClick={() => setHideVictory(false)}>Final scores</button>
              ) : (view.phase === "MAIN" || !mine) && (
                <>
                  <div className="build">
                    <button className={"btn build-btn" + (mode === "road" ? " on" : "")} disabled={!inMain || !types.has("BUILD_ROAD")} onClick={(e) => toggle("road", e)} data-testid="build-road">Road<Cost cost={COST.road} /></button>
                    <button className={"btn build-btn" + (mode === "settlement" ? " on" : "")} disabled={!inMain || !types.has("BUILD_SETTLEMENT")} onClick={(e) => toggle("settlement", e)} data-testid="build-settlement">Settlement<Cost cost={COST.settlement} /></button>
                    <button className={"btn build-btn" + (mode === "city" ? " on" : "")} disabled={!inMain || !types.has("BUILD_CITY")} onClick={(e) => toggle("city", e)} data-testid="build-city">City<Cost cost={COST.city} /></button>
                    <button className="btn build-btn" disabled={!inMain || !types.has("BUY_DEV") || busy} onClick={() => submit({ player: meSeat!, type: "BUY_DEV" })} data-testid="buy-dev">Dev card<Cost cost={COST.dev} /></button>
                  </div>
                  <div className="trade-btns">
                    <button className="btn" disabled={!inMain || !types.has("MARITIME_TRADE")} onClick={() => setDialog("maritime")} data-testid="bank-trade">Bank trade</button>
                    <button className="btn" disabled={!inMain || !types.has("OFFER_TRADE")} onClick={() => setDialog("offer")} data-testid="offer-trade">Offer trade</button>
                    <button className="btn primary" disabled={!inMain || busy} onClick={() => submit({ player: meSeat!, type: "END_TURN" })} data-testid="end-turn">End turn</button>
                  </div>
                </>
              )}
            </div>
          </>
        ) : (
          <div className="spectate muted">Spectating. All seats are AI.</div>
        )}
      </footer>

      {mine && view.phase === "DISCARD" && (
        <Discard key={view.discard_remaining[meSeat!] + ":" + view.action_count} view={view}
          onSubmitAll={(cards) => submitMany(cards.map((r) => ({ player: meSeat!, type: "DISCARD", value: r })))} />
      )}
      {dialog === "yop" && <YearOfPlenty legal={legal} onSubmit={submit} onClose={() => setDialog(null)} />}
      {dialog === "monopoly" && <Monopoly player={meSeat!} onSubmit={submit} onClose={() => setDialog(null)} />}
      {dialog === "maritime" && <MaritimeTrade view={view} legal={legal} onSubmit={submit} onClose={() => setDialog(null)} />}
      {dialog === "offer" && <TradeComposer view={view} mode="offer" onSubmit={submit} onClose={() => setDialog(null)} />}
      {dialog === "counter" && view.phase === "TRADE_RESPONSE" && <TradeComposer view={view} mode="counter" onSubmit={submit} onClose={() => setDialog(null)} />}
      {gate !== null && view.phase !== "GAME_OVER" && (
        <PassDevice seat={gate} view={view} onReady={() => { setViewer(gate); setGate(null); }} />
      )}
      {view.phase === "GAME_OVER" && !hideVictory && (
        <Victory view={view} onClose={() => setHideVictory(true)} onReplay={() => go(`/replay/game/${id}`)} onAnalyze={() => go(`/analysis/game/${id}`)} onNew={() => go("/new")} onMenu={() => go("/")} />
      )}
      {view.phase === "GAME_OVER" && hideVictory && !me && <button className="btn primary floating" onClick={() => setHideVictory(false)}>Final scores</button>}
      {mode && <div className="mode-hint">Building a {mode}. Click a highlighted spot (or Tab to one and press Enter), or press Esc. <button className="btn small ghost" onClick={() => setMode(null)}>Cancel</button></div>}
    </div>
  );
}
