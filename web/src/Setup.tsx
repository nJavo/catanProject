import { useEffect, useLayoutEffect, useState } from "react";
import { api, checkpointFor } from "./api";
import { go } from "./App";
import { PLAYER_COLORS, PLAYER_INK, PLAYER_NAMES, CREST } from "./rules";
import { Crest } from "./board/Board";

export interface SeatChoice {
  type: "human" | "agent";
  agent: string;
  checkpoint?: string;
}

export function CrestBadge({ p, size = 22 }: { p: number; size?: number }) {
  return (
    <svg width={size} height={size} viewBox="-0.5 -0.5 1 1" className="crest" aria-hidden>
      <circle r={0.47} fill={PLAYER_COLORS[p]} stroke={PLAYER_INK[p]} strokeWidth={0.07} />
      <Crest shape={CREST[p]} s={0.22} fill={PLAYER_INK[p]} />
    </svg>
  );
}

// Server-chosen checkpoint behind "Default" (null: the agent's own default).
// Set before setCheckpoints, so every AgentSelect re-renders with it.
let defaultCheckpoint: string | null = null;

export function AgentSelect({ value, agents, onChange, checkpoints, checkpoint, onCheckpoint }: {
  value: string; agents: string[]; onChange: (a: string) => void;
  checkpoints: string[]; checkpoint?: string; onCheckpoint: (c: string) => void;
}) {
  const [champ, setChamp] = useState<any>(null); // the gate-promoted model's summary
  const hasChamp = agents.includes("champion");
  useEffect(() => { if (hasChamp) fetch("/api/registry").then((r) => r.json()).then((r) => setChamp(r.champion ?? null)).catch(() => {}); }, [hasChamp]);
  // The option stays short so the select fits the seat row; the gate evidence goes in the tooltip.
  const champTitle = champ ? `${champ.label}. Win rate as 1 seat vs 3 opponents; chance is 25%.` : undefined;
  return (
    <>
      <select value={value} onChange={(e) => onChange(e.target.value)} aria-label="AI agent" title={value === "champion" ? champTitle : undefined}>
        {agents.map((a) => <option key={a} value={a} title={a === "champion" ? champTitle : undefined}>{a === "champion" ? `Champion${champ ? ` — ${champ.name}` : ""}` : a}</option>)}
      </select>
      {value === "rl" && (
        <select value={checkpoint ?? ""} onChange={(e) => onCheckpoint(e.target.value)} aria-label="Checkpoint">
          <option value="">Default checkpoint{defaultCheckpoint ? ` (${defaultCheckpoint.replace(/^runs\//, "")})` : ""}</option>
          {checkpoints.map((c) => <option key={c} value={c}>{c.replace(/^runs\//, "")}</option>)}
        </select>
      )}
    </>
  );
}

export function useAgents() {
  const [agents, setAgents] = useState<string[]>(["random"]);
  const [checkpoints, setCheckpoints] = useState<string[]>([]);
  const [agentsErr, setAgentsErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(true); // DS18: until /api/agents answers, seats only hold the "random" placeholder
  useEffect(() => {
    api.agents().then((r) => setAgents(r.agents)).finally(() => setLoading(false)).catch((e) => setAgentsErr(`Can't load the AI agents (${e.message}), so only random is offered. Is the server running?`));
    api.checkpoints().then((r) => { defaultCheckpoint = r.default ?? null; setCheckpoints(r.checkpoints); }).catch(() => {});
  }, []);
  return { agents, checkpoints, agentsErr, loading };
}

// One line per agent so a first-time player can choose without knowing the registry names.
export const AGENT_HINT: Record<string, string> = {
  random: "Plays random legal moves. For testing, not a real opponent.",
  basic: "Simple rule-based bot. The easiest real opponent.",
  intermediate: "Stronger rule-based bot. A fair first game.",
  intermediate_trader: "The intermediate bot, plus trade offers to other players.",
  search: "Rule-based bot that simulates ahead before each move. Strong, a little slower.",
  search_trader: "The search bot, plus trade offers to other players.",
  champion: "The trained model that last passed the promotion gate.",
  jointb_gated: "The champion's agent: value-net search plus a learned trade model. Slow.",
  rl: "A trained reinforcement-learning policy. Pick a checkpoint.",
  rl_greedy: "Scripted baseline: builds greedily and never trades.",
};
export const agentHint = (a: string) => AGENT_HINT[a] ?? "No description yet.";

export const seatPayload = (s: SeatChoice) =>
  s.type === "human" ? { type: "human" } : { type: "agent", agent: s.agent, kwargs: checkpointFor(s.agent, s.checkpoint) ? { checkpoint: checkpointFor(s.agent, s.checkpoint) } : {} };

export default function Setup() {
  const { agents, checkpoints, agentsErr, loading } = useAgents();
  const [n, setN] = useState(4);
  const [seats, setSeats] = useState<SeatChoice[]>([
    { type: "human", agent: "random" }, { type: "agent", agent: "random" },
    { type: "agent", agent: "random" }, { type: "agent", agent: "random" },
  ]);
  const [seed, setSeed] = useState("");
  const [vpText, setVpText] = useState("10"); // clamped on blur: clamping per keystroke turned a typed "12" into 20
  const vp = Math.max(3, Math.min(20, Number(vpText) || 10));
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [serverOnline, setServerOnline] = useState(false); // started with --online (docs/ONLINE.md)
  const [online, setOnline] = useState(false);
  useEffect(() => { api.health().then((h) => setServerOnline(!!h.online)).catch(() => {}); }, []);
  const best = agents.includes("intermediate") ? "intermediate" : agents.includes("basic") ? "basic" : agents[0];
  // layout effect: no painted frame pairs the loaded agent list with the random placeholder (DS17-10)
  useLayoutEffect(() => {
    setSeats((s) => s.map((x) => (x.type === "agent" && x.agent === "random" && best !== "random" ? { ...x, agent: best } : x)));
  }, [best]);

  const upd = (i: number, patch: Partial<SeatChoice>) => setSeats((s) => s.map((x, k) => (k === i ? { ...x, ...patch } : x)));
  const used = seats.slice(0, n);
  const humans = used.filter((s) => s.type === "human").length;

  async function start() {
    setBusy(true);
    setErr(null);
    try {
      const body: any = { players: used.map(seatPayload), vp_to_win: vp };
      if (seed.trim()) body.seed = Number(seed);
      if (online) {
        const r = await api.createRoom(body);
        go(`/room/${r.id}/${r.host_token}`);
      } else if (humans === 0) {
        const r = await api.labMatch({ agents: used.map((s) => ({ agent: s.agent, kwargs: seatPayload(s).kwargs as any })), seed: body.seed, vp_to_win: vp, max_turns: 500 });
        go(`/lab/match/${r.id}`);
      } else {
        const r = await api.createGame(body);
        go(`/game/${r.id}`);
      }
    } catch (e: any) {
      setErr(e.message);
      setBusy(false);
    }
  }

  return (
    <main className="page setup">
      <header className="page-head">
        <button className="btn ghost" onClick={() => go("/")}>Back</button>
        <h1>New game</h1>
      </header>
      <section className="setup-grid">
        <div className="field">
          <span className="field-label">Players</span>
          <div className="segmented" role="radiogroup" aria-label="Number of players">
            {[2, 3, 4].map((k) => (
              <button key={k} role="radio" aria-checked={n === k} className={n === k ? "on" : ""} onClick={() => setN(k)}>{k}</button>
            ))}
          </div>
          {n === 2 && <p className="note">Two players is an unofficial variant: standard rules, two seats.</p>}
        </div>
        {agents.includes("champion") && (
          <p className="note" data-testid="champion-quick">
            Want a challenge? The Champion is the AI model that last passed the promotion gate.{" "}
            <button className="btn small" onClick={() => setSeats((x) => x.map((y) => (y.type === "agent" ? { ...y, agent: "champion" } : y)))} data-testid="all-champion">Make every AI seat the Champion</button>
          </p>
        )}
        <label className="field online-toggle" title={serverOnline ? undefined : "Start the server with --online to enable this (docs/ONLINE.md)"}>
          <span><input type="checkbox" checked={online} disabled={!serverOnline} onChange={(e) => setOnline(e.target.checked)} data-testid="play-online" /> Play online with friends</span>
          <span className="muted small">{serverOnline ? "Each human seat gets its own secret link. Friends play on their own devices; AI seats run here." : "Needs the server started with --online."}</span>
        </label>
        <ol className="seat-list">
          {used.map((s, i) => (
            <li key={i} className="seat-row" style={{ ["--pc" as any]: PLAYER_COLORS[i] }}>
              <CrestBadge p={i} size={30} />
              <strong className="seat-name">{PLAYER_NAMES[i]}</strong>
              <div className="segmented small" role="radiogroup" aria-label={`Seat ${i + 1} controller`}>
                <button role="radio" aria-checked={s.type === "human"} className={s.type === "human" ? "on" : ""} onClick={() => upd(i, { type: "human" })}>Human</button>
                <button role="radio" aria-checked={s.type === "agent"} className={s.type === "agent" ? "on" : ""} onClick={() => upd(i, { type: "agent" })}>AI</button>
              </div>
              {s.type === "agent" ? (
                <AgentSelect value={s.agent} agents={agents} onChange={(a) => upd(i, { agent: a })}
                  checkpoints={checkpoints} checkpoint={s.checkpoint} onCheckpoint={(c) => upd(i, { checkpoint: c })} />
              ) : <span className="muted">{online ? "Gets a seat link" : humans > 1 ? "Hot-seat: pass the device on this seat's turn" : "You"}</span>}
              {s.type === "agent" && !loading && <span className="muted small seat-hint" data-testid="agent-hint">{agentHint(s.agent)}</span>}
            </li>
          ))}
        </ol>
        <div className="field-row">
          <label className="field">
            <span className="field-label">Seed</span>
            <input inputMode="numeric" placeholder="Random" value={seed} onChange={(e) => setSeed(e.target.value.replace(/\D/g, ""))} />
          </label>
          <label className="field">
            <span className="field-label">Points to win</span>
            <input type="number" min={3} max={20} value={vpText} onChange={(e) => setVpText(e.target.value)} onBlur={() => setVpText(String(vp))} />
          </label>
        </div>
        {humans === 0 && <p className="note">{online ? "An online room needs at least one human seat." : "No human seats: this opens the AI match viewer."}</p>}
        {agentsErr && <p className="error" role="alert">{agentsErr}</p>}
        {err && <p className="error" role="alert">{err}</p>}
        <div className="setup-go">
          <button className="btn primary big" disabled={busy || loading || (online && humans === 0)} onClick={start} data-testid="start-game">{loading ? "Loading AI agents…" : online ? "Create room" : humans === 0 ? "Watch AI match" : "Start game"}</button>
        </div>
      </section>
    </main>
  );
}
