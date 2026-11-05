import { useEffect, useState } from "react";
import { api, Lobby } from "./api";
import { go } from "./App";
import { CrestBadge } from "./Setup";
import { PLAYER_COLORS, PLAYER_NAMES, seatName } from "./rules";

// Online room lobby (docs/ONLINE.md). The token in the URL decides what this page shows:
// the host link lists every seat link; a seat link lets that player set a name and sit down.
export default function Room({ id, token }: { id: string; token: string }) {
  const [lobby, setLobby] = useState<Lobby | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [name, setName] = useState<string | null>(null); // null: not edited yet, show the saved name
  const [saved, setSaved] = useState(false);
  const [base, setBase] = useState(window.location.origin);
  const [copied, setCopied] = useState<number | null>(null);

  useEffect(() => {
    const load = () => api.lobby(id).then((l) => { setLobby(l); setErr(null); })
      .catch((e) => setErr(e.status === 403 || e.status === 404 ? "This room link is not valid. Ask the host for a new one." : e.message));
    load();
    const t = setInterval(() => { if (!document.hidden) load(); }, 2000);
    return () => clearInterval(t);
  }, [id, token]);

  if (!lobby) return <div className="fatal">{err ? <><h1>Room unavailable</h1><p>{err}</p></> : <p>Opening the room…</p>}</div>;
  const host = lobby.you === "host";
  const me = typeof lobby.you === "number" ? lobby.you : null;
  const link = (t: string) => `${base.replace(/\/+$/, "")}/#/room/${id}/${t}`;
  const copy = (i: number, text: string) => {
    navigator.clipboard?.writeText(text).then(() => { setCopied(i); setTimeout(() => setCopied(null), 2000); }).catch(() => {});
  };
  const saveName = async () => {
    try {
      await api.setName(id, (name ?? "").trim());
      setSaved(true);
      setName(null);
      setLobby(await api.lobby(id));
    } catch (e: any) { setErr(e.message); }
  };
  const started = lobby.phase !== "SETUP_SETTLEMENT" || lobby.turn > 0;

  return (
    <main className="page setup room">
      <header className="page-head">
        <button className="btn ghost" onClick={() => go("/")}>Back</button>
        <h1>{host ? "Online room" : me !== null ? `You play ${seatName(me, lobby.seats)}` : "Online room"}</h1>
      </header>
      <section className="setup-grid">
        {host && (
          <>
            <p className="note">Send each friend their own link. Anyone who has a link can play that seat, so share it only with that person. Keep this page's address to yourself: it shows every link.</p>
            <label className="field">
              <span className="field-label">Address your friends use</span>
              <input value={base} onChange={(e) => setBase(e.target.value)} placeholder="https://your-tunnel.example" data-testid="room-base" />
              <span className="muted small">Your Tailscale or tunnel address (docs/ONLINE.md). 127.0.0.1 only works on this computer.</span>
            </label>
          </>
        )}
        {me !== null && (
          <label className="field">
            <span className="field-label">Your name</span>
            <span className="room-name">
              <input value={name ?? lobby.seats[me].name ?? ""} maxLength={24} placeholder={PLAYER_NAMES[me]}
                onChange={(e) => { setName(e.target.value); setSaved(false); }} onKeyDown={(e) => e.key === "Enter" && name?.trim() && saveName()} data-testid="room-name" />
              <button className="btn" disabled={!name?.trim()} onClick={saveName} data-testid="room-name-save">Save</button>
            </span>
            {saved && <span className="muted small" role="status">Saved. Everyone now sees this name.</span>}
          </label>
        )}
        <ol className="seat-list room-seats" aria-label="Seats">
          {lobby.seats.map((s, i) => (
            <li key={i} className="room-seat" style={{ ["--pc" as any]: PLAYER_COLORS[i] }} data-testid={`room-seat-${i}`}>
              <CrestBadge p={i} size={30} />
              <div className="room-seat-main">
                <strong>{seatName(i, lobby.seats)}{i === me ? " (you)" : ""}</strong>
                <span className={"muted small room-status" + (s.connected ? " on" : "")} data-testid={`room-status-${i}`}>
                  {s.type === "agent" ? "AI, plays on the server" : s.connected ? "Connected" : "Not connected yet"}
                </span>
                {host && lobby.tokens?.[i] && (
                  <span className="room-link">
                    <input readOnly value={link(lobby.tokens[i]!)} onFocus={(e) => e.target.select()} aria-label={`Link for ${PLAYER_NAMES[i]}`} data-testid={`room-link-${i}`} />
                    <button className="btn small" onClick={() => copy(i, link(lobby.tokens![i]!))}>{copied === i ? "Copied" : "Copy"}</button>
                  </span>
                )}
              </div>
            </li>
          ))}
        </ol>
        {err && <p className="error" role="alert">{err}</p>}
        {me !== null && (
          <div className="setup-go">
            <button className="btn primary big" onClick={() => go(`/room/${id}/${token}/play`)} data-testid="take-seat">{started ? "Rejoin the game" : "Take your seat"}</button>
          </div>
        )}
        {host && <p className="muted small">To play a seat yourself, open its link. Bots move when a player's screen is open.</p>}
      </section>
    </main>
  );
}
