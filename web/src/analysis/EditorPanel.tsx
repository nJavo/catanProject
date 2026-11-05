// Left column of the analysis board: placement tools and every non-board
// field of the PositionSpec (turn state, viewer's cards, opponents' counts).
import type { PlayerSpec, PositionSpec } from "../api";
import { ResIcon } from "../board/Terrain";
import { CrestBadge } from "../Setup";
import { DEV_LABEL, PLAYER_COLORS, PLAYER_NAMES, RES_LABEL } from "../rules";
import { PHASES, PORT_KINDS, TERRAIN_LABEL, TERRAINS } from "./spec";

export type Tool = "settlement" | "city" | "road" | "erase" | "robber" | "terrain" | "number" | "port";
export const TOOLS: { id: Tool; label: string; key: string; hint: string }[] = [
  { id: "settlement", label: "Settlement", key: "s", hint: "Click a corner. Click your own settlement again to upgrade it." },
  { id: "city", label: "City", key: "c", hint: "Click a corner or a settlement." },
  { id: "road", label: "Road", key: "r", hint: "Click a side. Click your own road again to remove it." },
  { id: "erase", label: "Remove", key: "x", hint: "Click a building or road to remove it." },
  { id: "robber", label: "Robber", key: "b", hint: "Click the tile the robber sits on." },
  { id: "terrain", label: "Terrain", key: "t", hint: "Pick a terrain below, then click tiles." },
  { id: "number", label: "Number", key: "n", hint: "Pick a token below, then click tiles." },
  { id: "port", label: "Harbor", key: "p", hint: "Pick a harbor type below, then click one of the 9 harbor slots." },
];
export const NUMBER_BRUSH: (number | null)[] = [2, 3, 4, 5, 6, 8, 9, 10, 11, 12, null];
const PLAYER_PHASES = PHASES;
export const PHASE_NAME: Record<string, string> = { SETUP_SETTLEMENT: "Setup: settlement", SETUP_ROAD: "Setup: road", PRE_ROLL: "Before roll", MAIN: "Build & trade", MOVE_ROBBER: "Move robber", STEAL: "Steal", DISCARD: "Discard", ROAD_BUILDING: "Road Building",
  // only reachable from a game; shown read-only (DS3-11)
  TRADE_RESPONSE: "Responding to a trade", TRADE_CONFIRM: "Confirming a trade", GAME_OVER: "Game over" };

export interface Brush { terrain: string; number: number | null; port: string }

type Set = (f: (s: PositionSpec) => PositionSpec, key?: string) => void;

export function Stepper({ value, onChange, min = 0, max = 99, label, big, id }: {
  value: number; onChange: (v: number) => void; min?: number; max?: number; label: string; big?: boolean; id?: string;
}) {
  return (
    <span className={"stepper" + (big ? " big" : "")} data-testid={id}>
      <button onClick={() => onChange(Math.max(min, value - 1))} disabled={value <= min} aria-label={`Fewer ${label}`}>−</button>
      <output aria-label={label}>{value}</output>
      <button onClick={() => onChange(Math.min(max, value + 1))} disabled={value >= max} aria-label={`More ${label}`}>+</button>
    </span>
  );
}

function ResSteppers({ values, onChange, label, big, idPrefix, max = 19 }: {
  values: number[]; onChange: (r: number, v: number) => void; label: string; big?: boolean; idPrefix: string; max?: number;
}) {
  return (
    <div className={"an-res" + (big ? " big" : "")}>
      {values.map((n, r) => (
        <div key={r} className={`an-res-cell res-tint-${r}` + (n ? "" : " zero")}>
          <ResIcon r={r} size={big ? 26 : 18} />
          <Stepper value={n} onChange={(v) => onChange(r, v)} max={max} label={`${RES_LABEL[r]} ${label}`} big={big} id={`${idPrefix}-${r}`} />
        </div>
      ))}
    </div>
  );
}

const sum = (a?: number[]) => (a ?? []).reduce((x, y) => x + y, 0);

export default function EditorPanel({ spec, set, tool, setTool, seat, setSeat, brush, setBrush, readOnly }: {
  spec: PositionSpec; set: Set; tool: Tool | null; setTool: (t: Tool | null) => void; seat: number; setSeat: (s: number) => void;
  brush: Brush; setBrush: (b: Brush) => void; readOnly: boolean;
}) {
  const n = spec.num_players;
  const me = spec.players[spec.viewer];
  const upd = (p: number, f: (pl: PlayerSpec) => void, key?: string) => set((s) => { f(s.players[p]); return s; }, key);
  const field = <K extends keyof PositionSpec>(k: K, v: PositionSpec[K], key?: string) => set((s) => { s[k] = v; return s; }, key);
  const awardSel = (k: "longest_road_owner" | "largest_army_owner", label: string) => (
    <label className="field">
      <span className="field-label">{label}</span>
      <select value={spec[k] === undefined ? "auto" : String(spec[k])} aria-label={label}
        onChange={(e) => set((s) => { if (e.target.value === "auto") delete s[k]; else s[k] = Number(e.target.value); return s; })}>
        <option value="auto" title="Recomputed from the pieces on the board">Auto</option>
        <option value="-1">Nobody</option>
        {Array.from({ length: n }, (_, p) => <option key={p} value={p}>{PLAYER_NAMES[p]}</option>)}
      </select>
    </label>
  );

  return (
    <div className="an-editor" aria-label="Position editor">
      <fieldset className="an-card" disabled={readOnly}>
        <legend>Place pieces</legend>
        {readOnly && <p className="note small">Viewing a branch. Return to the position to edit.</p>}
        <div className="an-tools" role="toolbar" aria-label="Editor tools">
          {TOOLS.map((t) => (
            <button key={t.id} className={"chip" + (tool === t.id ? " on" : "")} onClick={() => setTool(tool === t.id ? null : t.id)}
              data-tool={t.id} title={t.hint} aria-pressed={tool === t.id}>
              {t.label}<kbd>{t.key.toUpperCase()}</kbd>
            </button>
          ))}
        </div>
        {(tool === "settlement" || tool === "city" || tool === "road" || !tool) && (
          <div className="an-seats" role="radiogroup" aria-label="Piece owner">
            {Array.from({ length: n }, (_, p) => (
              <button key={p} role="radio" aria-checked={seat === p} className={"an-seat" + (seat === p ? " on" : "")}
                style={{ ["--pc" as any]: PLAYER_COLORS[p] }} onClick={() => setSeat(p)} data-seat={p}>
                <CrestBadge p={p} size={18} />{PLAYER_NAMES[p]}<kbd>{p + 1}</kbd>
              </button>
            ))}
          </div>
        )}
        {tool === "terrain" && (
          <div className="an-brush">
            {TERRAINS.map((t) => (
              <button key={t} className={"chip" + (brush.terrain === t ? " on" : "")} onClick={() => setBrush({ ...brush, terrain: t })} data-brush={t}>
                <span className={`an-swatch t-${t}`} />{TERRAIN_LABEL[t]}
              </button>
            ))}
          </div>
        )}
        {tool === "number" && (
          <div className="an-brush tokens">
            {NUMBER_BRUSH.map((t) => (
              <button key={String(t)} className={"an-token" + (brush.number === t ? " on" : "") + (t === 6 || t === 8 ? " red" : "")}
                onClick={() => setBrush({ ...brush, number: t })} data-brush={String(t)} aria-label={t === null ? "No token" : `Token ${t}`}>
                {t ?? "∅"}
              </button>
            ))}
          </div>
        )}
        {tool === "port" && (
          <div className="an-brush">
            {PORT_KINDS.map((t) => (
              <button key={t} className={"chip" + (brush.port === t ? " on" : "")} onClick={() => setBrush({ ...brush, port: t })} data-brush={t}>
                {t === "3:1" ? "3:1 any" : `2:1 ${RES_LABEL[["wood", "brick", "sheep", "wheat", "ore"].indexOf(t)]}`}
              </button>
            ))}
          </div>
        )}
        <p className="muted small an-hint">{tool ? TOOLS.find((t) => t.id === tool)!.hint : "Choose a tool. Keys: S C R X B T N P, 1–4 owner, [ ] brush, Esc none."}</p>
      </fieldset>

      <fieldset className="an-card" disabled={readOnly}>
        <legend>Turn</legend>
        <div className="an-grid2">
          <label className="field"><span className="field-label">Players</span>
            <span className="segmented small">
              {[2, 3, 4].map((k) => <button key={k} className={n === k ? "on" : ""} onClick={() => set((s) => {
                s.num_players = k;
                s.viewer = Math.min(s.viewer, k - 1); s.current = Math.min(s.current, k - 1);
                s.vertex_owner = s.vertex_owner.map((o) => (o >= k ? -1 : o));
                s.vertex_level = s.vertex_level.map((l, v) => (s.vertex_owner[v] < 0 ? 0 : l));
                s.edge_owner = s.edge_owner.map((o) => (o >= k ? -1 : o));
                return s;
              })}>{k}</button>)}
            </span>
          </label>
          <label className="field"><span className="field-label" title="The player whose turn it is. In a trade reply or a discard someone else may act; the banner names who.">Whose turn</span>
            <select value={spec.current} onChange={(e) => field("current", Number(e.target.value))} aria-label="Current player">
              {Array.from({ length: n }, (_, p) => <option key={p} value={p}>{PLAYER_NAMES[p]}</option>)}
            </select>
          </label>
          <label className="field an-span2"><span className="field-label">Phase</span>
            <select value={spec.phase} onChange={(e) => set((s) => {
              s.phase = e.target.value;
              if (s.phase === "PRE_ROLL") s.rolled = false;
              if (["MAIN", "MOVE_ROBBER", "STEAL", "DISCARD", "ROAD_BUILDING"].includes(s.phase)) s.rolled = true;
              if (s.phase === "DISCARD" && !s.discard_remaining) s.discard_remaining = Array(4).fill(0);
              if (s.phase === "ROAD_BUILDING" && s.free_roads === undefined) s.free_roads = 2;
              return s;
            })} aria-label="Phase">
              {PLAYER_PHASES.map((ph) => <option key={ph} value={ph}>{PHASE_NAME[ph]}</option>)}
              {!PLAYER_PHASES.includes(spec.phase) && <option value={spec.phase} title={spec.phase}>{PHASE_NAME[spec.phase] ?? spec.phase} (from game)</option>}
            </select>
          </label>
          <label className="field"><span className="field-label">Turn number</span>
            <input type="number" min={1} value={spec.turn + 1} onChange={(e) => field("turn", Math.max(0, (Number(e.target.value) || 1) - 1), "turn")} aria-label="Turn number" />
          </label>
        </div>
        <div className="an-checks">
          <label><input type="checkbox" checked={spec.rolled} onChange={(e) => field("rolled", e.target.checked)} /> Dice rolled</label>
          <label><input type="checkbox" checked={spec.dev_played} onChange={(e) => field("dev_played", e.target.checked)} /> Dev card played this turn</label>
        </div>
        {spec.phase === "ROAD_BUILDING" && (
          <label className="an-inline"><span className="field-label">Free roads left</span>
            <Stepper value={spec.free_roads ?? 2} max={2} onChange={(v) => field("free_roads", v, "free")} label="free roads" /></label>
        )}
        {spec.phase === "DISCARD" && (
          <div className="an-discard">
            <span className="field-label">Cards each seat still has to discard</span>
            {Array.from({ length: n }, (_, p) => (
              <label key={p} className="an-inline"><CrestBadge p={p} size={16} /> {PLAYER_NAMES[p]}
                <Stepper value={spec.discard_remaining?.[p] ?? 0} max={20} label={`${PLAYER_NAMES[p]} discards`}
                  onChange={(v) => set((s) => { s.discard_remaining = [...(s.discard_remaining ?? [0, 0, 0, 0])]; s.discard_remaining[p] = v; return s; }, `disc${p}`)} />
              </label>
            ))}
          </div>
        )}
        <div className="an-grid2">{awardSel("longest_road_owner", "Longest Road")}{awardSel("largest_army_owner", "Largest Army")}</div>
      </fieldset>

      <fieldset className="an-card an-viewer" disabled={readOnly} style={{ ["--pc" as any]: PLAYER_COLORS[spec.viewer] }}>
        <legend>Your seat</legend>
        <label className="an-inline"><span className="field-label">Analyze as</span>
          <select value={spec.viewer} onChange={(e) => field("viewer", Number(e.target.value))} aria-label="Viewer seat" data-testid="viewer">
            {Array.from({ length: n }, (_, p) => <option key={p} value={p}>{PLAYER_NAMES[p]}</option>)}
          </select>
          <span className="muted small">{sum(me.hand)} cards</span>
        </label>
        <span className="field-label">Hand (exact)</span>
        <ResSteppers big values={me.hand ?? [0, 0, 0, 0, 0]} label="in hand" idPrefix="hand"
          onChange={(r, v) => upd(spec.viewer, (pl) => { pl.hand = [...(pl.hand ?? [0, 0, 0, 0, 0])]; pl.hand[r] = v; }, `hand${r}`)} />
        <span className="field-label">Unplayed development cards</span>
        <div className="an-devs">
          {DEV_LABEL.map((d, k) => (
            <label key={d} className="an-inline"><span>{d}</span>
              <Stepper value={me.dev_cards?.[k] ?? 0} max={14} label={d} id={`dev-${k}`}
                onChange={(v) => upd(spec.viewer, (pl) => {
                  pl.dev_cards = [...(pl.dev_cards ?? [0, 0, 0, 0, 0])]; pl.dev_cards[k] = v;
                  pl.dev_new = (pl.dev_new ?? [0, 0, 0, 0, 0]).map((x, i) => Math.min(x, pl.dev_cards![i]));
                }, `dev${k}`)} />
              {k !== 1 && (me.dev_cards?.[k] ?? 0) > 0 && (
                <span className="an-new" title="Bought this turn, can't be played yet">new
                  <Stepper value={me.dev_new?.[k] ?? 0} max={me.dev_cards?.[k] ?? 0} label={`new ${d}`}
                    onChange={(v) => upd(spec.viewer, (pl) => { pl.dev_new = [...(pl.dev_new ?? [0, 0, 0, 0, 0])]; pl.dev_new[k] = v; }, `new${k}`)} />
                </span>
              )}
            </label>
          ))}
          <label className="an-inline"><span>Knights played</span>
            <Stepper value={me.knights_played} max={14} label="your knights played" onChange={(v) => upd(spec.viewer, (pl) => { pl.knights_played = v; }, "myk")} />
          </label>
        </div>
      </fieldset>

      {Array.from({ length: n }, (_, p) => p).filter((p) => p !== spec.viewer).map((p) => {
        const pl = spec.players[p];
        return (
          <fieldset key={p} className="an-card an-opp" disabled={readOnly} style={{ ["--pc" as any]: PLAYER_COLORS[p] }} data-opp={p}>
            <legend><CrestBadge p={p} size={16} /> {PLAYER_NAMES[p]} <span className="muted small">public info</span></legend>
            <div className="an-opp-row">
              <label><span className="field-label">Cards</span><Stepper value={pl.resource_count ?? 0} max={95} label={`${PLAYER_NAMES[p]} cards`} onChange={(v) => upd(p, (x) => { x.resource_count = v; }, `rc${p}`)} id={`opp-${p}-cards`} /></label>
              <label><span className="field-label">Dev cards</span><Stepper value={pl.dev_count ?? 0} max={25} label={`${PLAYER_NAMES[p]} dev cards`} onChange={(v) => upd(p, (x) => { x.dev_count = v; }, `dc${p}`)} /></label>
              <label><span className="field-label">Knights</span><Stepper value={pl.knights_played} max={14} label={`${PLAYER_NAMES[p]} knights played`} onChange={(v) => upd(p, (x) => { x.knights_played = v; }, `kp${p}`)} /></label>
            </div>
            <details>
              <summary className="small">Known cards {sum(pl.known_hand_min) ? `(${sum(pl.known_hand_min)})` : "(optional)"}</summary>
              <ResSteppers values={pl.known_hand_min ?? [0, 0, 0, 0, 0]} label={`known in ${PLAYER_NAMES[p]}'s hand`} idPrefix={`known-${p}`} max={pl.resource_count ?? 0}
                onChange={(r, v) => upd(p, (x) => { x.known_hand_min = [...(x.known_hand_min ?? [0, 0, 0, 0, 0])]; x.known_hand_min[r] = v; }, `kh${p}${r}`)} />
            </details>
          </fieldset>
        );
      })}
    </div>
  );
}
