// PositionSpec helpers for the analysis board. The wire format is the
// contract in catan/analysis/position.py; the editor keeps every field for every
// seat and strips to viewer/opponent shape with toWire().
import type { Layout, PlayerSpec, PositionSpec, Topology, View } from "../api";
import { checkpointFor } from "../api";

export const TERRAINS = ["wood", "brick", "sheep", "wheat", "ore", "desert"] as const;
export const TERRAIN_LABEL: Record<string, string> = { wood: "Forest", brick: "Hills", sheep: "Pasture", wheat: "Fields", ore: "Mountains", desert: "Desert" };
export const TERRAIN_COUNTS: Record<string, number> = { wood: 4, brick: 3, sheep: 4, wheat: 4, ore: 3, desert: 1 };
export const TOKENS = [2, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8, 9, 9, 10, 10, 11, 11, 12];
export const PORT_KINDS = ["3:1", "wood", "brick", "sheep", "wheat", "ore"];
const PORT_SET = ["3:1", "3:1", "3:1", "3:1", "wood", "brick", "sheep", "wheat", "ore"];
const PORT_EDGE_OFFSETS = [0, 3, 6, 10, 13, 16, 20, 23, 26]; // mirrors catan/engine/board.py
export const DEV_TOTAL = [14, 5, 2, 2, 2];
export const SUPPLY = { settlement: 5, city: 4, road: 15 };

export const PHASES = ["SETUP_SETTLEMENT", "SETUP_ROAD", "PRE_ROLL", "MAIN", "MOVE_ROBBER", "STEAL", "DISCARD", "ROAD_BUILDING"];

const zeros = (n: number) => Array(n).fill(0);

/** The 9 harbor edges, computed exactly like generate_layout() does. */
export function portEdges(topo: Topology): number[] {
  const coast = topo.edges.filter((e) => topo.hexes.filter((h) => h.vertices.includes(e.v[0]) && h.vertices.includes(e.v[1])).length === 1)
    .map((e) => {
      const [a, b] = e.v.map((v) => topo.vertices[v]);
      return { id: e.id, ang: Math.atan2((a.y + b.y) / 2, (a.x + b.x) / 2) };
    })
    .sort((x, y) => x.ang - y.ang);
  return PORT_EDGE_OFFSETS.map((o) => coast[o].id);
}

export function hexNeighbors(topo: Topology, h: number): number[] {
  const vs = topo.hexes[h].vertices;
  return topo.hexes.filter((o) => o.id !== h && o.vertices.filter((v) => vs.includes(v)).length === 2).map((o) => o.id);
}

function shuffle<T>(a: T[]): T[] {
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

export function blankPlayer(): PlayerSpec {
  return { hand: zeros(5), dev_cards: zeros(5), dev_new: zeros(5), knights_played: 0, resource_count: 0, dev_count: 0, known_hand_min: zeros(5) };
}

function basePosition(topo: Topology, layout: Layout, n = 4): PositionSpec {
  return {
    version: 1, num_players: n, viewer: 0, layout,
    vertex_owner: Array(topo.vertices.length).fill(-1), vertex_level: zeros(topo.vertices.length),
    edge_owner: Array(topo.edges.length).fill(-1),
    robber: Math.max(0, layout.hex_terrain.indexOf("desert")),
    current: 0, phase: "SETUP_SETTLEMENT", rolled: false, dev_played: false, turn: 0,
    players: Array.from({ length: n }, blankPlayer),
  };
}

/** Random legal-looking island: official tile mix, no adjacent 6/8. */
export function randomPosition(topo: Topology): PositionSpec {
  const terrain = shuffle(Object.entries(TERRAIN_COUNTS).flatMap(([t, c]) => Array(c).fill(t)));
  const nb = topo.hexes.map((h) => hexNeighbors(topo, h.id));
  let numbers: (number | null)[] = [];
  for (let tries = 0; tries < 500; tries++) {
    const toks = shuffle([...TOKENS]);
    numbers = terrain.map((t) => (t === "desert" ? null : toks.pop()!));
    const red = (x: number | null) => x === 6 || x === 8;
    if (!numbers.some((x, h) => red(x) && nb[h].some((o) => red(numbers[o])))) break;
  }
  return basePosition(topo, { hex_terrain: terrain, hex_number: numbers, port_edges: portEdges(topo), port_types: shuffle([...PORT_SET]) });
}

/** All-desert island with no tokens: paint it to match a physical board. */
export function emptyPosition(topo: Topology): PositionSpec {
  return basePosition(topo, { hex_terrain: Array(topo.hexes.length).fill("desert"), hex_number: Array(topo.hexes.length).fill(null), port_edges: portEdges(topo), port_types: [...PORT_SET] });
}

/** Spec from a (fully revealed) replay/game view, keeping only what `viewer` could know. */
export function specFromView(v: View, viewer: number): PositionSpec {
  return {
    version: 1, num_players: v.num_players, viewer, layout: v.layout,
    vertex_owner: [...v.vertex_owner], vertex_level: [...v.vertex_level], edge_owner: [...v.edge_owner],
    robber: v.robber, current: v.current, phase: v.phase, rolled: v.rolled, dev_played: v.dev_played, turn: v.turn,
    players: v.players.map((p) => ({
      hand: p.id === viewer ? [...(p.hand ?? zeros(5))] : zeros(5),
      dev_cards: p.id === viewer ? [...(p.dev_cards ?? zeros(5))] : zeros(5),
      dev_new: p.id === viewer ? [...(p.dev_new ?? zeros(5))] : zeros(5),
      knights_played: p.knights_played, resource_count: p.resource_count, dev_count: p.dev_count, known_hand_min: zeros(5),
    })),
    longest_road_owner: v.longest_road_owner, largest_army_owner: v.largest_army_owner,
    bank: [...v.bank], dev_deck_count: v.dev_deck_count,
    discard_remaining: [...v.discard_remaining], free_roads: v.free_roads,
    // v1 optional fields: trade sub-phases need the pending trade and the actor
    ...({ trade: v.trade, trades_this_turn: v.trades_this_turn, ...(v.actor !== null ? { actor: v.actor } : {}) } as object),
  };
}

/** Contract shape: viewer gets exact cards, opponents only public counts. */
export function toWire(s: PositionSpec): PositionSpec {
  const out: PositionSpec = { ...s, players: s.players.slice(0, s.num_players).map((p, i) => {
    if (i === s.viewer) return { hand: p.hand ?? zeros(5), dev_cards: p.dev_cards ?? zeros(5), dev_new: p.dev_new ?? zeros(5), knights_played: p.knights_played };
    const o: PlayerSpec = { resource_count: p.resource_count ?? 0, dev_count: p.dev_count ?? 0, knights_played: p.knights_played };
    if (p.known_hand_min?.some((x) => x > 0)) o.known_hand_min = p.known_hand_min;
    return o;
  }) };
  if (s.phase !== "DISCARD") delete out.discard_remaining;
  else out.discard_remaining = (s.discard_remaining ?? []).slice(0, s.num_players);
  if (s.phase !== "ROAD_BUILDING") delete out.free_roads;
  for (const k of ["longest_road_owner", "largest_army_owner", "bank", "dev_deck_count"] as const) if (out[k] === undefined) delete out[k];
  return out;
}

/** Fill editor defaults into an imported spec; throws with a readable message. */
export function normalize(raw: any, topo: Topology): PositionSpec {
  if (!raw || typeof raw !== "object") throw new Error("Not a JSON object");
  const p = raw.position && raw.layout === undefined ? raw.position : raw; // accept evaluate responses too
  const need = (k: string, len?: number) => {
    if (!Array.isArray(p[k]) || (len !== undefined && p[k].length !== len)) throw new Error(`${k} must be an array${len ? ` of ${len}` : ""}`);
  };
  if (!p.layout) throw new Error("layout is missing");
  for (const k of ["hex_terrain", "hex_number"]) if (!Array.isArray(p.layout[k]) || p.layout[k].length !== topo.hexes.length) throw new Error(`layout.${k} must have ${topo.hexes.length} entries`);
  need("vertex_owner", topo.vertices.length); need("vertex_level", topo.vertices.length); need("edge_owner", topo.edges.length); need("players");
  const n = Number(p.num_players ?? p.players.length);
  if (!(n >= 2 && n <= 4)) throw new Error("num_players must be 2 to 4");
  if (p.version !== undefined && p.version !== 1) throw new Error(`unsupported version ${JSON.stringify(p.version)}`);
  if (p.players.length < n) throw new Error(`players must list ${n} seats`);
  // Bad seat data used to crash the page, and the autosave then crashed every reload.
  const count = (x: unknown, max: number) => Number.isInteger(x) && (x as number) >= 0 && (x as number) <= max;
  p.players.slice(0, n).forEach((q: any, i: number) => {
    if (!q || typeof q !== "object") throw new Error(`players[${i}] must be an object`);
    for (const k of ["hand", "dev_cards", "dev_new", "known_hand_min"])
      if (q[k] !== undefined && !(Array.isArray(q[k]) && q[k].length === 5 && q[k].every((x: unknown) => count(x, 25))))
        throw new Error(`players[${i}].${k} must be 5 whole numbers from 0 to 25`);
    for (const k of ["knights_played", "resource_count", "dev_count"])
      if (q[k] !== undefined && !count(q[k], 200)) throw new Error(`players[${i}].${k} must be a whole number ≥ 0`);
  });
  const players = Array.from({ length: 4 }, (_, i) => ({ ...blankPlayer(), ...(p.players[i] ?? {}) }));
  return {
    ...p, version: 1, num_players: n, viewer: Number(p.viewer ?? 0),
    layout: { ...p.layout, port_edges: p.layout.port_edges ?? portEdges(topo), port_types: p.layout.port_types ?? [...PORT_SET] },
    current: Number(p.current ?? 0), phase: String(p.phase ?? "MAIN"), rolled: !!p.rolled, dev_played: !!p.dev_played,
    turn: Number(p.turn ?? 0), robber: Number(p.robber ?? 0), players,
  };
}

/** Seat that must act: the discarding seat in DISCARD, otherwise `current`. */
export function actorOf(s: PositionSpec): number {
  const x = s as any;
  if (typeof x.actor === "number") return x.actor;
  if (x.trade?.pending?.length && (s.phase === "TRADE_RESPONSE")) return x.trade.pending[0];
  if (s.phase === "SETUP_SETTLEMENT" || s.phase === "SETUP_ROAD") { // snake order; the server ignores `current` here
    const n = s.num_players, k = s.vertex_owner.filter((o) => o >= 0).length - (s.phase === "SETUP_ROAD" ? 1 : 0);
    const i = Math.max(0, Math.min(2 * n - 1, k));
    return i < n ? i : 2 * n - 1 - i;
  }
  if (s.phase === "DISCARD") {
    const i = (s.discard_remaining ?? []).findIndex((x) => x > 0);
    if (i >= 0) return i;
  }
  return s.current;
}

export function policySpec(agent: string, checkpoint?: string): string {
  const c = checkpointFor(agent, checkpoint);
  return c ? `${agent}:checkpoint=${c}` : agent;
}

/** Board ids named in a validation error. Structured errors win; strings are scanned for "vertex 12", "edge 3", "hex 9". */
export function errorTargets(e: string | { message: string; vertices?: number[]; edges?: number[]; hexes?: number[] }) {
  if (typeof e !== "string") return { msg: e.message, vertices: e.vertices ?? [], edges: e.edges ?? [], hexes: e.hexes ?? [] };
  const grab = (re: RegExp) => [...e.matchAll(re)].map((m) => Number(m[1]));
  return { msg: e, vertices: grab(/\bvertex (\d+)/gi), edges: grab(/\b(?:edge|road) (?:on |at )?(\d+)/gi), hexes: grab(/\bhex (\d+)/gi) };
}

/** Eval as one number in 0..1 for charts; accepts a bare number or {win_rate}/{value_estimate}. */
export function evalNum(x: unknown, metric?: string): number | null {
  if (typeof x === "number") return x;
  if (x && typeof x === "object") {
    const o = x as any;
    if (metric) { const v = o[metric === "rollout_win_rate" ? "win_rate" : metric]; return typeof v === "number" ? v : null; }
    if (typeof o.win_rate === "number") return o.win_rate;
    if (typeof o.rollout?.win_rate === "number") return o.rollout.win_rate;
    if (typeof o.value_estimate === "number") return o.value_estimate;
  }
  return null;
}

/** Board target of an action, for highlights. */
export function actionTarget(a: { type: string; value?: unknown }): { kind: "v" | "e" | "h"; id: number } | null {
  if (typeof a.value !== "number") return null;
  if (a.type === "BUILD_SETTLEMENT" || a.type === "BUILD_CITY") return { kind: "v", id: a.value };
  if (a.type === "BUILD_ROAD") return { kind: "e", id: a.value };
  if (a.type === "MOVE_ROBBER") return { kind: "h", id: a.value };
  return null;
}

/** Pieces a seat has on the board. */
export function pieceCounts(s: PositionSpec, p: number) {
  let settlement = 0, city = 0, road = 0;
  s.vertex_owner.forEach((o, v) => { if (o === p) (s.vertex_level[v] === 2 ? city++ : settlement++); });
  s.edge_owner.forEach((o) => { if (o === p) road++; });
  return { settlement, city, road };
}
