// Thin typed client for catan/server/app.py. The server is the authority on
// legality; the UI only offers what `legal_actions` contains (or composes
// trade offers that the server validates).

export type ActionValue = null | number | number[] | number[][];
export interface Action {
  player: number;
  type: string;
  value?: ActionValue;
}

export interface GameEvent {
  seq: number;
  i: number;
  type: string;
  [k: string]: any;
}

export interface PlayerInfo {
  id: number;
  resource_count: number;
  dev_count: number;
  knights_played: number;
  settlements: number[];
  cities: number[];
  roads: number[];
  ports: string[];
  public_vp: number;
  road_length: number;
  supply: { road: number; settlement: number; city: number };
  hand?: number[];
  dev_cards?: number[];
  dev_new?: number[];
  vp?: number;
}

export interface Seat {
  type: "human" | "agent";
  agent: string | null;
  kwargs: Record<string, unknown>;
  name?: string; // online rooms: set from the seat link
}

export interface Trade {
  proposer: number;
  give: number[];
  get: number[];
  pending: number[];
  responses: Record<string, null | { give: number[]; get: number[] }>;
}

export interface Layout {
  hex_terrain: string[];
  hex_number: (number | null)[];
  port_edges: number[];
  port_types: string[];
}

export interface Decision {
  seat: number;
  agent: string;
  action: Action;
  info: Record<string, any>;
}

export interface View {
  viewer: number | null;
  num_players: number;
  layout: Layout;
  vertex_owner: number[];
  vertex_level: number[];
  edge_owner: number[];
  robber: number;
  bank: number[];
  dev_deck_count: number;
  phase: string;
  current: number;
  actor: number | null;
  turn: number;
  rolled: boolean;
  dev_played: boolean;
  last_roll: number[] | null;
  discard_remaining: number[];
  free_roads: number;
  trade: Trade | null;
  trades_this_turn: number;
  longest_road_owner: number;
  largest_army_owner: number;
  winner: number;
  truncated: boolean;
  players: PlayerInfo[];
  vp_to_win: number;
  game_id: string;
  source: string;
  seats: Seat[];
  legal_actions: Action[];
  events: GameEvent[];
  event_seq: number;
  action_count: number;
  trade_rates: number[] | null;
  max_trade_offers_per_turn: number | null;
  awaiting: "human" | "agent" | "none";
  last_decision: Decision | null;
  // replay-state only
  last_events?: GameEvent[];
  prev_players?: { hand: number[]; dev_cards: number[] }[] | null;
  span_events?: GameEvent[];
  turns?: TurnSegment[];
  total_actions?: number;
  index?: number;
}

export interface Lobby {
  id: string; you: number | "host"; phase: string; turn: number; actor: number | null;
  seats: (Seat & { connected: boolean })[];
  tokens?: (string | null)[]; // host link only
}

/** Replay indexes [start, end) of one turn; setup has one segment per placement pair. */
export interface TurnSegment { start: number; end: number; player: number; setup: boolean; turn: number }

export interface Topology {
  hexes: { id: number; q: number; r: number; x: number; y: number; vertices: number[] }[];
  vertices: { id: number; x: number; y: number; hexes: number[]; neighbors: number[] }[];
  edges: { id: number; v: number[] }[];
}

export interface SaveInfo {
  name: string;
  saved: number;
  seats: Seat[];
  turn: number;
  phase: string;
  winner: number;
}

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

// Online mode (docs/ONLINE.md): the server wants X-Room-Token on a room's routes and
// X-Catan-Admin on the rest. Patching fetch covers the components that call it directly.
// The admin key arrives once as ?admin=... and stays in this origin's localStorage.
let roomToken: string | null = null;
export const setRoomToken = (t: string | null) => { roomToken = t; };
{
  const u = new URL(window.location.href);
  const k = u.searchParams.get("admin");
  if (k) {
    localStorage.setItem("catan-admin-key", k);
    u.searchParams.delete("admin");
    history.replaceState(null, "", u.pathname + u.search + u.hash); // keep the key out of history and screenshots
  }
  const raw = window.fetch.bind(window);
  window.fetch = (input, init) => {
    const url = typeof input === "string" ? input : input instanceof URL ? input.pathname : input.url;
    const admin = localStorage.getItem("catan-admin-key");
    if (!url.startsWith("/api/") || (!roomToken && !admin)) return raw(input, init);
    const headers = new Headers(init?.headers ?? (input instanceof Request ? input.headers : undefined));
    if (roomToken) headers.set("X-Room-Token", roomToken);
    if (admin) headers.set("X-Catan-Admin", admin);
    return raw(input, { ...init, headers });
  };
}

async function req<T>(method: string, url: string, body?: unknown, signal?: AbortSignal): Promise<T> {
  const r = await fetch(url, {
    method,
    signal,
    headers: body !== undefined ? { "Content-Type": "application/json" } : undefined,
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  const text = await r.text();
  let data: any = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    data = { detail: text };
  }
  if (!r.ok) {
    const d = data?.detail;
    throw new ApiError(r.status, typeof d === "string" ? d : JSON.stringify(d ?? r.statusText));
  }
  return data as T;
}

/** A long analysis call the server can stop: aborting ``signal`` (Cancel/Esc,
 * leaving the page) also tells the server, so its slot and workers are free
 * at once instead of after the 120 s time limit. */
function cancellable<T>(url: string, body: object, signal?: AbortSignal): Promise<T> {
  const request_id = `r${Date.now().toString(36)}${Math.random().toString(36).slice(2, 10)}`;
  const onAbort = () => { req("POST", "/api/analysis/cancel", { request_id }).catch(() => {}); };
  signal?.addEventListener("abort", onAbort, { once: true });
  return req<T>("POST", url, { ...body, request_id }, signal).finally(() => signal?.removeEventListener("abort", onAbort));
}

const q = (v: number | "all" | null | undefined) => (v === null || v === undefined ? "" : `viewer=${v}`);

export const checkpointFor = (agent: string, c?: string): string | undefined => (agent === "rl" ? c || undefined : undefined);

export const api = {
  topology: () => req<Topology>("GET", "/api/topology"),
  health: () => req<{ ok: boolean; online?: boolean }>("GET", "/api/health"),
  createRoom: (body: { players: { type: string; agent?: string; kwargs?: Record<string, unknown> }[]; seed?: number; vp_to_win?: number }) =>
    req<{ id: string; host_token: string; tokens: (string | null)[] }>("POST", "/api/rooms", body),
  lobby: (id: string) => req<Lobby>("GET", `/api/rooms/${id}`),
  setName: (id: string, name: string) => req<{ name: string }>("POST", `/api/rooms/${id}/name`, { name }),
  agents: () => req<{ agents: string[] }>("GET", "/api/agents"),
  checkpoints: () => req<{ checkpoints: string[]; default?: string | null }>("GET", "/api/checkpoints"),
  createGame: (body: { players: { type: string; agent?: string; kwargs?: Record<string, unknown> }[]; seed?: number; vp_to_win?: number }) =>
    req<{ id: string; seed: number; seats: Seat[]; first_human: number | null }>("POST", "/api/games", body),
  view: (id: string, viewer: number | "all" | null) => req<View>("GET", `/api/games/${id}?${q(viewer)}`),
  act: (id: string, action: Action) => req<{ events: GameEvent[]; view: View }>("POST", `/api/games/${id}/actions`, { action }),
  advance: (id: string, viewer: number | "all" | null, steps = 1) =>
    req<{ steps: { action: Action; events: GameEvent[] }[]; view: View }>("POST", `/api/games/${id}/advance?steps=${steps}&${q(viewer)}`),
  saveGame: (id: string, name?: string) => req<{ name: string }>("POST", `/api/games/${id}/save`, name ? { name } : {}),
  saves: () => req<{ saves: SaveInfo[] }>("GET", "/api/saves"),
  resume: (name: string) => req<{ id: string; seats: Seat[]; first_human: number | null }>("POST", `/api/saves/${encodeURIComponent(name)}/resume`),
  replay: (id: string) => req<any>("GET", `/api/games/${id}/replay`),
  saveReplay: (id: string) => req<{ name: string }>("POST", `/api/games/${id}/replay/save`),
  replays: () => req<{ replays: any[] }>("GET", "/api/replays"),
  replayFile: (name: string) => req<any>("GET", `/api/replays/${encodeURIComponent(name)}`),
  replayState: (source: "game" | "file", ref: string, index: number, since?: number) =>
    req<View>("GET", `/api/replay-state?source=${source}&ref=${encodeURIComponent(ref)}&index=${index}${since === undefined ? "" : `&since=${since}`}`),
  labMatch: (body: { agents: (string | { agent: string; kwargs?: Record<string, unknown> })[]; seed?: number; max_turns?: number; vp_to_win?: number; run_to_end?: boolean }) =>
    req<{ id: string; seed: number; result: any }>("POST", "/api/lab/matches", body),
  runs: () => req<{ runs: any[] }>("GET", "/api/runs"),  // light list: no metric rows
  // one run; since = previous response's cursor -> only rows appended after it
  run: (id: string, since = 0) => req<any>("GET", `/api/runs/${encodeURIComponent(id)}?since=${since}`),
  tournaments: () => req<{ tournaments: any[] }>("GET", "/api/tournaments"),
  // Analysis engine (catan/server/analysis.py).
  analysis: {
    validate: (position: PositionSpec, signal?: AbortSignal) =>
      req<ValidateResult>("POST", "/api/analysis/validate", { position }, signal),
    legal: (position: PositionSpec, signal?: AbortSignal) =>
      req<{ legal: (Action & { label?: string })[] }>("POST", "/api/analysis/legal", { position }, signal),
    evaluate: (body: EvaluateRequest, signal?: AbortSignal) =>
      cancellable<AnalysisResult>("/api/analysis/evaluate", body, signal),
    game: (body: { source: AnalysisSource; viewer: number; policy: string; depth: Depth }, signal?: AbortSignal) =>
      cancellable<GameAnalysis>("/api/analysis/game", body, signal),
    continue: (body: { position: PositionSpec; action: Action; policy: string; steps?: number; until?: "turn_end" | "game_end"; seed?: number }, signal?: AbortSignal) =>
      cancellable<{ position: PositionSpec; events: GameEvent[]; done: boolean }>("/api/analysis/continue", body, signal),
  },
};

// ---------------------------------------------------------------- analysis types
export type Depth = "quick" | "standard" | "deep";
export type Verdict = "best" | "good" | "inaccuracy" | "mistake" | "blunder" | "unrated";

export interface PlayerSpec {
  hand?: number[];          // viewer only
  dev_cards?: number[];     // viewer only
  dev_new?: number[];       // viewer only
  knights_played: number;
  resource_count?: number;  // opponents
  dev_count?: number;       // opponents
  known_hand_min?: number[]; // opponents, optional lower bound
}

export interface PositionSpec {
  version: 1;
  num_players: number;
  viewer: number;
  layout: Layout;
  vertex_owner: number[];
  vertex_level: number[];
  edge_owner: number[];
  robber: number;
  current: number;
  phase: string;
  rolled: boolean;
  dev_played: boolean;
  turn: number;
  players: PlayerSpec[];
  longest_road_owner?: number;
  largest_army_owner?: number;
  bank?: number[];
  dev_deck_count?: number;
  discard_remaining?: number[];
  free_roads?: number;
}

export interface Rollout { win_rate: number; ci95: [number, number]; n: number; mean_vp?: number | null }
export interface Candidate {
  action: Action; label: string;
  policy_prob: number | null; value_estimate: number | null; search_score: number | null;
  rollout: Rollout | null; rank: number; verdict: Verdict;
  agent_choice?: boolean; search_value?: number | null; search_visits?: number | null;
}
export interface AnalysisSource { kind: "game" | "replay"; ref: string; index?: number }
export interface EvaluateRequest {
  position?: PositionSpec; source?: AnalysisSource; viewer?: number;
  policy: string; rollout_policy?: string; depth: Depth; top_k?: number; seed?: number;
  belief?: boolean;
}
export interface BeliefSummary {
  label: string; method: string; states: number; approx: boolean; unseen_steals: number;
  opponents: Record<string, { known: boolean; expected: number[]; dist: Record<string, number>[] }>;
}
export interface AnalysisResult {
  position: PositionSpec; actor: number; phase: string; legal_count: number;
  policy: string; rollout_policy: string | null; depth: Depth;
  position_eval: { value_estimate: number | null; rollout: Rollout | null };
  candidates: Candidate[];
  determinization: { samples: number; assumptions: string[]; hand_sampling?: "belief" | "uniform" };
  belief?: BeliefSummary | null;
  attribution?: {
    label: string; method: string; baseline: string; steps: number; action: Action; caveats: string[];
    logit: { at_obs: number; at_baseline: number; sum: number; completeness_gap: number };
    value: { at_obs: number; at_baseline: number; sum: number; completeness_gap: number };
    blocks: { block: string; logit: number; value: number }[];
    entities: Record<string, { id: number; logit: number; value: number }[]>;
  } | null;
  timing_ms: Record<string, number>;
  mock?: boolean;
  // v1 extras
  timed_out?: boolean; candidate_source?: string; search_score_kind?: string;
  policy_kind?: "learned_policy" | "search" | "agent" | "heuristic"; search_value_kind?: string | null;
  agent_choice?: Action | null; agent_info?: { source?: string; sims?: number; replaced?: Action; trade_layer?: string } | null; ranking?: string; verdict_thresholds?: unknown;
}
export interface ValidateResult { ok: boolean; errors: (string | { message: string; vertices?: number[]; edges?: number[]; hexes?: number[] })[]; derived?: Record<string, unknown>; mock?: boolean }
export interface TimelineEntry {
  index: number; actor: number; played: Action; played_label: string;
  best: Action | null; best_label: string | null;
  // v1: objects {win_rate, ci95, n, value_estimate, search_score}; `metric` says which field delta uses
  eval_before: number | EvalPoint | null; eval_after: number | EvalPoint | null; delta: number | null; verdict: Verdict;
  metric?: "rollout_win_rate" | "value_estimate" | "search_score"; turn?: number; phase?: string;
}
export interface EvalPoint { win_rate: number | null; ci95: [number, number] | null; n: number; value_estimate: number | null; search_score: number | null }
export interface GameAnalysis { timeline: TimelineEntry[]; summary: { accuracy?: number; mistakes: number; blunders: number }; mock?: boolean; depth?: Depth; policy?: string }
