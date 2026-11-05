// DEV-ONLY MOCK of /api/analysis. Used only while the
// real backend route is missing (404/405). Every response carries mock: true
// and the UI shows a MOCK banner: these numbers are NOT an evaluation.
// Validation here is a real but partial rule check, so the editor stays useful.
import type { Action, AnalysisResult, Candidate, Depth, GameAnalysis, GameEvent, PositionSpec, Topology, ValidateResult, Verdict } from "../api";
import { api } from "../api";
import { describeAction, RES_LABEL, seatShort } from "../rules";
import { actorOf, DEV_TOTAL, pieceCounts, SUPPLY, TERRAIN_COUNTS } from "./spec";

let TOPO: Topology | null = null;
export function setMockTopology(t: Topology) { TOPO = t; }
const topo = () => { if (!TOPO) throw new Error("mock topology not set"); return TOPO; };

const PIPS: Record<number, number> = { 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1 };
const COST = { road: [1, 1, 0, 0, 0], settlement: [1, 1, 1, 1, 0], city: [0, 0, 0, 2, 3], dev: [0, 0, 1, 1, 1] };

function rng(seedStr: string) {
  let h = 2166136261;
  for (let i = 0; i < seedStr.length; i++) h = Math.imul(h ^ seedStr.charCodeAt(i), 16777619);
  return () => {
    h = (h + 0x6d2b79f5) | 0;
    let t = Math.imul(h ^ (h >>> 15), 1 | h);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function sleep(ms: number, signal?: AbortSignal) {
  return new Promise<void>((res, rej) => {
    const t = setTimeout(res, ms);
    signal?.addEventListener("abort", () => { clearTimeout(t); rej(new DOMException("Aborted", "AbortError")); });
  });
}

const edgesAt = (v: number) => topo().edges.filter((e) => e.v.includes(v)).map((e) => e.id);
const settlementSpotOk = (s: PositionSpec, v: number) => s.vertex_owner[v] < 0 && topo().vertices[v].neighbors.every((u) => s.vertex_owner[u] < 0);

function roadConnected(s: PositionSpec, e: number, p: number) {
  return topo().edges[e].v.some((v) => s.vertex_owner[v] === p || (s.vertex_owner[v] < 0 && edgesAt(v).some((x) => x !== e && s.edge_owner[x] === p)));
}

// ------------------------------------------------------------------ validate
export function mockValidate(s: PositionSpec): ValidateResult {
  const t = topo(), errors: string[] = [], n = s.num_players;
  const L = s.layout;
  const counts: Record<string, number> = {};
  L.hex_terrain.forEach((x) => (counts[x] = (counts[x] ?? 0) + 1));
  for (const [k, c] of Object.entries(TERRAIN_COUNTS)) if ((counts[k] ?? 0) !== c) errors.push(`Terrain: ${counts[k] ?? 0} ${k} tiles, the base game has ${c}`);
  L.hex_terrain.forEach((x, h) => {
    if (x === "desert" && L.hex_number[h] !== null) errors.push(`hex ${h}: desert can't have a number token`);
    if (x !== "desert" && L.hex_number[h] === null) errors.push(`hex ${h}: ${x} tile needs a number token`);
  });
  if (!(s.robber >= 0 && s.robber < t.hexes.length)) errors.push("Robber is off the board");
  if (s.viewer >= n) errors.push(`Viewer seat ${s.viewer} is not in a ${n}-player game`);
  if (s.current >= n) errors.push(`Current seat ${s.current} is not in a ${n}-player game`);
  s.vertex_owner.forEach((o, v) => {
    if (o < 0) return;
    if (o >= n) errors.push(`vertex ${v}: owned by seat ${o}, which is not playing`);
    for (const u of t.vertices[v].neighbors) if (u > v && s.vertex_owner[u] >= 0) errors.push(`Distance rule: vertex ${v} and vertex ${u} are adjacent`);
  });
  s.edge_owner.forEach((o, e) => {
    if (o < 0) return;
    if (o >= n) errors.push(`edge ${e}: owned by seat ${o}, which is not playing`);
    else if (!roadConnected(s, e, o)) errors.push(`edge ${e}: ${seatShort(o)}'s road is not connected to their network`);
  });
  const handTotal = [0, 0, 0, 0, 0];
  for (let p = 0; p < n; p++) {
    const c = pieceCounts(s, p), pl = s.players[p];
    if (c.settlement > SUPPLY.settlement) errors.push(`${seatShort(p)} has ${c.settlement} settlements (max ${SUPPLY.settlement})`);
    if (c.city > SUPPLY.city) errors.push(`${seatShort(p)} has ${c.city} cities (max ${SUPPLY.city})`);
    if (c.road > SUPPLY.road) errors.push(`${seatShort(p)} has ${c.road} roads (max ${SUPPLY.road})`);
    if (p === s.viewer) (pl.hand ?? []).forEach((x, r) => (handTotal[r] += x));
    else (pl.known_hand_min ?? []).forEach((x, r) => (handTotal[r] += x));
    if (p !== s.viewer && (pl.known_hand_min ?? []).reduce((a, b) => a + b, 0) > (pl.resource_count ?? 0))
      errors.push(`${seatShort(p)}: known cards exceed their card count`);
  }
  handTotal.forEach((x, r) => x > 19 && errors.push(`${x} ${RES_LABEL[r]} in hands, only 19 exist`));
  const me = s.players[s.viewer];
  DEV_TOTAL.forEach((cap, d) => {
    const played = d === 0 ? s.players.slice(0, n).reduce((a, p) => a + p.knights_played, 0) : 0;
    if ((me?.dev_cards?.[d] ?? 0) + played > cap) errors.push(`Too many ${["knights", "VP cards", "Road Building", "Year of Plenty", "Monopoly"][d]}: only ${cap} exist`);
  });
  const settlements = s.vertex_owner.filter((o) => o >= 0).length;
  if (s.phase.startsWith("SETUP") && settlements > 2 * n) errors.push(`Setup phase but ${settlements} settlements are on the board`);
  if (s.phase === "DISCARD" && !(s.discard_remaining ?? []).some((x) => x > 0)) errors.push("Discard phase needs at least one seat with cards to discard");
  if (s.phase === "PRE_ROLL" && s.rolled) errors.push("Pre-roll phase but the dice are marked as rolled");
  const vp = Array.from({ length: n }, (_, p) => { const c = pieceCounts(s, p); return c.settlement + 2 * c.city; });
  return { ok: errors.length === 0, errors, derived: { public_vp_from_buildings: vp, setup_index: s.phase.startsWith("SETUP") ? settlements : null }, mock: true };
}

// ------------------------------------------------------------------ legal
const can = (hand: number[], cost: number[]) => cost.every((c, r) => (hand[r] ?? 0) >= c);

export function mockLegal(s: PositionSpec): Action[] {
  const t = topo(), p = actorOf(s), hand = s.players[p]?.hand ?? [0, 0, 0, 0, 0], out: Action[] = [];
  const A = (type: string, value?: any): Action => ({ player: p, type, value });
  const roadSpots = () => s.edge_owner.map((o, e) => (o < 0 && roadConnected(s, e, p) ? e : -1)).filter((e) => e >= 0);
  switch (s.phase) {
    case "SETUP_SETTLEMENT":
      t.vertices.forEach((v) => settlementSpotOk(s, v.id) && out.push(A("BUILD_SETTLEMENT", v.id)));
      break;
    case "SETUP_ROAD": case "ROAD_BUILDING":
      roadSpots().forEach((e) => out.push(A("BUILD_ROAD", e)));
      break;
    case "PRE_ROLL":
      out.push(A("ROLL"));
      if ((s.players[p]?.dev_cards?.[0] ?? 0) > 0 && !s.dev_played) out.push(A("PLAY_KNIGHT"));
      break;
    case "MOVE_ROBBER":
      t.hexes.forEach((h) => h.id !== s.robber && out.push(A("MOVE_ROBBER", h.id)));
      break;
    case "STEAL":
      [...new Set(t.hexes[s.robber].vertices.map((v) => s.vertex_owner[v]))].filter((o) => o >= 0 && o !== p).forEach((o) => out.push(A("STEAL", o)));
      break;
    case "DISCARD":
      hand.forEach((x, r) => x > 0 && out.push(A("DISCARD", r)));
      break;
    default: // MAIN
      if (can(hand, COST.settlement)) t.vertices.forEach((v) => settlementSpotOk(s, v.id) && edgesAt(v.id).some((e) => s.edge_owner[e] === p) && out.push(A("BUILD_SETTLEMENT", v.id)));
      if (can(hand, COST.city)) s.vertex_owner.forEach((o, v) => o === p && s.vertex_level[v] === 1 && out.push(A("BUILD_CITY", v)));
      if (can(hand, COST.road)) roadSpots().forEach((e) => out.push(A("BUILD_ROAD", e)));
      if (can(hand, COST.dev)) out.push(A("BUY_DEV"));
      hand.forEach((x, r) => x >= 4 && [0, 1, 2, 3, 4].forEach((g) => g !== r && out.push(A("MARITIME_TRADE", [r, g]))));
      out.push(A("END_TURN"));
  }
  return out;
}

// ------------------------------------------------------------------ evaluate
function heuristic(s: PositionSpec, a: Action) {
  const v = a.value as number;
  const pips = (vid: number) => topo().vertices[vid].hexes.reduce((x, h) => x + (PIPS[s.layout.hex_number[h] ?? 0] ?? 0), 0);
  switch (a.type) {
    case "BUILD_SETTLEMENT": case "BUILD_CITY": return pips(v) + (a.type === "BUILD_CITY" ? 4 : 2);
    case "BUILD_ROAD": return Math.max(...topo().edges[v].v.map(pips)) * 0.4;
    case "MOVE_ROBBER": return topo().hexes[v].vertices.filter((x) => s.vertex_owner[x] >= 0 && s.vertex_owner[x] !== a.player).length * 2 + (PIPS[s.layout.hex_number[v] ?? 0] ?? 0);
    case "BUY_DEV": return 4;
    case "ROLL": return 5;
    case "END_TURN": return 1;
    default: return 2;
  }
}

const wilson = (w: number, n: number): [number, number] => {
  const z = 1.96, p = w / n, d = 1 + (z * z) / n, c = p + (z * z) / (2 * n), m = z * Math.sqrt((p * (1 - p)) / n + (z * z) / (4 * n * n));
  return [Math.max(0, (c - m) / d), Math.min(1, (c + m) / d)];
};

export async function mockEvaluate(s: PositionSpec, policy: string, depth: Depth, topK = 5, signal?: AbortSignal, rolloutPolicy?: string): Promise<AnalysisResult> {
  const t0 = performance.now();
  const legal = mockLegal(s);
  const rand = rng(JSON.stringify([s.vertex_owner, s.edge_owner, s.phase, s.players, policy, depth]));
  const scored = legal.map((a) => ({ a, h: heuristic(s, a) + rand() * 1.5 })).sort((x, y) => y.h - x.h);
  const z = scored.reduce((acc, x) => acc + Math.exp(x.h / 2), 0);
  const k = depth === "deep" ? Math.max(topK, 8) : topK;
  const nRoll = depth === "quick" ? 0 : depth === "standard" ? 32 : 200;
  await sleep(depth === "quick" ? 120 : depth === "standard" ? 1400 : 4500, signal);
  const isRl = policy.startsWith("rl");
  const base = 0.15 + rand() * 0.4;
  const cands: Candidate[] = scored.slice(0, k).map((x, i) => {
    const wr = Math.max(0.02, Math.min(0.98, base + (k - i) * 0.03 - rand() * 0.06));
    const w = Math.round(wr * nRoll);
    return {
      action: x.a, label: describeAction(x.a),
      policy_prob: isRl ? Math.exp(x.h / 2) / z : null,
      value_estimate: isRl ? Math.max(0, Math.min(1, wr + (rand() - 0.5) * 0.08)) : null,
      search_score: isRl ? null : Number(x.h.toFixed(2)),
      rollout: nRoll ? { win_rate: w / nRoll, ci95: wilson(Math.max(w, 0.5), nRoll), n: nRoll, mean_vp: 5 + wr * 4 } : null,
      rank: i + 1, verdict: "unrated" as Verdict,
    };
  });
  if (nRoll) {
    const best = cands[0]?.rollout?.win_rate ?? 0;
    cands.forEach((c, i) => {
      const gap = best - (c.rollout?.win_rate ?? 0);
      c.verdict = i === 0 ? "best" : gap < 0.03 ? "good" : gap < 0.08 ? "inaccuracy" : gap < 0.15 ? "mistake" : "blunder";
    });
  }
  const ms = Math.round(performance.now() - t0);
  return {
    position: s, actor: actorOf(s), phase: s.phase, legal_count: legal.length, policy, rollout_policy: nRoll ? rolloutPolicy || policy : null, depth,
    position_eval: { value_estimate: isRl ? base : null, rollout: nRoll ? { win_rate: base, ci95: wilson(Math.round(base * nRoll * 4), nRoll * 4), n: nRoll * 4, mean_vp: 6 } : null },
    candidates: cands,
    determinization: { samples: nRoll ? 4 : 0, assumptions: ["MOCK: no determinization was run", "MOCK: numbers are generated from a pip-count heuristic plus noise"] },
    timing_ms: { policy: 3, rollouts: ms - 3, total: ms }, mock: true,
  };
}

// ------------------------------------------------------------------ game timeline
export async function mockGame(kind: "game" | "replay", ref: string, viewer: number, signal?: AbortSignal): Promise<GameAnalysis> {
  const r = kind === "game" ? await api.replay(ref) : await api.replayFile(ref);
  await sleep(600, signal);
  const actions: Action[] = r.actions ?? [];
  const rand = rng(ref + viewer);
  let ev = 0.25;
  const timeline = actions.map((a, index) => ({ a, index })).filter((x) => x.a.player === viewer).map(({ a, index }) => {
    const roll = rand();
    const verdict: Verdict = roll < 0.04 ? "blunder" : roll < 0.1 ? "mistake" : roll < 0.18 ? "inaccuracy" : roll < 0.5 ? "best" : "good";
    const delta = verdict === "blunder" ? -0.15 : verdict === "mistake" ? -0.08 : verdict === "inaccuracy" ? -0.03 : (rand() - 0.4) * 0.04;
    const before = ev;
    ev = Math.max(0.01, Math.min(0.99, ev + delta + (rand() - 0.5) * 0.03));
    const worse = verdict === "blunder" || verdict === "mistake" || verdict === "inaccuracy";
    return {
      index, actor: a.player, played: a, played_label: describeAction(a),
      best: worse ? null : a, best_label: worse ? "(mock) a different move" : describeAction(a),
      eval_before: before, eval_after: ev, delta: ev - before, verdict,
    };
  });
  return { timeline, summary: { mistakes: timeline.filter((x) => x.verdict === "mistake").length, blunders: timeline.filter((x) => x.verdict === "blunder").length }, mock: true };
}

// ------------------------------------------------------------------ continue
export async function mockContinue(s: PositionSpec, a: Action, signal?: AbortSignal): Promise<{ position: PositionSpec; events: GameEvent[]; done: boolean }> {
  await sleep(300, signal);
  const n = structuredClone(s), p = a.player, ev: GameEvent[] = [];
  const pay = (cost: number[]) => { const h = n.players[p].hand; if (h) cost.forEach((c, r) => (h[r] = Math.max(0, h[r] - c))); };
  const e = (type: string, extra: object = {}) => ev.push({ seq: ev.length, i: 0, type, player: p, ...extra });
  const v = a.value as number;
  switch (a.type) {
    case "BUILD_SETTLEMENT": n.vertex_owner[v] = p; n.vertex_level[v] = 1; if (s.phase === "MAIN") pay(COST.settlement); else n.phase = "SETUP_ROAD"; e("settlement_built", { free: s.phase !== "MAIN" }); break;
    case "BUILD_CITY": n.vertex_level[v] = 2; pay(COST.city); e("city_built"); break;
    case "BUILD_ROAD": n.edge_owner[v] = p; if (s.phase === "MAIN") pay(COST.road); e("road_built", { free: s.phase !== "MAIN" }); break;
    case "MOVE_ROBBER": n.robber = v; n.phase = "MAIN"; e("robber_moved"); break;
    case "ROLL": n.rolled = true; n.phase = "MAIN"; e("dice_rolled", { dice: [3, 4], total: 7 }); break;
    case "END_TURN": e("turn_started", { turn: s.turn + 1 }); break;
  }
  if (a.type === "END_TURN" || s.phase === "SETUP_ROAD") {
    n.current = (s.current + 1) % s.num_players; n.turn = s.turn + 1; n.rolled = false; n.dev_played = false;
    n.phase = s.phase.startsWith("SETUP") ? "SETUP_SETTLEMENT" : "PRE_ROLL";
  }
  return { position: n, events: ev, done: false };
}
