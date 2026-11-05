// "Quick update" for live play next to a physical board. These are client-side
// mirrors of the engine rules (catan/engine/game.py); results are an ESTIMATE
// and the backend validator remains the judge. Opponents' hands are tracked as
// counts plus a known lower bound (known_hand_min): production and trades are
// public, so cards received that way are known; unknown losses lower every bound.
import type { PositionSpec, Topology } from "../api";
import { RES } from "../rules";

export const COST = { road: [1, 1, 0, 0, 0], settlement: [1, 1, 1, 1, 0], city: [0, 0, 0, 2, 3], dev: [0, 0, 1, 1, 1] };
const Z = () => [0, 0, 0, 0, 0];
const sum = (a?: number[]) => (a ?? []).reduce((x, y) => x + y, 0);

export const cardCount = (s: PositionSpec, p: number) => (p === s.viewer ? sum(s.players[p].hand) : s.players[p].resource_count ?? 0);

/** Add (or with negative values remove) typed cards. Mutates s. */
function addTyped(s: PositionSpec, p: number, d: number[]) {
  const pl = s.players[p];
  if (p === s.viewer) { pl.hand = (pl.hand ?? Z()).map((x, r) => Math.max(0, x + d[r])); return; }
  pl.resource_count = Math.max(0, (pl.resource_count ?? 0) + sum(d));
  pl.known_hand_min = (pl.known_hand_min ?? Z()).map((x, r) => Math.max(0, x + d[r]));
  clampKnown(s, p);
}

/** An opponent lost `k` cards of unknown type: every known bound may drop by k. */
function loseUnknown(s: PositionSpec, p: number, k: number) {
  const pl = s.players[p];
  pl.resource_count = Math.max(0, (pl.resource_count ?? 0) - k);
  pl.known_hand_min = (pl.known_hand_min ?? Z()).map((x) => Math.max(0, x - k));
  clampKnown(s, p);
}

function clampKnown(s: PositionSpec, p: number) {
  const pl = s.players[p], km = pl.known_hand_min ?? Z();
  let over = sum(km) - (pl.resource_count ?? 0);
  for (let r = 4; r >= 0 && over > 0; r--) { const t = Math.min(km[r], over); km[r] -= t; over -= t; }
  pl.known_hand_min = km;
}

const bankAdd = (s: PositionSpec, d: number[]) => { if (s.bank) s.bank = s.bank.map((x, r) => x + d[r]); };

/** Production for a non-7 roll, same rules as Game._produce (robber blocks; shortage rule only when the bank is known). */
export function produce(s: PositionSpec, total: number, topo: Topology) {
  const n = s.num_players, owed = Array.from({ length: n }, Z);
  topo.hexes.forEach((h) => {
    if (s.layout.hex_number[h.id] !== total || h.id === s.robber) return;
    const r = RES.indexOf(s.layout.hex_terrain[h.id] as any);
    if (r < 0) return;
    h.vertices.forEach((v) => { const o = s.vertex_owner[v]; if (o >= 0 && o < n) owed[o][r] += s.vertex_level[v]; });
  });
  const gained = Array.from({ length: n }, Z), notes: string[] = [];
  for (let r = 0; r < 5; r++) {
    const need = owed.reduce((a, o) => a + o[r], 0);
    if (!need) continue;
    const claimants = owed.map((o, q) => (o[r] > 0 ? q : -1)).filter((q) => q >= 0);
    if (!s.bank || need <= s.bank[r]) claimants.forEach((q) => (gained[q][r] = owed[q][r]));
    else if (claimants.length === 1) { gained[claimants[0]][r] = s.bank[r]; notes.push(`Bank short of ${RES[r]}: partial payout`); }
    else notes.push(`Bank short of ${RES[r]}: nobody receives it`);
  }
  if (!s.bank) notes.push("Bank contents unknown: assumed it can pay everyone");
  gained.forEach((g, q) => { if (sum(g)) { addTyped(s, q, g); bankAdd(s, g.map((x) => -x)); } });
  s.rolled = true;
  s.phase = "MAIN";
  return { gained, notes };
}

/** Roll a 7: seats over 7 cards must discard half; otherwise move the robber. */
export function rollSeven(s: PositionSpec) {
  const d = Array.from({ length: 4 }, (_, p) => (p < s.num_players && cardCount(s, p) > 7 ? Math.floor(cardCount(s, p) / 2) : 0));
  s.rolled = true;
  s.discard_remaining = d;
  s.phase = d.some((x) => x > 0) ? "DISCARD" : "MOVE_ROBBER";
}

/** Discard: typed cards for the viewer, a count for opponents. */
export function discard(s: PositionSpec, p: number, cards: number[] | number) {
  if (Array.isArray(cards)) { addTyped(s, p, cards.map((x) => -x)); bankAdd(s, cards); } else loseUnknown(s, p, cards);
  const k = Array.isArray(cards) ? sum(cards) : cards;
  s.discard_remaining = (s.discard_remaining ?? [0, 0, 0, 0]).map((x, q) => (q === p ? Math.max(0, x - k) : x));
  if (s.phase === "DISCARD" && !s.discard_remaining.some((x) => x > 0)) s.phase = "MOVE_ROBBER";
}

/** Seats with a building on the robber's hex, other than the mover. */
export function stealVictims(s: PositionSpec, topo: Topology) {
  return [...new Set(topo.hexes[s.robber].vertices.map((v) => s.vertex_owner[v]))].filter((o) => o >= 0 && o !== s.current && cardCount(s, o) > 0);
}

export function moveRobber(s: PositionSpec, hex: number, topo: Topology) {
  s.robber = hex;
  s.phase = stealVictims(s, topo).length ? "STEAL" : "MAIN";
}

/** Steal one card. `r` is known when the viewer is thief or victim; null otherwise. */
export function steal(s: PositionSpec, thief: number, victim: number, r: number | null) {
  const one = Z();
  if (r !== null) { one[r] = 1; addTyped(s, victim, one.map((x) => -x)); addTyped(s, thief, one); }
  else { loseUnknown(s, victim, 1); const pl = s.players[thief]; pl.resource_count = (pl.resource_count ?? 0) + 1; } // opponent to opponent only
  s.phase = "MAIN";
}

/** Pay for a build. Opponents spend typed cards, so their known bound drops by exactly the cost. */
export function pay(s: PositionSpec, p: number, cost: number[]) {
  addTyped(s, p, cost.map((x) => -x));
  bankAdd(s, cost);
}

/** Trade `give` from `a` for `get`, with seat `b` or the bank (b = -1). */
export function trade(s: PositionSpec, a: number, b: number, give: number[], get: number[]) {
  addTyped(s, a, give.map((x, r) => get[r] - x));
  if (b >= 0) addTyped(s, b, give.map((x, r) => x - get[r]));
  else bankAdd(s, give.map((x, r) => x - get[r]));
}

export function buyDev(s: PositionSpec, p: number, card: number | null) {
  pay(s, p, COST.dev);
  const pl = s.players[p];
  if (p === s.viewer && card !== null) {
    pl.dev_cards = [...(pl.dev_cards ?? Z())]; pl.dev_cards[card]++;
    pl.dev_new = [...(pl.dev_new ?? Z())]; pl.dev_new[card]++;
  } else pl.dev_count = (pl.dev_count ?? 0) + 1;
  if (s.dev_deck_count !== undefined) s.dev_deck_count = Math.max(0, s.dev_deck_count - 1);
}

export function playKnight(s: PositionSpec, p: number) {
  const pl = s.players[p];
  if (p === s.viewer) { pl.dev_cards = [...(pl.dev_cards ?? Z())]; pl.dev_cards[0] = Math.max(0, pl.dev_cards[0] - 1); }
  else pl.dev_count = Math.max(0, (pl.dev_count ?? 0) - 1);
  pl.knights_played++;
  s.dev_played = true;
  s.phase = "MOVE_ROBBER";
}

// ---- setup (snake order 0..n-1, n-1..0), mirrors _do_build_settlement / _do_build_road
const settlements = (s: PositionSpec) => s.vertex_owner.filter((o) => o >= 0).length;
const setupSeat = (s: PositionSpec, k: number) => (k < s.num_players ? k : 2 * s.num_players - 1 - k);

export function setupSettlement(s: PositionSpec, v: number, topo: Topology) {
  const p = s.current, k = settlements(s);
  s.vertex_owner[v] = p; s.vertex_level[v] = 1;
  if (k >= s.num_players) { // second settlement: one card per adjacent producing hex
    const g = Z();
    topo.vertices[v].hexes.forEach((h) => { const r = RES.indexOf(s.layout.hex_terrain[h] as any); if (r >= 0 && (!s.bank || s.bank[r] - g[r] > 0)) g[r]++; });
    addTyped(s, p, g); bankAdd(s, g.map((x) => -x));
  }
  s.phase = "SETUP_ROAD";
}

export function setupRoad(s: PositionSpec, e: number) {
  s.edge_owner[e] = s.current;
  const k = settlements(s);
  if (k >= 2 * s.num_players) { s.phase = "PRE_ROLL"; s.current = 0; s.turn = 0; s.rolled = false; }
  else { s.phase = "SETUP_SETTLEMENT"; s.current = setupSeat(s, k); }
}

export function nextPlayer(s: PositionSpec) {
  if (s.phase.startsWith("SETUP")) { s.phase = "SETUP_SETTLEMENT"; s.current = setupSeat(s, Math.min(settlements(s), 2 * s.num_players - 1)); return; }
  s.current = (s.current + 1) % s.num_players;
  s.turn += 1;
  s.phase = "PRE_ROLL";
  s.rolled = false;
  s.dev_played = false;
  s.discard_remaining = [0, 0, 0, 0];
  // a pending trade (from a replay's trade step) belongs to the old turn; left in, the validator rejected PRE_ROLL
  const x = s as any;
  delete x.trade; delete x.actor;
  if (x.trades_this_turn !== undefined) x.trades_this_turn = 0;
  const me = s.players[s.viewer];
  me.dev_new = Z();
}
