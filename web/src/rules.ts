// Display vocabulary mirrored from catan/engine (indices are the contract).
import type { Action, GameEvent, Layout, Seat, Topology } from "./api";

export const RES = ["wood", "brick", "sheep", "wheat", "ore"] as const;
export const RES_LABEL = ["Lumber", "Brick", "Wool", "Grain", "Ore"];
export const DEV = ["knight", "victory_point", "road_building", "year_of_plenty", "monopoly"] as const;
export const DEV_LABEL = ["Knight", "Victory Point", "Road Building", "Year of Plenty", "Monopoly"];
export const DEV_HELP = [
  "Move the robber and steal one card.",
  "Worth 1 hidden victory point. Never played.",
  "Place 2 roads for free.",
  "Take any 2 cards from the bank.",
  "Name a resource. Every opponent gives you all of theirs.",
];
export const COST: Record<string, number[]> = {
  road: [1, 1, 0, 0, 0],
  settlement: [1, 1, 1, 1, 0],
  city: [0, 0, 0, 2, 3],
  dev: [0, 0, 1, 1, 1],
};

// Okabe-Ito derived, ordered so any 2-4 subset stays distinguishable for
// common color-vision deficiencies. Each seat also has a crest shape.
export const PLAYER_COLORS = ["#E2611B", "#2C7FC2", "#EDE6D6", "#9E5BB0"];
export const PLAYER_INK = ["#5A2208", "#0E3354", "#4A4236", "#3B1A45"];
export const PLAYER_NAMES = ["Ember", "Tide", "Chalk", "Heather"];
export const CREST = ["circle", "diamond", "triangle", "square"] as const;

// Registry name of the gate-promoted model behind the `champion` agent (set once by App).
let champName: string | null = null, champKind: string | null = null;
export const setChampion = (c?: { name?: string; kind?: string } | null) => { champName = c?.name ?? null; champKind = c?.kind ?? null; };
export const championKind = () => champKind;
export const championName = () => champName;

/** `long` adds the Champion's model name; narrow spots (opponent panel, banner) use the short form. */
export function seatName(i: number, seats?: Seat[], long = false): string {
  const s = seats?.[i];
  if (!s || s.type === "human") return s?.name || PLAYER_NAMES[i];
  if (s.agent === "champion") return `${PLAYER_NAMES[i]} (Champion${long && champName ? ` ${champName}` : ""})`;
  // Replay specs carry the full checkpoint path; the short form keeps the run and checkpoint names.
  return `${PLAYER_NAMES[i]} (${long || !s.agent ? s.agent : shortSpec(s.agent)})`;
}

/** "rl:checkpoint=runs/baseline_r1/checkpoints/ckpt_00200.pt" -> "rl:baseline_r1/ckpt_00200". */
export function shortSpec(spec: string) {
  const i = spec.indexOf(":");
  if (i < 0) return spec;
  const kws = spec.slice(i + 1).split(",").map((kv) => {
    if (!kv.includes("=")) return kv; // already short ("rl:baseline_r1/ckpt_00120")
    const [k, v = ""] = kv.split("=");
    if (!v.includes("/") && !/\.(pt|pth|ckpt)$/.test(v)) return `${k}=${v}`;
    // runs/<run>/[checkpoints/]<file>: keep the run, or every "runs/X/bc.pt" read as plain "bc" (DS33-02)
    const m = v.match(/(?:^|\/)runs\/([^/]+)\/(?:checkpoints\/)?([^/]+?)(?:\.\w+)?$/);
    return m ? `${m[1]}/${m[2]}` : v.split("/").pop()!.replace(/\.\w+$/, "");
  });
  return `${spec.slice(0, i)}:${kws.join(",")}`;
}


export function seatShort(i: number): string {
  return PLAYER_NAMES[i];
}

const resList = (v: number[]) =>
  v.map((n, r) => (n ? `${n} ${RES_LABEL[r]}` : "")).filter(Boolean).join(", ") || "nothing";

/** One log line per discard, not per card: consecutive "discarded" events of a player are merged
 *  into one event carrying per-resource counts (it keeps the first event's seq). */
export function groupDiscards(events: GameEvent[]): GameEvent[] {
  const out: GameEvent[] = [];
  for (const e of events) {
    const last = out[out.length - 1];
    if (e.type === "discarded" && last?.type === "discarded" && last.player === e.player && Array.isArray(last.resources)) {
      last.resources[e.resource] += 1;
      continue;
    }
    if (e.type === "discarded") {
      const resources = [0, 0, 0, 0, 0];
      resources[e.resource] = 1;
      out.push({ ...e, resources });
    } else out.push(e);
  }
  return out;
}

export function describeEvent(e: GameEvent, seats?: Seat[], layout?: Layout): string | null {
  const n = (p: number) => seatShort(p);
  switch (e.type) {
    case "game_started": return "The island is laid out. Each player places two settlements and two roads.";
    case "game_resumed": return `Game resumed from save "${e.save}"`;
    case "settlement_built": return `${n(e.player)} built a settlement${e.free ? " (setup)" : ""}`;
    case "city_built": return `${n(e.player)} upgraded to a city`;
    case "road_built": return `${n(e.player)} built a road${e.free ? " (free)" : ""}`;
    case "dice_rolled": return `${n(e.player)} rolled ${e.dice[0]} + ${e.dice[1]} = ${e.total}`;
    case "resources_gained":
      return `${n(e.player)} received ${resList(e.resources)}${e.reason === "setup" ? " (starting hand)" : e.reason === "year_of_plenty" ? " (Year of Plenty)" : ""}`;
    case "discard_required": {
      const who = (e.amounts as number[]).map((a, p) => (a ? `${n(p)} ${a}` : "")).filter(Boolean).join(", ");
      return `Robber! Discard half: ${who}`;
    }
    case "discarded":
      if (Array.isArray(e.resources))  // grouped by groupDiscards (DS3-08)
        return `${n(e.player)} discarded ${e.resources.map((k: number, r: number) => (k ? `${k} ${RES_LABEL[r]}` : "")).filter(Boolean).join(", ")}`;
      return `${n(e.player)} discarded 1 ${RES_LABEL[e.resource]}`;
    case "robber_moved": return `${n(e.player)} moved the robber${layout && e.hex != null ? ` to ${tileName(layout, e.hex)}` : ""}`;
    case "stolen": return `${n(e.thief)} stole a card from ${n(e.victim)}`;
    case "stolen_detail": return `The stolen card was ${RES_LABEL[e.resource]}`;
    case "dev_bought": return `${n(e.player)} bought a development card`;
    // replays and lab matches reveal every hand, so "You drew" named the wrong player there
    case "dev_bought_detail": return `${n(e.player)} drew ${DEV_LABEL[DEV.indexOf(e.card)]}`;
    case "dev_played": return `${n(e.player)} played ${DEV_LABEL[DEV.indexOf(e.card)]}`;
    case "monopoly": return `${n(e.player)} claimed ${e.amount} ${RES_LABEL[e.resource]} by Monopoly`;
    case "maritime_trade": return `${n(e.player)} traded ${e.amount} ${RES_LABEL[e.give]} to the bank for 1 ${RES_LABEL[e.get]}`;
    case "trade_offered": return `${n(e.player)} offers ${resList(e.give)} for ${resList(e.get)}`;
    case "trade_response":
      return e.response === "counter"
        // counter bundles are from the proposer's side; worded like an offer: the responder gives `get`, wants `give`
        ? `${n(e.player)} counters: offers ${resList(e.get)} for ${resList(e.give)}`
        : `${n(e.player)} ${e.response === "accept" ? "accepts" : "declines"}`;
    // the proposer (player) pays `give` and receives `get`
    case "trade_executed": return `${n(e.player)} gave ${n(e.partner)} ${resList(e.give)} for ${resList(e.get)}`;
    case "trade_cancelled": return e.reason === "no_takers" ? "No one took the offer" : `${n(e.player)} withdrew the offer`;
    case "turn_started": return `Turn ${e.turn + 1}: ${seatName(e.player, seats)}`;
    case "award_changed": {
      const award = e.award === "longest_road" ? "Longest Road" : "Largest Army";
      return e.to === -1 ? `${award} is now unclaimed` : `${n(e.to)} takes ${award}`;
    }
    case "game_won": return `${n(e.player)} wins with ${e.victory_points} victory points`;
    case "game_truncated": return `Game stopped at the turn limit (${e.turn})`;
    default: return null;
  }
}

/** Board places named by their tiles ("Ore 8, Grain 2"), shared by the board's aria-labels and move lists. */
export interface Where { topo: Topology; layout: Layout }
export function tileName(layout: Layout, h: number): string {
  const t = layout.hex_terrain[h], n = layout.hex_number[h];
  return t === "desert" ? "Desert" : `${RES_LABEL[RES.indexOf(t as any)]} ${n ?? ""}`.trim();
}
export const tilesName = (layout: Layout, hs: number[]) => hs.map((h) => tileName(layout, h)).join(", ") || "coast";
const vertexName = (w: Where, v: number) => tilesName(w.layout, w.topo.vertices[v].hexes);
const edgeName = (w: Where, e: number) => {
  const [a, b] = w.topo.edges[e].v.map((id) => w.topo.vertices[id].hexes);
  return tilesName(w.layout, a.filter((h) => b.includes(h)));
};

/** With `at`, board moves name tiles ("Settlement on Ore 8, Grain 2"); without it, engine ids. */
export function describeAction(a: Action, at?: Where): string {
  const v = a.value as any;
  switch (a.type) {
    case "BUILD_SETTLEMENT": return at ? `Settlement on ${vertexName(at, v)}` : `Settlement at ${v}`;
    case "BUILD_CITY": return at ? `City on ${vertexName(at, v)}` : `City at ${v}`;
    case "BUILD_ROAD": return at ? `Road along ${edgeName(at, v)}` : `Road on ${v}`;
    case "ROLL": return "Roll dice";
    case "END_TURN": return "End turn";
    case "BUY_DEV": return "Buy development card";
    case "PLAY_KNIGHT": return "Play Knight";
    case "PLAY_ROAD_BUILDING": return "Play Road Building";
    case "PLAY_YEAR_OF_PLENTY": return `Year of Plenty: ${RES_LABEL[v[0]]} + ${RES_LABEL[v[1]]}`;
    case "PLAY_MONOPOLY": return `Monopoly on ${RES_LABEL[v]}`;
    case "DISCARD": return `Discard ${RES_LABEL[v]}`;
    case "MOVE_ROBBER": return at ? `Robber to ${tileName(at.layout, v)}` : `Robber to hex ${v}`;
    case "STEAL": return `Steal from ${seatShort(v)}`;
    case "MARITIME_TRADE": return `Bank: ${RES_LABEL[v[0]]} for ${RES_LABEL[v[1]]}`;
    case "OFFER_TRADE": return `Offer ${resList(v[0])} for ${resList(v[1])}`;
    case "COUNTER_OFFER": return `Counter: ${resList(v[1])} for ${resList(v[0])}`; // like "Offer X for Y": X is what the actor gives
    // trade replies come from every seat in turn, so name who replied (DS8-11)
    case "ACCEPT_TRADE": return `${seatShort(a.player)} accepts the trade`;
    case "REJECT_TRADE": return `${seatShort(a.player)} declines the trade`;
    case "CONFIRM_TRADE": return `Trade with ${seatShort(v)}`;
    case "CANCEL_TRADE": return "Cancel trade";
    default: return a.type;
  }
}

/** The turn number shown to people (1-based): the turn in progress, or after the game the last turn played.
 *  Stats' ``turns`` count END_TURNs instead: a won game's winning turn never ends, so
 *  this is turns + 1 there; a truncated game stops right after an END_TURN, so it equals turns (DS2-09). */
export const turnNumber = (turn: number, over = false, truncated = false) => (over && truncated ? turn : turn + 1);

export const PHASE_LABEL: Record<string, string> = {
  SETUP_SETTLEMENT: "Setup",
  SETUP_ROAD: "Setup",
  PRE_ROLL: "Roll",
  DISCARD: "Discard",
  MOVE_ROBBER: "Robber",
  STEAL: "Steal",
  MAIN: "Build & trade",
  ROAD_BUILDING: "Road Building",
  TRADE_RESPONSE: "Trade offer",
  TRADE_CONFIRM: "Trade offer",
  GAME_OVER: "Game over",
};
