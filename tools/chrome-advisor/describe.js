// Move names in words, ported from web/src/rules.ts describeAction (board moves name their tiles).
// Keep the two in step: tools/chrome-advisor/test/describe.test.mjs pins the shared strings.
export const RES = ["wood", "brick", "sheep", "wheat", "ore"];
export const RES_LABEL = ["Lumber", "Brick", "Wool", "Grain", "Ore"];
export const PLAYER_NAMES = ["Ember", "Tide", "Chalk", "Heather"];

const resList = (v) => v.map((n, r) => (n ? `${n} ${RES_LABEL[r]}` : "")).filter(Boolean).join(", ") || "nothing";

export function tileName(layout, h) {
  const t = layout.hex_terrain[h], n = layout.hex_number[h];
  return t === "desert" ? "Desert" : `${RES_LABEL[RES.indexOf(t)]} ${n ?? ""}`.trim();
}
const tilesName = (layout, hs) => hs.map((h) => tileName(layout, h)).join(", ") || "coast";
const vertexName = (w, v) => tilesName(w.layout, w.topo.vertices[v].hexes);
const edgeName = (w, e) => {
  const [a, b] = w.topo.edges[e].v.map((id) => w.topo.vertices[id].hexes);
  return tilesName(w.layout, a.filter((h) => b.includes(h)));
};

/** `at` = {topo, layout}: board moves name tiles. `seat(i)` names a seat (default: our seat names). */
export function describeAction(a, at, seat = (i) => PLAYER_NAMES[i]) {
  const v = a.value;
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
    case "STEAL": return `Steal from ${seat(v)}`;
    case "MARITIME_TRADE": return `Bank: ${RES_LABEL[v[0]]} for ${RES_LABEL[v[1]]}`;
    case "OFFER_TRADE": return `Offer ${resList(v[0])} for ${resList(v[1])}`;
    case "COUNTER_OFFER": return `Counter: ${resList(v[1])} for ${resList(v[0])}`;
    case "ACCEPT_TRADE": return `${seat(a.player)} accepts the trade`;
    case "REJECT_TRADE": return `${seat(a.player)} declines the trade`;
    case "CONFIRM_TRADE": return `Trade with ${seat(v)}`;
    case "CANCEL_TRADE": return "Cancel trade";
    default: return a.type;
  }
}
