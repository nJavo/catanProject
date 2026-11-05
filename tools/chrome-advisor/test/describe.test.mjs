// node --test tools/chrome-advisor/test/
// Strings must match web/src/rules.ts describeAction for the same action.
import assert from "node:assert/strict";
import { test } from "node:test";
import { describeAction, tileName } from "../describe.js";

const layout = { hex_terrain: ["ore", "wheat", "desert"], hex_number: [8, 2, null] };
const topo = {
  vertices: [{ hexes: [0, 1] }, { hexes: [0, 1, 2] }, { hexes: [] }],
  edges: [{ v: [0, 1] }, { v: [1, 2] }],
};
const at = { topo, layout };

test("board moves name tiles", () => {
  assert.equal(tileName(layout, 2), "Desert");
  assert.equal(describeAction({ type: "BUILD_SETTLEMENT", value: 1 }, at), "Settlement on Ore 8, Grain 2, Desert");
  assert.equal(describeAction({ type: "BUILD_CITY", value: 0 }, at), "City on Ore 8, Grain 2");
  assert.equal(describeAction({ type: "BUILD_ROAD", value: 0 }, at), "Road along Ore 8, Grain 2");
  assert.equal(describeAction({ type: "BUILD_ROAD", value: 1 }, at), "Road along coast");
  assert.equal(describeAction({ type: "MOVE_ROBBER", value: 1 }, at), "Robber to Grain 2");
});

test("other moves", () => {
  assert.equal(describeAction({ type: "MARITIME_TRADE", value: [4, 0] }), "Bank: Ore for Lumber");
  assert.equal(describeAction({ type: "OFFER_TRADE", value: [[1, 0, 0, 0, 0], [0, 0, 0, 0, 2]] }), "Offer 1 Lumber for 2 Ore");
  assert.equal(describeAction({ type: "STEAL", value: 1 }), "Steal from Tide");
  assert.equal(describeAction({ type: "STEAL", value: 1 }, undefined, () => "blue"), "Steal from blue");
  assert.equal(describeAction({ type: "END_TURN" }), "End turn");
});
