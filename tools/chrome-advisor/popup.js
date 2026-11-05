import { describeAction, PLAYER_NAMES, tileName } from "./describe.js";

const $ = (id) => document.getElementById(id);
const COLOURS = ["red", "blue", "orange", "white", "green", "black", "purple", "bronze"];
const FIELD = { hands: "your hand", resource_counts: "opponents' card counts", dev_cards: "dev cards", knights_played: "knights played",
  current: "whose turn", turn: "turn number", phase: "phase" };
const PHASE = { SETUP_SETTLEMENT: "Setup: place a settlement", SETUP_ROAD: "Setup: place a road", MAIN: "Main turn" };
const FRESH_MS = 30_000; // reopening the popup within this shows the last result instead of a new (billed) read

const li = (ul, text, cls) => { const e = document.createElement("li"); e.textContent = text; if (cls) e.className = cls; ul.append(e); };
const b64url = (s) => btoa(s).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");

function hand() {
  const v = [...document.querySelectorAll("#hand input")].map((i) => i.value);
  if (v.every((x) => x === "")) return null;
  return v.map((x) => Math.max(0, Math.min(19, Number(x) || 0)));
}

function fillColours(colors, mine) {
  const sel = $("color");
  for (const c of new Set([...COLOURS, ...Object.keys(colors ?? {}), mine].filter(Boolean)))
    if (![...sel.options].some((o) => o.value === c)) sel.add(new Option(c[0].toUpperCase() + c.slice(1), c));
  sel.value = mine ?? "";
}

export function render(st, topo) {
  const busy = st?.status === "running";
  $("go").disabled = $("again").disabled = busy;
  $("status").textContent = busy ? "Reading the screenshot…" : st?.status === "error" ? st.error : "";
  $("status").className = st?.status === "error" ? "danger" : "mist";
  const r = st?.status === "done" ? st.result : null;
  $("result").hidden = !r;
  if (!r) return;
  const p = r.position, at = topo && { topo, layout: p.layout };
  const colourOf = Object.fromEntries(Object.entries(r.colors ?? {}).map(([c, s]) => [s, c]));
  const seat = (i) => colourOf[i] ?? PLAYER_NAMES[i] ?? `seat ${i}`;
  fillColours(r.colors, $("color").value);

  const c = r.suggestion?.candidates ?? [];
  $("move").textContent = c.length ? describeAction(c[0].action, at, seat) : "No suggestion";
  $("alts").replaceChildren();
  for (const x of c.slice(1, 3)) li($("alts"), describeAction(x.action, at, seat));
  $("notes").textContent = [r.suggestion_note, r.override_note].filter(Boolean).join(" ");

  const pos = $("position");
  pos.replaceChildren();
  li(pos, `${PHASE[p.phase] ?? p.phase}, ${p.num_players} players`);
  li(pos, `You: ${seat(p.viewer)}. To move: ${seat(p.current)}.`);
  const h = p.players[p.viewer]?.hand;
  if (h) li(pos, `Your hand: ${h.map((n, i) => (n ? `${n} ${["Lumber", "Brick", "Wool", "Grain", "Ore"][i]}` : "")).filter(Boolean).join(", ") || "empty"}`);
  for (const e of r.errors ?? []) li(pos, typeof e === "string" ? e : e.message, "danger");

  const low = r.low_confidence, conf = $("confidence");
  conf.replaceChildren();
  li(conf, `Board fit ${Math.round((r.confidence?.lattice ?? 0) * 100)}%`);
  li(conf, low.hexes.length ? `Unsure tiles: ${low.hexes.map((x) => (at ? tileName(p.layout, x) : `hex ${x}`)).join(", ")}` : "All 19 tiles read with confidence", low.hexes.length ? "warn" : "");
  if (low.vertices.length || low.edges.length) li(conf, `Unsure pieces: ${low.vertices.length} corners, ${low.edges.length} roads or harbors`, "warn");
  if (low.fields.length) li(conf, `Not read: ${low.fields.map((f) => FIELD[f] ?? f).join(", ")}`, "warn");

  const u = r.usage;
  $("reader").textContent = (r.backend === "llm" ? `Reader: Gemini (${r.profile.replace(/^gemini:/, "")})` : `Reader: colour tables (profile ${r.profile})`)
    + (u ? `, ${u.calls} call${u.calls === 1 ? "" : "s"}, ${u.total_tokens} tokens${u.usd != null ? `, $${u.usd.toFixed(4)}` : ""}` : "")
    + (r.backend_note ? `. ${r.backend_note}` : "");
  $("open").href = `${st.server}/#/analysis/import/${b64url(JSON.stringify({ position: p, low_confidence: low }))}`;
}

async function show() {
  const { state } = await chrome.storage.session.get("state");
  const { topo } = await chrome.storage.local.get("topo");
  render(state, topo);
  return state;
}

async function main() {
  const { myColor } = await chrome.storage.local.get("myColor");
  fillColours({}, myColor || "");
  $("color").addEventListener("change", (e) => chrome.storage.local.set({ myColor: e.target.value }));
  $("go").addEventListener("click", () => chrome.runtime.sendMessage({ type: "analyze" }));
  $("again").addEventListener("click", () => chrome.runtime.sendMessage({ type: "analyze", rerun: true, hand: hand() }));
  chrome.storage.onChanged.addListener((ch, area) => { if (area === "session" && ch.state) show(); });
  const st = await show();
  // the toolbar button is the trigger: read the tab now unless a read is running or just finished
  if (st?.status !== "running" && !(st && Date.now() - st.at < FRESH_MS)) chrome.runtime.sendMessage({ type: "analyze" });
}

main();
