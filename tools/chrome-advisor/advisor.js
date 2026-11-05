// Capture the visible tab and ask the local Catan Lab server for a move.
// Reads only the screenshot pixels: no content scripts, no DOM, no network interception.
export const DEFAULTS = { server: "http://127.0.0.1:8000", backend: "auto", numPlayers: "4", myColor: "" };

export const settings = async () => ({ ...DEFAULTS, ...(await chrome.storage.local.get(Object.keys(DEFAULTS))) });

/** Loopback http origin only: the extension's host permission covers nothing else. */
export function serverOrigin(url) {
  const u = new URL(url);
  if (u.protocol !== "http:" || !["127.0.0.1", "localhost"].includes(u.hostname)) throw new Error("The server must be http://127.0.0.1:<port> or http://localhost:<port>.");
  return u.origin;
}

const state = (s) => chrome.storage.session.set({ state: s });

/** `rerun` sends the last screenshot again (with a new hand or colour) instead of capturing. */
export async function analyze({ rerun = false, hand = null } = {}) {
  const cfg = await settings();
  await state({ status: "running", at: Date.now() });
  chrome.action?.setBadgeText({ text: "…" });
  try {
    const server = serverOrigin(cfg.server);
    let shot = rerun ? (await chrome.storage.session.get("shot")).shot : null;
    if (!shot) {
      shot = await chrome.tabs.captureVisibleTab({ format: "png" });
      // the screenshot stays in memory-only session storage for "Analyze again"; skip it when over quota
      await chrome.storage.session.set({ shot }).catch(() => {});
    }
    const fd = new FormData();
    fd.append("image", await (await fetch(shot)).blob(), "tab.png");
    fd.append("backend", cfg.backend);
    fd.append("num_players", cfg.numPlayers);
    if (cfg.myColor) fd.append("my_color", cfg.myColor);
    if (hand) fd.append("my_hand", hand.join(","));
    const r = await fetch(`${server}/api/analysis/import-image`, { method: "POST", body: fd });
    const j = await r.json().catch(() => ({ detail: r.statusText }));
    if (!r.ok) throw new Error(`Server ${r.status}: ${typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail)}`);
    let { topo } = await chrome.storage.local.get("topo");
    if (!topo) {
      topo = await (await fetch(`${server}/api/topology`)).json();
      await chrome.storage.local.set({ topo });
    }
    await state({ status: "done", at: Date.now(), result: j, server });
    chrome.action?.setBadgeText({ text: "" });
  } catch (e) {
    const msg = /Failed to fetch|NetworkError/.test(e.message) ? `Can't reach ${cfg.server}. Start the lab server (uv run python -m catan.server).` : e.message;
    await state({ status: "error", at: Date.now(), error: msg });
    chrome.action?.setBadgeText({ text: "!" });
  }
}
