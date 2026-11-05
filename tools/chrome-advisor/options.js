import { DEFAULTS, serverOrigin, settings } from "./advisor.js";

const $ = (id) => document.getElementById(id);
const cfg = await settings();
for (const k of ["server", "backend", "numPlayers"]) $(k).value = cfg[k];

$("f").addEventListener("submit", async (e) => {
  e.preventDefault();
  try {
    const server = serverOrigin($("server").value || DEFAULTS.server);
    // only the one configured loopback origin is granted (optional_host_permissions in manifest.json)
    if (!(await chrome.permissions.request({ origins: [`${server}/*`] }))) throw new Error("Permission for that server was refused.");
    await chrome.storage.local.set({ server, backend: $("backend").value, numPlayers: $("numPlayers").value });
    $("server").value = server;
    $("msg").textContent = "Saved.";
  } catch (err) {
    $("msg").textContent = err.message;
  }
});
