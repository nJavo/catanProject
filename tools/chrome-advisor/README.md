# Catan Lab move advisor (Chrome extension)

Advisor for private games with friends who agree. Doesn't read or control colonist.io.

When you press the toolbar button or the hotkey, the extension takes a screenshot of the visible tab. It sends the PNG to your own lab server (`POST /api/analysis/import-image`) and shows:

- the detected position, with a confidence summary that names the unsure tiles and the fields that were not read;
- the bot's suggested move in words ("Settlement on Ore 8, Grain 2, Wool 5"), plus two alternatives;
- which reader ran (Gemini or colour tables), with the call count, tokens and price when Gemini reports them;
- an **Open in Analysis** link that loads the position into the Analysis board, with the unsure fields outlined, so you can fix it and run a full analysis.

You make every move yourself.

## What it does not do

- It reads only the pixels of the visible tab, and only when you press the button or the hotkey.
- It has no content scripts. It does not read colonist.io's page, DOM, websocket or network traffic, and it uses no `webRequest` or `debugger` permission.
- It never clicks, types or sends keys to any page.
- The screenshot goes only to the lab server you configure, which must be on `127.0.0.1` or `localhost`. With the Gemini reader, that server forwards it to Google (see `docs/VISION_IMPORT.md`).

## Permissions

| Permission | Why |
|---|---|
| `activeTab` | Lets `chrome.tabs.captureVisibleTab` screenshot the tab you are looking at. Chrome grants it only when you click the toolbar button or press the hotkey. |
| `storage` | Keeps your settings and colour (`storage.local`), and the last screenshot and result in memory (`storage.session`). |
| `http://127.0.0.1:8000/*` | The default lab server. |
| optional `http://127.0.0.1:*/*`, `http://localhost:*/*` | Asked for only when you save a different server port in Options. Only that one origin is granted. |

No `<all_urls>`, no `tabs`, `scripting`, `webRequest` or `debugger`.

## Install (load unpacked)

1. Start the lab server: `uv run python -m catan.server` at the repo root. It serves http://127.0.0.1:8000. For Gemini, put `GEMINI_API_KEY=...` in `.env` first (`docs/VISION_IMPORT.md`).
2. In Chrome, open `chrome://extensions` and turn on **Developer mode**.
3. Click **Load unpacked** and pick the `tools/chrome-advisor/` folder.
4. Pin "Catan Lab move advisor" in the toolbar. To use a different server port, the reader or the player count, open the extension's **Options**.
5. Open your game tab and click the icon, or press **Alt+Shift+K**. Change the hotkey at `chrome://extensions/shortcuts`.

The extension id is always `iohkkbaieljdpkhpniegbceehkkbimio`, because the `key` in `manifest.json` pins it. The server lets that origin, and only it, post to `/api/analysis/import-image` (`catan/server/guard.py`). If you change or remove the key, add your new id to the server's environment: `CATAN_EXTENSION_ORIGINS=chrome-extension://<id>`.

## Using it

- **Analyze as:** pick your colour. It is stored. The server then reads the position from your seat: your hand is yours, and the opponents' hands stay hidden counts.
- **My hand:** the reader cannot always see your cards. Enter them and press **Again with my hand**. That re-sends the last screenshot instead of taking a new one.
- Opening the popup takes a screenshot, unless the last one is under 30 seconds old or still running. This avoids paying for a Gemini call every time you look at the result.
- The analysis runs in the service worker, so you can close the popup. The icon shows `…` while it runs and `!` on an error. The hotkey runs an analysis without opening the popup; click the icon to see the result.
- The colour reader does not know colour names, so **Analyze as** only works with the Gemini reader. With the colour reader, set your seat in Analysis.

## Tests

- `.venv/bin/python -m pytest tests/test_chrome_advisor.py`: checks the manifest permissions and the pinned id. Then it loads `popup.html` in Playwright, with a stubbed `chrome` API and a mocked server, and checks the move text, the reader line, the Analysis link and the uploaded fields.
- `node --test tools/chrome-advisor/test/describe.test.mjs`: the move wording. It must match `describeAction` in `web/src/rules.ts`.

## Files

`manifest.json`, `sw.js` (service worker: hotkey and messages), `advisor.js` (capture and upload), `describe.js` (move names, a port of `describeAction`), `popup.*`, `options.*`.
The CSS copies the tokens from `web/src/styles.css`. The extension has no build step, so the tokens can't be imported.
