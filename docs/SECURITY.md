# Security

_Regression tests: `tests/security/test_server_security.py`._

## Threat model

Catan Lab is a **local, single-user** app. One FastAPI process serves the UI, the JSON API, the bots and policy inference.
It binds `127.0.0.1` by default and has **no authentication**.
Anyone who can send it a request can read every hand and drive every seat.
For hot-seat play that is by design: the person at the keyboard owns all seats.

What we defend against:

| Adversary | Can do | Defence |
|---|---|---|
| **A web page in the user's browser** (any site the user visits while the server runs) | Send cross-site requests to `127.0.0.1:8000`. Simple POSTs need no CORS preflight. It can also DNS-rebind its own domain to 127.0.0.1, which gives it full same-origin read access | Host allowlist, refusal of cross-origin writes, no CORS headers |
| **A malformed or hostile API client** (a buggy script, a fuzzer) | Bad payloads, huge bodies, traversal names, expensive options | Validation at every endpoint, size caps, allowlisted options, confined file access |
| **A human player peeking** (UI or API) at hidden state during a live game with AI seats | Read AI hands, dev cards or the deck through side channels | Per-viewer filtering of views and events. No agent diagnostics while hands are hidden |

Out of scope:
- Other local users or processes on the same machine. They can reach the loopback port.
- Network attackers when someone binds `0.0.0.0` (see below).
- Malicious files planted in the data directory. Checkpoints are still loaded with `weights_only=True`.

### Binding to a network interface is unsafe

`uv run python -m catan.server --host 0.0.0.0` exposes an unauthenticated server.
Anyone on the network can see every hand, play any seat, fill the disk with saves and burn CPU with analysis.
The CLI now prints a warning for any non-loopback host.
Requests must also carry an allowed `Host`. Set `CATAN_ALLOWED_HOSTS=my-laptop.lan,192.168.1.20`, or `*` to disable the check, and only do that on a network you trust.

## Findings and fixes

| id | sev | finding | fix | test |
|---|---|---|---|---|
| S1 | high | **AI diagnostics leaked hidden info.** Every game view had `last_decision.info`, the bot's ranked candidate moves with scores and explanations. That reveals what the AI can afford (its hand) and which dev cards it can play, to a human opponent in a live game. | `last_decision` is sent only when every hand is visible anyway: `viewer=all` on AI-only games, games with no human seats, or finished games | `test_advance_and_view_do_not_leak_ai_private_info` |
| S2 | high | **`/advance` returned raw step events.** That included `dev_bought_detail`, the AI's drawn card, and `stolen_detail` between other seats, which carry `visible_to` and must be filtered. `/actions` had the same unfiltered `events` field. | Step events are filtered with the same `visible_to` rule as the view | same test |
| S3 | high | **DNS rebinding.** With no Host check, a site could re-point its domain at 127.0.0.1 and read every view, including all hands, and play the games. | `RequestGuard` (`catan/server/guard.py`) refuses a `Host` outside loopback names plus `CATAN_ALLOWED_HOSTS` | `test_dns_rebinding_host_refused`, `test_allowed_hosts_env` |
| S4 | medium | **Cross-site writes.** JSON endpoints were already safe, because FastAPI parses bodies only for `application/json`. Body-less POSTs (`/advance`, `/save`, `/replay/save`, `/saves/{name}/resume`) could be fired from any site with a form. | A non-GET request whose `Origin` differs from its `Host` gets 403. Non-browser clients send no Origin and are unaffected. The Vite dev proxy keeps Origin and Host equal | `test_cross_origin_writes_refused` |
| S5 | medium | **Agent options passed straight to constructors.** Any scalar kwarg reached `make_agent` from `/api/games` and `/api/lab/matches`. Examples: `search` with `rollouts=10**9`, `time_limit=1e9`, `device=...`. That's a CPU DoS from one request. | Allowlist with bounds, the same as the analysis API: `checkpoint`, `deterministic`, `rollouts` 1..32, `candidates` 1..12 | `test_agent_kwargs_allowlisted` |
| S6 | medium | **Analysis DoS.** `/api/analysis/game` has no time limit, and at `deep` it runs 8×25 rollouts with 4 workers per decision over a whole game. The regression test hung for minutes against the old code. Analyses also queue on an engine lock, each holding a server worker thread, so enough parallel requests stall every endpoint. | `deep` is refused for whole-game review (use it per position via `/evaluate`, which has a 120 s cap). At most 2 analysis requests at a time (1 running, 1 waiting); the rest get 429 | `test_analysis_bounds`, `test_analysis_concurrency_capped` |
| S7 | medium | **No request body limit.** uvicorn accepts any size. | `RequestGuard` buffers non-GET bodies up to 1 MB, then returns 413. This also covers chunked uploads without Content-Length. The largest real body, an analysis position, is about 10 KB. Deeply nested JSON inside the cap gets an error response and the server keeps serving | `test_body_size_cap`, `test_deeply_nested_json_does_not_kill_server` |
| S8 | low | **Checkpoint confinement was inconsistent.** `/api/checkpoints` lists `models/*.pt`, but game creation only accepted `runs/`, so a listed model couldn't be used. Analysis accepted both. | One rule everywhere: an existing file under `runs/` or `models/`, after `resolve()`. Symlinks that point outside are refused | `test_checkpoint_paths_confined` |
| S9 | low | **Unbounded saves.** Repeated `/save` calls could fill the disk. | At most 500 saves (507 after that). Overwriting an existing name is still allowed | `test_save_count_capped` |
| S10 | info | 0.0.0.0 binding wasn't flagged as unsafe. | CLI warning, plus this document | `test_server_cli_warns_on_public_bind` |

### Checked, no change needed
- **`torch.load`.** Every call site uses `weights_only=True`: `catan/rl/model.py`, two in `catan/rl/ppo.py`, and `tests/rl`. There is no `pickle`, `yaml.load`, `eval` or `exec` on request data.
- **Path traversal.** Each named-file endpoint validates the name with a strict regex and a `resolve()`/parent check:
  - replays (`/api/replays/{name}`, `replay-state?source=file`, analysis `source.kind=replay`)
  - saves (`/save` names and `/saves/{name}/resume`)
  - behavior profiles (`/api/behavior/{name}`)

  Tournaments and runs are listed by glob and take no name. Starlette's `StaticFiles` handles the UI. Encoded `..`, NUL, absolute paths and over-long names are covered by `test_named_file_endpoints_confined`.
- **CORS.** No `CORSMiddleware` is installed, so responses carry no `Access-Control-Allow-*` headers. Browsers block cross-origin reads, and preflighted requests fail (`test_no_cors_headers`). Keep it that way. Don't add `allow_origins=["*"]`. The one exception is the move-advisor extension (`tools/chrome-advisor/`). Its pinned origin `chrome-extension://iohkkbaieljdpkhpniegbceehkkbimio`, plus any origin in `CATAN_EXTENSION_ORIGINS`, may POST to `/api/analysis/import-image` only. That response echoes the origin in `Access-Control-Allow-Origin`. Every other path and origin is unchanged (`test_advisor_extension_origin_only_on_import_image`).
- **Hidden info elsewhere.**
  - `/api/games` lists only public fields (seats, phase, turn, winner).
  - Seat views need a human seat (`viewer=all` only for AI-only or finished games).
  - Replays of live human games return 403 on `/replay`, `/replay/save` and `replay-state?source=game`.
  - Analysis `source.kind=game` on a live human game is refused unless the viewer is a human seat, and positions are built from `player_view(viewer)`.
  - `/api/saves` returns metadata only. Save contents (hands, deck, RNG) are never served.

  Covered by `test_live_human_game_not_revealed_anywhere`.
- **Input validation.**
  - Actions go through `Action.from_json` and the engine's `is_legal`, which never raises.
  - Numeric query params are bounded (`steps` 1..5000, `index`/`since` ≥ 0).
  - Game options are bounded: seed, vp_to_win and max_turns.
  - Analysis options are bounded: `time_limit_ms`, `top_k`, `steps`, `seed`, `depth`, and policy options via the allowlist.
- **Dependencies.**
  - `pip-audit` over the 59 installed Python packages found no known vulnerabilities.
  - `npm audit --omit=dev` (the shipped bundle) found 0.
  - Full `npm audit` reports esbuild ≤0.24.2 via vite 5 (GHSA-67mh-4wv8-2f99). It affects only the **Vite dev server** (`npm run dev`), which a website could query. The fix is vite 6.4.4+ or 8, a major upgrade, so it is left for now. Don't run `npm run dev -- --host`, and stop the dev server when you're not using it.

## Residual risks

- **No authentication.** Any local process, or any network peer if you bind publicly with an allowed Host, has full control. Hot-seat play needs this. For multi-device play, `--online` adds per-seat tokens and an admin key: see [ONLINE.md](ONLINE.md).
- **CPU-heavy legitimate requests.**
  - A single `/evaluate` can use up to 120 s of CPU, `deep` with 4 worker processes.
  - `/game` at `standard` on a long game can take minutes.
  - `/advance?steps=5000` and `/lab/matches` `run_to_end` with `search` bots hold one game's lock for minutes.

  Those are bounded per request, but there is no global CPU budget.
- **Cancelled analyses still hold a slot.** The server doesn't stop work when the client aborts. If a user cancels a long `/evaluate` and starts two more, the third can get 429 until the abandoned one finishes, which takes at most 120 s. Before, the abandoned requests ran anyway, so the CPU cost is the same.
- **Memory.** Sessions are capped at 200, 50 of them finished. The analysis worker pool and loaded RL models stay resident.
- **Checkpoint parsing.** `weights_only=True` blocks pickle code execution. A crafted `.pt` could still hit bugs in torch's tensor deserializer. Only put trusted files in `runs/` and `models/`.
- **Information in errors and seats.** Some error messages, and seat `kwargs`, include absolute server paths, such as the resolved checkpoint. That is acceptable for a local app, but don't expose it.
- **Same-machine browsers on other ports.** Origin checks treat `http://localhost:3000` as foreign, which is intended. A malicious local web app on another port can still read nothing and write nothing. A local *non-browser* process is unrestricted, as stated in the threat model.
- **Vite dev-server advisory**, as above.

## Running the checks

```sh
uv run pytest tests/security
uv run ruff check .               # CI lint job (E9 + pyflakes)
uvx pip-audit -r <(uv export --no-hashes)
(cd web && npm audit --omit=dev)
```
