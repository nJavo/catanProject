# Architecture and shared contracts

This document is the contract between the engine, the bots, the RL stack, the server and the web UI.
Change it only together with the code it describes.

## Layout

```
catan/engine/    authoritative rules (pure Python, no deps). board.py, actions.py, game.py
catan/agents/    Agent interface, RandomAgent, registry (name -> factory)
catan/bots/      heuristic and search bots (they register themselves in the registry)
catan/eval/      tournaments, statistics, ratings
catan/zoo/       model registry, promotion gate, ladders, leaderboard, champion
catan/rl/        observation/action encoding, env, PPO, self-play, checkpoints
catan/research/  the networks the champion (jointb_gated) is built from
catan/analysis/  position analysis (Analysis board)
catan/vision/    screenshot import
catan/server/    FastAPI app. Serves the JSON API and the built web UI
web/             Vite + React + TypeScript frontend
tools/           Chrome move-advisor extension
models/          shipped weights and models/registry.json
configs/         training YAML
scripts/         training, tournaments, promotion, ladders, behavior reports
tests/           pytest
runs/, results/  local outputs (gitignored)
```

One process (the FastAPI server) hosts games, bots and policy inference. We don't use microservices.

## Engine API (`catan.engine`)

```python
g = Game(num_players=4, seed=123)          # or Game(config=GameConfig(...), seed=..., layout=...)
g.actor                                    # seat that must act now
g.legal_actions() -> list[Action]          # only for g.actor
g.is_legal(action) -> bool
g.step(action) -> list[event dict]         # raises IllegalAction
g.is_over, g.winner (-1 = none), g.truncated
g.victory_points(p), g.public_victory_points(p)
g.player_view(viewer|None) -> dict         # information-set view (JSON-safe)
g.to_dict() / Game.from_dict(d)            # full state including RNG
g.clone()                                  # fast copy for search
Game.replay(seed, config, actions)         # deterministic re-simulation
```

`Action(player, type, value)` is a NamedTuple. JSON form: `{"player":0,"type":"BUILD_ROAD","value":17}`.
Tuples become nested lists in JSON.

| type | value | phases |
|---|---|---|
| BUILD_SETTLEMENT | vertex id | SETUP_SETTLEMENT, MAIN |
| BUILD_ROAD | edge id | SETUP_ROAD, MAIN, ROAD_BUILDING |
| BUILD_CITY | vertex id | MAIN |
| ROLL | – | PRE_ROLL |
| BUY_DEV | – | MAIN |
| PLAY_KNIGHT / PLAY_ROAD_BUILDING | – | PRE_ROLL, MAIN |
| PLAY_YEAR_OF_PLENTY | (r1, r2), sorted | PRE_ROLL, MAIN |
| PLAY_MONOPOLY | r | PRE_ROLL, MAIN |
| DISCARD | r (one card per action) | DISCARD |
| MOVE_ROBBER | hex id | MOVE_ROBBER |
| STEAL | victim seat | STEAL (only when there are 2 or more victims; 1 victim is stolen from automatically) |
| MARITIME_TRADE | (give_r, get_r). Rate = best port (2, 3 or 4) | MAIN |
| OFFER_TRADE | (give[5], get[5]), from the proposer's side | MAIN |
| ACCEPT_TRADE / REJECT_TRADE | – | TRADE_RESPONSE |
| COUNTER_OFFER | (give[5], get[5]), from the proposer's side | TRADE_RESPONSE |
| CONFIRM_TRADE | partner seat | TRADE_CONFIRM |
| CANCEL_TRADE | – | TRADE_CONFIRM |
| END_TURN | – | MAIN |

Resources are indexed `0 wood, 1 brick, 2 sheep, 3 wheat, 4 ore`. Dev types are indexed
`0 knight, 1 victory_point, 2 road_building, 3 year_of_plenty, 4 monopoly`.

### Phases

`SETUP_SETTLEMENT → SETUP_ROAD` runs in snake order 0..n-1, n-1..0. The second settlement pays starting resources.
Then the turn loop is `PRE_ROLL → (ROLL) → MAIN → END_TURN → next PRE_ROLL`.

- A 7 leads to `DISCARD` (each player with more than 7 cards discards one card per action, in seat order from the roller), then `MOVE_ROBBER`, then `STEAL` if needed, then back to `MAIN`.
- A knight leads to `MOVE_ROBBER`, then `STEAL`, then back to PRE_ROLL or MAIN, wherever it was played.
- Road Building leads to `ROAD_BUILDING`, which places up to 2 free roads and then goes back.
- A domestic trade goes `MAIN → TRADE_RESPONSE` (every other seat answers in order from the proposer's left) `→ TRADE_CONFIRM` (the proposer picks one acceptor or counter-offerer, or cancels) `→ MAIN`.
  If nobody accepts or counters, the offer cancels itself.
  The protocol caps offers at `max_trade_offers_per_turn` (default 3). This cap is a platform limit, not an official rule, and `None` removes it.
  Counter-offers go one round only, so every negotiation terminates.
  Whether a counter-offer is legal depends only on the responder's own hand. If the engine checked the proposer's hand too, the legal-action list would leak it. `CONFIRM_TRADE` is only legal for a response the proposer can currently pay.

### Rules decisions (official 5th edition unless noted)

- Variable set-up. Terrain is random. Tokens go A..R in a spiral from a random corner and skip the desert. The 9 harbor pieces are shuffled onto 9 fixed frame positions: coastal edge offsets 0,3,6,10,13,16,20,23,26 around the coast.
- One dev card per turn, at any time during your own turn, including before the roll. A card bought this turn can't be played that turn. VP cards are never "played": they count toward the hidden total.
- You win as soon as you have 10 or more VP during your own turn. That check runs after every action and at the start of your turn.
- Production shortage: if the bank can't pay everyone a resource, nobody gets it, unless only one player is owed. That player gets whatever is left.
- Longest Road needs 5 or more segments. The holder keeps it on a tie. If the holder loses the lead and the new top length is tied, or is under 5, nobody holds the card.
- Largest Army needs 3 or more knights. A new owner must have strictly more knights than the holder.
- Year of Plenty requires both cards to be available in the bank.
- Two players is an **unofficial variant**: standard rules with 2 seats.
- `GameConfig.max_turns` truncates automated games with no winner. Human games leave it at None.

### Events

`step` returns dicts like `{"type": "dice_rolled", "dice": [3, 4], "total": 7, ...}`.
Events that carry a `visible_to` list are private to those seats, for example `stolen_detail` and `dev_bought_detail`.
The server must filter events per viewer.

### Randomness and replay

All randomness comes from `Game.rng`, which is seeded with `seed`.
The draw order is: layout, then the dev deck shuffle, then dice and steal draws in action order.
So `(seed, config, actions)` reproduces a game exactly.
A replay file is JSON (`catan.engine.replay`):
`{"format": "catan-replay", "version": 1, "rules_digest", "seed", "config", "agents", "actions": [Action json...], "meta": {...}}`.

- `make_replay(...)` writes it. The server, `catan.eval.tournament` and `catan.play --save` all use it.
- `rules_digest` fingerprints the engine rules: it is the final state of 3 fixed random games. `check_rules(doc)` warns when a file was written under different rules, because re-simulation may then diverge.
- `load_replay(doc)` validates and migrates a file. Files written before versioning have no `format`/`version` field; they load as version 1, since the layout is the same, with `rules_digest: null`. Unknown versions or formats raise `ReplayFormatError`, and the server answers 422 (`/api/replays/{name}`, `/api/replay-state`, `/api/analysis/*`).
- A new version must keep a migration from every older version in `load_replay` and add it to `SUPPORTED_VERSIONS`.

**Determinism.** The same seed and actions give byte-identical games across CPython versions. The engine uses only `random.Random(int)` seeding, `randint`, `randrange`, `shuffle` and `choice`, and no game logic depends on set or dict iteration order of strings.

## Agent interface (`catan.agents.base.Agent`)

`act(game, legal) -> Action`, `act_with_info(game, legal) -> (Action, info)`, `reset(seat, seed)`.
Agents may only use information visible to `game.actor` (see the module docstring).
Register agents with `catan.agents.registry.register(name, factory, trusted=True)`.

**Information-set view (`catan.engine.infoset`).** `game.observe_for(seat)` returns an `InfoSetGame`. It is a sandboxed snapshot with:
- `legal_actions`, `is_legal` and `player_view`, plus every public field;
- `hand_count(q)`, `dev_count(q)` and `dev_new_count(q)`;
- `public_victory_points`, `supply`, `trade_rate` and `longest_road`;
- the seat's own `hand` and `dev`.

Hidden state raises `HiddenInformationError`: opponents' `hand`, `dev` and `dev_new`, `victory_points(opponent)`, the dev deck order (`len()` still works), `rng`, `seed` and `to_dict()`. `clone()` returns a **determinized** real `Game`, through `Game.determinized(me, rng)`, the same sampling `catan.bots.common.determinize` uses. `step()` on the view raises; step a clone instead. Mutating the view never touches the real game.

It is opt-in: `run_tournament(..., infoset=True)` / `scripts/tournament.py --infoset`, and the server's `create_app(infoset_agents=True)` or `CATAN_INFOSET_AGENTS=1`. It is mandatory for agents registered with `trusted=False`. It guards against accidental peeking; it is not a security boundary, so run hostile code out of process. `tests/bots/test_infoset_agents.py` runs every registered agent through it.
Modules `catan.bots` and `catan.rl.policy_agent` are imported automatically.

## Artifact formats

- Training run: `runs/<run_id>/` contains:
  - `config.yaml`
  - `meta.json` with `{run_id, git_commit, seeds, started, hardware, ...}`
  - `metrics.jsonl`, one JSON object per log step, with `{"step": int, "time": float, ...}`
  - `checkpoints/*.pt`
  - `eval.jsonl`
- Tournament result: `results/tournaments/<id>.json` holds `{id, agents, games, seeds, per_agent: {name: {wins, games, win_rate, ci95, avg_vp, ...}}, pairwise, created}`.
- Replays: `results/replays/<id>.json` (format above).
- Saved games: `results/saves/<name>.json` holds `{format: "catan-save", version: 1, name, saved, source, seats, game: Game.to_dict(), events, decisions}`.
  - `POST /api/games/{id}/save` takes `{name?}`. Names match `[A-Za-z0-9][A-Za-z0-9_-]{0,63}`.
  - `GET /api/saves` returns metadata only. A save holds hands, deck order and RNG, so the server never sends its contents to clients.
  - `POST /api/saves/{name}/resume` starts a new session with the same seats. The usual hidden-info rules apply.

## Server HTTP API (`catan/server`)

Start the server with `uv run python -m catan.server [--host 127.0.0.1] [--port 8000]`.
It reads and writes `runs/`, `results/` and `models/` under the data directory. That is `CATAN_DATA_DIR`, or the repo root when the variable isn't set.
Every route is in `catan/server/app.py` unless noted. The analysis routes are in `catan/server/analysis.py`.

### Viewer rule

`viewer` is a seat number, `spectator` (public information only) or `all` (everything revealed).
- `all` is refused (403) for a live game that has human seats.
- A seat number for an AI seat is refused (403) until the game ends.
- Any client may ask for any *human* seat. That is how hot-seat play works. Without `--online` there is no seat authentication ([ONLINE.md](ONLINE.md)).
- Events that carry `visible_to` are filtered per viewer.

### Games

| Route | Body / query | Returns |
|---|---|---|
| `POST /api/games` | `{players: [{type: "human"\|"agent", agent?, kwargs?}], seed?, vp_to_win?, max_turns?}` | `{id, seed, seats, first_human}` |
| `GET /api/games` | none | session list |
| `GET /api/games/{id}` | `?viewer=&since=` | the view: board, players, legal actions for the viewer, and events after `since` |
| `POST /api/games/{id}/actions` | `{action: Action json, since?}` | `{events, view}`. Returns 400 if the action is illegal or malformed, or if it's another seat's turn or an AI seat's turn |
| `POST /api/games/{id}/advance` | `?steps=1..5000&viewer=` | plays AI seats until a human must act: `{steps: [{action, events}], view}` |
| `GET /api/games/{id}/replay` | none | the replay JSON. Returns 403 for a live game with human seats |
| `POST /api/games/{id}/replay/save` | none | writes `results/replays/<ts>-<id>.json` |
| `POST /api/games/{id}/save`, `GET /api/saves`, `POST /api/saves/{name}/resume` | see "Artifact formats" | |

Agent `kwargs` must be scalars. A `checkpoint` must be an existing file under `runs/` or `models/`.
Each AI decision's `info` (scores and explanations) is kept with the session and included in replays under `meta.decisions`.
The server keeps up to 200 sessions. When it's full, it evicts finished games first and then the least recently used, so a live human game is never evicted. At most 50 finished games are kept.

### Replays and the hand inspector

| Route | Returns |
|---|---|
| `GET /api/replays` | metadata for `results/replays/*.json` |
| `GET /api/replays/{name}` | one replay file |
| `GET /api/replay-state?source=game\|file&ref=&index=` | The board after the first `index` actions, with **every hand revealed**. Also returns `last_events` (the events of action `index-1`), `prev_players` (hands and dev cards before that action, for the hand inspector diff), `total_actions`, `index` and the recorded `decisions`. The server re-simulates the replay, caching the most recent 8, and returns 422 if it doesn't re-simulate. `source=game` follows the same 403 rule as `/replay` |

### Lab, runs, tournaments and behavior

| Route | Returns |
|---|---|
| `POST /api/lab/matches` | Body `{agents: [name \| {agent, kwargs}], seed?, vp_to_win?, max_turns? (default 500), run_to_end?}`. Returns `{id, seed, seats, result?}`. `run_to_end` plays the whole game while holding the session lock |
| `GET /api/runs` | light list of `runs/*/` for polling: `{id, meta, kind ("rl"\|"bc"), updated, metrics_total, eval_total, last_eval, checkpoints (count), bc}`. No metric rows. ETag + `Cache-Control: no-cache` (304 when nothing changed) |
| `GET /api/runs/{id}?since=` | one run: `metrics` (on `since=0` the whole log, downsampled to 2000 points; otherwise only complete lines appended after byte offset `since`, raw), `cursor` (pass back as `since`), `reset` (log rewritten: `metrics` is the whole log again), `eval`, checkpoint names, `bc_summary`, and `config` on full fetches only. Path-safe id; ETag per cursor |
| `GET /api/checkpoints` | `runs/*/checkpoints/*.pt` and `models/**/*.pt`, plus `default`, the checkpoint an `rl` seat uses when it names none. That is the newest `runs/*/checkpoints/latest.pt` |
| `GET /api/tournaments` | every `results/tournaments/*.json`, in the format described under "Artifact formats" |
| `GET /api/behavior`, `GET /api/behavior/{file}` | `results/behavior/*.json`, written by `scripts/behavior_report.py` (`catan/server/behavior.py`) |
| `GET /api/agents`, `/api/topology`, `/api/rules`, `/api/health` | the registry names, the board graph, costs and resource names, and a liveness check |
