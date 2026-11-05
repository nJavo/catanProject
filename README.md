# Catan Lab

A complete Catan base game, its bots, and a lab to study them. Everything shares one rules engine (`catan/engine`).

- **The game.** Play in the browser against bots, hot-seat at one screen, or online with friends.
- **The bots.** Heuristic bots (`random`, `basic`, `intermediate`, `search`) and the trained champion, `jointb_gated`.
- **The lab.** Watch AI matches, step through replays, follow training runs, run tournaments, and analyze positions.
- **The Chrome advisor.** An extension that screenshots your game tab and shows the champion's suggested move.

## Install

You need [uv](https://docs.astral.sh/uv/) and Node 22.

```bash
git clone https://github.com/nJavo/catanProject.git && cd catanProject
uv sync                                  # Python 3.11+, all dependencies and the dev tools
(cd web && npm ci && npm run build)      # builds the web UI into web/dist
```

## Play

```bash
uv run python -m catan.server            # http://127.0.0.1:8000
```

Open the URL and choose **New game**. The server reads the agent list once, at startup, so restart it after you change the bots.

A headless game from the command line:

```bash
uv run python -m catan.play --agents intermediate basic basic random --seed 1
```

## Agents

| Agent | What it does |
|---|---|
| `random` | Picks uniformly from the legal actions. |
| `basic` | Places by pips, then builds greedily. Simple robber targeting and bank trading. |
| `intermediate` | Looks one move ahead with a board-aware evaluation, and plans trades. |
| `search` | Starts from intermediate's candidates, re-samples the hidden cards, and scores each candidate with rollouts. |
| `champion` (`jointb_gated`) | Search guided by a learned value net, plus a learned trade model that it uses only against opponents who negotiate. Slow: allow a few seconds per move. |
| `rl` | A PPO policy from a checkpoint you trained (`checkpoint=path`). |

`intermediate_trader` and `search_trader` are the same bots plus trade offers to other players.

The champion passed the promotion gate (`catan/zoo/gate.py`, entry `jointb_gated-v1` in `models/registry.json`). As one seat against three copies of the same bot, where chance is 25%:

| Opponents | Win rate | 95% CI | Games |
|---|---|---|---|
| 3 × basic | 70.5% | 63.8–76.4% | 200 |
| 3 × intermediate | 76.5% | 70.2–81.8% | 200 |

Its weights are in `models/` (about 12 MB in total): `jointb_best.pt` (trade head and core), `value_best.pt` (value net), `league_best.pt` (search prior), `nego_best.pt` (trade model) and `opp_best.pt` (opponent classifier). The registry checks each file's sha256 before it loads the champion.

## The lab

Open http://127.0.0.1:8000/#/lab with the server running.

| Tab | Shows |
|---|---|
| AI match | Seat 2 to 4 agents and watch them play, or simulate the whole game. |
| Replays | Saved games, move by move, with every hand. |
| Training runs | Curves and evaluations from `runs/<id>/`. |
| Tournaments | `results/tournaments/*.json`: win rates with confidence intervals. |
| Behavior | `results/behavior/*.json`: how each agent plays. |
| Leaderboard | The model registry and the champion's gate evidence. |

The commands that fill those tabs:

```bash
# train a PPO agent; the run appears under Training runs
uv run python scripts/train.py --config configs/smoke.yaml --run-id smoke          # about a minute
uv run python scripts/train.py --config configs/ppo_baseline.yaml --run-id my_run

# a tournament
uv run python scripts/tournament.py --agents search intermediate basic random --games 400 --workers 4 --seed 0

# behavior profiles
uv run python scripts/behavior_report.py --help

# register and gate a checkpoint; a model that passes becomes the champion
uv run python scripts/promote.py --help
uv run python scripts/ladder.py --help
```

The **Analysis** board (from the menu) sets up any position and asks a bot for the best move. **Import screenshot** reads a board from a picture: see [docs/VISION_IMPORT.md](docs/VISION_IMPORT.md). For the optional Gemini reader, copy `.env.example` to `.env` and set `GEMINI_API_KEY`.

## Online rooms

```bash
uv run python -m catan.server --online --port 8190
```

Each friend plays from a private seat link on their own device. See [docs/ONLINE.md](docs/ONLINE.md) for how to make the server reachable with Tailscale or a tunnel, and for the security model.

## Chrome advisor

Load `tools/chrome-advisor/` unpacked at `chrome://extensions`, start the server, and press **Alt+Shift+K** on your game tab. It only screenshots the visible tab when you ask. It does not read or control the game site. Use it for review and for private games where everyone agreed. See [its README](tools/chrome-advisor/README.md).

## Tests

```bash
uv run pytest                            # Python: engine, bots, server, security, vision, RL
uv run ruff check .
(cd web && npm run typecheck)
```

`tests/test_chrome_advisor.py` also drives the extension popup when Playwright is installed (`uv run --with playwright playwright install chromium`); otherwise that test is skipped.

## Security

The server has no authentication and binds to 127.0.0.1. Don't expose it on a network without `--online`. See [docs/SECURITY.md](docs/SECURITY.md).

## More

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): engine API, action and phase contracts, file formats, server HTTP API.
- [docs/ONLINE.md](docs/ONLINE.md), [docs/SECURITY.md](docs/SECURITY.md), [docs/VISION_IMPORT.md](docs/VISION_IMPORT.md).
