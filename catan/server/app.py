"""FastAPI game server: hosts in-memory game sessions, AI seats, replays and
lab artifacts, and serves the built web UI from ``web/dist``.

The engine is the only authority on legality. Every client action goes
through ``Action.from_json`` and ``Game.step``; the client never decides.
Hidden information is filtered here: ``player_view(viewer)`` plus
``visible_to`` on events.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, Header, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from catan.agents.registry import available_agents, is_trusted, make_agent
from catan.server.behavior import behavior_router
from catan.server.turns import turn_segments
from catan.server.checkpoints import resolve_checkpoint
from catan.server.guard import RequestGuard, allowed_hosts_from_env, extension_origins_from_env
from catan.server.online import OnlineGate, Room, admin_key
from catan.engine import (COST, DEV_TYPES, RESOURCES, Action, Game, GameConfig, IllegalAction,
                          topology_dict)
from catan.engine.replay import ReplayFormatError, load_replay as load_replay_doc, make_replay
from catan.server.leaderboard import make_router as make_zoo_router
from catan.server.analysis import IMAGE_BODY

ROOT = Path(__file__).resolve().parents[2]
SAFE_NAME = re.compile(r"^[A-Za-z0-9_.\-]{1,128}$")
SAVE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-]{0,63}$")
MAX_SESSIONS = 200
MAX_FINISHED = 50   # a finished long AI game holds ~2-6 MB of events/decisions
MAX_METRIC_POINTS = 2000
MAX_BODY = 1_000_000  # bytes; the largest legitimate body (an analysis position) is ~10 KB
MAX_SAVES = 500
# Agent constructor options a client may set, with bounds. Anything else (e.g. a
# huge search ``rollouts`` or ``time_limit``) is a CPU knob, so it is refused.
AGENT_KWARGS = {"checkpoint": str, "deterministic": bool, "rollouts": (1, 32), "candidates": (1, 12)}


# ----------------------------------------------------------------- helpers
def _hands(g: Game) -> list[dict]:
    return [{"hand": pl.hand[:], "dev_cards": pl.dev[:]} for pl in g.players]


def _jsonable(x: Any) -> Any:
    """Agent diagnostics may hold numpy/tuples; coerce to plain JSON."""
    try:
        return json.loads(json.dumps(x, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    except (TypeError, ValueError):
        return {"repr": str(x)}


def _safe_child(base: Path, name: str) -> Path:
    """Resolve ``base/name`` refusing traversal, absolute paths and odd names."""
    if not SAFE_NAME.match(name or "") or name in (".", ".."):
        raise HTTPException(400, "invalid name")
    p = (base / name).resolve()
    if p.parent != base.resolve():
        raise HTTPException(400, "invalid name")
    return p


def _read_jsonl(path: Path) -> list[dict]:
    out = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except OSError:
        pass
    return out


def _downsample(rows: list, n: int = MAX_METRIC_POINTS) -> list:
    if len(rows) <= n:
        return rows
    step = math.ceil(len(rows) / n)
    out = rows[::step]
    if out[-1] is not rows[-1]:
        out.append(rows[-1])
    return out


def _read_jsonl_from(path: Path, offset: int) -> tuple[list[dict], int, bool]:
    """Complete JSON lines of ``path`` from byte ``offset``: (rows, new offset, reset).
    A partial last line (a writer mid-append) is left for the next call. ``reset``
    means the cursor no longer fits the file (truncated or rewritten, e.g. a resumed
    run trimmed its log), so the rows are the whole file from the start."""
    try:
        size = path.stat().st_size
        with open(path, "rb") as f:
            reset = offset > size
            if offset and not reset:
                f.seek(offset - 1)
                reset = f.read(1) != b"\n"
            if reset:
                offset = 0
            f.seek(offset)
            data = f.read()
    except OSError:
        return [], 0, offset > 0
    end = data.rfind(b"\n") + 1
    rows = []
    for line in data[:end].splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows, offset + end, reset


def _count_lines(path: Path) -> int:
    try:
        with open(path, "rb") as f:
            return sum(chunk.count(b"\n") for chunk in iter(lambda: f.read(1 << 20), b""))
    except OSError:
        return 0


def _last_json_line(path: Path) -> Any:
    """Last parseable line of a JSONL file (eval logs are small)."""
    rows = _read_jsonl(path)
    return rows[-1] if rows else None


def _stat_key(paths) -> str:
    parts = []
    for q in paths:
        try:
            st = q.stat()
            parts.append(f"{q.name}:{st.st_size}:{st.st_mtime_ns}")
        except OSError:
            parts.append(f"{q.name}:-")
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]


def _read_json(path: Path) -> Any:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


# ---------------------------------------------------------------- sessions
class Seat:
    def __init__(self, kind: str, agent: str | None, kwargs: dict, agent_obj=None):
        self.kind = kind
        self.agent = agent
        self.kwargs = kwargs
        self.agent_obj = agent_obj
        self.name: str | None = None   # online rooms: the display name a seat link set

    def to_json(self):
        d = {"type": self.kind, "agent": self.agent, "kwargs": self.kwargs}
        if self.name:
            d["name"] = self.name
        return d


class Session:
    def __init__(self, gid: str, game: Game, seats: list[Seat], source: str):
        self.id = gid
        self.game = game
        self.seats = seats
        self.source = source            # "play" or "lab"
        self.created = self.touched = time.time()
        self.lock = threading.Lock()
        self.events: list[dict] = [{"seq": 0, "i": -1, "type": "game_started"}]
        self.decisions: dict[int, dict] = {}   # action index -> agent info
        self.fallback_rng = random.Random(game.seed)
        self.room: Room | None = None          # online mode only (catan/server/online.py)

    @property
    def has_humans(self):
        return any(s.kind == "human" for s in self.seats)

    def record(self, events: list[dict], step: bool = True):
        i = len(self.game.action_log) - 1
        for e in events:
            self.events.append({"seq": len(self.events), "i": i, **e})
        if not step:
            return
        # agents that track public events (belief search, traders) see their seat's view, as in tournaments
        for q, s in enumerate(self.seats):
            obs = getattr(s.agent_obj, "observe", None)
            if callable(obs):
                try:
                    obs(self.game, [e for e in events if e.get("visible_to") is None or q in e["visible_to"]])
                except Exception:  # a tracker bug must not break the table; the agent falls back
                    pass

    def replay_json(self):
        g = self.game
        return make_replay(g.seed, g.config.to_dict(), [s.agent if s.kind == "agent" else "human" for s in self.seats],
                           g.action_log,
                           {"game_id": self.id, "source": self.source, "created": self.created,
                            "winner": g.winner, "truncated": g.truncated, "turns": g.turn,
                            "vp": [g.victory_points(p) for p in range(g.n)],
                            "decisions": {str(k): v for k, v in self.decisions.items()}})


def _visible(e: dict, viewer: int | None) -> bool:
    vt = e.get("visible_to")
    return vt is None or (viewer is not None and viewer in vt)


def _view(sess: Session, viewer: int | None, since: int | None = None, reveal: bool = False) -> dict:
    g = sess.game
    if reveal:
        # Spectator "god view" for AI-only matches: every hand visible.
        v = g.player_view(None)
        for q, info in enumerate(v["players"]):
            full = g.player_view(q)["players"][q]
            info.update({k: full[k] for k in ("hand", "dev_cards", "dev_new", "vp")})
    else:
        v = g.player_view(viewer)
    evs = sess.events if since is None else sess.events[since:]
    evs = [e for e in evs if reveal or _visible(e, viewer)]
    if since is None:
        evs = evs[-120:]
    actor = None if g.is_over else g.actor
    legal = []
    if viewer is not None and actor == viewer and sess.seats[viewer].kind == "human":
        # Enumerated COUNTER_OFFERs are filtered by the proposer's hand, so
        # they would leak it. Clients compose counters; the engine validates.
        legal = [a.to_json() for a in g.legal_actions() if a.type != "COUNTER_OFFER"]
    v.update({
        "game_id": sess.id,
        "source": sess.source,
        "seats": [s.to_json() for s in sess.seats],
        "legal_actions": legal,
        "events": evs,
        "event_seq": len(sess.events),
        "action_count": len(g.action_log),
        "trade_rates": [g.trade_rate(viewer, r) for r in range(5)] if viewer is not None else None,
        "max_trade_offers_per_turn": g.config.max_trade_offers_per_turn,
        "awaiting": ("none" if actor is None else sess.seats[actor].kind),
        # An agent's diagnostics rank its own legal moves, which reveals its hand
        # and dev cards: only shown when every hand is visible anyway.
        "last_decision": (sess.decisions.get(len(g.action_log) - 1)
                          if reveal or not sess.has_humans or g.is_over else None),
    })
    return v


def _check_kwargs(kwargs: Any, data_dir: Path) -> dict:
    if kwargs is None:
        return {}
    if not isinstance(kwargs, dict) or len(kwargs) > 16:
        raise HTTPException(400, "kwargs must be an object")
    out = {}
    for k, val in kwargs.items():
        rule = AGENT_KWARGS.get(k)
        if rule is None:
            raise HTTPException(400, f"kwarg {k!r} not allowed; allowed: {sorted(AGENT_KWARGS)}")
        if isinstance(rule, tuple):
            if type(val) is not int or not rule[0] <= val <= rule[1]:
                raise HTTPException(400, f"kwarg {k} must be an integer {rule[0]}..{rule[1]}")
        elif type(val) is not rule:
            raise HTTPException(400, f"kwarg {k} must be {rule.__name__}")
        if k == "checkpoint":
            val = resolve_checkpoint(data_dir, val)
        out[k] = val
    return out


_ENV = object()


def create_app(data_dir: str | os.PathLike | None = None, static_dir: str | os.PathLike | None = None,
               allowed_hosts: set[str] | None | object = _ENV, infoset_agents: bool | None = None,
               online: bool | None = None) -> FastAPI:
    """``infoset_agents`` (default: env CATAN_INFOSET_AGENTS=1) hands AI seats
    ``game.observe_for(seat)`` instead of the full game (catan.engine.infoset).
    ``online`` (default: env CATAN_ONLINE=1) enables rooms and token checks (docs/ONLINE.md)."""
    if infoset_agents is None:
        infoset_agents = os.environ.get("CATAN_INFOSET_AGENTS") == "1"
    if online is None:
        online = os.environ.get("CATAN_ONLINE") == "1"
    data = Path(data_dir or os.environ.get("CATAN_DATA_DIR") or ROOT).resolve()
    static = Path(static_dir or ROOT / "web" / "dist")
    app = FastAPI(title="Catan Lab server")
    sessions: dict[str, Session] = {}
    app.state.admin_key = admin_key(data) if online else None
    if online:  # added first, so RequestGuard (Host, Origin, 1 MB cap) runs before it
        app.add_middleware(OnlineGate, sessions=sessions, key=app.state.admin_key)
    app.add_middleware(RequestGuard, allowed_hosts=allowed_hosts_from_env() if allowed_hosts is _ENV else allowed_hosts,
                       max_body=MAX_BODY, path_max_body={"/api/analysis/import-image": IMAGE_BODY},
                       ext_origins=extension_origins_from_env(), ext_paths={"/api/analysis/import-image"})
    app.include_router(behavior_router(data))
    slock = threading.Lock()
    replay_cache: dict[tuple, tuple] = {}   # (source, ref, seed) -> (game, events, last events, hands per index)
    replay_lock = threading.Lock()

    def get(gid: str) -> Session:
        s = sessions.get(gid)
        if s is None:
            raise HTTPException(404, "game not found")
        s.touched = time.time()
        return s

    def parse_viewer(sess: Session, viewer: str | None) -> tuple[int | None, bool]:
        if viewer in (None, "", "spectator"):
            return None, False
        if viewer == "all":
            if sess.has_humans and not sess.game.is_over:
                raise HTTPException(403, "full reveal is only available for AI-only or finished games")
            return None, True
        try:
            v = int(viewer)
        except ValueError:
            raise HTTPException(400, "viewer must be a seat number, 'spectator' or 'all'")
        if not 0 <= v < sess.game.n:
            raise HTTPException(400, "viewer out of range")
        if sess.seats[v].kind != "human" and not sess.game.is_over:
            raise HTTPException(403, "can only view as a human seat")
        return v, False

    def room_viewer(request: Request, viewer: str | None) -> str | None:
        """In a room the token decides the view: a seat sees only itself, the host spectates."""
        who = request.scope.get("state", {}).get("catan_room_seat")
        return viewer if who is None else "spectator" if who == "host" else str(who)

    def default_checkpoint() -> str | None:
        """Checkpoint for an ``rl`` seat that names none. The agent's built-in
        default (models/...) is used when present; otherwise the most recently
        written runs/*/checkpoints/latest.pt, so "Default" works on a fresh clone."""
        try:
            from catan.rl.policy_agent import DEFAULT_CHECKPOINT
        except Exception:  # RL stack not installed
            return None
        if Path(DEFAULT_CHECKPOINT).is_file():
            return None
        c = sorted((data / "runs").glob("*/checkpoints/latest.pt"), key=lambda p: p.stat().st_mtime)
        return str(c[-1].resolve()) if c else None

    def new_session(players: list, seed, vp_to_win, max_turns, source: str, game: Game | None = None) -> Session:
        if not isinstance(players, list) or not 2 <= len(players) <= 4:
            raise HTTPException(400, "need 2 to 4 players")
        if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**31):
            raise HTTPException(400, "seed must be an integer in [0, 2^31)")
        if vp_to_win is not None and (not isinstance(vp_to_win, int) or not 3 <= vp_to_win <= 20):
            raise HTTPException(400, "vp_to_win must be 3..20")
        if max_turns is not None and (not isinstance(max_turns, int) or not 10 <= max_turns <= 5000):
            raise HTTPException(400, "max_turns must be 10..5000")
        seed = seed if seed is not None else random.randrange(2**31)
        known = available_agents()
        seats = []
        for i, p in enumerate(players):
            if not isinstance(p, dict) or p.get("type") not in ("human", "agent"):
                raise HTTPException(400, f"player {i}: type must be 'human' or 'agent'")
            if p["type"] == "human":
                seats.append(Seat("human", None, {}))
                continue
            name = p.get("agent") or "random"
            if name not in known:
                raise HTTPException(400, f"player {i}: unknown agent {name!r}; known: {known}")
            kw = _check_kwargs(p.get("kwargs"), data)
            if name == "rl" and "checkpoint" not in kw and (d := default_checkpoint()) is not None:
                kw["checkpoint"] = d
            try:
                obj = make_agent(name, **kw)
                obj.reset(i, seed * 1000 + i)
            except Exception as e:  # agent constructors are third-party code
                raise HTTPException(400, f"player {i}: could not create agent {name!r}: {e}")
            seats.append(Seat("agent", name, kw, obj))
        cfg = GameConfig(num_players=len(seats), vp_to_win=vp_to_win or 10, max_turns=max_turns)
        if game is not None and game.config.num_players != len(seats):
            raise HTTPException(400, "seat count does not match the game")
        sess = Session(uuid.uuid4().hex[:12], game or Game(config=cfg, seed=seed), seats, source)
        with slock:
            # Evict finished games first, then the least recently used, so a
            # burst of lab matches never drops a live human game.
            done = sorted((s for s in sessions.values() if s.game.is_over), key=lambda s: s.touched)
            for s in done[:max(0, len(done) - MAX_FINISHED + 1)]:
                sessions.pop(s.id, None)
            if len(sessions) >= MAX_SESSIONS:
                victim = min(sessions.values(), key=lambda s: (not s.game.is_over, s.touched))
                sessions.pop(victim.id, None)
            sessions[sess.id] = sess
        return sess

    def ai_step(sess: Session) -> dict:
        g = sess.game
        seat = sess.seats[g.actor]
        legal = g.legal_actions()
        info: dict = {}
        try:
            view = infoset_agents or not is_trusted(seat.agent)
            act, info = seat.agent_obj.act_with_info(g.observe_for(g.actor) if view else g, legal)
            if not g.is_legal(act):
                raise IllegalAction(f"agent returned illegal action {act}")
        except Exception as e:  # keep the table running; surface the failure
            act = sess.fallback_rng.choice(legal)
            info = {"agent_error": f"{type(e).__name__}: {e}", "fallback": "uniform random legal action"}
        ev = g.step(act)
        sess.record(ev)
        sess.decisions[len(g.action_log) - 1] = {"seat": act.player, "agent": seat.agent,
                                                  "action": act.to_json(), "info": _jsonable(info or {})}
        return {"action": act.to_json(), "events": ev}

    # ------------------------------------------------------------ meta
    @app.get("/api/health")
    def health():
        return {"ok": True, "online": online}

    @app.get("/api/topology")
    def topology():
        return topology_dict()

    @app.get("/api/rules")
    def rules():
        return {"resources": list(RESOURCES), "dev_types": list(DEV_TYPES),
                "costs": {k: list(v) for k, v in COST.items()}}

    @app.get("/api/agents")
    def agents():
        return {"agents": available_agents()}

    @app.get("/api/checkpoints")
    def checkpoints():
        runs = data / "runs"
        out = []
        if runs.is_dir():
            for p in sorted(runs.glob("*/checkpoints/*.pt")):
                out.append(str(p.relative_to(data)))
        out += sorted(str(p.relative_to(data)) for p in (data / "models").glob("**/*.pt"))
        d = default_checkpoint()
        return {"checkpoints": out,
                "default": None if d is None else next((c for c in out if (data / c).resolve() == Path(d)), d)}

    # ------------------------------------------------------------ games
    @app.post("/api/games")
    def create_game(body: dict = Body(...)):
        sess = new_session(body.get("players"), body.get("seed"), body.get("vp_to_win"),
                           body.get("max_turns"), "play")
        first_human = next((i for i, s in enumerate(sess.seats) if s.kind == "human"), None)
        return {"id": sess.id, "seed": sess.game.seed, "seats": [s.to_json() for s in sess.seats],
                "first_human": first_human}

    @app.get("/api/games")
    def list_games():
        return {"games": [{"id": s.id, "source": s.source, "created": s.created,
                           "seats": [x.to_json() for x in s.seats], "phase": s.game.phase,
                           "turn": s.game.turn, "winner": s.game.winner}
                          for s in sorted(sessions.values(), key=lambda s: -s.created)]}

    @app.get("/api/games/{gid}")
    def game_view(gid: str, request: Request, viewer: str | None = None, since: int | None = Query(None, ge=0)):
        sess = get(gid)
        with sess.lock:
            v, reveal = parse_viewer(sess, room_viewer(request, viewer))
            return _view(sess, v, since, reveal)

    @app.post("/api/games/{gid}/actions")
    def post_action(gid: str, request: Request, body: dict = Body(...)):
        sess = get(gid)
        seat = room_viewer(request, None)
        with sess.lock:
            g = sess.game
            try:
                act = Action.from_json(body.get("action"))
            except (ValueError, TypeError) as e:
                raise HTTPException(400, f"malformed action: {e}")
            if g.is_over:
                raise HTTPException(400, "game is over")
            if act.player != g.actor:
                raise HTTPException(400, f"not seat {act.player}'s move (waiting on seat {g.actor})")
            if sess.seats[act.player].kind != "human":
                raise HTTPException(400, "that seat is controlled by an AI")
            if seat is not None and seat != str(act.player):
                raise HTTPException(403, "your link controls another seat")
            try:
                ev = g.step(act)
            except IllegalAction as e:
                raise HTTPException(400, f"illegal action: {e}")
            except (TypeError, ValueError, IndexError, KeyError) as e:
                raise HTTPException(400, f"illegal action value: {e}")
            sess.record(ev)
            autosave(sess)
            since = body.get("since") if isinstance(body.get("since"), int) else None
            return {"events": [e for e in ev if _visible(e, act.player)], "view": _view(sess, act.player, since)}

    @app.post("/api/games/{gid}/advance")
    def advance(gid: str, request: Request, steps: int = Query(1, ge=1, le=5000), viewer: str | None = None,
                since: int | None = Query(None, ge=0)):
        """Play up to ``steps`` AI actions, stopping when a human must act."""
        sess = get(gid)
        with sess.lock:
            v, reveal = parse_viewer(sess, room_viewer(request, viewer))
            done = []
            for _ in range(steps):
                g = sess.game
                if g.is_over or sess.seats[g.actor].kind != "agent":
                    break
                st = ai_step(sess)
                # raw step events hold private details (an AI's drawn dev card, stolen cards)
                st["events"] = [e for e in st["events"] if reveal or _visible(e, v)]
                done.append(st)
            if done:
                autosave(sess)
            return {"steps": done, "view": _view(sess, v, since, reveal)}

    @app.get("/api/games/{gid}/replay")
    def game_replay(gid: str):
        sess = get(gid)
        with sess.lock:
            if sess.has_humans and not sess.game.is_over:
                raise HTTPException(403, "replay of a game with human seats is available after it ends")
            return sess.replay_json()

    @app.post("/api/games/{gid}/replay/save")
    def save_replay(gid: str):
        sess = get(gid)
        with sess.lock:
            if sess.has_humans and not sess.game.is_over:
                raise HTTPException(403, "finish the game first")
            out = data / "results" / "replays"
            out.mkdir(parents=True, exist_ok=True)
            name = f"{time.strftime('%Y%m%d-%H%M%S')}-{sess.id}.json"
            with open(out / name, "w") as f:
                json.dump(sess.replay_json(), f)
            return {"name": name}

    # ------------------------------------------------------------ saves
    # A save holds the full engine state (hands, deck order, RNG), so it is
    # never served to clients: they can only list saves and resume them.
    saves_dir = data / "results" / "saves"

    @app.post("/api/games/{gid}/save")
    def save_game(gid: str, body: dict | None = Body(None)):
        sess = get(gid)
        name = (body or {}).get("name") or f"{time.strftime('%Y%m%d-%H%M%S')}-{sess.id}"
        if not isinstance(name, str) or not SAVE_NAME.match(name):
            raise HTTPException(400, "save names use letters, digits, '-' and '_' (max 64)")
        p = _safe_child(saves_dir, name + ".json")
        if not p.exists() and saves_dir.is_dir() and sum(1 for _ in saves_dir.glob("*.json")) >= MAX_SAVES:
            raise HTTPException(507, f"{MAX_SAVES} saves already: delete some in results/saves/")
        with sess.lock:
            write_save(sess, p, name)
        return {"name": name}

    def write_save(sess: Session, p: Path, name: str):
        """Caller holds ``sess.lock``."""
        doc = {"format": "catan-save", "version": 1, "name": name, "saved": time.time(),
               "source": sess.source, "seats": [s.to_json() for s in sess.seats],
               "game": sess.game.to_dict(), "events": sess.events,
               "decisions": {str(k): v for k, v in sess.decisions.items()}}
        if sess.room is not None:
            doc["room"] = {"id": sess.id, **sess.room.to_json()}
        saves_dir.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(doc))
        tmp.replace(p)   # atomic: a crash mid-write never leaves a torn save

    def autosave(sess: Session):
        """Rooms are saved after every change so they survive a server restart. Caller holds the lock."""
        if sess.room is not None:
            write_save(sess, saves_dir / f"room-{sess.id}.json", f"room-{sess.id}")

    @app.get("/api/saves")
    def list_saves():
        out = []
        if saves_dir.is_dir():
            for p in sorted(saves_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
                d = _read_json(p)
                if not isinstance(d, dict) or d.get("format") != "catan-save":
                    continue
                g = d.get("game") or {}
                out.append({"name": p.stem, "saved": d.get("saved"), "seats": d.get("seats"),
                            "turn": g.get("turn"), "phase": g.get("phase"), "winner": g.get("winner")})
        return {"saves": out}

    @app.post("/api/saves/{name}/resume")
    def resume_save(name: str):
        if not SAVE_NAME.match(name):
            raise HTTPException(400, "invalid name")
        d = _read_json(_safe_child(saves_dir, name + ".json"))
        if not isinstance(d, dict) or d.get("format") != "catan-save":
            raise HTTPException(404, "save not found")
        sess = restore(d, name)
        first_human = next((i for i, s in enumerate(sess.seats) if s.kind == "human"), None)
        return {"id": sess.id, "seed": sess.game.seed, "seats": [s.to_json() for s in sess.seats],
                "first_human": first_human}

    def restore(d: dict, name: str) -> Session:
        try:
            g = Game.from_dict(d["game"])
            g.legal_actions()   # smoke-test the restored state
            players = [{"type": s["type"], "agent": s.get("agent"), "kwargs": s.get("kwargs") or {}}
                       for s in d["seats"]]
            if len(players) != g.n:
                raise ValueError("seat count does not match the game")
        except (KeyError, TypeError, ValueError, IndexError, AttributeError) as e:
            raise HTTPException(422, f"save is corrupt: {e}")
        sess = new_session(players, g.seed, g.config.vp_to_win, g.config.max_turns,
                           d.get("source") if d.get("source") in ("play", "lab") else "play", game=g)
        if isinstance(d.get("events"), list):
            sess.events = d["events"]
        if isinstance(d.get("decisions"), dict):
            sess.decisions = {int(k): v for k, v in d["decisions"].items() if str(k).isdigit()}
        for s, js in zip(sess.seats, d["seats"]):
            s.name = js.get("name") if isinstance(js.get("name"), str) else None
        sess.record([{"type": "game_resumed", "save": name}], step=False)
        return sess

    # ------------------------------------------------------------ replays
    def load_replay(name: str) -> dict:
        p = _safe_child(data / "results" / "replays", name)
        r = _read_json(p)
        if not isinstance(r, dict) or "actions" not in r or "seed" not in r:
            raise HTTPException(404, "replay not found")
        try:
            return load_replay_doc(r)  # files without a version are v1
        except ReplayFormatError as e:
            raise HTTPException(422, str(e))

    def replay_source(kind: str, ref: str) -> dict:
        if kind == "game":
            return game_replay(ref)
        if kind == "file":
            return load_replay(ref)
        raise HTTPException(400, "source must be 'game' or 'file'")

    @app.get("/api/replays")
    def list_replays():
        d = data / "results" / "replays"
        out = []
        if d.is_dir():
            for p in sorted(d.glob("*.json")):  # by name, so ties in time keep a stable order
                r = _read_json(p)
                if not isinstance(r, dict):
                    continue
                meta = r.get("meta") or {}
                out.append({"name": p.name, "agents": r.get("agents"), "seed": r.get("seed"),
                            "actions": len(r.get("actions") or []), "winner": meta.get("winner"),
                            "turns": meta.get("turns"), "truncated": meta.get("truncated"), "vp": meta.get("vp"),
                            "modified": p.stat().st_mtime, "created": meta.get("created")})
        # newest game first: mtime alone is arbitrary for copied or synced files (DS8-10)
        out.sort(key=lambda x: x["created"] if isinstance(x["created"], (int, float)) else x["modified"], reverse=True)
        return {"replays": out}

    @app.get("/api/replays/{name}")
    def get_replay(name: str):
        return load_replay(name)

    @app.get("/api/replay-state")
    def replay_state(source: str, ref: str, index: int = Query(0, ge=0), since: int | None = Query(None, ge=0)):
        """Board after the first ``index`` actions (full reveal) plus the
        events of action ``index-1``. Re-simulated: the engine is deterministic.
        With ``since`` < index, ``prev_players``/``span_events`` cover actions
        [since, index) instead (turn summaries)."""
        r = replay_source(source, ref)
        actions = r.get("actions") or []
        index = min(index, len(actions))
        key = (source, ref, r.get("seed"))
        with replay_lock:
            cached = replay_cache.pop(key, None)
            # Step forward from the cached position when possible (playback is
            # sequential); otherwise re-simulate from the start.
            if cached is not None:
                log = cached[0].action_log
                try:
                    stale = len(log) > index or (log and log[-1] != Action.from_json(actions[len(log) - 1]))
                except (ValueError, TypeError):
                    stale = True
                if stale:
                    cached = None
            try:
                if cached:
                    g, all_events, evs, snaps = cached
                else:
                    cfg = GameConfig(**r["config"]) if isinstance(r.get("config"), dict) else GameConfig()
                    g, all_events, evs = Game(config=cfg, seed=r["seed"]), [], []
                    snaps = [_hands(g)]
                for i in range(len(g.action_log), index):
                    evs = g.step(Action.from_json(actions[i]))
                    snaps.append(_hands(g))   # snaps[k]: hands/dev cards after k actions
                    all_events.extend({"seq": len(all_events), "i": i, **e} for e in evs)
            except (IllegalAction, ValueError, TypeError, KeyError) as e:
                raise HTTPException(422, f"replay does not re-simulate: {e}")
            replay_cache[key] = (g, all_events, evs, snaps)
            while len(replay_cache) > 8:
                replay_cache.pop(next(iter(replay_cache)))
            g = g.clone()
            all_events = all_events[:]
        agents_ = r.get("agents") or []
        names = [agents_[i] if i < len(agents_) else "?" for i in range(g.n)]
        seats = [Seat("human" if a == "human" else "agent", a, {}) for a in names]
        sess = Session("replay", g, seats, "replay")
        sess.events = all_events
        decisions = (r.get("meta") or {}).get("decisions") or {}
        sess.decisions = {int(k): v for k, v in decisions.items() if str(k).isdigit()}
        view = _view(sess, None, None, reveal=True)
        view["last_events"] = evs
        lo = since if since is not None and since < index else index - 1
        view["prev_players"] = snaps[lo] if index else None
        view["span_events"] = [e for e in all_events if lo <= e["i"] < index]
        view["turns"] = turn_segments(actions, g.n)
        view["total_actions"] = len(actions)
        view["index"] = index
        return view

    # ------------------------------------------------------------ lab
    @app.post("/api/lab/matches")
    def lab_match(body: dict = Body(...)):
        agents_ = body.get("agents")
        if not isinstance(agents_, list):
            raise HTTPException(400, "agents must be a list of agent names or {agent, kwargs}")
        players = [{"type": "agent", "agent": a} if isinstance(a, str) else {"type": "agent", **a}
                   for a in agents_]
        for p in players:
            p["type"] = "agent"
        sess = new_session(players, body.get("seed"), body.get("vp_to_win"),
                           body.get("max_turns") or 500, "lab")  # never unbounded: run_to_end holds the lock
        result = None
        if body.get("run_to_end"):
            with sess.lock:
                while not sess.game.is_over:
                    ai_step(sess)
                g = sess.game
                result = {"winner": g.winner, "truncated": g.truncated, "turns": g.turn,
                          "vp": [g.victory_points(p) for p in range(g.n)]}
        return {"id": sess.id, "seed": sess.game.seed, "seats": [s.to_json() for s in sess.seats],
                "result": result}

    # Runs: a light list (polled) and a per-run detail with a byte cursor into
    # metrics.jsonl, so polling a live run only transfers the new rows. Both carry
    # ETags derived from file sizes/mtimes; with no-cache the browser revalidates
    # (If-None-Match -> 304) instead of re-downloading unchanged runs.
    runs_dir = data / "runs"
    RUN_FILES = ("meta.json", "config.yaml", "metrics.jsonl", "eval.jsonl", "bc_summary.json")

    def _run_paths(p: Path):
        ck = p / "checkpoints"
        return [p / f for f in RUN_FILES] + ([ck] if ck.is_dir() else [])

    def _etag_response(payload: dict, etag: str, if_none_match: str | None):
        headers = {"ETag": etag, "Cache-Control": "no-cache"}
        if if_none_match and etag in [t.strip() for t in if_none_match.split(",")]:
            return Response(status_code=304, headers=headers)
        return JSONResponse(payload, headers=headers)

    def _bc_summary(p: Path):
        b = _read_json(p / "bc_summary.json")
        return b if isinstance(b, dict) else None

    def _eval_brief(row):
        if not isinstance(row, dict):
            return None
        per = row.get("per_opponent") or {}
        return {"step": row.get("step"), "checkpoint": row.get("checkpoint"),
                "per_opponent": {o: {k: v.get(k) for k in ("win_rate", "ci95", "games")}
                                 for o, v in per.items() if isinstance(v, dict)}}

    def _run_ckpts(p: Path) -> list[str]:
        """checkpoints/*.pt plus *.pt in the run root (BC runs write bc.pt there)."""
        return sorted({c.name for c in [*(p / "checkpoints").glob("*.pt"), *p.glob("*.pt")]})

    def run_summary(p: Path) -> dict:
        meta = _read_json(p / "meta.json")
        metrics_total = _count_lines(p / "metrics.jsonl")
        bc = _bc_summary(p)
        try:
            updated = max(q.stat().st_mtime for q in _run_paths(p) if q.exists())
        except ValueError:
            updated = None
        hist = (bc or {}).get("history") or []
        return {"id": p.name, "meta": meta if isinstance(meta, dict) else None,
                "kind": "bc" if bc is not None and not metrics_total else "rl",
                "updated": updated, "metrics_total": metrics_total,
                "eval_total": _count_lines(p / "eval.jsonl"),
                "last_eval": _eval_brief(_last_json_line(p / "eval.jsonl")),
                "checkpoints": len(_run_ckpts(p)),
                "bc": None if bc is None else {"teacher": bc.get("teacher"), "arch": bc.get("arch"),
                                               "decisions": bc.get("decisions"), "params": bc.get("params"),
                                               "epochs": len(hist),
                                               "heldout_acc": (hist[-1] or {}).get("heldout_acc") if hist else None}}

    @app.get("/api/runs")
    def runs(if_none_match: str | None = Header(None)):
        """Light list for polling: no metric rows (see /api/runs/{id})."""
        dirs = sorted(x for x in runs_dir.iterdir() if x.is_dir()) if runs_dir.is_dir() else []
        etag = '"' + _stat_key([q for d in dirs for q in [d, *_run_paths(d)]]) + '"'
        if if_none_match and etag in [t.strip() for t in if_none_match.split(",")]:
            return Response(status_code=304, headers={"ETag": etag, "Cache-Control": "no-cache"})
        return _etag_response({"runs": [run_summary(p) for p in dirs]}, etag, None)

    @app.get("/api/runs/{run_id}")
    def run_detail(run_id: str, since: int = Query(0, ge=0), if_none_match: str | None = Header(None)):
        """One run. ``since`` is the ``cursor`` of a previous response (a byte offset
        into metrics.jsonl): only rows appended after it are returned, raw. ``reset``
        true means the log was rewritten and ``metrics`` is the whole (downsampled) log."""
        p = _safe_child(runs_dir, run_id)
        if not p.is_dir():
            raise HTTPException(404, "run not found")
        etag = f'"{_stat_key(_run_paths(p))}-{since}"'
        if if_none_match and etag in [t.strip() for t in if_none_match.split(",")]:
            return Response(status_code=304, headers={"ETag": etag, "Cache-Control": "no-cache"})
        rows, cursor, reset = _read_jsonl_from(p / "metrics.jsonl", since)
        full = since == 0 or reset
        payload = {**run_summary(p), "metrics": _downsample(rows) if full else rows,
                   "downsampled": full and len(rows) > MAX_METRIC_POINTS,
                   "since": since, "cursor": cursor, "reset": bool(reset and since),
                   "eval": _downsample(_read_jsonl(p / "eval.jsonl")),
                   "checkpoints": _run_ckpts(p),
                   "bc_summary": _bc_summary(p)}
        if full:  # config.yaml is static: incremental polls keep the client's copy
            try:
                payload["config"] = (p / "config.yaml").read_text()[:20000]
            except OSError:
                payload["config"] = None
        return _etag_response(payload, etag, None)

    @app.get("/api/tournaments")
    def tournaments():
        d = data / "results" / "tournaments"
        out = []
        if d.is_dir():
            for p in sorted(d.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
                t = _read_json(p)
                # checkpoint ladders have their own schema and charts (Leaderboard tab, /api/ladders)
                if isinstance(t, dict) and t.get("kind") != "ladder":
                    t.setdefault("id", p.stem)
                    t["file"] = p.name
                    out.append(t)
        return {"tournaments": out}

    if online:
        from catan.server.online import mount_rooms
        mount_rooms(app, sessions, saves_dir, new_session, restore, autosave)

    from catan.analysis.engine import shutdown_pool
    app.router.add_event_handler("shutdown", shutdown_pool)  # stop rollout workers with the server
    from .analysis import make_router
    app.include_router(make_router(data, sessions))
    app.include_router(make_zoo_router(data))

    @app.exception_handler(404)
    async def not_found(request, exc):
        detail = getattr(exc, "detail", "not found")
        if request.url.path.startswith("/api/") or not (static / "index.html").is_file():
            return JSONResponse({"detail": detail}, status_code=404)
        return FileResponse(static / "index.html")

    if static.is_dir():
        app.mount("/", StaticFiles(directory=static, html=True), name="web")
    else:
        @app.get("/")
        def no_ui():
            return JSONResponse({"detail": "web UI not built: run `cd web && npm install && npm run build`"})

    return app


app = create_app()
