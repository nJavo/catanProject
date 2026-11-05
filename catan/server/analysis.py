"""HTTP layer for the analysis engine: ``/api/analysis``.

Hidden-information rules: positions are always built with
``position_from_game(game, viewer)``; live games with human seats can only be
analysed from a human seat's view (as for ``/api/games/{id}?viewer=``).
Checkpoints must be files under ``runs/`` or ``models/`` of the data dir.
"""
from __future__ import annotations

import functools
import json
import re
import threading
from pathlib import Path

from email.parser import BytesParser
from email.policy import HTTP

from fastapi import APIRouter, Body, HTTPException, Request
from fastapi.concurrency import run_in_threadpool

from catan.agents.registry import available_agents
from catan.bots.belief import HandPosterior
from catan.analysis import (DEPTHS, analyze, analyze_game, continue_position, derived, legal_moves,
                            parse_policy, position_from_game, validate_position_detailed)
from catan.analysis.engine import Cancelled
from catan.server.checkpoints import resolve_checkpoint
from catan.engine import Action, Game, GameConfig, IllegalAction
from catan.engine.replay import ReplayFormatError, load_replay

SAFE_NAME = re.compile(r"^[A-Za-z0-9_.\-]{1,128}$")
REQUEST_ID = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")
# Agent options the API accepts, with bounds (anything else could be a DoS knob).
MAX_IMAGE_BYTES = 8_000_000                 # screenshot upload cap (the file part)
IMAGE_BODY = MAX_IMAGE_BYTES + 65_536       # whole multipart body; app.py raises the guard's cap to this
IMAGE_TYPES = {"image/png", "image/jpeg", "image/webp"}
MAX_TIME_MS = 120_000  # server-side cap on rollout time per /evaluate (partial results, timed_out=true)
ALLOWED_KW = {"checkpoint": str, "deterministic": bool, "rollouts": (int, 1, 32), "candidates": (int, 1, 12)}
# Search agents run a search on every move. As the rollout policy that is a search per move of
# every simulated game (minutes per batch), so they may rank candidates but not roll out.
SLOW_ROLLOUT = {"search", "search_trader", "jointb_gated"}


MY_COLOR = re.compile(r"^[a-z]{1,20}$")


def apply_overrides(out: dict, my_color: str | None, my_hand: list[int] | None) -> None:
    """The user's "analyze as" colour and hand on top of a screenshot reading, in place.
    Opponents' hands stay hidden: only the viewer's seat gets an exact hand."""
    spec, low, notes = out["position"], out["low_confidence"]["fields"], []
    if my_color:
        seat = (out.get("colors") or {}).get(my_color)
        if seat is None:
            notes.append(f"Colour {my_color} was not seen or this reader doesn't name colours: set your seat in Analysis.")
        elif spec["phase"] != "MAIN":
            if seat != spec["viewer"]:
                notes.append(f"Setup: the next placement looks like {next(c for c, s in out['colors'].items() if s == spec['viewer'])}'s, not yours.")
        elif seat != spec["viewer"]:
            pls, old = spec["players"], spec["viewer"]
            o = pls[old]
            pls[old] = {"resource_count": sum(o["hand"]), "dev_count": sum(o["dev_cards"]), "knights_played": o["knights_played"]}
            pls[seat] = {"hand": [0] * 5, "dev_cards": [0] * 5, "knights_played": pls[seat]["knights_played"]}
            spec["viewer"] = seat
            low[:] = sorted(set(low) | {"hands", "dev_cards"})
        if spec["phase"] == "MAIN" and seat is not None and "current" in low:
            spec["current"] = seat  # whose turn was not read: assume the user asks on their own turn
    if my_hand:
        if spec["phase"] == "MAIN":
            spec["players"][spec["viewer"]]["hand"] = list(my_hand)
            low[:] = [f for f in low if f != "hands"]
        else:
            notes.append("Setup: your hand is ignored until the main game.")
    out["errors"] = validate_position_detailed(spec)
    out["override_note"] = " ".join(notes) or None


def make_router(data: Path, sessions: dict) -> APIRouter:
    """``sessions`` is the server's live session dict; it is only read."""
    data = Path(data).resolve()
    r = APIRouter(prefix="/api/analysis")
    # Rollout analyses run one at a time (engine lock). Without a cap, queued
    # requests each hold a server worker thread until the pool is exhausted and
    # the game endpoints stall too. One running + one waiting; the rest get 429.
    # Quick depth (no rollouts) has its own engine lock and its own cap, so it
    # never waits behind a deep analysis.
    heavy_slots = threading.BoundedSemaphore(2)
    quick_slots = threading.BoundedSemaphore(4)
    running: dict[str, threading.Event] = {}   # request_id -> cancel event
    # A cancel can arrive before its request (the client aborts right after
    # sending). Remember recent unknown ids so that request starts cancelled.
    early: dict[str, None] = {}
    running_lock = threading.Lock()
    ctx = threading.local()                     # the current request's cancel event

    def busy_guard(default_depth):
        def deco(fn):
            @functools.wraps(fn)  # keeps the signature FastAPI reads the body from
            def wrapped(*a, **kw):
                body = kw.get("body") if isinstance(kw.get("body"), dict) else {}
                rid = body.get("request_id")
                if rid is not None and (not isinstance(rid, str) or not REQUEST_ID.match(rid)):
                    raise HTTPException(400, "request_id must match [A-Za-z0-9_-]{1,64}")
                slots = quick_slots if (body.get("depth") or default_depth) == "quick" else heavy_slots
                if not slots.acquire(blocking=False):
                    raise HTTPException(429, "analysis is busy; retry when the current one finishes")
                ev = threading.Event()
                try:
                    if rid is not None:
                        with running_lock:
                            if rid in running:
                                raise HTTPException(409, "request_id already running")
                            running[rid] = ev
                            if early.pop(rid, 0) is None:
                                ev.set()
                    ctx.cancel = ev
                    try:
                        return fn(*a, **kw)
                    except Cancelled:
                        raise HTTPException(409, "analysis cancelled")
                finally:
                    ctx.cancel = None
                    if rid is not None:
                        with running_lock:
                            if running.get(rid) is ev:
                                del running[rid]
                    slots.release()
            return wrapped
        return deco

    @r.post("/cancel")
    def cancel(body: dict = Body(...)):
        """Stop a running /evaluate, /game or /continue by its ``request_id``.
        Its slot and its rollout workers are free within about CANCEL_POLL_S."""
        rid = body.get("request_id")
        if not isinstance(rid, str) or not REQUEST_ID.match(rid):
            raise HTTPException(400, "request_id must match [A-Za-z0-9_-]{1,64}")
        with running_lock:
            ev = running.get(rid)
            if ev is None:
                early[rid] = None
                while len(early) > 256:
                    early.pop(next(iter(early)))
        if ev is not None:
            ev.set()
        return {"cancelled": ev is not None}

    def policy_arg(spec, default):
        spec = default if spec is None else spec
        try:
            name, kw = parse_policy(spec)
        except ValueError as e:
            raise HTTPException(400, str(e))
        if name not in available_agents():
            raise HTTPException(400, f"unknown policy {name!r}; known: {available_agents()}")
        out = []
        for k, v in kw.items():
            rule = ALLOWED_KW.get(k)
            if rule is None:
                raise HTTPException(400, f"policy option {k!r} not allowed; allowed: {sorted(ALLOWED_KW)}")
            if k == "checkpoint":
                v = resolve_checkpoint(data, str(v))
            elif isinstance(rule, tuple):
                if type(v) is not int or not rule[1] <= v <= rule[2]:
                    raise HTTPException(400, f"policy option {k} must be an integer {rule[1]}..{rule[2]}")
            elif type(v) is not rule:
                raise HTTPException(400, f"policy option {k} must be {rule.__name__}")
            out.append(f"{k}={v}")
        return name + (":" + ",".join(out) if out else "")

    def rollout_arg(spec):
        p = policy_arg(spec, "basic")
        name = p.split(":", 1)[0]
        if name == "champion":
            from catan.zoo.champion import current
            e = current(data)
            name = (e or {}).get("spec", "").split(":", 1)[0]
        if name in SLOW_ROLLOUT:
            raise HTTPException(400, f"{p.split(':', 1)[0]!r} ({name}) searches on every move, so it is too slow to "
                                     f"play rollouts; pick a fast rollout policy such as basic or intermediate")
        return p

    def int_arg(body, k, default, lo, hi):
        v = body.get(k, default)
        if v is None:
            return default
        if type(v) is not int or not lo <= v <= hi:
            raise HTTPException(400, f"{k} must be an integer {lo}..{hi}")
        return v

    def depth_arg(body, default):
        d = body.get("depth") or default
        if d not in DEPTHS:
            raise HTTPException(400, f"depth must be one of {list(DEPTHS)}")
        return d

    def load_source(src, viewer):
        """(replay-like dict {seed, config, actions}, n) for a source; enforces view rules."""
        if not isinstance(src, dict):
            raise HTTPException(400, "source must be {kind, ref, index?}")
        kind, ref = src.get("kind"), src.get("ref")
        if not isinstance(ref, str):
            raise HTTPException(400, "source.ref must be a string")
        if kind == "game":
            sess = sessions.get(ref)
            if sess is None:
                raise HTTPException(404, "game not found")
            with sess.lock:
                g = sess.game
                rep = {"seed": g.seed, "config": g.config.to_dict(), "actions": [a.to_json() for a in g.action_log]}
                n = g.n
                if type(viewer) is not int or not 0 <= viewer < n:
                    raise HTTPException(400, "viewer must be a seat number")
                if sess.has_humans and not g.is_over and sess.seats[viewer].kind != "human":
                    raise HTTPException(403, "live games with human seats can only be analysed from a human seat's view")
            return rep
        if kind in ("replay", "file"):
            base = (data / "results" / "replays").resolve()
            if not SAFE_NAME.match(ref) or ref in (".", ".."):
                raise HTTPException(400, "invalid replay name")
            p = (base / ref).resolve()
            if p.parent != base:
                raise HTTPException(400, "invalid replay name")
            try:
                rep = json.loads(p.read_text())
            except (OSError, json.JSONDecodeError):
                raise HTTPException(404, "replay not found")
            if not isinstance(rep, dict) or "actions" not in rep or "seed" not in rep:
                raise HTTPException(404, "replay not found")
            try:
                rep = load_replay(rep)  # files without a version are v1
            except ReplayFormatError as e:
                raise HTTPException(422, str(e))
            n = (rep.get("config") or {}).get("num_players", 4)
            if type(viewer) is not int or not 0 <= viewer < n:
                raise HTTPException(400, "viewer must be a seat number")
            return rep
        raise HTTPException(400, "source.kind must be 'game' or 'replay'")

    def game_at(rep, index):
        actions = rep["actions"]
        index = len(actions) if index is None else index
        if type(index) is not int or not 0 <= index <= len(actions):
            raise HTTPException(400, f"source.index must be 0..{len(actions)}")
        try:
            cfg = GameConfig(**rep["config"]) if isinstance(rep.get("config"), dict) else GameConfig()
            return Game.replay(rep["seed"], cfg, actions[:index])
        except (IllegalAction, ValueError, TypeError, KeyError) as e:
            raise HTTPException(422, f"source does not re-simulate: {e}")

    def spec_arg(body):
        spec = body.get("position")
        if not isinstance(spec, dict):
            raise HTTPException(400, "position must be an object")
        return spec

    def run(fn, *a, **kw):
        try:
            return fn(*a, **kw)
        except (ValueError, KeyError, TypeError, IllegalAction) as e:
            raise HTTPException(400, str(e))

    @r.post("/validate")
    def validate(body: dict = Body(...)):
        spec = spec_arg(body)
        errs = validate_position_detailed(spec)
        return {"ok": not errs, "errors": errs, "derived": None if errs else derived(spec)}

    @r.post("/legal")
    def legal(body: dict = Body(...)):
        return {"legal": run(legal_moves, spec_arg(body))}

    @r.post("/evaluate")
    @busy_guard("standard")
    def evaluate(body: dict = Body(...)):
        belief = None
        if body.get("belief") not in (None, False, True):
            raise HTTPException(400, "belief must be a boolean")
        if body.get("source") is not None:
            src = body["source"]
            viewer = body.get("viewer")
            rep = load_source(src, viewer)
            g = game_at(rep, src.get("index"))
            if g.is_over:
                raise HTTPException(400, "the game is over at this index")
            spec = position_from_game(g, viewer)
            if body.get("belief"):
                # only the viewer's visible events from the replayed history are read
                cfg = GameConfig(**rep["config"]) if isinstance(rep.get("config"), dict) else GameConfig()
                belief, _ = HandPosterior.from_history(rep["seed"], cfg, [Action.from_json(a) for a in rep["actions"]],
                                                       viewer, src.get("index"))
                if belief.lost:
                    belief = None
        else:
            if body.get("belief"):
                raise HTTPException(400, "belief needs a source game or replay (a hand-built position has no event log)")
            spec = spec_arg(body)
        return run(analyze, spec, policy_arg(body.get("policy"), "intermediate"),
                   rollout_arg(body.get("rollout_policy")), depth_arg(body, "standard"),
                   int_arg(body, "top_k", None, 1, 20), int_arg(body, "seed", 0, 0, 2**31 - 1),
                   time_limit_s=int_arg(body, "time_limit_ms", MAX_TIME_MS, 100, MAX_TIME_MS) / 1000,
                   belief=belief, cancel=ctx.cancel)

    @r.post("/game")
    @busy_guard("quick")
    def game(body: dict = Body(...)):
        viewer = body.get("viewer")
        if body.get("depth") == "deep":
            # analyze_game has no time limit: deep (8x25 rollouts, 4 workers) per decision
            # over a whole game runs for hours. Use deep on single positions (/evaluate).
            raise HTTPException(400, "depth 'deep' is not available for whole-game review; use quick or standard")
        rep = load_source(body.get("source"), viewer)
        return run(analyze_game, rep, viewer, policy_arg(body.get("policy"), "intermediate"),
                   depth_arg(body, "quick"), rollout_arg(body.get("rollout_policy")),
                   int_arg(body, "seed", 0, 0, 2**31 - 1), belief=body.get("belief") is True, cancel=ctx.cancel)

    @r.post("/continue")
    @busy_guard("continue")
    def cont(body: dict = Body(...)):
        try:
            action = Action.from_json(body.get("action"))
        except (ValueError, TypeError) as e:
            raise HTTPException(400, f"malformed action: {e}")
        until = body.get("until")
        if until not in (None, "turn_end", "game_end"):
            raise HTTPException(400, "until must be 'turn_end' or 'game_end'")
        return run(continue_position, spec_arg(body), action, policy_arg(body.get("policy"), "basic"),
                   int_arg(body, "steps", None, 0, 2000), until, int_arg(body, "seed", 0, 0, 2**31 - 1),
                   cancel=ctx.cancel)

    image_slot = threading.BoundedSemaphore(1)  # one screenshot at a time (about 1 s of CPU each)

    @r.post("/import-image")
    async def import_image(request: Request):
        """Multipart ``image`` (PNG/JPEG/WebP, <= MAX_IMAGE_BYTES) plus optional ``profile``,
        ``num_players``, ``backend``, ``my_color`` and ``my_hand`` fields -> detected PositionSpec, per-field confidence and the champion's quick
        suggestion (docs/VISION_IMPORT.md). Reads only the uploaded file."""
        ctype = request.headers.get("content-type", "")
        if not ctype.lower().startswith("multipart/form-data"):
            raise HTTPException(415, "send multipart/form-data with an 'image' file field")
        body = await request.body()
        if len(body) > IMAGE_BODY:
            raise HTTPException(413, f"request body over {IMAGE_BODY} bytes")
        # stdlib MIME parser: no python-multipart dependency for one upload field
        msg = BytesParser(policy=HTTP).parsebytes(b"Content-Type: " + ctype.encode("latin-1") + b"\r\n\r\n" + body)
        if not msg.is_multipart():
            raise HTTPException(400, "malformed multipart body")
        image, fields = None, {}
        for part in msg.iter_parts():
            name = part.get_param("name", header="content-disposition")
            if name == "image":
                if image is not None:
                    raise HTTPException(400, "send exactly one image")
                if part.get_content_type() not in IMAGE_TYPES:
                    raise HTTPException(415, f"image must be one of {sorted(IMAGE_TYPES)}")
                image = part.get_payload(decode=True) or b""
            elif name in ("profile", "num_players", "backend", "my_color", "my_hand"):
                fields[name] = (part.get_payload(decode=True) or b"").decode("utf-8", "replace").strip()
        if not image:
            raise HTTPException(400, "missing 'image' file field")
        if len(image) > MAX_IMAGE_BYTES:
            raise HTTPException(413, f"image over {MAX_IMAGE_BYTES} bytes")
        profile = fields.get("profile") or None
        if profile is not None and not SAFE_NAME.match(profile):
            raise HTTPException(400, "invalid profile name")
        n = fields.get("num_players") or "4"
        if n not in ("2", "3", "4"):
            raise HTTPException(400, "num_players must be 2, 3 or 4")
        backend = fields.get("backend") or "auto"
        if backend not in ("auto", "llm", "color"):
            raise HTTPException(400, "backend must be auto, llm or color")
        my_color = (fields.get("my_color") or "").lower() or None
        if my_color is not None and not MY_COLOR.match(my_color):
            raise HTTPException(400, "my_color must be a colour name (letters, 1-20)")
        my_hand = fields.get("my_hand") or None
        if my_hand is not None:
            try:
                my_hand = [int(x) for x in my_hand.split(",")]
            except ValueError:
                raise HTTPException(400, "my_hand must be 5 comma-separated counts (wood,brick,sheep,wheat,ore)")
            if len(my_hand) != 5 or not all(0 <= x <= 19 for x in my_hand):
                raise HTTPException(400, "my_hand must be 5 comma-separated counts 0..19 (wood,brick,sheep,wheat,ore)")
        if not image_slot.acquire(blocking=False):
            raise HTTPException(429, "an import is already running; retry when it finishes")
        try:
            return await run_in_threadpool(import_work, image, profile, int(n), backend, my_color, my_hand)
        finally:
            image_slot.release()

    def import_work(image, profile, n, backend, my_color=None, my_hand=None):
        from catan.vision import VisionUnavailable, import_image as read
        from catan.vision import llm
        note = None
        if backend != "color" and not llm.config()[0]:
            note = "No GEMINI_API_KEY in .env: used the colour backend instead of Gemini."
            backend = "color"
        try:
            if backend == "color":
                out = read(image, profile, extra_profiles=data / "data" / "screenshots", num_players=n)
            else:
                out = llm.import_image_llm(image, num_players=n)
        except VisionUnavailable as e:
            raise HTTPException(501, str(e))
        except llm.LLMError as e:
            raise HTTPException(502, str(e))
        except ValueError as e:
            raise HTTPException(400, str(e))
        out["backend"] = "color" if backend == "color" else "llm"
        if note:
            out["backend_note"] = note
        if my_color or my_hand:
            apply_overrides(out, my_color, my_hand)
        out["suggestion"] = None
        if out["errors"]:
            out["suggestion_note"] = "Fix the flagged problems in the editor, then ask for the best move."
            return out
        policy = "champion" if "champion" in available_agents() else "intermediate"
        if not quick_slots.acquire(blocking=False):
            out["suggestion_note"] = "Analysis is busy; ask for the best move from the editor."
            return out
        try:
            res = analyze(out["position"], policy, "basic", "quick", 3, 0)
            out["suggestion"] = {"policy": policy, "depth": "quick", "candidates": [
                {k: c.get(k) for k in ("action", "label", "policy_prob", "value_estimate", "search_score", "rank")}
                for c in res["candidates"]]}
            if out["position"]["phase"] == "MAIN" and "hands" in out["low_confidence"]["fields"]:
                out["suggestion_note"] = "Hands are not read from the screenshot: enter yours in the editor for a useful suggestion."
        except (ValueError, KeyError, TypeError, IllegalAction) as e:
            out["suggestion_note"] = f"No suggestion: {e}"
        finally:
            quick_slots.release()
        return out

    return r
