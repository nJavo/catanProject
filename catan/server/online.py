"""``--online`` mode: private rooms reached through secret seat links (docs/ONLINE.md).

Every ``/api`` request is authorised here, before any route runs:

- ``/api/health``, ``/api/topology``, ``/api/rules``: public (no game data).
- ``/api/games/{id}[/...]`` and ``/api/rooms/{id}[/...]`` of a room: need that
  room's seat token or host token in ``X-Room-Token``. A seat token may only
  view, act, advance bots, read the finished replay and set its name; the
  route then sees ``scope["state"]["catan_room_seat"]`` (seat number or "host").
- Anything else (room creation, lab, analysis, saves, plain games): needs the
  server admin key in ``X-Catan-Admin``. Client IP or Host are not trusted,
  because a tunnel delivers every remote request from 127.0.0.1.
"""
from __future__ import annotations

import hmac
import json
import os
import re
import secrets
import time
from pathlib import Path

from fastapi import Body, HTTPException, Request

PUBLIC = {"/api/health", "/api/topology", "/api/rules"}
ROOM_PATH = re.compile(r"^/api/(games|rooms)/([0-9a-f]{12})(/[a-z/]*)?$")
SEAT_PATHS = {"", "/actions", "/advance", "/replay", "/name"}
MAX_ROOM_BODY = 8192          # an action is ~100 bytes
RATE_LIMIT, RATE_WINDOW = 60, 5.0   # requests per token per window: polling + Fast AI is ~6/s
CONNECTED_S = 8.0             # a seat polls every 1.5 s


def new_token() -> str:
    return secrets.token_urlsafe(24)  # 192 bits


def admin_key(data: Path) -> str:
    """``CATAN_ADMIN_KEY``, else a key kept in ``results/online_admin_key`` so it survives restarts."""
    if os.environ.get("CATAN_ADMIN_KEY"):
        return os.environ["CATAN_ADMIN_KEY"]
    p = data / "results" / "online_admin_key"
    try:
        k = p.read_text().strip()
        if len(k) >= 22:
            return k
    except OSError:
        pass
    k = new_token()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(k)
    p.chmod(0o600)
    return k


def _eq(a: str | None, b: str | None) -> bool:
    return bool(a and b) and hmac.compare_digest(a.encode(), b.encode())


class Room:
    """Tokens of one online game. ``tokens[i]`` is None for bot seats."""

    def __init__(self, host: str, tokens: list[str | None]):
        self.host, self.tokens = host, tokens
        self.seen: dict = {}
        self.hits: dict[str, list] = {}   # token -> [window start, count]

    def who(self, token: str | None):
        if _eq(token, self.host):
            return "host"
        return next((i for i, t in enumerate(self.tokens) if _eq(token, t)), None)

    def allow(self, token: str, now: float) -> bool:
        w = self.hits.setdefault(token, [now, 0])
        if now - w[0] >= RATE_WINDOW:
            w[0], w[1] = now, 0
        w[1] += 1
        return w[1] <= RATE_LIMIT

    def connected(self, who, now: float) -> bool:
        return now - self.seen.get(who, 0) < CONNECTED_S

    def to_json(self) -> dict:
        return {"host": self.host, "tokens": self.tokens}


class OnlineGate:
    """ASGI middleware; ``sessions`` is the app's live dict, rooms hang off ``session.room``."""

    def __init__(self, app, sessions: dict, key: str):
        self.app, self.sessions, self.key = app, sessions, key

    async def __call__(self, scope, receive, send):
        path = scope.get("path", "")
        if scope["type"] != "http" or not path.startswith("/api/") or path in PUBLIC:
            return await self.app(scope, receive, send)
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope["headers"]}
        m = ROOM_PATH.match(path)
        room = getattr(self.sessions.get(m.group(2)), "room", None) if m else None
        if room is None:
            if not _eq(headers.get("x-catan-admin"), self.key):
                return await _reply(send, 403, "online mode: this route needs the admin key")
            return await self.app(scope, receive, send)
        token = headers.get("x-room-token")
        who = room.who(token)
        if who is None:
            return await _reply(send, 403, "missing or wrong room token")
        if who != "host" and (m.group(3) or "") not in SEAT_PATHS:
            return await _reply(send, 403, "a seat link can't do that")
        now = time.time()
        if not room.allow(token, now):
            return await _reply(send, 429, "too many requests: slow down")
        # Buffer the body (RequestGuard already capped it at 1 MB) and apply the tighter room cap.
        body, more = b"", True
        while more:
            msg = await receive()
            if msg["type"] == "http.disconnect":
                return
            body += msg.get("body", b"")
            more = msg.get("more_body", False)
            if len(body) > MAX_ROOM_BODY:
                return await _reply(send, 413, f"room requests are capped at {MAX_ROOM_BODY} bytes")
        room.seen[who] = now
        scope.setdefault("state", {})["catan_room_seat"] = who
        sent = False

        async def replay():
            nonlocal sent
            if not sent:
                sent = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()
        return await self.app(scope, replay, send)


def mount_rooms(app, sessions: dict, saves_dir: Path, new_session, restore, autosave):
    """Room routes, and reload of rooms saved before a restart (``results/saves/room-*.json``)."""
    for p in sorted(saves_dir.glob("room-*.json")):
        try:
            d = json.loads(p.read_text())
            r = d["room"]
            sess = restore(d, p.stem)
            room = Room(r["host"], list(r["tokens"]))
            if len(room.tokens) != len(sess.seats) or not ROOM_PATH.match(f"/api/rooms/{r['id']}"):
                raise ValueError("room does not match its game")
        except (OSError, ValueError, KeyError, TypeError, HTTPException):
            continue  # a torn or foreign file must not stop the server
        sessions.pop(sess.id, None)
        sess.id, sess.room = r["id"], room
        sessions[sess.id] = sess

    def room_of(gid: str, request: Request):
        sess = sessions.get(gid)
        if sess is None or sess.room is None:
            raise HTTPException(404, "room not found")
        return sess, request.scope.get("state", {}).get("catan_room_seat")

    @app.post("/api/rooms")
    def create_room(body: dict = Body(...)):
        players = body.get("players")
        if not isinstance(players, list) or not any(isinstance(p, dict) and p.get("type") == "human" for p in players):
            raise HTTPException(400, "a room needs at least one human seat")
        sess = new_session(players, body.get("seed"), body.get("vp_to_win"), body.get("max_turns"), "play")
        with sess.lock:
            sess.room = Room(new_token(), [new_token() if s.kind == "human" else None for s in sess.seats])
            autosave(sess)
        return {"id": sess.id, "host_token": sess.room.host, "tokens": sess.room.tokens}

    @app.get("/api/rooms/{gid}")
    def lobby(gid: str, request: Request):
        sess, who = room_of(gid, request)
        now, g = time.time(), sess.game
        out = {"id": gid, "you": who, "phase": g.phase, "turn": g.turn, "actor": None if g.is_over else g.actor,
               "seats": [{**s.to_json(), "connected": s.kind == "agent" or sess.room.connected(i, now)}
                         for i, s in enumerate(sess.seats)]}
        if who == "host":
            out["tokens"] = sess.room.tokens
        return out

    @app.post("/api/rooms/{gid}/name")
    def set_name(gid: str, request: Request, body: dict = Body(...)):
        sess, who = room_of(gid, request)
        name = body.get("name")
        if who == "host":
            raise HTTPException(400, "the host link has no seat; open your seat link to set a name")
        if not isinstance(name, str) or not 1 <= len(name.strip()) <= 24 or not name.strip().isprintable():
            raise HTTPException(400, "name must be 1 to 24 printable characters")
        with sess.lock:
            sess.seats[who].name = name.strip()
            autosave(sess)
        return {"name": name.strip()}


async def _reply(send, status: int, detail: str):
    data = json.dumps({"detail": detail}).encode()
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(data)).encode())]})
    await send({"type": "http.response.body", "body": data})
