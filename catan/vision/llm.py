"""Screenshot -> PositionSpec with a multimodal LLM (Google Gemini). docs/VISION_IMPORT.md, "LLM backend".

The model never sees engine ids. It reports what it can see: terrain and number per hex in reading
order, pieces as (colour, hex, corner/side direction), harbors as (hex, side, type), plus the
colonist.io side panels. This module turns that into engine vertex/edge ids, runs the position
validator, and makes at most one repair call when the reading fails the checks.

The API key is read from the environment or the repo-root ``.env`` and is only ever sent to
Google's endpoint, in the ``x-goog-api-key`` header (never in a URL, never logged).
"""
from __future__ import annotations

import base64
import json
import math
import os
import random
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

from catan.analysis import validate_position_detailed
from catan.engine import TOPOLOGY as T
from catan.engine.board import PORT_TYPES, RESOURCES, TERRAIN_COUNTS, TOKEN_SEQUENCE, generate_layout

from .detect import LOW, load_image

ENV_FILE = Path(__file__).resolve().parents[2] / ".env"
DEFAULT_MODEL = "gemini-3.5-flash-lite"
ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
TIMEOUT = int(os.environ.get("GEMINI_TIMEOUT", "180"))  # seconds per call; larger Flash models think longer
# USD per 1M tokens (input, output incl. thinking), paid tier, from ai.google.dev pricing on 2026-11-04.
# shortcut: hard-coded table, update when Google changes prices; unknown models report cost null.
PRICES = {"gemini-3.5-flash-lite": (0.30, 2.50), "gemini-3.1-flash-lite": (0.25, 1.50),
          "gemini-3.8-flash": (0.75, 3.75), "gemini-3.6-flash": (0.75, 3.75),
          "gemini-3-flash-preview": (0.50, 3.00), "gemini-2.5-flash-lite": (0.10, 0.40)}

# hex_vertices lists corners clockwise from the top-right (catan.engine.board, pointy-top, y down)
CORNERS = ("NE", "SE", "S", "SW", "NW", "N")
# side -> the two corner indexes it joins
SIDES = {"NE": (5, 0), "E": (0, 1), "SE": (1, 2), "SW": (2, 3), "W": (3, 4), "NW": (4, 5)}
TERRAINS = ("wood", "brick", "sheep", "wheat", "ore", "desert")
PORT_KINDS = ("3:1",) + RESOURCES
DEVS = ("knight", "vp", "road_building", "year_of_plenty", "monopoly")
MODEL_CONF, FLAG_CONF = 0.8, 0.3  # confidence for a field the model read, and for one a check flagged


class LLMError(RuntimeError):
    """The model call failed or returned something unusable."""


# ------------------------------------------------------------------ config
def read_env(path: Path = ENV_FILE) -> dict:
    """``KEY=value`` pairs from ``path`` (python-dotenv when installed)."""
    if not path.is_file():
        return {}
    try:
        from dotenv import dotenv_values
        return {k: v for k, v in dotenv_values(path).items() if v is not None}
    except ImportError:
        out = {}
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.removeprefix("export ").split("=", 1)
                out[k.strip()] = v.strip().strip("'\"")
        return out


def config(path: Path | None = None) -> tuple[str | None, str]:
    """(api key or None, model id). The process environment wins over ``.env``."""
    env = read_env(path or ENV_FILE)
    key = os.environ.get("GEMINI_API_KEY") or env.get("GEMINI_API_KEY") or None
    return key, os.environ.get("GEMINI_MODEL") or env.get("GEMINI_MODEL") or DEFAULT_MODEL


# ------------------------------------------------------------------ prompt and schema
PROMPT = """You are reading a screenshot of a Settlers of Catan board (usually colonist.io). Report only what you see.

BOARD GEOMETRY. The base board has 19 land hexes, pointy-top: every hex has a corner at the top (N) and the bottom (S).
Rows from top to bottom hold 3, 4, 5, 4, 3 hexes. Number the hexes in reading order, top row first, left to right:
  row 1: 0 1 2
  row 2: 3 4 5 6
  row 3: 7 8 9 10 11
  row 4: 12 13 14 15
  row 5: 16 17 18
Hex 9 is the centre. The inner ring is 4 5 10 14 13 8. The outer ring, clockwise from the top-left, is 0 1 2 6 11 15 18 17 16 12 7 3.
Each hex has 6 corners: N, NE, SE, S, SW, NW, and 6 sides: NE, E, SE, SW, W, NW.
Neighbours share sides: hex 4's W side is hex 3's E side; hex 4's NW side is hex 0's SE side; hex 4's NE side is hex 1's SW side.
A corner is shared by up to 3 hexes. Name it from any one hex that touches it.

REPORT:
- pointy_top: false if the hexes have flat tops instead (then the rest can be empty).
- hexes: exactly 19 entries in reading order: terrain (wood=forest, brick=hills/clay, sheep=pasture, wheat=fields/grain, ore=mountains/rock, desert) and the number on its token (null for the desert). A base board has 4 wood, 3 brick, 4 sheep, 4 wheat, 3 ore, 1 desert and the tokens 2,3,3,4,4,5,5,6,6,8,8,9,9,10,10,11,11,12.
- robber: the hex index under the robber.
- harbors: the 9 harbors (ships/docks in the sea). For each, the land hex it faces, that hex's side toward the harbor, and the type: "3:1" or the resource of a 2:1 harbor (4 are 3:1, one 2:1 per resource).
- buildings: every settlement and city: colour, kind, a hex it touches and the corner of that hex it sits on.
- roads: every road: colour, a hex it borders and the side of that hex it lies on. Every road touches its owner's other roads or buildings.
- board_points: pixel position of the centre of hex 0 and of hex 18, as x and y from 0 to 1000 (fractions of image width and height times 1000).
- panel (null fields when not visible): players in the order the side panel lists them, top to bottom, each with colour, resource card count, development card count, victory points, knights played and longest road length; my_color: the colour of the player whose own hand is shown at the bottom; my_hand: their resource card counts; my_dev_cards: their unplayed development cards by type; dice: the last roll as two dice; turn_color: whose turn it is.
Colours are lower-case words as shown (red, blue, orange, white, green, black, purple, ...). Use the same word on the board and in the panel."""

_INT = {"type": "integer"}
_NINT = {"type": ["integer", "null"]}
_HEX = {"type": "integer", "minimum": 0, "maximum": 18}
_COUNTS = lambda keys: {"type": ["object", "null"], "properties": {k: _INT for k in keys}, "required": list(keys)}  # noqa: E731
SCHEMA = {
    "type": "object",
    "properties": {
        "pointy_top": {"type": "boolean"},
        "hexes": {"type": "array", "minItems": 19, "maxItems": 19, "items": {
            "type": "object", "required": ["terrain", "number"],
            "properties": {"terrain": {"type": "string", "enum": list(TERRAINS)}, "number": {"type": ["integer", "null"], "minimum": 2, "maximum": 12}}}},
        "robber": _HEX,
        "harbors": {"type": "array", "maxItems": 12, "items": {
            "type": "object", "required": ["hex", "side", "type"],
            "properties": {"hex": _HEX, "side": {"type": "string", "enum": list(SIDES)}, "type": {"type": "string", "enum": list(PORT_KINDS)}}}},
        "buildings": {"type": "array", "items": {
            "type": "object", "required": ["color", "kind", "hex", "corner"],
            "properties": {"color": {"type": "string"}, "kind": {"type": "string", "enum": ["settlement", "city"]},
                           "hex": _HEX, "corner": {"type": "string", "enum": list(CORNERS)}}}},
        "roads": {"type": "array", "items": {
            "type": "object", "required": ["color", "hex", "side"],
            "properties": {"color": {"type": "string"}, "hex": _HEX, "side": {"type": "string", "enum": list(SIDES)}}}},
        "board_points": {"type": ["object", "null"], "required": ["hex0", "hex18"], "properties": {
            p: {"type": "object", "required": ["x", "y"], "properties": {"x": _INT, "y": _INT}} for p in ("hex0", "hex18")}},
        "panel": {"type": ["object", "null"], "properties": {
            "players": {"type": "array", "maxItems": 6, "items": {
                "type": "object", "required": ["color"],
                "properties": {"color": {"type": "string"}, "cards": _NINT, "dev_cards": _NINT, "vp": _NINT,
                               "knights": _NINT, "road_length": _NINT}}},
            "my_color": {"type": ["string", "null"]},
            "my_hand": _COUNTS(RESOURCES),
            "my_dev_cards": _COUNTS(DEVS),
            "dice": {"type": ["array", "null"], "minItems": 2, "maxItems": 2, "items": {"type": "integer", "minimum": 1, "maximum": 6}},
            "turn_color": {"type": ["string", "null"]}}},
    },
    "required": ["pointy_top", "hexes", "robber", "harbors", "buildings", "roads"],
}


# ------------------------------------------------------------------ the model's words <-> engine ids
def corner_vertex(h: int, corner: str) -> int:
    return T.hex_vertices[h][CORNERS.index(corner)]


def side_edge(h: int, side: str) -> int:
    i, j = SIDES[side]
    return T.edge_lookup[frozenset((T.hex_vertices[h][i], T.hex_vertices[h][j]))]


def vertex_name(v: int) -> tuple[int, str]:
    """(hex, corner) for engine vertex ``v``: the lowest hex touching it."""
    h = min(T.vertex_hexes[v])
    return h, CORNERS[T.hex_vertices[h].index(v)]


def edge_name(e: int) -> tuple[int, str]:
    """(hex, side) for engine edge ``e``: the lowest hex bordering it."""
    vs = set(T.edge_vertices[e])
    for h in sorted(set(T.vertex_hexes[min(vs)]) & set(T.vertex_hexes[max(vs)])):
        for s in SIDES:
            if side_edge(h, s) == e:
                return h, s
    raise ValueError(f"edge {e} borders no hex")


def spec_to_reading(spec: dict, colors: list[str]) -> dict:
    """A PositionSpec's board in the model's terms (``colors[seat]``). The inverse of the board part of ``reading_to_board``."""
    lay = spec["layout"]
    return {
        "pointy_top": True,
        "hexes": [{"terrain": t, "number": n} for t, n in zip(lay["hex_terrain"], lay["hex_number"])],
        "robber": spec["robber"],
        "harbors": [dict(zip(("hex", "side"), edge_name(e)), type=t) for e, t in zip(lay["port_edges"], lay["port_types"])],
        "buildings": [dict(color=colors[o], kind="city" if spec["vertex_level"][v] == 2 else "settlement",
                           **dict(zip(("hex", "corner"), vertex_name(v)))) for v, o in enumerate(spec["vertex_owner"]) if o >= 0],
        "roads": [dict(color=colors[o], **dict(zip(("hex", "side"), edge_name(e)))) for e, o in enumerate(spec["edge_owner"]) if o >= 0],
    }


# The model mixes our app's seat names with plain colour words ("ember" and "orange" for one
# player), which split one player into two seats. Fold both to one plain name.
_ALIAS = {"ember": "orange", "tide": "blue", "chalk": "white", "cream": "white", "heather": "purple",
          "violet": "purple", "grey": "gray", "beige": "white"}


def _canon(c) -> str:
    c = (c or "").strip().lower()
    return _ALIAS.get(c, c)


def seat_colors(r: dict) -> list[str]:
    """Colour order = seat order: the side panel's order (top to bottom), then any colour only seen on the board."""
    seen = []
    for c in [p.get("color") for p in ((r.get("panel") or {}).get("players") or [])] + \
             [p["color"] for p in r.get("buildings", []) + r.get("roads", [])]:
        c = _canon(c)
        if c and c not in seen:
            seen.append(c)
    return seen


def reading_to_board(r: dict, colors: list[str]) -> tuple[dict, list[dict]]:
    """Model reading -> ``read_board``-shaped dict, plus problems found while mapping (duplicates, clashes)."""
    probs = []
    seat = {c: i for i, c in enumerate(colors)}
    hexes = r["hexes"]
    terrain = [x["terrain"] for x in hexes]
    number = [None if x["terrain"] == "desert" else x["number"] for x in hexes]
    port_edges, port_types = [], []
    for p in r.get("harbors", []):
        e = side_edge(p["hex"], p["side"])
        if e not in T.coast_edges:
            probs.append(_prob(f"harbor at hex {p['hex']} side {p['side']} is not on the coast", hexes=[p["hex"]]))
        elif e in port_edges:
            probs.append(_prob(f"two harbors on hex {p['hex']} side {p['side']}", hexes=[p["hex"]]))
        else:
            port_edges.append(e)
            port_types.append(p["type"])
    vo, vl, eo = [-1] * T.num_vertices, [0] * T.num_vertices, [-1] * T.num_edges
    for b in r.get("buildings", []):
        v = corner_vertex(b["hex"], b["corner"])
        o = seat[_canon(b["color"])]
        if vo[v] >= 0 and vo[v] != o:
            probs.append(_prob(f"two colours on the corner {b['corner']} of hex {b['hex']}", vertices=[v]))
        vo[v], vl[v] = o, max(vl[v], 2 if b["kind"] == "city" else 1)
    for d in r.get("roads", []):
        e = side_edge(d["hex"], d["side"])
        o = seat[_canon(d["color"])]
        if eo[e] >= 0 and eo[e] != o:
            probs.append(_prob(f"two colours on side {d['side']} of hex {d['hex']}", edges=[e]))
        eo[e] = o
    return {"layout": {"hex_terrain": terrain, "hex_number": number, "port_edges": port_edges, "port_types": port_types},
            "vertex_owner": vo, "vertex_level": vl, "edge_owner": eo, "robber": r["robber"]}, probs


def _prob(message, vertices=(), edges=(), hexes=()):
    return {"message": message, "vertices": list(vertices), "edges": list(edges), "hexes": list(hexes)}


def layout_checks(lay: dict) -> list[dict]:
    """Quota problems tied to the hexes that break them (the validator only says the totals are wrong),
    plus red numbers (6/8) on neighbouring hexes, which colonist's balanced boards avoid."""
    out = []
    terr, nums = lay["hex_terrain"], lay["hex_number"]
    for t, c in Counter(terr).items():
        if c > TERRAIN_COUNTS.get(t, 0):
            out.append(_prob(f"{c} {t} hexes, the base set has {TERRAIN_COUNTS[t]}", hexes=[h for h in range(19) if terr[h] == t]))
    want = Counter(TOKEN_SEQUENCE)
    for n, c in Counter(x for x in nums if x is not None).items():
        if c > want.get(n, 0):
            out.append(_prob(f"{c} hexes numbered {n}, the base set has {want.get(n, 0)}", hexes=[h for h in range(19) if nums[h] == n]))
    for a in range(19):
        for b in range(a + 1, 19):
            if nums[a] in (6, 8) and nums[b] in (6, 8) and len(set(T.hex_vertices[a]) & set(T.hex_vertices[b])) == 2:
                out.append(_prob(f"hexes {a} and {b} are neighbours and both have a red number ({nums[a]}, {nums[b]}); check both", hexes=[a, b]))
    if len(lay["port_edges"]) != 9 or Counter(lay["port_types"]) != Counter(PORT_TYPES):
        out.append(_prob(f"{len(lay['port_edges'])} harbors read; the base board has 9: four 3:1 and one 2:1 per resource"))
    return out


# ------------------------------------------------------------------ spec
def _n(x, hi=30):
    return min(max(int(x), 0), hi) if isinstance(x, int) else 0


def build_spec(r: dict, num_players: int) -> tuple[dict, list[str], list[str]]:
    """(PositionSpec, colour of each seat, panel fields not read)."""
    from . import board_to_spec
    colors = seat_colors(r)
    if len(colors) > 4:
        raise LLMError(f"the model saw {len(colors)} colours ({', '.join(colors)}); at most 4 players")
    b, _ = reading_to_board(r, colors)
    n = max(num_players, len(colors))
    while len(colors) < n:
        colors.append(f"seat{len(colors)}")
    spec = board_to_spec(b, n)
    panel = r.get("panel") or {}
    rows = {_canon(p.get("color")): p for p in panel.get("players") or []}
    me = _canon(panel.get("my_color"))
    turn = _canon(panel.get("turn_color"))
    unread = {"hands", "resource_counts", "dev_cards", "knights_played", "current", "turn", "phase"}
    if spec["phase"] == "MAIN":
        spec["viewer"] = colors.index(me) if me in colors else 0
        spec["current"] = colors.index(turn) if turn in colors else spec["viewer"]
        unread -= {"current"} if turn in colors else set()
    v = spec["viewer"]
    hand, devs = panel.get("my_hand"), panel.get("my_dev_cards")
    players = []
    for q in range(n):
        row = rows.get(colors[q], {})
        if q == v:
            players.append({"hand": [_n(hand.get(k), 19) for k in RESOURCES] if hand else [0] * 5,
                            "dev_cards": [_n(devs.get(k), 14) for k in DEVS] if devs else [0] * 5,
                            "knights_played": _n(row.get("knights"), 14)})
        else:
            players.append({"resource_count": _n(row.get("cards"), 95), "dev_count": _n(row.get("dev_cards"), 25),
                            "knights_played": _n(row.get("knights"), 14)})
    spec["players"] = players
    if hand and me == colors[v]:
        unread.discard("hands")
    if devs and me == colors[v]:
        unread.discard("dev_cards")
    if rows:
        unread -= {"resource_counts", "knights_played"}
    dice = panel.get("dice")
    if isinstance(dice, list) and len(dice) == 2 and all(isinstance(d, int) and 1 <= d <= 6 for d in dice):
        spec["last_roll"] = dice
    return spec, colors, sorted(unread)


def lattice_of(r: dict, w: int, h: int) -> dict | None:
    """Pixel lattice ``{cx, cy, r}`` from the model's hex 0 and hex 18 centres (for the preview overlay)."""
    bp = r.get("board_points")
    try:
        (x0, y0), (x1, y1) = [(bp[k]["x"] * w / 1000, bp[k]["y"] * h / 1000) for k in ("hex0", "hex18")]
    except (TypeError, KeyError):
        return None
    (a0, b0), (a1, b1) = T.hex_xy[0], T.hex_xy[18]
    rr = math.hypot(x1 - x0, y1 - y0) / math.hypot(a1 - a0, b1 - b0)
    if rr <= 0:
        return None
    return {"cx": (x0 + x1) / 2, "cy": (y0 + y1) / 2, "r": rr, "score": None, "width": w, "height": h}


# ------------------------------------------------------------------ the model call
def gemini_client(key: str, model: str):
    """``call(parts) -> (text, usage)`` for Gemini's generateContent REST endpoint (stdlib only)."""
    url = ENDPOINT.format(model=model)

    def call(parts):
        body = json.dumps({"contents": [{"role": "user", "parts": parts}],
                           "generationConfig": {"responseMimeType": "application/json", "responseJsonSchema": SCHEMA,
                                                "temperature": 0}}).encode()
        req = urllib.request.Request(url, body, {"Content-Type": "application/json", "x-goog-api-key": key})
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                j = json.load(resp)
        except urllib.error.HTTPError as e:
            try:
                msg = json.load(e).get("error", {}).get("message", "")
            except (ValueError, AttributeError):
                msg = ""
            raise LLMError(f"Gemini returned HTTP {e.code}: {msg[:300]}") from None
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise LLMError(f"cannot reach Gemini: {getattr(e, 'reason', e)}") from None
        try:
            text = "".join(p.get("text", "") for p in j["candidates"][0]["content"]["parts"] if not p.get("thought"))
        except (KeyError, IndexError, TypeError):
            raise LLMError(f"Gemini returned no answer ({(j.get('promptFeedback') or {}).get('blockReason', 'empty')})") from None
        return text, j.get("usageMetadata", {})
    return call


def _parse(text: str) -> dict:
    try:
        r = json.loads(text)
    except ValueError:
        raise LLMError("the model's answer is not JSON") from None
    if not isinstance(r, dict) or not r.get("pointy_top", True):
        raise LLMError("the model says the board is not pointy-top; only the standard orientation is supported")
    hexes = r.get("hexes")
    if not (isinstance(hexes, list) and len(hexes) == 19 and all(isinstance(x, dict) and x.get("terrain") in TERRAINS for x in hexes)):
        raise LLMError("the model did not report 19 hexes")
    if not (isinstance(r.get("robber"), int) and 0 <= r["robber"] < 19):
        r["robber"] = next(h for h in range(19) if hexes[h]["terrain"] == "desert") if any(x["terrain"] == "desert" for x in hexes) else 0
    ok = lambda x, k, names: isinstance(x, dict) and isinstance(x.get("hex"), int) and 0 <= x["hex"] < 19 and x.get(k) in names  # noqa: E731
    r["harbors"] = [x for x in r.get("harbors") or [] if ok(x, "side", SIDES) and x.get("type") in PORT_KINDS]
    r["buildings"] = [x for x in r.get("buildings") or [] if ok(x, "corner", CORNERS) and isinstance(x.get("color"), str)
                      and x["color"].strip() and x.get("kind") in ("settlement", "city")]
    r["roads"] = [x for x in r.get("roads") or [] if ok(x, "side", SIDES) and isinstance(x.get("color"), str) and x["color"].strip()]
    for x in hexes:
        if not (isinstance(x.get("number"), int) and 2 <= x["number"] <= 12 and x["number"] != 7):
            x["number"] = None
    return r


def _problems(r: dict, num_players: int) -> tuple[dict, list[str], list[str], list[dict], list[dict]]:
    """(spec, colours, unread panel fields, validator errors, every problem incl. heuristics)."""
    spec, colors, unread = build_spec(r, num_players)
    _, mapping = reading_to_board(r, colors)
    errors = validate_position_detailed(spec)
    # the validator stops at layout errors; check the pieces on a valid stand-in layout so one repair call sees both
    lay = generate_layout(random.Random(0)).to_dict()
    stand_in = validate_position_detailed(spec | {"layout": lay, "robber": lay["hex_terrain"].index("desert")})
    extra = [p for p in stand_in if p not in errors and (p["vertices"] or p["edges"])] if errors else []
    return spec, colors, unread, errors, mapping + layout_checks(spec["layout"]) + errors + extra


def _in_words(p: dict) -> str:
    """A problem with engine ids also named in the model's terms."""
    at = [f"hex {h} corner {c}" for h, c in map(vertex_name, p.get("vertices", []))] + \
         [f"hex {h} side {s}" for h, s in map(edge_name, p.get("edges", []))] + [f"hex {h}" for h in p.get("hexes", [])]
    return p["message"] + (f" (at {'; '.join(at)})" if at else "")


def _cost(model, usage):
    pin = sum(u.get("promptTokenCount", 0) for u in usage)
    pout = sum(u.get("candidatesTokenCount", 0) + u.get("thoughtsTokenCount", 0) for u in usage)
    price = PRICES.get(model)
    return {"calls": len(usage), "input_tokens": pin, "output_tokens": pout,
            "total_tokens": sum(u.get("totalTokenCount", 0) for u in usage),
            "usd": round((pin * price[0] + pout * price[1]) / 1e6, 5) if price else None,
            "price_source": "ai.google.dev/gemini-api/docs/pricing (paid tier, 2026-11-04)" if price else "unknown model"}


def _mime(data: bytes) -> str:
    return "image/png" if data[:4] == b"\x89PNG" else "image/webp" if data[8:12] == b"WEBP" else "image/jpeg"


def import_image_llm(data: bytes, num_players: int = 4, client=None, model: str | None = None) -> dict:
    """Image bytes -> the same result shape as ``catan.vision.import_image``, plus ``colors``,
    ``panel``, ``usage`` and ``repaired``. ``client(parts) -> (text, usage)``; default: Gemini with the configured key."""
    img = load_image(data)  # format and size checks, and the pixel size for the overlay
    if client is None:
        key, model = config()
        if not key:
            raise LLMError("no GEMINI_API_KEY configured")
        client = gemini_client(key, model)
    model = model or DEFAULT_MODEL
    image = {"inlineData": {"mimeType": _mime(data), "data": base64.b64encode(data).decode()}}
    usage = []
    text, u = client([{"text": PROMPT}, image])
    usage.append(u)
    r = _parse(text)
    spec, colors, unread, errors, probs = _problems(r, num_players)
    repaired = False
    if probs:  # one repair call with the image, the previous JSON and the problem list
        fix = "Your previous reading of this image:\n" + json.dumps(r) + "\n\nIt fails these checks:\n- " + \
              "\n- ".join(_in_words(p) for p in probs) + \
              "\n\nLook at the image again and return the corrected full reading in the same JSON format. Change only what is wrong."
        text, u = client([{"text": PROMPT}, image, {"text": fix}])
        usage.append(u)
        try:
            r2 = _parse(text)
            got = _problems(r2, num_players)
        except LLMError:
            got = None  # a broken repair keeps the first reading
        if got is not None and len(got[3]) <= len(errors):
            r, (spec, colors, unread, errors, probs), repaired = r2, got, True
    # confidence: model-read fields MODEL_CONF, fields named by a remaining problem FLAG_CONF
    bad_h = {h for p in probs for h in p.get("hexes", [])}
    bad_v = {v for p in probs for v in p.get("vertices", [])}
    bad_e = {e for p in probs for e in p.get("edges", [])}
    c = lambda bad, i: FLAG_CONF if i in bad else MODEL_CONF  # noqa: E731
    lay = spec["layout"]
    lat = lattice_of(r, int(img.shape[1]), int(img.shape[0]))
    conf = {"hex_terrain": [c(bad_h, h) for h in range(19)], "hex_number": [c(bad_h, h) for h in range(19)],
            "ports": [FLAG_CONF if any("harbor" in p["message"] for p in probs) else MODEL_CONF for _ in lay["port_edges"]],
            "robber": MODEL_CONF, "vertices": [c(bad_v, v) for v in range(T.num_vertices)],
            "edges": [c(bad_e, e) for e in range(T.num_edges)], "lattice": MODEL_CONF if lat else 0.0,
            "panel": {k: 0.0 if k in unread else MODEL_CONF for k in
                      ("current", "phase", "turn", "hands", "resource_counts", "dev_cards", "knights_played")}}
    low = {"hexes": sorted(h for h in range(19) if conf["hex_terrain"][h] < LOW),
           "vertices": [v for v, x in enumerate(conf["vertices"]) if x < LOW],
           "edges": [e for e, x in enumerate(conf["edges"]) if x < LOW]
                    + [e for e, x in zip(lay["port_edges"], conf["ports"]) if x < LOW],
           "fields": [k for k, x in conf["panel"].items() if x < LOW]}
    return {"position": spec, "errors": errors, "confidence": conf, "low_confidence": low, "lattice": lat,
            "profile": f"gemini:{model}", "colors": {col: s for s, col in enumerate(colors)},
            "panel": r.get("panel"), "checks": [_in_words(p) for p in probs], "repaired": repaired,
            "usage": _cost(model, usage)}
