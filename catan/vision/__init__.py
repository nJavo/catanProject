"""Screenshot -> PositionSpec importer (docs/VISION_IMPORT.md).

Reads only an image the user supplies. For post-game review and consenting
private games; it never talks to any game site.
"""
from __future__ import annotations

from pathlib import Path

from catan.analysis import validate_position_detailed
from .detect import LOW, VisionUnavailable, load_image, load_profile, profiles, read_board

__all__ = ["import_image", "board_to_spec", "VisionUnavailable", "load_image", "load_profile", "profiles", "read_board"]

# Read from the image: none of these. They get defaults and confidence 0 so the editor flags them.
PANEL_FIELDS = ("current", "phase", "turn", "hands", "resource_counts", "dev_cards", "knights_played")


def board_to_spec(b: dict, num_players: int = 4) -> dict:
    """PositionSpec from ``read_board`` output. Phase is a guess from the piece counts: a board that
    could be mid-setup is set to the next setup placement (so the suggestion is an opening move),
    anything else to the viewer's main phase with an empty hand."""
    vo, vl, eo = b["vertex_owner"], b["vertex_level"], b["edge_owner"]
    n = num_players
    sett, roads = sum(o >= 0 for o in vo), sum(o >= 0 for o in eo)
    order = list(range(n)) + list(reversed(range(n)))
    setup = not any(level == 2 for level in vl) and sett <= 2 * n and roads in (sett, sett - 1) and (sett, roads) != (2 * n, 2 * n)
    if setup:
        phase = "SETUP_SETTLEMENT" if roads == sett else "SETUP_ROAD"
        viewer = order[sett if roads == sett else sett - 1]
    else:
        phase, viewer = "MAIN", 0
    players = [{"hand": [0] * 5, "dev_cards": [0] * 5, "knights_played": 0} if q == viewer
               else {"resource_count": 0, "dev_count": 0, "knights_played": 0} for q in range(n)]
    return {"version": 1, "num_players": n, "viewer": viewer, "layout": b["layout"],
            "vertex_owner": vo, "vertex_level": vl, "edge_owner": eo, "robber": b["robber"],
            "current": viewer if phase == "MAIN" else 0, "phase": phase, "rolled": phase == "MAIN",
            "dev_played": False, "turn": 0, "players": players}


def import_image(data: bytes, profile: str | None = None, extra_profiles: Path | None = None, num_players: int = 4) -> dict:
    """Image bytes -> ``{position, errors, confidence, low_confidence, lattice, profile}``.
    ``profile`` names a profile file; None tries every known profile and keeps the best lattice fit."""
    img = load_image(data)
    cands = [p for p in profiles(extra_profiles) if profile is None or p.stem == profile]
    if not cands:
        raise ValueError(f"unknown profile {profile!r}; known: {[p.stem for p in profiles(extra_profiles)]}")
    best = None
    for p in cands:
        prof = load_profile(p)
        b = read_board(img, prof)
        if best is None or b["lattice"]["score"] > best[1]["lattice"]["score"]:
            best = (prof["name"], b)
    name, b = best
    # pieces need a seat for every colour seen
    used = {o for o in b["vertex_owner"] + b["edge_owner"] if o >= 0}
    spec = board_to_spec(b, max([num_players] + [o + 1 for o in used]))
    conf = b["confidence"] | {"panel": {k: 0.0 for k in PANEL_FIELDS}}
    low = {
        "hexes": sorted({h for h in range(19) if min(conf["hex_terrain"][h], conf["hex_number"][h]) < LOW}
                        | ({b["robber"]} if conf["robber"] < LOW else set())),
        "vertices": [v for v, c in enumerate(conf["vertices"]) if c < LOW],
        "edges": [e for e, c in enumerate(conf["edges"]) if c < LOW]
                 + [e for e, c in zip(spec["layout"]["port_edges"], conf["ports"]) if c < LOW],
        "fields": list(PANEL_FIELDS),
    }
    return {"position": spec, "errors": validate_position_detailed(spec), "confidence": conf,
            "low_confidence": low, "lattice": b["lattice"], "profile": name}
