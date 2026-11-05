"""Human-readable action labels, e.g. "Build settlement at vertex 23 (wheat 6, ore 8, 3:1 port)"."""
from __future__ import annotations

from catan.engine import TOPOLOGY as T, RESOURCES, A
from catan.engine.board import DESERT


def _vertex_desc(layout, v):
    parts = [f"{layout.hex_terrain[h]} {layout.hex_number[h]}" if layout.hex_terrain[h] != DESERT else "desert"
             for h in T.vertex_hexes[v]]
    port = layout.vertex_ports().get(v)
    if port:
        parts.append(f"{port} port" if port == "3:1" else f"{port} 2:1 port")
    return ", ".join(parts)


def _hex_desc(layout, h):
    t = layout.hex_terrain[h]
    return "desert" if t == DESERT else f"{t} {layout.hex_number[h]}"


def _cards(vec):
    return " + ".join(f"{n} {RESOURCES[i]}" for i, n in enumerate(vec) if n) or "nothing"


def label(game, a) -> str:
    """Label for action ``a`` in ``game`` (uses only public board info and the
    actor's own trade rates)."""
    t, v, lay = a.type, a.value, game.layout
    if t == A.BUILD_SETTLEMENT:
        return f"Build settlement at vertex {v} ({_vertex_desc(lay, v)})"
    if t == A.BUILD_CITY:
        return f"Upgrade to city at vertex {v} ({_vertex_desc(lay, v)})"
    if t == A.BUILD_ROAD:
        x, y = T.edge_vertices[v]
        free = " (free)" if game.phase in ("SETUP_ROAD", "ROAD_BUILDING") else ""
        return f"Build road at edge {v} (vertices {x}-{y}){free}"
    if t == A.ROLL:
        return "Roll the dice"
    if t == A.END_TURN:
        return "End turn"
    if t == A.BUY_DEV:
        return "Buy a development card"
    if t == A.PLAY_KNIGHT:
        return "Play Knight"
    if t == A.PLAY_ROAD_BUILDING:
        return "Play Road Building"
    if t == A.PLAY_YEAR_OF_PLENTY:
        return f"Play Year of Plenty: {RESOURCES[v[0]]} + {RESOURCES[v[1]]}"
    if t == A.PLAY_MONOPOLY:
        return f"Play Monopoly on {RESOURCES[v]}"
    if t == A.MOVE_ROBBER:
        owners = sorted({game.vertex_owner[x] for x in T.hex_vertices[v]} - {-1})
        blocks = f"; touches player {', '.join(map(str, owners))}" if owners else ""
        return f"Move robber to hex {v} ({_hex_desc(lay, v)}{blocks})"
    if t == A.STEAL:
        return f"Steal from player {v}"
    if t == A.DISCARD:
        return f"Discard 1 {RESOURCES[v]}"
    if t == A.MARITIME_TRADE:
        g, r = v
        rate = game.trade_rate(a.player, g)
        how = {4: "bank", 3: "3:1 port", 2: f"{RESOURCES[g]} 2:1 port"}[rate]
        return f"Trade {rate} {RESOURCES[g]} → 1 {RESOURCES[r]} ({how})"
    if t == A.OFFER_TRADE:
        return f"Offer {_cards(v[0])} for {_cards(v[1])} (to all players)"
    if t == A.COUNTER_OFFER:
        return f"Counter-offer: they give {_cards(v[0])}, you give {_cards(v[1])}"
    if t == A.ACCEPT_TRADE:
        return "Accept the trade"
    if t == A.REJECT_TRADE:
        return "Reject the trade"
    if t == A.CONFIRM_TRADE:
        return f"Confirm trade with player {v}"
    if t == A.CANCEL_TRADE:
        return "Cancel the trade"
    return f"{t} {v}" if v is not None else t

