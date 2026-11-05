"""Action and phase vocabulary shared by engine, UI server, bots and RL."""
from __future__ import annotations

from typing import NamedTuple, Any


class Phase:
    SETUP_SETTLEMENT = "SETUP_SETTLEMENT"
    SETUP_ROAD = "SETUP_ROAD"
    PRE_ROLL = "PRE_ROLL"
    DISCARD = "DISCARD"
    MOVE_ROBBER = "MOVE_ROBBER"
    STEAL = "STEAL"
    MAIN = "MAIN"
    ROAD_BUILDING = "ROAD_BUILDING"
    TRADE_RESPONSE = "TRADE_RESPONSE"
    TRADE_CONFIRM = "TRADE_CONFIRM"
    GAME_OVER = "GAME_OVER"


class A:
    """Action type names."""
    ROLL = "ROLL"
    END_TURN = "END_TURN"
    BUILD_ROAD = "BUILD_ROAD"              # value: edge id
    BUILD_SETTLEMENT = "BUILD_SETTLEMENT"  # value: vertex id
    BUILD_CITY = "BUILD_CITY"              # value: vertex id
    BUY_DEV = "BUY_DEV"
    PLAY_KNIGHT = "PLAY_KNIGHT"
    PLAY_ROAD_BUILDING = "PLAY_ROAD_BUILDING"
    PLAY_YEAR_OF_PLENTY = "PLAY_YEAR_OF_PLENTY"  # value: (r1, r2) sorted
    PLAY_MONOPOLY = "PLAY_MONOPOLY"              # value: resource index
    MOVE_ROBBER = "MOVE_ROBBER"            # value: hex id
    STEAL = "STEAL"                        # value: victim player id
    DISCARD = "DISCARD"                    # value: resource index (one card)
    MARITIME_TRADE = "MARITIME_TRADE"      # value: (give_res, get_res)
    OFFER_TRADE = "OFFER_TRADE"            # value: (give5, get5) proposer perspective
    ACCEPT_TRADE = "ACCEPT_TRADE"
    REJECT_TRADE = "REJECT_TRADE"
    COUNTER_OFFER = "COUNTER_OFFER"        # value: (give5, get5) proposer perspective
    CONFIRM_TRADE = "CONFIRM_TRADE"        # value: partner player id
    CANCEL_TRADE = "CANCEL_TRADE"


ALL_ACTION_TYPES = tuple(v for k, v in vars(A).items() if not k.startswith("_"))


class Action(NamedTuple):
    player: int
    type: str
    value: Any = None

    def to_json(self) -> dict:
        p, t, v = self  # unpacking beats three NamedTuple attribute lookups
        if isinstance(v, tuple):
            v = [list(x) if isinstance(x, tuple) else x for x in v]
        return {"player": p, "type": t, "value": v}

    @staticmethod
    def from_json(d: dict) -> "Action":
        if not isinstance(d, dict):
            raise ValueError("action must be an object")
        t = d.get("type")
        if t not in ALL_ACTION_TYPES:
            raise ValueError(f"unknown action type {t!r}")
        p = d.get("player")
        if not isinstance(p, int) or isinstance(p, bool):
            raise ValueError("player must be an int")
        v = d.get("value")
        if isinstance(v, list):
            v = tuple(tuple(x) if isinstance(x, list) else x for x in v)
        return Action(p, t, v)
