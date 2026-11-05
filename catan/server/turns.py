"""Turn boundaries of a replay, derived from its action list alone.

A segment is a half-open range of replay indexes ``[start, end)``: index k is
the state after k actions, so ``start`` is the turn's opening position (the
PRE_ROLL of a main turn). Setup is split into one segment per settlement+road
placement pair; every main turn ends with its END_TURN.
"""


def turn_segments(actions: list[dict], n_players: int) -> list[dict]:
    actions = [a if isinstance(a, dict) else {} for a in actions]   # replay files are untrusted
    segs, start, setup_left, turn = [], 0, 2 * n_players, 0
    for i, a in enumerate(actions):
        t = a.get("type")
        if setup_left and t == "BUILD_ROAD":
            setup_left -= 1
            segs.append({"start": start, "end": i + 1, "player": actions[start].get("player"), "setup": True,
                         "turn": 2 * n_players - setup_left})
            start = i + 1
        elif not setup_left and t == "END_TURN":
            turn += 1
            segs.append({"start": start, "end": i + 1, "player": actions[start].get("player"), "setup": False,
                         "turn": turn})
            start = i + 1
    if start < len(actions):   # game ended (won) mid-turn, or setup still in progress
        segs.append({"start": start, "end": len(actions), "player": actions[start].get("player"),
                     "setup": setup_left > 0, "turn": (2 * n_players - setup_left + 1) if setup_left else turn + 1})
    return segs
