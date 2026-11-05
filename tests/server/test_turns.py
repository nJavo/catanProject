"""Turn boundaries for the replay turn stepper (catan/server/turns.py)."""
import pytest
from fastapi.testclient import TestClient

from catan.agents.registry import make_agent
from catan.engine import Game, GameConfig
from catan.server.app import create_app
from catan.server.turns import turn_segments


def play(n, seed, max_turns=None):
    """Random game; returns (actions as json, events per action, final game)."""
    g = Game(config=GameConfig(num_players=n, max_turns=max_turns), seed=seed)
    bots = [make_agent("random", seed=seed + i) for i in range(n)]
    acts, evs = [], []
    while not g.is_over:
        a = bots[g.actor].act(g, g.legal_actions())
        acts.append(a.to_json())
        evs.append(g.step(a))
    return acts, evs, g


def check(acts, evs, n):
    segs = turn_segments(acts, n)
    # contiguous, non-empty, covering every action
    assert segs[0]["start"] == 0 and segs[-1]["end"] == len(acts)
    assert all(a["end"] == b["start"] for a, b in zip(segs, segs[1:]))
    assert all(s["end"] > s["start"] for s in segs)
    setup = [s for s in segs if s["setup"]]
    assert [s["player"] for s in setup] == list(range(n)) + list(range(n))[::-1]
    assert all(s["end"] - s["start"] == 2 for s in setup) and [s["turn"] for s in setup] == list(range(1, 2 * n + 1))
    main = segs[2 * n:]
    assert not any(s["setup"] for s in main)
    for k, s in enumerate(main):
        assert s["turn"] == k + 1 and s["player"] == k % n
        # boundary agrees with the engine's turn_started event at the previous action
        ts = [e for e in evs[s["start"] - 1] if e["type"] == "turn_started"]
        assert ts == [{"type": "turn_started", "player": s["player"], "turn": s["turn"] - 1}]
        assert acts[s["start"]]["player"] == s["player"]
        if s is not main[-1]:
            assert acts[s["end"] - 1]["type"] == "END_TURN"
    return segs


@pytest.mark.parametrize("n,seed", [(4, 1), (3, 7), (2, 3)])
def test_boundaries_full_game_won_mid_turn(n, seed):
    acts, evs, g = play(n, seed)
    assert g.winner >= 0
    segs = check(acts, evs, n)
    # the game ends inside the winner's turn: the last segment has no END_TURN
    assert acts[-1]["type"] != "END_TURN" and segs[-1]["player"] == g.winner
    start = Game.replay(g.seed, g.config, acts[:segs[2 * n]["start"]])
    assert start.phase == "PRE_ROLL" and start.current == 0


def test_boundaries_truncated_at_end_turn():
    acts, evs, g = play(2, 5, max_turns=6)
    assert g.truncated and acts[-1]["type"] == "END_TURN"
    segs = check(acts, evs, 2)
    assert len(segs) == 4 + 6   # no empty trailing segment after the final END_TURN


def test_boundaries_partial_setup_and_bad_input():
    acts, _, _ = play(3, 2)
    segs = turn_segments(acts[:5], 3)
    assert [(s["start"], s["end"], s["setup"]) for s in segs] == [(0, 2, True), (2, 4, True), (4, 5, True)]
    assert turn_segments([], 4) == []
    assert turn_segments(["junk", {"type": "BUILD_ROAD"}], 2)[0]["end"] == 2


def test_replay_state_turns_and_since(tmp_path):
    (tmp_path / "results" / "replays").mkdir(parents=True)
    c = TestClient(create_app(data_dir=tmp_path, static_dir=tmp_path / "nodist"))
    gid = c.post("/api/lab/matches", json={"agents": ["random"] * 2, "seed": 8, "max_turns": 100,
                                           "run_to_end": True}).json()["id"]
    rep = c.get(f"/api/games/{gid}/replay").json()
    st = c.get("/api/replay-state", params={"source": "game", "ref": gid, "index": 30}).json()
    assert st["turns"] == turn_segments(rep["actions"], 2)
    seg = next(s for s in st["turns"] if not s["setup"] and s["turn"] == 3)
    # turn summary: hands at the turn's start and every event inside it
    st = c.get("/api/replay-state", params={"source": "game", "ref": gid, "index": seg["end"], "since": seg["start"]}).json()
    cfg = GameConfig(**rep["config"])
    before = Game.replay(rep["seed"], cfg, rep["actions"][:seg["start"]])
    assert [p["hand"] for p in st["prev_players"]] == [pl.hand for pl in before.players]
    assert {e["i"] for e in st["span_events"]} == set(range(seg["start"], seg["end"]))
    # since >= index falls back to the single last move
    st = c.get("/api/replay-state", params={"source": "game", "ref": gid, "index": 10, "since": 10}).json()
    assert {e["i"] for e in st["span_events"]} <= {9}
    assert [p["hand"] for p in st["prev_players"]] == [pl.hand for pl in Game.replay(rep["seed"], cfg, rep["actions"][:9]).players]
