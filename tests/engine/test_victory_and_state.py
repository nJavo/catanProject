import json
import random

import pytest

from catan.engine import A, Action, Phase, Game, GameConfig, IllegalAction, COST
from catan.engine.actions import ALL_ACTION_TYPES
from catan.engine.game import VICTORY_POINT

from .helpers import main_game, put_settlement, put_road, give, vertex_path


def _nine_vp(g, p):
    vs, es = vertex_path(30)
    spots = [v for v in vs[::2]][:4]
    for v in spots[:3]:
        put_settlement(g, p, v, city=True)       # 6
    put_settlement(g, p, spots[3])               # 7
    g.largest_army_owner = p                     # 9
    g.players[p].knights = 3
    return vs, es


def test_win_on_own_turn_by_building():
    g = main_game()
    vs, es = _nine_vp(g, 0)
    assert g.victory_points(0) == 9
    put_road(g, 0, es[14])
    put_road(g, 0, es[15])
    give(g, 0, COST["settlement"])
    tgt = vs[16] if g._settlement_spot_free(vs[16]) else vs[15]
    assert g.is_legal(Action(0, A.BUILD_SETTLEMENT, tgt))
    ev = g.step(Action(0, A.BUILD_SETTLEMENT, tgt))
    assert g.is_over and g.winner == 0 and g.phase == Phase.GAME_OVER
    assert ev[-1]["type"] == "game_won"
    assert g.legal_actions() == []
    with pytest.raises(IllegalAction):
        g.step(Action(0, A.END_TURN))
    for t in ALL_ACTION_TYPES:
        assert not g.is_legal(Action(0, t))


def test_hidden_vp_card_wins_on_own_turn_only():
    g = main_game()
    _nine_vp(g, 1)
    g.players[1].dev[VICTORY_POINT] = 1
    assert g.victory_points(1) == 10 and g.public_victory_points(1) == 9
    # p0's turn: p1 not declared winner on p0's actions
    give(g, 0, (4, 0, 0, 0, 0))
    g.step(Action(0, A.MARITIME_TRADE, (0, 1)))
    assert not g.is_over
    ev = g.step(Action(0, A.END_TURN))
    assert g.is_over and g.winner == 1
    assert {"type": "game_won", "player": 1, "victory_points": 10} in ev


def test_buying_vp_card_wins_immediately():
    g = main_game()
    _nine_vp(g, 0)
    g.dev_deck.append(VICTORY_POINT)
    give(g, 0, COST["dev"])
    g.step(Action(0, A.BUY_DEV))
    assert g.winner == 0


def test_truncation():
    g = main_game(max_turns=2)
    g.step(Action(0, A.END_TURN))
    assert not g.is_over
    g.phase, g.rolled = Phase.MAIN, True
    g.step(Action(1, A.END_TURN))
    assert g.is_over and g.truncated and g.winner == -1


# ---------------------------------------------------------- serialization
def _random_game(seed, steps, n=4):
    g = Game(config=GameConfig(num_players=n), seed=seed)
    rng = random.Random(seed)
    for _ in range(steps):
        if g.is_over:
            break
        g.step(rng.choice(g.legal_actions()))
    return g, rng


@pytest.mark.parametrize("seed", range(5))
def test_json_roundtrip(seed):
    g, rng = _random_game(seed, 400 + 97 * seed)
    d = json.loads(json.dumps(g.to_dict()))
    h = Game.from_dict(d)
    assert h.to_dict() == g.to_dict()
    assert h.legal_actions() == g.legal_actions()
    for _ in range(300):
        if g.is_over:
            break
        a = rng.choice(g.legal_actions())
        assert g.step(a) == h.step(Action.from_json(json.loads(json.dumps(a.to_json()))))
    assert h.to_dict() == g.to_dict()


def test_roundtrip_during_trade():
    g = main_game()
    give(g, 0, (1, 0, 0, 0, 0))
    give(g, 1, (0, 0, 0, 0, 1))
    g.step(Action(0, A.OFFER_TRADE, ((1, 0, 0, 0, 0), (0, 0, 0, 0, 1))))
    g.step(Action(1, A.ACCEPT_TRADE))
    h = Game.from_dict(json.loads(json.dumps(g.to_dict())))
    assert h.to_dict() == g.to_dict() and h.actor == 2


def test_replay_determinism():
    g, _ = _random_game(7, 3000)
    r = Game.replay(g.seed, g.config, [a.to_json() for a in g.action_log])
    assert r.to_dict() == g.to_dict()


def test_clone_independent():
    g, rng = _random_game(8, 500)
    c = g.clone()
    assert c.to_dict() == g.to_dict()
    snapshot = json.dumps(g.to_dict())
    for _ in range(300):
        if c.is_over:
            break
        c.step(rng.choice(c.legal_actions()))
    assert json.dumps(g.to_dict()) == snapshot


def test_player_view_hides_private_info():
    g, _ = _random_game(9, 600)
    while g.is_over:
        g, _ = _random_game(g.seed + 100, 600)
    for viewer in range(4):
        v = g.player_view(viewer)
        json.dumps(v)
        for q, info in enumerate(v["players"]):
            if q == viewer:
                assert info["hand"] == g.players[q].hand
                assert info["dev_cards"] == g.players[q].dev
            else:
                assert "hand" not in info and "dev_cards" not in info and "vp" not in info
                assert info["resource_count"] == sum(g.players[q].hand)
        assert "dev_deck" not in v and "rng_state" not in v
    spect = g.player_view(None)
    assert all("hand" not in p for p in spect["players"])


# ---------------------------------------------------------- malformed input
GARBAGE = [None, -1, 54, 72, 19, 999, 10**30, 1.0, 0.5, True, False, "0", "x", [], [0], (0,), {},
           (0, 1, 2), ((1, 0, 0, 0, 0),), ((True, 0, 0, 0, 0), (0, 0, 0, 0, 1)),
           ((1.0, 0, 0, 0, 0), (0, 0, 0, 0, 1)), ([1, 0, 0, 0, 0], [0, 0, 0, 0, 1]),
           (float("nan"), 0), (0, True), (True, 1), object()]


def _malformed_ok(g):
    actor = g.actor if not g.is_over else 0
    for t in list(ALL_ACTION_TYPES) + ["NOPE", None, 3]:
        for v in GARBAGE:
            for p in (actor, actor + 1, -1, True, "0", None):
                a = Action(p, t, v)
                ok = g.is_legal(a)
                assert isinstance(ok, bool)
                if ok:
                    assert a in g.legal_actions() or t in (A.OFFER_TRADE, A.COUNTER_OFFER), a
                    assert not isinstance(v, (bool, float)) and not isinstance(p, bool), a
    for bad in (None, (0, A.ROLL, None), "ROLL", {"player": 0}):
        assert g.is_legal(bad) is False
        with pytest.raises(IllegalAction):
            g.step(bad)


def test_is_legal_rejects_malformed_without_raising():
    g, rng = _random_game(10, 0)
    for _ in range(400):
        if g.is_over:
            break
        _malformed_ok(g)
        g.step(rng.choice(g.legal_actions()))


def test_bool_and_float_values_rejected():
    g = main_game()
    put_settlement(g, 0, 1)
    give(g, 0, (1, 1, 0, 2, 3))
    e = g.legal_actions()
    road = next(a for a in e if a.type == A.BUILD_ROAD)
    assert not g.is_legal(Action(0, A.BUILD_CITY, True))
    assert not g.is_legal(Action(0, A.BUILD_CITY, 1.0))
    assert not g.is_legal(Action(0, A.BUILD_ROAD, float(road.value)))
    assert not g.is_legal(Action(False, A.END_TURN))
    with pytest.raises(IllegalAction):
        g.step(Action(0, A.BUILD_CITY, True))
    g.players[0].dev[4] = 1
    assert not g.is_legal(Action(0, A.PLAY_MONOPOLY, False))
    g.players[0].dev[3] = 1
    assert not g.is_legal(Action(0, A.PLAY_YEAR_OF_PLENTY, (True, 1)))


def test_action_json_rejects_garbage():
    for d in (None, [], {"type": "NOPE", "player": 0}, {"type": "ROLL", "player": True},
              {"type": "ROLL", "player": "0"}, {"type": "ROLL"}):
        with pytest.raises(ValueError):
            Action.from_json(d)
