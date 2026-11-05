"""EN-008: Game.observe_for(seat) exposes public state + own cards and raises on hidden state."""
import random

import pytest

from catan.engine import A, Action, Game, GameConfig, Phase
from catan.engine.actions import ALL_ACTION_TYPES
from catan.engine.infoset import HiddenInformationError, InfoSetGame

from .test_oracle import CAND_VALUES, _positions


def _mid_game(seed=3, steps=600):
    g = Game(config=GameConfig(max_turns=300), seed=seed)
    rng = random.Random(seed)
    for _ in range(steps):
        if g.is_over:
            break
        g.step(rng.choice(g.legal_actions()))
    return g


def test_hidden_state_raises():
    g = _mid_game()
    me = g.actor
    v = g.observe_for(me)
    opp = (me + 1) % g.n
    for read in (lambda: v.players[opp].hand, lambda: v.players[opp].dev, lambda: v.players[opp].dev_new,
                 lambda: list(v.dev_deck), lambda: v.dev_deck[0], lambda: v.dev_deck.pop(),
                 lambda: v.rng.random(), lambda: v.rng.getstate(), lambda: v.seed,
                 lambda: v.victory_points(opp), lambda: v.to_dict(), lambda: v.unseen_dev_pool(opp),
                 lambda: v.step(v.legal_actions()[0])):
        with pytest.raises(HiddenInformationError):
            read()
    # public and own information stays readable
    assert v.players[me].hand == g.players[me].hand and v.players[me].dev == g.players[me].dev
    assert len(v.dev_deck) == len(g.dev_deck) and bool(v.dev_deck) == bool(g.dev_deck)
    for q in range(g.n):
        assert v.hand_count(q) == g.hand_count(q) and v.dev_count(q) == g.dev_count(q)
        assert v.public_victory_points(q) == g.public_victory_points(q)
        assert v.players[q].settlements == g.players[q].settlements and v.players[q].knights == g.players[q].knights
        assert v.longest_road(q) == g.longest_road(q) and v.supply(q, "road") == g.supply(q, "road")
    assert v.victory_points(me) == g.victory_points(me)
    assert v.player_view(me) == g.player_view(me)
    assert v.action_log == g.action_log


def test_view_answers_like_the_game_for_its_seat():
    n = 0
    for g, i in _positions(range(8), steps=1500):
        if i % 7:
            continue
        p = g.actor
        v = g.observe_for(p)
        legal = g.legal_actions()
        assert v.legal_actions() == legal and v.actor == p and v.phase == g.phase
        if i % 21 == 0:
            for t in ALL_ACTION_TYPES:
                for val in CAND_VALUES[::3]:
                    a = Action(p, t, val)
                    assert v.is_legal(a) == g.is_legal(a), a
        n += 1
    assert n > 1000


def test_mutating_the_view_never_touches_the_game():
    g = _mid_game(5)
    before = g.to_dict()
    v = g.observe_for(g.actor)
    v.players[g.actor].hand[0] += 5
    v.vertex_owner[0] = 3
    v.bank[1] = 0
    v.config.vp_to_win = 3
    v._hexes_by_number.clear()
    v.action_log.append(Action(0, A.END_TURN))
    c = v.clone()
    c.step(c.legal_actions()[0])
    assert g.to_dict() == before


def test_clone_is_a_determinization_that_keeps_public_facts():
    g = _mid_game(7, 900)
    me = g.actor
    v = g.observe_for(me)
    c = v.clone()
    assert type(c) is Game and not isinstance(c, InfoSetGame)
    assert c.players[me].hand == g.players[me].hand and c.players[me].dev == g.players[me].dev
    for q in range(g.n):
        assert sum(c.players[q].hand) == g.hand_count(q) and sum(c.players[q].dev) == g.dev_count(q)
    for r in range(5):
        assert sum(pl.hand[r] for pl in c.players) == sum(pl.hand[r] for pl in g.players)  # conservation
    assert sorted(c.dev_deck + [t for q in range(g.n) if q != me for t in range(5)
                                for _ in range(c.players[q].dev[t])]) == g.unseen_dev_pool(me)
    assert c.seed is None and c.action_log == g.action_log
    c.to_dict()
    while not c.is_over and c.turn < g.turn + 3:  # a determinization is a playable game
        c.step(random.Random(1).choice(c.legal_actions()))


def test_view_determinization_equals_the_real_one():
    """Same RNG, same sampling: the view supplies exactly the public inputs the
    full-game determinization reads (so search decisions don't change)."""
    for seed in range(4):
        g = _mid_game(seed, 300 + 200 * seed)
        if g.is_over:
            continue
        me = g.actor
        a = g.determinized(me, random.Random(seed)).to_dict()
        b = g.observe_for(me).determinized(me, random.Random(seed)).to_dict()
        a.pop("seed"), b.pop("seed")
        assert a == b


def test_game_over_view_does_not_reveal_hands():
    g = Game(config=GameConfig(max_turns=40), seed=1)
    rng = random.Random(1)
    while not g.is_over:
        g.step(rng.choice(g.legal_actions()))
    assert g.phase == Phase.GAME_OVER
    pv = g.observe_for(0).player_view(0)
    assert "hand" in pv["players"][0] and all("hand" not in p for p in pv["players"][1:])
