import random

import numpy as np
import pytest

from catan.engine import Game, GameConfig, Phase
from catan.rl.spaces import (OBS_DIM, ACT_DIM, ACTIONS, encode_obs, action_mask, encode_action,
                             decode_action, static_features)


def _walk(seed, n_players, max_turns=400):
    """Yield (game, legal) at every decision of a random game."""
    g = Game(config=GameConfig(num_players=n_players, max_turns=max_turns), seed=seed)
    rng = random.Random(seed)
    while not g.is_over:
        legal = g.legal_actions()
        yield g, legal
        g.step(rng.choice(legal))


def test_action_space_layout():
    assert ACT_DIM == 72 + 54 + 54 + 19 + 3 + 5 + 15 + 5 + 20 + 60 + 60 + 2 + 3 + 1 + 5 == 378
    assert len(set(ACTIONS)) == ACT_DIM


@pytest.mark.parametrize("n_players", [2, 3, 4])
def test_mask_equals_legal_and_bijection(n_players):
    phases = set()
    for seed in range(12):
        for g, legal in _walk(seed, n_players):
            me = g.actor
            assert len(set(legal)) == len(legal)
            m = action_mask(legal, me, g.n)
            assert m.dtype == bool and m.shape == (ACT_DIM,)
            assert m.sum() == len(legal)
            for a in legal:
                i = encode_action(a, me, g.n)
                assert m[i] and decode_action(i, me, g.n) == a
            assert {decode_action(int(i), me, g.n) for i in np.nonzero(m)[0]} == set(legal)
            phases.add(g.phase)
    assert {Phase.DISCARD, Phase.TRADE_RESPONSE, Phase.TRADE_CONFIRM, Phase.MOVE_ROBBER,
            Phase.MAIN, Phase.PRE_ROLL, Phase.SETUP_ROAD} <= phases


def test_obs_shape_dtype_stable_and_finite():
    for seed in range(4):
        st = None
        for g, _ in _walk(seed, 4):
            st = static_features(g.layout) if st is None else st
            o = encode_obs(g, g.actor, st)
            assert o.shape == (OBS_DIM,) and o.dtype == np.float32
            assert np.isfinite(o).all() and np.abs(o).max() < 50
            assert np.array_equal(o, encode_obs(g, g.actor))  # cache == fresh


def _scramble_hidden(g, me, rng):
    """Change opponents' hand composition, dev composition, deck order and RNG
    while keeping every public count identical."""
    h = g.clone()
    for q, pl in enumerate(h.players):
        if q == me:
            continue
        n = sum(pl.hand)
        hand = [0] * 5
        for _ in range(n):
            hand[rng.randrange(5)] += 1
        pl.hand = hand
        # dev composition: keep total and the per-type "bought this turn" split
        # is hidden too; keep totals only.
        nd = sum(pl.dev)
        dev = [0] * 5
        for _ in range(nd):
            dev[rng.randrange(5)] += 1
        pl.dev = dev
        pl.dev_new = [0] * 5
    rng.shuffle(h.dev_deck)
    h.rng.seed(rng.random())
    return h


def test_no_hidden_info_leak():
    rng = random.Random(0)
    checked = 0
    for seed in range(6):
        for g, _ in _walk(seed, 4):
            if rng.random() > 0.05:
                continue
            for me in range(g.n):
                h = _scramble_hidden(g, me, rng)
                # bank depends on total hands only, so it is unchanged
                assert np.array_equal(encode_obs(g, me), encode_obs(h, me)), (g.phase, me)
                checked += 1
    assert checked > 200


def test_obs_changes_with_own_hand():
    g = Game(seed=3)
    rng = random.Random(3)
    while g.phase != Phase.MAIN:
        g.step(rng.choice(g.legal_actions()))
    me = g.actor
    h = g.clone()
    h.players[me].hand[0] += 1
    assert not np.array_equal(encode_obs(g, me), encode_obs(h, me))


def test_seat_rotation_me_is_index_zero():
    # The same position seen by seat q in a game equals "rotated" ownership.
    g = Game(seed=5)
    rng = random.Random(5)
    for _ in range(40):
        g.step(rng.choice(g.legal_actions()))
    from catan.rl.spaces import OFFSETS, EDGE_F, NE
    a, b = OFFSETS["edge"]
    for me in range(4):
        e = encode_obs(g, me)[a:b].reshape(NE, EDGE_F)
        for edge, owner in enumerate(g.edge_owner):
            if owner >= 0:
                assert e[edge, (owner - me) % 4] == 1
        own = [x for x, o in enumerate(g.edge_owner) if o == me]
        assert all(e[x, 0] == 1 for x in own)
