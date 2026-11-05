"""Fast encode_obs / action_mask vs. the original implementations."""
import random

import numpy as np
import pytest

from catan.engine import Game, GameConfig
from catan.rl.spaces import (OBS_DIM, MAXP, HEX_F, VDYN_F, EDGE_F, PUB_F, NV, NE, PORT_KINDS, PHASES,
                             PHASE_INDEX, OFFSETS, T, encode_obs, action_mask, encode_action,
                             decode_action, static_features)

_ADJ = np.zeros((NV, NV), dtype=np.int8)
for _v, _ns in enumerate(T.vertex_neighbors):
    _ADJ[_v, list(_ns)] = 1



def ref_encode_obs(g, me, static=None):
    """encode_obs before the fast-path rewrite, verbatim."""
    n = g.n
    out = np.zeros(OBS_DIM, dtype=np.float32)
    if static is None:
        static = static_features(g.layout)
    out[:len(static)] = static
    a, _ = OFFSETS["hex"]
    out[a + g.robber * HEX_F + HEX_F - 1] = 1

    rel_of = [(q - me) % n for q in range(n)]
    vo = np.asarray(g.vertex_owner)
    lv = np.asarray(g.vertex_level)
    occ = vo >= 0
    a, b = OFFSETS["vertex_dyn"]
    vd = out[a:b].reshape(NV, VDYN_F)
    idx = np.nonzero(occ)[0]
    if len(idx):
        rel = (vo[idx] - me) % n
        vd[idx, rel + MAXP * (lv[idx] - 1)] = 1
    vd[:, -1] = ~(occ | (_ADJ @ occ > 0))

    eo = np.asarray(g.edge_owner)
    idx = np.nonzero(eo >= 0)[0]
    a, b = OFFSETS["edge"]
    if len(idx):
        out[a:b].reshape(NE, EDGE_F)[idx, (eo[idx] - me) % n] = 1

    a, b = OFFSETS["players"]
    pf = out[a:b].reshape(MAXP, PUB_F)
    actor = g.actor if not g.is_over else -1
    for q, pl in enumerate(g.players):
        f = pf[rel_of[q]]
        f[0] = 1
        f[1] = sum(pl.hand) / 10
        f[2] = sum(pl.dev) / 5
        f[3] = pl.knights / 5
        f[4] = g.public_victory_points(q) / 10
        f[5] = g.road_length[q] / 15
        f[6] = g.longest_road_owner == q
        f[7] = g.largest_army_owner == q
        f[8] = (5 - len(pl.settlements)) / 5
        f[9] = (4 - len(pl.cities)) / 4
        f[10] = (15 - len(pl.roads)) / 15
        for k in pl.ports:
            f[11 + PORT_KINDS.index(k)] = 1
        f[17] = g.discard_remaining[q] / 10
        f[18] = g.current == q
        f[19] = actor == q

    a, _ = OFFSETS["own"]
    pl = g.players[me]
    out[a:a + 5] = np.asarray(pl.hand, dtype=np.float32) / 10
    out[a + 5:a + 10] = pl.dev
    out[a + 10:a + 15] = pl.dev_new
    out[a + 15] = g.victory_points(me) / 10

    a, _ = OFFSETS["global"]
    out[a:a + 5] = np.asarray(g.bank, dtype=np.float32) / 19
    out[a + 5] = len(g.dev_deck) / 25
    o = a + 6
    ph = PHASE_INDEX.get(g.phase)
    if ph is not None:
        out[o + ph] = 1
    o += len(PHASES)
    out[o] = g.rolled
    out[o + 1] = g.dev_played
    o += 2
    if g.last_roll:
        out[o + sum(g.last_roll) - 2] = 1
    o += 11
    out[o] = g.free_roads / 2
    out[o + 1] = g.trades_this_turn / 3
    out[o + 2] = g.turn / 200

    tr = g.trade
    if tr is not None:
        a, _ = OFFSETS["trade"]
        out[a] = 1
        out[a + 1 + rel_of[tr["proposer"]]] = 1
        out[a + 1 + MAXP:a + 6 + MAXP] = tr["give"]
        out[a + 6 + MAXP:a + 11 + MAXP] = tr["get"]
        base = a + 11 + MAXP
        for q in range(n):
            f = out[base + rel_of[q] * 14: base + rel_of[q] * 14 + 14]
            if q in tr["responses"]:
                r = tr["responses"][q]
                if r is None:
                    f[1] = 1
                elif r == (tr["give"], tr["get"]):
                    f[2] = 1
                else:
                    f[3] = 1
                    f[4:9] = r[0]
                    f[9:14] = r[1]
            elif q in tr["pending"]:
                f[0] = 1
    return out


def ref_action_mask(legal, me, n):
    m = np.zeros(len(action_mask([], me, n)), dtype=bool)
    m[[encode_action(a, me, n) for a in legal]] = True
    return m


@pytest.mark.parametrize("n_players", [2, 3, 4])
def test_fast_encoding_matches_reference(n_players):
    checked = 0
    for seed in range(6):
        g = Game(config=GameConfig(num_players=n_players, max_turns=300), seed=seed)
        rng = random.Random(seed)
        st = static_features(g.layout)
        while not g.is_over:
            legal = g.legal_actions()
            for me in range(g.n):
                assert np.array_equal(encode_obs(g, me, st), ref_encode_obs(g, me, st))
            want = ref_action_mask(legal, g.actor, g.n)
            assert np.array_equal(action_mask(legal, g.actor, g.n), want)
            # actions not from legal_actions() (no prebuilt identity) take the slow path
            dec = [decode_action(encode_action(a, g.actor, g.n), g.actor, g.n) for a in legal]
            assert np.array_equal(action_mask(dec, g.actor, g.n), want)
            checked += 1
            g.step(rng.choice(legal))
    assert checked > 1000
