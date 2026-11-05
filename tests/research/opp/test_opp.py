"""D21 opponent features: setup policy match and public-only inputs."""
import numpy as np

from catan.agents.registry import make_agent
from catan.engine import Game
from catan.research.opp.agent import _BySeat
from catan.research.opp.features import FEATURE_NAMES, seat_features


def _play(names, seed, turns=12):
    g = Game(4, seed=seed)
    ags = [make_agent(n) for n in names]
    for i, a in enumerate(ags):
        a.reset(i, seed)
    while not g.is_over and g.turn < turns:
        g.step(ags[g.actor].act(g, g.legal_actions()))
    return g


def test_setup_match_flags_identify_basic_and_intermediate():
    g = _play(["basic", "intermediate", "basic", "intermediate"], 3)
    f = seat_features(g.layout, g.config, g.action_log)
    s1b, s1i = FEATURE_NAMES.index("setup_s1_basic"), FEATURE_NAMES.index("setup_s1_int")
    s2b, s2i = FEATURE_NAMES.index("setup_s2_basic"), FEATURE_NAMES.index("setup_s2_int")
    for q in (0, 2):
        assert f[q, s1b] == 1 and f[q, s2b] == 1
    for q in (1, 3):
        assert f[q, s1i] == 1 and f[q, s2i] == 1
    assert (f[:, FEATURE_NAMES.index("setup_seen")] == 1).all()
    assert f[0, FEATURE_NAMES.index("offers")] == 0


def test_features_are_the_same_from_an_infoset_view():
    g = _play(["basic", "intermediate_trader", "intermediate", "basic"], 7)
    v = g.observe_for(2)
    assert np.array_equal(seat_features(g.layout, g.config, g.action_log),
                          seat_features(v.layout, v.config, v.action_log))


def test_by_seat_dispatch():
    class P:
        def __init__(self, k):
            self.k = k

        def act(self, c, legal):
            return self.k

    class G:
        actor = 2
    assert _BySeat(P("d"), {2: P("x")}).act(G, []) == "x"
    assert _BySeat(P("d"), {1: P("x")}).act(G, []) == "d"
