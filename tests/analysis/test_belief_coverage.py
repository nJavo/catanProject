"""AN-005 acceptance: belief-weighted determinization over 50 replayed games.

1. Samples never contradict a public event: the true hands are always in the
   posterior's support, every analysis determinization drawn with the belief
   takes its hands from that support, and each one matches the public hand
   sizes, per-resource totals and the viewer's own hand.
2. On game-sourced positions the average log-likelihood of the true hands is
   higher under the posterior than under uniform sampling (catan.analysis
   determinize without a belief: the unseen pool shuffled and dealt by hand size).
"""
import math
import random

from catan.agents.registry import make_agent
from catan.analysis import determinize, position_from_game
from catan.bots import common as C
from catan.bots.belief import HandPosterior, visible
from catan.engine import Game, GameConfig

LINEUPS = (["intermediate", "basic", "intermediate_trader", "random"], ["basic", "random", "intermediate", "basic"])


def _games(n_games, max_turns=150):
    """(game, viewer, posterior) after every step; the viewer rotates with the seed."""
    for seed in range(n_games):
        lineup = LINEUPS[seed % 2]
        ags = [make_agent(x, seed=seed * 4 + i) for i, x in enumerate(lineup)]
        for i, a in enumerate(ags):
            a.reset(i, seed * 4 + i)
        g = Game(config=GameConfig(max_turns=max_turns), seed=1000 + seed)
        me = seed % 4
        post = HandPosterior(g.n, me)
        while not g.is_over:
            ev = g.step(ags[g.actor].act(g, g.legal_actions()))
            post.observe(g, visible(ev, me))
            yield g, me, post


def _uniform_loglik(g, me, opps) -> float:
    """log P(true opponent hands) when the unseen pool is shuffled and dealt by hand size:
    multivariate hypergeometric = prod_r [tot_r! / prod_q h_qr!] / [N! / prod_q n_q!]."""
    tot = C.opponents_resource_totals(g, me)
    sizes = [C.hand_count(g, q) for q in opps]
    lg = math.lgamma
    num = sum(lg(t + 1) for t in tot) - sum(lg(g.players[q].hand[r] + 1) for q in opps for r in range(5))
    den = lg(sum(tot) + 1) - sum(lg(n + 1) for n in sizes)
    return num - den


def test_belief_support_and_samples_over_50_games():
    rng = random.Random(0)
    steps = samples = 0
    for g, me, post in _games(50):
        assert not post.lost
        true = tuple(tuple(g.players[q].hand) for q in post.opps)
        assert true in post.states  # the true hands never contradict the public record
        steps += 1
        if g.is_over or len(g.action_log) % 25 or g.actor != me:
            continue
        spec = position_from_game(g, me)
        cols = C.opponents_resource_totals(g, me)
        for d in determinize(spec, 3, rng.randrange(10**6), belief=post):
            hands = tuple(tuple(d.players[q].hand) for q in post.opps)
            assert hands in post.states  # drawn from the posterior, not the uniform fallback
            assert d.players[me].hand == g.players[me].hand
            assert [sum(h) for h in hands] == [C.hand_count(g, q) for q in post.opps]
            assert [sum(h[r] for h in hands) for r in range(5)] == cols
            samples += 1
    assert steps > 10_000 and samples > 100


def test_belief_loglik_beats_uniform_on_game_positions():
    bel, uni, n = 0.0, 0.0, 0
    for g, me, post in _games(50):
        if g.is_over or len(g.action_log) % 20 or g.turn < 8:
            continue
        true = tuple(tuple(g.players[q].hand) for q in post.opps)
        bel += math.log(post.states[true])
        uni += _uniform_loglik(g, me, post.opps)
        n += 1
    assert n > 200
    mean_bel, mean_uni = bel / n, uni / n
    print(f"mean log-likelihood of the true hands over {n} positions: belief {mean_bel:.3f}, uniform {mean_uni:.3f}")
    assert mean_bel > mean_uni
