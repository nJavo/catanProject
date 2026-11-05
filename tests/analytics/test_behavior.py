import random

from catan.analytics import aggregate_rows, analyze_game, dangling_roads
from catan.bots.common import board_info
from catan.engine import A, Game, GameConfig, Phase
from catan.engine.board import TOPOLOGY as T

CFG = GameConfig(num_players=4, max_turns=300)


def play(seed, policy):
    """Play a game with ``policy(game, legal, rng) -> action``; return the action log."""
    g = Game(config=CFG, seed=seed)
    rng = random.Random(seed)
    while not g.is_over:
        g.step(policy(g, g.legal_actions(), rng))
    return g


def rand(g, legal, rng):
    return rng.choice(legal)


def test_invariants_random_games():
    for seed in range(3):
        g = play(seed, rand)
        rows = analyze_game(seed, CFG, g.action_log)
        assert sum(sum(r["actions"].values()) for r in rows) == len(g.action_log)
        for q, r in enumerate(rows):
            pl = g.players[q]
            assert r["vp"] == g.victory_points(q)
            assert r["vp"] == (r["vp_settlements"] + r["vp_cities"] + r["vp_longest_road"]
                               + r["vp_largest_army"] + r["vp_cards"])
            assert r["cities_built"] == len(pl.cities)
            assert 2 + r["settlements_built"] == len(pl.settlements) + len(pl.cities)
            assert r["roads_built"] + r["free_roads"] + 2 == len(pl.roads)
            assert r["won"] == (g.winner == q)
            assert r["missed_any"] <= r["end_turns"]
            assert 0 <= r["dangling_roads"] <= len(pl.roads)


def test_opening_metrics_best_spot():
    bi = None

    def greedy_setup(g, legal, rng):
        nonlocal bi
        bi = bi or board_info(g.layout)
        if g.phase == Phase.SETUP_SETTLEMENT:
            return max(legal, key=lambda a: (sum(bi.vertex_pips[a.value]), -a.value))
        return rng.choice(legal)

    g = play(5, greedy_setup)
    rows = analyze_game(5, CFG, g.action_log)
    for q, r in enumerate(rows):
        assert r["open_pick_vs_best"] == 1.0 and r["open_rank_frac"] == 0.0
        first, second = [a.value for a in g.action_log
                         if a.type == A.BUILD_SETTLEMENT and a.player == q][:2]
        assert r["open_pips_1"] == sum(bi.vertex_pips[first])
        assert r["open_pips_2"] == sum(bi.vertex_pips[second])
        res = {i for v in (first, second) for i in range(5) if bi.vertex_pips[v][i]}
        assert r["open_resource_count"] == len(res)
        assert len(r["open_missing"]) == 5 - len(res)


def test_robber_self_block_detected():
    def self_robber(g, legal, rng):
        if g.phase == Phase.MOVE_ROBBER and g.actor == 0:
            own = [a for a in legal if any(g.vertex_owner[v] == 0 for v in T.hex_vertices[a.value])]
            return (own or legal)[0]
        return rng.choice(legal)

    g = play(11, self_robber)
    r = analyze_game(11, CFG, g.action_log)[0]
    moves = sum(1 for a in g.action_log if a.type == A.MOVE_ROBBER and a.player == 0)
    assert moves > 0 and r["robber_moves"] == moves
    assert r["robber_self_block"] == moves


def test_hoarder_misses_builds_and_discards():
    def hoard(g, legal, rng):
        types = {a.type: a for a in legal}
        if g.actor == 0:
            for t in (A.ROLL, A.END_TURN):
                if t in types:
                    return types[t]
        return rng.choice(legal)

    g = play(3, hoard)
    r = analyze_game(3, CFG, g.action_log)[0]
    assert r["settlements_built"] == r["cities_built"] == r["roads_built"] == r["dev_bought"] == 0
    assert r["missed_any"] > 0 and r["end_turn_afford_settlement"] > 0
    assert r["discarded_cards"] > 0 and r["hand_at_7_mean"] > 4
    assert r["offers"] == 0 and r["maritime"] == 0


def test_dangling_roads_crafted():
    g = Game(config=CFG, seed=0)
    pl = g.players[0]
    v0 = 0
    e1 = T.vertex_edges[v0][0]
    a, b = T.edge_vertices[e1]
    far = b if a == v0 else a
    e2 = next(e for e in T.vertex_edges[far] if e != e1)
    pl.settlements = [v0]
    g.vertex_owner[v0] = 0
    pl.roads = [e1, e2]
    assert dangling_roads(g, 0) == 2      # branch leads nowhere
    c, d = T.edge_vertices[e2]
    tip = d if c == far else c
    pl.settlements.append(tip)
    assert dangling_roads(g, 0) == 0      # now it connects two buildings


def test_aggregate_ratios_pool():
    rows = [{"seat": 0, "vp": 4, "robber_moves": 2, "robber_hits_leader": 1, "end_turns": 10,
             "missed_vp_build": 1, "actions": {"MAIN:END_TURN": 10, "MAIN:BUILD_ROAD": 10}},
            {"seat": 1, "vp": 6, "robber_moves": 2, "robber_hits_leader": 2, "end_turns": 10,
             "missed_vp_build": 0, "actions": {"MAIN:END_TURN": 10}}]
    agg = aggregate_rows(rows)
    assert agg["metrics"]["vp"]["mean"] == 5
    assert agg["ratios"]["robber_hits_leader_rate"]["rate"] == 0.75
    assert agg["ratios"]["missed_vp_build_rate"]["rate"] == 0.05
    assert agg["actions_per_game"]["MAIN"]["END_TURN"] == 10
    assert agg["action_share_by_phase"]["MAIN"]["BUILD_ROAD"] == 0.3333


def test_settlements_built_excludes_setup_settlements():
    """DS16-07: settlements_built counts paid builds only; the 2 free setup settlements
    show up in final_settlements / vp_settlements instead."""
    def hoard(g, legal, rng):
        types = {a.type: a for a in legal}
        if g.actor == 0 and g.phase not in (Phase.SETUP_SETTLEMENT, Phase.SETUP_ROAD):
            for t in (A.ROLL, A.END_TURN):
                if t in types:
                    return types[t]
        return rng.choice(legal)

    g = play(3, hoard)
    r = analyze_game(3, CFG, g.action_log)[0]
    assert r["settlements_built"] == 0
    assert r["final_settlements"] == 2 and r["vp_settlements"] == 2
    for q, row in enumerate(analyze_game(3, CFG, g.action_log)):
        assert row["final_settlements"] + row["final_cities"] == 2 + row["settlements_built"]
