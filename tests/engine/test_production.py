from catan.engine import TOPOLOGY as T

from .helpers import main_game, put_settlement, resource_hex, res_index, check_conservation


def _setup(seed=0):
    g = main_game(seed=seed)
    h = resource_hex(g)
    return g, h, g.layout.hex_number[h], res_index(g, h)


def test_settlement_one_city_two():
    g, h, num, r = _setup()
    v1, v3 = T.hex_vertices[h][0], T.hex_vertices[h][2]
    put_settlement(g, 1, v1)
    put_settlement(g, 2, v3, city=True)
    ev = []
    g._produce(num, ev)
    assert g.players[1].hand[r] >= 1 and g.players[2].hand[r] >= 2
    # exact amounts from this hex only (other hexes may share the number)
    expected = {1: 0, 2: 0}
    for hh in range(19):
        if g.layout.hex_number[hh] == num and hh != g.robber:
            for v, q, lvl in ((v1, 1, 1), (v3, 2, 2)):
                if v in T.hex_vertices[hh]:
                    expected[q] += lvl
    assert g.players[1].hand[r] == expected[1]
    assert g.players[2].hand[r] == expected[2]
    check_conservation(g)
    assert {e["player"] for e in ev if e["type"] == "resources_gained"} == {1, 2}


def test_robber_blocks_production():
    g, h, num, r = _setup()
    v = T.hex_vertices[h][0]
    put_settlement(g, 1, v, city=True)
    free = g.clone()
    g.robber = h
    free.robber = (h + 1) % 19 if (h + 1) % 19 != h else 0
    while free.robber == h or free.layout.hex_number[free.robber] == num:
        free.robber = (free.robber + 1) % 19
    g._produce(num, [])
    free._produce(num, [])
    assert free.players[1].hand[r] - g.players[1].hand[r] == 2
    check_conservation(g)


def test_bank_shortage_single_claimant_gets_remainder():
    g, h, num, r = _setup()
    put_settlement(g, 1, T.hex_vertices[h][0], city=True)
    g.bank[r] = 1
    g.players[3].hand[r] = 18
    g._produce(num, [])
    assert g.bank[r] == 0 and g.players[1].hand[r] == 1
    check_conservation(g)


def test_bank_shortage_multiple_claimants_nobody_gets():
    g, h, num, r = _setup()
    put_settlement(g, 1, T.hex_vertices[h][0], city=True)
    put_settlement(g, 2, T.hex_vertices[h][3], city=True)
    g.bank[r] = 3
    g.players[3].hand[r] = 16
    g._produce(num, [])
    assert g.bank[r] == 3 and g.players[1].hand[r] == 0 and g.players[2].hand[r] == 0
    check_conservation(g)


def test_bank_exact_amount_pays_all():
    g, h, num, r = _setup()
    put_settlement(g, 1, T.hex_vertices[h][0])
    put_settlement(g, 2, T.hex_vertices[h][3])
    owed = sum(1 for hh in range(19) if g.layout.hex_number[hh] == num and hh != g.robber
               for v in (T.hex_vertices[h][0], T.hex_vertices[h][3]) if v in T.hex_vertices[hh]
               and g.layout.hex_terrain[hh] == g.layout.hex_terrain[h])
    g.players[3].hand[r] = 19 - owed
    g.bank[r] = owed
    g._produce(num, [])
    assert g.bank[r] == 0
    check_conservation(g)


def test_roll_seven_produces_nothing_and_roll_goes_main():
    from catan.engine import Phase, A, Action
    from .helpers import pre_roll_game
    for seed in range(40):
        g = pre_roll_game(seed=seed)
        ev = g.step(Action(0, A.ROLL))
        tot = ev[0]["total"]
        assert 2 <= tot <= 12 and ev[0]["dice"][0] + ev[0]["dice"][1] == tot
        assert g.rolled
        if tot == 7:
            assert g.phase == Phase.MOVE_ROBBER  # no one has cards
            assert not any(e["type"] == "resources_gained" for e in ev)
        else:
            assert g.phase == Phase.MAIN
