
from catan.engine import A, Action, Phase, TOPOLOGY as T

from .helpers import main_game, put_settlement, give, check_conservation

Z = (0, 0, 0, 0, 0)


def vec(**kw):
    names = ("wood", "brick", "sheep", "wheat", "ore")
    return tuple(kw.get(n, 0) for n in names)


def _port_vertex(g, kind):
    i = g.layout.port_types.index(kind)
    return T.edge_vertices[g.layout.port_edges[i]][0]


# ------------------------------------------------------------- maritime
def test_maritime_4_to_1():
    g = main_game()
    give(g, 0, (4, 0, 0, 0, 0))
    assert g.trade_rate(0, 0) == 4
    assert not g.is_legal(Action(0, A.MARITIME_TRADE, (0, 0)))
    g.step(Action(0, A.MARITIME_TRADE, (0, 3)))
    assert g.players[0].hand == [0, 0, 0, 1, 0]
    check_conservation(g)
    give(g, 0, (3, 0, 0, 0, 0))
    assert not g.is_legal(Action(0, A.MARITIME_TRADE, (0, 3)))


def test_maritime_3_to_1_generic_port():
    g = main_game()
    put_settlement(g, 0, _port_vertex(g, "3:1"))
    give(g, 0, (0, 3, 0, 0, 0))
    assert g.trade_rate(0, 1) == 3
    g.step(Action(0, A.MARITIME_TRADE, (1, 4)))
    assert g.players[0].hand == [0, 0, 0, 0, 1]


def test_maritime_2_to_1_specific_port():
    g = main_game()
    put_settlement(g, 0, _port_vertex(g, "sheep"))
    put_settlement(g, 0, _port_vertex(g, "3:1"))
    assert g.trade_rate(0, 2) == 2
    assert g.trade_rate(0, 0) == 3
    give(g, 0, (0, 0, 2, 0, 0))
    g.step(Action(0, A.MARITIME_TRADE, (2, 0)))
    assert g.players[0].hand == [1, 0, 0, 0, 0]


def test_maritime_needs_bank_stock():
    g = main_game()
    give(g, 0, (4, 0, 0, 0, 0))
    g.players[1].hand[4] = g.bank[4]
    g.bank[4] = 0
    assert not g.is_legal(Action(0, A.MARITIME_TRADE, (0, 4)))
    assert g.is_legal(Action(0, A.MARITIME_TRADE, (0, 3)))


# ------------------------------------------------------------- domestic
def _offer(g, give_v, get_v, p=0):
    g.step(Action(p, A.OFFER_TRADE, (give_v, get_v)))


def test_offer_validation():
    g = main_game()
    give(g, 0, (2, 0, 0, 0, 0))
    ok = (vec(wood=1), vec(ore=1))
    assert g.is_legal(Action(0, A.OFFER_TRADE, ok))
    assert not g.is_legal(Action(0, A.OFFER_TRADE, (vec(wood=1), Z)))          # gift
    assert not g.is_legal(Action(0, A.OFFER_TRADE, (Z, vec(ore=1))))           # ask for gift
    assert not g.is_legal(Action(0, A.OFFER_TRADE, (vec(wood=1), vec(wood=1, ore=1))))  # like-for-like
    assert not g.is_legal(Action(0, A.OFFER_TRADE, (vec(wood=3), vec(ore=1))))  # can't pay
    assert not g.is_legal(Action(0, A.OFFER_TRADE, (vec(wood=1), (0, 0, 0, 0, -1))))
    assert not g.is_legal(Action(0, A.OFFER_TRADE, ((1, 0, 0, 0), vec(ore=1))))
    # arbitrary (non-template) bundle is accepted
    assert g.is_legal(Action(0, A.OFFER_TRADE, (vec(wood=2), vec(ore=1, wheat=2))))


def test_offer_only_after_roll():
    g = main_game()
    g.phase, g.rolled = Phase.PRE_ROLL, False
    give(g, 0, (1, 0, 0, 0, 0))
    assert not g.is_legal(Action(0, A.OFFER_TRADE, (vec(wood=1), vec(ore=1))))


def test_full_protocol_accept_and_confirm():
    g = main_game()
    give(g, 0, (1, 0, 0, 0, 0))
    give(g, 1, (0, 0, 0, 0, 1))
    give(g, 3, (0, 0, 0, 0, 2))
    _offer(g, vec(wood=1), vec(ore=1))
    assert g.phase == Phase.TRADE_RESPONSE
    order = []
    # responders in order from proposer's left; p2 can't pay so cannot accept
    assert g.actor == 1
    order.append(g.actor); g.step(Action(1, A.ACCEPT_TRADE))
    assert g.actor == 2
    assert not g.is_legal(Action(2, A.ACCEPT_TRADE))
    order.append(g.actor); g.step(Action(2, A.REJECT_TRADE))
    order.append(g.actor); g.step(Action(3, A.ACCEPT_TRADE))
    assert order == [1, 2, 3]
    assert g.phase == Phase.TRADE_CONFIRM and g.actor == 0
    legal = set(g.legal_actions())
    assert legal == {Action(0, A.CANCEL_TRADE), Action(0, A.CONFIRM_TRADE, 1), Action(0, A.CONFIRM_TRADE, 3)}
    assert not g.is_legal(Action(0, A.CONFIRM_TRADE, 2))
    g.step(Action(0, A.CONFIRM_TRADE, 3))
    assert g.players[0].hand == [0, 0, 0, 0, 1]
    assert g.players[3].hand == [1, 0, 0, 0, 1]
    assert g.players[1].hand == [0, 0, 0, 0, 1]
    assert g.phase == Phase.MAIN and g.trade is None
    check_conservation(g)


def test_counter_offer_and_confirm_counter():
    g = main_game()
    give(g, 0, (2, 0, 0, 0, 0))
    give(g, 1, (0, 0, 0, 0, 1))
    _offer(g, vec(wood=1), vec(ore=1))
    counter = (vec(wood=2), vec(ore=1))
    assert not g.is_legal(Action(1, A.COUNTER_OFFER, (vec(wood=1), vec(ore=1))))  # same as offer
    # A counter the proposer can't pay is legal to make (checking it would leak
    # the proposer's hand) but can't be confirmed.
    assert g.is_legal(Action(1, A.COUNTER_OFFER, (vec(wood=3), vec(ore=1))))
    assert not g.is_legal(Action(1, A.COUNTER_OFFER, (vec(wood=1), vec(ore=2))))  # responder can't pay
    g.step(Action(1, A.COUNTER_OFFER, counter))
    g.step(Action(2, A.REJECT_TRADE))
    g.step(Action(3, A.REJECT_TRADE))
    assert g.phase == Phase.TRADE_CONFIRM
    g.step(Action(0, A.CONFIRM_TRADE, 1))
    assert g.players[0].hand == [0, 0, 0, 0, 1] and g.players[1].hand == [2, 0, 0, 0, 0]
    check_conservation(g)


def test_unaffordable_counter_not_confirmable_and_no_leak():
    g = main_game()
    give(g, 0, (1, 0, 0, 0, 0))
    give(g, 1, (0, 0, 0, 0, 1))
    _offer(g, vec(wood=1), vec(ore=1))
    before = {a for a in g.legal_actions() if a.type == A.COUNTER_OFFER}
    g.players[0].hand[0] = 3  # proposer's hidden hand changes...
    after = {a for a in g.legal_actions() if a.type == A.COUNTER_OFFER}
    assert before == after  # ...responder's options don't reveal it
    g.players[0].hand[0] = 1
    g.step(Action(1, A.COUNTER_OFFER, (vec(wood=3), vec(ore=1))))
    g.step(Action(2, A.REJECT_TRADE))
    g.step(Action(3, A.REJECT_TRADE))
    assert g.phase == Phase.TRADE_CONFIRM
    assert not g.is_legal(Action(0, A.CONFIRM_TRADE, 1))
    assert g.legal_actions() == [Action(0, A.CANCEL_TRADE)]


def test_no_takers_auto_cancel():
    g = main_game()
    give(g, 0, (1, 0, 0, 0, 0))
    _offer(g, vec(wood=1), vec(ore=1))
    ev = []
    for p in (1, 2, 3):
        ev = g.step(Action(p, A.REJECT_TRADE))
    assert g.phase == Phase.MAIN and g.trade is None
    assert ev[-1]["type"] == "trade_cancelled" and ev[-1]["reason"] == "no_takers"
    assert g.players[0].hand == [1, 0, 0, 0, 0]


def test_proposer_cancel():
    g = main_game()
    give(g, 0, (1, 0, 0, 0, 0))
    give(g, 1, (0, 0, 0, 0, 1))
    _offer(g, vec(wood=1), vec(ore=1))
    g.step(Action(1, A.ACCEPT_TRADE))
    g.step(Action(2, A.REJECT_TRADE))
    g.step(Action(3, A.REJECT_TRADE))
    g.step(Action(0, A.CANCEL_TRADE))
    assert g.phase == Phase.MAIN and g.players[0].hand == [1, 0, 0, 0, 0]


def test_offer_cap_and_reset():
    g = main_game()
    give(g, 0, (5, 0, 0, 0, 0))
    for _ in range(3):
        _offer(g, vec(wood=1), vec(ore=1))
        for p in (1, 2, 3):
            g.step(Action(p, A.REJECT_TRADE))
    assert not g.is_legal(Action(0, A.OFFER_TRADE, (vec(wood=1), vec(ore=1))))
    assert not any(a.type == A.OFFER_TRADE for a in g.legal_actions())
    g.step(Action(0, A.END_TURN))
    assert g.trades_this_turn == 0


def test_offer_cap_none_unlimited():
    g = main_game(max_trade_offers_per_turn=None)
    give(g, 0, (1, 0, 0, 0, 0))
    for _ in range(10):
        _offer(g, vec(wood=1), vec(ore=1))
        for p in (1, 2, 3):
            g.step(Action(p, A.REJECT_TRADE))
    assert g.is_legal(Action(0, A.OFFER_TRADE, (vec(wood=1), vec(ore=1))))


def test_two_player_trade():
    g = main_game(n=2)
    give(g, 0, (1, 0, 0, 0, 0))
    give(g, 1, (0, 1, 0, 0, 0))
    _offer(g, vec(wood=1), vec(brick=1))
    g.step(Action(1, A.ACCEPT_TRADE))
    g.step(Action(0, A.CONFIRM_TRADE, 1))
    assert g.players[0].hand == [0, 1, 0, 0, 0]


def test_trade_response_actions_only_for_responder():
    g = main_game()
    give(g, 0, (1, 0, 0, 0, 0))
    _offer(g, vec(wood=1), vec(ore=1))
    assert not g.is_legal(Action(0, A.CANCEL_TRADE))
    assert not g.is_legal(Action(2, A.REJECT_TRADE))
    assert not g.is_legal(Action(1, A.END_TURN))
    assert all(a.player == 1 and a.type in (A.REJECT_TRADE, A.ACCEPT_TRADE, A.COUNTER_OFFER)
               for a in g.legal_actions())
