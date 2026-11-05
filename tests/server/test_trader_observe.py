from catan.agents.registry import make_agent
from catan.engine import Game, GameConfig
from catan.server.app import Seat, Session


def test_server_record_feeds_seat_filtered_events_to_trader():
    g = Game(config=GameConfig(max_turns=60), seed=3)
    agents = [make_agent("intermediate_trader", seed=i) for i in range(4)]
    for i, a in enumerate(agents):
        a.reset(i, i)
    seats = [Seat("agent", "intermediate_trader", {}, a) for a in agents]
    sess = Session("t", g, seats, "lab")
    private_seen = [0] * 4
    for _ in range(600):
        if g.is_over:
            break
        evs = g.step(agents[g.actor].act(g, g.legal_actions()))
        for e in evs:
            if "visible_to" in e:
                for q in e["visible_to"]:
                    private_seen[q] += 1
        sess.record(evs)
    for a in agents:
        assert a.tracker is not None and not a.tracker.lost
        assert a.tracker.synced == len(g.action_log)
    assert sum(private_seen) > 0
