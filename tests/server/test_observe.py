"""The server feeds agents' observe() hooks with their seat's visible events, like tournaments do."""
import random

from catan.bots.belief import HandPosterior
from catan.engine import Game
from catan.server.app import Seat, Session


class Watcher:
    def __init__(self, seat):
        self.post = HandPosterior(4, seat)
        self.seen = []

    def observe(self, game, events):
        self.seen += events
        self.post.observe(game, events)


def test_record_feeds_visible_events():
    g = Game(seed=3)
    ws = [Watcher(q) for q in range(4)]
    sess = Session("x", g, [Seat("agent", "w", {}, w) for w in ws], "lab")
    r = random.Random(0)
    for _ in range(300):
        sess.record(g.step(r.choice(g.legal_actions())))
    sess.record([{"type": "game_resumed"}], step=False)
    for q, w in enumerate(ws):
        assert all(e.get("visible_to") is None or q in e["visible_to"] for e in w.seen)
        assert not w.post.lost and w.post.synced == len(g.action_log)
