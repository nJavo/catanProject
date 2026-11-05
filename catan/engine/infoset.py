"""Information-set view of a game for agents that must not see hidden state.

``game.observe_for(seat)`` returns an ``InfoSetGame``: a sandboxed copy that
supports what bots and RL use (``legal_actions``, ``is_legal``, the public
fields, ``player_view``, ``supply``/``trade_rate``/``public_victory_points``,
``hand_count``/``dev_count``, the seat's own hand and dev cards) and removes
everything else:

- opponents' ``hand``, ``dev`` and ``dev_new`` are not stored; reading them
  raises ``HiddenInformationError`` (counts: ``hand_count(q)``, ``dev_count(q)``);
- ``dev_deck`` only answers ``len()``/truthiness; ``rng`` and ``seed`` raise;
- ``clone()`` returns a *determinized* real ``Game`` (opponents' hands and dev
  cards, deck order and dice resampled from public information);
- ``step()`` raises: the view is a snapshot, step a ``clone()`` instead.

Mutating the view never touches the real game. This is a guard against
accidental peeking by in-process agents, not a security boundary: run truly
untrusted code in another process and send it views over the wire.
"""
from __future__ import annotations

import random
from dataclasses import replace

from .game import Game, Player


class HiddenInformationError(RuntimeError):
    """An agent read information its seat may not see."""


def _deny(what):
    raise HiddenInformationError(f"{what} is hidden from this seat (InfoSetGame)")


class OpponentView:
    """Public part of an opponent's Player."""
    __slots__ = ("settlements", "cities", "roads", "ports", "knights", "_hand_count", "_dev_count",
                 "_dev_new_count")

    def __init__(self, pl: Player):
        self.settlements, self.cities, self.roads = pl.settlements[:], pl.cities[:], pl.roads[:]
        self.ports, self.knights = set(pl.ports), pl.knights
        self._hand_count, self._dev_count, self._dev_new_count = sum(pl.hand), sum(pl.dev), sum(pl.dev_new)

    hand = property(lambda self: _deny("an opponent's hand (use game.hand_count(q))"))
    dev = property(lambda self: _deny("an opponent's dev cards (use game.dev_count(q))"))
    dev_new = property(lambda self: _deny("an opponent's dev cards"))

    def to_dict(self):
        _deny("an opponent's hand")


class _HiddenDeck:
    """Development deck: only its size is public."""
    __slots__ = ("_n",)

    def __init__(self, n):
        self._n = n

    def __len__(self):
        return self._n

    def __bool__(self):
        return self._n > 0

    def __getattr__(self, name):
        _deny("the dev deck order")

    def __iter__(self):
        _deny("the dev deck order")

    def __getitem__(self, i):
        _deny("the dev deck order")


class _HiddenRNG:
    __slots__ = ()

    def __getattr__(self, name):
        _deny("the game RNG")


_HIDDEN_RNG = _HiddenRNG()


class InfoSetGame(Game):
    """See the module docstring. Build with ``Game.observe_for(seat)``."""
    _reveals_when_over = False  # player_view never reveals opponents' hands from a view

    @property
    def seed(self):
        _deny("the game seed (it determines dice and deck order)")

    def hand_count(self, p: int) -> int:
        pl = self.players[p]
        return pl._hand_count if type(pl) is OpponentView else sum(pl.hand)

    def dev_count(self, p: int) -> int:
        pl = self.players[p]
        return pl._dev_count if type(pl) is OpponentView else sum(pl.dev)

    def dev_new_count(self, p: int) -> int:
        pl = self.players[p]
        return pl._dev_new_count if type(pl) is OpponentView else sum(pl.dev_new)

    def unseen_dev_pool(self, me: int) -> list[int]:
        if me != self.seat:
            _deny("another seat's unseen dev pool")
        return list(self._unseen)

    def victory_points(self, p: int) -> int:
        if p != self.seat:
            _deny("an opponent's victory point cards (use public_victory_points)")
        return super().victory_points(p)

    def step(self, action):
        raise HiddenInformationError("InfoSetGame is a read-only snapshot; step a clone() instead")

    def to_dict(self):
        _deny("the full state")

    def clone(self) -> Game:
        """A determinized real Game: hidden information resampled from public info."""
        return self.determinized(self.seat, self._clone_rng)

    def _determinization_base(self) -> Game:
        g = Game.__new__(Game)
        g.__dict__.update(self.__dict__)
        for k in ("seat", "_unseen", "_clone_rng"):
            del g.__dict__[k]
        g.seed = None  # unknown to this seat
        g._rng, g._rng_shared = None, False  # determinized() installs a fresh RNG
        g.players = [pl.copy() if type(pl) is Player else _blank_player(pl) for pl in self.players]
        g.dev_deck = []
        for k in ("bank", "vertex_owner", "vertex_level", "edge_owner", "discard_remaining", "road_length",
                  "setup_order"):
            setattr(g, k, list(getattr(self, k)))
        self._share_log(g)
        if self.trade is not None:
            g.trade = dict(self.trade, pending=list(self.trade["pending"]), responses=dict(self.trade["responses"]))
        return g


def _blank_player(v: OpponentView) -> Player:
    p = Player()
    p.settlements, p.cities, p.roads = v.settlements[:], v.cities[:], v.roads[:]
    p.ports, p.knights = set(v.ports), v.knights
    return p  # hand/dev filled in by Game.determinized


def observe_for(g: Game, seat: int) -> InfoSetGame:
    v = InfoSetGame.__new__(InfoSetGame)
    d = dict(g.__dict__)
    d.pop("seed", None)
    v.__dict__.update(d)
    v.seat = seat
    v._rng, v._rng_shared = _HIDDEN_RNG, False
    v.config = replace(g.config)
    v._vertex_port = dict(g._vertex_port)
    v._hexes_by_number = {k: list(x) for k, x in g._hexes_by_number.items()}
    v.players = [pl.copy() if q == seat else OpponentView(pl) for q, pl in enumerate(g.players)]
    v.dev_deck = _HiddenDeck(len(g.dev_deck))
    for k in ("bank", "vertex_owner", "vertex_level", "edge_owner", "discard_remaining", "road_length",
              "setup_order"):
        setattr(v, k, list(getattr(g, k)))
    g._share_log(v)  # public: every action is announced
    if g.trade is not None:
        v.trade = dict(g.trade, pending=list(g.trade["pending"]), responses=dict(g.trade["responses"]))
    v._unseen = g.unseen_dev_pool(seat)  # sorted: its composition is public
    v._clone_rng = random.Random(len(v.action_log))  # seeded from public info only
    return v
