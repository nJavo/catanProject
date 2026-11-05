"""``opp_value_trade``: value_trade-v1 whose search plays each opponent with its classified type (D21).

    opp_value_trade                                    # classifier models/opp_best.pt, threshold 0.6
    opp_value_trade:threshold=0.8,learned=intermediate,pool=true

value_trade's search (catan.research.exit.search) plays every opponent with the basic bot inside
its simulations, including the opponents' setup replies. Here, before each decision, the online
classifier (``features`` + ``classifier``, public action log only) labels every opponent seat;
each class maps to a rollout policy and a seat whose most likely policy (class probabilities summed
per policy) has probability >= ``threshold`` is simulated with it:
basic -> basic, intermediate -> intermediate, intermediate_trader -> trader, search -> intermediate
(search's own base policy; the real search bot is too slow for rollouts), learned -> ``learned``
(default basic, i.e. unchanged). Unsure seats keep the basic rollout policy, which is value_trade-v1.
Everything else (value net, prior, trade layer) is value_trade-v1's.
"""
from __future__ import annotations

from catan.agents.registry import register
from catan.bots.intermediate import IntermediateAgent
from catan.bots.trading import TraderIntermediateAgent
from catan.research.value.trade import DEFAULT_VALUE, ValueTradeAgent

from .classifier import load
from .features import CLASSES, seat_features

DEFAULT_OPP = "models/opp_best.pt"
POLICY_OF = {"basic": "basic", "intermediate": "intermediate", "intermediate_trader": "trader",
             "search": "intermediate"}


class _BySeat:
    """Rollout opponent policy that dispatches on the acting seat."""

    def __init__(self, default, by_seat=None):
        self.default, self.by_seat = default, by_seat or {}

    def act(self, c, legal):
        return self.by_seat.get(c.actor, self.default).act(c, legal)


class OppValueTradeAgent(ValueTradeAgent):
    name = "opp_value_trade"

    def __init__(self, checkpoint: str = DEFAULT_VALUE, opp: str = DEFAULT_OPP, threshold: float = 0.6,
                 learned: str = "basic", pool: bool = False, seed: int | None = None, **kw):
        super().__init__(checkpoint=checkpoint, seed=seed, **kw)
        if learned not in ("basic", "intermediate", "trader"):
            raise ValueError(f"learned must be basic|intermediate|trader, got {learned!r}")
        self.clf = load(opp)
        self.threshold, self.learned = float(threshold), learned
        self.pool = bool(pool)  # unsure seats take the policy every confident opponent shares (table prior)
        self._make_policies(seed)

    def _make_policies(self, seed):
        s = None if seed is None else seed + 2
        self.policies = {"basic": self.net._opp_policy, "intermediate": IntermediateAgent(seed=s),
                         # shortcut: no counter-offers in rollouts (enumerating them was 60% of search time);
                         # accept/reject is the trader's own rule. Revisit if trader tables matter.
                         "trader": TraderIntermediateAgent(seed=s, counters=False)}
        self.net._opp_policy = self.dispatch = _BySeat(self.policies["basic"])
        self._key, self.labels = None, {}

    def reset(self, seat, seed=None):
        super().reset(seat, seed)  # replaces net._opp_policy with a freshly seeded basic bot
        if seed is not None:
            self._make_policies(seed)
        else:
            self.net._opp_policy = self.dispatch
            self._key, self.labels = None, {}

    def classify(self, game):
        """{seat: (class, prob)} for every opponent, from the public log (cached per log length)."""
        log = game.action_log
        if self._key != len(log):
            p = self.clf.proba(seat_features(game.layout, game.config, log))
            me = game.actor
            # threshold the rollout policy, not the class: intermediate and search share one policy
            # and split their probability (each ~0.5), which kept most intermediate seats "unsure".
            self.labels, by = {}, {}
            for q in range(game.n):
                if q == me:
                    continue
                grp: dict = {}
                for c, x in zip(CLASSES, p[q]):
                    k = POLICY_OF.get(c, self.learned)
                    grp[k] = grp.get(k, 0.0) + float(x)
                k = max(grp, key=grp.get)
                self.labels[q] = (k, grp[k])
                if grp[k] >= self.threshold:
                    by[q] = self.policies[k]
            sure = {self.labels[q][0] for q in by}
            if self.pool and len(sure) == 1:
                k = sure.pop()
                for q in self.labels:
                    by.setdefault(q, self.policies[k])
            self.dispatch.by_seat = by
            self._key = len(log)
        return self.labels

    def act_with_info(self, game, legal, explain=True):
        labels = self.classify(game)
        a, info = super().act_with_info(game, legal, explain=explain)
        if explain:
            info["opp"] = {q: [c, round(p, 3)] for q, (c, p) in labels.items()}
        return a, info


register("opp_value_trade", lambda seed=None, **kw: OppValueTradeAgent(seed=seed, **kw))
