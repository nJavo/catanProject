"""Heuristic and search bots. Importing this package registers them in
``catan.agents.registry`` (the registry imports it automatically).

``intermediate:trader=true`` / ``search:trader=true`` select the trading
trading variants (catan/bots/trading.py); the defaults stay
unchanged as benchmark anchors. ``domestic=false`` disables domestic trading
for any heuristic bot."""
from catan.agents.registry import register

from .basic import BasicHeuristicAgent
from .intermediate import IntermediateAgent
from .search import SearchAgent


def _intermediate(seed=None, trader=False, **kw):
    if trader:
        from .trading import TraderIntermediateAgent
        return TraderIntermediateAgent(seed=seed, **kw)
    return IntermediateAgent(seed=seed, **kw)


def _search(seed=None, trader=False, **kw):
    if trader:
        from .trading import TraderSearchAgent
        return TraderSearchAgent(seed=seed, **kw)
    return SearchAgent(seed=seed, **kw)


register("basic", lambda seed=None, **kw: BasicHeuristicAgent(seed=seed, **kw))
register("intermediate", _intermediate)
register("search", _search)
register("intermediate_trader", lambda seed=None, **kw: _intermediate(seed=seed, trader=True, **kw))
register("search_trader", lambda seed=None, **kw: _search(seed=seed, trader=True, **kw))
