"""D12 learned evaluator (docs/research/VALUE.md). Importing registers ``value_search`` and
``value_trade`` (value_search + the combo/nego trade layer)."""
from catan.agents.registry import register


def _make(**kw):
    from .search import ValueSearchAgent
    return ValueSearchAgent(**kw)


def _make_trade(**kw):
    from .trade import ValueTradeAgent
    return ValueTradeAgent(**kw)


register("value_search", _make)
register("value_trade", _make_trade)
