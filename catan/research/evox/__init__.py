"""D29 island-model evolution of the heuristic evaluator (docs/research/EVOX.md). Importing registers ``evox``."""
from catan.agents.registry import register


def _make(**kw):
    from .agent import make
    return make(**kw)


register("evox", _make)
