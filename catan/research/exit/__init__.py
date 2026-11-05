"""D4 Expert Iteration (see docs/research/EXIT.md). Importing registers ``exit_search`` and
``exit_greedy`` (an ``rl`` checkpoint played by argmax: ExIt-distilled policies are high-entropy,
so the sampling ``rl`` default loses most of their strength; the gate spec has no such flag)."""
from catan.agents.registry import register

from .search import ExitSearchAgent

register("exit_search", lambda **kw: ExitSearchAgent(**kw))


def _greedy(**kw):
    from catan.rl.policy_agent import PolicyAgent
    return PolicyAgent(**{**kw, "deterministic": True})


register("exit_greedy", _greedy)
