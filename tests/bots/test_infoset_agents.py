"""EN-008: every registered agent plays through Game.observe_for(seat).

Through the view, reading hidden state raises HiddenInformationError (counted
as an illegal move by the tournament runner). An agent that never peeks plays
exactly the same actions as with the full game."""
import os

import pytest

from catan.agents.registry import available_agents
from catan.engine import GameConfig
from catan.eval.tournament import play_one

FAST = {"search": "search:rollouts=3,candidates=3", "search_trader": "search_trader:rollouts=3,candidates=3",
        "rl": "rl:checkpoint=models/ppo_baseline.pt", "arch_rl": "arch_rl:checkpoint=models/arch_best.pt",
        "value_search": "value_search:checkpoint=models/league_best.pt,value=models/value_best.pt,sims=4",
        "value_trade": "value_trade:checkpoint=models/value_best.pt,prior=models/league_best.pt,sims=4",
        "belief_rl": "belief_rl:checkpoint=models/belief_v3.pt", "belief_combo": "belief_combo:checkpoint=models/belief_v3.pt",
        "distil_trade": "distil_trade:checkpoint=models/distil_best.pt,sims=4",
        "belief_value_trade": "belief_value_trade:checkpoint=models/value_best.pt,prior=models/league_best.pt,sims=4",
        "opp_value_trade": "opp_value_trade:checkpoint=models/value_best.pt,prior=models/league_best.pt,sims=4"}
CHECKPOINTS = {"rl": "models/ppo_baseline.pt", "arch_rl": "models/arch_best.pt",
               "value_search": "models/value_best.pt", "value_trade": "models/value_best.pt",
               "belief_rl": "models/belief_v3.pt", "belief_combo": "models/belief_v3.pt",
               "distil_trade": "models/distil_best.pt",
               "belief_value_trade": "models/value_best.pt", "opp_value_trade": "models/opp_best.pt"}
# Agents that read hidden state.
KNOWN_PEEKS: dict = {}


def _play(spec, seed, infoset):
    cfg = GameConfig(num_players=4, max_turns=80).to_dict()
    return play_one({"seed": seed, "rotation": 0, "lineup": (spec, "basic", "intermediate", spec),
                     "config": cfg, "save_replay": True, "infoset": infoset})


@pytest.mark.parametrize("name", [  # strict: once an agent is fixed its entry must be removed
    pytest.param(n, marks=pytest.mark.xfail(strict=True, reason=KNOWN_PEEKS[n])) if n in KNOWN_PEEKS else n
    for n in available_agents()])
def test_agent_does_not_peek(name):
    if name in CHECKPOINTS:
        pytest.importorskip("torch")
        if not os.path.isfile(CHECKPOINTS[name]):
            pytest.skip(f"{CHECKPOINTS[name]} not available")
    spec = FAST.get(name, name)
    base, view = _play(spec, 11, False), _play(spec, 11, True)
    assert base["illegal"] == [0] * 4
    assert view["illegal"] == [0] * 4, view["illegal_log"][:3]
    assert view["replay"]["actions"] == base["replay"]["actions"]


def test_untrusted_agents_always_get_the_view():
    from catan.agents.base import Agent
    from catan.agents.registry import _REGISTRY, _UNTRUSTED, is_trusted, register

    class Peeker(Agent):
        def act(self, game, legal):
            sum(game.players[(game.actor + 1) % game.n].hand)  # hidden from this seat
            return legal[0]

    register("_peeker", lambda seed=None, **_: Peeker(), trusted=False)
    try:
        assert not is_trusted("_peeker") and is_trusted("basic")
        cfg = GameConfig(num_players=2, max_turns=5).to_dict()
        rec = play_one({"seed": 1, "rotation": 0, "lineup": ("_peeker", "random"), "config": cfg})
        assert rec["illegal"][0] > 0 and rec["illegal"][1] == 0
        assert "HiddenInformationError" in rec["illegal_log"][0]["error"]
    finally:
        _REGISTRY.pop("_peeker", None)
        _UNTRUSTED.discard("_peeker")
