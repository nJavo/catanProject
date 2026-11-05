"""RL decision explanations: no effect on actions, attributions are complete, payload is small."""
import json

import pytest
import torch

from catan.agents.registry import make_agent
from catan.engine import Game, GameConfig
from catan.rl.model import PolicyNet, save_checkpoint

MAX_BYTES = 8000


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    torch.manual_seed(0)
    p = tmp_path_factory.mktemp("ck") / "a.pt"
    save_checkpoint(p, PolicyNet(hidden=64, blocks=2), {"model": {"hidden": 64, "blocks": 2}})
    return str(p)


def play(ckpt, seed, explain, deterministic, on_info=None, max_turns=40):
    agents = [make_agent("rl", checkpoint=ckpt, deterministic=deterministic, seed=seed)] + \
             [make_agent("basic", seed=seed + i) for i in range(1, 4)]
    for i, a in enumerate(agents):
        a.reset(i, seed * 10 + i)
    g = Game(config=GameConfig(max_turns=max_turns), seed=seed)
    while not g.is_over:
        legal = g.legal_actions()
        if g.actor == 0 and explain:
            a, info = agents[0].act_with_info(g, legal)
            if on_info:
                on_info(g, a, info)
        else:
            a = agents[g.actor].act(g, legal)
        g.step(a)
    return g.action_log


@pytest.mark.parametrize("deterministic", [True, False])
def test_explanations_do_not_change_rl_actions(ckpt, deterministic):
    for seed in range(3):
        assert play(ckpt, seed, False, deterministic) == play(ckpt, seed, True, deterministic), seed


def test_rl_explanation_contents(ckpt):
    n = [0]

    def check(g, a, info):
        assert "explain_error" not in info, info["explain_error"]
        raw = json.dumps(info, allow_nan=False)
        assert len(raw) < MAX_BYTES
        e = info["explain"]
        assert e["kind"] == "policy_model" and e["summary_source"] == "generated from model outputs"
        assert e["candidates"][0]["action"] == a.to_json()
        assert abs(e["candidates"][0]["prob"] - info["chosen_prob"]) < 1e-3 and e["value"] == info["value"]
        at = e["attribution"]
        assert at["label"] == "model-derived attribution (integrated gradients)" and e["caveats"]
        for head in ("logit", "value"):
            d = at[head]["at_obs"] - at[head]["at_baseline"]
            assert abs(at[head]["completeness_gap"]) <= 0.05 * max(1.0, abs(d)), (head, at[head])
        assert abs(sum(b["value"] for b in at["blocks"]) - at["value"]["sum"]) < 0.01
        assert all(c["n"] == e["value_probe"]["determinizations"] for c in e["candidates"])
        n[0] += 1

    play(ckpt, 5, True, True, on_info=check)
    assert n[0] > 20
