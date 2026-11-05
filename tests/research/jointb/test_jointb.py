import random

import torch

from catan.agents.registry import make_agent
from catan.engine import Game, GameConfig
from catan.engine.actions import A, Action
from catan.research.joint.agent import _bundle_head
from catan.research.jointb.features import EXTRA_B, OPP_F, JointBCore, opp_history
from catan.research.jointb.train import bundle, widen
from catan.research.joint.model import save
from catan.research.nego.model import batch
from catan.research.nego.options import OPT_F, STATE_F


class _G:  # minimal stand-in: opp_history only reads n and action_log
    n = 4

    def __init__(self, log):
        self.action_log = log


def test_opp_history_separates_offerers_from_rule_bots():
    z = ((0,) * 5, (0,) * 5)
    log = [Action(1, A.OFFER_TRADE, z), Action(2, A.REJECT_TRADE), Action(3, A.COUNTER_OFFER, z),
           Action(1, A.END_TURN), Action(2, A.ACCEPT_TRADE), Action(2, A.END_TURN)]
    x = opp_history(_G(log), 0)
    assert len(x) == OPP_F
    s1, s2, s3 = x[0:4], x[4:8], x[8:12]
    assert s1[0] > 0 and s2[0] == 0 and s3[0] == 0        # only seat 1 offered
    assert s2[1] > s3[1] > s1[1] == 0                      # response counts 2, 1, 0
    assert s2[2] == (1 + 0.5) / 4 and s3[3] == (1 + 0.5) / 3
    # relative seats: from seat 1, k=1 is seat 2 and k=3 is seat 0, which did nothing
    assert opp_history(_G(log), 1)[0:4] == s2
    assert opp_history(_G(log), 1)[8:12] == [0.0, 0.0, 0.25, 0.25]


def test_widen_keeps_d15_head_outputs():
    old = _bundle_head("models/joint_best.pt")
    new = widen(old).eval()
    rng = random.Random(0)
    k = 5
    x = [{"s": [rng.random() for _ in range(STATE_F + EXTRA_B)],
          "o": [[rng.random() for _ in range(OPT_F)] for _ in range(k)],
          "give": [[rng.randrange(3) for _ in range(5)] for _ in range(k)],
          "get": [[rng.randrange(3) for _ in range(5)] for _ in range(k)]} for _ in range(3)]
    ln, _ = new(*batch(x))
    lo, _ = old(*batch([{**d, "s": d["s"][:STATE_F + old.extra]} for d in x]))
    assert torch.allclose(ln, lo, atol=1e-6)


def test_jointb_vt_plays(tmp_path):
    head = tmp_path / "h.pt"
    save(widen(_bundle_head("models/joint_best.pt")), str(head), {})
    b = tmp_path / "b.pt"
    bundle(str(head), str(b))
    agents = [make_agent("jointb_vt", seed=0, checkpoint=str(b), sims=2)] + \
             [make_agent("intermediate_trader", seed=i) for i in (1, 2, 3)]
    for i, a in enumerate(agents):
        a.reset(i, i)
    g = Game(config=GameConfig(max_turns=25), seed=3)
    while not g.is_over:
        a = agents[g.actor].act(g, g.legal_actions())
        assert g.is_legal(a)
        evs = g.step(a)
        for i, ag in enumerate(agents):
            if hasattr(ag, "observe"):
                ag.observe(g, [e for e in evs if "visible_to" not in e or i in e["visible_to"]])
    assert len(JointBCore()(g, 0)) == EXTRA_B


def _play(name, seed, opp, **kw):
    agents = [make_agent(name, seed=seed, **kw)] + [make_agent(opp, seed=seed + i) for i in (1, 2, 3)]
    for i, a in enumerate(agents):
        a.reset(i, seed + i)
    g = Game(config=GameConfig(max_turns=500), seed=seed)
    while not g.is_over:
        evs = g.step(agents[g.actor].act(g, g.legal_actions()))
        for i, a in enumerate(agents):
            if callable(getattr(a, "observe", None)):
                a.observe(g, [e for e in evs if e.get("visible_to") is None or i in e["visible_to"]])
    return g, agents[0]


def test_gated_plays_v1_vs_basic_and_jointb_vs_negotiators():
    torch.set_num_threads(1)  # multi-threaded matmuls can flip near-tied search argmaxes between runs
    g1, a = _play("jointb_gated", 6800001, "basic")
    g2, _ = _play("value_trade", 6800001, "basic", checkpoint="models/value_best.pt")
    assert g1.action_log == g2.action_log and a.uses["jointb"] < 0.1 * a.uses["v1"]
    _, a = _play("jointb_gated", 6800001, "intermediate_trader")
    assert a.uses["jointb"] > 10 * a.uses["v1"]
