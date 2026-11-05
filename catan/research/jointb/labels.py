"""Counterfactual trade-option labels, D18 version (docs/research/JOINTB.md).

    python -m scripts.jointb_labels --games 1500 --workers 3 --out runs/jointb_labels

Same scheme as catan.research.joint.labels (learner = combo_zs; top-k options
replayed on clones with common random numbers; branch end scored by the D12
value net), with three changes:
  1. tables rebalanced toward rule bots: 30% 3x basic, 25% 3x intermediate,
     20% 3x intermediate_trader, 15% gate mixed, 10% 3x combo_zs;
  2. horizon: a branch runs until the learner's next turn starts, so for an
     offer every possible partner's next turn (and its build) is inside it;
  3. the recorded state also carries per-opponent offer/accept history
     (features.opp_history), so records are JointNet(extra=EXTRA_B) inputs.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import pickle
import random
import time
from multiprocessing import get_context

import torch

from catan.engine import Game, GameConfig
from catan.eval.seeds import TRAIN_SEED_BASE
from catan.research.combo.agent import ComboAgent
from catan.research.joint.labels import COMBO, Brancher, LabelTradeAgent, _make

from .features import JointBCore

TABLES = [  # (weight, opponent specs in seat order after the learner)
    (0.30, ["basic"] * 3),
    (0.25, ["intermediate"] * 3),
    (0.20, ["intermediate_trader"] * 3),
    (0.15, [COMBO, "intermediate", "intermediate"]),
    (0.10, [COMBO] * 3),
]


class RoundBrancher(Brancher):
    """Brancher whose branches end when the learner's next turn starts."""

    def __call__(self, g, me, actions, seed):
        stop = g.turn + ((me - g.current) % g.n or g.n)
        ends, fixed = [], {}
        for j, a in enumerate(actions):
            c = g.clone()
            c.rng = random.Random(seed)  # common random numbers across options
            c.step(a)
            steps = 0
            while not c.is_over and c.turn < stop and steps < self.max_steps:
                c.step(self.agents[c.actor].act(c, c.legal_actions()))
                steps += 1
            if c.is_over:
                fixed[j] = 1.0 if c.winner == me else 0.0
            else:
                ends.append((j, c))
        probs = self.core.win_probs([c for _, c in ends], me)
        out = dict(fixed)
        out.update({j: p for (j, _), p in zip(ends, probs)})
        return [out[j] for j in range(len(actions))]


def play(task):
    seed, frac, k = task["seed"], task["frac"], task["k"]
    torch.set_num_threads(1)
    rng = random.Random(seed)
    w, table = zip(*TABLES)
    opps = rng.choices(table, weights=w)[0]
    learner = rng.randrange(4)
    it = iter(opps)
    seat_specs = [COMBO if s == learner else next(it) for s in range(4)]
    core = JointBCore()
    br = RoundBrancher(core, seat_specs, max_steps=task.get("max_steps", 1500))
    br.set_table(seat_specs, seed + 100)
    agents = []
    for s, spec in enumerate(seat_specs):
        if s == learner:
            a = ComboAgent(checkpoint="models/combo_best.pt", seed=seed + s)
            a.trade = LabelTradeAgent(seed + s, core, br, k=k, frac=frac)
        else:
            a = _make(spec, seed + s)
        a.reset(s, seed + s)
        agents.append(a)
    g = Game(config=GameConfig(max_turns=500), seed=seed)
    while not g.is_over:
        evs = g.step(agents[g.actor].act(g, g.legal_actions()))
        for s, a in enumerate(agents):
            if callable(getattr(a, "observe", None)):
                a.observe(g, [e for e in evs if e.get("visible_to") is None or s in e["visible_to"]])
    traj = agents[learner].trade.trajectory
    for d in traj:
        d["won"] = g.winner == learner
    return {"seed": seed, "samples": traj, "won": g.winner == learner, "table": "+".join(opps),
            "turns": g.turn}


def run(games, workers, out, seed=0, frac=0.4, k=6, chunk=30, log=print):
    os.makedirs(out, exist_ok=True)
    base = TRAIN_SEED_BASE + 18_000_000 + seed * 1_000_000
    done = {r["seed"] for p in sorted(os.listdir(out)) if p.startswith("part_")
            for r in pickle.load(open(os.path.join(out, p), "rb"))}  # resume
    tasks = [{"seed": base + i, "frac": frac, "k": k} for i in range(games) if base + i not in done]
    part = len([p for p in os.listdir(out) if p.startswith("part_")])
    t0, buf, n = time.time(), [], 0
    with get_context("fork").Pool(workers, initializer=torch.set_num_threads, initargs=(1,)) as pool:
        for i, r in enumerate(pool.imap_unordered(play, tasks, chunksize=1)):
            buf.append(r)
            n += sum(1 for d in r["samples"] if "q" in d)
            if len(buf) >= chunk or i == len(tasks) - 1:
                with open(os.path.join(out, f"part_{part:04d}.pkl"), "wb") as f:
                    pickle.dump(buf, f)
                part += 1
                buf = []
                log(f"{len(done) + i + 1}/{games} games, {n} labelled decisions, {time.time() - t0:.0f}s")
