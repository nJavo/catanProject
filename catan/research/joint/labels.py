"""Counterfactual option labels for the trade head (D15, ExIt-style).

    python -m scripts.joint_labels --games 900 --workers 3 --out runs/joint_labels

A learner seat plays combo_zs (league GNN core + nego BC trade head) against
a sampled table. At a sampled fraction of its trade decisions, the top ``k``
options (always including the nego choice, the trader teacher's choice and
the no-trade option) are each played out on a clone of the true state
(training-time privileged information is fine for *labels*; the trained head
only ever sees the information-set features) with common random numbers until
the next turn starts, by fresh copies of the table's policies. Each branch end
is scored by the D12 value net as the learner's win probability (1/0 if the
game ended). Records: extended state (nego state + CoreFeatures), options, the
nego choice, the teacher choice and per-option values ``q``.
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

from catan.agents.registry import make_agent
from catan.engine import Game, GameConfig
from catan.eval.seeds import TRAIN_SEED_BASE
from catan.research.combo.agent import ComboAgent
from catan.research.nego import options as NO
from catan.research.nego.agent import NegoAgent
from catan.research.nego.model import batch

from .features import CoreFeatures

COMBO = "combo:checkpoint=models/combo_best.pt,deterministic=true"
TABLES = [  # (weight, opponent specs in seat order after the learner)
    (0.4, [COMBO, "intermediate", "intermediate"]),
    (0.2, [COMBO] * 3),
    (0.15, ["intermediate"] * 3),
    (0.15, ["intermediate_trader"] * 3),
    (0.1, ["basic"] * 3),
]


def _make(spec, seed):
    name, _, rest = spec.partition(":")
    kw = {}
    for part in filter(None, rest.split(",")):
        k, _, v = part.partition("=")
        kw[k] = {"true": True, "false": False}.get(v, v)
    return make_agent(name, seed=seed, **kw)


class LabelTradeAgent(NegoAgent):
    """Plays the nego BC head's choice; records extended state, teacher and branch values."""

    def __init__(self, seed, core, branch, k=6, frac=0.5):
        super().__init__(seed=seed, checkpoint="models/nego_best.pt", record=True)
        self.core, self.branch, self.k, self.frac = core, branch, k, frac
        self.lrng = random.Random(seed)

    def _nego_choose(self, g, me, decision, opts, teacher, est):
        if len(opts) == 1:
            return opts[0][0], {}
        base = NO.state_vec(g, me, est, decision)
        sample = {"s": base + self.core(g, me), "o": [o[1] for o in opts],
                  "give": [list(o[2]) for o in opts], "get": [list(o[3]) for o in opts]}
        with torch.no_grad():
            logits, _ = self.net(*batch([{**sample, "s": base}]))
        idx = int(torch.argmax(logits[0]))
        sample.update(a=idx, teacher=teacher, decision=decision, turn=g.turn)
        if self.lrng.random() < self.frac:
            order = sorted(range(len(opts)), key=lambda i: -float(logits[0][i]))
            cand = list(dict.fromkeys([idx, teacher, 0] + order))[:max(self.k, 3)]
            vals = self.branch(g, me, [opts[i][0] for i in cand], self.lrng.getrandbits(32))
            sample["q"] = dict(zip(cand, vals))
        self.trajectory.append(sample)
        return opts[idx][0], {"nego_decision": decision, "chosen": idx, "teacher": teacher}


class Brancher:
    def __init__(self, core: CoreFeatures, specs: list[str], max_steps=400):
        self.core, self.specs, self.max_steps = core, specs, max_steps
        self.agents = None

    def set_table(self, seat_specs, seed):
        # fresh branch policies (beliefs start from the public prior: they never see the history)
        self.agents = [_make(s, seed + i) for i, s in enumerate(seat_specs)]
        for i, a in enumerate(self.agents):
            a.reset(i, seed + i)

    def __call__(self, g, me, actions, seed):
        ends, fixed = [], {}
        for j, a in enumerate(actions):
            c = g.clone()
            c.rng = random.Random(seed)  # common random numbers across options
            c.step(a)
            start, steps = g.turn, 0
            while not c.is_over and c.turn == start and steps < self.max_steps:
                p = c.actor
                c.step(self.agents[p].act(c, c.legal_actions()))
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
    seat_specs, it = [], iter(opps)
    for s in range(4):
        seat_specs.append(COMBO if s == learner else next(it))
    core = CoreFeatures()
    br = Brancher(core, seat_specs)
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
        p = g.actor
        evs = g.step(agents[p].act(g, g.legal_actions()))
        for s, a in enumerate(agents):
            if callable(getattr(a, "observe", None)):
                a.observe(g, [e for e in evs if e.get("visible_to") is None or s in e["visible_to"]])
    traj = agents[learner].trade.trajectory
    for d in traj:
        d["won"] = g.winner == learner
    return {"seed": seed, "samples": traj, "won": g.winner == learner, "table": "+".join(opps),
            "turns": g.turn}


def run(games, workers, out, seed=0, frac=0.5, k=6, chunk=30, log=print):
    os.makedirs(out, exist_ok=True)
    base = TRAIN_SEED_BASE + 15_000_000 + seed * 1_000_000
    tasks = [{"seed": base + i, "frac": frac, "k": k} for i in range(games)]
    t0 = time.time()
    buf, part, n = [], 0, 0
    with get_context("fork").Pool(workers, initializer=torch.set_num_threads, initargs=(1,)) as pool:
        for i, r in enumerate(pool.imap_unordered(play, tasks, chunksize=1)):
            buf.append(r)
            n += sum(1 for d in r["samples"] if "q" in d)
            if len(buf) >= chunk or i == len(tasks) - 1:
                with open(os.path.join(out, f"part_{part:04d}.pkl"), "wb") as f:
                    pickle.dump(buf, f)
                part += 1
                buf = []
                log(f"{i + 1}/{games} games, {n} labelled decisions, {time.time() - t0:.0f}s")
