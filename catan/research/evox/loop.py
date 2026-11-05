"""D29 island-model GA over ``evox`` genomes (docs/research/EVOX.md).

    python -m catan.research.evox.loop --run-id evox_r1 --islands 120 --members 6 --generations 2000
    python -m catan.research.evox.loop --run-id evox_r1 --resume --generations 3000

Per generation every island seats its members ``per_table`` (default 3) per table, the other seats
opponents (basic, intermediate or a hall-of-fame genome). Common random numbers: table t of every island plays the
same seed, seating and opponents. Fitness = lifetime mean of (win + VP/20) shrunk toward the
population mean, so survivors sharpen and lucky one-game children do not take over. The bottom
third (``replace``) of each island is replaced by mutated (self-adaptive sigma) children of the top half,
with uniform crossover; every ``migrate`` generations each island's best replaces its ring
neighbour's worst. Every ``every`` generations: the top candidates are re-evaluated on selection
block 56 vs 3x basic / 3x intermediate, the best becomes the checkpoint champion and enters the
hall of fame, and that champion is measured on report block 60 vs 3x random / basic /
intermediate (never read by selection). Elo is fitted against the fixed anchors (random = 1000).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
import time
from multiprocessing import get_context
from pathlib import Path

from catan.engine import Game, GameConfig
from catan.eval.seeds import block_seeds
from catan.eval.stats import wilson

from .agent import DIM, INTERMEDIATE_Z, NAMES, SCALE

ANCHORS = ("random", "basic", "intermediate")
FIT_BLOCK, REPORT_BLOCK = 56, 60
SEL_OFFSET = 50_000            # block 56 seeds [0, 50k) = fitness games, [50k, ...) = checkpoint selection
TRAIN_CFG = GameConfig(max_turns=250).to_dict()
EVAL_CFG = GameConfig(max_turns=500).to_dict()
SHOW = ("prod", "hand", "spot0", "dev_ev", "setup_pips", "setup_exp")  # genes charted in the Population tab

# ------------------------------------------------------------------ games (worker side)
_ANCHOR_CACHE: dict = {}


def _seat(spec, seed):
    if isinstance(spec, str):
        a = _ANCHOR_CACHE.get(spec)
        if a is None:  # direct classes: the registry would import torch into every worker
            from catan.agents.base import RandomAgent
            from catan.bots import BasicHeuristicAgent, IntermediateAgent
            a = _ANCHOR_CACHE[spec] = {"random": RandomAgent, "basic": BasicHeuristicAgent,
                                       "intermediate": IntermediateAgent}[spec]()
        return a
    from .agent import make
    return make(z=spec)


def play(task):
    """task = (seed, [spec x4], cfg dict); spec = anchor name or genome list. -> (winner, vp list)."""
    seed, specs, cfg = task
    g = Game(config=GameConfig(**cfg), seed=seed)
    agents = [_seat(s, seed) for s in specs]
    for i, a in enumerate(agents):
        a.reset(i, seed * 1000 + i)
    fb = random.Random(seed ^ 0x5EED)
    while not g.is_over:
        legal = g.legal_actions()
        try:
            act = agents[g.actor].act(g, legal)
            if not g.is_legal(act):
                act = fb.choice(legal)
        except Exception:
            act = fb.choice(legal)
        g.step(act)
    return g.winner, [g.victory_points(q) for q in range(4)]


def play_batch(tasks):
    return [play(t) for t in tasks]


def _init_worker():
    os.environ["OMP_NUM_THREADS"] = "1"


class Runner:
    """Plays a task list on 2 pool workers plus this process (3 processes in total)."""

    def __init__(self, procs=3):
        self.procs = procs
        self.pool = get_context("spawn").Pool(procs - 1, initializer=_init_worker) if procs > 1 else None

    def run(self, tasks):
        if not self.pool:
            return play_batch(tasks)
        # small chunks for the workers, a fixed share for the driver; games vary in length
        k = len(tasks) // self.procs
        mine, rest = tasks[:k], tasks[k:]
        chunks = [rest[i:i + 8] for i in range(0, len(rest), 8)]
        fut = self.pool.map_async(play_batch, chunks, chunksize=1)
        out = play_batch(mine)
        for c in fut.get():
            out.extend(c)
        return out

    def close(self):
        if self.pool:
            self.pool.terminate()
            self.pool.join()


def game_score(res, seat):
    winner, vp = res
    return (1.0 if winner == seat else 0.0) + vp[seat] / 20.0


# ------------------------------------------------------------------ ratings vs fixed anchors
def anchor_elo(results_by_opp: dict, anchor_elo_map: dict) -> tuple[float, float]:
    """1-D Bradley-Terry MLE of one entry against anchors with fixed Elo.
    results_by_opp: {anchor: (pair_wins, pair_games)}. Returns (elo, se). 0.5 virtual win/loss per anchor."""
    data = [(anchor_elo_map[a], w + 0.5, n + 1.0) for a, (w, n) in results_by_opp.items() if n]
    if not data:
        return float("nan"), float("nan")
    k = math.log(10) / 400
    r = 1500.0
    for _ in range(100):
        g = h = 0.0
        for ra, w, n in data:
            p = 1 / (1 + 10 ** ((ra - r) / 400))
            g += k * (w - n * p)
            h += k * k * n * p * (1 - p)
        step = g / h
        r += max(-200.0, min(200.0, step))
        if abs(step) < 1e-6:
            break
    return r, 1 / math.sqrt(h)


def pair_counts(results, specs_label, label):
    """Pairwise finishing-order results of ``label`` vs every other label in 4-seat games.
    results: [(res, [label per seat])]. -> {other: (wins, games)}."""
    out: dict = {}
    for (winner, vp), labs in results:
        sc = [vp[q] + (100 if winner == q else 0) for q in range(4)]
        for q in range(4):
            if labs[q] != label:
                continue
            for o in range(4):
                if labs[o] == label:
                    continue
                w, n = out.get(labs[o], (0.0, 0))
                out[labs[o]] = (w + (1.0 if sc[q] > sc[o] else 0.5 if sc[q] == sc[o] else 0.0), n + 1)
    return out


def anchor_ratings(runner, seeds) -> dict:
    """Elo of random/basic/intermediate from mixed anchor tables, random pinned at 1000."""
    from catan.eval.ratings import bradley_terry, pairwise_outcomes
    tables = [("random", "basic", "intermediate", "intermediate"), ("random", "random", "basic", "intermediate"),
              ("basic", "basic", "intermediate", "intermediate"), ("random", "basic", "basic", "intermediate")]
    tasks, labs = [], []
    for s in seeds:
        for t in tables:
            for r in range(4):
                lab = [t[(i + r) % 4] for i in range(4)]
                tasks.append((s, lab, EVAL_CFG))
                labs.append(lab)
    res = runner.run(tasks)
    rankings = [[(labs[i][q], res[i][1][q] + (100 if res[i][0] == q else 0)) for q in range(4)] for i in range(len(res))]
    bt = bradley_terry(pairwise_outcomes(rankings), ANCHORS)
    return {a: round(bt[a] - bt["random"] + 1000, 1) for a in ANCHORS}


# ------------------------------------------------------------------ evolution state
def new_member(st, z, sigma, parent, island, root):
    st["next_id"] += 1
    m = {"id": f"e{st['next_id']:06d}", "z": list(z), "sigma": sigma, "parent": parent, "born": st["gen"],
         "island": island, "root": root, "n": 0, "sum": 0.0, "wins": 0}
    st["parents"][m["id"]] = (parent, st["gen"])
    return m


def fitness(m, prior, k):
    return (m["sum"] + k * prior) / (m["n"] + k)


def init_state(a, rng):
    st = {"gen": 0, "next_id": -1, "parents": {}, "hof": [], "islands": [], "roots": {},
          "champions": [], "best": None, "args": vars(a)}
    from .agent_x import DIM_X, INTERMEDIATE_ZX
    dim, inter = (DIM_X, INTERMEDIATE_ZX) if a.genome == "ext" else (DIM, INTERMEDIATE_Z)
    start = list(inter)
    if a.init_from:  # warm start; a base genome gets the extra genes at intermediate's values
        start = json.loads(Path(a.init_from).read_text())["z"]
        start = (start + list(inter[len(start):]))[:dim]
    for i in range(a.islands):
        isl = []
        for _ in range(a.members):
            if a.init == "random":
                z = [rng.gauss(0, 1) for _ in range(dim)]
            elif a.init == "zero":
                z = [0.0] * dim
            else:  # intermediate / --init-from: the start genome, jittered except member 0 of each island
                z = list(start) if not isl else [x + rng.gauss(0, a.sigma0) for x in start]
            m = new_member(st, z, a.sigma0, None, i, None)
            m["root"] = m["id"]
            st["roots"][m["id"]] = list(z)
            isl.append(m)
        st["islands"].append(isl)
    return st


def make_child(st, a, rng, isl, i, prior):
    ranked = sorted(isl, key=lambda m: -fitness(m, prior, a.shrink))
    top = ranked[: max(2, len(ranked) // 2)]
    p = max(rng.sample(top, min(2, len(top))), key=lambda m: fitness(m, prior, a.shrink))  # binary tournament
    z = list(p["z"])
    if len(top) > 1 and rng.random() < a.crossover:
        q = rng.choice([m for m in top if m is not p])
        z = [x if rng.random() < 0.5 else y for x, y in zip(z, q["z"])]
    sigma = min(1.0, max(a.sigma_min, p["sigma"] * math.exp(a.tau * rng.gauss(0, 1))))
    z = [x + sigma * rng.gauss(0, 1) for x in z]
    return new_member(st, z, sigma, p["id"], i, p["root"])


def generation(st, a, runner, rng):
    g = st["gen"]
    k = a.per_table
    T = a.members // k
    seeds = block_seeds(FIT_BLOCK, SEL_OFFSET)
    # same seed, seating and opponents for table t on every island (common random numbers)
    plan = []
    for t in range(T):
        seed = seeds[(g * T + t) % SEL_OFFSET]
        def draw():
            if st["hof"] and rng.random() < a.hof_frac:
                return rng.choice(st["hof"])["z"]
            return "intermediate" if rng.random() < a.int_frac else "basic"
        # "same": every opponent seat at the table is one agent (closest to the 1v3 panel tables)
        opps = [draw()] * (4 - k) if a.opp_mode == "same" else [draw() for _ in range(4 - k)]
        seats = rng.sample(range(4), 4)  # seats[:k] = member seats
        plan.append((seed, opps, seats))
    tasks, who = [], []
    for i, isl in enumerate(st["islands"]):
        order = rng.sample(range(len(isl)), len(isl))
        for t, (seed, opps, seats) in enumerate(plan):
            specs = [None] * 4
            ms = [isl[order[k * t + j]] for j in range(k)]
            for j, m in enumerate(ms):
                specs[seats[j]] = m["z"]
            for j, o in enumerate(opps):
                specs[seats[k + j]] = o
            tasks.append((seed, specs, TRAIN_CFG))
            who.append([(m, seats[j]) for j, m in enumerate(ms)])
    res = runner.run(tasks)
    for r, seated in zip(res, who):
        for m, seat in seated:
            m["n"] += 1
            m["sum"] += game_score(r, seat)
            m["wins"] += r[0] == seat
    allm = [m for isl in st["islands"] for m in isl]
    prior = sum(m["sum"] for m in allm) / max(1, sum(m["n"] for m in allm))
    # selection + reproduction per island
    for i, isl in enumerate(st["islands"]):
        isl.sort(key=lambda m: -fitness(m, prior, a.shrink))
        for k in range(a.replace):
            isl[-1 - k] = make_child(st, a, rng, isl[: len(isl) - a.replace], i, prior)
    # ring migration: best of island i replaces the worst of island i+1
    if a.migrate and (g + 1) % a.migrate == 0:
        bests = [max(isl, key=lambda m: fitness(m, prior, a.shrink)) for isl in st["islands"]]
        for i, b in enumerate(bests):
            dst = st["islands"][(i + 1) % len(bests)]
            j = min(range(len(dst)), key=lambda k: fitness(dst[k], prior, a.shrink))
            c = new_member(st, b["z"], b["sigma"], b["id"], (i + 1) % len(bests), b["root"])
            c.update(n=b["n"], sum=b["sum"], wins=b["wins"])  # same genome: its games stay valid evidence
            dst[j] = c
    fits = sorted((fitness(m, prior, a.shrink) for m in allm), reverse=True)
    return {"gen": g, "time": time.time(), "games": len(tasks), "prior": round(prior, 4),
            "best_fit": round(fits[0], 4), "median_fit": round(fits[len(fits) // 2], 4),
            "mean_sigma": round(sum(m["sigma"] for m in allm) / len(allm), 4),
            "member_win_share": round(sum(any(r[0] == seat for _, seat in w) for r, w in zip(res, who)) / len(res), 4)}


def one_v_three(z, opp, seeds, cfg=EVAL_CFG):
    """Tasks for z alone vs 3x opp, 4 seat rotations per seed."""
    out = []
    for s in seeds:
        for seat in range(4):
            specs = [opp] * 4
            specs[seat] = z
            out.append(((s, specs, cfg), seat))
    return out


def eval_panel(runner, z, opps, seeds):
    """{opp: {"win_rate", "ci95", "n", "score", "pairs"}} for z 1v3 vs each opp."""
    tasks, meta = [], []
    for o in opps:
        for t, seat in one_v_three(z, o, seeds):
            tasks.append(t)
            meta.append((o, seat))
    res = runner.run(tasks)
    out = {}
    for o in opps:
        rows = [(r, seat) for r, (oo, seat) in zip(res, meta) if oo == o]
        w = sum(r[0] == seat for r, seat in rows)
        n = len(rows)
        pc = pair_counts([(r, ["x" if q == seat else o for q in range(4)]) for r, seat in rows], None, "x")
        out[o] = {"win_rate": w / n, "ci95": list(wilson(w, n)), "n": n, "wins": w,
                  "score": sum(game_score(r, seat) for r, seat in rows) / n, "pairs": pc.get(o, (0, 0)),
                  "per_game": [int(r[0] == seat) for r, seat in rows]}
    return out


def checkpoint(st, a, runner, rng, out_dir, anchors):
    g = st["gen"]
    allm = [m for isl in st["islands"] for m in isl]
    prior = sum(m["sum"] for m in allm) / max(1, sum(m["n"] for m in allm))
    ranked = sorted(allm, key=lambda m: -fitness(m, prior, a.shrink))
    cands, seen = [], set()
    for m in ranked:
        key = tuple(m["z"])
        if key not in seen:
            seen.add(key)
            cands.append(m)
        if len(cands) == a.candidates:
            break
    sel_seeds = block_seeds(FIT_BLOCK, SEL_OFFSET + a.sel_seeds)[SEL_OFFSET:]
    rows = []
    for m in cands:
        p = eval_panel(runner, m["z"], ("basic", "intermediate"), sel_seeds)
        elo, se = anchor_elo({o: p[o]["pairs"] for o in p}, anchors)
        rows.append((m, p, elo, se))
    champ, cp, celo, cse = max(rows, key=lambda x: x[1]["basic"]["score"] + x[1]["intermediate"]["score"])
    sel_score = cp["basic"]["score"] + cp["intermediate"]["score"]
    st["hof"].append({"id": champ["id"], "z": list(champ["z"]), "gen": g})
    if st["best"] is None or sel_score > st["best"]["sel_score"]:
        st["best"] = {"id": champ["id"], "z": list(champ["z"]), "gen": g, "sel_score": sel_score, "root": champ["root"]}
    rep = eval_panel(runner, champ["z"], ANCHORS, block_seeds(REPORT_BLOCK, a.report_seeds))
    relo, rse = anchor_elo({o: rep[o]["pairs"] for o in rep}, anchors)
    members = [{"id": m["id"], "parent": m["parent"], "born": m["born"], "elo": round(elo, 1), "elo_se": round(se, 1),
                "hparams": {"sigma": round(m["sigma"], 4)}, "island": m["island"],
                "shaping": {k: round(m["z"][NAMES.index(k)] * SCALE[NAMES.index(k)], 4) for k in SHOW},
                "matches": p["basic"]["n"] + p["intermediate"]["n"],
                "wins": p["basic"]["wins"] + p["intermediate"]["wins"],
                "win_rate": (p["basic"]["wins"] + p["intermediate"]["wins"]) / (p["basic"]["n"] + p["intermediate"]["n"]),
                "train_fitness": round(fitness(m, prior, a.shrink), 4), "games_total": m["n"], "steps": 0}
               for m, p, elo, se in rows]
    pop = {"gen": g, "time": time.time(), "anchors": anchors, "champion": champ["id"], "members": members,
           "islands": len(st["islands"]), "population": len(allm), "hof_size": len(st["hof"])}
    ch = {"gen": g, "id": champ["id"], "elo": round(relo, 1), "elo_se": round(rse, 1), "root": champ["root"],
          "vs": {o: {k: rep[o][k] for k in ("win_rate", "ci95", "n", "score")} for o in ANCHORS},
          "selection": {o: {k: cp[o][k] for k in ("win_rate", "n", "score")} for o in ("basic", "intermediate")},
          "z": [round(x, 5) for x in champ["z"]]}
    with open(out_dir / "population.jsonl", "a") as f:
        f.write(json.dumps(pop) + "\n")
    with open(out_dir / "champions.jsonl", "a") as f:
        f.write(json.dumps(ch) + "\n")
    st["champions"].append(ch)
    best = st["best"]
    (out_dir / "best.json").write_text(json.dumps({"z": best["z"], "id": best["id"], "gen": best["gen"],
                                                    "root": best["root"], "root_z": st["roots"][best["root"]],
                                                    "names": NAMES, "scale": SCALE}, indent=1))
    return ch


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--islands", type=int, default=120)
    ap.add_argument("--members", type=int, default=6)
    ap.add_argument("--per-table", type=int, default=3, help="members per 4-seat table; the rest are opponents")
    ap.add_argument("--generations", type=int, default=2000)
    ap.add_argument("--init", choices=("random", "zero", "intermediate"), default="random")
    ap.add_argument("--genome", choices=("base", "ext"), default="base", help="ext = agent_x extra genes")
    ap.add_argument("--init-from", help="genome json to start from (with --init intermediate)")
    ap.add_argument("--sigma0", type=float, default=0.3)
    ap.add_argument("--tau", type=float, default=0.3)
    ap.add_argument("--sigma-min", type=float, default=0.01)
    ap.add_argument("--opp-mode", choices=("mixed", "same"), default="mixed")
    ap.add_argument("--crossover", type=float, default=0.3)
    ap.add_argument("--replace", type=int, default=2)
    ap.add_argument("--shrink", type=float, default=4.0)
    ap.add_argument("--migrate", type=int, default=10)
    ap.add_argument("--hof-frac", type=float, default=0.33)
    ap.add_argument("--int-frac", type=float, default=0.5)
    ap.add_argument("--every", type=int, default=40)
    ap.add_argument("--candidates", type=int, default=6)
    ap.add_argument("--sel-seeds", type=int, default=15)
    ap.add_argument("--report-seeds", type=int, default=50)
    ap.add_argument("--procs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--runs-dir", default="runs")
    a = ap.parse_args(argv)
    assert a.members % a.per_table == 0 and 1 <= a.per_table <= 4, "members must split into full tables"
    out_dir = Path(a.runs_dir) / a.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    state_p = out_dir / "state.pkl"
    runner = Runner(a.procs)
    try:
        if a.resume:
            with open(state_p, "rb") as f:
                st, rng_state = pickle.load(f)
            rng = random.Random()
            rng.setstate(rng_state)
            gens = a.generations
            a = argparse.Namespace(**{**vars(ap.parse_args(["--run-id", a.run_id])), **st["args"], "generations": gens})
        else:
            rng = random.Random(a.seed)
            st = init_state(a, rng)
            st["anchors"] = anchor_ratings(runner, block_seeds(REPORT_BLOCK, 25000)[20000:20000 + a.report_seeds])
            meta = {"run_id": a.run_id, "kind": "evox", "mode": f"evolve_only (island GA, {a.islands} islands x "
                    f"{a.members} members, init={a.init}{' from ' + a.init_from if a.init_from else ''}, "
                    f"evox {a.genome} genome)",
                    "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "anchors": st["anchors"], "args": vars(a),
                    "genes": NAMES, "scale": SCALE,
                    "rating": "Elo vs fixed anchors (random=1000) from 1v3 games; members on selection block 56, "
                              "champions on report block 60"}
            (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
        anchors = st["anchors"]
        while st["gen"] < a.generations:
            t0 = time.time()
            row = generation(st, a, runner, rng)
            ckpt = (st["gen"] + 1) % a.every == 0 or st["gen"] == 0
            if ckpt:
                ch = checkpoint(st, a, runner, rng, out_dir, anchors)
                row["champion"] = {"id": ch["id"], "elo": ch["elo"], **{o: ch["vs"][o]["win_rate"] for o in ANCHORS}}
            row["wall_s"] = round(time.time() - t0, 2)
            with open(out_dir / "generations.jsonl", "a") as f:
                f.write(json.dumps(row) + "\n")
            st["gen"] += 1
            if ckpt:
                with open(state_p, "wb") as f:
                    pickle.dump((st, rng.getstate()), f)
                print(json.dumps(row), flush=True)
        with open(state_p, "wb") as f:
            pickle.dump((st, rng.getstate()), f)
    finally:
        runner.close()


if __name__ == "__main__":
    main()
