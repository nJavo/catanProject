"""Analysis engine: candidate moves, model outputs and rollout statistics.

Honesty rules: ``policy_prob``, ``value_estimate``,
``search_score`` and ``rollout`` are separate fields; a method that does not
produce a number leaves it ``None``. Hidden information is never read from a
true game state: every analysis starts from a PositionSpec and works on
determinizations (``position.determinize``).
"""
from __future__ import annotations

import atexit
import os
import random
import threading
import time
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FuturesTimeout
import multiprocessing as mp

from catan.agents.registry import make_agent
from catan.bots.basic import HeuristicBase
from catan.bots.belief import HandPosterior
from catan.engine import Action, Game, GameConfig
from catan.eval.stats import wilson

from .labels import label
from .position import ASSUMPTIONS, assumptions, determinize, position_from_game, validate_position, derived

DEPTHS = {
    # samples = determinizations, rollouts = rollouts per determinization per candidate
    "quick": {"samples": 4, "rollouts": 0, "top_k": 5, "workers": 1},
    "standard": {"samples": 4, "rollouts": 8, "top_k": 5, "workers": 1},
    "deep": {"samples": 8, "rollouts": 25, "top_k": 8, "workers": 4},
}
DEFAULT_POLICY = "intermediate"
DEFAULT_ROLLOUT_POLICY = "basic"
ROLLOUT_TURNS = 200       # rollout cap, in turns after the analysed position
ROLLOUT_MAX_STEPS = 20000  # safety cap on actions per rollout (counted as truncated)
# Verdict thresholds, in win-rate points below the best candidate. A verdict
# only fires when the 95% Wilson CIs do not overlap (candidate hi < best lo).
INACCURACY, MISTAKE, BLUNDER = 0.05, 0.10, 0.20
MAX_WORKERS = int(os.environ.get("CATAN_ANALYSIS_MAX_WORKERS", 4))  # CPU budget; tests use 2

# shortcut: one lock serialises rollout analyses (shared cached agents, CPU budget); use per-request agents if concurrency matters.
_LOCK = threading.RLock()
_AGENTS: dict[str, object] = {}
# Quick depth (no rollouts) has its own lock and agent cache, so it never waits
# behind a deep analysis and never shares an agent object with one.
_QUICK_LOCK = threading.RLock()
_QUICK_AGENTS: dict[str, object] = {}
_POOL = None
CANCEL_POLL_S = 0.2
_on_lock_wait = None  # test hook: called when a request starts waiting for an engine lock


class Cancelled(Exception):
    """Raised when a request's cancel event is set."""


def _check(cancel):
    if cancel is not None and cancel.is_set():
        raise Cancelled()


class _locked:
    """``with _locked(lock, cancel)``: acquire ``lock``, giving up if cancelled while waiting."""
    def __init__(self, lock, cancel):
        self.lock, self.cancel = lock, cancel

    def __enter__(self):
        _check(self.cancel)
        if self.lock.acquire(blocking=False):
            return
        if _on_lock_wait is not None:
            _on_lock_wait()
        while not self.lock.acquire(timeout=CANCEL_POLL_S):
            _check(self.cancel)

    def __exit__(self, *exc):
        self.lock.release()


# ----------------------------------------------------------------- policies
def parse_policy(spec: str) -> tuple[str, dict]:
    """``"rl:checkpoint=runs/x.pt,deterministic=true"`` -> ("rl", {...})."""
    if not isinstance(spec, str) or not spec:
        raise ValueError("policy must be a non-empty string")
    name, _, rest = spec.partition(":")
    kw = {}
    for part in filter(None, rest.split(",")):
        k, eq, v = part.partition("=")
        if not eq:
            raise ValueError(f"bad policy option {part!r} (expected key=value)")
        low = v.lower()
        if low in ("true", "false"):
            kw[k] = low == "true"
        else:
            try:
                kw[k] = int(v)
            except ValueError:
                try:
                    kw[k] = float(v)
                except ValueError:
                    kw[k] = v
    return name, kw


def get_agent(spec: str, cache: dict | None = None):
    """Cached agent instance for a policy spec (checkpoints load once)."""
    cache = _AGENTS if cache is None else cache
    ag = cache.get(spec)
    if ag is None:
        name, kw = parse_policy(spec)
        ag = make_agent(name, **kw)
        if len(cache) > 16:
            cache.clear()
        cache[spec] = ag
    return ag


def _is_rl(agent):
    return hasattr(agent, "model") and hasattr(agent, "meta") and not _is_search(agent)


def _is_search(agent):
    from catan.research.exit.search import ExitSearchAgent
    return isinstance(agent, ExitSearchAgent)


def _own_choice(agent):
    """Agents scored by asking them for their move: everything except heuristic bots and plain RL policies
    (search agents and their trade/robber/setup wrappers, combo GNNs, ...)."""
    return not isinstance(agent, HeuristicBase) and not _is_rl(agent)


def _agent_decision(agent, g, me, seed):
    """The agent's own move on the viewer's information-set view: hidden state raises, so it cannot peek."""
    agent.reset(me, seed)
    view = g.observe_for(me)
    return agent.act_with_info(view, view.legal_actions())


SEARCH_VALUE_KIND = {
    "value": "value-net win probability at the search leaves (mean over its simulations)",
    "heur": "intermediate.evaluate margin at the search leaves (heuristic units, not a win probability)",
    "mix": "value-net win probability + heuristic margin / mix_scale at the search leaves",
}


_STATIC: dict = {}


def _static(lay):
    from catan.rl.spaces import static_features
    st = _STATIC.get(lay)
    if st is None:
        _STATIC.clear()
        st = _STATIC[lay] = static_features(lay)
    return st


def _obs(games, me, agent=None):
    """Observations as the agent encodes them: ``arch_rl`` models carry their own (obs v2) encoder."""
    import numpy as np
    import torch
    from catan.rl.spaces import encode_obs
    enc = getattr(agent, "encode", encode_obs)
    st = _static(games[0].layout)
    return torch.from_numpy(np.stack([enc(g, me, st) for g in games]))


def _mask(agent, g, me, legal):
    """The agent's own action mask, including the action filters it was trained with."""
    from catan.rl.env import filter_mask
    from catan.rl.spaces import action_mask
    if callable(getattr(agent, "mask", None)):
        return agent.mask(g, legal)
    m = action_mask(legal, me, g.n)
    return filter_mask(m, g, me, agent.env_cfg) if getattr(agent, "env_cfg", None) is not None else m


def _rl_policy(agent, g, me, legal):
    """(probability of each legal action, value head) at ``g`` for seat ``me``."""
    import torch
    from catan.rl.spaces import encode_action
    with torch.no_grad():
        logits, v = agent.model(_obs([g], me, agent), torch.from_numpy(_mask(agent, g, me, legal))[None])
    p = torch.softmax(logits[0].float(), -1)
    return {a: float(p[encode_action(a, me, g.n)]) for a in legal}, float(v[0])


def _rl_values(agent, games, me):
    """Value head for seat ``me`` in each game (batched)."""
    import torch
    with torch.no_grad():
        _, v = agent.model(_obs(games, me, agent))
    return [float(x) for x in v]


def _rl_attribution(agent, g, me, action):
    """Integrated-gradient attribution of the policy's logit for ``action`` and of the value, grouped by
    observation block and by board entity (catan/rl/explain.attribute). The viewer's observation reads
    only public information and the viewer's own cards, so any determinization gives the same input."""
    from catan.rl.explain import CAVEATS, attribute
    from catan.rl.spaces import encode_action
    out = attribute(agent.model, _obs([g], me, agent)[0], encode_action(action, me, g.n), len(_static(g.layout)), action)
    out["action"] = action.to_json()
    out["caveats"] = CAVEATS[:3]
    return out


# ----------------------------------------------------------------- rollouts
def _play_out(c: Game, agent, seed: int):
    agent.reset(0, seed)
    steps = 0
    while not c.is_over and steps < ROLLOUT_MAX_STEPS:
        c.step(agent.act(c, c.legal_actions()))
        steps += 1
    return c


def _rollout_batch(det: Game, action, seeds, agent, me, cancel=None):
    """Rollouts of ``action`` (None = let the policy move) from one determinization."""
    out = []
    for s in seeds:
        _check(cancel)
        c = det.clone()
        c.rng = random.Random(s)
        if action is not None:
            c.step(action)
        _play_out(c, agent, s)
        trunc = not c.is_over or c.truncated
        out.append((int(c.winner == me), int(trunc), c.victory_points(me)))
    return out


def _worker(det_dict, action_json, seeds, policy, me):
    det = Game.from_dict(det_dict)
    a = Action.from_json(action_json) if action_json is not None else None
    return _rollout_batch(det, a, seeds, get_agent(policy), me)


def _exit_with_parent(parent: int):
    """Worker initializer. If the server dies without shutting the pool down
    (SIGKILL, crash), workers would be reparented and live forever, keeping the
    resource tracker alive too. Exit as soon as the parent is gone."""
    def watch():
        while os.getppid() == parent:
            time.sleep(0.5)
        os._exit(0)
    threading.Thread(target=watch, daemon=True, name="parent-watchdog").start()


def _pool(workers):
    """Persistent spawn pool (worker start-up and checkpoint loads are paid once)."""
    global _POOL
    if _POOL is None or _POOL._max_workers != workers:
        if _POOL is not None:
            shutdown_pool()
        _POOL = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
                                    initializer=_exit_with_parent, initargs=(os.getpid(),))
    return _POOL


def shutdown_pool():
    """Stop the rollout pool now, killing workers mid-rollout (server shutdown)."""
    global _POOL
    pool, _POOL = _POOL, None
    if pool is None:
        return
    procs = list((getattr(pool, "_processes", None) or {}).values())
    pool.shutdown(wait=False, cancel_futures=True)
    for pr in procs:
        if pr.is_alive():
            pr.terminate()
    for pr in procs:
        pr.join(2)


atexit.register(shutdown_pool)


def _rollout_stats(results, batches=None):
    """``batches``: the same results split per determinization (AN-006). Their
    win-rate spread separates "which hidden hands" from rollout noise."""
    n = len(results)
    if n == 0:
        return None
    w = sum(r[0] for r in results)
    lo, hi = wilson(w, n)
    out = {"win_rate": round(w / n, 4), "ci95": [round(lo, 4), round(hi, 4)], "n": n,
           "mean_vp": round(sum(r[2] for r in results) / n, 3), "truncated": sum(r[1] for r in results)}
    rates = [sum(r[0] for r in b) / len(b) for b in (batches or []) if b]
    if len(rates) >= 2:
        m = sum(rates) / len(rates)
        out["between_sample_sd"] = round((sum((x - m) ** 2 for x in rates) / (len(rates) - 1)) ** 0.5, 4)
        out["samples"] = len(rates)
    else:
        out["between_sample_sd"] = None
        out["between_sample_sd_reason"] = "fewer than 2 determinizations finished"
    return out


def _run_rollouts(dets, actions, seeds, policy, me, workers, deadline=None, cancel=None, batches=None):
    """({action or None: [results]}, timed_out). Common random numbers across
    actions; jobs go determinization-major so a deadline cuts all actions evenly.
    ``cancel`` (threading.Event) is polled every CANCEL_POLL_S; on cancel the
    pool's busy workers are killed so the CPU is free at once."""
    jobs = [(i, a) for i in range(len(dets)) for a in actions]
    res = {a: [] for a in actions}
    if batches is not None:  # per-determinization results, filled alongside ``res``
        batches.update({a: [] for a in actions})
    if workers > 1 and len(jobs) > 1:
        dd = [d.to_dict() for d in dets]
        futs = [(a, _pool(workers).submit(_worker, dd[i], None if a is None else a.to_json(), seeds[i], policy, me))
                for i, a in jobs]
        for k, (a, f) in enumerate(futs):
            while True:
                left = None if deadline is None else deadline - time.perf_counter()
                try:
                    b = f.result(CANCEL_POLL_S if left is None else max(0.0, min(CANCEL_POLL_S, left)))
                    res[a].extend(b)
                    if batches is not None:
                        batches[a].append(b)
                    break
                except FuturesTimeout:
                    if cancel is not None and cancel.is_set():
                        for _, g in futs[k:]:
                            g.cancel()
                        shutdown_pool()  # running jobs can't be cancelled: stop their workers
                        raise Cancelled()
                    if left is not None and left <= CANCEL_POLL_S:
                        for _, g in futs[k:]:
                            g.cancel()
                        if any(g.running() for _, g in futs[k:]):
                            shutdown_pool()  # running jobs would keep the workers busy after we return
                        return res, True
    else:
        agent = get_agent(policy)
        for i, a in jobs:
            if deadline is not None and time.perf_counter() > deadline:
                return res, True
            b = _rollout_batch(dets[i], a, seeds[i], agent, me, cancel)
            res[a].extend(b)
            if batches is not None:
                batches[a].append(b)
    return res, False


# ----------------------------------------------------------------- verdicts
def verdict(cand_roll, best_roll) -> str:
    if cand_roll is None or best_roll is None:
        return "unrated"
    gap = best_roll["win_rate"] - cand_roll["win_rate"]
    separated = cand_roll["ci95"][1] < best_roll["ci95"][0]
    if gap < INACCURACY:
        return "good"
    if not separated:
        return "unrated"
    if gap >= BLUNDER:
        return "blunder"
    if gap >= MISTAKE:
        return "mistake"
    return "inaccuracy"


# ----------------------------------------------------------------- analyze
def analyze(spec: dict, policy: str = DEFAULT_POLICY, rollout_policy: str = DEFAULT_ROLLOUT_POLICY,
            depth: str = "standard", top_k: int | None = None, seed: int = 0, include=(),
            samples: int | None = None, rollouts: int | None = None, workers: int | None = None,
            time_limit_s: float | None = None, belief=None, cancel=None, attribution: bool = True) -> dict:
    """Rank candidate moves for the viewer (who must be the actor). ``include``
    forces extra actions into the candidate list (e.g. the move actually played).
    ``belief`` (a ``HandPosterior`` for the viewer, from the game's public events) samples
    opponents' hands from the posterior instead of uniformly."""
    if depth not in DEPTHS:
        raise ValueError(f"depth must be one of {list(DEPTHS)}")
    errs = validate_position(spec)
    if errs:
        raise ValueError("invalid position: " + "; ".join(errs))
    cfg = DEPTHS[depth]
    top_k = int(top_k or cfg["top_k"])
    D = int(samples or cfg["samples"])
    R = cfg["rollouts"] if rollouts is None else int(rollouts)
    workers = min(MAX_WORKERS, int(workers or cfg["workers"]))
    quick = R == 0
    cache = _QUICK_AGENTS if quick else None
    with _locked(_QUICK_LOCK if quick else _LOCK, cancel):
        t0 = time.perf_counter()
        me = spec["viewer"]
        d = derived(spec)
        if d["actor"] != me:
            raise ValueError(f"it is seat {d['actor']}'s decision, not the viewer's ({me}); "
                             "analysis needs the viewer's own decision")
        dets = determinize(spec, D, seed, max_turns=spec["turn"] + ROLLOUT_TURNS, belief=belief)
        g0 = dets[0]
        legal = g0.legal_actions()
        agent = get_agent(policy, cache)
        probs = scores = None
        value_pos = None
        source = "policy"
        kind = "learned_policy"
        choice, sinfo, root = None, {}, {}
        want_attr, attribution = attribution, None
        if _own_choice(agent):
            choice, sinfo = _agent_decision(agent, g0, me, seed)
            root = {Action.from_json(r["action"]): r for r in sinfo.get("root") or []}
            kind = "search" if root else "agent"
        if root:
            source = "search_root"
            # the agent's pick first, then the search's own preference (mean leaf value, then visits)
            order = sorted(root, key=lambda a: (a != choice, -(root[a]["q"] if root[a]["q"] is not None else -1e9),
                                                -root[a]["n"]))
            order = [a for a in order if a in set(legal)]
            if choice not in order:  # e.g. the trade head replaced the search's pick with an offer
                order.insert(0, choice)
            score_kind = None
        elif _is_rl(agent):
            probs, value_pos = _rl_policy(agent, g0, me, legal)
            order = sorted(legal, key=lambda a: -probs[a])
            score_kind = None
            if want_attr:
                attribution = _rl_attribution(agent, g0, me, order[0])
        else:
            scorer = agent if isinstance(agent, HeuristicBase) else get_agent("intermediate", cache)
            if scorer is agent:
                kind = "heuristic"
            else:
                source = "intermediate_pruning"
            scorer.reset(me, seed)
            raw = {}
            legal_set = set(legal)
            for s, a in scorer.score(g0.clone(), legal):
                if a in legal_set and (a not in raw or s > raw[a]):
                    raw[a] = float(s)
            # only actions the heuristic scored are candidates (it skips e.g. most domestic offers)
            order = sorted(raw, key=lambda a: -raw[a]) or legal
            scores = raw if scorer is agent else None
            score_kind = f"{policy} heuristic score (VP-like units; higher is better)" if scores else None
            if scores and policy.split(":")[0] == "search":
                score_kind = ("search: mean rollout leaf value (intermediate.evaluate units) when it searched; "
                              "intermediate heuristic score when the heuristic was already decisive")
        cands = order[:max(top_k, len(root))]  # keep every root candidate of a search
        if choice is not None and choice not in cands and choice in set(legal):
            cands.append(choice)
        for a in include:
            a = a if isinstance(a, Action) else Action.from_json(a)
            if a not in cands:
                if a not in set(legal):
                    raise ValueError(f"included action {a} is not legal here")
                cands.append(a)
        # value head after each candidate, averaged over determinizations
        values = {}
        if probs is not None:
            after = []
            for a in cands:
                for det in dets:
                    c = det.clone()
                    c.step(a)
                    after.append(c)
            live = [c for c in after if not c.is_over]
            vs = _rl_values(agent, live, me) if live else []
            it = iter(vs)
            flat = [(1.0 if c.winner == me else 0.0) if c.is_over else next(it) for c in after]
            for i, a in enumerate(cands):
                chunk = flat[i * D:(i + 1) * D]
                values[a] = sum(chunk) / len(chunk)
        t1 = time.perf_counter()
        roll = {}
        pos_roll = None
        timed_out = False
        if R > 0:
            srng = random.Random(seed ^ 0x5EED)
            seeds = [[srng.getrandbits(63) for _ in range(R)] for _ in range(D)]
            deadline = None if time_limit_s is None else t0 + time_limit_s
            bt: dict = {}
            res, timed_out = _run_rollouts(dets, [None] + cands, seeds, rollout_policy, me, workers, deadline, cancel,
                                           batches=bt)
            pos_roll = _rollout_stats(res[None], bt[None])
            roll = {a: _rollout_stats(res[a], bt[a]) for a in cands}
        t2 = time.perf_counter()
        out = []
        for a in cands:
            out.append({"action": a.to_json(), "label": label(g0, a),
                        "policy_prob": None if probs is None else round(probs[a], 4),
                        "value_estimate": None if a not in values else round(values[a], 4),
                        # AN-006: the value head has never been calibrated (RL-014), so say so per number
                        "value_estimate_calibrated": None if a not in values else False,
                        "search_score": None if scores is None or a not in scores else round(scores[a], 4),
                        "agent_choice": a == choice,
                        "search_value": root[a]["q"] if a in root else None,
                        # same caveat as the value head: leaf values are model outputs, never calibrated (RL-014)
                        "search_value_calibrated": False if a in root and root[a]["q"] is not None else None,
                        "search_visits": root[a]["n"] if a in root else None,
                        "rollout": roll.get(a)})
        prior_rank = {a: i for i, a in enumerate(cands)}
        if roll:
            out.sort(key=lambda c: (c["rollout"] is None, -(c["rollout"] or {}).get("win_rate", 0),
                                    -(c["rollout"] or {}).get("mean_vp", 0),
                                    prior_rank[Action.from_json(c["action"])]))
            best = out[0]["rollout"]
            for i, c in enumerate(out):
                c["rank"] = i + 1
                c["verdict"] = "best" if i == 0 and best else verdict(c["rollout"], best)
        else:
            for i, c in enumerate(out):
                c["rank"] = i + 1
                c["verdict"] = "unrated"
        t3 = time.perf_counter()
        return {
            "position": spec, "actor": me, "phase": spec["phase"], "legal_count": len(legal),
            "policy": policy, "rollout_policy": rollout_policy if R > 0 else None, "depth": depth,
            "position_eval": {"value_estimate": None if value_pos is None else round(value_pos, 4),
                              "value_estimate_calibrated": None if value_pos is None else False,
                              "rollout": pos_roll},
            "candidates": out,
            "candidate_source": source, "search_score_kind": score_kind,
            "policy_kind": kind,
            "agent_choice": None if choice is None else choice.to_json(),
            "agent_info": None if choice is None else {
                k: sinfo[k] for k in ("source", "sims", "candidates", "trade_layer", "replaced") if k in sinfo},
            "search_value_kind": SEARCH_VALUE_KIND.get(sinfo.get("leaf")) if root else None,
            "ranking": "rollout win rate (ties: mean VP)" if roll else
                       ("search: agent's choice, then search value" if root else
                        "policy probability" if probs is not None else "heuristic score"),
            "determinization": {"samples": D, "rollouts_per_sample": R, "seed": seed,
                                "rollout_turn_cap": ROLLOUT_TURNS if R else None,
                                "truncation": "truncated rollouts count as losses; counted in rollout.truncated",
                                "hand_sampling": "belief" if belief is not None else "uniform",
                                "assumptions": assumptions(belief)},
            "belief": belief_summary(belief),
            "attribution": attribution,
            "verdict_thresholds": {"inaccuracy": INACCURACY, "mistake": MISTAKE, "blunder": BLUNDER,
                                   "rule": "only when the candidate's 95% CI upper bound is below the best's lower bound"},
            "timed_out": timed_out,
            "timing_ms": {"policy": round((t1 - t0) * 1000, 1), "rollouts": round((t2 - t1) * 1000, 1),
                          "total": round((t3 - t0) * 1000, 1)},
        }


def belief_summary(belief) -> dict | None:
    """Opponent-hand posterior for the UI, labeled as an estimate from public events."""
    if belief is None:
        return None
    return {"label": "estimate from public events", "method": "exact posterior over hands (catan/bots/belief.py)",
            "states": len(belief.states), "approx": belief.approx, "unseen_steals": belief.unseen_steals,
            "opponents": {str(q): m for q, m in belief.marginals().items()}}


def legal_moves(spec: dict) -> list[dict]:
    errs = validate_position(spec)
    if errs:
        raise ValueError("invalid position: " + "; ".join(errs))
    d = derived(spec)
    if d["actor"] != spec["viewer"]:
        raise ValueError(f"it is seat {d['actor']}'s decision, not the viewer's ({spec['viewer']})")
    g = determinize(spec, 1, 0)[0]
    return [{**a.to_json(), "label": label(g, a)} for a in g.legal_actions()]


# ----------------------------------------------------------------- game timeline
def _metric(c, rl):
    if c.get("rollout"):
        return "rollout_win_rate", c["rollout"]["win_rate"]
    if rl and c.get("value_estimate") is not None:
        return "value_estimate", c["value_estimate"]
    if c.get("search_value") is not None:
        return "search_value", c["search_value"]
    return "search_score", c.get("search_score")


def _eval_obj(c):
    ro = c.get("rollout") or {}
    return {"win_rate": ro.get("win_rate"), "ci95": ro.get("ci95"), "n": ro.get("n", 0),
            "value_estimate": c.get("value_estimate"), "search_score": c.get("search_score"),
            "search_value": c.get("search_value")}


def analyze_game(source, viewer: int, policy: str = DEFAULT_POLICY, depth: str = "quick",
                 rollout_policy: str = DEFAULT_ROLLOUT_POLICY, seed: int = 0, top_k: int | None = None,
                 belief: bool = False, cancel=None) -> dict:
    """Timeline over ``viewer``'s decisions in a finished/live game or replay
    dict (``{seed, config, actions}``). Forced moves (one legal action) are skipped."""
    if isinstance(source, Game):
        actions = list(source.action_log)
        # Passing layout= skips the layout RNG draws, so only do it for custom layouts.
        g = Game(config=source.config, seed=source.seed)
        if g.layout != source.layout:
            g = Game(config=source.config, seed=source.seed, layout=source.layout)
    else:
        actions = [Action.from_json(a) for a in source.get("actions") or []]
        cfg = GameConfig(**source["config"]) if isinstance(source.get("config"), dict) else GameConfig()
        g = Game(config=cfg, seed=source["seed"])
    if not 0 <= viewer < g.n:
        raise ValueError("viewer out of range")
    t0 = time.perf_counter()
    post = HandPosterior(g.n, viewer) if belief else None
    timeline, skipped = [], 0
    rl = _is_rl(get_agent(policy, _QUICK_AGENTS if DEPTHS[depth]["rollouts"] == 0 else None))
    for i, played in enumerate(actions):
        _check(cancel)
        if g.actor == viewer and played.player == viewer:
            legal = g.legal_actions()
            if len(legal) <= 1:
                skipped += 1
            else:
                spec = position_from_game(g, viewer)
                res = analyze(spec, policy, rollout_policy, depth, top_k, seed + i, include=[played], attribution=False,
                              belief=post if post is not None and not post.lost else None, cancel=cancel)
                cs = res["candidates"]
                best = cs[0]
                pj = played.to_json()
                pc = next(c for c in cs if c["action"] == pj)
                metric, before = _metric(best, rl)
                _, after = _metric(pc, rl)
                delta = None if before is None or after is None else round(after - before, 4)
                timeline.append({"index": i, "actor": viewer, "turn": g.turn, "phase": g.phase,
                                 "played": pj, "played_label": pc["label"],
                                 "best": best["action"], "best_label": best["label"],
                                 "eval_before": _eval_obj(best), "eval_after": _eval_obj(pc),
                                 "delta": delta, "metric": metric,
                                 "verdict": "best" if pc is best else pc["verdict"],
                                 "legal_count": res["legal_count"]})
        ev = g.step(played)
        if post is not None:
            post.observe(g, ev)
    counts = {k: sum(1 for t in timeline if t["verdict"] == k)
              for k in ("best", "good", "inaccuracy", "mistake", "blunder", "unrated")}
    n = len(timeline)
    return {"viewer": viewer, "policy": policy, "depth": depth,
            "rollout_policy": rollout_policy if DEPTHS[depth]["rollouts"] else None,
            "timeline": timeline,
            "summary": {"decisions": n, "skipped_forced": skipped,
                        "accuracy": round(counts["best"] / n, 4) if n else None,
                        "accuracy_definition": "share of analysed decisions where the played move was ranked best",
                        "inaccuracies": counts["inaccuracy"], "mistakes": counts["mistake"],
                        "blunders": counts["blunder"], "unrated": counts["unrated"],
                        "good": counts["good"], "best": counts["best"]},
            "timing_ms": {"total": round((time.perf_counter() - t0) * 1000, 1)}}


# ----------------------------------------------------------------- what-if
def continue_position(spec: dict, action, policy: str = DEFAULT_ROLLOUT_POLICY, steps: int | None = None,
                      until: str | None = None, seed: int = 0, cancel=None) -> dict:
    """Apply ``action`` (the viewer's move) in one determinization, then let
    ``policy`` play every seat for ``steps`` actions or ``until`` turn_end /
    game_end (default: turn_end). Returns the new position from the viewer's view."""
    if until not in (None, "turn_end", "game_end"):
        raise ValueError("until must be 'turn_end' or 'game_end'")
    if steps is not None and (type(steps) is not int or not 0 <= steps <= 5000):
        raise ValueError("steps must be 0..5000")
    if steps is None and until is None:
        until = "turn_end"
    errs = validate_position(spec)
    if errs:
        raise ValueError("invalid position: " + "; ".join(errs))
    viewer = spec["viewer"]
    if derived(spec)["actor"] != viewer:
        raise ValueError("the position must be the viewer's decision")
    with _locked(_LOCK, cancel):
        g = determinize(spec, 1, seed, max_turns=spec["turn"] + ROLLOUT_TURNS)[0]
        a = action if isinstance(action, Action) else Action.from_json(action)
        if not g.is_legal(a):
            raise ValueError(f"illegal action {a.to_json()} in this position")
        agent = get_agent(policy)
        agent.reset(0, seed)
        log = [{"action": a.to_json(), "label": label(g, a)}]
        events = g.step(a)
        start_turn = spec["turn"]
        n = 0
        cap = steps if steps is not None else 5000
        while not g.is_over and n < cap:
            if until == "turn_end" and g.turn > start_turn:
                break
            if n % 50 == 0:
                _check(cancel)
            b = agent.act(g, g.legal_actions())
            log.append({"action": b.to_json(), "label": label(g, b)})
            events += g.step(b)
            n += 1
        vis = [e for e in events if e.get("visible_to") is None or viewer in e["visible_to"]]
        return {"position": position_from_game(g, viewer), "events": vis, "actions": log,
                "done": g.is_over, "winner": g.winner if g.is_over else None, "truncated": g.truncated,
                "assumptions": ASSUMPTIONS}
