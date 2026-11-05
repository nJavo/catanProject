"""Tournament runner.

Seat rotation: for each seed every lineup rotation is played on the same board
and dice seed ("cyclic": n rotations, each agent once per seat; "all": all n!
permutations), so agent comparisons are paired. Seeds come from a fixed
held-out block (``catan.eval.seeds``, the seed registry).

Agents are given as registry specs ``name`` or ``name:key=val,key2=val2``.
Illegal or crashing agent moves are caught, counted, logged and replaced by
a seeded random legal action. Agents with an ``observe(game, events)`` method
receive every step's events, filtered by ``visible_to`` for their seat.
With ``infoset=True`` agents get ``game.observe_for(seat)`` (catan.engine.infoset)
instead of the full game; reading hidden state then raises and counts as an
illegal move (``illegal_log`` shows the HiddenInformationError).
"""
from __future__ import annotations

import itertools
import json
import math
import os
import random
import subprocess
import time
import traceback
from collections import Counter
from multiprocessing import get_context

from catan.agents.registry import is_trusted, make_agent
from catan.engine import Game, GameConfig
from catan.engine.replay import make_replay

from .ratings import ratings
from .seeds import HELDOUT_BASE, block_seeds
from .stats import binom_two_sided_p, mean_ci, wilson

DEV_CARDS = ("knight", "victory_point", "road_building", "year_of_plenty", "monopoly")


def _coerce(v: str):
    low = v.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    for t in (int, float):
        try:
            return t(v)
        except ValueError:
            pass
    return v


def parse_spec(spec: str) -> tuple[str, dict]:
    """``"rl:checkpoint=a.pt,temp=0.5"`` -> ``("rl", {"checkpoint": "a.pt", "temp": 0.5})``."""
    name, _, rest = spec.partition(":")
    kw = {}
    if rest:
        for part in rest.split(","):
            k, eq, v = part.partition("=")
            if not eq or not k:
                raise ValueError(f"bad agent spec {spec!r}: expected key=val, got {part!r}")
            kw[k.strip()] = _coerce(v.strip())
    if not name:
        raise ValueError(f"bad agent spec {spec!r}")
    return name, kw


def heldout_seeds(seed: int, count: int) -> list[int]:
    """Seeds of held-out block ``seed`` (see catan.eval.seeds for block purposes)."""
    return block_seeds(seed, count)


def lineups(agents: list[str], num_players: int, rotation: str = "cyclic") -> list[tuple]:
    """Seat assignments played for every seed."""
    if not 1 <= len(agents) <= num_players:
        raise ValueError(f"need 1..{num_players} agents, got {len(agents)}")
    base = [agents[i % len(agents)] for i in range(num_players)]
    if rotation == "cyclic":
        out = [tuple(base[(i + r) % num_players] for i in range(num_players)) for r in range(num_players)]
    elif rotation == "all":
        # order by seat positions, not agent names: rotation r must mean the same seating pattern for
        # any specs, or paired comparisons keyed on (seed, rotation) pair different seatings
        out = [tuple(base[i] for i in p) for p in itertools.permutations(range(num_players))]
    elif rotation == "none":
        out = [tuple(base)]
    else:
        raise ValueError(f"unknown rotation {rotation!r}")
    # drop duplicate seatings (e.g. "basic random random random" rotations are distinct; A A A A is one)
    seen, uniq = set(), []
    for x in out:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


# Reserved spec key that marks a second, label-distinct copy of an agent (the gate's same-seat
# champion copy). It is stripped before the agent is built, so the copy plays exactly like the original.
COPY_KEY = "gate_copy"


def copy_spec(spec: str) -> str:
    """``spec`` with the copy marker: same agent, different label in tournament results."""
    return f"{spec}{',' if ':' in spec else ':'}{COPY_KEY}=1"


# ------------------------------------------------------------------ worker
_AGENT_CACHE: dict = {}


def _agent(spec: str, seat: int):
    key = (spec, seat)
    if key not in _AGENT_CACHE:
        name, kw = parse_spec(spec)
        kw.pop(COPY_KEY, None)
        _AGENT_CACHE[key] = make_agent(name, **kw)
    return _AGENT_CACHE[key]


def play_one(task: dict) -> dict:
    """Play one game; returns a JSON-safe record. Runs in worker processes."""
    seed, lineup, cfg = task["seed"], task["lineup"], GameConfig(**task["config"])
    n = cfg.num_players
    g = Game(config=cfg, seed=seed)
    agents = [_agent(s, i) for i, s in enumerate(lineup)]
    for i, a in enumerate(agents):
        a.reset(i, seed * 1000 + i + 17 * task["rotation"])
    # optional hook: agents that track public events (e.g. trade beliefs)
    observers = [(i, a.observe) for i, a in enumerate(agents) if callable(getattr(a, "observe", None))]
    # per seat: opt-in for everyone, mandatory for agents registered trusted=False
    infoset = [bool(task.get("infoset")) or not is_trusted(parse_spec(s)[0]) for s in lineup]
    fallback = random.Random(seed ^ 0x5EED)
    illegal = [0] * n
    illegal_log = []
    tsum, tmax, tcnt = [0.0] * n, [0.0] * n, [0] * n
    dev_bought = [0] * n
    dev_played = [Counter() for _ in range(n)]
    offers, offers_exec, partner_exec, counters, maritime = ([0] * n for _ in range(5))
    while not g.is_over:
        p = g.actor
        legal = g.legal_actions()
        t0 = time.perf_counter()
        err = None
        try:
            act = agents[p].act(g.observe_for(p) if infoset[p] else g, legal)
            ok = g.is_legal(act)
        except Exception as e:  # agent crashed: count as illegal
            act, ok, err = None, False, repr(e)
        dt = time.perf_counter() - t0
        tsum[p] += dt
        tcnt[p] += 1
        tmax[p] = max(tmax[p], dt)
        if not ok:
            illegal[p] += 1
            if len(illegal_log) < 20:
                illegal_log.append({"seat": p, "agent": lineup[p], "phase": g.phase, "turn": g.turn,
                                    "action": act.to_json() if hasattr(act, "to_json") else repr(act),
                                    "error": err})
            act = fallback.choice(legal)
        evs = g.step(act)
        for q, obs in observers:
            obs(g.observe_for(q) if infoset[q] else g,
                [e for e in evs if "visible_to" not in e or q in e["visible_to"]])
        for ev in evs:
            t = ev["type"]
            if t == "dev_bought":
                dev_bought[ev["player"]] += 1
            elif t == "dev_played":
                dev_played[ev["player"]][ev["card"]] += 1
            elif t == "trade_offered":
                offers[ev["player"]] += 1
            elif t == "trade_executed":
                offers_exec[ev["player"]] += 1
                partner_exec[ev["partner"]] += 1
            elif t == "trade_response" and ev["response"] == "counter":
                counters[ev["player"]] += 1
            elif t == "maritime_trade":
                maritime[ev["player"]] += 1
    rec = {
        "seed": seed, "rotation": task["rotation"], "lineup": list(lineup),
        "winner": g.winner, "truncated": g.truncated, "turns": g.turn,
        "actions": len(g.action_log),
        "vp": [g.victory_points(q) for q in range(n)],
        "longest_road": g.longest_road_owner, "largest_army": g.largest_army_owner,
        "dev_bought": dev_bought, "dev_played": [dict(c) for c in dev_played],
        "offers": offers, "offers_executed": offers_exec, "partner_executed": partner_exec,
        "counters": counters, "maritime": maritime,
        "illegal": illegal, "illegal_log": illegal_log,
        "time_sum": tsum, "time_max": tmax, "moves": tcnt,
    }
    if task.get("save_replay"):
        rec["replay"] = make_replay(seed, cfg.to_dict(), lineup, g.action_log,
                                    {"winner": g.winner, "turns": g.turn, "truncated": g.truncated, "vp": rec["vp"]})
    return rec


def _safe_play(task):
    try:
        return play_one(task)
    except Exception:
        return {"error": traceback.format_exc(), "seed": task["seed"], "rotation": task["rotation"]}


# ------------------------------------------------------------------ aggregate
def rank_scores(rec) -> list[float]:
    """Per-seat ranking key: winner first, then VP."""
    return [rec["vp"][q] + (100 if rec["winner"] == q else 0) for q in range(len(rec["vp"]))]


def aggregate(records: list[dict], labels: list[str]) -> dict:
    n = len(records[0]["lineup"])
    per = {}
    for lab in labels:
        games = [r for r in records if lab in r["lineup"]]
        seat_rows = [(r, q) for r in games for q in range(n) if r["lineup"][q] == lab]
        wins = sum(1 for r in games if r["winner"] >= 0 and r["lineup"][r["winner"]] == lab)
        ng = len(games)
        share = sum(r["lineup"].count(lab) for r in games) / (n * ng) if ng else 0
        lo, hi = wilson(wins, ng)
        moves = sum(r["moves"][q] for r, q in seat_rows)
        tsum = sum(r["time_sum"][q] for r, q in seat_rows)
        seats = {}
        for s in range(n):
            rows = [r for r in games if r["lineup"][s] == lab]
            w = sum(1 for r in rows if r["winner"] == s)
            if rows:
                seats[str(s)] = {"games": len(rows), "wins": w, "win_rate": w / len(rows),
                                 "ci95": list(wilson(w, len(rows)))}
        played = Counter()
        for r, q in seat_rows:
            played.update(r["dev_played"][q])
        ns = len(seat_rows)
        offers = sum(r["offers"][q] for r, q in seat_rows)
        oexec = sum(r["offers_executed"][q] for r, q in seat_rows)
        per[lab] = {
            "games": ng, "seat_games": ns, "wins": wins,
            "win_rate": wins / ng if ng else 0.0, "ci95": [lo, hi],
            "expected_win_rate_if_equal": share,
            "avg_vp": mean_ci([r["vp"][q] for r, q in seat_rows]),
            "seats": seats,
            "longest_road_freq": sum(1 for r, q in seat_rows if r["longest_road"] == q) / max(1, ns),
            "largest_army_freq": sum(1 for r, q in seat_rows if r["largest_army"] == q) / max(1, ns),
            "dev_bought_per_game": sum(r["dev_bought"][q] for r, q in seat_rows) / max(1, ns),
            "dev_played_per_game": {c: played[c] / max(1, ns) for c in DEV_CARDS},
            "trade": {
                "offers_per_game": offers / max(1, ns),
                "offers_executed_per_game": oexec / max(1, ns),
                "offer_success_rate": oexec / offers if offers else None,
                "accepted_as_partner_per_game": sum(r["partner_executed"][q] for r, q in seat_rows) / max(1, ns),
                "counters_per_game": sum(r["counters"][q] for r, q in seat_rows) / max(1, ns),
                "maritime_per_game": sum(r["maritime"][q] for r, q in seat_rows) / max(1, ns),
            },
            "illegal_actions": sum(r["illegal"][q] for r, q in seat_rows),
            "moves": moves,
            "sec_per_move": tsum / moves if moves else None,
            "max_sec_per_move": max((r["time_max"][q] for r, q in seat_rows), default=None),
        }
    # pairwise: in games with both, how often a finished ahead of b
    pair = {}
    for a, b in itertools.permutations(labels, 2):
        ahead = tie = tot = 0
        for r in records:
            if a not in r["lineup"] or b not in r["lineup"]:
                continue
            sc = rank_scores(r)
            sa = max(sc[q] for q in range(n) if r["lineup"][q] == a)
            sb = max(sc[q] for q in range(n) if r["lineup"][q] == b)
            tot += 1
            ahead += sa > sb
            tie += sa == sb
        if tot:
            k = ahead + 0.5 * tie
            decisive = tot - tie
            pair[f"{a} vs {b}"] = {"games": tot, "ahead": ahead, "ties": tie,
                                   "ahead_rate": k / tot, "ci95": list(wilson(k, tot)),
                                   "sign_test_p": binom_two_sided_p(ahead, decisive) if decisive else 1.0}
    turns = [r["turns"] for r in records]
    finished = [r["turns"] for r in records if not r["truncated"]]
    seat_w = [sum(1 for r in records if r["winner"] == s) for s in range(n)]
    glob = {
        "games": len(records),
        "truncation_rate": sum(r["truncated"] for r in records) / len(records),
        "turns": mean_ci(turns),
        "turns_finished_games": mean_ci(finished),
        "median_turns": sorted(turns)[len(turns) // 2],
        "seat_win_rate": [{"seat": s, "wins": seat_w[s], "rate": seat_w[s] / len(records),
                           "ci95": list(wilson(seat_w[s], len(records)))} for s in range(n)],
        "illegal_actions": sum(sum(r["illegal"]) for r in records),
    }
    rk = [[(r["lineup"][q], sc) for q, sc in enumerate(rank_scores(r))] for r in records]
    return {"per_agent": per, "pairwise": pair, "global": glob, "ratings": ratings(rk, labels)}


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def run_tournament(agents: list[str], games: int, seed: int = 0, workers: int = 1,
                   num_players: int = 4, max_turns: int = 500, rotation: str = "cyclic",
                   out_dir: str | None = "results", tid: str | None = None,
                   save_replays: int = 0, progress: bool = False, config_extra: dict | None = None,
                   infoset: bool = False) -> dict:
    for s in agents:
        parse_spec(s)  # fail fast on bad specs
    cfg = GameConfig(num_players=num_players, max_turns=max_turns, **(config_extra or {}))
    lus = lineups(agents, num_players, rotation)
    nseeds = max(1, math.ceil(games / len(lus)))
    seeds = heldout_seeds(seed, nseeds)
    tasks = [{"seed": s, "rotation": ri, "lineup": lu, "config": cfg.to_dict(),
              "save_replay": False, "infoset": infoset}
             for s in seeds for ri, lu in enumerate(lus)][:max(games, 1)]
    for t in tasks[:save_replays]:
        t["save_replay"] = True
    t0 = time.time()
    records = []
    try:  # forked workers inherit this; torch defaults to one thread per core (load 80+ with ~10 jobs)
        import torch
        torch.set_num_threads(1)
    except ImportError:
        pass
    if workers > 1:
        with get_context("spawn" if os.name == "nt" else "fork").Pool(workers) as pool:
            for i, rec in enumerate(pool.imap(_safe_play, tasks, chunksize=1)):
                records.append(rec)
                if progress and (i + 1) % max(1, len(tasks) // 20) == 0:
                    print(f"  {i + 1}/{len(tasks)} games, {time.time() - t0:.0f}s", flush=True)
    else:
        records = [_safe_play(t) for t in tasks]
    errors = [r for r in records if "error" in r]
    records = [r for r in records if "error" not in r]
    if not records:
        raise RuntimeError("all games failed:\n" + (errors[0]["error"] if errors else ""))
    labels = list(dict.fromkeys(agents))
    agg = aggregate(records, labels)
    tid = tid or time.strftime("%Y%m%d-%H%M%S") + "-" + "-".join(parse_spec(a)[0] for a in labels)
    result = {
        "id": tid, "agents": labels, "games": len(records), "seeds": seeds,
        "per_agent": agg["per_agent"], "pairwise": agg["pairwise"],
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "global": agg["global"], "ratings": agg["ratings"],
        "config": cfg.to_dict(), "rotation": rotation, "lineups": [list(x) for x in lus],
        "seed": seed, "seed_set": f"heldout base {HELDOUT_BASE} + seed*100000 + i",
        "workers": workers, "elapsed_sec": round(time.time() - t0, 1), "git_commit": _git_commit(),
        "failed_games": [{"seed": e["seed"], "rotation": e["rotation"], "error": e["error"][-2000:]}
                         for e in errors],
        "illegal_log": [x for r in records for x in r["illegal_log"]][:50],
        "game_summaries": [{k: r[k] for k in ("seed", "rotation", "lineup", "winner", "truncated",
                                              "turns", "vp")} for r in records],
    }
    if out_dir:
        os.makedirs(os.path.join(out_dir, "tournaments"), exist_ok=True)
        with open(os.path.join(out_dir, "tournaments", f"{tid}.json"), "w") as f:
            json.dump(result, f, indent=1)
        reps = [r["replay"] for r in records if "replay" in r]
        if reps:
            os.makedirs(os.path.join(out_dir, "replays"), exist_ok=True)
            for i, rp in enumerate(reps):
                with open(os.path.join(out_dir, "replays", f"{tid}-g{i}.json"), "w") as f:
                    json.dump(rp, f)
    return result


def format_table(res: dict) -> str:
    lines = [f"Tournament {res['id']}: {res['games']} games, rotation={res['rotation']}, "
             f"truncation={res['global']['truncation_rate']:.1%}, "
             f"turns={res['global']['turns']['mean']:.1f}",
             f"{'agent':<28}{'games':>6}{'wins':>6}{'win%':>8}{'95% CI':>17}{'avgVP':>7}"
             f"{'Elo':>7}{'ms/move':>9}{'illegal':>8}"]
    for lab, s in res["per_agent"].items():
        ci = s["ci95"]
        lines.append(f"{lab:<28}{s['games']:>6}{s['wins']:>6}{s['win_rate']:>8.1%}"
                     f"{f'[{ci[0]:.1%},{ci[1]:.1%}]':>17}{s['avg_vp']['mean']:>7.2f}"
                     f"{res['ratings'][lab]['elo']:>7.0f}{1000 * (s['sec_per_move'] or 0):>9.2f}"
                     f"{s['illegal_actions']:>8}")
    for k, v in res["pairwise"].items():
        a, b = k.split(" vs ")
        if list(res["per_agent"]).index(a) < list(res["per_agent"]).index(b):
            lines.append(f"  {k}: ahead {v['ahead_rate']:.1%} [{v['ci95'][0]:.1%},{v['ci95'][1]:.1%}] "
                         f"n={v['games']} p={v['sign_test_p']:.2g}")
    return "\n".join(lines)

