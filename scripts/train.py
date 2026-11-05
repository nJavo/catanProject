"""Train a masked-PPO Catan agent.

    python scripts/train.py --config configs/ppo_baseline.yaml [--run-id NAME]
    python scripts/train.py --resume runs/<run_id>
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml  # noqa: E402

from catan.rl.ppo import load_config, merge, DEFAULTS, train  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--resume", help="run directory (or run id under runs/) to resume")
    ap.add_argument("--run-id")
    ap.add_argument("--total-steps", type=int, help="override total_steps (e.g. to extend on resume)")
    a = ap.parse_args(argv)
    if a.resume:
        run_dir = a.resume if os.path.isdir(a.resume) else os.path.join("runs", a.resume)
        with open(os.path.join(run_dir, "config.yaml")) as f:
            cfg = merge(DEFAULTS, yaml.safe_load(f))
    elif a.config:
        cfg = load_config(a.config)
        run_dir = os.path.join("runs", a.run_id or f"{cfg['run_name']}_{time.strftime('%Y%m%d_%H%M%S')}")
        if os.path.exists(os.path.join(run_dir, "config.yaml")):  # a dir without a config (only logs) is fine
            ap.error(f"{run_dir} exists; use --resume")
    else:
        ap.error("--config or --resume required")
    if a.total_steps:
        cfg["total_steps"] = a.total_steps
    print(f"run dir: {run_dir}", flush=True)
    train(cfg, run_dir, resume=bool(a.resume), log=lambda s: print(s, flush=True))


if __name__ == "__main__":
    main()
