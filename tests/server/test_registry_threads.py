"""Concurrent first calls to the agent registry all see the torch plugins (rl, jointb_gated).

Server threads used to race in ``_load_plugins``: ``_loaded`` was set before the imports, so a
second request arriving during the (slow) torch import saw a partial registry, and the lab
answered "unknown agent"."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

SCRIPT = """
import threading
from catan.agents import registry
out = []
ts = [threading.Thread(target=lambda: out.append(set(registry.available_agents()))) for _ in range(8)]
[t.start() for t in ts]; [t.join() for t in ts]
assert len(out) == 8 and all({"rl", "jointb_gated", "search"} <= s for s in out), [sorted(s) for s in out]
print("ok")
"""


def test_concurrent_first_calls_see_all_plugins():
    r = subprocess.run([sys.executable, "-c", SCRIPT], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr[-2000:]
