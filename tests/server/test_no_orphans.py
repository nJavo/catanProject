"""The analysis rollout pool must not outlive the server.

Before the fix, killing a server during a deep analysis left 4 spawn workers
and a resource_tracker reparented to init/launchd for good."""
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _post(base, path, body, timeout=60):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def _group(pgid):
    """Live pids in process group ``pgid`` (the server's session)."""
    out = subprocess.run(["ps", "-axo", "pid=,pgid=,stat="], capture_output=True, text=True).stdout
    return [int(p) for p, g, st in (ln.split()[:3] for ln in out.splitlines() if ln.strip())
            if int(g) == pgid and not st.startswith("Z")]


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")
@pytest.mark.parametrize("sig", [signal.SIGKILL, signal.SIGTERM])
def test_killing_server_mid_deep_analysis_leaves_no_workers(tmp_path, sig):
    port = _port()
    env = dict(os.environ, CATAN_DATA_DIR=str(tmp_path), PYTHONPATH=str(ROOT), CATAN_ANALYSIS_MAX_WORKERS="2")
    proc = subprocess.Popen([sys.executable, "-m", "catan.server", "--port", str(port)], cwd=ROOT, env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(600):  # slow on a loaded machine
            try:
                urllib.request.urlopen(base + "/api/health")
                break
            except OSError:
                time.sleep(0.1)
        gid = _post(base, "/api/lab/matches", {"agents": ["random"] * 4, "seed": 3, "max_turns": 50})["id"]
        # deep = 2 worker processes here; long time limit so they are busy when the server dies
        threading.Thread(target=lambda: _quiet(_post, base, "/api/analysis/evaluate",
                         {"source": {"kind": "game", "ref": gid, "index": 0}, "viewer": 0, "depth": "deep",
                          "time_limit_ms": 60000}, 120), daemon=True).start()
        for _ in range(300):  # wait for the pool: server + resource tracker + workers
            if len(_group(proc.pid)) >= 4:
                break
            time.sleep(0.1)
        assert len(_group(proc.pid)) >= 4, "deep analysis did not start its worker pool"
        proc.send_signal(sig)  # only the server, as a crash or an impatient kill would
        proc.wait(20)
        for _ in range(100):  # workers poll their parent every 0.5 s
            if not _group(proc.pid):
                break
            time.sleep(0.1)
        left = _group(proc.pid)
        assert not left, f"orphaned processes after {sig.name}: {left}"
    finally:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):  # group already gone
            pass
        proc.wait(5)


def _quiet(fn, *a):
    try:
        fn(*a)
    except (OSError, urllib.error.URLError, ValueError):
        pass
