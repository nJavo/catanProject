"""python -m catan.server [--host 127.0.0.1] [--port 8000] [--online]

The server has no authentication: anyone who can reach it can read every hand
and drive every seat. Keep the default loopback host unless you trust the whole
network (docs/SECURITY.md).

``--online`` turns on private rooms with secret seat links: every game route
then needs a room token and every other API route the admin key printed at
startup (docs/ONLINE.md).
"""
import argparse
import os
import sys

import uvicorn

from catan.server.guard import LOOPBACK


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--online", action="store_true", help="rooms with secret seat links; see docs/ONLINE.md")
    a = ap.parse_args()
    if a.online:
        os.environ["CATAN_ONLINE"] = "1"  # read by create_app when uvicorn imports catan.server.app
        from catan.server.app import app
        print(f"Online mode. On this machine open http://127.0.0.1:{a.port}/?admin={app.state.admin_key} "
              "to create rooms and use the lab. Keep that key private; share only seat links.", file=sys.stderr)
        if a.host not in LOOPBACK:
            print(f"Note: binding to {a.host}. Prefer the default 127.0.0.1 plus Tailscale or a tunnel "
                  "(docs/ONLINE.md); traffic to this port is plain HTTP.", file=sys.stderr)
    elif a.host not in LOOPBACK:
        print(f"WARNING: binding to {a.host}. This server has no authentication: anyone on the network can see "
              "every hand, drive every seat and write saves. Requests must also use a Host listed in "
              "CATAN_ALLOWED_HOSTS (comma list, or '*').", file=sys.stderr)
    # bounded graceful shutdown: a running analysis must not keep SIGTERM waiting for minutes
    uvicorn.run("catan.server.app:app", host=a.host, port=a.port, timeout_graceful_shutdown=5)


if __name__ == "__main__":
    main()
