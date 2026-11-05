"""Request guard for a no-auth local server (docs/SECURITY.md).

- Host allowlist: blocks DNS rebinding, where a web page's own domain is
  re-pointed at 127.0.0.1 so the browser treats this server as same-origin.
- Cross-origin writes: a non-GET request whose Origin differs from its Host is
  refused. A page on another site can send a "simple" POST without CORS
  preflight; this stops it from driving games or writing saves.
- Extension origins: ``ext_origins`` (the move-advisor Chrome extension,
  tools/chrome-advisor/) may POST to ``ext_paths`` only, and get a CORS
  ``Access-Control-Allow-Origin`` echo there. Every other origin and path is
  still refused, and no other response carries CORS headers.
- Body size cap: request bodies are buffered up to ``max_body`` bytes, then 413.
  ``path_max_body`` raises the cap for named paths (the screenshot upload).
"""
from __future__ import annotations

import json
import os
from urllib.parse import urlsplit

LOOPBACK = {"127.0.0.1", "localhost", "::1"}
SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}
# tools/chrome-advisor/manifest.json pins this id with its "key"; CATAN_EXTENSION_ORIGINS adds others.
ADVISOR_ORIGIN = "chrome-extension://iohkkbaieljdpkhpniegbceehkkbimio"


def extension_origins_from_env() -> set[str]:
    """``CATAN_EXTENSION_ORIGINS``: comma list of extra ``chrome-extension://<id>`` origins."""
    raw = os.environ.get("CATAN_EXTENSION_ORIGINS", "")
    return {ADVISOR_ORIGIN} | {o.strip().lower() for o in raw.split(",") if o.strip().lower().startswith("chrome-extension://")}


def allowed_hosts_from_env() -> set[str] | None:
    """``CATAN_ALLOWED_HOSTS``: comma list of hostnames, or ``*`` to disable the check."""
    raw = os.environ.get("CATAN_ALLOWED_HOSTS", "").strip()
    if raw == "*":
        return None
    # "testserver" is Starlette's TestClient host; a single-label name an attacker can't register.
    return LOOPBACK | {"testserver"} | {h.strip().lower() for h in raw.split(",") if h.strip()}


def _hostname(host: str) -> str:
    host = host.strip().lower()
    if host.startswith("["):  # [::1]:8000
        return host[1:].split("]")[0]
    return host.rsplit(":", 1)[0] if host.count(":") == 1 else host


class RequestGuard:
    def __init__(self, app, allowed_hosts: set[str] | None, max_body: int = 1_000_000, path_max_body: dict | None = None,
                 ext_origins: set[str] = frozenset(), ext_paths: set[str] = frozenset()):
        self.app, self.allowed, self.max_body, self.path_max = app, allowed_hosts, max_body, path_max_body or {}
        self.ext_origins, self.ext_paths = ext_origins, ext_paths

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope["headers"]}
        host = headers.get("host", "")
        if self.allowed is not None and _hostname(host) not in self.allowed:
            return await _reply(send, 400, "unknown Host header (set CATAN_ALLOWED_HOSTS to serve other names)")
        method = scope["method"]
        origin = headers.get("origin")
        ext = origin is not None and origin.lower() in self.ext_origins and scope["path"] in self.ext_paths
        if ext:
            send = _with_cors(send, origin)
        elif method not in SAFE_METHODS and origin is not None and urlsplit(origin).netloc.lower() != host.lower():
            return await _reply(send, 403, "cross-origin request refused")
        if method in SAFE_METHODS:
            return await self.app(scope, receive, send)
        limit = self.path_max.get(scope["path"], self.max_body)
        try:
            declared = int(headers.get("content-length", "0") or 0)
        except ValueError:
            return await _reply(send, 400, "bad Content-Length")
        if declared > limit:
            return await _reply(send, 413, f"request body over {limit} bytes")
        # Buffer the body (also catches chunked uploads without Content-Length).
        body, more = b"", True
        while more:
            msg = await receive()
            if msg["type"] == "http.disconnect":
                return
            body += msg.get("body", b"")
            more = msg.get("more_body", False)
            if len(body) > limit:
                return await _reply(send, 413, f"request body over {limit} bytes")
        sent = False

        async def replay():
            nonlocal sent
            if not sent:
                sent = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()
        return await self.app(scope, replay, send)


def _with_cors(send, origin: str):
    async def wrapped(msg):
        if msg["type"] == "http.response.start":
            msg = {**msg, "headers": list(msg.get("headers", [])) + [
                (b"access-control-allow-origin", origin.encode("latin-1")), (b"vary", b"Origin")]}
        await send(msg)
    return wrapped


async def _reply(send, status: int, detail: str):
    data = json.dumps({"detail": detail}).encode()
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(data)).encode())]})
    await send({"type": "http.response.body", "body": data})
