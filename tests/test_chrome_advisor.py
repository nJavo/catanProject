"""tools/chrome-advisor: manifest limits, and the popup against a mocked lab server (Playwright)."""
import base64
import hashlib
import json
import re
from pathlib import Path

import pytest

from catan.engine.board import topology_dict
from catan.server.guard import ADVISOR_ORIGIN

EXT = Path(__file__).resolve().parents[1] / "tools" / "chrome-advisor"


def test_manifest_permissions_are_minimal():
    m = json.loads((EXT / "manifest.json").read_text())
    assert m["manifest_version"] == 3
    assert m["permissions"] == ["activeTab", "storage"]
    assert m["host_permissions"] == ["http://127.0.0.1:8000/*"]
    assert all(re.match(r"^http://(127\.0\.0\.1|localhost):\*/\*$", o) for o in m["optional_host_permissions"])
    assert not {"content_scripts", "externally_connectable", "web_accessible_resources"} & set(m)
    # the "key" pins the extension id that the server's guard lets through
    digest = hashlib.sha256(base64.b64decode(m["key"])).hexdigest()[:32]
    assert "chrome-extension://" + "".join(chr(97 + int(c, 16)) for c in digest) == ADVISOR_ORIGIN
    src = " ".join(p.read_text() for p in EXT.glob("*.js"))
    for banned in ("executeScript", "webRequest", "debugger", "WebSocket", "document.querySelector('canvas"):
        assert banned not in src


STUB = """
const store = (area) => ({ data: {}, async get(k) { const ks = typeof k === "string" ? [k] : k; return Object.fromEntries(ks.filter((x) => x in this.data).map((x) => [x, this.data[x]])); },
  async set(o) { Object.assign(this.data, o); for (const f of listeners) f(Object.fromEntries(Object.keys(o).map((k) => [k, { newValue: o[k] }])), area); } });
const listeners = [];
window.sent = [];
window.chrome = {
  storage: { local: store("local"), session: store("session"), onChanged: { addListener: (f) => listeners.push(f) } },
  tabs: { captureVisibleTab: async () => "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==" },
  action: { setBadgeText() {} },
  runtime: { sendMessage: (msg) => { window.sent.push(msg); import("./advisor.js").then((m) => m.analyze(msg)); } },
};
chrome.storage.local.data.myColor = "blue";
"""


def test_popup_against_mocked_server(tmp_path):
    sync_api = pytest.importorskip("playwright.sync_api")
    pytest.importorskip("PIL")
    from fastapi.testclient import TestClient
    from catan.server.app import create_app

    from catan.vision import llm
    from tests.vision.synth import fake, jpeg, main_spec, reading
    r = reading(main_spec())
    orig = llm.config, llm.gemini_client
    llm.config, llm.gemini_client = (lambda path=None: ("k", llm.DEFAULT_MODEL)), (lambda key, model: fake(r))
    try:
        res = TestClient(create_app(data_dir=tmp_path, static_dir=tmp_path / "x")).post(
            "/api/analysis/import-image", files={"image": ("b.jpg", jpeg(), "image/jpeg")},
            data={"my_color": "blue"}).json()
    finally:
        llm.config, llm.gemini_client = orig
    res["backend"] = "color"  # the popup's reader line for the offline colour tables
    res["usage"] = {"calls": 2, "total_tokens": 5100, "usd": 0.0123}  # as the Gemini reader reports it
    uploads = []

    def server(route):
        req = route.request
        if req.url.endswith("/api/topology"):
            return route.fulfill(json=topology_dict(), headers={"access-control-allow-origin": "*"})
        uploads.append(req.post_data_buffer)
        route.fulfill(json=res, headers={"access-control-allow-origin": "*"})

    with sync_api.sync_playwright() as pw:
        b = pw.chromium.launch()
        page = b.new_page()
        page.route("http://ext.test/**", lambda r: r.fulfill(path=str(EXT / r.request.url.split("/")[-1])))
        page.route("http://127.0.0.1:8000/**", server)
        page.add_init_script(STUB)
        page.goto("http://ext.test/popup.html")
        page.get_by_test_id("move").wait_for(timeout=10_000)
        move = page.get_by_test_id("move").inner_text()
        position = page.get_by_test_id("position").inner_text()
        reader = page.get_by_test_id("reader").inner_text()
        href = page.get_by_test_id("open").get_attribute("href")
        assert page.get_by_test_id("color").input_value() == "blue"
        assert "Doesn't read or control colonist.io" in page.content()
        page.locator("#hand input[data-res='3']").fill("2")
        page.get_by_test_id("again").click()
        page.wait_for_function("window.sent.length === 2")
        page.wait_for_timeout(300)
        b.close()

    first = res["suggestion"]["candidates"][0]["action"]["type"]
    assert {"BUILD_SETTLEMENT": "Settlement on ", "END_TURN": "End turn", "BUILD_ROAD": "Road along "}[first] in move
    assert "You: blue" in position and "Main turn" in position
    assert "colour tables" in reader and "$0.0123" in reader
    assert href.startswith("http://127.0.0.1:8000/#/analysis/import/")
    payload = json.loads(base64.urlsafe_b64decode(href.rsplit("/", 1)[1] + "=="))
    assert payload["position"] == res["position"] and payload["low_confidence"] == res["low_confidence"]
    assert len(uploads) == 2 and b'name="my_color"\r\n\r\nblue' in uploads[0]
    assert b'name="my_hand"\r\n\r\n0,0,0,2,0' in uploads[1]
