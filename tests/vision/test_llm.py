"""Gemini backend: the model's (hex, corner/side) words <-> engine ids, the repair loop, fallback. No network."""
import json
import math

import pytest
from fastapi.testclient import TestClient

pytest.importorskip("PIL")

from catan.engine import TOPOLOGY as T  # noqa: E402
from catan.server.app import create_app  # noqa: E402
from catan.vision import llm  # noqa: E402

from .synth import COLORS, fake, jpeg, main_spec, reading as _reading, setup_spec  # noqa: E402

# bearings in degrees, y up, of a pointy-top hex's corners and of its side midpoints
CORNER_DEG = {"N": 90, "NE": 30, "SE": -30, "S": -90, "SW": -150, "NW": 150}
SIDE_DEG = {"NE": 60, "E": 0, "SE": -60, "SW": -120, "W": 180, "NW": 120}


def bearing(dx, dy):
    return math.degrees(math.atan2(-dy, dx))  # engine y points down


def close(a, b):
    return abs((a - b + 180) % 360 - 180) < 1


@pytest.mark.parametrize("h", range(19))
def test_every_corner_and_side_points_where_its_name_says(h):
    hx, hy = T.hex_xy[h]
    for c in llm.CORNERS:
        v = llm.corner_vertex(h, c)
        x, y = T.vertex_xy[v]
        assert close(bearing(x - hx, y - hy), CORNER_DEG[c]), (h, c)
        assert llm.corner_vertex(*llm.vertex_name(v)) == v
    for s in llm.SIDES:
        e = llm.side_edge(h, s)
        (ax, ay), (bx, by) = (T.vertex_xy[i] for i in T.edge_vertices[e])
        assert close(bearing((ax + bx) / 2 - hx, (ay + by) / 2 - hy), SIDE_DEG[s]), (h, s)
        assert llm.side_edge(*llm.edge_name(e)) == e


def test_reading_order_is_rows_3_4_5_4_3():
    ys = [round(y, 3) for _, y in T.hex_xy]
    assert [ys.count(y) for y in sorted(set(ys))] == [3, 4, 5, 4, 3]
    assert all((T.hex_xy[a][1], T.hex_xy[a][0]) < (T.hex_xy[a + 1][1], T.hex_xy[a + 1][0]) for a in range(18))
    assert all(llm.vertex_name(v) for v in range(T.num_vertices)) and all(llm.edge_name(e) for e in range(T.num_edges))


@pytest.mark.parametrize("make", [setup_spec, main_spec, lambda: main_spec(seed=5, turn=40)], ids=["setup", "main", "late"])
def test_positions_round_trip_through_the_models_words(make):
    truth = make()
    r = llm.spec_to_reading(truth, COLORS)
    b, probs = llm.reading_to_board(r, COLORS)
    assert probs == []
    for k in ("vertex_owner", "vertex_level", "edge_owner", "robber"):
        assert b[k] == truth[k], k
    lay = truth["layout"]
    assert b["layout"]["hex_terrain"] == lay["hex_terrain"] and b["layout"]["hex_number"] == lay["hex_number"]
    assert dict(zip(b["layout"]["port_edges"], b["layout"]["port_types"])) == dict(zip(lay["port_edges"], lay["port_types"]))


def reading(spec=setup_spec):
    truth = spec()
    return _reading(truth), truth


def test_clean_reading_needs_one_call():
    r, truth = reading()
    r["panel"] = {"players": [{"color": c} for c in COLORS], "my_color": "blue", "turn_color": "blue", "dice": [3, 4],
                  "my_hand": None, "my_dev_cards": None}
    client = fake(r)
    out = llm.import_image_llm(jpeg(), client=client, model="gemini-3.5-flash-lite")
    assert len(client.sent) == 1 and not out["repaired"] and out["errors"] == [], out["checks"]
    assert client.sent[0][1]["inlineData"]["mimeType"] == "image/jpeg"
    assert out["position"]["vertex_owner"] == truth["vertex_owner"] and out["colors"] == {c: i for i, c in enumerate(COLORS)}
    assert out["lattice"]["r"] > 0 and out["low_confidence"]["hexes"] == []
    assert out["usage"]["calls"] == 1 and out["usage"]["usd"] == pytest.approx((2000 * 0.30 + 500 * 2.50) / 1e6)


def test_bad_reading_triggers_one_repair_and_flags_what_stays_wrong():
    good, _ = reading()
    bad = json.loads(json.dumps(good))
    bad["roads"].append({"color": "red", "hex": 9, "side": "E"})          # a road touching nothing of red's
    bad["hexes"][0], bad["hexes"][1] = {"terrain": "ore", "number": 6}, {"terrain": "ore", "number": 6}
    client = fake(bad, bad)
    out = llm.import_image_llm(jpeg(), client=client)
    assert len(client.sent) == 2  # never more than one repair
    fix = client.sent[1][2]["text"]
    assert "not connected" in fix and "hex 9 side E" in fix and "ore" in fix and "both have a red number" in fix
    assert out["errors"] and llm.side_edge(9, "E") in out["low_confidence"]["edges"]
    assert {0, 1} <= set(out["low_confidence"]["hexes"])
    # a good repair is taken
    out = llm.import_image_llm(jpeg(), client=fake(bad, good))
    assert out["repaired"] and out["errors"] == [] and out["usage"]["calls"] == 2


def test_panel_maps_to_seats_and_hands():
    r, truth = reading(main_spec)  # past setup
    r["panel"] = {"players": [{"color": c, "cards": 3, "dev_cards": 1, "knights": 1} for c in COLORS],
                  "my_color": "orange", "turn_color": "white", "dice": [6, 2],
                  "my_hand": {"wood": 1, "brick": 2, "sheep": 0, "wheat": 0, "ore": 1}, "my_dev_cards": None}
    spec, colors, unread = llm.build_spec(r, 4)
    assert colors == COLORS and spec["last_roll"] == [6, 2] and spec["phase"] == "MAIN"
    assert spec["viewer"] == 2 and spec["current"] == 3 and spec["players"][2]["hand"] == [1, 2, 0, 0, 1]
    assert spec["players"][0]["resource_count"] == 3 and spec["players"][0]["dev_count"] == 1
    assert "hands" not in unread and "current" not in unread and "dev_cards" in unread
    assert spec["vertex_owner"] == truth["vertex_owner"]


def test_env_parser_without_dotenv(tmp_path, monkeypatch):
    p = tmp_path / ".env"
    p.write_text("# comment\nGEMINI_API_KEY='abc'\nexport GEMINI_MODEL=gemini-3.8-flash\n")
    monkeypatch.setitem(__import__("sys").modules, "dotenv", None)
    assert llm.read_env(p) == {"GEMINI_API_KEY": "abc", "GEMINI_MODEL": "gemini-3.8-flash"}
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    assert llm.config(p) == ("abc", "gemini-3.8-flash")
    assert llm.config(tmp_path / "none") == (None, llm.DEFAULT_MODEL)


URL = "/api/analysis/import-image"


@pytest.fixture
def client(tmp_path):
    return TestClient(create_app(data_dir=tmp_path, static_dir=tmp_path / "nodist"))


def post(client, **data):
    return client.post(URL, files={"image": ("b.jpg", jpeg(), "image/jpeg")}, data=data)


def test_no_key_falls_back_to_colour(client, monkeypatch):
    monkeypatch.setattr(llm, "config", lambda path=None: (None, llm.DEFAULT_MODEL))
    j = post(client).json()
    assert j["backend"] == "color" and "GEMINI_API_KEY" in j["backend_note"] and j["profile"] == "ours"
    assert post(client, backend="nope").status_code == 400


def test_endpoint_uses_gemini_when_a_key_is_set(client, monkeypatch):
    r, _ = reading()
    monkeypatch.setattr(llm, "config", lambda path=None: ("k", "gemini-3.5-flash-lite"))
    monkeypatch.setattr(llm, "gemini_client", lambda key, model: fake(r))
    j = post(client).json()
    assert j["backend"] == "llm" and j["profile"] == "gemini:gemini-3.5-flash-lite" and j["errors"] == []
    assert j["usage"]["calls"] == 1 and j["suggestion"]["candidates"]
    assert post(client, backend="color").json()["backend"] == "color"
    monkeypatch.setattr(llm, "gemini_client", lambda key, model: (_ for _ in ()).throw(llm.LLMError("boom")))
    assert post(client).status_code == 502


def test_gemini_request_shape_and_key_stays_in_the_header(monkeypatch):
    import io
    import urllib.error
    seen = {}

    def urlopen(req, timeout):
        seen.update(url=req.full_url, headers=dict(req.header_items()), body=json.loads(req.data), timeout=timeout)
        return io.BytesIO(json.dumps({"candidates": [{"content": {"parts": [{"text": "{}"}]}}],
                                      "usageMetadata": {"promptTokenCount": 1}}).encode())
    monkeypatch.setattr(llm.urllib.request, "urlopen", urlopen)
    text, usage = llm.gemini_client("SECRET", "gemini-3.5-flash-lite")([{"text": "hi"}])
    assert text == "{}" and usage == {"promptTokenCount": 1} and seen["timeout"] == llm.TIMEOUT
    assert seen["url"].startswith("https://generativelanguage.googleapis.com/") and "SECRET" not in seen["url"]
    assert seen["headers"]["X-goog-api-key"] == "SECRET"
    gc = seen["body"]["generationConfig"]
    assert gc["responseMimeType"] == "application/json" and gc["responseJsonSchema"] == llm.SCHEMA

    def fail(req, timeout):
        raise urllib.error.HTTPError(req.full_url, 400, "bad", {}, io.BytesIO(b'{"error": {"message": "API key not valid"}}'))
    monkeypatch.setattr(llm.urllib.request, "urlopen", fail)
    with pytest.raises(llm.LLMError, match="HTTP 400: API key not valid") as e:
        llm.gemini_client("SECRET", "m")([{"text": "hi"}])
    assert "SECRET" not in str(e.value)


def test_colour_aliases_fold_to_one_seat():
    from catan.vision.llm import seat_colors
    r = {"buildings": [{"color": "Ember"}, {"color": "orange"}, {"color": "tide"}],
         "roads": [{"color": "blue"}, {"color": "Heather"}, {"color": "purple"}]}
    assert seat_colors(r) == ["orange", "blue", "purple"]
