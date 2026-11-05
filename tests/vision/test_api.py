"""POST /api/analysis/import-image: boundary checks."""
import pytest
from fastapi.testclient import TestClient

pytest.importorskip("PIL")

from catan.server.analysis import IMAGE_BODY, MAX_IMAGE_BYTES  # noqa: E402
from catan.server.app import create_app  # noqa: E402

URL = "/api/analysis/import-image"


@pytest.fixture
def client(tmp_path):
    return TestClient(create_app(data_dir=tmp_path, static_dir=tmp_path / "nodist"))


@pytest.mark.parametrize("files,data,code", [
    ({"image": ("a.gif", b"GIF89a....", "image/gif")}, {}, 415),
    ({"image": ("a.png", b"not a png", "image/png")}, {}, 400),
    ({"other": ("a.png", b"x", "image/png")}, {}, 400),
    ({"image": ("a.png", b"x", "image/png")}, {"profile": "../etc"}, 400),
    ({"image": ("a.png", b"x", "image/png")}, {"profile": "nope"}, 400),
    ({"image": ("a.png", b"x", "image/png")}, {"num_players": "9"}, 400),
])
def test_rejects_bad_uploads(client, files, data, code):
    assert client.post(URL, files=files, data=data).status_code == code


def test_rejects_json_and_oversize(client):
    assert client.post(URL, json={"image": "x"}).status_code == 415
    big = b"\x89PNG" + b"0" * (MAX_IMAGE_BYTES + 10)
    assert client.post(URL, files={"image": ("a.png", big, "image/png")}).status_code == 413
    assert IMAGE_BODY < 10 * MAX_IMAGE_BYTES  # the guard's raised cap only applies to this path
    assert client.post("/api/analysis/validate", content=b"x" * 1_100_000,
                       headers={"content-type": "application/json"}).status_code == 413


@pytest.mark.parametrize("data", [{"my_color": "../x"}, {"my_color": "a" * 21}, {"my_hand": "1,2,3"},
                                  {"my_hand": "1,2,3,4,x"}, {"my_hand": "1,2,3,4,20"}, {"my_hand": "-1,0,0,0,0"}])
def test_rejects_bad_overrides(client, data):
    assert client.post(URL, files={"image": ("a.png", b"x", "image/png")}, data=data).status_code == 400
