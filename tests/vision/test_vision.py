"""Screenshot import: input checks, the quota assignment and the phase guess.

The colour reader's accuracy on rendered boards is in docs/VISION_IMPORT.md.
"""
import io

import pytest

pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from catan.vision import import_image, board_to_spec  # noqa: E402
from catan.vision.detect import assign  # noqa: E402


def test_rejects_non_images():
    with pytest.raises(ValueError):
        import_image(b"GIF89a not an image at all", "ours")
    buf = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buf, "BMP")
    with pytest.raises(ValueError, match="unsupported image format"):
        import_image(buf.getvalue(), "ours")


def test_assign_respects_quota():
    import numpy as np
    prob = np.array([[0.9, 0.1], [0.8, 0.2], [0.6, 0.4]])
    assert assign(prob, {"a": 1, "b": 2}, ("a", "b")) == [0, 1, 1]


def test_board_to_spec_phase_guess():
    b = {"layout": {}, "vertex_owner": [-1] * 54, "vertex_level": [0] * 54, "edge_owner": [-1] * 72, "robber": 0}
    assert board_to_spec(b)["phase"] == "SETUP_SETTLEMENT"
    b["vertex_owner"][0], b["vertex_level"][0] = 0, 1
    s = board_to_spec(b)
    assert (s["phase"], s["viewer"]) == ("SETUP_ROAD", 0)
    b["vertex_level"][0] = 2
    assert board_to_spec(b)["phase"] == "MAIN"

