"""Test inputs made at test time: bot-game PositionSpecs and a plain image to upload."""
import io
import json

from catan.analysis import position_from_game
from catan.engine import Phase

from tests.analysis.helpers import play_until

COLORS = ["red", "blue", "orange", "white"]


def setup_spec(seed=1):
    """7 settlements and 7 roads down: the 8th setup settlement is next."""
    g = play_until(seed, lambda g: g.phase == Phase.SETUP_SETTLEMENT and sum(1 for o in g.vertex_owner if o >= 0) == 7)
    return position_from_game(g, 0)


def main_spec(seed=2, turn=10):
    return position_from_game(play_until(seed, lambda g: g.turn >= turn), 0)


def jpeg(w=1280, h=800):
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (20, 70, 100)).save(buf, "JPEG")
    return buf.getvalue()


def reading(truth):
    """What a perfect Gemini reading of ``truth`` looks like."""
    from catan.vision import llm
    colors = COLORS[:truth["num_players"]]
    r = llm.spec_to_reading(truth, colors)
    r["board_points"] = {"hex0": {"x": 300, "y": 200}, "hex18": {"x": 600, "y": 800}}
    r["panel"] = {"players": [{"color": c} for c in colors]}  # panel order = seat order
    return r


def fake(*replies):
    """A Gemini client that returns ``replies`` in turn and records what it was sent."""
    sent = []

    def call(parts):
        sent.append(parts)
        return json.dumps(replies[len(sent) - 1]), {"promptTokenCount": 2000, "candidatesTokenCount": 500, "totalTokenCount": 2500}
    call.sent = sent
    return call
