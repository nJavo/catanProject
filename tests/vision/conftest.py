import pytest


@pytest.fixture(autouse=True)
def no_gemini_key(monkeypatch, tmp_path):
    """No test may reach Gemini, even when the developer has a key in .env or the environment."""
    from catan.vision import llm
    monkeypatch.setattr(llm, "ENV_FILE", tmp_path / "no.env")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.setattr(llm.urllib.request, "urlopen", lambda *a, **k: pytest.fail("network call in a test"))
