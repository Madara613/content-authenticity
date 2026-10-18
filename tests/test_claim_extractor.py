from unittest.mock import MagicMock, patch
import pytest

from backend.modules.claim_extractor import (
    clear_claim_cache,
    extract_claim,
    _resolve_gemini_model,
)


@pytest.fixture(autouse=True)
def reset_cache():
    """Ensure clean claim cache before and after each test."""
    clear_claim_cache()
    yield
    clear_claim_cache()


def test_extract_claim_success():
    """Valid factual claim is extracted as a single concise sentence."""
    mock_client = MagicMock()
    mock_response = MagicMock(text="Wildfire broke out in Ventura County on October 12, 2026.")
    mock_client.models.generate_content.return_value = mock_response

    with patch("backend.modules.claim_extractor.genai.Client", return_value=mock_client):
        claim = extract_claim(
            caption="Breaking news: huge fire burning right now! Wildfire broke out in Ventura County on October 12, 2026.",
            api_key="test_api_key_123",
        )

    assert claim == "Wildfire broke out in Ventura County on October 12, 2026."
    mock_client.models.generate_content.assert_called_once()


def test_extract_claim_null_for_opinion_or_description():
    """Pure description or opinion returns None."""
    mock_client = MagicMock()
    mock_response = MagicMock(text="null")
    mock_client.models.generate_content.return_value = mock_response

    with patch("backend.modules.claim_extractor.genai.Client", return_value=mock_client):
        claim = extract_claim(
            caption="Just a pretty golden retriever enjoying the warm morning sunshine.",
            api_key="test_api_key_123",
        )

    assert claim is None


def test_extract_claim_empty_caption_returns_none_without_api_call():
    """Empty or whitespace caption returns None immediately without calling Gemini."""
    with patch("backend.modules.claim_extractor.genai.Client") as mock_client_cls:
        assert extract_claim("") is None
        assert extract_claim("   ") is None
        mock_client_cls.assert_not_called()


def test_extract_claim_missing_api_key_returns_none_gracefully(monkeypatch):
    """Missing API key returns None without crashing."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with patch("backend.modules.claim_extractor.settings.gemini_api_key", ""):
        claim = extract_claim("The Federal Reserve cut interest rates today by 50 bps.")
        assert claim is None


def test_extract_claim_api_failure_returns_none_fail_safe():
    """If Gemini API raises an exception, return None without crashing."""
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = RuntimeError("Quota exceeded or network timeout")

    with patch("backend.modules.claim_extractor.genai.Client", return_value=mock_client):
        claim = extract_claim("A claim to verify", api_key="test_key")
        assert claim is None


def test_extract_claim_in_memory_caching():
    """Repeated extraction of same caption uses in-memory cache."""
    mock_client = MagicMock()
    mock_response = MagicMock(text="Earthquake struck off coast of Japan.")
    mock_client.models.generate_content.return_value = mock_response

    with patch("backend.modules.claim_extractor.genai.Client", return_value=mock_client):
        claim1 = extract_claim("Earthquake off Japan coast today.", api_key="test_key")
        claim2 = extract_claim("Earthquake off Japan coast today.", api_key="test_key")

    assert claim1 == claim2 == "Earthquake struck off coast of Japan."
    assert mock_client.models.generate_content.call_count == 1


def test_extract_claim_model_id_from_env(monkeypatch):
    """Verify Gemini model ID is resolved from GEMINI_MODEL env var."""
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-pro")
    assert _resolve_gemini_model() == "gemini-2.5-pro"
    assert _resolve_gemini_model("custom-model") == "custom-model"
