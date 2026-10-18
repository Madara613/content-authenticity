import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from backend.modules.evidence import (
    clear_evidence_cache,
    evidence_signal,
    grounded_check,
    local_lookup,
)
from backend.schemas import SignalResult


@pytest.fixture(autouse=True)
def reset_cache():
    """Ensure clean evidence cache before and after each test."""
    clear_evidence_cache()
    yield
    clear_evidence_cache()


def _build_mock_gemini_grounded_response(
    verdict: str,
    confidence: float,
    reason: str,
    sources: list,
    wrap_in_code_block: bool = False,
):
    """Build a mock Gemini response with text and grounding metadata."""
    payload = {
        "verdict": verdict,
        "confidence": confidence,
        "reason": reason,
    }
    json_str = json.dumps(payload)
    if wrap_in_code_block:
        json_str = f"```json\n{json_str}\n```"

    mock_chunks = []
    for s in sources:
        mock_web = MagicMock(title=s.get("title", ""), uri=s.get("url", ""))
        mock_chunks.append(MagicMock(web=mock_web))

    mock_gm = MagicMock(grounding_chunks=mock_chunks)
    mock_candidate = MagicMock(grounding_metadata=mock_gm)
    mock_response = MagicMock(
        text=json_str,
        candidates=[mock_candidate],
    )
    return mock_response


def test_grounded_check_supported_with_sources():
    """Verify grounded_check returns supported verdict when corroborated by web sources."""
    sources = [{"title": "Reuters", "url": "https://reuters.com/news1"}]
    mock_resp = _build_mock_gemini_grounded_response("supported", 0.90, "Confirmed by Reuters report.", sources)

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_resp

    with patch("backend.modules.evidence.genai.Client", return_value=mock_client):
        res = grounded_check("NASA launched Europa Clipper mission.", api_key="test_key")

    assert res["verdict"] == "supported"
    assert res["confidence"] == 0.90
    assert "Confirmed" in res["reason"]
    assert len(res["sources"]) == 1
    assert res["sources"][0]["title"] == "Reuters"


def test_grounded_check_contradicted_with_sources():
    """Verify grounded_check returns contradicted verdict when debunked by sources."""
    sources = [{"title": "AP Fact Check", "url": "https://apnews.com/fact-check"}]
    mock_resp = _build_mock_gemini_grounded_response("contradicted", 0.95, "Debunked by AP Fact Check.", sources)

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_resp

    with patch("backend.modules.evidence.genai.Client", return_value=mock_client):
        res = grounded_check("Eiffel Tower was destroyed in fire.", api_key="test_key")

    assert res["verdict"] == "contradicted"
    assert res["confidence"] == 0.95
    assert len(res["sources"]) == 1


def test_grounded_check_no_sources_forces_verdict_unclear():
    """If there are no grounding sources, verdict must be forced to 'unclear'."""
    mock_resp = _build_mock_gemini_grounded_response(
        verdict="supported",
        confidence=0.85,
        reason="Claim appears plausible.",
        sources=[],  # No sources!
    )

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_resp

    with patch("backend.modules.evidence.genai.Client", return_value=mock_client):
        res = grounded_check("Some uncorroborated assertion", api_key="test_key")

    assert res["verdict"] == "unclear"
    assert res["confidence"] <= 0.30
    assert len(res["sources"]) == 0


def test_grounded_check_strips_markdown_code_fences():
    """Grounded check properly strips ```json markdown fences."""
    sources = [{"title": "BBC", "url": "https://bbc.com/article"}]
    mock_resp = _build_mock_gemini_grounded_response(
        "supported", 0.88, "Corroborated by BBC.", sources, wrap_in_code_block=True
    )

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_resp

    with patch("backend.modules.evidence.genai.Client", return_value=mock_client):
        res = grounded_check("Claim with code fences", api_key="test_key")

    assert res["verdict"] == "supported"
    assert res["confidence"] == 0.88


def test_local_lookup_with_mock_faiss():
    """Verify local_lookup queries FAISS and filters results by min_sim."""
    mock_model = MagicMock()
    mock_model.encode.return_value = np.zeros((1, 384), dtype=np.float32)

    mock_index = MagicMock()
    # Return distances: [0.85, 0.55] and indices: [0, 1]
    mock_index.search.return_value = (
        np.array([[0.85, 0.55]], dtype=np.float32),
        np.array([[0, 1]], dtype=np.int64),
    )

    metadata = [
        {"statement": "Vaccines cause microchips.", "label": "pants-fire", "speaker": "blogger"},
        {"statement": "Economy grew by 2% last quarter.", "label": "true", "speaker": "economist"},
    ]

    with patch("backend.modules.evidence._get_local_index_components", return_value=(mock_model, mock_index, metadata)):
        matches = local_lookup("Vaccines contain tracking devices", min_sim=0.60)

    # 0.85 >= 0.60 (included), 0.55 < 0.60 (excluded)
    assert len(matches) == 1
    assert matches[0]["statement"] == "Vaccines cause microchips."
    assert matches[0]["label"] == "pants-fire"
    assert matches[0]["similarity"] == 0.85


def test_local_lookup_missing_index_returns_empty_list():
    """If FAISS index has not been built, local_lookup returns empty list safely."""
    with patch("backend.modules.evidence._get_local_index_components", return_value=(None, None, None)):
        matches = local_lookup("Any statement")
        assert matches == []


def test_evidence_signal_supported_low_suspicion():
    """Supported claim produces low suspicion score and high confidence."""
    mock_web = {
        "verdict": "supported",
        "confidence": 0.90,
        "reason": "Official NASA mission records corroborate launch.",
        "sources": [{"title": "NASA", "url": "https://nasa.gov/europa"}],
    }

    with patch("backend.modules.evidence.grounded_check", return_value=mock_web), \
         patch("backend.modules.evidence.local_lookup", return_value=[]):
        sig = evidence_signal("NASA launched Europa Clipper.", api_key="test_key")

    assert isinstance(sig, SignalResult)
    assert sig.name == "claim_evidence"
    assert sig.score <= 0.20
    assert sig.confidence >= 0.70
    assert "supports" in sig.explanation.lower()
    assert len(sig.details["sources"]) == 1


def test_evidence_signal_contradicted_high_suspicion():
    """Contradicted claim produces high suspicion score and high confidence."""
    mock_web = {
        "verdict": "contradicted",
        "confidence": 0.92,
        "reason": "Debunked by police statements and video evidence.",
        "sources": [{"title": "Reuters", "url": "https://reuters.com/factcheck"}],
    }

    with patch("backend.modules.evidence.grounded_check", return_value=mock_web), \
         patch("backend.modules.evidence.local_lookup", return_value=[]):
        sig = evidence_signal("False event claim", api_key="test_key")

    assert sig.name == "claim_evidence"
    assert sig.score >= 0.80
    assert sig.confidence >= 0.70
    assert "contradicts" in sig.explanation.lower()


def test_evidence_signal_unclear_gives_low_confidence():
    """Unclear or uncorroborated evidence must produce LOW confidence and neutral suspicion."""
    mock_web = {
        "verdict": "unclear",
        "confidence": 0.25,
        "reason": "No credible reporting found.",
        "sources": [],
    }

    with patch("backend.modules.evidence.grounded_check", return_value=mock_web), \
         patch("backend.modules.evidence.local_lookup", return_value=[]):
        sig = evidence_signal("An obscure unverified rumor", api_key="test_key")

    assert sig.name == "claim_evidence"
    assert sig.score == 0.50
    assert sig.confidence <= 0.35, f"Expected low confidence for unclear evidence, got {sig.confidence}"
    assert "inconclusive" in sig.explanation.lower() or "uncorroborated" in sig.explanation.lower()


def test_evidence_signal_empty_claim_returns_confidence_zero():
    """If no claim is provided, return confidence 0.0 instead of crashing."""
    sig1 = evidence_signal(None)
    assert sig1.name == "claim_evidence"
    assert sig1.confidence == 0.0
    assert sig1.score == 0.50

    sig2 = evidence_signal("   ")
    assert sig2.confidence == 0.0


def test_evidence_signal_missing_api_key_returns_confidence_zero(monkeypatch):
    """Missing API key returns signal with confidence 0.0."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with patch("backend.modules.evidence.settings.gemini_api_key", ""):
        sig = evidence_signal("A claim to verify")
        assert sig.name == "claim_evidence"
        assert sig.confidence == 0.0
        assert "not configured" in sig.explanation.lower()
