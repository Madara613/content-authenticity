from unittest.mock import MagicMock, patch
import pytest
import torch

from tests.conftest import make_jpeg_bytes
from backend.modules.clip_consistency import (
    analyze_clip_consistency,
    clear_clip_cache,
    get_clip_components,
)
from backend.schemas import SignalResult


@pytest.fixture(autouse=True)
def reset_cache():
    """Ensure clean cache before and after each test."""
    clear_clip_cache()
    yield
    clear_clip_cache()


def _build_mock_clip(cosine_target: float):
    """Create a mock CLIP model and processor that outputs the desired cosine similarity."""
    mock_processor = MagicMock()
    mock_processor.return_value = {
        "pixel_values": torch.zeros((1, 3, 224, 224)),
        "input_ids": torch.zeros((1, 10), dtype=torch.long),
    }

    mock_model = MagicMock()
    # Return 2D embeddings such that their dot product equals cosine_target
    # E.g. image = [1.0, 0.0], text = [cosine_target, sqrt(1 - cosine_target^2)]
    img_feat = torch.tensor([[1.0, 0.0]])
    txt_y = (max(0.0, 1.0 - cosine_target ** 2)) ** 0.5
    txt_feat = torch.tensor([[cosine_target, txt_y]])

    mock_model.get_image_features.return_value = img_feat
    mock_model.get_text_features.return_value = txt_feat
    return mock_model, mock_processor


def test_clip_consistency_high_similarity():
    """High semantic similarity (e.g. 0.32) produces a low suspicion score (genuine pair)."""
    mock_model, mock_processor = _build_mock_clip(cosine_target=0.32)

    with patch("backend.modules.clip_consistency.get_clip_components", return_value=(mock_model, mock_processor)):
        jpeg_bytes = make_jpeg_bytes()
        result = analyze_clip_consistency(
            jpeg_bytes,
            caption="A sunny mountain landscape with snow peaks.",
        )

    assert isinstance(result, SignalResult)
    assert result.name == "clip_consistency"
    assert result.score <= 0.15, f"Expected low suspicion for matching caption, got {result.score}"
    assert result.confidence >= 0.80
    assert "aligns strongly" in result.explanation.lower()
    assert result.details["cosine_similarity"] >= 0.30


def test_clip_consistency_low_similarity_cheapfake():
    """Low semantic similarity (e.g. 0.10) produces a high suspicion score (mismatched cheapfake)."""
    mock_model, mock_processor = _build_mock_clip(cosine_target=0.10)

    with patch("backend.modules.clip_consistency.get_clip_components", return_value=(mock_model, mock_processor)):
        jpeg_bytes = make_jpeg_bytes()
        result = analyze_clip_consistency(
            jpeg_bytes,
            caption="Protesters clash with riot police outside parliament building.",
        )

    assert result.name == "clip_consistency"
    assert result.score >= 0.75, f"Expected high suspicion for mismatched caption, got {result.score}"
    assert result.confidence >= 0.80
    assert "divergence" in result.explanation.lower() or "cheapfake" in result.explanation.lower()
    assert result.details["cosine_similarity"] <= 0.12


def test_clip_consistency_empty_caption_handling():
    """Empty caption yields neutral score with low confidence without calling model."""
    jpeg_bytes = make_jpeg_bytes()
    result = analyze_clip_consistency(jpeg_bytes, caption="   ")

    assert result.name == "clip_consistency"
    assert result.score == 0.50
    assert result.confidence == 0.30
    assert "empty caption" in result.explanation.lower()


def test_clip_consistency_model_load_failure_returns_confidence_zero():
    """If model fails to load, return confidence 0 instead of crashing."""
    with patch(
        "backend.modules.clip_consistency.get_clip_components",
        side_effect=RuntimeError("CLIP weights repository unavailable offline"),
    ):
        jpeg_bytes = make_jpeg_bytes()
        result = analyze_clip_consistency(jpeg_bytes, caption="Any caption")

    assert isinstance(result, SignalResult)
    assert result.name == "clip_consistency"
    assert result.confidence == 0.0
    assert "unavailable" in result.explanation.lower() or "failed" in result.explanation.lower()
    assert "error" in result.details
    assert "unavailable offline" in result.details["error"]


def test_clip_consistency_inference_failure_returns_confidence_zero():
    """If inference raises an exception, return confidence 0 instead of crashing."""
    mock_model = MagicMock()
    mock_processor = MagicMock()
    mock_model.get_image_features.side_effect = RuntimeError("Token index out of range")

    with patch("backend.modules.clip_consistency.get_clip_components", return_value=(mock_model, mock_processor)):
        jpeg_bytes = make_jpeg_bytes()
        result = analyze_clip_consistency(jpeg_bytes, caption="Sample caption")

    assert result.name == "clip_consistency"
    assert result.confidence == 0.0
    assert "inference failed" in result.explanation.lower()


def test_clip_consistency_contract_types_and_bounds():
    """Verify strict adherence to {name, score, confidence, explanation} contract."""
    mock_model, mock_processor = _build_mock_clip(cosine_target=0.22)

    with patch("backend.modules.clip_consistency.get_clip_components", return_value=(mock_model, mock_processor)):
        jpeg_bytes = make_jpeg_bytes()
        result = analyze_clip_consistency(jpeg_bytes, caption="Sample caption")

    res_dict = result.model_dump()
    assert res_dict["name"] == "clip_consistency"
    assert isinstance(res_dict["score"], float)
    assert isinstance(res_dict["confidence"], float)
    assert isinstance(res_dict["explanation"], str)
    assert 0.0 <= res_dict["score"] <= 1.0
    assert 0.0 <= res_dict["confidence"] <= 1.0
