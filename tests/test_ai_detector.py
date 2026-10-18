from unittest.mock import MagicMock, patch
import pytest
import torch

from tests.conftest import make_jpeg_bytes
from backend.modules.ai_detector import (
    analyze_ai_generation,
    clear_ai_detector_cache,
    get_ai_detector_components,
)
from backend.schemas import SignalResult


@pytest.fixture(autouse=True)
def reset_cache():
    """Ensure clean cache before and after each test."""
    clear_ai_detector_cache()
    yield
    clear_ai_detector_cache()


def test_ai_detector_synthetic_prediction():
    """Verify AI detector returns high suspicion score when model detects synthetic artifacts."""
    mock_model = MagicMock()
    mock_processor = MagicMock()

    # Configure logits: [artificial=4.0, human=-2.0] -> synthetic prob > 0.99
    mock_logits = torch.tensor([[4.0, -2.0]])
    mock_model.return_value = MagicMock(logits=mock_logits)
    mock_model.config.id2label = {0: "artificial", 1: "human"}

    with patch("backend.modules.ai_detector.get_ai_detector_components", return_value=(mock_model, mock_processor)):
        jpeg_bytes = make_jpeg_bytes()
        result = analyze_ai_generation(jpeg_bytes, filename="synthetic.jpg")

    assert isinstance(result, SignalResult)
    assert result.name == "ai_detector"
    assert result.score >= 0.85, f"Expected high synthetic suspicion score, got {result.score}"
    assert result.confidence >= 0.80
    assert "AI-generated" in result.explanation or "synthetic" in result.explanation.lower()
    assert result.details["synthetic_prob"] is not None
    assert result.details["synthetic_prob"] >= 0.85


def test_ai_detector_authentic_prediction():
    """Verify AI detector returns low suspicion score when model detects authentic photographic patterns."""
    mock_model = MagicMock()
    mock_processor = MagicMock()

    # Configure logits: [artificial=-3.0, human=3.0] -> synthetic prob < 0.01
    mock_logits = torch.tensor([[-3.0, 3.0]])
    mock_model.return_value = MagicMock(logits=mock_logits)
    mock_model.config.id2label = {0: "artificial", 1: "human"}

    with patch("backend.modules.ai_detector.get_ai_detector_components", return_value=(mock_model, mock_processor)):
        jpeg_bytes = make_jpeg_bytes()
        result = analyze_ai_generation(jpeg_bytes, filename="camera_real.jpg")

    assert result.name == "ai_detector"
    assert result.score <= 0.15, f"Expected low synthetic score for authentic image, got {result.score}"
    assert result.confidence >= 0.80
    assert "authentic" in result.explanation.lower()
    assert result.details["synthetic_prob"] <= 0.15
    assert result.details["authentic_prob"] >= 0.85


def test_ai_detector_model_load_failure_returns_confidence_zero():
    """If model fails to load, return a signal with confidence 0 instead of crashing."""
    with patch(
        "backend.modules.ai_detector.get_ai_detector_components",
        side_effect=RuntimeError("Model weights corrupted or download connection failed"),
    ):
        jpeg_bytes = make_jpeg_bytes()
        result = analyze_ai_generation(jpeg_bytes, filename="test.jpg")

    assert isinstance(result, SignalResult)
    assert result.name == "ai_detector"
    assert result.confidence == 0.0
    assert "unavailable" in result.explanation.lower() or "failed" in result.explanation.lower()
    assert "error" in result.details
    assert "Model weights corrupted" in result.details["error"]


def test_ai_detector_inference_failure_returns_confidence_zero():
    """If inference throws an unexpected error, return confidence 0 instead of crashing."""
    mock_model = MagicMock(side_effect=RuntimeError("CUDA/CPU tensor allocation error"))
    mock_processor = MagicMock()

    with patch("backend.modules.ai_detector.get_ai_detector_components", return_value=(mock_model, mock_processor)):
        jpeg_bytes = make_jpeg_bytes()
        result = analyze_ai_generation(jpeg_bytes, filename="test.jpg")

    assert result.name == "ai_detector"
    assert result.confidence == 0.0
    assert "inference failed" in result.explanation.lower()


def test_ai_detector_contract_types_and_bounds():
    """Verify strict adherence to {name, score, confidence, explanation} contract."""
    mock_model = MagicMock()
    mock_processor = MagicMock()
    mock_logits = torch.tensor([[0.5, 0.5]])
    mock_model.return_value = MagicMock(logits=mock_logits)
    mock_model.config.id2label = {0: "artificial", 1: "human"}

    with patch("backend.modules.ai_detector.get_ai_detector_components", return_value=(mock_model, mock_processor)):
        jpeg_bytes = make_jpeg_bytes()
        result = analyze_ai_generation(jpeg_bytes)

    res_dict = result.model_dump()
    assert res_dict["name"] == "ai_detector"
    assert isinstance(res_dict["score"], float)
    assert isinstance(res_dict["confidence"], float)
    assert isinstance(res_dict["explanation"], str)
    assert 0.0 <= res_dict["score"] <= 1.0
    assert 0.0 <= res_dict["confidence"] <= 1.0


def test_ai_detector_reads_model_from_env(monkeypatch):
    """Verify model ID is loaded from AI_DETECTOR_MODEL environment variable."""
    custom_model_id = "Organism/custom-synthetic-detector"
    monkeypatch.setenv("AI_DETECTOR_MODEL", custom_model_id)

    mock_model = MagicMock()
    mock_processor = MagicMock()
    mock_logits = torch.tensor([[2.0, -1.0]])
    mock_model.return_value = MagicMock(logits=mock_logits)
    mock_model.config.id2label = {0: "artificial", 1: "human"}

    with patch("backend.modules.ai_detector.AutoImageProcessor.from_pretrained", return_value=mock_processor) as mock_proc_loader, \
         patch("backend.modules.ai_detector.AutoModelForImageClassification.from_pretrained", return_value=mock_model) as mock_mod_loader:
        jpeg_bytes = make_jpeg_bytes()
        result = analyze_ai_generation(jpeg_bytes)

        assert result.details["model_id"] == custom_model_id
        mock_proc_loader.assert_called_with(custom_model_id)
        mock_mod_loader.assert_called_with(custom_model_id)


def test_ai_detector_default_model_when_env_not_set(monkeypatch):
    """Verify default model umm-maybe/AI-image-detector is used when AI_DETECTOR_MODEL is not set."""
    monkeypatch.delenv("AI_DETECTOR_MODEL", raising=False)

    mock_model = MagicMock()
    mock_processor = MagicMock()
    mock_logits = torch.tensor([[2.0, -1.0]])
    mock_model.return_value = MagicMock(logits=mock_logits)
    mock_model.config.id2label = {0: "artificial", 1: "human"}

    with patch("backend.modules.ai_detector.AutoImageProcessor.from_pretrained", return_value=mock_processor) as mock_proc_loader, \
         patch("backend.modules.ai_detector.AutoModelForImageClassification.from_pretrained", return_value=mock_model) as mock_mod_loader:
        jpeg_bytes = make_jpeg_bytes()
        result = analyze_ai_generation(jpeg_bytes)

        assert result.details["model_id"] == "umm-maybe/AI-image-detector"
        mock_proc_loader.assert_called_with("umm-maybe/AI-image-detector")
        mock_mod_loader.assert_called_with("umm-maybe/AI-image-detector")

