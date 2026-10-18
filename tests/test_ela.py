from pathlib import Path
from PIL import Image
import pytest

from tests.conftest import (
    make_jpeg_bytes,
    make_png_bytes,
    make_spliced_jpeg_bytes,
    make_webp_bytes,
)
from backend.modules.ela import analyze_ela
from backend.schemas import SignalResult


def test_ela_uniform_image(tmp_path):
    """Uniform image should show low error discrepancy and low suspicion."""
    jpeg_bytes = make_jpeg_bytes(width=150, height=150, color=(120, 120, 120))
    result = analyze_ela(jpeg_bytes, filename="uniform.jpg", output_dir=tmp_path)

    assert isinstance(result, SignalResult)
    assert result.name == "ela"
    assert 0.0 <= result.score <= 1.0
    assert 0.0 <= result.confidence <= 1.0
    assert result.score <= 0.35, f"Expected low suspicion for uniform image, got {result.score}"
    assert result.details["high_discrepancy_clusters"] == 0
    assert "uniform" in result.explanation.lower()


def test_ela_spliced_tampered_image(tmp_path):
    """Spliced image with uncompressed patch should show localized discrepancy clusters."""
    spliced_bytes = make_spliced_jpeg_bytes(width=200, height=200)
    result = analyze_ela(spliced_bytes, filename="spliced.jpg", output_dir=tmp_path)

    assert result.name == "ela"
    assert result.details["high_discrepancy_clusters"] > 0
    assert result.score >= 0.50, f"Expected elevated suspicion for spliced image, got {result.score}"
    assert "discrepancies" in result.explanation.lower() or "clusters" in result.explanation.lower()


def test_ela_heatmap_file_generation(tmp_path):
    """Verify ELA module generates a readable PNG heatmap on disk."""
    jpeg_bytes = make_jpeg_bytes(width=160, height=120)
    result = analyze_ela(jpeg_bytes, filename="test_heatmap.jpg", output_dir=tmp_path)

    heatmap_path_str = result.details["heatmap_path"]
    heatmap_file = Path(heatmap_path_str)

    assert heatmap_file.exists(), f"Heatmap file was not created: {heatmap_path_str}"
    assert heatmap_file.is_file()
    assert heatmap_file.suffix.lower() == ".png"

    # Verify heatmap is valid readable image matching original dimensions
    with Image.open(heatmap_file) as h_img:
        assert h_img.format == "PNG"
        assert h_img.size == (160, 120)


def test_ela_contract_types_and_bounds(tmp_path):
    """Verify strict adherence to {name, score, confidence, explanation} contract."""
    jpeg_bytes = make_jpeg_bytes()
    result = analyze_ela(jpeg_bytes, output_dir=tmp_path)

    res_dict = result.model_dump()
    assert "name" in res_dict
    assert "score" in res_dict
    assert "confidence" in res_dict
    assert "explanation" in res_dict

    assert res_dict["name"] == "ela"
    assert isinstance(res_dict["score"], float)
    assert isinstance(res_dict["confidence"], float)
    assert isinstance(res_dict["explanation"], str)

    assert 0.0 <= res_dict["score"] <= 1.0
    assert 0.0 <= res_dict["confidence"] <= 1.0
    assert len(res_dict["explanation"]) > 0


def test_ela_spliced_score_higher_than_uniform(tmp_path):
    """Spliced image must produce a higher suspicion score than a clean uniform image."""
    uniform_bytes = make_jpeg_bytes(width=200, height=200, color=(120, 120, 120))
    spliced_bytes = make_spliced_jpeg_bytes(width=200, height=200)

    res_uniform = analyze_ela(uniform_bytes, filename="uniform.jpg", output_dir=tmp_path)
    res_spliced = analyze_ela(spliced_bytes, filename="spliced.jpg", output_dir=tmp_path)

    assert res_spliced.score > res_uniform.score
    assert res_spliced.details["high_discrepancy_clusters"] > res_uniform.details["high_discrepancy_clusters"]


def test_ela_png_lossless_low_confidence(tmp_path):
    """PNG is lossless, so ELA should lower confidence to ~0.15 and warn in explanation."""
    png_bytes = make_png_bytes(width=150, height=150)
    result = analyze_ela(png_bytes, filename="graphic.png", output_dir=tmp_path)

    assert result.name == "ela"
    assert result.confidence == 0.15
    assert "unreliable for lossless images" in result.explanation.lower()
    assert result.details["is_jpeg"] is False
    assert result.details["image_format"] == "PNG"


def test_ela_webp_lossless_low_confidence(tmp_path):
    """WEBP non-JPEG image should lower confidence to ~0.15 and warn in explanation."""
    webp_bytes = make_webp_bytes(width=150, height=150)
    result = analyze_ela(webp_bytes, filename="photo.webp", output_dir=tmp_path)

    assert result.name == "ela"
    assert result.confidence == 0.15
    assert "unreliable for lossless images" in result.explanation.lower()
    assert result.details["is_jpeg"] is False


def test_ela_jpeg_behavior_unchanged(tmp_path):
    """JPEG image retains standard high confidence and does not warn about lossless format."""
    jpeg_bytes = make_jpeg_bytes(width=150, height=150)
    result = analyze_ela(jpeg_bytes, filename="photo.jpg", output_dir=tmp_path)

    assert result.name == "ela"
    assert result.confidence >= 0.70
    assert "unreliable for lossless images" not in result.explanation.lower()
    assert result.details["is_jpeg"] is True
    assert result.details["image_format"] == "JPEG"

