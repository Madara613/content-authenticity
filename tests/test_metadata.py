import pytest
from tests.conftest import make_jpeg_bytes
from backend.modules.metadata import analyze_metadata
from backend.schemas import SignalResult


def test_metadata_camera_info_intact():
    """Verify metadata parser recognizes authentic camera EXIF."""
    # 271 = Make, 272 = Model
    jpeg_bytes = make_jpeg_bytes(
        exif_dict={
            271: "Sony",
            272: "ILCE-7RM5",
        }
    )

    result = analyze_metadata(jpeg_bytes, filename="camera_photo.jpg")

    assert isinstance(result, SignalResult)
    assert result.name == "metadata"
    assert 0.0 <= result.score <= 1.0
    assert 0.0 <= result.confidence <= 1.0
    assert result.score <= 0.30, f"Expected low suspicion for authentic camera, got {result.score}"
    assert result.details["has_camera_info"] is True
    assert result.details["camera_make"] == "Sony"
    assert result.details["camera_model"] == "ILCE-7RM5"
    assert result.details["editing_detected"] is False
    assert "Sony" in result.explanation


def test_metadata_rich_exif_without_make_model():
    """Verify rich capture EXIF without Make/Model is treated as likely camera-captured.
    
    5 or more capture tags with no editing software must give a low suspicion score
    around 0.05-0.15, and list the capture tags in explanation and details.
    """
    # 33434: ExposureTime, 33437: FNumber, 34855: ISOSpeedRatings,
    # 36867: DateTimeOriginal, 37386: FocalLength, 37385: Flash
    jpeg_bytes = make_jpeg_bytes(
        exif_dict={
            33434: (1, 250),
            33437: (28, 10),
            34855: 400,
            36867: "2026:08:14 15:30:00",
            37386: (50, 1),
            37385: 0,
        }
    )

    result = analyze_metadata(jpeg_bytes, filename="phone_no_make_model.jpg")

    assert result.details["camera_make"] is None
    assert result.details["camera_model"] is None
    assert result.details["is_likely_camera_captured"] is True
    assert result.details["has_camera_info"] is True
    assert result.details["is_rich_capture"] is True
    assert result.details["capture_tag_count"] >= 5

    # Check scoring: 0.05 - 0.15 for rich capture EXIF without editing
    assert 0.05 <= result.score <= 0.15, f"Expected low score (0.05-0.15), got {result.score}"
    assert result.confidence >= 0.80

    # Explanation and details should list the found capture tags
    tags_found = result.details["capture_tags_found"]
    for expected_tag in ["ExposureTime", "FNumber", "ISOSpeedRatings", "DateTimeOriginal", "FocalLength", "Flash"]:
        assert expected_tag in tags_found
        assert expected_tag in result.explanation


def test_metadata_no_exif():
    """Verify fully missing EXIF stays medium (around 0.45) with honest confidence."""
    jpeg_bytes = make_jpeg_bytes()

    result = analyze_metadata(jpeg_bytes, filename="stripped.jpg")

    assert result.details["has_exif"] is False
    assert result.details["has_camera_info"] is False
    assert result.details["is_likely_camera_captured"] is False
    assert result.details["capture_tag_count"] == 0
    assert 0.40 <= result.score <= 0.50, f"Expected medium score around 0.45, got {result.score}"
    assert result.confidence <= 0.60, f"Expected lower honest confidence, got {result.confidence}"
    assert any("Missing camera make/model and capture metadata" in w for w in result.details["warnings"])


def test_metadata_null_bytes_and_whitespace_stripped():
    """Verify null bytes (\x00, \u0000) and surrounding whitespace are stripped from EXIF strings."""
    # 271: Make, 272: Model, 305: Software
    jpeg_bytes = make_jpeg_bytes(
        exif_dict={
            305: "MediaTek Camera Application\x00\x00",
            271: "  Sony  \x00",
            272: "ILCE-7RM5\u0000",
        }
    )

    result = analyze_metadata(jpeg_bytes, filename="null_bytes.jpg")

    assert result.details["software"] == "MediaTek Camera Application"
    assert "\x00" not in result.details["software"]
    assert "\u0000" not in result.details["software"]
    assert result.details["camera_make"] == "Sony"
    assert result.details["camera_model"] == "ILCE-7RM5"
    assert result.details["software_is_camera_app"] is True
    assert result.details["is_likely_camera_captured"] is True
    assert result.score <= 0.15


def test_metadata_editing_software_flagged():
    """Verify editing software (e.g. Adobe Photoshop, Canva, GIMP) scores high."""
    # 271 = Make, 272 = Model, 305 = Software
    jpeg_bytes = make_jpeg_bytes(
        exif_dict={
            271: "Nikon",
            272: "Z8",
            305: "Adobe Photoshop 2024 (Windows)",
        }
    )

    result = analyze_metadata(jpeg_bytes, filename="edited_photo.jpg")

    assert result.name == "metadata"
    assert result.score >= 0.70, f"Expected high suspicion for edited photo, got {result.score}"
    assert result.details["editing_detected"] is True
    assert result.details["software"] == "Adobe Photoshop 2024 (Windows)"
    assert any("Adobe Photoshop" in w for w in result.details["warnings"])
    assert "editing software" in result.explanation.lower()


def test_metadata_canva_editing_detected():
    """Verify Canva editing software tag scores high."""
    jpeg_bytes = make_jpeg_bytes(exif_dict={305: "Canva Design Studio"})
    result = analyze_metadata(jpeg_bytes)
    assert result.details["editing_detected"] is True
    assert result.score >= 0.70


def test_metadata_gimp_editing_detected():
    """Verify GIMP editing software tag scores high."""
    jpeg_bytes = make_jpeg_bytes(exif_dict={305: "GIMP 2.10.34"})
    result = analyze_metadata(jpeg_bytes)
    assert result.details["editing_detected"] is True
    assert result.score >= 0.70


def test_metadata_c2pa_fails_gracefully():
    """Verify C2PA parser handles images without manifests gracefully."""
    jpeg_bytes = make_jpeg_bytes()

    result = analyze_metadata(jpeg_bytes, filename="no_c2pa.jpg")

    assert result.name == "metadata"
    assert result.details["c2pa_manifest_detected"] is False
    assert result.details["c2pa_synthetic_detected"] is False
    assert isinstance(result.details["warnings"], list)


def test_metadata_contract_types_and_bounds():
    """Verify strict adherence to {name, score, confidence, explanation} contract."""
    jpeg_bytes = make_jpeg_bytes(exif_dict={271: "Canon"})
    result = analyze_metadata(jpeg_bytes)

    res_dict = result.model_dump()
    assert "name" in res_dict
    assert "score" in res_dict
    assert "confidence" in res_dict
    assert "explanation" in res_dict

    assert isinstance(res_dict["name"], str)
    assert isinstance(res_dict["score"], float)
    assert isinstance(res_dict["confidence"], float)
    assert isinstance(res_dict["explanation"], str)

    assert 0.0 <= res_dict["score"] <= 1.0
    assert 0.0 <= res_dict["confidence"] <= 1.0
    assert len(res_dict["explanation"]) > 0


def test_metadata_corrupted_bytes_handling():
    """Verify metadata parser does not crash on arbitrary corrupted byte streams."""
    result = analyze_metadata(b"Not An Image", filename="corrupted.jpg")
    assert isinstance(result, SignalResult)
    assert result.name == "metadata"
    assert any("Could not read" in w for w in result.details["warnings"])
