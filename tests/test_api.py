import io
from unittest.mock import patch
import pytest

from tests.conftest import make_jpeg_bytes, make_png_bytes, make_webp_bytes
from backend.schemas import SignalResult


@pytest.fixture(autouse=True)
def mock_phase2_models():
    """Mock Phase 2 AI and CLIP models in API tests so they run fast and offline."""
    mock_ai_signal = SignalResult(
        name="ai_detector",
        score=0.10,
        confidence=0.88,
        explanation="Mocked AI detector signal.",
        details={"synthetic_prob": 0.10, "authentic_prob": 0.90},
    )
    mock_clip_signal = SignalResult(
        name="clip_consistency",
        score=0.15,
        confidence=0.85,
        explanation="Mocked CLIP consistency signal.",
        details={"cosine_similarity": 0.29},
    )

    with patch("backend.main.analyze_ai_generation", return_value=mock_ai_signal), \
         patch("backend.main.analyze_clip_consistency", return_value=mock_clip_signal):
        yield


def test_health_endpoint(client):
    """GET /health returns healthy status and Phase 1 & 2 modules."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["version"] == "1.0.0"
    for expected_mod in ["metadata", "ela", "ai_detector", "clip_consistency"]:
        assert expected_mod in data["modules_loaded"]


def test_api_health_endpoint_alias(client):
    """GET /api/health works as well."""
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_analyze_with_jpeg(client):
    """POST /analyze with JPEG and caption returns all 4 signals."""
    jpeg_bytes = make_jpeg_bytes()
    response = client.post(
        "/analyze",
        data={"caption": "Protest in downtown Seattle on Sunday morning."},
        files={"image": ("protest.jpg", jpeg_bytes, "image/jpeg")},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "protest.jpg"
    assert data["caption"] == "Protest in downtown Seattle on Sunday morning."
    assert "signals" in data
    assert isinstance(data["signals"], list)
    assert len(data["signals"]) == 4

    # Check signal modules
    signal_names = [s["name"] for s in data["signals"]]
    for expected_name in ["metadata", "ela", "ai_detector", "clip_consistency"]:
        assert expected_name in signal_names

    # Check contract fields for each signal
    for s in data["signals"]:
        assert "name" in s
        assert "score" in s
        assert "confidence" in s
        assert "explanation" in s
        assert 0.0 <= s["score"] <= 1.0
        assert 0.0 <= s["confidence"] <= 1.0


def test_analyze_with_png(client):
    """POST /analyze with PNG is accepted."""
    png_bytes = make_png_bytes()
    response = client.post(
        "/analyze",
        data={"caption": "Screenshot of press release."},
        files={"image": ("document.png", png_bytes, "image/png")},
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["signals"]) == 4


def test_analyze_with_webp(client):
    """POST /analyze with WEBP is accepted."""
    webp_bytes = make_webp_bytes()
    response = client.post(
        "/analyze",
        data={"caption": "Modern web article illustration."},
        files={"image": ("article.webp", webp_bytes, "image/webp")},
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["signals"]) == 4


def test_analyze_disallowed_extension(client):
    """POST /analyze rejects disallowed extensions like .gif."""
    gif_bytes = b"GIF89a\x01\x00\x01\x00\x80\x00\x00\x00\x00\x00\xff\xff\xff!\xf9\x04\x01\x00\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;"
    response = client.post(
        "/analyze",
        data={"caption": "Animated image claim."},
        files={"image": ("animation.gif", gif_bytes, "image/gif")},
    )
    assert response.status_code == 400
    assert "unsupported" in response.json()["detail"].lower()


def test_analyze_disallowed_mime_type(client):
    """POST /analyze rejects disallowed mime types."""
    response = client.post(
        "/analyze",
        data={"caption": "Document claim."},
        files={"image": ("doc.pdf", b"%PDF-1.4 ...", "application/pdf")},
    )
    assert response.status_code == 400
    assert "unsupported" in response.json()["detail"].lower()


def test_analyze_corrupted_image(client):
    """POST /analyze rejects corrupted image bytes."""
    corrupted_bytes = b"NotARealImage" * 20
    response = client.post(
        "/analyze",
        data={"caption": "Corrupted file."},
        files={"image": ("bad.jpg", corrupted_bytes, "image/jpeg")},
    )
    assert response.status_code == 400
    assert "corrupted" in response.json()["detail"].lower() or "invalid" in response.json()["detail"].lower()


def test_analyze_empty_file(client):
    """POST /analyze rejects 0-byte file."""
    response = client.post(
        "/analyze",
        data={"caption": "Empty file."},
        files={"image": ("empty.jpg", b"", "image/jpeg")},
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_analyze_exceeds_max_size(client):
    """POST /analyze rejects uploads exceeding 10 MB."""
    oversized_bytes = b"\xff\xd8\xff\xe0" + b"\x00" * (10 * 1024 * 1024 + 500 * 1024)
    response = client.post(
        "/analyze",
        data={"caption": "Large file upload."},
        files={"image": ("huge.jpg", oversized_bytes, "image/jpeg")},
    )
    assert response.status_code in (413, 400)
    assert "exceeds" in response.json()["detail"].lower()


def test_analyze_missing_caption(client):
    """POST /analyze without caption returns 422 Unprocessable Entity."""
    jpeg_bytes = make_jpeg_bytes()
    response = client.post(
        "/analyze",
        files={"image": ("photo.jpg", jpeg_bytes, "image/jpeg")},
    )
    assert response.status_code == 422


def test_analyze_missing_image(client):
    """POST /analyze without image returns 422 Unprocessable Entity."""
    response = client.post(
        "/analyze",
        data={"caption": "A claim without any image."},
    )
    assert response.status_code == 422
