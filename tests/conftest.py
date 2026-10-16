import io
import sys
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.main import app


@pytest.fixture
def client():
    """FastAPI TestClient fixture."""
    return TestClient(app)


def make_jpeg_bytes(
    width: int = 150,
    height: int = 150,
    color=(120, 140, 160),
    exif_dict: Optional[Dict[int, Any]] = None,
    quality: int = 95,
) -> bytes:
    """Generate JPEG bytes in-memory with optional EXIF tags."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()

    if exif_dict:
        exif = img.getexif()
        for tag_id, val in exif_dict.items():
            exif[tag_id] = val
        img.save(buf, format="JPEG", quality=quality, exif=exif)
    else:
        img.save(buf, format="JPEG", quality=quality)

    return buf.getvalue()


def make_png_bytes(width: int = 120, height: int = 120, color=(80, 120, 180)) -> bytes:
    """Generate PNG bytes in-memory."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def make_webp_bytes(width: int = 120, height: int = 120, color=(180, 120, 80)) -> bytes:
    """Generate WEBP bytes in-memory."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="WEBP")
    return buf.getvalue()


def make_spliced_jpeg_bytes(width: int = 200, height: int = 200) -> bytes:
    """Generate an image with an uncompressed spliced patch simulating tampering.
    
    The base image is saved at low JPEG quality, and then an uncompressed patch
    is spliced onto it, producing prominent localized ELA discrepancies.
    """
    # Base gradient image saved at lower quality
    base = Image.linear_gradient("L").resize((width, height)).convert("RGB")
    buf_low = io.BytesIO()
    base.save(buf_low, format="JPEG", quality=65)
    buf_low.seek(0)
    base = Image.open(buf_low).copy()

    # Create distinct high-contrast patch
    patch_size = 60
    patch = Image.new("RGB", (patch_size, patch_size), color=(255, 0, 0))

    # Paste patch into the center
    paste_x = (width - patch_size) // 2
    paste_y = (height - patch_size) // 2
    base.paste(patch, (paste_x, paste_y))

    buf_final = io.BytesIO()
    base.save(buf_final, format="JPEG", quality=95)
    return buf_final.getvalue()
