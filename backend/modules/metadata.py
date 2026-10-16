import io
import json
import logging
import os
import tempfile
from typing import Any, Dict, List, Optional
from PIL import Image, ExifTags

from backend.schemas import SignalResult

logger = logging.getLogger(__name__)

# Try importing c2pa gracefully
try:
    import c2pa
    HAS_C2PA = True
except ImportError:
    c2pa = None
    HAS_C2PA = False

# Known editing and generative software signatures (lowercase)
KNOWN_EDITING_SOFTWARE = [
    "photoshop",
    "gimp",
    "lightroom",
    "canva",
    "snapseed",
    "picsart",
    "pixlr",
    "vsco",
    "affinity",
    "paint.net",
    "coreldraw",
    "capture one",
    "darktable",
    "rawtherapee",
    "fotor",
    "facetune",
    "meitu",
    "procreate",
    "midjourney",
    "stable diffusion",
    "dall-e",
    "firefly",
    "generative",
]

# Standard camera capture tags indicating physical sensor / optics metadata
CAPTURE_TAG_NAMES = {
    "ExposureTime",
    "FNumber",
    "ISOSpeedRatings",
    "PhotographicSensitivity",
    "DateTimeOriginal",
    "DateTimeDigitized",
    "FocalLength",
    "Flash",
    "WhiteBalance",
    "MeteringMode",
    "ExposureProgram",
    "LensModel",
    "LensMake",
    "ShutterSpeedValue",
    "ApertureValue",
    "BrightnessValue",
    "ExposureBiasValue",
    "MaxApertureValue",
    "FocalLengthIn35mmFilm",
    "SceneCaptureType",
}

# Software substrings indicating stock/phone camera capture apps
CAMERA_APP_SUBSTRINGS = [
    "camera",
    "gcam",
    "snapdragon",
    "mediatek camera",
    "qualcomm camera",
    "android camera",
    "samsung camera",
]


def clean_exif_val(val: Any) -> Any:
    """Clean EXIF values by stripping null bytes and surrounding whitespace."""
    if isinstance(val, bytes):
        try:
            val = val.decode("utf-8", errors="ignore")
        except Exception:
            try:
                val = val.decode("latin-1", errors="ignore")
            except Exception:
                return f"<binary {len(val)} bytes>"
    if isinstance(val, str):
        val = val.replace("\x00", "").replace("\u0000", "").strip()
    return val


def extract_c2pa_manifest(image_bytes: bytes, file_ext: str = ".jpg") -> tuple[bool, Optional[Dict[str, Any]], List[str]]:
    """Extract C2PA manifest from image bytes if c2pa library is present.
    
    Fails gracefully if the library is missing or if no manifest is found.
    """
    if not HAS_C2PA or c2pa is None:
        return False, None, ["c2pa-python library not available in environment"]

    tmp_path = None
    try:
        suffix = file_ext if file_ext.startswith(".") else f".{file_ext}"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_file:
            tmp_file.write(image_bytes)
            tmp_path = tmp_file.name

        reader = c2pa.Reader(tmp_path)
        manifest_raw = reader.json()
        manifest_data = json.loads(manifest_raw) if isinstance(manifest_raw, str) else manifest_raw
        return True, manifest_data, []
    except Exception as exc:
        logger.debug(f"C2PA manifest check returned: {exc}")
        return False, None, []
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def analyze_metadata(image_bytes: bytes, filename: str = "") -> SignalResult:
    """Analyze image EXIF and C2PA metadata.
    
    Returns a SignalResult with suspicion score (0-1), confidence (0-1),
    and detailed breakdown.
    """
    warnings: List[str] = []
    exif_details: Dict[str, Any] = {}
    camera_make: Optional[str] = None
    camera_model: Optional[str] = None
    software_tag: Optional[str] = None
    has_exif = False

    file_ext = os.path.splitext(filename)[1].lower() if filename else ".jpg"
    if not file_ext:
        file_ext = ".jpg"

    # 1. Parse EXIF with Pillow (including sub-IFDs)
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            raw_exif = img.getexif()
            if raw_exif and len(raw_exif) > 0:
                has_exif = True
                # Read root tags
                for tag_id, val in raw_exif.items():
                    tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                    cleaned = clean_exif_val(val)
                    exif_details[tag_name] = cleaned

                # Also inspect sub-IFD (e.g. Exif IFD containing ExposureTime, FNumber, etc.)
                try:
                    for ifd_id in (ExifTags.IFD.Exif, ExifTags.IFD.GPSInfo):
                        ifd = raw_exif.get_ifd(ifd_id)
                        for tag_id, val in ifd.items():
                            tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                            if tag_name not in exif_details:
                                exif_details[tag_name] = clean_exif_val(val)
                except Exception:
                    pass

                # Extract and clean primary identification tags
                if "Make" in exif_details and exif_details["Make"]:
                    camera_make = str(exif_details["Make"])
                if "Model" in exif_details and exif_details["Model"]:
                    camera_model = str(exif_details["Model"])
                if "Software" in exif_details and exif_details["Software"]:
                    software_tag = str(exif_details["Software"])
    except Exception as exc:
        warnings.append(f"Could not read image EXIF: {str(exc)}")

    # 2. Check for editing software signatures
    editing_detected = False
    detected_editing_software: List[str] = []
    if software_tag:
        software_lower = software_tag.lower()
        for kw in KNOWN_EDITING_SOFTWARE:
            if kw in software_lower:
                editing_detected = True
                detected_editing_software.append(software_tag)
                break

    for tag_name, val in exif_details.items():
        if tag_name.lower() in ("software", "processingsoftware", "history", "imagehistory"):
            val_str = str(val)
            val_lower = val_str.lower()
            for kw in KNOWN_EDITING_SOFTWARE:
                if kw in val_lower and val_str not in detected_editing_software:
                    editing_detected = True
                    detected_editing_software.append(val_str)

    if editing_detected:
        for sw in detected_editing_software:
            warnings.append(f"Editing software signature detected: {sw}")

    # 3. Detect camera capture tags & camera app names
    capture_tags_found = sorted([tag for tag in exif_details if tag in CAPTURE_TAG_NAMES and exif_details[tag] != ""])
    capture_tag_count = len(capture_tags_found)

    software_is_camera_app = False
    if software_tag and not editing_detected:
        sw_lower = software_tag.lower()
        for kw in CAMERA_APP_SUBSTRINGS:
            if kw in sw_lower:
                software_is_camera_app = True
                break

    has_make_or_model = bool((camera_make and camera_make.strip()) or (camera_model and camera_model.strip()))
    is_rich_capture = capture_tag_count >= 5
    is_likely_camera_captured = bool(
        is_rich_capture
        or has_make_or_model
        or software_is_camera_app
        or capture_tag_count >= 3
    )

    if not is_likely_camera_captured:
        warnings.append("Missing camera make/model and capture metadata (EXIF stripped or synthetic media)")
    elif not has_make_or_model:
        warnings.append("Camera make/model absent, but capture tags or camera software confirm camera-capture origin")

    # 4. Extract C2PA manifest (fails gracefully)
    c2pa_detected, c2pa_manifest, c2pa_warnings = extract_c2pa_manifest(image_bytes, file_ext=file_ext)
    warnings.extend(c2pa_warnings)

    c2pa_synthetic_detected = False
    if c2pa_detected and c2pa_manifest:
        manifest_str = json.dumps(c2pa_manifest).lower()
        if "trainedalgorithmicmedia" in manifest_str or "compositesynthetic" in manifest_str:
            c2pa_synthetic_detected = True
            warnings.append("C2PA manifest explicitly declares AI/synthetic generation source")

    # 5. Score calculation (0.0 = authentic, 1.0 = suspicious/tampered) & Honest Confidence
    score: float
    confidence: float
    explanation_parts: List[str] = []

    if c2pa_synthetic_detected:
        score = 0.95
        confidence = 0.95
        explanation_parts.append("C2PA manifest explicitly indicates algorithmic/synthetic image origin.")
    elif c2pa_detected:
        if editing_detected:
            score = 0.65
            confidence = 0.90
            explanation_parts.append("C2PA manifest present but digital editing recorded in provenance history.")
        else:
            score = 0.05
            confidence = 0.95
            explanation_parts.append("Valid C2PA provenance manifest verified with authentic hardware capture lineage.")
    elif editing_detected:
        score = 0.80
        confidence = 0.88
        explanation_parts.append(f"Digital editing software traces detected ({', '.join(detected_editing_software)}).")
    elif is_rich_capture:
        # 5 or more capture tags with no editing software -> low suspicion around 0.05-0.15
        if has_make_or_model:
            score = 0.08
            confidence = 0.90
            cam_str = f"{camera_make or ''} {camera_model or ''}".strip()
            explanation_parts.append(
                f"Authentic camera origin verified with hardware tags ({cam_str}) "
                f"and {capture_tag_count} capture tags: {', '.join(capture_tags_found)}."
            )
        else:
            score = 0.10
            confidence = 0.85
            explanation_parts.append(
                f"Likely camera-captured image: rich capture metadata present without Make/Model "
                f"({capture_tag_count} capture tags: {', '.join(capture_tags_found)})."
            )
    elif software_is_camera_app or capture_tag_count >= 2:
        score = 0.15
        confidence = 0.75
        details_str = []
        if software_is_camera_app:
            details_str.append(f"camera app '{software_tag}'")
        if capture_tags_found:
            details_str.append(f"{capture_tag_count} capture tags: {', '.join(capture_tags_found)}")
        explanation_parts.append(
            f"Likely camera-captured image supported by {' and '.join(details_str)} with no editing traces."
        )
    elif has_make_or_model:
        score = 0.22
        confidence = 0.65
        cam_str = f"{camera_make or ''} {camera_model or ''}".strip()
        explanation_parts.append(
            f"Camera hardware tag present ({cam_str}) with limited capture settings metadata."
        )
    elif has_exif:
        score = 0.38
        confidence = 0.50
        explanation_parts.append("Partial EXIF present without identifiable camera capture metadata or hardware tags.")
    else:
        # Fully missing EXIF stays medium around 0.45 with lower confidence
        score = 0.45
        confidence = 0.55
        explanation_parts.append("No EXIF metadata found; stripped metadata is common in social media re-encodings and AI generation.")

    if not c2pa_detected and HAS_C2PA:
        explanation_parts.append("No C2PA provenance credentials attached.")

    explanation = " ".join(explanation_parts)

    return SignalResult(
        name="metadata",
        score=round(max(0.0, min(1.0, score)), 3),
        confidence=round(max(0.0, min(1.0, confidence)), 3),
        explanation=explanation,
        details={
            "has_exif": has_exif,
            "has_camera_info": is_likely_camera_captured,
            "is_likely_camera_captured": is_likely_camera_captured,
            "is_rich_capture": is_rich_capture,
            "camera_make": camera_make,
            "camera_model": camera_model,
            "software": software_tag,
            "software_is_camera_app": software_is_camera_app,
            "editing_detected": editing_detected,
            "detected_editing_software": detected_editing_software,
            "capture_tags_found": capture_tags_found,
            "capture_tag_count": capture_tag_count,
            "c2pa_manifest_detected": c2pa_detected,
            "c2pa_synthetic_detected": c2pa_synthetic_detected,
            "warnings": warnings,
            "exif_tag_count": len(exif_details),
        },
    )
