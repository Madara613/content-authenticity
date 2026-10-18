import io
import os
import uuid
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
from PIL import Image

from backend.schemas import SignalResult


def analyze_ela(
    image_bytes: bytes,
    filename: str = "",
    output_dir: Optional[Path] = None,
    quality: int = 90,
) -> SignalResult:
    """Perform Error Level Analysis (ELA) using PIL and OpenCV.
    
    1. Resaves the image at a controlled JPEG quality (default 90).
    2. Calculates pixel discrepancy between original and resaved versions.
    3. Identifies anomalous localized compression discrepancies/clusters.
    4. Generates and saves a forensic colormap heatmap image.
    5. Returns a SignalResult with suspicion score (0-1) and explanation.
    """
    if output_dir is None:
        output_dir = Path("data/heatmaps")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Open image and convert to RGB
    with Image.open(io.BytesIO(image_bytes)) as orig_img:
        orig_format = (orig_img.format or "").upper()
        orig_rgb = orig_img.convert("RGB")
        width, height = orig_rgb.size

    # 2. Resave to in-memory JPEG at controlled quality
    resave_buffer = io.BytesIO()
    orig_rgb.save(resave_buffer, format="JPEG", quality=quality)
    resave_buffer.seek(0)
    with Image.open(resave_buffer) as resaved_img:
        resaved_rgb = resaved_img.convert("RGB")

    # 3. Calculate absolute pixel difference
    orig_arr = np.array(orig_rgb, dtype=np.int16)
    resaved_arr = np.array(resaved_rgb, dtype=np.int16)
    diff_rgb = np.abs(orig_arr - resaved_arr).astype(np.uint8)
    diff_gray = cv2.cvtColor(diff_rgb, cv2.COLOR_RGB2GRAY)

    # Calculate basic statistical metrics
    mean_error = float(np.mean(diff_gray))
    max_error = float(np.max(diff_gray))
    std_error = float(np.std(diff_gray))

    # 4. Detect localized high-discrepancy clusters (splicing / tampering)
    # Adaptive threshold sensitive to statistical discrepancies
    anom_threshold = max(6.0, mean_error + 2.0 * std_error)
    _, thresh_mask = cv2.threshold(
        diff_gray, int(min(250, anom_threshold)), 255, cv2.THRESH_BINARY
    )

    # Morphological opening to clean single-pixel noise while retaining clusters
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned_mask = cv2.morphologyEx(thresh_mask, cv2.MORPH_OPEN, kernel)

    # Find spatial clusters
    contours, _ = cv2.findContours(
        cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    total_pixels = width * height
    min_cluster_area = max(15, total_pixels // 4000)
    significant_clusters = [c for c in contours if cv2.contourArea(c) >= min_cluster_area]
    cluster_count = len(significant_clusters)

    # Calculate area ratio of high-discrepancy regions
    anom_pixel_count = int(np.count_nonzero(cleaned_mask))
    anom_area_ratio = float(anom_pixel_count / max(1, total_pixels))

    # 5. Generate and save heatmap
    # Scale difference for visual forensic inspection (amplification)
    amplified = np.clip(diff_gray.astype(np.float32) * 15.0, 0, 255).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(amplified, cv2.COLORMAP_INFERNO)

    # Draw subtle contour outlines around detected clusters on heatmap
    if significant_clusters:
        cv2.drawContours(heatmap_bgr, significant_clusters, -1, (0, 255, 255), 1)

    stem = Path(filename).stem if filename else "image"
    safe_stem = "".join(c for c in stem if c.isalnum() or c in ("-", "_"))[:30] or "img"
    heatmap_filename = f"ela_{safe_stem}_{uuid.uuid4().hex[:8]}.png"
    heatmap_path = output_dir / heatmap_filename
    cv2.imwrite(str(heatmap_path), heatmap_bgr)

    # 6. Compute suspicion score and confidence
    score: float
    confidence: float
    explanation: str

    if cluster_count > 0:
        # Detected localized discrepancy clusters
        base_score = 0.50 + min(0.30, cluster_count * 0.10)
        ratio_boost = min(0.15, anom_area_ratio * 4.0)
        score = min(0.95, base_score + ratio_boost)
        confidence = 0.82
        explanation = (
            f"Localized compression discrepancies detected: {cluster_count} anomalous cluster(s) "
            f"covering {anom_area_ratio * 100:.1f}% of image (mean error: {mean_error:.1f}, "
            f"max error: {max_error:.1f}). Indicates possible localized splicing or digital editing."
        )
    elif max_error > 50.0 and std_error > 10.0:
        score = 0.45
        confidence = 0.72
        explanation = (
            f"Elevated compression variance detected (std error: {std_error:.1f}, max error: {max_error:.1f}), "
            "without concentrated spatial clusters. Suggests moderate post-processing or re-encoding."
        )
    else:
        score = min(0.25, max(0.05, mean_error / 50.0))
        confidence = 0.85
        explanation = (
            f"Compression error levels are uniform across the image (mean error: {mean_error:.1f}, "
            f"max error: {max_error:.1f}). No localized discrepancy clusters detected."
        )

    # Small image penalty on confidence (for JPEG images)
    if width < 120 or height < 120:
        confidence = max(0.50, confidence - 0.20)
        explanation += " Note: Small image dimensions reduce ELA sensitivity."

    # Lossless / Non-JPEG handling: ELA is only meaningful for lossy formats.
    # For PNG, WEBP lossless or any image that was not JPEG-compressed,
    # lower the confidence to about 0.15 and state that ELA is unreliable for lossless images.
    is_jpeg = orig_format in ("JPEG", "JPG")
    if not is_jpeg:
        confidence = 0.15
        format_label = orig_format or "non-JPEG"
        explanation += (
            f" Note: Image format is {format_label}. ELA is only meaningful for lossy formats "
            f"and is unreliable for lossless images."
        )

    return SignalResult(
        name="ela",
        score=round(max(0.0, min(1.0, score)), 3),
        confidence=round(max(0.0, min(1.0, confidence)), 3),
        explanation=explanation,
        details={
            "mean_error": round(mean_error, 3),
            "max_error": round(max_error, 3),
            "std_error": round(std_error, 3),
            "high_discrepancy_clusters": cluster_count,
            "anomalous_area_ratio": round(anom_area_ratio, 4),
            "heatmap_path": str(heatmap_path).replace("\\", "/"),
            "heatmap_filename": heatmap_filename,
            "resave_quality": quality,
            "image_dimensions": [width, height],
            "image_format": orig_format,
            "is_jpeg": is_jpeg,
        },
    )
