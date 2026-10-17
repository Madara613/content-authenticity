"""AI-Generated Image Detection Module (Phase 2).

Uses a pretrained Hugging Face Vision Transformer classifier to evaluate the probability
that an image was synthetically generated (diffusion, GANs) versus captured organically.

Model Selection:
    - Model: `umm-maybe/AI-image-detector` (fine-tuned google/vit-base-patch16-224-in21k)
    - License: Apache 2.0 (open source, permissive, clear commercial/academic license)
    - Rationale: High community adoption (>1M downloads on Hugging Face), specifically
      benchmarked across modern generative models (Midjourney, Stable Diffusion, DALL-E),
      and lightweight enough for reliable CPU inference.
"""

import io
import logging
from typing import Any, Dict, Optional, Tuple
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

from backend.config import settings
from backend.schemas import SignalResult

logger = logging.getLogger(__name__)

# Cached model instances for lazy loading
_ai_model: Optional[Any] = None
_ai_processor: Optional[Any] = None
_ai_cached_model_id: Optional[str] = None


def get_ai_detector_components(model_id: Optional[str] = None) -> Tuple[Any, Any]:
    """Lazily load and cache the Hugging Face AI detector model on CPU."""
    global _ai_model, _ai_processor, _ai_cached_model_id

    target_model_id = model_id or settings.ai_detector_model_id

    if _ai_model is not None and _ai_processor is not None and _ai_cached_model_id == target_model_id:
        return _ai_model, _ai_processor

    logger.info(f"Loading AI detector model '{target_model_id}' on CPU...")
    processor = AutoImageProcessor.from_pretrained(target_model_id)
    model = AutoModelForImageClassification.from_pretrained(target_model_id)
    model.to(torch.device("cpu"))
    model.eval()

    _ai_model = model
    _ai_processor = processor
    _ai_cached_model_id = target_model_id
    return _ai_model, _ai_processor


def clear_ai_detector_cache() -> None:
    """Clear cached model and processor (primarily for testing)."""
    global _ai_model, _ai_processor, _ai_cached_model_id
    _ai_model = None
    _ai_processor = None
    _ai_cached_model_id = None


def analyze_ai_generation(
    image_bytes: bytes,
    filename: str = "",
    model_id: Optional[str] = None,
) -> SignalResult:
    """Classify image as authentic vs synthetic/AI-generated.
    
    Returns standard SignalResult:
        - name: "ai_detector"
        - score: Suspicion score between 0.0 (authentic) and 1.0 (synthetic)
        - confidence: Reliability metric (0.0 if model fails to load)
        - explanation: Clear human-readable finding
    """
    target_model_id = model_id or settings.ai_detector_model_id

    # 1. Lazily load model components with fail-safe error handling
    try:
        model, processor = get_ai_detector_components(target_model_id)
    except Exception as exc:
        logger.warning(f"Failed to load AI detector model '{target_model_id}': {exc}")
        return SignalResult(
            name="ai_detector",
            score=0.5,
            confidence=0.0,
            explanation=f"AI image detector unavailable: {str(exc)}",
            details={
                "error": str(exc),
                "model_id": target_model_id,
                "synthetic_prob": None,
                "authentic_prob": None,
            },
        )

    # 2. Run inference on CPU
    try:
        with Image.open(io.BytesIO(image_bytes)) as pil_img:
            image_rgb = pil_img.convert("RGB")

        inputs = processor(images=image_rgb, return_tensors="pt")
        inputs = {k: v.to(torch.device("cpu")) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).squeeze()

        # Map labels dynamically based on model id2label
        id2label = getattr(model.config, "id2label", {0: "artificial", 1: "human"})
        synthetic_idx = None
        human_idx = None

        for idx, lbl in id2label.items():
            lbl_lower = str(lbl).lower()
            if any(k in lbl_lower for k in ("artificial", "synthetic", "ai", "fake")):
                synthetic_idx = int(idx)
            elif any(k in lbl_lower for k in ("human", "real", "authentic")):
                human_idx = int(idx)

        # Fallback if label names don't match heuristics
        if synthetic_idx is None:
            synthetic_idx = 0
        if human_idx is None:
            human_idx = 1 if synthetic_idx == 0 else 0

        synthetic_prob = float(probs[synthetic_idx].item())
        authentic_prob = float(probs[human_idx].item())

        score = round(max(0.0, min(1.0, synthetic_prob)), 3)

        # Confidence: highest when prediction is decisive (close to 0 or 1), lower near 0.50
        gap = abs(synthetic_prob - 0.50)
        confidence = round(min(0.95, max(0.50, 0.50 + gap * 0.90)), 3)

        if score >= 0.70:
            explanation = (
                f"Deep learning classifier detected statistical signatures characteristic of "
                f"AI-generated imagery (synthetic probability: {synthetic_prob * 100:.1f}%)."
            )
        elif score <= 0.30:
            explanation = (
                f"Deep learning classifier detected authentic photographic patterns "
                f"(synthetic probability: {synthetic_prob * 100:.1f}%, authentic: {authentic_prob * 100:.1f}%)."
            )
        else:
            explanation = (
                f"Ambiguous generative visual features detected; inconclusive between real and synthetic "
                f"(synthetic probability: {synthetic_prob * 100:.1f}%)."
            )

        return SignalResult(
            name="ai_detector",
            score=score,
            confidence=confidence,
            explanation=explanation,
            details={
                "synthetic_prob": round(synthetic_prob, 4),
                "authentic_prob": round(authentic_prob, 4),
                "model_id": target_model_id,
            },
        )
    except Exception as exc:
        logger.warning(f"Error during AI image detector inference: {exc}")
        return SignalResult(
            name="ai_detector",
            score=0.5,
            confidence=0.0,
            explanation=f"AI image detector inference failed: {str(exc)}",
            details={
                "error": str(exc),
                "model_id": target_model_id,
                "synthetic_prob": None,
                "authentic_prob": None,
            },
        )
