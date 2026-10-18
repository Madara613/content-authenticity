"""Cross-Modal CLIP Image-Caption Semantic Consistency Module (Phase 2).

Uses OpenAI CLIP (openai/clip-vit-base-patch32) to compute cross-modal cosine similarity
between the visual content of the image and the textual claim/caption.

Low similarity indicates high suspicion of contextual manipulation (out-of-context caption,
cheapfake, or recycled media paired with false text assertions).
"""

import io
import logging
from typing import Any, Dict, Optional, Tuple
from PIL import Image
import torch
from transformers import CLIPModel, CLIPProcessor

from backend.config import settings
from backend.schemas import SignalResult

logger = logging.getLogger(__name__)

# Cached model instances for lazy loading
_clip_model: Optional[Any] = None
_clip_processor: Optional[Any] = None
_clip_cached_model_id: Optional[str] = None


def get_clip_components(model_id: Optional[str] = None) -> Tuple[Any, Any]:
    """Lazily load and cache the CLIP model and processor on CPU."""
    global _clip_model, _clip_processor, _clip_cached_model_id

    target_model_id = model_id or settings.clip_model_id

    if _clip_model is not None and _clip_processor is not None and _clip_cached_model_id == target_model_id:
        return _clip_model, _clip_processor

    logger.info(f"Loading CLIP model '{target_model_id}' on CPU...")
    processor = CLIPProcessor.from_pretrained(target_model_id)
    model = CLIPModel.from_pretrained(target_model_id)
    model.to(torch.device("cpu"))
    model.eval()

    _clip_model = model
    _clip_processor = processor
    _clip_cached_model_id = target_model_id
    return _clip_model, _clip_processor


def clear_clip_cache() -> None:
    """Clear cached CLIP model and processor (primarily for testing)."""
    global _clip_model, _clip_processor, _clip_cached_model_id
    _clip_model = None
    _clip_processor = None
    _clip_cached_model_id = None


def analyze_clip_consistency(
    image_bytes: bytes,
    caption: str,
    model_id: Optional[str] = None,
) -> SignalResult:
    """Evaluate semantic alignment between image and caption using CLIP.
    
    Returns standard SignalResult:
        - name: "clip_consistency"
        - score: Suspicion score (0.0 = high similarity / genuine match,
                 1.0 = severe divergence / cheapfake)
        - confidence: Reliability metric (0.0 if model fails to load)
        - explanation: Clear human-readable summary
    """
    target_model_id = model_id or settings.clip_model_id
    cleaned_caption = caption.strip() if caption else ""

    # 1. Check for empty caption edge case before loading heavy model
    if not cleaned_caption:
        return SignalResult(
            name="clip_consistency",
            score=0.50,
            confidence=0.30,
            explanation="Empty caption provided; cross-modal semantic alignment cannot be computed.",
            details={
                "cosine_similarity": 0.0,
                "caption": "",
                "model_id": target_model_id,
            },
        )

    # 2. Lazily load model components with fail-safe error handling
    try:
        model, processor = get_clip_components(target_model_id)
    except Exception as exc:
        logger.warning(f"Failed to load CLIP model '{target_model_id}': {exc}")
        return SignalResult(
            name="clip_consistency",
            score=0.5,
            confidence=0.0,
            explanation=f"CLIP consistency model unavailable: {str(exc)}",
            details={
                "error": str(exc),
                "model_id": target_model_id,
                "cosine_similarity": None,
            },
        )

    # 3. Run inference on CPU
    try:
        with Image.open(io.BytesIO(image_bytes)) as pil_img:
            image_rgb = pil_img.convert("RGB")

        inputs = processor(
            text=[cleaned_caption],
            images=image_rgb,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(torch.device("cpu")) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            image_embeds = outputs.image_embeds
            text_embeds = outputs.text_embeds

            # L2-normalize both explicitly
            image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
            text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

            cosine_sim = float((image_embeds @ text_embeds.T).squeeze().item())

        # Calibration: low similarity means high suspicion
        # In CLIP, random image/text pairs are around 0.10-0.15; strong matches are >= 0.28.
        if cosine_sim >= 0.28:
            score = max(0.05, 0.10 - (cosine_sim - 0.28) * 0.5)
        elif cosine_sim <= 0.12:
            score = min(0.95, 0.85 + (0.12 - cosine_sim) * 0.5)
        else:
            # Linear scaling between (0.12 -> 0.85) and (0.28 -> 0.10)
            fraction = (0.28 - cosine_sim) / (0.28 - 0.12)
            score = 0.10 + fraction * (0.85 - 0.10)

        score = round(max(0.0, min(1.0, score)), 3)

        # Confidence: lower for very short captions
        if len(cleaned_caption.split()) < 3:
            confidence = 0.60
        else:
            confidence = 0.85

        if score >= 0.70:
            explanation = (
                f"Significant semantic divergence detected between image visual content and caption "
                f"(CLIP similarity: {cosine_sim:.3f}); suggests possible out-of-context caption reuse or cheapfake."
            )
        elif score <= 0.25:
            explanation = (
                f"Image visual content aligns strongly with caption claims (CLIP similarity: {cosine_sim:.3f})."
            )
        else:
            explanation = (
                f"Moderate semantic correlation between image and caption (CLIP similarity: {cosine_sim:.3f})."
            )

        return SignalResult(
            name="clip_consistency",
            score=score,
            confidence=confidence,
            explanation=explanation,
            details={
                "cosine_similarity": round(cosine_sim, 4),
                "caption": cleaned_caption,
                "model_id": target_model_id,
            },
        )
    except Exception as exc:
        logger.warning(f"Error during CLIP consistency inference: {exc}")
        return SignalResult(
            name="clip_consistency",
            score=0.5,
            confidence=0.0,
            explanation=f"CLIP consistency inference failed: {str(exc)}",
            details={
                "error": str(exc),
                "model_id": target_model_id,
                "cosine_similarity": None,
            },
        )
