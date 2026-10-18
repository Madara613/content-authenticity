"""Factual Claim Extraction Module (Phase 3).

Uses Google GenAI SDK (gemini-3.1-flash-lite) to extract the primary checkable
factual claim from an image caption or claim string. If the caption contains
only subjective opinions, pure visual descriptions, or empty text, it returns None.
"""

import logging
import os
import re
from typing import Dict, Optional

from google import genai
from google.genai import types

from backend.config import settings

logger = logging.getLogger(__name__)

DEFAULT_GEMINI_MODEL = "gemini-3.1-flash-lite"

# In-memory cache for claim extraction per caption
_claim_cache: Dict[str, Optional[str]] = {}


def clear_claim_cache() -> None:
    """Clear in-memory claim extraction cache (useful for testing)."""
    global _claim_cache
    _claim_cache.clear()


def _resolve_gemini_model(model_id: Optional[str] = None) -> str:
    """Resolve Gemini model ID from argument, environment variable, or default."""
    if model_id and model_id.strip():
        return model_id.strip()
    env_model = os.getenv("GEMINI_MODEL")
    if env_model and env_model.strip():
        return env_model.strip()
    return getattr(settings, "gemini_model", DEFAULT_GEMINI_MODEL) or DEFAULT_GEMINI_MODEL


def _resolve_gemini_api_key(api_key: Optional[str] = None) -> Optional[str]:
    """Resolve Gemini API key without ever logging it."""
    if api_key and api_key.strip():
        return api_key.strip()
    env_key = os.getenv("GEMINI_API_KEY")
    if env_key and env_key.strip():
        return env_key.strip()
    cfg_key = getattr(settings, "gemini_api_key", None)
    if cfg_key and cfg_key.strip():
        return cfg_key.strip()
    return None


def extract_claim(
    caption: str,
    api_key: Optional[str] = None,
    model_id: Optional[str] = None,
) -> Optional[str]:
    """Extract the main factual claim from a caption in one short sentence.
    
    If the caption has no checkable factual claim (pure description, opinion,
    or empty text), returns None. Results are cached in memory per caption.
    """
    clean_caption = caption.strip() if caption else ""
    if not clean_caption:
        return None

    # Check in-memory cache
    if clean_caption in _claim_cache:
        return _claim_cache[clean_caption]

    resolved_key = _resolve_gemini_api_key(api_key)
    if not resolved_key:
        logger.warning("GEMINI_API_KEY is not configured; skipping claim extraction.")
        _claim_cache[clean_caption] = None
        return None

    resolved_model = _resolve_gemini_model(model_id)

    prompt = (
        "You are an expert fact-checking claim extractor.\n"
        "Analyze the following text/caption and extract the single primary checkable factual claim "
        "as one concise sentence.\n"
        "A checkable factual claim is an assertion about real-world events, public figures, policies, "
        "historical occurrences, statistics, or actions that can be verified as true or false.\n\n"
        "Rules:\n"
        "1. If the caption asserts a checkable factual claim, extract and return it as a single objective sentence.\n"
        "2. If the caption contains NO checkable factual claim (such as pure visual descriptions like "
        "'a cute cat on a table' or 'sunset over mountains', subjective opinions, feelings, questions, "
        "or greetings), output exactly: null\n"
        "3. Output ONLY the claim string or the word null. Do not include markdown code fences, quotes, "
        "or explanations.\n\n"
        f'Caption: "{clean_caption}"'
    )

    try:
        client = genai.Client(api_key=resolved_key)
        response = client.models.generate_content(
            model=resolved_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
            ),
        )

        raw_text = (response.text or "").strip()
        # Clean potential quotes and markdown formatting
        cleaned_text = re.sub(r"^[`'\"]+|[`'\"]+$", "", raw_text).strip()

        if not cleaned_text or cleaned_text.lower() in ("null", "none", "n/a", "no claim", "null.", "none."):
            extracted_claim = None
        else:
            extracted_claim = cleaned_text

        _claim_cache[clean_caption] = extracted_claim
        return extracted_claim
    except Exception as exc:
        # Never log the API key!
        logger.warning(f"Gemini claim extraction failed: {type(exc).__name__}: {str(exc)}")
        _claim_cache[clean_caption] = None
        return None
