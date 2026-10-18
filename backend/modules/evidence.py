"""Fact-Checking and Cross-Reference Evidence Module (Phase 3).

Integrates:
1. Gemini with Google Search tool grounding (grounded_check)
2. Sentence-Transformers + FAISS similarity lookup over LIAR benchmark (local_lookup)
3. Synthesis into standard SignalResult contract (evidence_signal)
"""

import json
import logging
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from google import genai
from google.genai import types

from backend.config import settings
from backend.schemas import SignalResult

logger = logging.getLogger(__name__)

DEFAULT_GEMINI_MODEL = "gemini-3.1-flash-lite"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# In-memory cache for evidence signals per claim
_evidence_cache: Dict[str, SignalResult] = {}

# Cached local model & FAISS index
_st_model: Optional[Any] = None
_faiss_index: Optional[Any] = None
_faiss_metadata: Optional[List[Dict[str, Any]]] = None


def clear_evidence_cache() -> None:
    """Clear in-memory evidence signal cache and reset cached models."""
    global _evidence_cache, _st_model, _faiss_index, _faiss_metadata
    _evidence_cache.clear()
    _st_model = None
    _faiss_index = None
    _faiss_metadata = None


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


def _clean_json_text(text: str) -> str:
    """Strip markdown code block fences and trailing formatting."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)
    return cleaned.strip()


def grounded_check(
    claim: str,
    api_key: Optional[str] = None,
    model_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Call Gemini with the Google Search tool to verify a factual claim.
    
    Returns:
        {
            "verdict": "supported" | "contradicted" | "unclear",
            "confidence": float (0.0 to 1.0),
            "reason": str,
            "sources": [{"title": str, "url": str}, ...]
        }
    If there are no sources returned, forces verdict to "unclear".
    """
    clean_claim = claim.strip() if claim else ""
    if not clean_claim:
        return {
            "verdict": "unclear",
            "confidence": 0.0,
            "reason": "Empty claim provided.",
            "sources": [],
        }

    resolved_key = _resolve_gemini_api_key(api_key)
    if not resolved_key:
        logger.warning("GEMINI_API_KEY is not configured; grounded search unavailable.")
        return {
            "verdict": "unclear",
            "confidence": 0.0,
            "reason": "Gemini API key is not configured.",
            "sources": [],
        }

    resolved_model = _resolve_gemini_model(model_id)

    prompt = (
        "You are a strict, objective fact-checking assistant.\n"
        "Verify the following factual claim using Google Search:\n"
        f'Claim: "{clean_claim}"\n\n'
        "Evaluate whether reliable recent search results support or contradict this claim.\n"
        "Return your assessment strictly as a raw JSON object with NO markdown code fences, matching this schema:\n"
        "{\n"
        '  "verdict": "supported" | "contradicted" | "unclear",\n'
        '  "confidence": <float between 0.0 and 1.0>,\n'
        '  "reason": "<one or two sentence factual summary of findings>"\n'
        "}"
    )

    try:
        client = genai.Client(api_key=resolved_key)
        response = client.models.generate_content(
            model=resolved_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                temperature=0.0,
            ),
        )

        # 1. Collect sources from grounding metadata
        sources: List[Dict[str, str]] = []
        if response.candidates:
            candidate = response.candidates[0]
            gm = getattr(candidate, "grounding_metadata", None)
            if gm:
                chunks = getattr(gm, "grounding_chunks", None) or []
                for chunk in chunks:
                    web = getattr(chunk, "web", None)
                    if web:
                        uri = getattr(web, "uri", None) or getattr(web, "url", None) or ""
                        title = getattr(web, "title", None) or uri
                        if uri and not any(s["url"] == uri for s in sources):
                            sources.append({"title": title, "url": uri})

        # 2. Parse JSON response text
        raw_text = response.text or ""
        cleaned_json = _clean_json_text(raw_text)

        parsed_data: Dict[str, Any] = {}
        try:
            parsed_data = json.loads(cleaned_json)
        except Exception:
            # Fallback regex extraction if model returned slightly imperfect JSON
            v_match = re.search(r'"verdict"\s*:\s*"([^"]+)"', cleaned_json, re.IGNORECASE)
            c_match = re.search(r'"confidence"\s*:\s*([0-9.]+)', cleaned_json)
            r_match = re.search(r'"reason"\s*:\s*"([^"]+)"', cleaned_json)
            parsed_data = {
                "verdict": v_match.group(1).lower() if v_match else "unclear",
                "confidence": float(c_match.group(1)) if c_match else 0.30,
                "reason": r_match.group(1) if r_match else cleaned_json[:200],
            }

        verdict = str(parsed_data.get("verdict", "unclear")).lower().strip()
        if verdict not in ("supported", "contradicted", "unclear"):
            verdict = "unclear"

        try:
            confidence = float(parsed_data.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))
        except (ValueError, TypeError):
            confidence = 0.5

        reason = str(parsed_data.get("reason", "Web search analysis completed.")).strip()

        # Rule: If there are no sources, force verdict "unclear"
        if not sources:
            verdict = "unclear"
            confidence = min(confidence, 0.30)
            reason = (
                f"{reason} (Note: No verifiable external web sources were retrieved by search grounding.)"
                if reason
                else "No external web sources retrieved by search grounding; verdict forced to unclear."
            )

        return {
            "verdict": verdict,
            "confidence": round(confidence, 3),
            "reason": reason,
            "sources": sources,
        }
    except Exception as exc:
        # Never log the API key!
        logger.warning(f"Grounded search check failed: {type(exc).__name__}: {str(exc)}")
        return {
            "verdict": "unclear",
            "confidence": 0.0,
            "reason": f"Grounded search check error: {str(exc)}",
            "sources": [],
        }


def _get_local_index_components(index_dir: Optional[Path] = None) -> Tuple[Optional[Any], Optional[Any], Optional[List[Dict[str, Any]]]]:
    """Lazily load SentenceTransformer model and FAISS LIAR index."""
    global _st_model, _faiss_index, _faiss_metadata

    target_dir = index_dir or settings.liar_index_dir
    index_file = target_dir / "liar_index.faiss"
    metadata_file = target_dir / "liar_metadata.json"

    # Lazy-load embedding model
    if _st_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model '{EMBEDDING_MODEL_NAME}' on CPU...")
            _st_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")
        except Exception as exc:
            logger.warning(f"Failed to load SentenceTransformer '{EMBEDDING_MODEL_NAME}': {exc}")
            return None, None, None

    # Lazy-load FAISS index and metadata if available on disk
    if _faiss_index is None or _faiss_metadata is None:
        if index_file.exists() and metadata_file.exists():
            try:
                import faiss
                logger.info(f"Loading FAISS LIAR index from {index_file}...")
                _faiss_index = faiss.read_index(str(index_file))
                with open(metadata_file, "r", encoding="utf-8") as f:
                    _faiss_metadata = json.load(f)
            except Exception as exc:
                logger.warning(f"Failed to load FAISS index from {target_dir}: {exc}")
                _faiss_index = None
                _faiss_metadata = None
        else:
            # Index has not been built yet
            return _st_model, None, None

    return _st_model, _faiss_index, _faiss_metadata


def local_lookup(
    claim: str,
    k: int = 3,
    min_sim: float = 0.6,
    index_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Search the local LIAR FAISS index for semantically similar checked claims.
    
    Returns list of items:
        [{"statement": str, "label": str, "similarity": float}, ...]
    filtered by similarity >= min_sim, or an empty list.
    """
    clean_claim = claim.strip() if claim else ""
    if not clean_claim:
        return []

    try:
        model, index, metadata = _get_local_index_components(index_dir=index_dir)
        if model is None or index is None or metadata is None:
            return []

        # Encode query claim to normalized vector
        query_emb = model.encode([clean_claim], normalize_embeddings=True)
        query_vec = np.array(query_emb, dtype=np.float32)

        distances, indices = index.search(query_vec, k)
        results: List[Dict[str, Any]] = []

        if len(distances) > 0 and len(indices) > 0:
            for sim, idx in zip(distances[0], indices[0]):
                if idx < 0 or idx >= len(metadata):
                    continue
                similarity = float(sim)
                if similarity >= min_sim:
                    item = metadata[idx]
                    results.append({
                        "statement": item.get("statement", ""),
                        "label": item.get("label", "unknown"),
                        "similarity": round(similarity, 3),
                        "speaker": item.get("speaker", ""),
                    })

        return results
    except Exception as exc:
        logger.warning(f"Local LIAR index lookup failed: {exc}")
        return []


def evidence_signal(
    claim: Optional[str],
    api_key: Optional[str] = None,
    model_id: Optional[str] = None,
    index_dir: Optional[Path] = None,
) -> SignalResult:
    """Synthesize Google Search grounded check and local LIAR index into standard SignalResult.
    
    Contract:
        {name: "claim_evidence", score (0-1 suspicion), confidence, explanation, details}
    Web verdict weighs more than the local lookup. Low or unclear evidence must give
    LOW confidence, not a strong score.
    """
    clean_claim = claim.strip() if claim else ""
    if not clean_claim:
        return SignalResult(
            name="claim_evidence",
            score=0.50,
            confidence=0.0,
            explanation="No checkable factual claim extracted from caption.",
            details={
                "claim": None,
                "web_verdict": "unclear",
                "sources": [],
                "local_matches": [],
            },
        )

    # In-memory caching per claim
    if clean_claim in _evidence_cache:
        return _evidence_cache[clean_claim]

    # Check API key before proceeding
    resolved_key = _resolve_gemini_api_key(api_key)
    if not resolved_key:
        result = SignalResult(
            name="claim_evidence",
            score=0.50,
            confidence=0.0,
            explanation="Gemini API key is not configured; claim evidence verification unavailable.",
            details={
                "claim": clean_claim,
                "web_verdict": "unclear",
                "sources": [],
                "local_matches": [],
            },
        )
        _evidence_cache[clean_claim] = result
        return result

    # 1. Web Grounded Search
    web_res = grounded_check(clean_claim, api_key=resolved_key, model_id=model_id)

    # 2. Local LIAR index lookup
    local_matches = local_lookup(clean_claim, k=3, min_sim=0.6, index_dir=index_dir)

    web_verdict = web_res.get("verdict", "unclear")
    web_conf = web_res.get("confidence", 0.0)
    web_reason = web_res.get("reason", "")
    sources = web_res.get("sources", [])

    # If the call failed with confidence 0.0
    if web_conf == 0.0 and web_verdict == "unclear" and "error" in web_reason.lower():
        result = SignalResult(
            name="claim_evidence",
            score=0.50,
            confidence=0.0,
            explanation=f"Grounded evidence search failed: {web_reason}",
            details={
                "claim": clean_claim,
                "web_verdict": "unclear",
                "sources": [],
                "local_matches": local_matches,
            },
        )
        _evidence_cache[clean_claim] = result
        return result

    # 3. Combine into score & confidence
    # Web verdict weighs more than local lookup.
    # Low or unclear evidence must give LOW confidence, not a strong score.
    score: float
    confidence: float
    explanation: str

    if web_verdict == "supported":
        # Corroborated factual claim -> low suspicion
        score = round(max(0.05, 0.15 - web_conf * 0.10), 3)
        confidence = round(max(0.60, min(0.95, web_conf)), 3)
        explanation = f"Search evidence supports the claim: {web_reason}"
        if local_matches:
            top_m = local_matches[0]
            explanation += f" (Local benchmark match: '{top_m['statement'][:60]}...' labeled {top_m['label']})."
    elif web_verdict == "contradicted":
        # Contradicted claim -> high suspicion
        score = round(min(0.95, 0.85 + web_conf * 0.10), 3)
        confidence = round(max(0.65, min(0.95, web_conf)), 3)
        explanation = f"Search evidence contradicts the claim: {web_reason}"
        if local_matches:
            top_m = local_matches[0]
            explanation += f" (Local benchmark match: '{top_m['statement'][:60]}...' labeled {top_m['label']})."
    else:
        # Unclear or insufficient external sources -> LOW confidence, neutral suspicion
        score = 0.50
        confidence = round(min(0.35, max(0.15, web_conf)), 3)
        explanation = f"Evidence is inconclusive or uncorroborated: {web_reason}"
        if local_matches:
            top_m = local_matches[0]
            explanation += f" Found similar past claim in LIAR dataset: '{top_m['statement'][:60]}...' labeled {top_m['label']} (similarity: {top_m['similarity']:.2f})."

    # Subtle confidence adjustment from local match corroboration
    if local_matches and web_verdict != "unclear":
        confidence = round(min(0.98, confidence + 0.05), 3)

    result = SignalResult(
        name="claim_evidence",
        score=score,
        confidence=confidence,
        explanation=explanation,
        details={
            "claim": clean_claim,
            "web_verdict": web_verdict,
            "web_confidence": web_conf,
            "reason": web_reason,
            "sources": sources,
            "local_matches": local_matches,
        },
    )

    _evidence_cache[clean_claim] = result
    return result
