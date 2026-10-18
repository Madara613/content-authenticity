"""Claim extraction module re-export."""
from backend.modules.claim_extractor import (
    extract_claim,
    clear_claim_cache,
)

__all__ = [
    "extract_claim",
    "clear_claim_cache",
]
