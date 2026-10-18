"""Evidence verification module re-export."""
from backend.modules.evidence import (
    grounded_check,
    local_lookup,
    evidence_signal,
    clear_evidence_cache,
)

__all__ = [
    "grounded_check",
    "local_lookup",
    "evidence_signal",
    "clear_evidence_cache",
]
