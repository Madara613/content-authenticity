"""AI detector module re-export."""
from backend.modules.ai_detector import (
    analyze_ai_generation,
    get_ai_detector_components,
    clear_ai_detector_cache,
)

__all__ = [
    "analyze_ai_generation",
    "get_ai_detector_components",
    "clear_ai_detector_cache",
]
