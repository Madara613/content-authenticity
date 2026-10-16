from datetime import datetime, timezone
from typing import Any, Dict, List
from pydantic import BaseModel, Field


class SignalResult(BaseModel):
    name: str = Field(..., description="Identifier of the analysis module")
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Suspicion score between 0.0 (authentic) and 1.0 (anomalous/tampered)",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 (uncertain) and 1.0 (highly certain)",
    )
    explanation: str = Field(..., description="Human-readable explanation of signal findings")
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Module-specific forensic metrics and flags",
    )


class AnalyzeResponse(BaseModel):
    filename: str
    caption: str
    signals: List[SignalResult]


class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str = "1.0.0"
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    modules_loaded: List[str]
