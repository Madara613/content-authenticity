import io
import os
from pathlib import Path
from typing import List
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from backend.config import settings
from backend.modules.ai_detector import analyze_ai_generation
from backend.modules.claim_extractor import extract_claim
from backend.modules.clip_consistency import analyze_clip_consistency
from backend.modules.ela import analyze_ela
from backend.modules.evidence import evidence_signal
from backend.modules.metadata import analyze_metadata
from backend.schemas import AnalyzeResponse, HealthResponse, SignalResult

app = FastAPI(
    title="Multimodal Content Authenticity API",
    description="Backend API for verifying image and claim authenticity (Phase 1, Phase 2, & Phase 3).",
    version="1.0.0",
)

# CORS setup
origins = [origin.strip() for origin in settings.cors_origins.split(",") if origin.strip()]
if not origins or origins == ["*"]:
    allow_origins = ["*"]
else:
    allow_origins = origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure heatmap output directory exists
settings.heatmap_dir.mkdir(parents=True, exist_ok=True)
app.mount(
    "/heatmaps",
    StaticFiles(directory=str(settings.heatmap_dir), check_dir=False),
    name="heatmaps",
)


@app.get("/health", response_model=HealthResponse, tags=["System"])
@app.get("/api/health", response_model=HealthResponse, include_in_schema=False)
async def health_check() -> HealthResponse:
    """Return system readiness probe and loaded Phase 1, 2, & 3 modules."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        modules_loaded=[
            "metadata",
            "ela",
            "ai_detector",
            "clip_consistency",
            "claim_extractor",
            "claim_evidence",
        ],
    )


async def validate_and_read_image(image: UploadFile) -> bytes:
    """Validate upload format, size, and integrity.
    
    Returns raw image bytes if valid, or raises HTTPException with clear error.
    """
    filename = image.filename or ""
    ext = Path(filename).suffix.lower()

    # 1. Extension check
    if ext and ext not in settings.allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported file format '{ext}'. "
                f"Only JPEG, PNG, and WEBP images are supported."
            ),
        )

    # 2. Content-Type check (if provided by client)
    if image.content_type and image.content_type not in settings.allowed_mime_types:
        # If client sends an explicit disallowed mime type like application/pdf or image/gif
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported MIME type '{image.content_type}'. "
                f"Only image/jpeg, image/png, and image/webp are supported."
            ),
        )

    # 3. Read content with size enforcement
    contents = await image.read()
    if len(contents) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    if len(contents) > settings.max_upload_size_bytes:
        max_mb = settings.max_upload_size_bytes // (1024 * 1024)
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"File size ({len(contents) / (1024 * 1024):.2f} MB) exceeds maximum allowed limit of {max_mb} MB.",
        )

    # 4. Verify image readability and actual format with Pillow
    try:
        with Image.open(io.BytesIO(contents)) as pil_img:
            format_name = (pil_img.format or "").upper()
            if format_name not in ("JPEG", "PNG", "WEBP"):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=(
                        f"Detected image format '{format_name}' is not supported. "
                        f"Only JPEG, PNG, and WEBP are supported."
                    ),
                )
            # Verify image data stream integrity
            pil_img.verify()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Corrupted or invalid image data: {str(exc)}",
        )

    return contents


@app.post("/analyze", response_model=AnalyzeResponse, tags=["Verification"])
@app.post("/api/analyze", response_model=AnalyzeResponse, include_in_schema=False)
async def analyze_endpoint(
    image: UploadFile = File(..., description="Image file to verify (JPEG, PNG, WEBP, <= 10MB)"),
    caption: str = Form(..., description="Caption or claim text accompanying the image"),
) -> AnalyzeResponse:
    """Analyze image using metadata, ELA, AI detector, and CLIP cross-modal consistency."""
    image_bytes = await validate_and_read_image(image)
    filename = image.filename or "upload.jpg"

    # Module 1: Metadata analysis (EXIF & C2PA)
    metadata_signal: SignalResult = analyze_metadata(image_bytes, filename=filename)

    # Module 2: Error Level Analysis (ELA & Heatmap)
    ela_signal: SignalResult = analyze_ela(
        image_bytes,
        filename=filename,
        output_dir=settings.heatmap_dir,
    )

    # Module 3: AI-Generated Image Detector (Lazy loaded on CPU)
    ai_signal: SignalResult = analyze_ai_generation(image_bytes, filename=filename)

    # Module 4: Cross-Modal CLIP Consistency (Lazy loaded on CPU)
    clip_signal: SignalResult = analyze_clip_consistency(image_bytes, caption=caption)

    # Module 5 & 6: Claim Extraction & Grounded Evidence Verification (Phase 3)
    try:
        extracted_claim: Optional[str] = extract_claim(caption)
    except Exception as exc:
        extracted_claim = None

    try:
        claim_evidence_signal: SignalResult = evidence_signal(extracted_claim)
    except Exception as exc:
        claim_evidence_signal = SignalResult(
            name="claim_evidence",
            score=0.50,
            confidence=0.0,
            explanation=f"Evidence verification failed: {str(exc)}",
            details={"claim": extracted_claim},
        )

    signals: List[SignalResult] = [
        metadata_signal,
        ela_signal,
        ai_signal,
        clip_signal,
        claim_evidence_signal,
    ]

    return AnalyzeResponse(
        filename=filename,
        caption=caption,
        claim=extracted_claim,
        signals=signals,
    )
