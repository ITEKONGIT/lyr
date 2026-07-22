"""
API route definitions — Phase 7 Enhanced

Endpoints:
    POST   /api/v1/register              — Synchronous face registration with quality
    PUT    /api/v1/identities/{id}/re-enroll — Update existing identity
    POST   /api/v1/identify              — One-shot identification
    GET    /api/v1/identities            — List all enrolled identities
    DELETE /api/v1/identities/{id}       — Delete an identity
    GET    /api/v1/health                — System health and stats
    GET    /api/v1/debug/liveness        — Live liveness calibration
    WS     /api/v1/stream/detections     — Real-time identification stream
"""

import sys
import os
from datetime import datetime, timezone
from typing import Optional, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from .dependencies import (
    get_controller,
    get_broadcaster,
    is_controller_running,
)

from .dependencies import (
    get_controller,
    get_broadcaster,
    is_controller_running,
    verify_api_key,  # ADD THIS
)

router = APIRouter(
    prefix="/api/v1",
    tags=["Face Recognition"],
    dependencies=[Depends(verify_api_key)]  # ADD THIS — protects ALL routes
)


# ──────────────────────────────────────────────────
# REQUEST / RESPONSE MODELS
# ──────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    user_id: str = Field(..., min_length=1, description="Unique user identifier")
    name: str = Field(..., min_length=1, description="Display name")
    metadata: Optional[Dict] = Field(default=None, description="Optional metadata")


class ReEnrollRequest(BaseModel):
    """Re-enrollment only needs user_id — identity is verified by face."""
    pass


class QualityInfo(BaseModel):
    """Enrollment quality details."""
    overall_score: float
    grade: str
    avg_confidence: Optional[float] = None
    avg_liveness_variance: Optional[float] = None
    lighting_condition: Optional[str] = None
    pose_diversity: Optional[float] = None
    avg_self_similarity: Optional[float] = None
    frames_used: Optional[int] = None
    frames_captured: Optional[int] = None
    correlation: Optional[float] = None
    blend_ratio: Optional[float] = None


class RegisterResponse(BaseModel):
    status: str
    transaction_id: str
    user_id: str
    name: str
    liveness: Optional[Dict]
    embedding_dim: int
    processing_time_ms: float
    timestamp: str
    error: str
    # Phase 7: Quality fields
    quality: Optional[QualityInfo] = None
    quality_score: float = 0.0
    quality_grade: str = ""
    lighting_condition: str = ""
    frames_used: int = 0
    frames_captured: int = 0
    avg_self_similarity: float = 0.0
    recommendation: str = ""


class IdentityItem(BaseModel):
    id: str
    user_id: str
    name: str
    stored_at: str


class IdentityListResponse(BaseModel):
    count: int
    identities: List[IdentityItem]


class IdentifyResponse(BaseModel):
    identity: str
    user_id: str
    name: str
    confidence: float
    distance: float
    is_known: bool
    is_live: bool
    liveness_variance: float
    bbox: List[int]
    processing_time_ms: float
    authorization: str = ""
    state: str = ""
    state_reason: str = ""


class HealthResponse(BaseModel):
    status: str
    uptime_seconds: float
    database_count: int
    registrations_completed: int
    camera: Dict
    liveness: Dict
    watcher: Optional[Dict]
    face_capper: Optional[Dict]
    ema_guard: Optional[Dict]
    broadcaster: Optional[Dict]


class LivenessDebugResponse(BaseModel):
    variance: float
    threshold: float
    is_live: bool
    running_mean: float
    samples_seen: int


# ──────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────

def _require_controller():
    """Raise 503 if controller is not running."""
    if not is_controller_running():
        raise HTTPException(
            status_code=503,
            detail="Face recognition system not running"
        )
    return get_controller()


def _build_register_response(result) -> RegisterResponse:
    """Build a RegisterResponse from a RegistrationResult."""
    quality = None
    if result.quality:
        quality = QualityInfo(
            overall_score=result.quality.get("overall_score", 0),
            grade=result.quality.get("grade", ""),
            avg_confidence=result.quality.get("avg_confidence"),
            avg_liveness_variance=result.quality.get("avg_liveness_variance"),
            lighting_condition=result.quality.get("lighting_condition"),
            pose_diversity=result.quality.get("pose_diversity"),
            avg_self_similarity=result.quality.get("avg_self_similarity"),
            frames_used=result.quality.get("frames_used"),
            frames_captured=result.quality.get("frames_captured"),
            correlation=result.quality.get("correlation"),
            blend_ratio=result.quality.get("blend_ratio"),
        )
    
    return RegisterResponse(
        status=result.status,
        transaction_id=result.transaction_id,
        user_id=result.user_id,
        name=result.name,
        liveness=result.liveness,
        embedding_dim=result.embedding_dim,
        processing_time_ms=result.processing_time_ms,
        timestamp=result.timestamp,
        error=result.error,
        quality=quality,
        quality_score=result.quality_score,
        quality_grade=result.quality_grade,
        lighting_condition=result.lighting_condition,
        frames_used=result.frames_used,
        frames_captured=result.frames_captured,
        avg_self_similarity=result.avg_self_similarity,
        recommendation=result.recommendation,
    )


# ──────────────────────────────────────────────────
# REGISTRATION ENDPOINTS
# ──────────────────────────────────────────────────

@router.post("/register", response_model=RegisterResponse, status_code=201)
async def register_face(payload: RegisterRequest):
    """
    Register a face synchronously with quality assessment.
    
    The camera captures 5 frames over ~3 seconds. Each frame is assessed
    for lighting quality, detection confidence, and liveness. A quality-
    weighted centroid is computed and stored.

    """
    controller = _require_controller()
    
    result = controller.register_face(
        user_id=payload.user_id,
        name=payload.name,
        metadata=payload.metadata,
    )
    
    if result.status == "success":
        return _build_register_response(result)
    
    # Map error types to HTTP codes
    if "already exists" in result.error.lower():
        raise HTTPException(status_code=409, detail=result.error)
    elif result.liveness and not result.liveness.get("is_live"):
        raise HTTPException(status_code=400, detail=result.error)
    elif "lighting" in result.error.lower() or "insufficient" in result.error.lower():
        raise HTTPException(status_code=400, detail=result.error)
    else:
        raise HTTPException(status_code=422, detail=result.error)


@router.put("/identities/{user_id}/re-enroll", response_model=RegisterResponse)
async def re_enroll_face(user_id: str):
    """
    Update an existing identity with fresh face data.

    """
    controller = _require_controller()
    
    result = controller.re_enroll_face(user_id)
    
    if result.status == "success":
        return _build_register_response(result)
    
    if "no existing identity" in result.error.lower():
        raise HTTPException(status_code=404, detail=result.error)
    elif "doesn't correlate" in result.error.lower():
        raise HTTPException(status_code=400, detail=result.error)
    else:
        raise HTTPException(status_code=422, detail=result.error)


# ──────────────────────────────────────────────────
# IDENTIFICATION ENDPOINTS
# ──────────────────────────────────────────────────

@router.post("/identify", response_model=IdentifyResponse)
async def identify_once():
    """
    One-shot identification — capture a single frame and identify
    the largest face in it.

    """
    controller = _require_controller()
    
    result = controller.identify_once()
    
    if result is None:
        raise HTTPException(status_code=503, detail="Camera not available")
    
    if result.get("identity") == "No face detected":
        raise HTTPException(status_code=422, detail="No face detected in frame")
    
    return IdentifyResponse(
        identity=result["identity"],
        user_id=result["user_id"],
        name=result["name"],
        confidence=result["confidence"],
        distance=result["distance"],
        is_known=result["is_known"],
        is_live=result["is_live"],
        liveness_variance=result["liveness_variance"],
        bbox=result["bbox"],
        processing_time_ms=result["processing_time_ms"],
        authorization=result.get("authorization", ""),
        state=result.get("state", ""),
        state_reason=result.get("state_reason", ""),
    )


# ──────────────────────────────────────────────────
# WEBSOCKET STREAMING ENDPOINT
# ──────────────────────────────────────────────────

@router.websocket("/stream/detections")
async def stream_detections(websocket: WebSocket):
    """
    Real-time identification stream at ~10 FPS.
    
    Requires API key as query parameter: ?api_key=your_key
    """
    # Check API key for WebSocket connection
    from .dependencies import verify_websocket_key
    
    if not await verify_websocket_key(websocket):
        await websocket.accept()
        await websocket.send_json({
            "error": "Invalid or missing API key. Provide ?api_key= in the URL."
        })
        await websocket.close(code=1008, reason="Invalid API key")
        return
    
    broadcaster = get_broadcaster()

    if broadcaster is None:
        await websocket.accept()
        await websocket.send_json({
            "error": "Broadcaster not running. Start the system first."
        })
        await websocket.close()
        return

    await broadcaster.connect(websocket)

    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        await broadcaster.disconnect(websocket)

# ──────────────────────────────────────────────────
# IDENTITY MANAGEMENT ENDPOINTS
# ──────────────────────────────────────────────────

@router.get("/identities", response_model=IdentityListResponse)
async def list_identities():
    """List all enrolled identities (without embedding vectors)."""
    controller = _require_controller()
    
    identities = controller.database.list_all()
    
    return IdentityListResponse(
        count=len(identities),
        identities=[IdentityItem(**i) for i in identities]
    )


@router.delete("/identities/{doc_id}")
async def delete_identity(doc_id: str):
    """Delete an enrolled identity by document ID."""
    controller = _require_controller()
    
    success = controller.database.delete(doc_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Identity not found")
    
    return {"status": "deleted", "id": doc_id}


# ──────────────────────────────────────────────────
# HEALTH & DEBUG ENDPOINTS
# ──────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """System health and throughput statistics."""
    controller = _require_controller()
    stats = controller.get_stats()
    
    return HealthResponse(
        status="healthy" if stats["is_running"] else "degraded",
        uptime_seconds=stats["uptime_seconds"],
        database_count=stats["database_count"],
        registrations_completed=stats.get("registrations_completed", 0),
        camera=stats.get("camera", {}),
        liveness=stats.get("liveness", {}),
        watcher=stats.get("watcher"),
        face_capper=stats.get("face_capper"),
        ema_guard=stats.get("ema_guard"),
        broadcaster=stats.get("broadcaster"),
    )


@router.get("/debug/liveness", response_model=LivenessDebugResponse)
async def debug_liveness():
    """
    Live liveness variance monitor for threshold calibration.
    
    Captures current frame, detects face, runs liveness
    on the original-resolution face region.
    """
    controller = _require_controller()
    
    frame = controller.camera.get_latest_frame()
    if frame is None:
        raise HTTPException(status_code=503, detail="Camera not available")
    
    face = controller.face_detector.detect_largest_face(frame)
    
    if face:
        liveness = controller.liveness_gate.check_full_frame_region(
            frame.pixel_array, face.bbox
        )
        variance = liveness.variance
        is_live = liveness.is_live
    else:
        variance = controller.liveness_gate._compute_laplacian_variance(
            frame.pixel_array
        )
        is_live = variance >= controller.liveness_gate.threshold
    
    stats = controller.liveness_gate.get_stats()
    
    return LivenessDebugResponse(
        variance=round(float(variance), 2),
        threshold=stats["threshold"],
        is_live=is_live,
        running_mean=stats["running_mean_variance"],
        samples_seen=stats["samples_seen"],
    )