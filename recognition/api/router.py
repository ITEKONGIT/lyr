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
from typing import Any, Optional, Dict, List

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

from recognition.sensor_contracts import (
    MAX_BATCH_READINGS,
    SensorReading,
    SensorType,
    SensorUnit,
)
from recognition.sensor_history import (
    HistoryBackpressureError,
    MAX_QUERY_LIMIT,
    QueryFilter,
    get_store,
)
from recognition.sensor_registry import get_registry

router = APIRouter(
    prefix="/api/v1",
    tags=["Face Recognition"],
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


class SensorReadingPayload(BaseModel):
    sensor_id: str = Field(..., min_length=1)
    sensor_type: str = Field(...)
    value: float
    timestamp: Optional[datetime] = None
    unit: Optional[str] = None
    confidence_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    source: Optional[str] = None
    device_info: Optional[Dict[str, Any]] = None
    raw_data: Optional[Dict[str, Any]] = None
    location: Optional[Dict[str, float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    reading_id: Optional[str] = None
    face_data: Optional[Dict[str, Any]] = None


class SensorBatchPayload(BaseModel):
    readings: List[Dict[str, Any]] = Field(..., min_length=1)
    source_node: Optional[str] = None


class SensorQueryPayload(BaseModel):
    filters: List[Dict[str, Any]] = Field(default_factory=list)
    limit: int = Field(default=100, ge=1, le=MAX_QUERY_LIMIT)
    order: str = Field(default="desc")


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
# SENSOR INGESTION ENDPOINTS
# ──────────────────────────────────────────────────

@router.post("/sensors/ingest", status_code=201)
async def ingest_sensor_reading(
    payload: SensorReadingPayload,
    api_key: str = Depends(verify_api_key),
):
    """Ingest one generic sensor reading."""
    reading = _payload_to_sensor_reading(payload)
    _record_sensor_reading(reading)
    return {
        "status": "accepted",
        "reading": _public_sensor_reading(reading),
    }


@router.post("/sensors/ingest/batch")
async def ingest_sensor_batch(
    payload: SensorBatchPayload,
    api_key: str = Depends(verify_api_key),
):
    """
    Ingest a batch with partial-success semantics.

    Each item gets its own accepted/rejected status so one malformed reading
    does not discard legitimate readings from the same sensor burst.
    """
    if len(payload.readings) > MAX_BATCH_READINGS:
        raise HTTPException(
            status_code=422,
            detail=f"Batch too large: {len(payload.readings)} > {MAX_BATCH_READINGS}",
        )

    results = []
    accepted = 0
    failed = 0
    for index, item in enumerate(payload.readings):
        try:
            reading_payload = SensorReadingPayload.model_validate(item)
            reading = _payload_to_sensor_reading(reading_payload)
            _record_sensor_reading(reading)
            accepted += 1
            results.append({
                "index": index,
                "status": "accepted",
                "reading_id": reading.reading_id,
            })
        except HTTPException as exc:
            failed += 1
            results.append({
                "index": index,
                "status": "rejected",
                "error": exc.detail,
            })
        except Exception as exc:
            failed += 1
            results.append({
                "index": index,
                "status": "rejected",
                "error": str(exc),
            })

    if accepted and failed:
        status_text = "partial_success"
    elif accepted:
        status_text = "success"
    else:
        status_text = "failed"

    return {
        "status": status_text,
        "accepted": accepted,
        "failed": failed,
        "results": results,
    }


@router.websocket("/sensors/ingest/ws/{sensor_id}")
async def ingest_sensor_stream(websocket: WebSocket, sensor_id: str):
    """Ingest sensor readings over WebSocket with explicit ack/nack replies."""
    from .dependencies import verify_websocket_key

    if not await verify_websocket_key(websocket):
        await websocket.accept()
        await websocket.send_json({
            "status": "rejected",
            "error": "Invalid or missing API key. Provide ?api_key= in the URL.",
        })
        await websocket.close(code=1008, reason="Invalid API key")
        return

    await websocket.accept()
    while True:
        try:
            message = await websocket.receive_json()
            if "sensor_id" in message and message["sensor_id"] != sensor_id:
                await websocket.send_json({
                    "status": "rejected",
                    "error": "sensor_id in message does not match WebSocket path",
                })
                continue

            message["sensor_id"] = sensor_id
            reading_payload = SensorReadingPayload.model_validate(message)
            reading = _payload_to_sensor_reading(reading_payload)
            _record_sensor_reading(reading)
            await websocket.send_json({
                "status": "accepted",
                "reading_id": reading.reading_id,
            })
        except WebSocketDisconnect:
            break
        except HTTPException as exc:
            await websocket.send_json({
                "status": "rejected",
                "error": exc.detail,
            })
        except Exception as exc:
            await websocket.send_json({
                "status": "rejected",
                "error": str(exc),
            })


@router.get("/sensors/status/{sensor_id}")
async def get_sensor_status(
    sensor_id: str,
    api_key: str = Depends(verify_api_key),
):
    """Return registry metadata for a known sensor."""
    sensor = get_registry().get_sensor(sensor_id)
    if sensor is None:
        raise HTTPException(status_code=404, detail="Sensor not found")
    return sensor.to_dict()


@router.get("/sensors/history/{sensor_id}")
async def get_sensor_history(
    sensor_id: str,
    limit: int = 100,
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
    api_key: str = Depends(verify_api_key),
):
    """Return recent readings for a known sensor."""
    if get_registry().get_sensor(sensor_id) is None:
        raise HTTPException(status_code=404, detail="Sensor not found")
    try:
        readings = get_store().get_history(
            sensor_id=sensor_id,
            limit=limit,
            since=since,
            until=until,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "sensor_id": sensor_id,
        "count": len(readings),
        "readings": [_public_sensor_reading(reading) for reading in readings],
    }


@router.post("/sensors/query")
async def query_sensor_history(
    payload: SensorQueryPayload,
    api_key: str = Depends(verify_api_key),
):
    """Structured generic sensor-history query."""
    filters = _build_query_filters(payload.filters)
    try:
        readings = get_store().query(
            filters=filters,
            limit=payload.limit,
            order=payload.order,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "count": len(readings),
        "readings": [_public_sensor_reading(reading) for reading in readings],
    }


# ──────────────────────────────────────────────────
# REGISTRATION ENDPOINTS
# ──────────────────────────────────────────────────

@router.post("/register", response_model=RegisterResponse, status_code=201)
async def register_face(
    payload: RegisterRequest,
    api_key: str = Depends(verify_api_key),
):
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
async def re_enroll_face(
    user_id: str,
    api_key: str = Depends(verify_api_key),
):
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
async def identify_once(api_key: str = Depends(verify_api_key)):
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
async def list_identities(api_key: str = Depends(verify_api_key)):
    """List all enrolled identities (without embedding vectors)."""
    controller = _require_controller()
    
    identities = controller.database.list_all()
    
    return IdentityListResponse(
        count=len(identities),
        identities=[IdentityItem(**i) for i in identities]
    )


@router.delete("/identities/{doc_id}")
async def delete_identity(
    doc_id: str,
    api_key: str = Depends(verify_api_key),
):
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
async def health_check(api_key: str = Depends(verify_api_key)):
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
async def debug_liveness(api_key: str = Depends(verify_api_key)):
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


PROTECTED_FACE_METADATA_KEYS = {"identity", "name", "user_id"}


def _payload_to_sensor_reading(payload: SensorReadingPayload) -> SensorReading:
    data = payload.model_dump(exclude_none=True)
    try:
        if "sensor_type" in data:
            data["sensor_type"] = SensorType.from_string(data["sensor_type"])
        if "unit" in data:
            data["unit"] = SensorUnit.from_string(data["unit"])
        return SensorReading(**data)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _record_sensor_reading(reading: SensorReading) -> None:
    try:
        get_registry().record_reading(reading)
        get_store().record(reading)
    except HistoryBackpressureError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _public_sensor_reading(reading: SensorReading) -> Dict[str, Any]:
    result = reading.to_dict()
    if reading.sensor_type == SensorType.FACE and "metadata" in result:
        result["metadata"] = {
            key: value
            for key, value in result["metadata"].items()
            if key not in PROTECTED_FACE_METADATA_KEYS
        }
        result["metadata_redacted"] = True
    return result


def _build_query_filters(filters: List[Dict[str, Any]]) -> List[QueryFilter]:
    built = []
    for item in filters:
        try:
            query_filter = QueryFilter(
                field=item["field"],
                op=item["op"],
                value=item["value"],
            )
        except KeyError as exc:
            raise HTTPException(
                status_code=422,
                detail=f"Missing query filter field: {exc.args[0]}",
            ) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        if query_filter.op == "LIKE":
            raise HTTPException(
                status_code=403,
                detail="LIKE queries require elevated sensor-query privileges.",
            )
        built.append(query_filter)
    return built
