"""
Shared data contracts for the face recognition pipeline.
Every module produces and consumes these verifiable data structures.

"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
from datetime import datetime


# ──────────────────────────────────────────────────
# PHASE 7: QUALITY & ENROLLMENT TYPES
# ──────────────────────────────────────────────────

@dataclass
class QualityMetadata:
    """
    Per-frame quality tracking during enrollment.

    """
    frame_index: int              # Which frame (0-4)
    confidence: float             # Face detection confidence [0, 1]
    liveness_variance: float      # Laplacian variance
    is_live: bool                
    embedding_quality: float      
    lighting: str                 
    weight: float = 1.0           
    
    def __repr__(self) -> str:
        return (f"QualityMetadata(frame={self.frame_index}, "
                f"confidence={self.confidence:.2f}, "
                f"lighting={self.lighting}, weight={self.weight:.2f})")


@dataclass
class EnrollmentQuality:
    """
    Composite enrollment quality assessment.

    """
    overall_score: float          # 0.0–1.0 composite quality
    grade: str                    # "excellent", "good", "marginal", "poor"
    avg_confidence: float         # Mean detection confidence
    avg_liveness_variance: float  # Mean Laplacian variance
    lighting_condition: str       # Overall lighting: "excellent", "good", "acceptable", "low"
    pose_diversity: float         # 0.0–1.0 how varied the poses were
    avg_self_similarity: float    # Mean similarity of each frame to centroid
    frames_captured: int          # Total frames attempted
    frames_used: int              # Frames that passed all gates
    recommendation: str           # Human-readable recommendation
    
    def is_acceptable(self) -> bool:
        """Returns True if this enrollment meets minimum quality standards."""
        return self.grade in ("excellent", "good", "marginal")
    
    def __repr__(self) -> str:
        return (f"EnrollmentQuality(grade={self.grade}, "
                f"score={self.overall_score:.2f}, "
                f"frames={self.frames_used}/{self.frames_captured})")


@dataclass
class ReEnrollmentResult:
    """
    Phase 7 output — Result of updating an existing identity.

    """
    status: str                   # "success" or "failed"
    user_id: str
    name: str = ""
    correlation: float = 0.0      # Similarity between old and new centroid
    blend_ratio: float = 0.0      # How much new data was blended (0.0–1.0)
    old_quality: Optional[Dict] = None   # Previous enrollment quality
    new_quality: Optional[Dict] = None   # New enrollment quality
    processing_time_ms: float = 0.0
    timestamp: str = ""
    error: str = ""
    
    def is_success(self) -> bool:
        return self.status == "success"
    
    def __repr__(self) -> str:
        return (f"ReEnrollmentResult({self.status}, "
                f"correlation={self.correlation:.3f}, "
                f"blend={self.blend_ratio:.2f})")


class LightingVarianceError(ValueError):
    """
    Raised when a frame's Laplacian variance falls below the liveness threshold.
    
    Indicates the frame has insufficient texture detail — likely a spoof
    (photo/screen) or extremely poor lighting conditions.
    """
    def __init__(self, variance: float, threshold: float,
                 message: str = ""):
        self.variance = variance
        self.threshold = threshold
        if not message:
            message = (f"Lighting variance {variance:.1f} below "
                       f"threshold {threshold:.1f}")
        super().__init__(message)


# ──────────────────────────────────────────────────
# PHASE 1: FRAME DATA
# ──────────────────────────────────────────────────

@dataclass
class FrameData:
    """
    Phase 1 output — A single captured frame with metadata.
    """
    pixel_array: np.ndarray
    frame_number: int
    timestamp: float
    resolution: Tuple[int, int]
    capture_latency_ms: float = 0.0
    
    def verify_integrity(self) -> bool:
        """Self-test: Is this frame valid and usable?"""
        if self.pixel_array is None:
            print("[FAIL] pixel_array is None")
            return False
        
        if len(self.pixel_array.shape) != 3:
            print(f"[FAIL] Expected 3 dimensions, got {len(self.pixel_array.shape)}")
            return False
        
        if self.pixel_array.shape[2] != 3:
            print(f"[FAIL] Expected 3 channels, got {self.pixel_array.shape[2]}")
            return False
        
        actual_height, actual_width = self.pixel_array.shape[:2]
        expected_width, expected_height = self.resolution
        
        if (actual_width, actual_height) != (expected_width, expected_height):
            print(f"[FAIL] Resolution mismatch: expected {self.resolution}, "
                  f"got ({actual_width}, {actual_height})")
            return False
        
        if np.mean(self.pixel_array) < 1.0:
            print("[FAIL] Frame appears to be completely black")
            return False
        
        if self.frame_number < 0:
            print(f"[FAIL] Invalid frame number: {self.frame_number}")
            return False
        
        return True
    
    def verify_consistency_with(self, other: 'FrameData') -> bool:
        """Cross-frame check: does this frame logically follow another?"""
        if self.frame_number <= other.frame_number:
            print(f"[FAIL] Frame number didn't increment: "
                  f"{other.frame_number} -> {self.frame_number}")
            return False
        
        if self.timestamp <= other.timestamp:
            print(f"[FAIL] Timestamp didn't advance")
            return False
        
        if self.resolution != other.resolution:
            print(f"[FAIL] Resolution changed mid-stream: "
                  f"{other.resolution} -> {self.resolution}")
            return False
        
        return True
    
    def __repr__(self) -> str:
        return (f"FrameData(#{self.frame_number}, "
                f"{self.resolution[0]}x{self.resolution[1]}, "
                f"mean_pixel={np.mean(self.pixel_array):.1f})")


# ──────────────────────────────────────────────────
# PHASE 2: FACE DATA & LIVENESS
# ──────────────────────────────────────────────────

@dataclass
class FaceData:
    """
    Phase 2 output — A detected and aligned face crop.
    """
    cropped_face: np.ndarray
    bbox: Tuple[int, int, int, int]
    left_eye: Tuple[int, int]
    right_eye: Tuple[int, int]
    confidence: float
    source_frame_number: int
    target_size: Tuple[int, int] = (160, 160)
    
    def verify_alignment(self) -> bool:
        """Self-test: Is this face properly aligned?"""
        if self.cropped_face is None:
            print("[FAIL] cropped_face is None")
            return False
        
        actual_h, actual_w = self.cropped_face.shape[:2]
        expected_w, expected_h = self.target_size
        
        if (actual_w, actual_h) != (expected_w, expected_h):
            print(f"[FAIL] Crop dimensions: expected {self.target_size}, "
                  f"got ({actual_w}, {actual_h})")
            return False
        
        if len(self.cropped_face.shape) != 3:
            print(f"[FAIL] Expected 3 dimensions, got {len(self.cropped_face.shape)}")
            return False
        
        if self.cropped_face.shape[2] != 3:
            print(f"[FAIL] Expected 3 channels, got {self.cropped_face.shape[2]}")
            return False
        
        eye_y_diff = abs(self.left_eye[1] - self.right_eye[1])
        eye_x_dist = abs(self.left_eye[0] - self.right_eye[0])
        
        if eye_x_dist > 0:
            angle = abs(np.degrees(np.arctan(eye_y_diff / eye_x_dist)))
            if angle > 5.0:
                print(f"[WARN] Eyes not horizontal: {angle:.1f} degrees")
        
        if not (0.0 <= self.confidence <= 1.0):
            print(f"[FAIL] Confidence out of range: {self.confidence}")
            return False
        
        x, y, w, h = self.bbox
        if w <= 0 or h <= 0:
            print(f"[FAIL] Invalid bbox dimensions: {self.bbox}")
            return False
        
        if self.source_frame_number < 0:
            print(f"[FAIL] Invalid source frame: {self.source_frame_number}")
            return False
        
        return True
    
    def __repr__(self) -> str:
        return (f"FaceData(bbox={self.bbox}, "
                f"crop={self.target_size}, "
                f"confidence={self.confidence:.3f})")


@dataclass
class LivenessResult:
    """
    Phase 2b output — Result of the liveness check.
    """
    is_live: bool
    variance: float
    threshold: float
    reason: str = ""
    
    def verify_threshold(self) -> bool:
        """Self-test: Does the variance measurement make physical sense?"""
        if self.variance < 0:
            print("[FAIL] Variance cannot be negative")
            return False
        if self.threshold <= 0:
            print("[FAIL] Threshold must be positive")
            return False
        return True
    
    def __repr__(self) -> str:
        status = "LIVE" if self.is_live else "SPOOF"
        return (f"LivenessResult({status}, variance={self.variance:.1f}, "
                f"threshold={self.threshold:.0f})")


# ──────────────────────────────────────────────────
# PHASE 3: EMBEDDING DATA
# ──────────────────────────────────────────────────

@dataclass
class EmbeddingData:
    """
    Phase 3 output — L2-normalized face embedding vector.
    """
    vector: np.ndarray
    dimension: int = 512
    source_frame_number: int = -1
    model_name: str = "arcface"
    extraction_time_ms: float = 0.0
    
    def verify_normalization(self) -> bool:
        """Self-test: Is this a proper unit vector?"""
        if self.vector is None:
            print("[FAIL] vector is None")
            return False
        
        if len(self.vector.shape) != 1:
            print(f"[FAIL] Expected 1D vector, got shape {self.vector.shape}")
            return False
        
        if self.vector.shape[0] != self.dimension:
            print(f"[FAIL] Dimension mismatch: expected {self.dimension}, "
                  f"got {self.vector.shape[0]}")
            return False
        
        if np.any(np.isnan(self.vector)):
            print("[FAIL] Vector contains NaN values")
            return False
        
        if np.any(np.isinf(self.vector)):
            print("[FAIL] Vector contains Inf values")
            return False
        
        norm = np.linalg.norm(self.vector)
        if abs(norm - 1.0) > 1e-4:
            print(f"[FAIL] L2 norm is {norm:.6f}, expected 1.0")
            return False
        
        return True
    
    def cosine_similarity(self, other: 'EmbeddingData') -> float:
        """
        Compute cosine similarity with another embedding.
        Since both are L2-normalized, this is just the dot product.
        
        Returns:
            Similarity score [0.0, 1.0] where 1.0 = identical
        """
        return float(np.dot(self.vector, other.vector))
    
    def cosine_distance(self, other: 'EmbeddingData') -> float:
        """
        Cosine distance = 1 - cosine_similarity.
        Returns:
            Distance [0.0, 2.0] where 0.0 = identical
        """
        return 1.0 - self.cosine_similarity(other)
    
    def __repr__(self) -> str:
        return (f"EmbeddingData(dim={self.dimension}, "
                f"norm={np.linalg.norm(self.vector):.4f}, "
                f"model={self.model_name})")


# ──────────────────────────────────────────────────
# PHASE 4: STORAGE RESULT
# ──────────────────────────────────────────────────

@dataclass
class StorageResult:
    """
    Phase 4 output — Result of a database operation.
    """
    document_id: str
    user_id: str
    name: str = ""
    stored_vector: Optional[np.ndarray] = None
    retrieved_vector: Optional[np.ndarray] = None
    query_latency_ms: float = 0.0
    timestamp: str = ""
    
    def verify_roundtrip(self) -> bool:
        """
        Self-test: Did we get back what we stored?

        """
        if self.stored_vector is None:
            print("[FAIL] stored_vector is None")
            return False
        
        if self.retrieved_vector is None:
            print("[FAIL] retrieved_vector is None — storage roundtrip failed")
            return False
        
        if not np.allclose(self.stored_vector, self.retrieved_vector, atol=1e-6):
            diff = np.max(np.abs(self.stored_vector - self.retrieved_vector))
            print(f"[FAIL] Vectors differ beyond tolerance — max delta: {diff:.10f}")
            return False
        
        if not self.document_id:
            print("[FAIL] Document ID is empty")
            return False
        
        return True
    
    def __repr__(self) -> str:
        return (f"StorageResult(id={self.document_id[:8]}..., "
                f"user={self.user_id}, "
                f"roundtrip={'OK' if self.verify_roundtrip() else 'FAIL'})")


# ──────────────────────────────────────────────────
# PHASE 5: REGISTRATION RESULT (EXTENDED FOR PHASE 7)
# ──────────────────────────────────────────────────

@dataclass
class RegistrationResult:
    """
    Phase 5/7 output — Result of a complete face registration.
    
    Phase 7 adds quality, lighting, and enrollment metadata fields.
    """
    status: str
    transaction_id: str = ""
    user_id: str = ""
    name: str = ""
    liveness: Optional[Dict[str, Any]] = None
    embedding_dim: int = 512
    processing_time_ms: float = 0.0
    timestamp: str = ""
    error: str = ""
    
    # Phase 7: Quality fields
    quality: Optional[Dict[str, Any]] = None       # EnrollmentQuality as dict
    quality_score: float = 0.0                      # 0.0–1.0
    quality_grade: str = ""                         # "excellent", "good", "marginal", "poor"
    lighting_condition: str = ""                    # Overall lighting assessment
    frames_used: int = 0                            # Frames that passed all gates
    frames_captured: int = 0                        # Total frames attempted
    avg_self_similarity: float = 0.0                # Mean similarity to centroid
    recommendation: str = ""                        # Human-readable feedback
    
    def is_success(self) -> bool:
        return self.status == "success"
    
    def __repr__(self) -> str:
        return (f"RegistrationResult({self.status}, "
                f"user={self.user_id}, "
                f"quality={self.quality_grade}, "
                f"time={self.processing_time_ms:.1f}ms)")


# ──────────────────────────────────────────────────
# PHASE 6: IDENTIFICATION & DETECTION
# ──────────────────────────────────────────────────

@dataclass
class IdentificationResult:
    """
    Phase 6 output — Single face identification from the watcher.
    
    Fields:
        identity: Human-readable label ("Known: {name}", "Unknown", "Spoof", "Error")
        authorization: Security posture — "authorized", "unauthorized", "alert", "error", "idle"
        state: State identifier from the 10-state authorization matrix
        state_reason: Human-readable explanation of the state decision
    """
    identity: str
    user_id: str
    name: str
    confidence: float
    distance: float
    is_known: bool
    is_live: bool
    liveness_variance: float
    bbox: Tuple[int, int, int, int]
    processing_time_ms: float
    
    # Authorization layer
    authorization: str = "idle"
    state: str = "empty_frame"
    state_reason: str = ""
    
    def __repr__(self) -> str:
        return (f"IdentificationResult({self.identity}, "
                f"confidence={self.confidence:.3f}, "
                f"live={self.is_live}, "
                f"auth={self.authorization})")


@dataclass
class DetectionEvent:
    """
    Phase 6 output — Emitted by the watcher for every processed frame.
    Pushed to WebSocket clients for real-time dashboard updates.
    """
    timestamp: str
    system_state: str
    frame_number: int
    detections: List[Dict]
    watcher_fps: float
    
    def __repr__(self) -> str:
        return (f"DetectionEvent(frame=#{self.frame_number}, "
                f"detections={len(self.detections)}, "
                f"fps={self.watcher_fps:.1f})")