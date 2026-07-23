"""
Lyr Sensor Contracts - Universal Sensor Ingestion Layer

This module extends the face recognition contracts with generic sensor support.
All sensors (temperature, humidity, accelerometer, etc.) use these contracts.

The existing contracts.py remains untouched for face recognition.
"""

from typing import Any, Dict, Optional, List, Union
from datetime import datetime, timedelta, timezone
from enum import Enum
from dataclasses import dataclass, field
import uuid

# Import the existing face contracts for compatibility
from .contracts import (
    FrameData,
    FaceData,
    LivenessResult,
    EmbeddingData,
    StorageResult,
    RegistrationResult,
    IdentificationResult,
    DetectionEvent,
    QualityMetadata,
    EnrollmentQuality,
    ReEnrollmentResult,
    LightingVarianceError,
)


CRITICAL_CONFIDENCE_THRESHOLD = 0.95
HIGH_CONFIDENCE_THRESHOLD = 0.80
MEDIUM_CONFIDENCE_THRESHOLD = 0.60
LOW_CONFIDENCE_THRESHOLD = 0.40

MIN_AUTONOMOUS_ACTION_THRESHOLD = 0.95
DEFAULT_AUTONOMOUS_ACTION_THRESHOLD = 0.99
STALE_READING_MAX_AGE_SECONDS = 300
FUTURE_TIMESTAMP_TOLERANCE_SECONDS = 5
MAX_BATCH_READINGS = 1000


def _utc_now_naive() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _normalize_utc_naive(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value
    return value.astimezone(timezone.utc).replace(tzinfo=None)


# ──────────────────────────────────────────────────
# SENSOR TYPES
# ──────────────────────────────────────────────────

class SensorType(str, Enum):
    """Supported sensor types across all tiers."""
    
    # Environmental
    TEMPERATURE = "temperature"
    HUMIDITY = "humidity"
    PRESSURE = "pressure"
    AIR_QUALITY = "air_quality"
    GAS = "gas"
    SMOKE = "smoke"
    RAINFALL = "rainfall"
    WATER_LEVEL = "water_level"
    WATER_FLOW = "water_flow"
    SOIL_MOISTURE = "soil_moisture"
    
    # Motion / Position
    ACCELEROMETER = "accelerometer"
    GYROSCOPE = "gyroscope"
    MAGNETOMETER = "magnetometer"
    GPS = "gps"
    ULTRA_SOUND = "ultra_sound"
    
    # Light / Audio
    AMBIENT_LIGHT = "ambient_light"
    MICROPHONE = "microphone"
    ULTRASONIC = "ultrasonic"
    
    # Vision (existing face recognition)
    FACE = "face"
    MOTION_DETECTION = "motion_detection"
    OBJECT_DETECTION = "object_detection"
    
    # Power / Electrical
    BATTERY = "battery"
    CURRENT = "current"
    VOLTAGE = "voltage"
    POWER = "power"
    
    # Generic / Unknown
    UNKNOWN = "unknown"
    
    @classmethod
    def from_string(cls, value: str) -> "SensorType":
        """Convert string to SensorType, defaulting to UNKNOWN."""
        try:
            return cls(value.lower())
        except ValueError:
            return cls.UNKNOWN


# ──────────────────────────────────────────────────
# SENSOR UNITS
# ──────────────────────────────────────────────────

class SensorUnit(str, Enum):
    """Standard units for sensor readings."""
    
    # Temperature
    CELSIUS = "°C"
    FAHRENHEIT = "°F"
    KELVIN = "K"
    
    # Humidity / Pressure
    PERCENT = "%"
    HPA = "hPa"
    MBAR = "mbar"
    PA = "Pa"
    KPA = "kPa"
    ATM = "atm"
    
    # Position / Motion
    METERS = "m"
    METERS_PER_SECOND = "m/s"
    METERS_PER_SECOND_SQUARED = "m/s²"
    RADIANS = "rad"
    DEGREES = "°"
    RADIANS_PER_SECOND = "rad/s"
    DEGREES_PER_SECOND = "°/s"
    
    # Light / Sound
    LUX = "lux"
    DECIBELS = "dB"
    CANDELA = "cd"
    LUMEN = "lm"
    
    # Electrical
    VOLT = "V"
    AMPERE = "A"
    WATT = "W"
    AMPERE_HOUR = "Ah"
    WATT_HOUR = "Wh"
    
    # Generic
    COUNT = "count"
    BOOLEAN = "boolean"
    RATING = "rating"
    NONE = ""
    
    @classmethod
    def from_string(cls, value: str) -> "SensorUnit":
        """Convert string to SensorUnit, defaulting to NONE."""
        try:
            return cls(value)
        except ValueError:
            return cls.NONE


# ──────────────────────────────────────────────────
# CONFIDENCE LEVELS
# ──────────────────────────────────────────────────

class ConfidenceLevel(str, Enum):
    """
    Confidence levels for sensor readings and AI decisions.
    
    CRITICAL: ≥ 95% - Act autonomously
    HIGH:     ≥ 80% - Recommend action, human approval requested
    MEDIUM:   ≥ 60% - Log + notify
    LOW:      ≥ 40% - Just log
    NONE:     < 40% - Discard / ignore
    """
    CRITICAL = "critical"      # ≥ 95%
    HIGH = "high"              # ≥ 80%
    MEDIUM = "medium"          # ≥ 60%
    LOW = "low"                # ≥ 40%
    NONE = "none"              # < 40%
    
    @classmethod
    def from_score(cls, score: Optional[float]) -> "ConfidenceLevel":
        """Convert a numeric confidence score to a ConfidenceLevel."""
        if score is None:
            return cls.NONE
        
        if score >= CRITICAL_CONFIDENCE_THRESHOLD:
            return cls.CRITICAL
        elif score >= HIGH_CONFIDENCE_THRESHOLD:
            return cls.HIGH
        elif score >= MEDIUM_CONFIDENCE_THRESHOLD:
            return cls.MEDIUM
        elif score >= LOW_CONFIDENCE_THRESHOLD:
            return cls.LOW
        else:
            return cls.NONE


# ──────────────────────────────────────────────────
# UNIVERSAL SENSOR READING
# ──────────────────────────────────────────────────

@dataclass
class SensorReading:
    """
    Universal sensor reading contract.
    
    Any sensor (temperature, accelerometer, face, etc.) emits this.
    All tiers (1-4) use this same contract.
    
    The face recognition pipeline can also emit this by wrapping
    its existing IdentificationResult.
    """
    
    # ─── Required Fields ──────────────────────────
    
    sensor_id: str = field(metadata={"description": "Unique identifier for the physical sensor"})
    sensor_type: SensorType = field(metadata={"description": "Type of sensor"})
    value: float = field(metadata={"description": "The numeric reading value"})
    timestamp: datetime = field(default_factory=_utc_now_naive)
    
    # ─── Optional but Recommended ─────────────────
    
    unit: Optional[SensorUnit] = field(
        default=None,
        metadata={"description": "Unit of measurement (if applicable)"}
    )
    
    confidence_score: Optional[float] = field(
        default=None,
        metadata={"description": "Confidence in the reading (0-1)"}
    )
    
    # ─── Metadata ──────────────────────────────────
    
    source: Optional[str] = field(
        default=None,
        metadata={"description": "Source of the reading (phone_browser, weather_api, camera_module)"}
    )
    
    device_info: Optional[Dict[str, Any]] = field(
        default=None,
        metadata={"description": "Device-specific information"}
    )
    
    raw_data: Optional[Dict[str, Any]] = field(
        default=None,
        metadata={"description": "Original unprocessed data from the sensor"}
    )
    
    location: Optional[Dict[str, float]] = field(
        default=None,
        metadata={"description": "GPS or relative location (lat, lng, altitude)"}
    )
    
    metadata: Dict[str, Any] = field(
        default_factory=dict,
        metadata={"description": "Sensor-specific extra data"}
    )
    
    # ─── System Fields ────────────────────────────
    
    reading_id: str = field(
        default_factory=lambda: str(uuid.uuid4()),
        metadata={"description": "Unique ID for this specific reading"}
    )
    
    # ─── Face Recognition Integration ─────────────
    
    face_data: Optional[Dict[str, Any]] = field(
        default=None,
        metadata={"description": "Face recognition data (if sensor_type is FACE)"}
    )
    
    def __post_init__(self):
        """Validate after initialization."""
        self._validate()
    
    def _validate(self):
        """Internal validation."""
        self.timestamp = _normalize_utc_naive(self.timestamp)

        if not self.sensor_id or len(self.sensor_id) < 1:
            raise ValueError("sensor_id must be non-empty")
        
        if self.confidence_score is not None:
            if not (0.0 <= self.confidence_score <= 1.0):
                raise ValueError(f"confidence_score must be between 0 and 1, got {self.confidence_score}")
        
        max_timestamp = (
            _utc_now_naive()
            + timedelta(seconds=FUTURE_TIMESTAMP_TOLERANCE_SECONDS)
        )
        if self.timestamp > max_timestamp:
            raise ValueError("timestamp cannot be in the future")
    
    # ──────────────────────────────────────────────
    # METHODS
    # ──────────────────────────────────────────────
    
    def get_confidence_level(self) -> ConfidenceLevel:
        """Return the confidence level based on the confidence_score."""
        return ConfidenceLevel.from_score(self.confidence_score)
    
    def is_autonomous_action_safe(
        self,
        required_confidence: float = DEFAULT_AUTONOMOUS_ACTION_THRESHOLD,
    ) -> bool:
        """
        Determine if this reading is confident enough for autonomous action.
        
        This is the HIGHEST LEVERAGE design decision in the entire system.
        
        - If confidence >= required_confidence (default 0.99): Safe to act
        - If confidence < required_confidence: Fail TOWARD hard alert
        """
        if required_confidence < MIN_AUTONOMOUS_ACTION_THRESHOLD:
            raise ValueError(
                "required_confidence cannot be below "
                f"{MIN_AUTONOMOUS_ACTION_THRESHOLD:.2f}"
            )

        if self.confidence_score is None:
            return False
        
        return self.confidence_score >= required_confidence
    
    def is_stale(
        self,
        max_age_seconds: int = STALE_READING_MAX_AGE_SECONDS,
    ) -> bool:
        """Check if the reading is stale (older than max_age_seconds)."""
        age = (_utc_now_naive() - self.timestamp).total_seconds()
        return age > max_age_seconds
    
    def is_high_confidence(self) -> bool:
        """Returns True if confidence is CRITICAL or HIGH."""
        level = self.get_confidence_level()
        return level in (ConfidenceLevel.CRITICAL, ConfidenceLevel.HIGH)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excludes raw_data for size)."""
        result = {
            'sensor_id': self.sensor_id,
            'sensor_type': self.sensor_type.value,
            'value': self.value,
            'timestamp': self.timestamp.isoformat(),
            'reading_id': self.reading_id,
        }
        
        if self.unit:
            result['unit'] = self.unit.value
        if self.confidence_score is not None:
            result['confidence_score'] = self.confidence_score
        if self.source:
            result['source'] = self.source
        if self.device_info:
            result['device_info'] = self.device_info
        if self.location:
            result['location'] = self.location
        if self.metadata:
            result['metadata'] = self.metadata
        if self.face_data:
            result['face_data'] = self.face_data
        
        return result
    
    def __repr__(self) -> str:
        return (f"SensorReading(id={self.reading_id[:8]}, "
                f"type={self.sensor_type.value}, "
                f"value={self.value}, "
                f"conf={self.confidence_score or 0:.2f})")


# ──────────────────────────────────────────────────
# BATCH READING
# ──────────────────────────────────────────────────

@dataclass
class SensorReadingBatch:
    """
    Batch of sensor readings for bulk ingestion.
    
    Used by Tier 1 for edge nodes that cache data and burst-upload.
    """
    
    readings: List[SensorReading] = field(
        metadata={"description": "List of sensor readings"}
    )
    
    batch_id: str = field(
        default_factory=lambda: str(uuid.uuid4()),
        metadata={"description": "Unique ID for this batch"}
    )
    
    source_node: Optional[str] = field(
        default=None,
        metadata={"description": "ID of the edge node that sent this batch"}
    )
    
    timestamp: datetime = field(
        default_factory=_utc_now_naive,
        metadata={"description": "UTC timestamp of the batch creation"}
    )
    
    def __post_init__(self):
        """Validate the batch."""
        if not self.readings:
            raise ValueError("Batch must contain at least one reading")
        
        self.timestamp = _normalize_utc_naive(self.timestamp)

        if len(self.readings) > MAX_BATCH_READINGS:
            raise ValueError(
                f"Batch too large: {len(self.readings)} > {MAX_BATCH_READINGS}"
            )
    
    def __repr__(self) -> str:
        return (f"SensorReadingBatch(id={self.batch_id[:8]}, "
                f"readings={len(self.readings)}, "
                f"node={self.source_node or 'unknown'})")


# ──────────────────────────────────────────────────
# SENSOR METADATA (For Registry)
# ──────────────────────────────────────────────────

@dataclass
class SensorMetadata:
    """
    Metadata stored in the sensor registry.
    
    Used by Tier 1 to track all known sensors.
    """
    
    sensor_id: str
    sensor_type: SensorType
    registered_at: datetime = field(default_factory=_utc_now_naive)
    last_reading_at: Optional[datetime] = None
    reading_count: int = 0
    
    # Optional metadata
    location: Optional[Dict[str, float]] = None
    device_info: Optional[Dict[str, Any]] = None
    calibration_data: Optional[Dict[str, Any]] = None
    
    # Status
    is_active: bool = True
    is_trusted: bool = True
    
    # For sensor-specific configuration
    config: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self) -> str:
        return (f"SensorMetadata(id={self.sensor_id}, "
                f"type={self.sensor_type.value}, "
                f"readings={self.reading_count}, "
                f"active={self.is_active})")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'sensor_id': self.sensor_id,
            'sensor_type': self.sensor_type.value,
            'registered_at': self.registered_at.isoformat(),
            'last_reading_at': self.last_reading_at.isoformat() if self.last_reading_at else None,
            'reading_count': self.reading_count,
            'location': self.location,
            'device_info': self.device_info,
            'is_active': self.is_active,
            'is_trusted': self.is_trusted,
            'config': self.config,
        }


# ──────────────────────────────────────────────────
# SENSOR REGISTRY STATE
# ──────────────────────────────────────────────────

@dataclass
class SensorRegistryState:
    """
    Full state of the sensor registry.
    
    Used for status reporting and debugging.
    """
    
    total_sensors: int
    active_sensors: int
    sensor_types: Dict[str, int]  # type -> count
    total_readings: int
    last_update: datetime
    
    def __repr__(self) -> str:
        return (f"SensorRegistryState(sensors={self.total_sensors}, "
                f"active={self.active_sensors}, "
                f"readings={self.total_readings})")


# ──────────────────────────────────────────────────
# WRAPPER FOR FACE RECOGNITION
# ──────────────────────────────────────────────────

def identification_to_sensor_reading(
    result: IdentificationResult,
    sensor_id: str = "face_camera_1",
) -> SensorReading:
    """
    Convert an IdentificationResult to a SensorReading.
    
    This bridges the existing face recognition pipeline to the
    universal sensor contract.
    
    Args:
        result: The IdentificationResult from the face recognition pipeline
        sensor_id: The sensor ID (defaults to "face_camera_1")
    
    Returns:
        SensorReading: The universal sensor reading
    """
    # Determine confidence score from the identification result
    confidence = result.confidence
    
    # If it's a spoof, confidence should be low
    if not result.is_live:
        confidence = min(confidence, 0.3)
    
    # If it's unknown, confidence is moderate
    if result.identity == "Unknown" or not result.is_known:
        confidence = min(confidence, 0.5)
    
    return SensorReading(
        sensor_id=sensor_id,
        sensor_type=SensorType.FACE,
        value=confidence,  # The primary value is the confidence
        confidence_score=confidence,
        timestamp=_utc_now_naive(),
        source="face_recognition_pipeline",
        metadata={
            'identity': result.identity,
            'user_id': result.user_id,
            'name': result.name,
            'is_known': result.is_known,
            'is_live': result.is_live,
            'liveness_variance': result.liveness_variance,
            'bbox': result.bbox,
            'distance': result.distance,
            'processing_time_ms': result.processing_time_ms,
            'authorization': result.authorization,
            'state': result.state,
            'state_reason': result.state_reason,
        }
    )


# ──────────────────────────────────────────────────
# HELPER FUNCTIONS
# ──────────────────────────────────────────────────

def create_sensor_reading(
    sensor_id: str,
    sensor_type: Union[str, SensorType],
    value: float,
    confidence: Optional[float] = None,
    unit: Optional[Union[str, SensorUnit]] = None,
    source: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    location: Optional[Dict[str, float]] = None,
) -> SensorReading:
    """
    Convenience function to create a SensorReading.
    
    Args:
        sensor_id: Unique sensor identifier
        sensor_type: Type of sensor (string or SensorType enum)
        value: The numeric reading
        confidence: Confidence score (0-1)
        unit: Unit of measurement
        source: Source of the reading
        metadata: Additional metadata
        location: GPS/location data
    
    Returns:
        SensorReading: The created reading
    """
    # Convert string to enum if needed
    if isinstance(sensor_type, str):
        sensor_type = SensorType.from_string(sensor_type)
    
    if isinstance(unit, str):
        unit = SensorUnit.from_string(unit)
    
    return SensorReading(
        sensor_id=sensor_id,
        sensor_type=sensor_type,
        value=value,
        confidence_score=confidence,
        unit=unit,
        source=source,
        metadata=metadata or {},
        location=location,
    )


# ──────────────────────────────────────────────────
# EXPOSE ALL CONTRACTS
# ──────────────────────────────────────────────────

__all__ = [
    # Decision constants
    'CRITICAL_CONFIDENCE_THRESHOLD',
    'HIGH_CONFIDENCE_THRESHOLD',
    'MEDIUM_CONFIDENCE_THRESHOLD',
    'LOW_CONFIDENCE_THRESHOLD',
    'MIN_AUTONOMOUS_ACTION_THRESHOLD',
    'DEFAULT_AUTONOMOUS_ACTION_THRESHOLD',
    'STALE_READING_MAX_AGE_SECONDS',
    'FUTURE_TIMESTAMP_TOLERANCE_SECONDS',
    'MAX_BATCH_READINGS',

    # Enums
    'SensorType',
    'SensorUnit',
    'ConfidenceLevel',
    
    # Core contracts
    'SensorReading',
    'SensorReadingBatch',
    'SensorMetadata',
    'SensorRegistryState',
    
    # Bridge function
    'identification_to_sensor_reading',
    'create_sensor_reading',
    
    # Re-export face contracts (for convenience)
    'FrameData',
    'FaceData',
    'LivenessResult',
    'EmbeddingData',
    'StorageResult',
    'RegistrationResult',
    'IdentificationResult',
    'DetectionEvent',
    'QualityMetadata',
    'EnrollmentQuality',
    'ReEnrollmentResult',
    'LightingVarianceError',
]
