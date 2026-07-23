"""
Lyr Sensor Registry - Tracks all known sensors and their metadata.

The registry is a thread-safe singleton that:
- Auto-registers new sensors on first reading
- Maintains per-sensor metadata (type, location, last reading, etc.)
- Provides status and statistics
- Supports querying by sensor type, location, or status

All operations are thread-safe for concurrent sensor ingestion.
"""

import threading
from typing import Any, Dict, Optional, List, Set
from datetime import datetime
from collections import defaultdict

from .sensor_contracts import (
    SensorReading,
    SensorType,
    SensorMetadata,
    SensorRegistryState,
)


class SensorRegistry:
    """
    Thread-safe registry for all known sensors.
    
    Singleton pattern — use get_registry() to access the global instance.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the registry."""
        if self._initialized:
            return
        
        self._sensors: Dict[str, SensorMetadata] = {}
        self._sensors_by_type: Dict[SensorType, Set[str]] = defaultdict(set)
        self._sensors_by_location: Dict[str, Set[str]] = defaultdict(set)
        self._registry_lock = threading.RLock()
        
        self._total_readings = 0
        self._last_cleanup = datetime.utcnow()
        
        self._initialized = True
        print(f"[SensorRegistry] Initialized at {datetime.utcnow().isoformat()}")
    
    # ──────────────────────────────────────────────
    # REGISTRATION
    # ──────────────────────────────────────────────
    
    def register(
        self,
        sensor_id: str,
        sensor_type: SensorType,
        location: Optional[Dict[str, float]] = None,
        device_info: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> SensorMetadata:
        """
        Register a new sensor or update an existing one.
        
        Args:
            sensor_id: Unique sensor identifier
            sensor_type: Type of sensor
            location: GPS or relative location
            device_info: Device-specific information
            config: Sensor-specific configuration
        
        Returns:
            SensorMetadata: The updated metadata
        """
        with self._registry_lock:
            # Check if sensor exists
            if sensor_id in self._sensors:
                # Update existing
                metadata = self._sensors[sensor_id]
                if location:
                    metadata.location = location
                if device_info:
                    metadata.device_info = device_info
                if config:
                    metadata.config = config
                metadata.is_active = True
                metadata.last_reading_at = datetime.utcnow()
                return metadata
            
            # Create new sensor
            metadata = SensorMetadata(
                sensor_id=sensor_id,
                sensor_type=sensor_type,
                location=location,
                device_info=device_info,
                config=config or {},
                registered_at=datetime.utcnow(),
                is_active=True,
            )
            
            self._sensors[sensor_id] = metadata
            self._sensors_by_type[sensor_type].add(sensor_id)
            
            if location and 'room' in location:
                room = str(location['room'])
                self._sensors_by_location[room].add(sensor_id)
            
            print(f"[SensorRegistry] Registered: {sensor_id} ({sensor_type.value})")
            return metadata
    
    def record_reading(self, reading: SensorReading) -> None:
        """
        Record a sensor reading and update metadata.
        
        This is called by the ingest endpoints for every valid reading.
        """
        with self._registry_lock:
            # Register or update the sensor
            metadata = self.register(
                sensor_id=reading.sensor_id,
                sensor_type=reading.sensor_type,
                location=reading.location,
                device_info=reading.device_info,
            )
            
            # Update metadata
            metadata.last_reading_at = reading.timestamp
            metadata.reading_count += 1
            metadata.is_active = True
            
            self._total_readings += 1
    
    # ──────────────────────────────────────────────
    # QUERIES
    # ──────────────────────────────────────────────
    
    def get_sensor(self, sensor_id: str) -> Optional[SensorMetadata]:
        """Get metadata for a specific sensor."""
        with self._registry_lock:
            return self._sensors.get(sensor_id)
    
    def get_all_sensors(self) -> List[SensorMetadata]:
        """Get metadata for all registered sensors."""
        with self._registry_lock:
            return list(self._sensors.values())
    
    def get_sensors_by_type(self, sensor_type: SensorType) -> List[SensorMetadata]:
        """Get all sensors of a specific type."""
        with self._registry_lock:
            sensor_ids = self._sensors_by_type.get(sensor_type, set())
            return [self._sensors[sid] for sid in sensor_ids if sid in self._sensors]
    
    def get_sensors_by_location(self, location: str) -> List[SensorMetadata]:
        """Get all sensors in a specific location."""
        with self._registry_lock:
            sensor_ids = self._sensors_by_location.get(location, set())
            return [self._sensors[sid] for sid in sensor_ids if sid in self._sensors]
    
    def get_active_sensors(self) -> List[SensorMetadata]:
        """Get all active sensors."""
        with self._registry_lock:
            return [s for s in self._sensors.values() if s.is_active]
    
    def get_inactive_sensors(self) -> List[SensorMetadata]:
        """Get all inactive sensors."""
        with self._registry_lock:
            return [s for s in self._sensors.values() if not s.is_active]
    
    def sensor_exists(self, sensor_id: str) -> bool:
        """Check if a sensor is registered."""
        with self._registry_lock:
            return sensor_id in self._sensors
    
    # ──────────────────────────────────────────────
    # STATUS & STATISTICS
    # ──────────────────────────────────────────────
    
    def get_state(self) -> SensorRegistryState:
        """Get the current state of the registry."""
        with self._registry_lock:
            # Count by type
            type_counts = {}
            for sensor_type, ids in self._sensors_by_type.items():
                active_count = sum(
                    1 for sid in ids 
                    if sid in self._sensors and self._sensors[sid].is_active
                )
                type_counts[sensor_type.value] = active_count
            
            active_count = sum(1 for s in self._sensors.values() if s.is_active)
            
            return SensorRegistryState(
                total_sensors=len(self._sensors),
                active_sensors=active_count,
                sensor_types=type_counts,
                total_readings=self._total_readings,
                last_update=datetime.utcnow(),
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get detailed statistics for monitoring."""
        state = self.get_state()
        return {
            "total_sensors": state.total_sensors,
            "active_sensors": state.active_sensors,
            "inactive_sensors": state.total_sensors - state.active_sensors,
            "sensor_types": state.sensor_types,
            "total_readings": state.total_readings,
            "last_update": state.last_update.isoformat(),
        }
    
    # ──────────────────────────────────────────────
    # MAINTENANCE
    # ──────────────────────────────────────────────
    
    def mark_inactive(self, sensor_id: str, reason: str = "") -> None:
        """Mark a sensor as inactive."""
        with self._registry_lock:
            if sensor_id in self._sensors:
                self._sensors[sensor_id].is_active = False
                print(f"[SensorRegistry] Marked inactive: {sensor_id} ({reason})")
    
    def cleanup_stale_sensors(self, max_age_seconds: int = 86400) -> int:
        """
        Mark sensors as inactive if they haven't reported in max_age_seconds.
        
        Default: 24 hours.
        
        Returns:
            int: Number of sensors marked inactive.
        """
        now = datetime.utcnow()
        stale_count = 0
        
        with self._registry_lock:
            for sensor_id, metadata in self._sensors.items():
                if not metadata.is_active:
                    continue
                
                if metadata.last_reading_at is None:
                    continue
                
                age = (now - metadata.last_reading_at).total_seconds()
                if age > max_age_seconds:
                    metadata.is_active = False
                    stale_count += 1
                    print(f"[SensorRegistry] Stale: {sensor_id} (age: {age/3600:.1f}h)")
        
        return stale_count
    
    # ──────────────────────────────────────────────
    # RESET
    # ──────────────────────────────────────────────
    
    def reset(self) -> None:
        """Clear all registry data (for testing)."""
        with self._registry_lock:
            self._sensors.clear()
            self._sensors_by_type.clear()
            self._sensors_by_location.clear()
            self._total_readings = 0
            print("[SensorRegistry] Reset complete")


# ──────────────────────────────────────────────
# GLOBAL INSTANCE
# ──────────────────────────────────────────────

_registry: Optional[SensorRegistry] = None


def get_registry() -> SensorRegistry:
    """Get the global SensorRegistry instance."""
    global _registry
    if _registry is None:
        _registry = SensorRegistry()
    return _registry


def reset_registry() -> None:
    """Reset the global registry (for testing)."""
    global _registry
    if _registry is not None:
        _registry.reset()
    _registry = None