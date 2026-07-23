"""
Test the SensorRegistry.

Run with:
    python tests/unit/test_registry.py
"""

import sys
import os

# Add the project root to the Python path
# tests/unit/test_registry.py -> tests/ -> project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from recognition.sensor_registry import get_registry, reset_registry
from recognition.sensor_contracts import create_sensor_reading, SensorType, SensorUnit


def test_registry():
    """Basic registry test."""
    # Reset registry for clean test
    reset_registry()
    registry = get_registry()
    
    print("=== Testing SensorRegistry ===\n")
    
    # Create a temperature reading
    reading1 = create_sensor_reading(
        sensor_id="test_sensor_001",
        sensor_type=SensorType.TEMPERATURE,
        value=23.5,
        confidence=0.95,
        unit=SensorUnit.CELSIUS,
        source="test_script",
        metadata={"location": "lab", "test": True}
    )
    
    # Record it
    registry.record_reading(reading1)
    print(f"✓ Recorded reading: {reading1.sensor_id}")
    
    # Create a humidity reading from same sensor
    reading2 = create_sensor_reading(
        sensor_id="test_sensor_001",
        sensor_type=SensorType.HUMIDITY,
        value=45.0,
        confidence=0.90,
        unit=SensorUnit.PERCENT,
        source="test_script",
    )
    
    registry.record_reading(reading2)
    print(f"✓ Recorded second reading: {reading2.sensor_id}")
    
    # Create a different sensor
    reading3 = create_sensor_reading(
        sensor_id="test_sensor_002",
        sensor_type=SensorType.ACCELEROMETER,
        value=9.81,
        confidence=0.85,
        unit=SensorUnit.METERS_PER_SECOND_SQUARED,
        source="test_script",
    )
    
    registry.record_reading(reading3)
    print(f"✓ Recorded third reading: {reading3.sensor_id}\n")
    
    # Check state
    state = registry.get_state()
    print("=== Registry State ===")
    print(f"Total sensors: {state.total_sensors}")
    print(f"Active sensors: {state.active_sensors}")
    print(f"Total readings: {state.total_readings}")
    print(f"Sensor types: {state.sensor_types}")
    print()
    
    # Get specific sensor
    sensor = registry.get_sensor("test_sensor_001")
    print(f"=== Sensor: test_sensor_001 ===")
    print(f"Type: {sensor.sensor_type.value}")
    print(f"Readings: {sensor.reading_count}")
    print(f"Registered: {sensor.registered_at}")
    print(f"Last reading: {sensor.last_reading_at}")
    print(f"Active: {sensor.is_active}")
    print()
    
    # Get all sensors
    all_sensors = registry.get_all_sensors()
    print(f"=== All Sensors ({len(all_sensors)}) ===")
    for s in all_sensors:
        print(f"  - {s.sensor_id}: {s.sensor_type.value} ({s.reading_count} readings)")
    
    print("\n✅ All tests passed!")


if __name__ == "__main__":
    test_registry()