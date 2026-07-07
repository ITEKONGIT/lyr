"""
Phase 6: Compute-Aware Dynamic Face Capping (Solution #9)
"""

import collections
import numpy as np


class FaceCapper:
    """
    Dynamically limits the number of faces processed per frame.
    
    Monitors cycle times and adjusts the cap to prevent the watcher
    from falling behind its target FPS.
    
    Decision matrix:
        - Cycle time < 80% of budget  → Process all faces (up to max)
        - Cycle time 80-120% of budget → Cap at 3 faces
        - Cycle time > 120% of budget  → Cap at 1 face (survival mode)
    """
    
    # Budget thresholds as fraction of cycle time
    HEALTHY_THRESHOLD = 0.80    # Below this: process all
    WARNING_THRESHOLD = 1.20    # Above this: throttle to 1 face
    
    # Face limits per tier
    HEALTHY_LIMIT = 10          # Process up to 10 faces when fast
    WARNING_LIMIT = 3           # Cap at 3 faces when moderate
    CRITICAL_LIMIT = 1          # Cap at 1 face when slow
    
    def __init__(self, cycle_time_ms: float = 100.0, max_faces: int = 10):
        """
        Args:
            cycle_time_ms: Target cycle time in milliseconds (100ms = 10 FPS)
            max_faces: Absolute maximum faces to process per frame
        """
        self.cycle_time_ms = cycle_time_ms
        self.max_faces = max_faces
        
        # Rolling window of recent cycle durations (last 30 cycles)
        self._history = collections.deque(maxlen=30)
        
        # Stats
        self._caps_applied = 0
        self._faces_skipped = 0
    
    def cap(self, faces: list) -> list:
        """
        Apply the dynamic cap to a list of detected faces.
        
        Args:
            faces: List of FaceData objects from the detector
            
        Returns:
            Capped list — may be shorter than input
        """
        if not faces:
            return faces
        
        limit = self._compute_limit()
        capped = faces[:limit]
        
        skipped = len(faces) - len(capped)
        if skipped > 0:
            self._caps_applied += 1
            self._faces_skipped += skipped
        
        return capped
    
    def record_cycle(self, duration_ms: float):
        """
        Record the duration of a completed watcher cycle.

        """
        self._history.append(duration_ms)
    
    def _compute_limit(self) -> int:
        """
        Determine the face limit based on recent cycle performance.

        """
        if len(self._history) < 5:
            # Not enough data — be optimistic
            return min(self.HEALTHY_LIMIT, self.max_faces)
        
        avg_duration = np.mean(self._history)
        
        healthy_budget = self.cycle_time_ms * self.HEALTHY_THRESHOLD
        warning_budget = self.cycle_time_ms * self.WARNING_THRESHOLD
        
        if avg_duration < healthy_budget:
            return min(self.HEALTHY_LIMIT, self.max_faces)
        elif avg_duration < warning_budget:
            return min(self.WARNING_LIMIT, self.max_faces)
        else:
            return min(self.CRITICAL_LIMIT, self.max_faces)
    
    def get_stats(self) -> dict:
        """Return capping statistics for health monitoring."""
        return {
            "cycle_time_ms": self.cycle_time_ms,
            "max_faces": self.max_faces,
            "current_limit": self._compute_limit(),
            "avg_cycle_ms": round(float(np.mean(self._history)), 1) if self._history else 0,
            "caps_applied": self._caps_applied,
            "faces_skipped": self._faces_skipped,
            "history_size": len(self._history),
        }