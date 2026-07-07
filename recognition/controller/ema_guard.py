"""
Phase 6: Rate-Limited EMA Guard (Solution #6)

Design:
    - In-memory timestamp tracker (no persistence needed — resets on restart)
    - Per-identity cooldown: default 6 hours
    - Thread-safe for concurrent watcher cycles
    - Zero external dependencies
"""

import time
import threading
from typing import Dict


class EmaGuard:
    """
    Rate-limits EMA vector updates per identity.
    
    Each identity (document ID) can only receive one EMA update
    within the cooldown window. Subsequent attempts are silently ignored.

    """
    
    # ──────────────────────────────────────────────
    # CONFIGURATION
    # ──────────────────────────────────────────────
    
    DEFAULT_COOLDOWN_SECONDS = 21600  # 6 hours
    
    # ──────────────────────────────────────────────
    # LIFECYCLE
    # ──────────────────────────────────────────────
    
    def __init__(self, cooldown_seconds: float = None):
        """
        Args:
            cooldown_seconds: Minimum seconds between EMA updates per identity.
                              Defaults to 6 hours (21600 seconds).
                              Set to 3600 for 1 hour, 60 for testing.
        """
        self.cooldown_seconds = cooldown_seconds or self.DEFAULT_COOLDOWN_SECONDS
        
        # Per-identity last update timestamps
        # Key: document_id (str), Value: timestamp (float)
        self._last_update: Dict[str, float] = {}
        
        # Thread safety — watcher runs in a thread
        self._lock = threading.Lock()
        
        # Stats
        self._attempts = 0
        self._allowed = 0
        self._blocked = 0
    
    # ──────────────────────────────────────────────
    # PUBLIC API
    # ──────────────────────────────────────────────
    
    def try_update(self, doc_id: str) -> bool:
        """
        Attempt to acquire an EMA update slot for an identity.
        
        Thread-safe. Returns True if the update is allowed (cooldown
        has expired), False if it should be skipped.

        """
        if not doc_id:
            return False
        
        self._attempts += 1
        now = time.time()
        
        with self._lock:
            last = self._last_update.get(doc_id)
            
            if last is None:
                # First update for this identity — allow it
                self._last_update[doc_id] = now
                self._allowed += 1
                return True
            
            elapsed = now - last
            
            if elapsed >= self.cooldown_seconds:
                # Cooldown expired — allow update
                self._last_update[doc_id] = now
                self._allowed += 1
                return True
            else:
                # Still in cooldown — block
                self._blocked += 1
                return False
    
    def reset_identity(self, doc_id: str):
        """
        Manually reset the cooldown for a specific identity.
        
        Useful for testing or when an admin forces a re-enrollment.
        """
        with self._lock:
            self._last_update.pop(doc_id, None)
    
    def reset_all(self):
        """Reset all cooldowns. Use with caution."""
        with self._lock:
            self._last_update.clear()
    
    def get_cooldown_remaining(self, doc_id: str) -> float:
        """
        Get remaining cooldown time for an identity in seconds.
        
        Returns:
            Seconds remaining, or 0 if no cooldown is active.
        """
        with self._lock:
            last = self._last_update.get(doc_id)
            if last is None:
                return 0.0
            
            elapsed = time.time() - last
            remaining = self.cooldown_seconds - elapsed
            
            return max(0.0, remaining)
    
    # ──────────────────────────────────────────────
    # STATS
    # ──────────────────────────────────────────────
    
    def get_stats(self) -> dict:
        """Return guard statistics for health monitoring."""
        with self._lock:
            active_cooldowns = sum(
                1 for t in self._last_update.values()
                if time.time() - t < self.cooldown_seconds
            )
        
        return {
            "cooldown_seconds": self.cooldown_seconds,
            "cooldown_hours": round(self.cooldown_seconds / 3600, 1),
            "total_attempts": self._attempts,
            "allowed": self._allowed,
            "blocked": self._blocked,
            "active_cooldowns": active_cooldowns,
            "total_tracked": len(self._last_update),
        }