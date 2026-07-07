import cv2
import numpy as np
from typing import Tuple, Optional
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from contracts import FaceData, LivenessResult


class LivenessGate:
    """
    Stateless liveness detector using Laplacian variance.

    """
    
    # Pre-configured thresholds for common scenarios
    PRESETS = {
        "strict": 150.0,     # High-security: fewer false accepts
        "balanced": 100.0,   # Default: good balance
        "permissive": 50.0,  # Low-light: fewer false rejects
    }
    
    def __init__(self, threshold: float = 100.0):
        """
        Args:
            threshold: Minimum Laplacian variance to consider a face "live".
                       Tune this against your specific camera and lighting.
        """
        if threshold <= 0:
            raise ValueError(f"Threshold must be positive, got {threshold}")
        
        self.threshold = threshold
        self._samples_seen = 0
        self._running_mean = 0.0
        
        print(f"[LivenessGate] Initialized with threshold={threshold:.1f}")
    
    # ──────────────────────────────────────────────────
    # PUBLIC API
    # ──────────────────────────────────────────────────
    
    def check(self, face_crop: np.ndarray) -> LivenessResult:
        """
        Run liveness check on a single face crop.
        
        The face crop can be the original frame region or an aligned crop.
        For best results, use the original (non-resized) face region to 
        preserve texture detail.

        """
        # Validate input
        if face_crop is None or face_crop.size == 0:
            return LivenessResult(
                is_live=False,
                variance=0.0,
                threshold=self.threshold,
                reason="Empty or null face crop"
            )
        
        # Compute Laplacian variance
        variance = self._compute_laplacian_variance(face_crop)
        
        # Track statistics for monitoring
        self._samples_seen += 1
        self._running_mean += (variance - self._running_mean) / self._samples_seen
        
        # Decision
        is_live = variance >= self.threshold
        
        reason = ""
        if not is_live:
            if variance < self.threshold * 0.33:
                reason = (f"Very low texture (variance={variance:.1f}) — "
                          f"likely a digital screen or high-gloss photo")
            elif variance < self.threshold * 0.66:
                reason = (f"Low texture (variance={variance:.1f}) — "
                          f"possible printed photo or very flat lighting")
            else:
                reason = (f"Borderline texture (variance={variance:.1f}) — "
                          f"just below threshold {self.threshold:.1f}")
        
        return LivenessResult(
            is_live=is_live,
            variance=round(float(variance), 2),
            threshold=self.threshold,
            reason=reason
        )
    
    def check_face_data(self, face_data: FaceData) -> LivenessResult:
        """
        Convenience method — run liveness on a FaceData object.
        Uses the cropped (aligned) face.
        """
        return self.check(face_data.cropped_face)
    
    def check_full_frame_region(self, frame: np.ndarray, 
                                 bbox: Tuple[int, int, int, int]) -> LivenessResult:
        """
        Run liveness on the face region from the original frame.
        

        """
        x, y, w, h = bbox
        
        # Clamp to frame boundaries
        x = max(0, x)
        y = max(0, y)
        w = min(w, frame.shape[1] - x)
        h = min(h, frame.shape[0] - y)
        
        if w <= 0 or h <= 0:
            return LivenessResult(
                is_live=False,
                variance=0.0,
                threshold=self.threshold,
                reason="Invalid bounding box dimensions"
            )
        
        face_region = frame[y:y+h, x:x+w]
        return self.check(face_region)
    
    def get_stats(self) -> dict:
        """
        Return running statistics for threshold calibration.

        """
        return {
            "threshold": self.threshold,
            "samples_seen": self._samples_seen,
            "running_mean_variance": round(self._running_mean, 2),
            "preset": self._get_current_preset_name()
        }
    
    def set_preset(self, preset_name: str) -> None:
        """
        Apply a pre-configured threshold preset.

        """
        if preset_name not in self.PRESETS:
            raise ValueError(
                f"Unknown preset '{preset_name}'. "
                f"Available: {list(self.PRESETS.keys())}"
            )
        
        self.threshold = self.PRESETS[preset_name]
        print(f"[LivenessGate] Preset '{preset_name}' applied: "
              f"threshold={self.threshold:.1f}")
    
    # ──────────────────────────────────────────────────
    # INTERNAL
    # ──────────────────────────────────────────────────
    
    def _compute_laplacian_variance(self, image: np.ndarray) -> float:
        """
        Compute the variance of the Laplacian of an image.
        

        """
        # Convert to grayscale if color
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif len(image.shape) == 2:
            gray = image
        else:
            raise ValueError(f"Unexpected image shape: {image.shape}")
        
        # Ensure sufficient size for meaningful Laplacian
        h, w = gray.shape[:2]
        if h < 20 or w < 20:
            # Image too small — upsample for meaningful measurement
            gray = cv2.resize(gray, (max(w, 40), max(h, 40)),
                            interpolation=cv2.INTER_LINEAR)
        
        # Compute Laplacian (64-bit float to preserve precision)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        
        # Variance of the Laplacian
        variance = laplacian.var()
        
        return float(variance)
    
    def _get_current_preset_name(self) -> str:
        """Identify which preset (if any) matches current threshold."""
        for name, value in self.PRESETS.items():
            if abs(self.threshold - value) < 0.01:
                return name
        return "custom"
    
    # ──────────────────────────────────────────────────
    # CALIBRATION UTILITY
    # ──────────────────────────────────────────────────
    
    @staticmethod
    def calibrate_threshold(live_samples: list, spoof_samples: list) -> dict:
        """
        Given lists of variance values from known-live and known-spoof
        samples, compute the optimal threshold.
        
        Uses the midpoint between the two distributions' means,
        with a safety margin toward the spoof side.
        
        Args:
            live_samples: List of variance values from real faces
            spoof_samples: List of variance values from photos/screens
            
        Returns:
            Dict with recommended_threshold, live_stats, spoof_stats
        """
        if not live_samples or not spoof_samples:
            return {"error": "Both live and spoof samples are required"}
        
        live_mean = np.mean(live_samples)
        live_std = np.std(live_samples)
        spoof_mean = np.mean(spoof_samples)
        spoof_std = np.std(spoof_samples)
        
        # Set threshold at spoof_mean + 2*spoof_std
        # This puts it above ~95% of spoof samples
        # If live_mean is higher, use midpoint instead
        recommended = max(
            spoof_mean + 2.0 * spoof_std,
            (live_mean + spoof_mean) / 2.0
        )
        
        return {
            "recommended_threshold": round(recommended, 1),
            "live": {
                "mean": round(float(live_mean), 2),
                "std": round(float(live_std), 2),
                "min": round(float(np.min(live_samples)), 2),
                "max": round(float(np.max(live_samples)), 2),
                "count": len(live_samples)
            },
            "spoof": {
                "mean": round(float(spoof_mean), 2),
                "std": round(float(spoof_std), 2),
                "min": round(float(np.min(spoof_samples)), 2),
                "max": round(float(np.max(spoof_samples)), 2),
                "count": len(spoof_samples)
            },
            "separation": round(float(live_mean - spoof_mean), 2)
        }