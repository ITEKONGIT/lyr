import cv2
import threading
import time
import numpy as np
from typing import Optional, Tuple
import sys
import os

# Add parent directory to path so we can import contracts
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from contracts import FrameData


class CameraModule:
    """
    Asymmetric buffer camera.
    
    Key behavior:
    - Internal _current_frame holds exactly ONE frame (no queue)
    - Old frames are dropped silently when new ones arrive
    - get_latest_frame() always returns the freshest frame
    - Memory footprint stays constant regardless of runtime
    """
    
    def __init__(self, 
                 camera_id: int = 0,
                 native_resolution: Tuple[int, int] = (1280, 720),
                 process_resolution: Tuple[int, int] = (640, 480)):
        """
        Args:
            camera_id: Which /dev/videoX or USB camera index
            native_resolution: What the camera captures at
            process_resolution: What we downscale to for downstream processing
        """
        self.camera_id = camera_id
        self.native_resolution = native_resolution
        self.process_resolution = process_resolution
        
        self._cap: Optional[cv2.VideoCapture] = None
        self._is_running = False
        self._lock = threading.Lock()
        self._current_frame: Optional[np.ndarray] = None
        self._capture_thread: Optional[threading.Thread] = None
        
        # Stats
        self._frame_count = 0
        self._start_time: Optional[float] = None
        self._dropped_frames = 0
    
    # ──────────────────────────────────────────────────
    # PUBLIC API
    # ──────────────────────────────────────────────────
    
    def start(self) -> bool:
        """
        Initialize camera hardware and begin continuous capture.
        
        Returns:
            True if camera started successfully
        """
        if self._is_running:
            print("[CameraModule] Already running")
            return True
        
        self._cap = cv2.VideoCapture(self.camera_id)
        
        # Set native resolution
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.native_resolution[0])
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.native_resolution[1])
        self._cap.set(cv2.CAP_PROP_FPS, 30)
        # CRITICAL: Minimize internal OpenCV buffer to 1 frame
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not self._cap.isOpened():
            print(f"[CameraModule] ERROR: Cannot open camera {self.camera_id}")
            return False
        
        actual_w = self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h = self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"[CameraModule] Camera {self.camera_id} opened: {actual_w}x{actual_h}")
        
        self._is_running = True
        self._start_time = time.time()
        self._frame_count = 0
        
        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            daemon=True,
            name="CameraCapture"
        )
        self._capture_thread.start()
        
        print(f"[CameraModule] Capture thread started. "
              f"Process resolution: {self.process_resolution}")
        return True
    
    def get_latest_frame(self) -> Optional[FrameData]:
        """
        Get the most recent frame as a FrameData object.
        Non-blocking — returns immediately.
        
        Returns:
            FrameData if a frame is available, None otherwise
        """
        with self._lock:
            if self._current_frame is None:
                return None
            
            # Copy so the caller owns their data
            pixel_copy = self._current_frame.copy()
            frame_num = self._frame_count
            timestamp = time.time()
            resolution = self.process_resolution
        
        return FrameData(
            pixel_array=pixel_copy,
            frame_number=frame_num,
            timestamp=timestamp,
            resolution=resolution,
            capture_latency_ms=0.0  # We could measure this more precisely
        )
    
    def stop(self) -> None:
        """
        Graceful shutdown. Releases camera hardware.
        Safe to call multiple times.
        """
        self._is_running = False
        
        if self._capture_thread is not None:
            self._capture_thread.join(timeout=3.0)
            if self._capture_thread.is_alive():
                print("[CameraModule] WARNING: Capture thread did not exit cleanly")
        
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        
        print(f"[CameraModule] Stopped. Total frames: {self._frame_count}")
    
    def get_stats(self) -> dict:
        """
        Runtime statistics for monitoring.
        
        Returns:
            Dictionary with frame count, FPS, uptime, buffer info
        """
        elapsed = time.time() - self._start_time if self._start_time else 0
        fps = self._frame_count / elapsed if elapsed > 0 else 0
        
        return {
            "camera_id": self.camera_id,
            "is_running": self._is_running,
            "frames_captured": self._frame_count,
            "runtime_seconds": round(elapsed, 1),
            "fps": round(fps, 2),
            "buffer_depth": 1,  # Always 1 — this is the asymmetric buffer
            "process_resolution": f"{self.process_resolution[0]}x{self.process_resolution[1]}",
            "native_resolution": f"{self.native_resolution[0]}x{self.native_resolution[1]}"
        }
    
    # ──────────────────────────────────────────────────
    # INTERNAL
    # ──────────────────────────────────────────────────
    
    def _capture_loop(self) -> None:
        """
        Continuous capture loop running in daemon thread.
        Reads from camera, downscales, and does atomic swap
        of the single-frame buffer.
        """
        while self._is_running:
            ret, frame = self._cap.read()
            
            if not ret:
                print("[CameraModule] WARNING: Frame read failed")
                time.sleep(0.001)
                continue
            
            # Downscale to process resolution
            if (frame.shape[1], frame.shape[0]) != self.process_resolution:
                frame = cv2.resize(
                    frame,
                    self.process_resolution,
                    interpolation=cv2.INTER_AREA  # Good for downscaling
                )
            
            # Atomic swap: old frame is garbage collected, new frame takes its place
            with self._lock:
                old_frame = self._current_frame
                self._current_frame = frame
                self._frame_count += 1
                if old_frame is not None:
                    self._dropped_frames += 1
            
            # Small sleep prevents 100% CPU usage
            time.sleep(0.001)