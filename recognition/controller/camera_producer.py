"""
Camera Producer Thread

Continuously captures frames and pushes them into a queue.
Uses asymmetric buffer (maxsize=1) so old frames are dropped
when the consumer is busy.
"""

import queue
import threading
import time
from typing import Optional

from camera.camera_module import CameraModule
from contracts import FrameData


class CameraProducer:
    """
    Single-responsibility: pump frames from camera into a queue.
    
    The queue has maxsize=1. If the consumer hasn't picked up the
    previous frame yet, the new frame replaces it silently.
    """
    
    def __init__(self, camera: CameraModule, frame_queue: queue.Queue):
        self.camera = camera
        self.frame_queue = frame_queue
        self._thread: Optional[threading.Thread] = None
        self._is_running = False
    
    def start(self):
        """Start the capture thread."""
        if self._is_running:
            return
        
        self._is_running = True
        self._thread = threading.Thread(
            target=self._loop,
            daemon=True,
            name="CameraProducer"
        )
        self._thread.start()
        print("[CameraProducer] Started")
    
    def stop(self):
        """Signal the thread to stop."""
        self._is_running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
            if self._thread.is_alive():
                print("[CameraProducer] WARNING: did not exit cleanly")
        print("[CameraProducer] Stopped")
    
    def _loop(self):
        """Main capture loop."""
        while self._is_running:
            frame_data = self.camera.get_latest_frame()
            
            if frame_data is not None and frame_data.verify_integrity():
                try:
                    self.frame_queue.put_nowait(frame_data)
                except queue.Full:
                    # Consumer is busy — drop old frame silently
                    pass
            
            time.sleep(0.001)