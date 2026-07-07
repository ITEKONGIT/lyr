"""
Face Registration Orchestrator — Phase 7 Enhanced

"""

import queue
import threading
import time
import asyncio
from typing import Optional, Dict

from camera.camera_module import CameraModule
from face.face_module import FaceModule
from face.liveness import LivenessGate
from embedding.embedding_module import EmbeddingModule
from database.database_module import DatabaseModule
from contracts import RegistrationResult

from .camera_producer import CameraProducer
from .registration_handler import RegistrationHandler
from .db_worker import DatabaseWorker
from .face_capper import FaceCapper
from .ema_guard import EmaGuard
from .broadcaster import Broadcaster
from .watcher import Watcher


class FaceRegController:
    """
    Central orchestrator for the face registration and identification system.

    """
    
    DEFAULT_LIVENESS_THRESHOLD = 40.0
    
    def __init__(self, liveness_threshold: float = None):
        self._liveness_threshold = liveness_threshold or self.DEFAULT_LIVENESS_THRESHOLD
        
        # Modules (loaded in start())
        self.camera: Optional[CameraModule] = None
        self.face_detector: Optional[FaceModule] = None
        self.liveness_gate: Optional[LivenessGate] = None
        self.embedder: Optional[EmbeddingModule] = None
        self.database: Optional[DatabaseModule] = None
        
        # Queues
        self.frame_queue = queue.Queue(maxsize=1)   # Camera → Watcher
        self.db_queue = queue.Queue()                # Watcher → DB Worker
        
        # Phase 5 components
        self._camera_producer: Optional[CameraProducer] = None
        self._registration_handler: Optional[RegistrationHandler] = None
        self._db_worker: Optional[DatabaseWorker] = None
        
        # Phase 6 components
        self._face_capper: Optional[FaceCapper] = None
        self._ema_guard: Optional[EmaGuard] = None
        self._broadcaster: Optional[Broadcaster] = None
        self._watcher: Optional[Watcher] = None
        
        # State
        self._is_running = False
        self._start_time: Optional[float] = None
        self._lock = threading.Lock()
    
    # ──────────────────────────────────────────────────
    # LIFECYCLE
    # ──────────────────────────────────────────────────
    
    def start(self) -> bool:
        """Initialize all modules and start background threads."""
        if self._is_running:
            print("[Orchestrator] Already running")
            return True
        
        print("\n" + "=" * 50)
        print("[Orchestrator] Starting Face Registration System")
        print("=" * 50)
        
        try:
            # ── Load modules ──
            print("\n[Orchestrator] Initializing Database...")
            self.database = DatabaseModule()
            
            print("\n[Orchestrator] Loading Face Detector...")
            self.face_detector = FaceModule()
            
            print("\n[Orchestrator] Initializing Liveness Gate...")
            self.liveness_gate = LivenessGate(threshold=self._liveness_threshold)
            
            print("\n[Orchestrator] Loading Embedding Model...")
            self.embedder = EmbeddingModule()
            
            print("\n[Orchestrator] Starting Camera...")
            self.camera = CameraModule()
            if not self.camera.start():
                print("[Orchestrator] FAILED to start camera")
                return False
            
            # ── Phase 6 components ──
            print("\n[Orchestrator] Initializing Face Capper...")
            self._face_capper = FaceCapper(
                cycle_time_ms=100.0,
                max_faces=5
            )
            
            print("\n[Orchestrator] Initializing EMA Guard...")
            self._ema_guard = EmaGuard(cooldown_seconds=21600)
            
            print("\n[Orchestrator] Initializing Broadcaster...")
            self._broadcaster = Broadcaster()
            
            # ── Phase 5 + 7 components ──
            self._camera_producer = CameraProducer(self.camera, self.frame_queue)
            
            # RegistrationHandler now includes Phase 7 quality features
            self._registration_handler = RegistrationHandler(
                self.camera, self.face_detector, self.liveness_gate,
                self.embedder, self.database
            )
            
            self._db_worker = DatabaseWorker(self.database, self.db_queue)
            
            # ── Phase 6: Watcher ──
            print("\n[Orchestrator] Starting Watcher...")
            self._watcher = Watcher(
                face_detector=self.face_detector,
                liveness_gate=self.liveness_gate,
                embedder=self.embedder,
                database=self.database,
                frame_queue=self.frame_queue,
                broadcaster=self._broadcaster,
                face_capper=self._face_capper,
                ema_guard=self._ema_guard,
                db_queue=self.db_queue,
            )
            
            # ── Start threads ──
            self._is_running = True
            self._start_time = time.time()
            
            self._camera_producer.start()
            self._db_worker.start()
            self._watcher.start()
            
            print("\n[Orchestrator] All systems started")
            print("[Orchestrator] Watcher running — connect to "
                  "WS /api/v1/stream/detections")
            print("[Orchestrator] Ready to accept registration requests")
            return True
            
        except Exception as e:
            print(f"[Orchestrator] Start failed: {e}")
            self.stop()
            return False
    
    async def start_broadcaster(self):
        """Start the broadcaster's asyncio background task."""
        if self._broadcaster is None:
            print("[Orchestrator] WARNING: Broadcaster not initialized")
            return
        
        loop = asyncio.get_event_loop()
        self._broadcaster.attach_loop(loop)
        await self._broadcaster.start()
        print("[Orchestrator] Broadcaster started")
    
    async def stop_broadcaster(self):
        """Stop the broadcaster's asyncio task."""
        if self._broadcaster:
            await self._broadcaster.stop()
    
    def stop(self) -> None:
        """Graceful shutdown — stops all threads and releases hardware."""
        print("\n[Orchestrator] Shutting down...")
        self._is_running = False
        
        if self._watcher:
            self._watcher.stop()
        
        for component in [self._db_worker, self._camera_producer]:
            if component:
                component.stop()
        
        if self.camera:
            self.camera.stop()
        
        print(f"[Orchestrator] Stopped.")
    
    # ──────────────────────────────────────────────────
    # PUBLIC API — REGISTRATION
    # ──────────────────────────────────────────────────
    
    def register_face(self, user_id: str, name: str,
                      metadata: Optional[Dict] = None) -> RegistrationResult:
        """
        Register a face synchronously with quality assessment.

        """
        if not self._is_running:
            return RegistrationResult(
                status="failed",
                user_id=user_id,
                name=name,
                error="System not running"
            )
        
        return self._registration_handler.register(user_id, name, metadata)
    
    def re_enroll_face(self, user_id: str) -> RegistrationResult:
        """
        Update an existing identity with fresh face data.

        """
        if not self._is_running:
            return RegistrationResult(
                status="failed",
                user_id=user_id,
                error="System not running"
            )
        
        return self._registration_handler.re_enroll(user_id)
    
    # ──────────────────────────────────────────────────
    # PUBLIC API — IDENTIFICATION
    # ──────────────────────────────────────────────────
    
    def identify_once(self) -> Optional[Dict]:
        """
        Run a single identification cycle on the current frame.

        """
        if not self._is_running or self.camera is None:
            return None
        
        frame = self.camera.get_latest_frame()
        if frame is None:
            return None
        
        face = self.face_detector.detect_largest_face(frame)
        if face is None:
            return {
                "identity": "No face detected",
                "is_known": False,
                "is_live": False,
                "detections": [],
            }
        
        result = self._watcher._process_single_face(frame, face)
        
        return {
            "identity": result.identity,
            "user_id": result.user_id,
            "name": result.name,
            "confidence": result.confidence,
            "distance": result.distance,
            "is_known": result.is_known,
            "is_live": result.is_live,
            "liveness_variance": result.liveness_variance,
            "bbox": list(result.bbox),
            "processing_time_ms": result.processing_time_ms,
            "authorization": result.authorization,
            "state": result.state,
            "state_reason": result.state_reason,
        }
    
    # ──────────────────────────────────────────────────
    # PUBLIC API — UTILITIES
    # ──────────────────────────────────────────────────
    
    def get_broadcaster(self) -> Optional[Broadcaster]:
        """Get the broadcaster instance for WebSocket route handlers."""
        return self._broadcaster
    
    def get_stats(self) -> Dict:
        """Return comprehensive system statistics."""
        uptime = time.time() - self._start_time if self._start_time else 0
        
        stats = {
            "uptime_seconds": round(uptime, 1),
            "is_running": self._is_running,
            "camera": self.camera.get_stats() if self.camera else {},
            "database_count": self.database.count() if self.database else 0,
            "liveness": self.liveness_gate.get_stats() if self.liveness_gate else {},
            "registrations_completed": self._db_worker.registrations_completed if self._db_worker else 0,
        }
        
        if self._watcher:
            stats["watcher"] = self._watcher.get_stats()
        if self._face_capper:
            stats["face_capper"] = self._face_capper.get_stats()
        if self._ema_guard:
            stats["ema_guard"] = self._ema_guard.get_stats()
        if self._broadcaster:
            stats["broadcaster"] = self._broadcaster.get_stats()
        
        return stats
    
    def set_liveness_threshold(self, threshold: float) -> None:
        """Update liveness threshold at runtime."""
        if threshold <= 0:
            raise ValueError(f"Threshold must be positive, got {threshold}")
        self._liveness_threshold = threshold
        if self.liveness_gate:
            self.liveness_gate.threshold = threshold
            print(f"[Orchestrator] Liveness threshold updated to {threshold:.1f}")