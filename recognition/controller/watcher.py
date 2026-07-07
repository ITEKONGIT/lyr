"""
Phase 6: The Watcher — Continuous Face Identification Engine

Architecture:
    Camera Producer → frame_queue (maxsize=1)
                           │
                           ▼
    Watcher Loop (10 FPS, daemon thread)
        │
        ├─► Pull latest frame (non-blocking, Solution #1)
        ├─► Detect faces → Cap to N (Solution #9)
        ├─► Per face:
        │     ├─► Liveness check (Solution #4, Fail-Fast)
        │     ├─► Extract embedding
        │     ├─► Query DB (nearest neighbor, n_results=2)
        │     ├─► Decision matrix:
        │     │     ├─► distance < 0.40  → "Known: {name}"
        │     │     └─► distance >= 0.40 → "Unknown"
        │     └─► EMA update if similarity > 0.95 (Solutions #5 + #6)
        │
        └─► Broadcast DetectionEvent to WebSocket clients (Solution #8)

Solutions implemented:
    #1  Producer-Consumer (frame_queue)
    #4  Fail-Fast Liveness Gate
    #5  Continuous Learning (EMA)
    #6  Rate-Limited EMA (cooldown guard)
    #8  Threading Bridge (sync → async via Broadcaster)
    #9  Compute-Aware Face Capping
"""

import threading
import time
import queue
import gc
from datetime import datetime, timezone
from typing import Optional, List

from face.face_module import FaceModule
from face.liveness import LivenessGate
from embedding.embedding_module import EmbeddingModule
from database.database_module import DatabaseModule
from contracts import (
    FrameData, FaceData, EmbeddingData,
    IdentificationResult, DetectionEvent
)
from controller.face_capper import FaceCapper
from controller.ema_guard import EmaGuard
from controller.broadcaster import Broadcaster


class Watcher:
    """
    Continuous face identification engine.
    
    Runs at a target FPS, processing every available frame through
    the full pipeline: detect → liveness → embed → identify → broadcast.
    
    The watcher NEVER blocks the camera. It pulls from the asymmetric
    frame queue and drops frames it can't process in time.
    """
    
    TARGET_FPS = 10
    CYCLE_TIME = 1.0 / TARGET_FPS
    KNOWN_THRESHOLD = 0.40
    EMA_SIMILARITY_THRESHOLD = 0.95
    
    def __init__(
        self,
        face_detector: FaceModule,
        liveness_gate: LivenessGate,
        embedder: EmbeddingModule,
        database: DatabaseModule,
        frame_queue: queue.Queue,
        broadcaster: Broadcaster,
        face_capper: FaceCapper,
        ema_guard: EmaGuard,
        db_queue: queue.Queue,
    ):
        self.face_detector = face_detector
        self.liveness_gate = liveness_gate
        self.embedder = embedder
        self.database = database
        self.frame_queue = frame_queue
        self.broadcaster = broadcaster
        self.face_capper = face_capper
        self.ema_guard = ema_guard
        self.db_queue = db_queue
        
        self._thread: Optional[threading.Thread] = None
        self._is_running = False
        
        self._current_fps = 0.0
        self._fps_samples = []
        self._cycles_completed = 0
        self._faces_processed = 0
        self._identifications_made = 0
        self._spoofs_blocked = 0
        self._ema_updates_queued = 0
        self._frames_dropped = 0
        
        self._empty_frame_streak = 0
    
    # ──────────────────────────────────────────────────
    # AUTHORIZATION STATE MATRIX
    # ──────────────────────────────────────────────────
    
    @staticmethod
    def _classify_state(
        is_live: bool,
        is_known: bool,
        confidence: float,
        has_error: bool,
        db_empty: bool,
        is_empty_frame: bool,
    ) -> tuple:
        """
        10-state authorization matrix.

        Returns:
            (state_label, authorization, state_number, state_reason)
        """
        # State 1: Empty frame
        if is_empty_frame:
            return ("empty_frame", "idle", 1,
                    "No face in frame — system idle")
        
        # State 8: Pipeline error
        if has_error:
            return ("pipeline_error", "error", 8,
                    "Face detection or embedding pipeline failed")
        
        # State 2: Spoof detected
        if not is_live:
            return ("spoof_detected", "alert", 2,
                    "Spoof attempt detected — photo or screen presented to camera")
        
        # State 10: DB empty — no identities enrolled
        if db_empty:
            return ("unenrolled_system", "unauthorized", 10,
                    "No identities enrolled in database")
        
        # State 3: Live but not known
        if not is_known:
            return ("unknown_intrusion", "unauthorized", 3,
                    "Unknown person detected — unauthorized access")
        
        # State 4: Known with high confidence
        if confidence > 0.70:
            return ("known_high_confidence", "authorized", 4,
                    "Known identity — access granted")
        
        # State 5: Known with moderate confidence
        if confidence >= 0.50:
            return ("known_low_confidence", "authorized", 5,
                    "Known identity — access granted (low confidence match)")
        
        # State 6: Known but very low confidence — possible impersonation
        return ("uncertain_match", "unauthorized", 6,
                "Uncertain match — possible impersonation attempt")
    
    def start(self):
        if self._is_running:
            print("[Watcher] Already running")
            return
        
        self._is_running = True
        self._thread = threading.Thread(
            target=self._loop,
            daemon=True,
            name="Watcher"
        )
        self._thread.start()
        print(f"[Watcher] Started — target {self.TARGET_FPS} FPS "
              f"(cycle={self.CYCLE_TIME*1000:.0f}ms)")
    
    def stop(self):
        self._is_running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
            if self._thread.is_alive():
                print("[Watcher] WARNING: did not exit cleanly")
        
        print(f"[Watcher] Stopped — {self._cycles_completed} cycles, "
              f"{self._identifications_made} identifications, "
              f"{self._spoofs_blocked} spoofs blocked, "
              f"{self._ema_updates_queued} EMA updates")
    
    def _loop(self):
        while self._is_running:
            cycle_start = time.monotonic()
            
            try:
                frame = self._get_latest_frame()
                if frame is None:
                    self._empty_frame_streak += 1
                    self._broadcast_empty_frame()
                    self._sleep_until_next_cycle(cycle_start)
                    continue
                
                self._empty_frame_streak = 0
                
                face_list = self.face_detector.detect_faces(frame)
                face_list = self.face_capper.cap(face_list)
                
                detections = []
                for face_data in face_list:
                    result = self._process_single_face(frame, face_data)
                    detections.append(result)
                
                self._faces_processed += len(face_list)
                
                event = self._build_event(frame, detections)
                self._broadcast(event)
                
                self._cycles_completed += 1
                self._update_fps(cycle_start)
                self.face_capper.record_cycle(
                    (time.monotonic() - cycle_start) * 1000
                )
                
                if self._cycles_completed % 100 == 0:
                    gc.collect()
                
            except Exception as e:
                print(f"[Watcher] Cycle error: {e}")
            
            self._sleep_until_next_cycle(cycle_start)
    
    def _get_latest_frame(self) -> Optional[FrameData]:
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            self._frames_dropped += 1
            return None
    
    def _process_single_face(
        self, frame: FrameData, face_data: FaceData
    ) -> IdentificationResult:
        bbox = face_data.bbox
        processing_start = time.monotonic()
        
        liveness = self.liveness_gate.check_full_frame_region(
            frame.pixel_array, bbox
        )
        
        db_empty = self.database.count() == 0
        
        if not liveness.is_live:
            self._spoofs_blocked += 1
            state, auth, state_num, reason = self._classify_state(
                is_live=False, is_known=False, confidence=0.0,
                has_error=False, db_empty=db_empty, is_empty_frame=False
            )
            return IdentificationResult(
                identity="Spoof",
                user_id="",
                name="",
                confidence=0.0,
                distance=1.0,
                is_known=False,
                is_live=False,
                liveness_variance=round(liveness.variance, 2),
                bbox=bbox,
                processing_time_ms=round(
                    (time.monotonic() - processing_start) * 1000, 2
                ),
                authorization=auth,
                state=state,
                state_reason=reason,
            )
        
        embedding = self.embedder.extract(face_data)
        if embedding is None or not embedding.verify_normalization():
            state, auth, state_num, reason = self._classify_state(
                is_live=True, is_known=False, confidence=0.0,
                has_error=True, db_empty=db_empty, is_empty_frame=False
            )
            return IdentificationResult(
                identity="Error",
                user_id="",
                name="",
                confidence=0.0,
                distance=1.0,
                is_known=False,
                is_live=True,
                liveness_variance=round(liveness.variance, 2),
                bbox=bbox,
                processing_time_ms=round(
                    (time.monotonic() - processing_start) * 1000, 2
                ),
                authorization=auth,
                state=state,
                state_reason=reason,
            )
        
        identity, user_id, name, confidence, distance, is_known = (
            self._identify_embedding(embedding)
        )
        
        self._identifications_made += 1
        
        if is_known and confidence > self.EMA_SIMILARITY_THRESHOLD:
            if self.ema_guard.try_update(user_id):
                self._queue_ema_update(user_id, embedding)
                self._ema_updates_queued += 1
        
        state, auth, state_num, reason = self._classify_state(
            is_live=True, is_known=is_known, confidence=confidence,
            has_error=False, db_empty=db_empty, is_empty_frame=False
        )
        
        return IdentificationResult(
            identity=identity,
            user_id=user_id,
            name=name,
            confidence=round(confidence, 4),
            distance=round(distance, 4),
            is_known=is_known,
            is_live=True,
            liveness_variance=round(liveness.variance, 2),
            bbox=bbox,
            processing_time_ms=round(
                (time.monotonic() - processing_start) * 1000, 2
            ),
            authorization=auth,
            state=state,
            state_reason=reason,
        )
    
    def _identify_embedding(self, embedding: EmbeddingData) -> tuple:
        if self.database.count() == 0:
            return (
                "Unknown (no identities enrolled)",
                "",
                "",
                0.0,
                1.0,
                False,
            )
        
        matches = self.database.query_nearest(embedding, n_results=2)
        
        if not matches:
            return ("Unknown", "", "", 0.0, 1.0, False)
        
        top = matches[0]
        distance = top["distance"]
        similarity = top["similarity"]
        
        if distance < self.KNOWN_THRESHOLD:
            identity = f"Known: {top['name']}"
            user_id = top["user_id"]
            name = top["name"]
            is_known = True
        else:
            identity = "Unknown"
            user_id = ""
            name = ""
            is_known = False
        
        return (identity, user_id, name, similarity, distance, is_known)
    
    def _queue_ema_update(self, doc_id: str, embedding: EmbeddingData):
        task = {
            "action": "ema_update",
            "doc_id": doc_id,
            "embedding": embedding,
            "alpha": 0.1,
        }
        try:
            self.db_queue.put_nowait(task)
        except queue.Full:
            pass
    
    def _build_event(
        self, frame: FrameData, detections: List[IdentificationResult]
    ) -> DetectionEvent:
        return DetectionEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            system_state="watching",
            frame_number=frame.frame_number,
            detections=[self._result_to_dict(d) for d in detections],
            watcher_fps=round(self._current_fps, 1),
        )
    
    def _result_to_dict(self, result: IdentificationResult) -> dict:
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
    
    def _broadcast(self, event: DetectionEvent):
        try:
            self.broadcaster.push_sync({
                "timestamp": event.timestamp,
                "system_state": event.system_state,
                "frame_number": event.frame_number,
                "detections": event.detections,
                "watcher_fps": event.watcher_fps,
            })
        except Exception as e:
            print(f"[Watcher] Broadcast error: {e}")
    
    def _broadcast_empty_frame(self):
        if self._empty_frame_streak % 10 != 0:
            return
        
        try:
            self.broadcaster.push_sync({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "system_state": "watching",
                "frame_number": 0,
                "detections": [{
                    "identity": "",
                    "user_id": "",
                    "name": "",
                    "confidence": 0.0,
                    "distance": 1.0,
                    "is_known": False,
                    "is_live": False,
                    "liveness_variance": 0.0,
                    "bbox": [0, 0, 0, 0],
                    "processing_time_ms": 0.0,
                    "authorization": "idle",
                    "state": "empty_frame",
                    "state_reason": "No face in frame — system idle",
                }],
                "watcher_fps": round(self._current_fps, 1),
            })
        except Exception:
            pass
    
    def _sleep_until_next_cycle(self, cycle_start: float):
        elapsed = time.monotonic() - cycle_start
        sleep_time = self.CYCLE_TIME - elapsed
        
        if sleep_time > 0:
            time.sleep(sleep_time)
    
    def _update_fps(self, cycle_start: float):
        elapsed = time.monotonic() - cycle_start
        if elapsed > 0:
            instantaneous_fps = 1.0 / elapsed
            self._fps_samples.append(instantaneous_fps)
            
            if len(self._fps_samples) > 30:
                self._fps_samples.pop(0)
            
            self._current_fps = (
                sum(self._fps_samples) / len(self._fps_samples)
                if self._fps_samples else 0.0
            )
    
    def get_stats(self) -> dict:
        return {
            "is_running": self._is_running,
            "target_fps": self.TARGET_FPS,
            "current_fps": round(self._current_fps, 1),
            "cycles_completed": self._cycles_completed,
            "faces_processed": self._faces_processed,
            "identifications_made": self._identifications_made,
            "spoofs_blocked": self._spoofs_blocked,
            "ema_updates_queued": self._ema_updates_queued,
            "frames_dropped": self._frames_dropped,
        }