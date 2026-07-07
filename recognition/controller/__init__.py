"""
Face Recognition Controller Package

Exports:
    FaceRegController   — main orchestrator
    RegistrationHandler — synchronous multi-frame registration
    CameraProducer      — frame capture thread
    DatabaseWorker      — sequential DB writer
    Watcher             — continuous identification engine
    Broadcaster         — WebSocket broadcast manager
    FaceCapper          — compute-aware dynamic face capping
    EmaGuard            — rate-limited EMA cooldown tracker
"""

from .orchestrator import FaceRegController
from .registration_handler import RegistrationHandler
from .camera_producer import CameraProducer
from .db_worker import DatabaseWorker
from .watcher import Watcher
from .broadcaster import Broadcaster
from .face_capper import FaceCapper
from .ema_guard import EmaGuard

__all__ = [
    "FaceRegController",
    "RegistrationHandler",
    "CameraProducer",
    "DatabaseWorker",
    "Watcher",
    "Broadcaster",
    "FaceCapper",
    "EmaGuard",
]