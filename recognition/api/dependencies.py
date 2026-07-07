"""
Controller singleton and lifecycle management.

The controller is a module-level singleton. It is started once
and shared across all route handlers.

"""

import sys
import os
import threading
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from controller import FaceRegController

_controller: Optional[FaceRegController] = None
_lock = threading.Lock()


def get_controller() -> Optional[FaceRegController]:
    """
    Returns the singleton controller instance.

    """
    global _controller
    with _lock:
        return _controller


def get_broadcaster():
    """
    Returns the broadcaster instance for WebSocket route handlers.
    
    Returns None if controller or broadcaster hasn't been started.
    """
    ctrl = get_controller()
    if ctrl is None:
        return None
    return ctrl.get_broadcaster()


def start_controller() -> bool:
    """
    Initialize and start the FaceRegController (synchronous part).

    """
    global _controller
    
    with _lock:
        if _controller is not None and _controller._is_running:
            print("[Dependencies] Controller already running")
            return True
        
        print("[Dependencies] Starting controller...")
        _controller = FaceRegController()
        
        if not _controller.start():
            print("[Dependencies] Controller start failed")
            _controller = None
            return False
        
        print("[Dependencies] Controller started successfully "
              "(broadcaster pending)")
        return True


async def start_broadcaster():
    """
    Start the broadcaster's asyncio background task.

    """
    ctrl = get_controller()
    if ctrl is None:
        print("[Dependencies] WARNING: Cannot start broadcaster — "
              "controller not running")
        return
    
    await ctrl.start_broadcaster()


async def stop_broadcaster():
    """
    Stop the broadcaster's asyncio background task.

    """
    ctrl = get_controller()
    if ctrl:
        await ctrl.stop_broadcaster()


def stop_controller() -> None:
    """
    Gracefully stop the controller and release hardware.

    """
    global _controller
    
    with _lock:
        if _controller is not None:
            print("[Dependencies] Stopping controller...")
            _controller.stop()
            _controller = None
            print("[Dependencies] Controller stopped")


def is_controller_running() -> bool:
    """Check if the controller is active."""
    ctrl = get_controller()
    return ctrl is not None and ctrl._is_running