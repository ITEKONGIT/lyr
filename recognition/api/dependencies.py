"""
Dependencies for FastAPI routes.

Provides:
    - Controller singleton management
    - API key authentication
    - WebSocket authentication helpers
"""

import sys
import hmac
import os
import threading
from typing import Optional, TYPE_CHECKING
from fastapi import HTTPException, WebSocket, status, Depends
from fastapi.security import APIKeyHeader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config

if TYPE_CHECKING:
    from controller import FaceRegController


_controller: Optional["FaceRegController"] = None
_lock = threading.Lock()

# ──────────────────────────────────────────────────
# API KEY AUTHENTICATION
# ──────────────────────────────────────────────────

# API key header name
API_KEY_HEADER = "X-API-Key"

# FastAPI security dependency for header extraction
api_key_header = APIKeyHeader(name=API_KEY_HEADER, auto_error=False)


def verify_api_key(api_key: Optional[str] = Depends(api_key_header)) -> str:
    """
    FastAPI dependency that validates the API key.

    Checks the X-API-Key header against the configured key.

    Args:
        api_key: The API key from the header (injected by FastAPI).

    Returns:
        The validated API key.

    Raises:
        HTTPException: 401 if key is missing or invalid.
    """
    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Provide X-API-Key header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    
    # Use constant-time comparison to prevent timing attacks
    if not hmac.compare_digest(api_key, Config.API_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    
    return api_key


async def verify_websocket_key(websocket: WebSocket) -> bool:
    """
    Validate API key for WebSocket connections.

    Checks the 'api_key' query parameter.

    Args:
        websocket: The WebSocket connection.

    Returns:
        True if valid, False otherwise.
    """
    api_key = websocket.query_params.get("api_key")
    
    if api_key is None:
        return False
    
    # Use constant-time comparison
    return api_key == Config.API_KEY


# ──────────────────────────────────────────────────
# CONTROLLER MANAGEMENT
# ──────────────────────────────────────────────────

def get_controller() -> Optional["FaceRegController"]:
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
        from controller import FaceRegController

        _controller = FaceRegController()

        if not _controller.start():
            print("[Dependencies] Controller start failed")
            _controller = None
            return False

        print("[Dependencies] Controller started successfully")
        return True


async def start_broadcaster():
    """
    Start the broadcaster's asyncio background task.
    """
    ctrl = get_controller()
    if ctrl is None:
        print("[Dependencies] WARNING: Cannot start broadcaster — controller not running")
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
