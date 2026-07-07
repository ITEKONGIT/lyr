"""
Phase 6: Unified Face Recognition API

Exports:
    get_router()        — Returns APIRouter for mounting in external FastAPI apps
    create_app()        — Creates a standalone FastAPI app with full lifecycle
    get_controller()    — Returns the singleton FaceRegController
    get_broadcaster()   — Returns the WebSocket broadcaster
    start_controller()  — Starts the controller (sync part)
    start_broadcaster() — Starts the broadcaster (async, after event loop ready)
    stop_broadcaster()  — Stops the broadcaster (async, before event loop closes)
    stop_controller()   — Stops the controller (sync part)
    is_controller_running() — Check if controller is active
"""

from .router import router
from .dependencies import (
    get_controller,
    get_broadcaster,
    start_controller,
    start_broadcaster,
    stop_broadcaster,
    stop_controller,
    is_controller_running,
)

__all__ = [
    "router",
    "get_controller",
    "get_broadcaster",
    "start_controller",
    "start_broadcaster",
    "stop_broadcaster",
    "stop_controller",
    "is_controller_running",
    "create_app",
]


def get_router():
    """
    Returns the APIRouter for mounting in external FastAPI applications.
    
    Usage from a parent project:
    
        from fastapi import FastAPI
        from recognition.api import (
            get_router, start_controller, start_broadcaster,
            stop_broadcaster, stop_controller,
        )
        
        app = FastAPI()
        
        @app.on_event("startup")
        async def startup():
            start_controller()       # Sync: camera, watcher, DB
            await start_broadcaster() # Async: WebSocket broadcast loop
        
        @app.on_event("shutdown")
        async def shutdown():
            await stop_broadcaster()  # Async: stop broadcast loop first
            stop_controller()         # Sync: stop threads, release hardware
        
        app.include_router(get_router())
    """
    return router


def create_app() -> "FastAPI":
    """
    Creates a standalone FastAPI application with full lifecycle management.
    
    Starts the controller, watcher, and WebSocket broadcaster on startup.
    Stops everything gracefully on shutdown.
    
    Usage:
        app = create_app()
        uvicorn.run(app, host="0.0.0.0", port=8000)
    """
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from contextlib import asynccontextmanager
    
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # ── Startup ──
        print("\n" + "=" * 60)
        print("  Face Recognition System — Starting Up")
        print("=" * 60)
        
        # Step 1: Start controller (sync — camera, watcher, DB threads)
        if not start_controller():
            raise RuntimeError("Controller failed to start")
        
        # Step 2: Start broadcaster (async — WebSocket broadcast loop)
        await start_broadcaster()
        
        print("\n[API] All systems ready")
        print("[API] REST API at http://0.0.0.0:8000")
        print("[API] WebSocket at ws://0.0.0.0:8000/api/v1/stream/detections")
        print("[API] Docs at http://0.0.0.0:8000/docs")
        
        yield  # ── Application runs here ──
        
        # ── Shutdown ──
        print("\n" + "=" * 60)
        print("  Face Recognition System — Shutting Down")
        print("=" * 60)
        
        # Step 1: Stop broadcaster (async — close WebSocket connections)
        await stop_broadcaster()
        
        # Step 2: Stop controller (sync — stop threads, release camera)
        stop_controller()
        
        print("[API] Goodbye.")
    
    app = FastAPI(
        title="Face Recognition System",
        description=(
            "Local facial recognition API with continuous identification.\n\n"
            "**Features:**\n"
            "- Face registration with multi-frame centroid averaging\n"
            "- Real-time identification via WebSocket stream\n"
            "- Liveness detection (anti-spoofing)\n"
            "- Automatic identity adaptation (EMA learning)\n"
            "- Compute-aware dynamic face capping\n"
        ),
        version="2.0.0",
        lifespan=lifespan,
    )
    
    # CORS — allow local development and dashboard access
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Register routes
    app.include_router(router)
    
    # Root endpoint
    @app.get("/")
    async def root():
        return {
            "service": "Face Recognition System",
            "version": "2.0.0",
            "docs": "/docs",
            "endpoints": {
                "register": "POST /api/v1/register",
                "identify": "POST /api/v1/identify",
                "identities": "GET /api/v1/identities",
                "delete_identity": "DELETE /api/v1/identities/{id}",
                "health": "GET /api/v1/health",
                "debug_liveness": "GET /api/v1/debug/liveness",
                "stream_detections": "WS /api/v1/stream/detections",
            },
        }
    
    return app