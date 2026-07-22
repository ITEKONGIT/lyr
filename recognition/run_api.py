"""
Standalone launcher for the Face Registration System.

Usage:
    python run_api.py
    python run_api.py --port 9000
"""

import sys
import os
import uvicorn

# Add recognition directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from api import create_app
from config import Config


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Lyr - Face Registration System")
    parser.add_argument("--host", default=Config.HOST)
    parser.add_argument("--port", type=int, default=Config.PORT)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    # Log configuration
    print(f"[Lyr] Environment: {Config.ENVIRONMENT}")
    print(f"[Lyr] API Key configured: {'Yes' if Config.API_KEY else 'No'}")
    print(f"[Lyr] Starting server on {args.host}:{args.port}")
    print(f"[Lyr] Use X-API-Key header for authentication")

    app = create_app()
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload
    )


if __name__ == "__main__":
    main()