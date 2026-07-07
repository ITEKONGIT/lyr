"""
Standalone launcher for the Face Registration System.

Usage:
    python run.py
    python run.py --port 9000
"""

import sys
import os
import uvicorn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from api import create_app

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Face Registration System")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    
    app = create_app()
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)