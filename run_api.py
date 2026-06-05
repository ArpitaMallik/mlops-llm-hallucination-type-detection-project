"""Run the FastAPI hallucination API with uvicorn.

Usage (from the project root):

    python run_api.py                # default: 127.0.0.1:8000
    python run_api.py --port 9000    # custom port
    python run_api.py --reload       # auto-reload on file changes
"""
from __future__ import annotations

import argparse

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the hallucination detection API.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()

    uvicorn.run(
        "src.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
