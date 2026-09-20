#!/usr/bin/env python3
import argparse

import uvicorn

from terminal_web.api.app import create_app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the terminal inspection API")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    uvicorn.run(create_app(), host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
