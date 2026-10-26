"""Run the demo site locally: ``uv run python -m giant_sp.demo``.

Reads data from Arraylake using the login cached by ``arraylake auth login``
(or ``ARRAYLAKE_TOKEN``); see ``ArraylakeSource.from_env`` for the other
environment overrides.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import uvicorn

from giant_sp.demo.dashboard import Dashboard
from giant_sp.demo.server import DEFAULT_SITE_DIR, create_app
from giant_sp.demo.source import ArraylakeSource


def main() -> None:
    """Parse CLI args and serve the demo with uvicorn."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--site-dir", type=Path, default=DEFAULT_SITE_DIR)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:     %(message)s")
    app = create_app(Dashboard(ArraylakeSource.from_env()), site_dir=args.site_dir)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
