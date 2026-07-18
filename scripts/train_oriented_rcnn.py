#!/usr/bin/env python3
"""Convenience entry point for Oriented R-CNN training."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.models.oriented_rcnn.train import main


if __name__ == "__main__":
    main()

