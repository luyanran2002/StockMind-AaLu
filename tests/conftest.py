"""Ensure the repo root is importable even when pytest is run without install."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

