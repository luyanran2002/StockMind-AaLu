"""Quick NVDA demo (kept for convenience / Phase 1 parity).

Run:

    python examples/nvda_research.py --lang en

For multiple tickers, use ``examples/research.py`` instead.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from research import run_ticker


def main() -> None:
    parser = argparse.ArgumentParser(description="NVDA quick demo")
    parser.add_argument("--lang", choices=["en", "zh"], default=os.getenv("STOCKMIND_LANGUAGE", "en"))
    args = parser.parse_args()
    run_ticker("NVDA", args.lang)


if __name__ == "__main__":
    main()

