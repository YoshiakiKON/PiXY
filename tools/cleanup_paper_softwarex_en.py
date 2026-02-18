"""Cleanup helper for paper_softwarex.md.

This repository occasionally ends up with duplicated manuscript blocks appended
after the References section (e.g., an extra YAML front matter starting with
"---\ntitle:"). Some editors/tools may also display only the first block.

This script truncates paper_softwarex.md to the first block by removing anything
from the *second* YAML front matter start ("\n---\ntitle:") onward.

Usage (from repo root):
    py tools/cleanup_paper_softwarex_en.py
"""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "paper_softwarex.md"


def main() -> int:
    text = PATH.read_text(encoding="utf-8")

    marker = "\n---\ntitle:"
    cut = text.find(marker, 1)
    if cut == -1:
        print("OK: no duplicated YAML block detected")
        return 0

    cleaned = text[:cut].rstrip() + "\n"
    PATH.write_text(cleaned, encoding="utf-8", newline="\n")
    print(f"CLEANED: truncated at byte index {cut}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
