"""Write tools/strict_powf/constants.py's block into the header, between
its CONSTANTS markers:

    python tools/strict_powf/write_header.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import constants  # noqa: E402

HEADER = Path(__file__).resolve().parents[2] / "woof" / "backend" / "translate" / "strict_powf.cuh"
BEGIN = "// CONSTANTS BEGIN (tools/strict_powf/constants.py)\n"
END = "// CONSTANTS END\n"


def with_block(text: str) -> str:
    head, rest = text.split(BEGIN, 1)
    _, tail = rest.split(END, 1)
    return head + BEGIN + constants.block() + "\n" + END + tail


if __name__ == "__main__":
    HEADER.write_text(with_block(HEADER.read_text()))
