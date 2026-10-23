"""Write the reader oracle's input decks (synthetic turbines only).

    python make_deck.py <DECK_ROOT>

Two decks, each a run directory as WRF would see it:

* ``ij``: ``windturbines-ij.txt`` (``windfarm_ij = 1``) and two tables;
* ``latlon``: ``windturbines.txt`` (``windfarm_ij = 0``) and one table.

The values are chosen to exercise what a list-directed READ does with a
decimal: commas as separators, ``d`` exponents, integers read as REALs,
values a record carries past what the READ asks for, a READ that continues
onto the next record, and decimals a hair above a binary32 rounding
midpoint, where rounding through binary64 first gives the other word
(strtof, which libgfortran uses, rounds the decimal itself).
"""
from __future__ import annotations

import struct
import sys
from decimal import Decimal, getcontext
from fractions import Fraction
from pathlib import Path

getcontext().prec = 120


def _f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _next_up(value: float) -> float:
    """The next binary32 above ``value`` (a binary32, nonzero)."""
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    bits = bits + 1 if value > 0 else bits - 1
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def above_midpoint(value: float) -> str:
    """A decimal just above the midpoint after float32(value): binary64
    rounds it onto the midpoint, and ties-to-even then picks the even word,
    where the correctly rounded read picks the upper one."""
    low = _f32(value)
    high = _next_up(low)
    mid = (Fraction(low) + Fraction(high)) / 2
    exact = mid + Fraction(1, 2 ** 90)
    text = Decimal(exact.numerator) / Decimal(exact.denominator)
    return format(text, "f")


def main(argv=None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        raise SystemExit(__doc__.splitlines()[2].strip())
    root = Path(args[0])
    ij = root / "ij"
    ij.mkdir(parents=True, exist_ok=True)
    (ij / "windturbines-ij.txt").write_text(
        "5 6 1\n"
        "5 6 1\n"
        "5,6,3\n"
        "9 9 1 7 8\n"
        "12 4 3\n", encoding="ascii")
    (ij / "wind-turbine-1.tbl").write_text(
        "4 99\n"
        "94.0\n"
        "112.0, 0.13\n"
        "3.0 extra values\n"
        "3 0.8 10\n"
        f"{above_midpoint(7.3)} {above_midpoint(0.79)} 1.05d3\n"
        "12.5D0 0.6 3.0E+03\n"
        "25. .21 3000.\n", encoding="ascii")
    (ij / "wind-turbine-3.tbl").write_text(
        "3\n"
        "150. 220. 0.11 8.0\n"
        "3.5 0.82 25\n"
        f"11.0 {above_midpoint(0.4123)} {above_midpoint(7999.9)}\n"
        "28. 0.05 8000\n", encoding="ascii")
    latlon = root / "latlon"
    latlon.mkdir(parents=True, exist_ok=True)
    (latlon / "windturbines.txt").write_text(
        f"{above_midpoint(36.0915)} -98.987263 2\n"
        f"36.091546 {above_midpoint(-98.9805)} 2\n"
        "36.0969,-98.98058,2\n"
        "3.6097d1 -98.973904 2 surplus\n", encoding="ascii")
    (latlon / "wind-turbine-2.tbl").write_text(
        "2\n"
        "100 117 0.13 3.45\n"
        "3. 0.8 20.\n"
        "25 0.15 3450\n", encoding="ascii")
    print(f"decks in {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
