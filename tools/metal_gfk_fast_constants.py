"""Emit the binary32 constant tables of woof/backend/metal/gfk_fast.h.

gfk_fast.h evaluates glibc's logf, expf and powf (glibc_flt32.cuh's
gfk_log, gfk_exp, gfk_pow) in double-float, so every binary64 constant of
those routines is carried as an unevaluated sum of binary32 values: hi =
RN32(d), lo = RN32(d - hi) (two terms, 48 bits) or a third term for the
expf reduction constant, whose product with x must be exact to 2^-46.
The values are read from glibc_flt32.cuh itself, so a table here cannot
drift from the routine it shortcuts: tests/test_metal_gfk_fast.py checks
that the header carries exactly this script's output.

usage: python tools/metal_gfk_fast_constants.py   (prints the block)
"""
from __future__ import annotations

import re
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "woof" / "core" / "kernels" / "glibc_flt32.cuh"
BEGIN = "// BEGIN tools/metal_gfk_fast_constants.py"
END = "// END tools/metal_gfk_fast_constants.py"


def f32(d: float) -> float:
    return struct.unpack("<f", struct.pack("<f", d))[0]


def split(d: float, terms: int = 2) -> list[float]:
    out = []
    for _ in range(terms - 1):
        hi = f32(d)
        out.append(hi)
        d = d - hi          # exact in binary64: hi is d rounded to 24 bits
    out.append(f32(d))
    return out


def lit(x: float) -> str:
    return "0.0f" if x == 0.0 else f"{x.hex()}f".replace("0x", "0x", 1)


def _array(text: str, name: str) -> list[float]:
    body = re.search(r"%s\[\d+\]\s*=\s*\{(.*?)\};" % re.escape(name), text, re.S).group(1)
    return [float.fromhex(v.strip()) for v in body.replace("\n", " ").split(",") if v.strip()]


def _define(text: str, name: str) -> float:
    value = re.search(r"#define %s \(?([-+0-9a-fx.p]+)\)?" % re.escape(name), text).group(1)
    return float.fromhex(value)


def tables(text: str | None = None) -> dict[str, list[float]]:
    text = HEADER.read_text(encoding="utf-8") if text is None else text
    out: dict[str, list[float]] = {}
    for name in ("GFK_LOGF_INVC", "GFK_LOGF_LOGC", "GFK_POWF_INVC", "GFK_POWF_LOGC"):
        values = _array(text, name)
        out[name + "_HI"] = [split(v)[0] for v in values]
        out[name + "_LO"] = [split(v)[1] for v in values]
    bits = re.search(r"GFK_EXP2F_TAB\[32\]\s*=\s*\{(.*?)\};", text, re.S).group(1)
    tab = [int(v.strip().rstrip("ULL").rstrip("ull"), 16) for v in bits.split(",") if v.strip()]
    two = [struct.unpack("<d", struct.pack("<Q", (t + (j << 47)) & (2**64 - 1)))[0] for j, t in enumerate(tab)]
    out["GFK_EXP2_T_HI"] = [split(v)[0] for v in two]
    out["GFK_EXP2_T_LO"] = [split(v)[1] for v in two]
    scalars = {
        "GFK_LOGF_LN2": _define(text, "GFK_LOGF_LN2"),
        "GFK_LOGF_A0": _define(text, "GFK_LOGF_A0"),
        "GFK_LOGF_A1": _define(text, "GFK_LOGF_A1"),
        "GFK_LOGF_A2": _define(text, "GFK_LOGF_A2"),
        # gfk_exp's reduction constant and its pre-scaled polynomial
        "GFK_EXPF_P0S": _define(text, "GFK_EXP2F_P0") / 32.0 / 32.0 / 32.0,
        "GFK_EXPF_P1S": _define(text, "GFK_EXP2F_P1") / 32.0 / 32.0,
        "GFK_EXPF_P2S": _define(text, "GFK_EXP2F_P2") / 32.0,
        "GFK_EXP2F_P0": _define(text, "GFK_EXP2F_P0"),
        "GFK_EXP2F_P1": _define(text, "GFK_EXP2F_P1"),
        "GFK_EXP2F_P2": _define(text, "GFK_EXP2F_P2"),
    }
    for i, v in enumerate(_array(text, "GFK_POWF_A")):
        scalars[f"GFK_POWF_A{i}"] = v
    for name, v in scalars.items():
        out[name] = split(v)
    invln2n = re.search(r"const double invln2n = ([-+0-9a-fx.p]+) \* 32\.0;", text).group(1)
    out["GFK_EXPF_INVLN2N"] = split(float.fromhex(invln2n) * 32.0, 3)
    return out


def block(text: str | None = None) -> str:
    lines = [BEGIN, "// generated: do not edit by hand"]
    for name, values in tables(text).items():
        if name.endswith(("_HI", "_LO")) and len(values) > 3:
            lines.append(f"constant float W{name}[{len(values)}] = {{")
            for i in range(0, len(values), 4):
                lines.append("    " + ", ".join(lit(v) for v in values[i:i + 4]) + ",")
            lines.append("};")
        else:
            for part, v in zip(("HI", "LO", "LO2"), values):
                lines.append(f"#define W{name}_{part} {lit(v)}")
    lines.append(END)
    return "\n".join(lines)


if __name__ == "__main__":
    print(block())
