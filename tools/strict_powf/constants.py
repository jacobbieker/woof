"""Emit the constant block of woof/backend/translate/strict_powf.cuh.

Every constant is computed here with Python's ``decimal`` at 100 digits
(no third-party code, no library tables), so the header is reproducible:

    python tools/strict_powf/constants.py > block.txt

The header carries the printed block verbatim between its CONSTANTS
markers; tests/test_strict_powf.py regenerates it and compares.

What the constants are:

* the reduction table.  ``m`` in [0.75, 1.5) is split at the centres
  ``c_i = 1 + i/128`` (i = -32 .. 64); ``invc_i`` is 1/c_i rounded to 24
  significant bits (exactly 1 at i = 0), so ``m * invc_i`` is exact in
  binary64 and ``r = m * invc_i - 1`` is exact too; ``logc_i`` is
  -log2(invc_i) as a double-double;
* 2^(j/64), j = 0 .. 63, as double-doubles;
* Taylor coefficients: log2(1 + r) = sum (-1)^(k+1) r^k / (k ln 2) and
  2^g = sum (g ln 2)^k / k!, as doubles for the fast phase and as
  double-doubles for the accurate one.  Truncation bounds are in the
  header's comments.
"""
from __future__ import annotations

import decimal
import math
from decimal import Decimal

decimal.getcontext().prec = 100
LN2 = Decimal(2).ln()
LOG_LO, LOG_HI = -32, 64
FAST_LOG_TERMS, ACCURATE_LOG_TERMS = 10, 18
FAST_EXP_TERMS, ACCURATE_EXP_TERMS = 7, 14


def nearest(v: Decimal) -> float:
    """The binary64 nearest to ``v`` (float(str) rounds correctly)."""
    return float(str(v))


def dd(v: Decimal) -> tuple[float, float]:
    hi = nearest(v)
    return hi, nearest(v - Decimal(hi))


def round_bits(v: Decimal, bits: int) -> float:
    """``v`` (positive) rounded to ``bits`` significant bits, ties to even."""
    e = math.frexp(nearest(v))[1]          # 2^(e-1) <= v < 2^e
    q = int((v * Decimal(2) ** (bits - e)).to_integral_value(rounding=decimal.ROUND_HALF_EVEN))
    return math.ldexp(q, e - bits)         # exact: q has at most bits + 1 bits


def log2(v: Decimal) -> Decimal:
    return v.ln() / LN2


def h(x: float) -> str:
    if x == 0:
        return "0x0p+0"
    return x.hex() if x > 0 else "(" + x.hex() + ")"


def table(name: str, values) -> str:
    rows = ",\n".join("    " + ", ".join(h(v) for v in chunk)
                      for chunk in (values[i:i + 3] for i in range(0, len(values), 3)))
    return f"WSP_TABLE double {name}[{len(values)}] = {{\n{rows}\n}};"


def block() -> str:
    invc, logc_hi, logc_lo = [], [], []
    for i in range(LOG_LO, LOG_HI + 1):
        c = 1 + Decimal(i) / 128
        v = Decimal(1) if i == 0 else Decimal(round_bits(1 / c, 24))
        invc.append(float(v))
        hi, lo = dd(-log2(v))
        logc_hi.append(hi)
        logc_lo.append(lo)
    exp_hi, exp_lo = [], []
    for j in range(64):
        hi, lo = dd(Decimal(2) ** (Decimal(j) / 64))
        exp_hi.append(hi)
        exp_lo.append(lo)
    log_coeff = [(-1) ** (k + 1) / (Decimal(k) * LN2) for k in range(1, ACCURATE_LOG_TERMS + 1)]
    exp_coeff, fact = [], Decimal(1)
    for k in range(1, ACCURATE_EXP_TERMS + 1):
        fact *= k
        exp_coeff.append(LN2 ** k / fact)
    out = [
        f"#define WSP_LOG_LO ({LOG_LO})",
        f"#define WSP_LOG_ENTRIES {LOG_HI - LOG_LO + 1}",
        f"#define WSP_FAST_LOG_TERMS {FAST_LOG_TERMS}",
        f"#define WSP_ACCURATE_LOG_TERMS {ACCURATE_LOG_TERMS}",
        f"#define WSP_FAST_EXP_TERMS {FAST_EXP_TERMS}",
        f"#define WSP_ACCURATE_EXP_TERMS {ACCURATE_EXP_TERMS}",
        table("wsp_invc", invc),
        table("wsp_logc_hi", logc_hi),
        table("wsp_logc_lo", logc_lo),
        table("wsp_exp2_hi", exp_hi),
        table("wsp_exp2_lo", exp_lo),
        "// log2(1 + r) = sum_k wsp_log_hi[k-1] r^k (+ wsp_log_lo for the accurate phase)",
        table("wsp_log_hi", [dd(v)[0] for v in log_coeff]),
        table("wsp_log_lo", [dd(v)[1] for v in log_coeff]),
        "// 2^g = 1 + sum_k wsp_exp_hi[k-1] g^k (+ wsp_exp_lo for the accurate phase)",
        table("wsp_exp_hi", [dd(v)[0] for v in exp_coeff]),
        table("wsp_exp_lo", [dd(v)[1] for v in exp_coeff]),
    ]
    return "\n".join(out)


if __name__ == "__main__":
    print(block())
