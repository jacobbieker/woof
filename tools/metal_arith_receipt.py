"""Measure what Apple-GPU Metal arithmetic does, word for word.

The CUDA side of woof pins its arithmetic with NVRTC options (``--fmad``,
``--ftz``, ``--prec-div``, ``--prec-sqrt``) and per-operation intrinsics.
Metal has no such flags.  This receipt measures, on the GPU it runs on and
through the same launcher production uses (``woof.backend.metal.ext``), the
facts the CUDA-to-Metal translation must rely on:

``ops``         binary32 ``+ - * /``, ``sqrt``, ``fma`` against IEEE-754
                round-to-nearest (NumPy's binary32 arithmetic), including
                subnormal operands and results, signed zero, infinities.
``contraction`` whether ``a*b + c`` is fused, under each spelling that could
                stop it.
``subnormals``  whether subnormal operands and results survive.
``conversions`` float<->int rounding, and NaN payloads through copies.
``libm``        how far ``metal::precise::`` and default-namespace exp/log/
                pow/... are from correctly rounded (informational: woof's
                pinned kernels use the in-repo word-exact libm ports).

No verdict is typed here: each is derived from counted bit mismatches.

Usage::

    pixi run -e mlx python tools/metal_arith_receipt.py \
        --out woof/backend/metal/arith_receipt.json
"""

from __future__ import annotations

import argparse
import json
import platform
import struct
import sys
from pathlib import Path

import numpy as np

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SAFE, RELAXED, FAST = 0, 1, 2
_HEADER = "#include <metal_stdlib>\nusing namespace metal;\n"


def _kernel(name: str, body: str, *, pragma: str = "", arity: int = 3,
            out_type: str = "float", in_type: str = "float") -> str:
    params = ", ".join(
        f"const device {in_type}* x{k} [[buffer({k})]]" for k in range(arity))
    return (_HEADER + pragma + "\n"
            f"[[kernel]] void {name}({params}, device {out_type}* out [[buffer({arity})]],\n"
            f"    constant uint& n [[buffer({arity + 1})]],\n"
            "    uint3 tid [[thread_position_in_threadgroup]],\n"
            "    uint3 bid [[threadgroup_position_in_grid]],\n"
            "    uint3 bdim [[threads_per_threadgroup]]) {\n"
            "  uint i = bid.x * bdim.x + tid.x;\n"
            "  if (i >= n) return;\n"
            + "".join(f"  {in_type} v{k} = x{k}[i];\n" for k in range(arity))
            + f"  out[i] = {body};\n}}\n")


class Runner:
    def __init__(self):
        import mlx.core as mx
        from woof.backend.metal import ext
        self.mx = mx
        self.launch = ext.load().launch
        self._serial = 0

    def run(self, source: str, name: str, inputs, out_dtype, *, math_mode=SAFE):
        mx = self.mx
        n = len(inputs[0])
        xs = [mx.array(x) for x in inputs]
        out = mx.zeros((n,), dtype=out_dtype)
        self._serial += 1
        groups = (n + 255) // 256
        buffers = xs + [out]
        writable = [False] * len(xs) + [True]
        order = list(range(len(buffers))) + [-1]
        (result,) = self.launch(
            f"woof-arith-receipt-{self._serial}", source, name, (groups, 1, 1),
            (256, 1, 1), buffers, [0] * len(buffers), writable,
            [struct.pack("I", n)], order, 0, math_mode)
        return np.array(result)


def _bits32(x):
    return np.asarray(x, dtype=np.float32).view(np.uint32)


def _same_words(got, want, *, nan_any=True):
    """Bitwise equal; with ``nan_any`` every NaN matches every NaN."""
    g, w = _bits32(got), _bits32(want)
    same = g == w
    if nan_any:
        same |= np.isnan(np.asarray(got, np.float32)) & np.isnan(np.asarray(want, np.float32))
    return same


def _ulp(got, want):
    """|ULP| distance between binary32 words (monotone key), NaN-aware."""
    from woof.core.fp32_ulp import fp32_ulp_distance
    return np.asarray(fp32_ulp_distance(np.asarray(got, np.float32),
                                        np.asarray(want, np.float32)))


def _operands(rng, n):
    """Random binary32 words over every class, plus a dense subnormal band."""
    words = rng.integers(0, 2**32, size=n, dtype=np.uint64).astype(np.uint32)
    x = words.view(np.float32)
    k = n // 8
    sub = rng.integers(1, 2**23, size=k, dtype=np.uint64).astype(np.uint32)
    sub |= (rng.integers(0, 2, size=k, dtype=np.uint64).astype(np.uint32) << 31)
    x[:k] = sub.view(np.float32)
    # Normal operands with modest exponents, so products and quotients land
    # in range and exercise rounding rather than overflow.
    m = n // 4
    x[k:k + m] = (rng.standard_normal(m) * np.exp2(rng.integers(-20, 20, m))).astype(np.float32)
    specials = np.array([0.0, -0.0, np.inf, -np.inf, np.nan, 1.0, -1.0,
                         np.finfo(np.float32).tiny, np.finfo(np.float32).max,
                         np.float32(1e-45), np.float32(-1e-45)], np.float32)
    x[k + m:k + m + specials.size] = specials
    return x


def _summary(mask, got, want):
    bad = int((~mask).sum())
    row = {"samples": int(mask.size), "mismatches": bad}
    if bad:
        ulp = _ulp(got[~mask], want[~mask])
        finite = ulp[np.isfinite(ulp.astype(np.float64))]
        row["max_ulp"] = int(finite.max()) if finite.size else None
        idx = np.flatnonzero(~mask)[:4]
        row["examples"] = [{"got": hex(int(_bits32(got)[i])), "want": hex(int(_bits32(want)[i]))}
                           for i in idx]
    return row


def _subnormal(x):
    x = np.asarray(x, np.float32)
    return (x != 0) & (np.abs(x) < np.finfo(np.float32).tiny)


def _flush(x):
    """Subnormal binary32 words to the zero of the same sign."""
    x = np.asarray(x, np.float32)
    return np.where(_subnormal(x), np.copysign(np.float32(0), x), x).astype(np.float32)


def measure_ops(r: Runner, rng, n):
    """Each op against two references.

    ``ieee``: NumPy binary32, subnormals kept.  ``ftz``: operands flushed
    to signed zero, the IEEE op, then a subnormal result flushed (after
    rounding) -- CUDA's ``.ftz`` model.  ``normal_only`` restricts the IEEE
    comparison to samples with no subnormal operand and no subnormal IEEE
    result, which isolates rounding from flushing.
    """
    a, b, c = (_operands(rng, n) for _ in range(3))
    fa, fb, fc = _flush(a), _flush(b), _flush(c)
    with np.errstate(all="ignore"):
        cases = {
            "add": ("v0 + v1", lambda x, y, z: x + y),
            "sub": ("v0 - v1", lambda x, y, z: x - y),
            "mul": ("v0 * v1", lambda x, y, z: x * y),
            "div": ("v0 / v1", lambda x, y, z: x / y),
            "sqrt": ("sqrt(v0)", lambda x, y, z: np.sqrt(x)),
            "precise_sqrt": ("precise::sqrt(v0)", lambda x, y, z: np.sqrt(x)),
            "precise_divide": ("precise::divide(v0, v1)", lambda x, y, z: x / y),
            "fma": ("fma(v0, v1, v2)", _fma32),
        }
    out = {}
    any_sub_in = _subnormal(a) | _subnormal(b) | _subnormal(c)
    for key, (expr, fn) in cases.items():
        with np.errstate(all="ignore"):
            ieee = np.asarray(fn(a, b, c), np.float32)
            ftz = _flush(fn(fa, fb, fc))
        normal = ~any_sub_in & ~_subnormal(ieee)
        for mode_name, mode in (("safe", SAFE), ("fast", FAST)):
            got = r.run(_kernel("k", expr, pragma="#pragma METAL fp contract(off)"), "k",
                        [a, b, c], r.mx.float32, math_mode=mode)
            out[f"{key}/{mode_name}"] = {
                "ieee": _summary(_same_words(got, ieee), got, ieee),
                "ftz": _summary(_same_words(got, ftz), got, ftz),
                "normal_only": _summary(_same_words(got[normal], ieee[normal]),
                                        got[normal], ieee[normal]),
            }
    return out


def _fma32(a, b, c):
    """Correctly rounded binary32 fma.

    The binary32 product is exact in binary64.  TwoSum gives the binary64
    sum ``s`` and its exact error ``e``; rounding ``s`` to binary32 is then
    correct unless ``s`` sits exactly on a binary32 halfway point and ``e``
    is non-zero, in which case the true sum lies on ``e``'s side of it.
    """
    a64, b64, c64 = (np.asarray(v, np.float64) for v in (a, b, c))
    with np.errstate(all="ignore"):
        p = a64 * b64
        s = p + c64
        bp = s - c64
        e = (p - bp) + (c64 - (s - bp))
        r = s.astype(np.float32)
        lo = np.nextafter(r, np.float32(-np.inf))
        hi = np.nextafter(r, np.float32(np.inf))
        r64 = r.astype(np.float64)
        below = (r64 > s) & ((r64 + lo.astype(np.float64)) / 2 == s)  # s at midpoint (lo, r)
        above = (r64 < s) & ((r64 + hi.astype(np.float64)) / 2 == s)  # s at midpoint (r, hi)
        ok = np.isfinite(s) & np.isfinite(e) & (e != 0)
        r = np.where(ok & below & (e < 0), lo, r)
        r = np.where(ok & above & (e > 0), hi, r)
    return r.astype(np.float32)


def measure_contraction(r: Runner, rng, n):
    # Operands where fused and unfused a*b+c differ: a*b's low bits matter.
    a = (1.0 + rng.integers(1, 2**23, n) * 2.0**-23).astype(np.float32)
    b = (1.0 + rng.integers(1, 2**23, n) * 2.0**-23).astype(np.float32)
    c = (-(a.astype(np.float64) * b.astype(np.float64))).astype(np.float32)
    unfused = (a * b) + c
    fused = _fma32(a, b, c)
    differ = ~_same_words(unfused, fused)
    variants = {
        "plain": ("", SAFE),
        "plain_relaxed": ("", RELAXED),
        "plain_fast": ("", FAST),
        "metal_fp_contract_off": ("#pragma METAL fp contract(off)", SAFE),
        "metal_fp_contract_off_fast": ("#pragma METAL fp contract(off)", FAST),
        "clang_fp_contract_off": ("#pragma clang fp contract(off)", SAFE),
        "metal_fp_math_mode_safe": ("#pragma METAL fp math_mode(safe)", SAFE),
    }
    out = {"distinguishing_samples": int(differ.sum())}
    for name, (pragma, mode) in variants.items():
        try:
            got = r.run(_kernel("k", "v0 * v1 + v2", pragma=pragma), "k", [a, b, c],
                        r.mx.float32, math_mode=mode)
        except Exception as exc:  # a pragma the compiler rejects is a finding
            out[name] = {"compiles": False, "error": str(exc).splitlines()[0][:200]}
            continue
        g = got[differ]
        out[name] = {
            "compiles": True,
            "fused": int(_same_words(g, fused[differ]).sum()),
            "unfused": int(_same_words(g, unfused[differ]).sum()),
        }
    got = r.run(_kernel("k", "v2 + v0 * v1"), "k", [a, b, c], r.mx.float32)
    g = got[differ]
    out["plain_commuted"] = {"compiles": True,
                             "fused": int(_same_words(g, fused[differ]).sum()),
                             "unfused": int(_same_words(g, unfused[differ]).sum())}
    return out


def measure_subnormals(r: Runner):
    tiny = np.float32(np.finfo(np.float32).tiny)
    sub = np.array([1e-45, 1e-40, -3e-39, 5e-39], np.float32)
    ones = np.ones_like(sub)
    checks = {
        "operand_times_one": ("v0 * v1", [sub, ones, ones], sub * ones),
        "operand_plus_zero": ("v0 + v1", [sub, np.zeros_like(sub), ones], sub + 0.0),
        "result_tiny_half": ("v0 * v1", [np.full(4, tiny, np.float32), np.full(4, 0.5, np.float32), ones],
                             np.full(4, tiny, np.float32) * np.float32(0.5)),
        "copy": ("v0", [sub, ones, ones], sub),
    }
    out = {}
    for mode_name, mode in (("safe", SAFE), ("fast", FAST)):
        for key, (expr, ins, want) in checks.items():
            got = r.run(_kernel("k", expr), "k", ins, r.mx.float32, math_mode=mode)
            out[f"{key}/{mode_name}"] = {
                "retained": bool(_same_words(got, want, nan_any=False).all()),
                "got": [hex(int(w)) for w in _bits32(got)],
                "want": [hex(int(w)) for w in _bits32(want)]}
    return out


def measure_conversions(r: Runner, rng, n):
    mx = r.mx
    out = {}
    f = (rng.standard_normal(n) * 1e6).astype(np.float32)
    got = r.run(_kernel("k", "int(v0)", arity=1, out_type="int"), "k", [f], mx.int32)
    want = np.trunc(f).astype(np.int32)
    out["float_to_int_trunc"] = {"samples": n, "mismatches": int((got != want).sum())}
    i = rng.integers(-2**31, 2**31, n, dtype=np.int64).astype(np.int32)
    got = r.run(_kernel("k", "float(v0)", arity=1, in_type="int"), "k", [i], mx.float32)
    want = i.astype(np.float32)
    out["int_to_float_rne"] = _summary(_same_words(got, want), got, want)
    u = rng.integers(0, 2**32, n, dtype=np.uint64).astype(np.uint32)
    got = r.run(_kernel("k", "float(v0)", arity=1, in_type="uint"), "k", [u], mx.float32)
    want = u.astype(np.float32)
    out["uint_to_float_rne"] = _summary(_same_words(got, want), got, want)
    payloads = np.array([0x7FC00001, 0x7FA00000, 0xFFC12345, 0x7F800001], np.uint32)
    got = r.run(_kernel("k", "v0", arity=1, out_type="uint", in_type="uint"), "k", [payloads], mx.uint32)
    out["nan_payload_copy_as_uint"] = bool((got == payloads).all())
    got = r.run(_kernel("k", "v0", arity=1), "k", [payloads.view(np.float32)], mx.float32)
    out["nan_payload_copy_as_float"] = [hex(int(w)) for w in _bits32(got)]
    got = r.run(_kernel("k", "-v0", arity=1), "k", [np.array([0.0, -0.0], np.float32)], mx.float32)
    out["negate_zero"] = [hex(int(w)) for w in _bits32(got)]
    return out


def measure_libm(r: Runner, rng, n):
    x = (rng.uniform(-20, 20, n)).astype(np.float32)
    pos = np.abs(x) * np.exp2(rng.integers(-30, 30, n)).astype(np.float32) + np.float32(1e-30)
    y = rng.uniform(-4, 4, n).astype(np.float32)
    fns = {
        "exp": ([x], lambda a: np.exp(a.astype(np.float64))),
        "log": ([pos], lambda a: np.log(a.astype(np.float64))),
        "pow": ([pos, y], lambda a, b: np.power(a.astype(np.float64), b.astype(np.float64))),
        "sin": ([x], lambda a: np.sin(a.astype(np.float64))),
        "cos": ([x], lambda a: np.cos(a.astype(np.float64))),
        "tanh": ([x], lambda a: np.tanh(a.astype(np.float64))),
        "log10": ([pos], lambda a: np.log10(a.astype(np.float64))),
        "atan": ([x], lambda a: np.arctan(a.astype(np.float64))),
        "cbrt": ([pos], lambda a: np.cbrt(a.astype(np.float64))),
    }
    out = {}
    for name, (ins, ref) in fns.items():
        with np.errstate(all="ignore"):
            want = ref(*ins).astype(np.float32)
        args = ", ".join(f"v{k}" for k in range(len(ins)))
        for space in ("precise::", ""):
            key = f"{space or 'default::'}{name}"
            try:
                got = r.run(_kernel("k", f"{space}{name}({args})", arity=len(ins)), "k", ins, r.mx.float32)
            except RuntimeError as exc:
                if "Unable to build metal library" not in str(exc):
                    raise
                out[key] = {"available": False}
                continue
            same = _same_words(got, want)
            ulp = _ulp(got, want)
            out[key] = {
                "available": True, "samples": n, "correctly_rounded": int(same.sum()),
                "max_ulp": int(np.nanmax(ulp.astype(np.float64)))}
    return out


def verdicts(receipt):
    ops = receipt["ops"]
    con = receipt["contraction"]
    sub = receipt["subnormals"]

    def clean(key, ref):
        return ops[key][ref]["mismatches"] == 0

    v = {}
    for op in ("add", "sub", "mul", "div", "precise_divide", "sqrt", "precise_sqrt", "fma"):
        for mode in ("safe", "fast"):
            v[f"{op}/{mode}/correctly_rounded_normals"] = clean(f"{op}/{mode}", "normal_only")
            v[f"{op}/{mode}/equals_ftz_model"] = clean(f"{op}/{mode}", "ftz")
    v["plain_mul_add_fused_safe"] = con["plain"].get("unfused", 1) == 0
    v["contract_off_pragma_unfuses"] = (con["metal_fp_contract_off"].get("compiles", False)
                                        and con["metal_fp_contract_off"]["fused"] == 0)
    v["subnormals_retained_safe"] = all(x["retained"] for k, x in sub.items() if k.endswith("/safe"))
    v["subnormals_retained_fast"] = all(x["retained"] for k, x in sub.items() if k.endswith("/fast"))
    return v


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=Path("woof/backend/metal/arith_receipt.json"))
    parser.add_argument("--samples", type=int, default=1 << 22)
    parser.add_argument("--seed", type=int, default=20261029)
    args = parser.parse_args(argv)
    import mlx.core as mx
    r = Runner()
    rng = np.random.default_rng(args.seed)
    receipt = {
        "schema": "woof.metal-arith-receipt/v1",
        "device": mx.device_info(),
        "mlx": mx.__version__,
        "macos": platform.mac_ver()[0],
        "seed": args.seed,
        "ops": measure_ops(r, rng, args.samples),
        "contraction": measure_contraction(r, rng, args.samples),
        "subnormals": measure_subnormals(r),
        "conversions": measure_conversions(r, rng, args.samples),
        "libm": measure_libm(r, rng, args.samples),
    }
    receipt["verdicts"] = verdicts(receipt)
    receipt["device"] = {k: (v if isinstance(v, (int, str, float)) else str(v))
                         for k, v in receipt["device"].items()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=1, sort_keys=True) + "\n")
    print(json.dumps(receipt["verdicts"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
