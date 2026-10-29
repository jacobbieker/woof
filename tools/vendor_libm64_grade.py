"""Grade the facade backends' binary64 libm, word for word, on the GPU.

A CUDA unit calling CUDA's binary64 libm names runs through the real path
(``cupy.RawModule`` on the mlx facade: woof.backend.translate, then Metal).
Its words are compared with the implementation each name maps to
(woof/backend/translate/vendor_libm64.cuh):

* atan2, hypot, log2, expm1, tanh, tgamma: the Rust libm 0.2.16 crate the
  header transcribes, built here from tools/rustwx/vendor/crates-io/libm;
* exp, log, log10, sin, atan, pow: woof's correctly rounded f64cr.cuh,
  built here as host C++ from the same header text.

Every NaN matches every NaN.  Exit status is non-zero on any mismatch.

Usage::

    pixi run -e mlx python tools/vendor_libm64_grade.py --samples 1000000
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LIBM = ROOT / "tools" / "rustwx" / "vendor" / "crates-io" / "libm"
F64CR = ROOT / "woof" / "core" / "kernels" / "f64cr.cuh"

RUST_FNS = ("atan2", "hypot", "log2", "expm1", "tanh", "tgamma", "lgamma", "log1p")
CR_FNS = ("exp", "log", "log10", "sin", "atan", "pow")
BINARY = {"atan2", "hypot", "pow"}
CR_EXTRA = ("cbrt", "cos")      # cr_cbrt and wlm_cos: graded against exact decimal values
F32_FNS = ("cbrtf", "tgammaf", "log1pf")   # binary32: Rust (cbrtf, tgammaf), glibc (log1pf)
ALL = RUST_FNS + CR_FNS + CR_EXTRA

CUDA_UNIT = """
extern "C" __global__ void grade(const double* x, const double* y, double* out, float* outf, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double a = x[i], b = y[i];
    float af = (float)a;
""" + "".join(
    f"    out[(size_t)i * {len(ALL)} + {k}] = {fn}(a{', b' if fn in BINARY else ''});\n"
    for k, fn in enumerate(ALL)) + "".join(
    f"    outf[(size_t)i * {len(F32_FNS)} + {k}] = {fn}(af);\n" for k, fn in enumerate(F32_FNS)) + "}\n"

RUST_MAIN = """
use std::io::{Read, Write};
fn main() {
    let mut buf = Vec::new();
    std::io::stdin().read_to_end(&mut buf).unwrap();
    let n = buf.len() / 16;
    let word = |k: usize| f64::from_le_bytes(buf[8 * k..8 * k + 8].try_into().unwrap());
    let mut out = std::io::BufWriter::new(std::io::stdout());
    for i in 0..n {
        let (a, b) = (word(i), word(n + i));
        for v in [libm::atan2(a, b), libm::hypot(a, b), libm::log2(a), libm::expm1(a),
                  libm::tanh(a), libm::tgamma(a), libm::lgamma(a), libm::log1p(a)] {
            out.write_all(&v.to_bits().to_le_bytes()).unwrap();
        }
        let af = a as f32;
        for v in [libm::cbrtf(af), libm::tgammaf(af)] {
            out.write_all(&(v as f64).to_bits().to_le_bytes()).unwrap();
        }
    }
}
"""

HOST_CR = """
#include "f64cr.cuh"
#include <cstdio>
#include <cstdlib>
#include <vector>
int main() {
    std::vector<double> buf;
    double v;
    while (fread(&v, 8, 1, stdin) == 1) buf.push_back(v);
    size_t n = buf.size() / 2;
    for (size_t i = 0; i < n; ++i) {
        double a = buf[i], b = buf[n + i];
        double r[6] = {wcr_exp(a), wcr_log(a), wcr_log10(a), wcr_sin(a), wcr_atan(a), wcr_pow(a, b)};
        fwrite(r, 8, 6, stdout);
    }
}
"""


def operands(n: int, seed: int):
    rng = np.random.default_rng(seed)
    parts = []
    for _ in range(2):
        x = rng.standard_normal(n) * np.exp2(rng.integers(-12, 12, n))
        k = n // 8
        x[:k] = rng.uniform(-200, 200, k)                       # tgamma/exp/tanh range
        x[k:2 * k] = rng.uniform(0, 4, k)
        x[2 * k:2 * k + 32] = [0.0, -0.0, 1.0, -1.0, 0.5, 2.0, 3.0, 10.0, 0.1, -0.1, 1e-300, -1e-300,
                               1e300, -1e300, np.inf, -np.inf, np.nan, 5e-324, -5e-324, 171.0, 172.0,
                               -2.5, -1.5, 23.0, 24.0, 0.25, 1e-20, -1e-20, 709.0, -745.0, 20.0, -20.0]
        x[2 * k + 32:3 * k] = np.floor(rng.uniform(-30, 30, k - 32))  # integers
        parts.append(x)
    return parts


def rust_reference(a, b) -> np.ndarray:
    cargo = shutil.which("cargo")
    if cargo is None:
        raise RuntimeError("cargo is needed to build the Rust libm reference")
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        (t / "src").mkdir()
        (t / "src" / "main.rs").write_text(RUST_MAIN)
        (t / "Cargo.toml").write_text(
            '[package]\nname = "libm_ref"\nversion = "0.0.0"\nedition = "2021"\n\n'
            f'[dependencies]\nlibm = {{ path = "{LIBM}" }}\n')
        subprocess.run([cargo, "build", "--release", "--offline", "-q"], cwd=t, check=True)
        data = np.concatenate([a, b]).astype("<f8").tobytes()
        out = subprocess.run([str(t / "target" / "release" / "libm_ref")], input=data,
                             capture_output=True, check=True).stdout
    return np.frombuffer(out, "<f8").reshape(len(a), len(RUST_FNS) + 2)


def cr_reference(a, b) -> np.ndarray:
    cxx = os.environ.get("CXX") or shutil.which("clang++") or shutil.which("c++")
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        (t / "ref.cpp").write_text(HOST_CR)
        subprocess.run([cxx, "-O2", "-std=c++17", "-ffp-contract=off", f"-I{F64CR.parent}",
                        str(t / "ref.cpp"), "-o", str(t / "ref")], check=True)
        data = np.concatenate([a, b]).astype("<f8").tobytes()
        out = subprocess.run([str(t / "ref")], input=data, capture_output=True, check=True).stdout
    return np.frombuffer(out, "<f8").reshape(len(a), len(CR_FNS))


def gpu(a, b) -> np.ndarray:
    os.environ.setdefault("WOOF_BACKEND", "mlx")
    import woof  # noqa: F401
    import cupy as cp
    n = len(a)
    x, y = cp.asarray(a), cp.asarray(b)
    out = cp.zeros((n * len(ALL),), dtype=cp.float64)
    outf = cp.zeros((n * len(F32_FNS),), dtype=cp.float32)
    kern = cp.RawModule(code=CUDA_UNIT, options=("-std=c++17", "--fmad=false")).get_function("grade")
    kern(((n + 127) // 128,), (128,), (x, y, out, outf, np.int32(n)))
    return out.get().reshape(n, len(ALL)), outf.get().reshape(n, len(F32_FNS))


def exact_cbrt(a) -> np.ndarray:
    """Correctly rounded binary64 cube roots, from 60-digit decimal roots."""
    import decimal
    ctx = decimal.Context(prec=60)
    out = np.empty(len(a))
    for i, v in enumerate(a):
        if not np.isfinite(v) or v == 0:
            out[i] = np.cbrt(v)
            continue
        d = ctx.power(ctx.abs(decimal.Decimal(float(v))), ctx.divide(1, 3))
        r = float(d)          # nearest binary64 to a 60-digit root: correct barring exact ties
        out[i] = -r if v < 0 else r
    return out


def exact_cos(a) -> np.ndarray:
    """Correctly rounded binary64 cosines below 2^20, from an 80-digit
    reduction and series; NaN (ungraded: the known gap) above."""
    import decimal
    ctx = decimal.Context(prec=80)
    D = decimal.Decimal
    pi = D("3.14159265358979323846264338327950288419716939937510582097494459230781640628620899862803482534211706798214808651")
    out = np.empty(len(a))
    for i, v in enumerate(a):
        if not np.isfinite(v):
            out[i] = np.nan
            continue
        if abs(v) >= 2.0 ** 20:
            out[i] = np.nan
            continue
        x = ctx.remainder_near(D(float(v)), ctx.multiply(2, pi))
        x2, term, total, n = ctx.multiply(x, x), D(1), D(1), 0
        while True:
            n += 2
            term = ctx.divide(ctx.multiply(-term, x2), (n - 1) * n)
            if abs(term) < D(10) ** -78:
                break
            total = ctx.add(total, term)
        out[i] = float(total)
    return out


def log1pf_reference(af) -> np.ndarray:
    from woof.core.noahmp_libm import log1pf
    return np.array([log1pf(np.float32(v)) for v in af], dtype=np.float32)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--samples", type=int, default=1 << 20)
    ap.add_argument("--seed", type=int, default=20261029)
    args = ap.parse_args(argv)
    a, b = operands(args.samples, args.seed)
    got, gotf = gpu(a, b)
    rust = rust_reference(a, b)
    want = np.concatenate([rust[:, :len(RUST_FNS)], cr_reference(a, b),
                           exact_cbrt(a)[:, None], exact_cos(a)[:, None]], axis=1)
    af = a.astype(np.float32)
    wantf = np.stack([rust[:, len(RUST_FNS)].astype(np.float32), rust[:, len(RUST_FNS) + 1].astype(np.float32),
                      log1pf_reference(af)], axis=1)
    summary, status = {}, 0
    for k, fn in enumerate(F32_FNS):
        g, w = gotf[:, k], wantf[:, k]
        ok = (g.view(np.uint32) == w.view(np.uint32)) | (np.isnan(g) & np.isnan(w))
        bad = np.flatnonzero(~ok)
        summary[fn] = {"samples": len(g), "mismatches": int(bad.size)}
        if bad.size:
            summary[fn]["example"] = {"a": repr(af[bad[0]]), "got": repr(g[bad[0]]), "want": repr(w[bad[0]])}
            status = 1
    for k, fn in enumerate(ALL):
        g, w = got[:, k], want[:, k]
        ok = (g.view(np.uint64) == w.view(np.uint64)) | (np.isnan(g) & np.isnan(w))
        if fn == "cos":         # NaN reference: inf/nan operands and the |x| >= 2^20 known gap
            ok |= np.isnan(w) & (np.isnan(g) | (np.abs(a) >= 2.0 ** 20))
        bad = np.flatnonzero(~ok)
        summary[fn] = {"samples": len(g), "mismatches": int(bad.size)}
        if bad.size:
            i = bad[0]
            summary[fn]["example"] = {"a": repr(a[i]), "b": repr(b[i]), "got": repr(g[i]), "want": repr(w[i])}
            status = 1
    print(json.dumps(summary, indent=1))
    return status


if __name__ == "__main__":
    sys.exit(main())
