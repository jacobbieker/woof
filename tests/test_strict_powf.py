"""WOOF's correctly rounded powf (woof/backend/translate/strict_powf.cuh),
without a GPU: its constants are the generator's output, its host C++
build rounds correctly on sampled arguments, and the opt-in transcription
of CUDA's powf stays out of the package.

The Metal translation is checked word for word by
tests/test_strict_float32_power.py.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "woof" / "backend" / "translate" / "strict_powf.cuh"
sys.path.insert(0, str(ROOT / "tools" / "strict_powf"))

import reference  # noqa: E402
import write_header  # noqa: E402

_HARNESS = r"""
#include <cstdio>
#include <vector>
#include "strict_powf.cuh"
int main(int argc, char** argv) {
    FILE* f = std::fopen(argv[1], "rb");
    std::fseek(f, 0, SEEK_END);
    long n = std::ftell(f) / 8;
    std::rewind(f);
    std::vector<float> xy(2 * n), z(n);
    if (std::fread(xy.data(), 4, 2 * n, f) != (size_t)(2 * n)) return 1;
    for (long i = 0; i < n; ++i) z[i] = woof_strict_powf(xy[2 * i], xy[2 * i + 1]);
    FILE* o = std::fopen(argv[2], "wb");
    std::fwrite(z.data(), 4, n, o);
    std::fclose(o);
    return 0;
}
"""


def test_constants_are_the_generator_output():
    text = HEADER.read_text(encoding="utf-8")
    assert write_header.with_block(text) == text


def test_the_cuda_transcription_is_not_package_data():
    """The opt-in lives in tools/cuda_powf, which no wheel carries."""
    assert not list((ROOT / "woof").rglob("cuda_powf*"))
    assert (ROOT / "tools" / "cuda_powf" / "cuda_powf.cuh").is_file()
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    setuptools = config["tool"]["setuptools"]
    assert not any(name.startswith("tools.") for name in setuptools["packages"]["find"]["include"])
    assert not any("cuda_powf" in pattern
                   for patterns in setuptools["package-data"].values() for pattern in patterns)


@pytest.fixture(scope="module")
def host_powf(tmp_path_factory):
    compiler = os.environ.get("CXX") or shutil.which("c++") or shutil.which("clang++") or shutil.which("g++")
    if not compiler:
        pytest.skip("no C++ compiler")
    work = tmp_path_factory.mktemp("strict_powf")
    (work / "harness.cpp").write_text(_HARNESS)
    exe = work / "harness"
    subprocess.run([compiler, "-O2", "-ffp-contract=off", "-std=c++17", f"-I{HEADER.parent}",
                    str(work / "harness.cpp"), "-o", str(exe)], check=True)

    def run(x, y):
        xy = np.stack([np.asarray(x, np.float32), np.asarray(y, np.float32)], axis=1)
        xy.tofile(work / "in.bin")
        subprocess.run([str(exe), str(work / "in.bin"), str(work / "out.bin")], check=True)
        return np.fromfile(work / "out.bin", np.float32)
    return run


def _check(run, x, y):
    got = run(x, y)
    want, _ = reference.powf(x, y)
    bad = np.flatnonzero((got.view(np.uint32) != want.view(np.uint32))
                         & ~(np.isnan(got) & np.isnan(want)))
    assert bad.size == 0, [(float(x[i]), float(y[i]), float(got[i]), float(want[i])) for i in bad[:5]]


def test_host_build_is_correctly_rounded_on_samples(host_powf):
    rng = np.random.default_rng(2026)
    n = 200_000
    # the Exner function's range, random positive x over every exponent,
    # x near 1 with integer and fractional y, negative x with integer y
    x = np.concatenate([rng.uniform(0.5, 1.0, n).astype(np.float32),
                        rng.integers(1, 0x7F800000, n, dtype=np.uint32).view(np.float32),
                        rng.uniform(0.01, 3.0, n).astype(np.float32),
                        -rng.uniform(0.01, 3.0, n).astype(np.float32)])
    y = np.concatenate([np.full(n, 0.2857142857142857, np.float32),
                        (rng.standard_normal(n) * 3).astype(np.float32),
                        rng.uniform(-8, 8, n).astype(np.float32),
                        np.round(rng.uniform(-8, 8, n)).astype(np.float32)])
    _check(host_powf, x, y)


def test_host_build_special_values_and_midpoints(host_powf):
    specials = np.array([0.0, -0.0, 1.0, -1.0, np.inf, -np.inf, np.nan, 2.0, 0.5,
                         1e-45, -1e-45, 3.4e38, 1e-38, 3.0, -3.0, 2.5], np.float32)
    x, y = (a.ravel() for a in np.meshgrid(specials, specials))
    _check(host_powf, x, y)
    # exact binary32 midpoints: odd x of 13 bits squared, 257^3 as 66049^1.5
    odd = np.arange(4097, 8192, 2, dtype=np.float32)
    squares = np.arange(257, 323, 2, dtype=np.float32) ** 2
    _check(host_powf, np.concatenate([odd, squares]),
           np.concatenate([np.full(odd.size, 2, np.float32), np.full(squares.size, 1.5, np.float32)]))
