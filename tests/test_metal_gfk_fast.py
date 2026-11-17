"""glibc_flt32.cuh's gfk_log, gfk_exp and gfk_pow on Metal: the binary32
shortcut of woof/backend/metal/gfk_fast.h returns the binary64 routine's
word, or defers to it.

On Metal the routines' binary64 arithmetic is integer soft-float, which made
sfclay's option-1 surface layer (a secant loop over logs and powers) cost
about 12 ms for 4,565 points on an M3 Max; with the shortcut it is about
2 ms, with the same words.  tools/metal_gfk_fast_grade.py grades every
logf/expf input and every powf base for a set of exponents; these tests run
a strided sample of the same probes, check the translator's rewrite and the
opt-out, and check the header's constants against their generator.
"""

from __future__ import annotations

import importlib.util
import os
import re
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import metal_gfk_fast_constants as constants  # noqa: E402

HEADER = ROOT / "woof" / "core" / "kernels" / "glibc_flt32.cuh"
FAST = ROOT / "woof" / "backend" / "metal" / "gfk_fast.h"


def test_the_header_carries_the_generators_constants():
    text = FAST.read_text()
    begin = text.index(constants.BEGIN)
    end = text.index(constants.END) + len(constants.END)
    assert text[begin:end] == constants.block()


def test_the_constants_sum_to_the_binary64_values():
    tables = constants.tables()
    text = HEADER.read_text(encoding="utf-8")
    for name in ("GFK_LOGF_INVC", "GFK_LOGF_LOGC", "GFK_POWF_INVC", "GFK_POWF_LOGC"):
        exact = constants._array(text, name)
        hi, lo = tables[name + "_HI"], tables[name + "_LO"]
        assert all(abs((h + l) - d) <= abs(d) * 2.0 ** -48 for h, l, d in zip(hi, lo, exact))
    hi, lo, lo2 = tables["GFK_EXPF_INVLN2N"]
    assert hi + lo + lo2 == float.fromhex("0x1.71547652b82fep+0") * 32.0


needs_clang = pytest.mark.skipif(shutil.which("clang") is None and shutil.which("clang++") is None,
                                 reason="the translator preprocesses with clang")


@needs_clang
def test_the_translator_routes_the_three_routines(monkeypatch):
    from woof.backend import translate
    monkeypatch.delenv(translate.GFK_FAST_ENV, raising=False)
    src = translate.preprocess(HEADER.read_text(encoding="utf-8"))
    out = translate.gfk_fast(src)
    for name, params, args in translate._GFK_SHORTCUTS:
        assert f"float {name}_soft({params}) {{" in out
        assert f"unsigned int __b = __woof_{name}({args});" in out
    assert out.count("__woof_gfk_") == 3
    monkeypatch.setenv(translate.GFK_FAST_ENV, "0")
    assert translate.gfk_fast(src) == src


@needs_clang
def test_only_glibc_definitions_are_rerouted(monkeypatch):
    """A same-signature routine that is not glibc_flt32.cuh's, or glibc's
    body over a different table, keeps its own words: no wrapper."""
    from woof.backend import translate
    monkeypatch.delenv(translate.GFK_FAST_ENV, raising=False)
    impostor = "float gfk_log(float x) { return x; }\nfloat gfk_exp(float x) { return x; }\n"
    assert translate.gfk_fast(impostor) == impostor
    src = translate.preprocess(HEADER.read_text(encoding="utf-8"))
    head, sep, tail = src.partition("float gfk_log(float x)")
    body = tail[:tail.index("\n}\n") + 3]
    swapped = src.replace(sep + body, "float gfk_log(float x)\n{\n    return x;\n}\n")
    out = translate.gfk_fast(swapped)
    assert "gfk_log_soft" not in out and "__woof_gfk_exp(" in out and "__woof_gfk_pow(" in out
    # one table word changed: the routines reading GFK_EXP2F_TAB keep their bodies
    table = src.index("GFK_EXP2F_TAB")
    first = re.compile(r"0x[0-9a-fA-F]+").search(src, src.index("{", table))
    altered = src[:first.start()] + hex(int(first.group(), 16) ^ 1) + src[first.end():]
    out = translate.gfk_fast(altered)
    assert "__woof_gfk_log(" in out and "__woof_gfk_exp(" not in out and "__woof_gfk_pow(" not in out


needs_mlx = pytest.mark.skipif(importlib.util.find_spec("mlx") is None,
                               reason="needs mlx (Apple silicon)")


def _mlx_or_skip():
    from woof import backend
    if backend.active() != "mlx":
        pytest.skip("runs under WOOF_BACKEND=mlx")
    if os.environ.get("WOOF_METAL_GFK_FAST", "1").strip() == "0":
        pytest.skip("WOOF_METAL_GFK_FAST=0")


@needs_mlx
@pytest.mark.parametrize("contract", ["ftz", "noftz"])
def test_logf_and_expf_shortcuts_give_the_binary64_words(contract):
    _mlx_or_skip()
    import metal_gfk_fast_grade as grade
    module = grade.probe_module(grade.OPTION_SETS[contract])
    for which in (0, 1):
        # an odd stride walks every exponent and sign with varied mantissas
        bad, deferred, total, first, _ = grade.run_unary(module, which, stride=65_521)
        assert bad == 0, f"first differing input 0x{first:08x}"
        assert deferred < total
        # and densely through the range the physics uses: |x| in [2^-8, 2^8)
        for start in (0x3b800000, 0xbb800000):
            bad, _, _, first, _ = grade.run_unary(module, which, stride=7, start=start,
                                                  stop=start + 0x08000000)
            assert bad == 0, f"first differing input 0x{first:08x}"


@needs_mlx
@pytest.mark.parametrize("contract", ["ftz", "noftz"])
def test_powf_shortcut_gives_the_binary64_words(contract):
    _mlx_or_skip()
    import metal_gfk_fast_grade as grade
    module = grade.probe_module(grade.OPTION_SETS[contract])
    for y in grade.POW_EXPONENTS:
        bad, _, _, first = grade.run_pow_sweep(module, y, stride=4099)
        assert bad == 0, f"powf(0x{first:08x}, {y!r})"
    bad, deferred, total, pairs = grade.run_pow_random(module, 1 << 20, seed=7)
    assert bad == 0, pairs
    assert deferred < total // 2
