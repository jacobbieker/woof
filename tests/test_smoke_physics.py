"""woof.smoke: the injection profiles, and the package as a whole.

The injection profiles (``woof.smoke.injection``) are woof's own, with no
WRF counterpart: their CuPy kernels and NumPy references agree at zero ULP,
and both are held to a float64 evaluation of the same formulas and to the
column normalisation within float32 rounding.  The WRF ports have their own
oracle parity files (``test_smoke_*_wrf471_parity.py``).
"""
from __future__ import annotations

import importlib

import numpy as np
import pytest

from conftest import requires_gpu
from woof.smoke import injection as inj
from woof.verify import smoke_oracle as so

PBL = 1500.0


def _columns(seed: int = 3, ny: int = 3, nx: int = 4):
    """Stretched interfaces (25 m, 8 % growth, 400 m cap), perturbed."""
    rng = np.random.default_rng(seed)
    dz = np.minimum(25.0 * 1.08 ** np.arange(50), 400.0)
    zw1 = np.concatenate([[0.0], np.cumsum(dz)])
    z_w = (zw1[:, None, None] * (1.0 + 0.1 * rng.random((ny, nx)))
           ).astype(np.float32)
    zm = 0.5 * (z_w[:-1] + z_w[1:])
    rho = (1.2 * np.exp(-zm / 8500.0)).astype(np.float32)
    flux = (1e-6 + 1e-4 * rng.random((ny, nx))).astype(np.float32)
    return z_w, rho, flux


def _uniform_f64(z_w, rho, flux, bot, top):
    """The same formulas in float64, column by column."""
    z_w, rho = np.float64(z_w), np.float64(rho)
    cols = z_w.shape[1:]
    out = np.zeros(rho.shape)
    for col in np.ndindex(cols):
        zw = z_w[(slice(None),) + col]
        b = min(max(float(np.broadcast_to(bot, cols)[col]), zw[0]), zw[-1])
        t = min(max(float(np.broadcast_to(top, cols)[col]), zw[0]), zw[-1])
        f = float(np.broadcast_to(flux, cols)[col])
        r = rho[(slice(None),) + col]
        if t > b:
            for k in range(zw.size - 1):
                ov = min(zw[k + 1], t) - max(zw[k], b)
                if ov > 0:
                    out[(k,) + col] = (f * ov / (t - b)
                                       / (r[k] * (zw[k + 1] - zw[k])))
        else:
            k = next((k for k in range(zw.size - 1)
                      if zw[k] <= b < zw[k + 1]), zw.size - 2)
            out[(k,) + col] = f / (r[k] * (zw[k + 1] - zw[k]))
    return out


def _integral(source, z_w, rho):
    return np.sum(np.float64(source) * np.float64(rho)
                  * np.diff(np.float64(z_w), axis=0), axis=0)


INTERVALS = [(0.0, 1500.0), (812.3, 1673.9), (50.0, 1e5), (35.0, 35.0),
             (1e5, 90.0), (-5.0, 0.0), (0.0, 10.0)]


@pytest.mark.parametrize("bot,top", INTERVALS)
def test_uniform_profile_is_its_float64_formula_and_integrates(bot, top):
    z_w, rho, flux = _columns()
    got = inj.uniform_profile_np(z_w, rho, flux, bot, top)
    # A partial layer's overlap is a difference of two heights of up to a
    # few km, each a float32: 1 ULP of 1.7 km on a 15 m overlap is 8e-6.
    np.testing.assert_allclose(got, _uniform_f64(z_w, rho, flux, bot, top),
                               rtol=2e-5, atol=0)
    np.testing.assert_allclose(_integral(got, z_w, rho), flux, rtol=5e-6)


def test_lowest_layer_and_pbl_profiles():
    z_w, rho, flux = _columns()
    low = inj.lowest_layer_profile_np(z_w, rho, flux)
    assert np.all(low[1:] == 0)
    np.testing.assert_allclose(_integral(low, z_w, rho), flux, rtol=1e-6)
    for fraction in (1.0, 0.5, 0.0):
        got = inj.pbl_uniform_profile_np(z_w, rho, flux,
                                         np.full(flux.shape, PBL), fraction)
        np.testing.assert_allclose(_integral(got, z_w, rho), flux,
                                   rtol=5e-6)
        if fraction == 0.0:
            so.assert_words_equal(got, low, "zero-depth PBL = lowest layer")
        else:
            assert np.all(got[z_w[:-1] >= fraction * PBL] == 0)
    with pytest.raises(ValueError):
        inj.pbl_uniform_profile_np(z_w, rho, flux, PBL, fraction=1.5)
    with pytest.raises(ValueError, match="increase"):
        inj.uniform_profile_np(z_w[::-1], rho, flux, 0.0, 10.0)


def test_oracle_fixtures_are_the_pinned_wrf_and_their_own_receipts():
    """Every published word is the one build.sh wrote, from pinned sources,
    by the harness now in tools/smoke_wrf471_oracle."""
    import hashlib

    pins = [line.split() for line in
            (so.TOOL / "SOURCES.sha256").read_text().splitlines()
            if line and not line.startswith("#")]
    recorded = [line.split() for line in
                (so.ORACLE / "SOURCES.sha256").read_text().splitlines() if line]
    assert recorded == pins
    listed = set()
    for line in (so.ORACLE / "FIXTURES.sha256").read_text().splitlines():
        digest, rel = line.split()
        listed.add(rel.removeprefix("./"))
        assert hashlib.sha256((so.ORACLE / rel).read_bytes()).hexdigest() \
            == digest, rel
    on_disk = {p.relative_to(so.ORACLE).as_posix()
               for p in so.ORACLE.rglob("*")
               if p.is_file() and (p.suffix == ".bin"
                                   or p.name == "MANIFEST.txt")}
    assert on_disk == listed
    for line in (so.ORACLE / "HARNESS.sha256").read_text().splitlines():
        digest, rel = line.split()
        assert hashlib.sha256((so.TOOL / rel).read_bytes()).hexdigest() \
            == digest, rel
    report = (so.ORACLE / "libmvec-report.txt").read_text()
    head = report.split("at WRF's -O2")[0]
    assert "_ZGV" not in head        # the -O0 reference took no vector libm
    assert "_ZGVbN4v_expf" in report  # and the guard can see one


def test_word_comparison_sees_signed_zero_and_one_ulp():
    for dtype in (np.float32, np.float64):
        zero = np.zeros(1, dtype)
        assert so.max_ulp(-zero, zero) == 1
        tiny = np.array([np.nextafter(dtype(0), dtype(1))])
        assert so.max_ulp(-tiny, -zero) == 1 and so.max_ulp(-tiny, tiny) == 3
        one = np.ones(1, dtype)
        assert so.max_ulp(np.nextafter(one, dtype(2)), one) == 1
        with pytest.raises(AssertionError):
            so.assert_words_equal(-zero, zero, "signed zero")


def test_package_imports_nothing_and_its_modules_import():
    import woof.smoke as smoke

    # Its only non-dunder attributes are submodules someone imported.
    import types
    assert all(isinstance(value, types.ModuleType)
               for name, value in vars(smoke).items()
               if not name.startswith("__"))
    for name in ("emission", "injection", "plume_rise", "deposition",
                 "_kernels"):
        importlib.import_module(f"woof.smoke.{name}")


@pytest.mark.gpu
@requires_gpu
def test_the_smoke_unit_compiles_within_its_priced_frame():
    import re

    import cupy as cp

    from woof.core import preflight as pf
    from woof.smoke import _kernels

    priced = pf.CHAINED_TRANSLATION_UNIT_FRAMES["smoke_composed"]
    module = _kernels._module(cp.cuda.Device().id)
    names = set(re.findall(r'extern "C" __global__ void (\w+)',
                           _kernels.module_source()))
    assert len(names) == 9, sorted(names)
    frames = {name: int(cp.cuda.driver.funcGetAttribute(
        cp.cuda.driver.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
        module.get_function(name).ptr)) for name in names}
    assert max(frames.values()) <= priced.max_local_size_bytes, frames


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("bot,top", INTERVALS)
def test_device_injection_is_the_reference_word_for_word(bot, top):
    import cupy as cp

    z_w, rho, flux = _columns(seed=7, ny=5, nx=6)
    d = [cp.asarray(a) for a in (z_w, rho, flux)]
    so.assert_words_equal(inj.uniform_profile(*d, bot, top).get(),
                          inj.uniform_profile_np(z_w, rho, flux, bot, top),
                          f"uniform [{bot}, {top}]")
    so.assert_words_equal(inj.lowest_layer_profile(*d).get(),
                          inj.lowest_layer_profile_np(z_w, rho, flux),
                          "lowest layer")
    pbl = np.linspace(0.0, 3000.0, flux.size).reshape(flux.shape)
    for fraction in (1.0, 0.7, 0.0):
        so.assert_words_equal(
            inj.pbl_uniform_profile(*d, cp.asarray(pbl), fraction).get(),
            inj.pbl_uniform_profile_np(z_w, rho, flux, pbl, fraction),
            f"pbl x {fraction}")
