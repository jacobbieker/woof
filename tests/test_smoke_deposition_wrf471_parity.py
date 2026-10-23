"""Smoke particle removal against WRF v4.7.1 itself, word for word.

The fixtures are WRF's own Fortran, byte-unmodified and pinned by sha256,
compiled at -O0 by ``tools/smoke_wrf471_oracle/build.sh`` and run by its
drivers:

* ``zhang2001``       -- ``zhang2001_aer_drydepvel_1``
  (chem/module_aer_drydep.F:1650-1941) over every surface type (Zhang and
  USGS, with out-of-range ones), every season, dry and wet, eight diameters
  across both clamps, both ``zhang_optaa`` variants and every moment switch
  (``run_zhang2001.F90``);
* ``gocart_settling`` -- GOCART ``settling`` (chem/module_gocart_settling.F:
  206-367, REAL*8) fed as ``gocart_settling_driver`` feeds dust: smoke and
  dust bins, the UoC density override, time steps from under a second to
  an hour (the 12-substep cap) and negative mixing ratios
  (``run_gocart_settling.F90``).

Both the NumPy references and the device kernels must equal every output
word (max ULP 0).  The woof-own pieces (aerodynamic resistance, the
surface deposition tendency) have no WRF routine to equal: their two forms
equal each other at zero ULP and the float64 formula within a tolerance.
"""
from __future__ import annotations

import numpy as np
import pytest

from conftest import requires_gpu
from woof.smoke import deposition as dep
from woof.verify.smoke_oracle import (assert_words_equal, case_names,
                                      load_case, max_ulp)

ZHANG_CASES = case_names("zhang2001")
SETTLING_CASES = case_names("gocart_settling")
ZHANG_FIELDS = ("depvel_0", "depvel_3", "resist_s0", "resist_s3",
                "vsettl_0", "vsettl_3")


def _zhang_call(fn, d):
    args = [d[k] for k in ("dg_num", "lnsig", "dens_part", "airtemp",
                           "airdens", "ustar", "resist_a")]
    return fn(*args, isurftype=d["isurftype"], iseason=d["iseason"],
              iwetsurf=d["iwetsurf"], modesurftype=int(d["modesurftype"]),
              zhang_optaa=int(d["zhang_optaa"]),
              airkinvisc=d["airkinvisc_in"], freepath=d["freepath_in"],
              ido_mom0=int(d["ido_mom0"]), ido_mom3=int(d["ido_mom3"]))


def _zhang_words(result, to_host=np.asarray):
    out = {k: to_host(getattr(result, k)) for k in ZHANG_FIELDS}
    out["airkinvisc_out"] = to_host(result.airkinvisc)
    out["freepath_out"] = to_host(result.freepath)
    return out


def _settling_call(fn, d):
    """Fixture (nz, nbin, ncol) -> the API's (nbin, nz, ncol, 1)."""
    chem = np.ascontiguousarray(np.transpose(d["chem"], (1, 0, 2))[..., None])
    fields = [np.ascontiguousarray(d[k][..., None])
              for k in ("t_phy", "p_phy", "dz8w", "rho_phy")]
    return fn(chem, *fields, d["dt"], d["reff"], d["den"], g=float(d["g"]),
              dyn_visc=float(d["dyn_visc"]), uoc=int(d["uoc"]))


def _settling_words(result, to_host=np.asarray):
    tc = to_host(result.tc)[..., 0]
    return {"tc_out": np.ascontiguousarray(np.transpose(tc, (1, 0, 2))),
            "graset": to_host(result.graset)[..., 0],
            "grasetvel": to_host(result.grasetvel)[..., 0]}


def _get(array):
    return array.get()


# --------------------------------------------------------------------------
# the fixtures reach what they claim to reach


def test_zhang_fixtures_reach_every_branch():
    seen_types, seen_usgs, optaa, moments = set(), set(), set(), set()
    for case in ZHANG_CASES:
        d = load_case(f"zhang2001/{case}")
        if int(d["modesurftype"]) <= 1:
            seen_types |= set(d["isurftype"].tolist())
        else:
            seen_usgs |= set(d["isurftype"].tolist())
        optaa.add(int(d["zhang_optaa"]))
        moments.add((int(d["ido_mom0"]), int(d["ido_mom3"])))
        assert set(d["iwetsurf"].tolist()) == {0, 1}
        assert set(d["iseason"].tolist()) == set(range(-1, 7))
    assert set(range(-1, 17)) <= seen_types
    assert set(range(-1, 28)) <= seen_usgs
    assert optaa == {1, 2}
    assert moments == {(1, 1), (1, 0), (0, 1), (0, 0)}


def test_settling_fixtures_reach_the_substep_cap_and_the_reset():
    caps, zero_step, negatives, uoc = set(), False, False, set()
    for case in SETTLING_CASES:
        d = load_case(f"gocart_settling/{case}")
        uoc.add(int(d["uoc"]))
        zero_step |= int(np.float32(d["dt"])) == 0
        negatives |= bool((d["chem"] < 0).any())
        vsettl = (np.float64(np.float32(4.0) / np.float32(9.0)
                             * np.float32(d["g"])) * d["den"] * d["reff"] ** 2
                  / np.float64(d["dyn_visc"]))
        dzmin = d["dz8w"].astype(np.float64).min(axis=0)
        ratio = int(np.float32(d["dt"])) / (dzmin[None, :] / vsettl[:, None])
        caps.add(bool((ratio >= 13).any()))
    assert caps == {True, False}
    assert zero_step and negatives and uoc == {0, 1}


# --------------------------------------------------------------------------
# CPU references against the compiled WRF


@pytest.mark.parametrize("case", ZHANG_CASES)
def test_zhang2001_reference_is_wrfs_word_for_word(case):
    d = load_case(f"zhang2001/{case}")
    got = _zhang_words(_zhang_call(dep.zhang2001_depvel_np, d))
    for name, words in got.items():
        assert_words_equal(words, d[name], f"zhang2001/{case}/{name}")


@pytest.mark.parametrize("case", SETTLING_CASES)
def test_gocart_settling_reference_is_wrfs_word_for_word(case):
    d = load_case(f"gocart_settling/{case}")
    got = _settling_words(_settling_call(dep.gocart_settling_np, d))
    for name, words in got.items():
        assert words.dtype == np.float64
        assert_words_equal(words, d[name], f"gocart_settling/{case}/{name}")


def test_single_bin_settling_equals_its_bin_in_a_stack():
    d = load_case(f"gocart_settling/{SETTLING_CASES[0]}")
    stacked = _settling_call(dep.gocart_settling_np, d)
    chem = np.ascontiguousarray(
        np.transpose(d["chem"], (1, 0, 2))[1][..., None])
    fields = [d[k][..., None] for k in ("t_phy", "p_phy", "dz8w", "rho_phy")]
    one = dep.gocart_settling_np(
        chem, *fields, d["dt"], d["reff"][1], d["den"][1], g=float(d["g"]),
        dyn_visc=float(d["dyn_visc"]))
    assert max_ulp(one.tc, stacked.tc[1]) == 0
    assert max_ulp(one.graset, stacked.graset[1]) == 0


# --------------------------------------------------------------------------
# woof-own pieces: float64 formula within a tolerance


def _resistance_inputs():
    rng = np.random.default_rng(4)
    shape = (7, 9)
    ustar = (0.05 + rng.random(shape)).astype(np.float32)
    z0 = (1e-3 + rng.random(shape)).astype(np.float32)
    zref = (z0 * (2.0 + 40.0 * rng.random(shape))).astype(np.float32)
    length = (np.where(rng.random(shape) < 0.5, -1.0, 1.0)
              * (2.0 + 800.0 * rng.random(shape))).astype(np.float32)
    length[0, :3] = np.inf
    return ustar, z0, zref, length


def test_aerodynamic_resistance_reference_is_its_float64_formula():
    ustar, z0, zref, length = _resistance_inputs()
    got = dep.aerodynamic_resistance_np(ustar, z0, zref, length)
    want = dep.aerodynamic_resistance_f64(ustar, z0, zref, length)
    assert got.dtype == np.float32
    np.testing.assert_allclose(got, want, rtol=2e-5, atol=1e-6)
    neutral = dep.aerodynamic_resistance_np(0.3, 0.05, 12.5)
    assert float(neutral) == pytest.approx(np.log(250.0) / 0.12, rel=1e-6)
    stable = dep.aerodynamic_resistance_np(0.3, 0.05, 12.5, 50.0)
    unstable = dep.aerodynamic_resistance_np(0.3, 0.05, 12.5, -50.0)
    assert float(unstable) < float(neutral) < float(stable)


def test_surface_tendency_reference_is_its_float64_formula():
    rng = np.random.default_rng(5)
    q0 = rng.random((5, 6)).astype(np.float32) * 1e-8
    dz0 = (20.0 + 40.0 * rng.random((5, 6))).astype(np.float32)
    vd = (1e-3 + 4e-3 * rng.random((5, 6))).astype(np.float32)
    rho0 = (1.0 + 0.2 * rng.random((5, 6))).astype(np.float32)
    got = dep.surface_deposition_tendency_np(q0, dz0, vd, rho0)
    q, z, v, r = (a.astype(np.float64) for a in (q0, dz0, vd, rho0))
    np.testing.assert_allclose(got.tendency, -v * q / z, rtol=1e-6)
    np.testing.assert_allclose(got.flux, r * v * q, rtol=1e-6)
    # Column mass: rho0 dz0 tendency is minus the flux.
    np.testing.assert_allclose(r * z * got.tendency.astype(np.float64),
                               -got.flux.astype(np.float64), rtol=1e-6)


def test_smoke_mode_deposits_inside_the_measured_range():
    """PM2.5 smoke (number median 0.15 um, sigma 1.6) over grass and forest
    at u* = 0.4: a few tenths of a cm/s, inside 0.05-1 cm/s."""
    ra = dep.aerodynamic_resistance_np(0.4, np.array([0.05, 0.8]), 25.0)
    out = dep.zhang2001_depvel_np(0.15e-6, np.log(1.6), 1400.0, 293.0, 1.19,
                                  0.4, ra, isurftype=np.array([6, 1]),
                                  iseason=1)
    assert np.all((out.depvel_3 > 5e-4) & (out.depvel_3 < 1e-2))
    assert np.all(out.vsettl_3 < 1e-4)


# --------------------------------------------------------------------------
# the device kernels against the compiled WRF and the references


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("case", ZHANG_CASES)
def test_gpu_zhang2001_is_wrfs_and_the_references_word_for_word(case):
    import cupy  # noqa: F401  (device test)

    d = load_case(f"zhang2001/{case}")
    gpu = _zhang_words(_zhang_call(dep.zhang2001_depvel, d), _get)
    cpu = _zhang_words(_zhang_call(dep.zhang2001_depvel_np, d))
    for name, words in gpu.items():
        assert words.dtype == np.float32
        assert_words_equal(words, d[name], f"gpu zhang2001/{case}/{name}")
        assert_words_equal(words, cpu[name],
                           f"gpu-cpu zhang2001/{case}/{name}")


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("case", SETTLING_CASES)
def test_gpu_gocart_settling_is_wrfs_and_the_references_word_for_word(case):
    import cupy  # noqa: F401  (device test)

    d = load_case(f"gocart_settling/{case}")
    gpu = _settling_words(_settling_call(dep.gocart_settling, d), _get)
    cpu = _settling_words(_settling_call(dep.gocart_settling_np, d))
    for name, words in gpu.items():
        assert_words_equal(words, d[name], f"gpu gocart/{case}/{name}")
        assert_words_equal(words, cpu[name], f"gpu-cpu gocart/{case}/{name}")


@pytest.mark.gpu
@requires_gpu
def test_gpu_woof_own_pieces_equal_their_references():
    import cupy as cp

    ustar, z0, zref, length = _resistance_inputs()
    got = dep.aerodynamic_resistance(cp.asarray(ustar), cp.asarray(z0),
                                     cp.asarray(zref), cp.asarray(length))
    assert_words_equal(got.get(), dep.aerodynamic_resistance_np(
        ustar, z0, zref, length), "resist_a")
    neutral = dep.aerodynamic_resistance(ustar, z0, zref)
    assert_words_equal(neutral.get(), dep.aerodynamic_resistance_np(
        ustar, z0, zref), "resist_a neutral")
    q0 = (ustar * 1e-8).astype(np.float32)
    out = dep.surface_deposition_tendency(cp.asarray(q0), zref, 3e-3, 1.1)
    ref = dep.surface_deposition_tendency_np(q0, zref, 3e-3, 1.1)
    assert_words_equal(out.tendency.get(), ref.tendency, "tendency")
    assert_words_equal(out.flux.get(), ref.flux, "flux")


@pytest.mark.gpu
@requires_gpu
def test_gpu_whole_domain_shapes_round_trip():
    """(nz, ny, nx) fields in, one launch, the same shapes out."""
    import cupy as cp

    d = load_case(f"gocart_settling/{SETTLING_CASES[0]}")
    nz, nbin, ncol = d["chem"].shape
    chem = np.ascontiguousarray(np.transpose(d["chem"], (1, 0, 2)))[0]
    chem = np.stack([chem] * 3, axis=-1)            # (nz, ncol, 3)
    fields = [np.stack([d[k]] * 3, axis=-1)
              for k in ("t_phy", "p_phy", "dz8w", "rho_phy")]
    gpu = dep.gocart_settling(cp.asarray(chem), *fields, d["dt"],
                              d["reff"][0], d["den"][0])
    cpu = dep.gocart_settling_np(chem, *fields, d["dt"], d["reff"][0],
                                 d["den"][0])
    assert gpu.tc.shape == (nz, ncol, 3) and gpu.graset.shape == (ncol, 3)
    assert_words_equal(gpu.tc.get(), cpu.tc, "tc")
    assert_words_equal(gpu.graset.get(), cpu.graset, "graset")
