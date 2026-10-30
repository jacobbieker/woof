"""woof.radtran against libRadtran's uvspec (black-box oracle fixtures).

Fixtures: tests/data/oracles/uvspec/<case>.json, written by
``python -m tools.uvspec_oracle.generate`` (uvspec 2.0.6 from the pixi
``libradtran`` environment; receipt.json records version, build and data
hashes).  The inputs are rebuilt here from the user's libRadtran data
(tools/uvspec_oracle/cases.py), never stored in woof; woof runs its solver's
host build (bit-identical to the device builds, tests/test_radtran_kernel_words*.py).

Tolerances are MEASURED, not targets: each is about twice the agreement
measured on 2026-10-30 (macOS arm64), with the reason the agreement is not
tighter.  uvspec prints irradiances with 7 significant digits and radiances
with 10, so 5e-7 is the print floor for irradiance.
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import numpy as np
import pytest

FIX = Path(__file__).parent / "data" / "oracles" / "uvspec"

PRINT = (1.5e-6, "uvspec prints 7 significant digits (quantum 5e-7); measured <= 6.6e-7")

#: case -> {quantity: (relative tolerance, reason)}; quantities not listed use PRINT
TOL = {
    "o3_huggins": {q: (1.6e-5, "uvspec interpolates the O3/NO2 cross-section files at a single-precision "
                              "representative wavelength (woof: binary64); with a float32 wavelength woof "
                              "agrees to 1.5e-6; measured 8.1e-6")
                   for q in ("edir", "edn", "eup", "uavgdir", "uavgdn", "uavgup")},
    "model_h2o": {q: (6e-6, "uvspec holds altitudes in single precision; model-level altitudes are not "
                           "float32-exact, so its layer thicknesses differ by ~1e-6 relative; measured 2.7e-6")
                  for q in ("edir", "edn", "eup", "uavgdir", "uavgdn", "uavgup")},
    "wc_hu_500": {q: (3e-5, "optically thick cloud (tau ~ 31+20): the diffuse field moves by 1-2e-5 when the "
                           "layer single-scattering albedo is rounded to binary32 (uvspec's internal optical "
                           "properties are single precision); measured 1.4e-5")
                  for q in ("edn", "eup", "uavgdn", "uavgup")},
    "model_wc": {q: (4e-5, "thick cloud, as wc_hu_500 (binary32 single-scattering albedo in uvspec); "
                          "measured 1.9e-5")
                 for q in ("edn", "eup", "uavgdn", "uavgup")},
    "twostr_wc": {q: (3.5e-3, "uvspec's twostr departs from the two-stream equations' exact solution for "
                             "thick clouds (gap grows ~tau^2: 4e-6 at tau 2.5, 1.7e-3 at tau 50); woof's "
                             "two-stream reproduces the closed form (Meador & Weaver 1980) to 1e-11 "
                             "(test_radtran_reference.py); measured 1.7e-3")
                  for q in ("edn", "eup")},
}
#: pseudo-spherical direct beam: uvspec reports a direct irradiance that is
#: not exp(-slant tau) of the beam its own solver deposits (its diffuse
#: fields match woof's to <=6e-6 and an energy check shows the reported
#: direct beam inconsistent with them).  woof reports its solver's beam
#: (Dahlback & Stamnes homogeneous shells).  Measured: 1.4e-5 (60 deg),
#: 2.2e-3 (85 deg), 2.8e-2 (89 deg).
_SPH = "uvspec's reported pseudo-spherical direct beam differs from its solver's own beam (docs/dev/radtran.md)"
for _c, _t in (("sph_sza60", 3e-5), ("sph_sza85", 4.5e-3), ("sph_sza89", 6e-2), ("twostr_sph85", 4.5e-3),
               ("rad16_sph85", 4.5e-3)):
    TOL.setdefault(_c, {}).update({"edir": (_t, _SPH), "uavgdir": (_t, _SPH)})
TOL["sph_sza89"].update({q: (1.4e-5, "SZA 89: diffuse field measured 6.4e-6 (layer secants near grazing)")
                         for q in ("edn", "eup", "uavgdn", "uavgup")})
TOL["twostr_sph85"].update({q: (3e-6, "two-stream pseudo-spherical diffuse field; measured 1.3e-6")
                            for q in ("edn", "eup")})

RAD_TOL = (1e-6, "uvspec prints radiances with 10 digits; measured <= 4.7e-7")
#: rad16_wc_moments: uvspec's disort_intcor moments applies Nakajima &
#: Tanaka's IMS (secondary-scattering) correction as well as TMS; woof
#: implements TMS only.  The IMS acts on downward radiance near the solar
#: almucantar (umu = -0.9 with umu0 = 0.866): measured 1.6e-2 there,
#: <= 2.4e-6 elsewhere.
IMS_ROWS = {"rad16_wc_moments": ([0], 3.5e-2)}


def _cases():
    return sorted(p.stem for p in FIX.glob("*.json") if p.stem != "receipt")


def _need_data():
    from woof.radtran import data
    try:
        data.data_dir()
    except data.LibradtranDataMissing as exc:
        pytest.skip(str(exc))


@pytest.mark.parametrize("name", _cases())
def test_radtran_matches_uvspec(name):
    _need_data()
    from tools.uvspec_oracle import cases
    from tools.uvspec_oracle.generate import inputs_digest
    from woof.radtran import driver
    rec = json.loads((FIX / f"{name}.json").read_text())
    opts, col, cl = cases.build(name)
    if inputs_digest(col, cl) != rec["inputs_sha256"]:
        warnings.warn(f"{name}: rebuilt inputs differ in some bit from the fixture's (platform libm); "
                      "the comparison still holds at the stated tolerances")
    r = driver.compute(opts, col, cl, backend="host")
    uv = {k: np.asarray(v, dtype=np.float64) for k, v in rec["uvspec"].items() if v is not None}
    assert np.allclose(r.wavelength, uv["lambda"], atol=6e-4), "band centres"
    quantities = (("edir", "edn", "eup") if opts.rte_solver == "twostr"
                  else ("edir", "edn", "eup", "uavgdir", "uavgdn", "uavgup"))
    tol = TOL.get(name, {})
    for q in quantities:
        a = getattr(r, q)[0].T            # (nband, nout)
        b = uv[q]
        big = np.abs(b) > 1e-6 * np.abs(b).max()
        rel = np.abs(a - b)[big] / np.abs(b)[big]
        t, reason = tol.get(q, PRINT)
        assert rel.max() <= t, f"{name} {q}: max relative error {rel.max():.2e} > {t:.1e} ({reason})"
    if opts.umu:
        a = np.moveaxis(r.uu[0], -1, 0)   # (nband, nout, numu, nphi)
        b = uv["uu"]
        rows = np.ones(len(opts.umu), bool)
        if name in IMS_ROWS:
            ims, t_ims = IMS_ROWS[name]
            rows[ims] = False
            m = np.abs(b[:, :, ims]) > 1e-8
            rel = np.abs(a[:, :, ims] - b[:, :, ims])[m] / np.abs(b[:, :, ims])[m]
            assert rel.max() <= t_ims, f"{name} IMS rows: {rel.max():.2e}"
        m = np.abs(b[:, :, rows]) > 1e-8
        rel = np.abs(a[:, :, rows] - b[:, :, rows])[m] / np.abs(b[:, :, rows])[m]
        t = RAD_TOL[0] if name not in IMS_ROWS else 5e-6
        assert rel.max() <= t, f"{name} radiance: max relative error {rel.max():.2e} > {t:.1e}"


def test_receipt_names_the_oracle():
    r = json.loads((FIX / "receipt.json").read_text())
    assert r["uvspec_version"].startswith("2.0.6")
    assert r["package"]["name"] == "rubin-libradtran"
    assert set(r["cases"]) == set(_cases())
    assert all(len(h) == 64 for h in r["data_sha256"].values())


def test_sun_distance_is_spencer_1971():
    from woof.radtran.driver import sun_distance_factor
    assert sun_distance_factor(None) == 1.0
    assert math.isclose(sun_distance_factor(3), 1.0342, rel_tol=1e-3)
    assert math.isclose(sun_distance_factor(185), 0.9667, rel_tol=1e-3)
