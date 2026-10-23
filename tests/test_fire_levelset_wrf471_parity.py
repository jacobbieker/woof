"""The fire model, host and device, against WRF v4.7.1 itself: max ULP 0.

The fixture is WRF's own Fortran -- ``phys/module_fr_fire_{core,phys,util}.F``
byte-unmodified and pinned by sha256 -- compiled at -O0 with REAL(4) by
``tools/fire_levelset_wrf471_oracle/build.sh`` and run by its
``run_fire.F90``, which calls WRF's routines in ``fire_model``'s order of
passes 4, 5 and 6 and the driver's aggregation (PROVENANCE.md beside the
fixtures has the toolchain).  The cases reach every arm the port carries:

* ``point_iso`` -- an isotropic point fire (WENO5 band, viscosity ramp,
  reinitialization, the ignition growing out to its radius);
* ``line_wind`` -- a line walked from inside the ENO1 edge band, a varying
  wind and slope (fire_advection = 1) and a no-fuel patch the fire stops at;
* ``substep`` -- the same fire in outer steps of four fire steps, with the
  outer step's mean atmospheric flux;
* ``options`` -- fire_advection = 0, no reinitialization, two ignitions, one
  starting later and held for 5 s;
* ``cap`` -- fire_ros's 6 m/s cap taken out of the wind and slope parts;
* ``asym`` -- a 48 x 40 mesh with dx = 10 m, dy = 8 m, and every numeric
  switch the port exposes off its Registry value (viscosity and its band
  and ramp, the WENO band, slope factor, feedback, two reinitialization
  iterations, a half-biased edge extrapolation);
* ``weight_7`` ... ``weight_900`` -- fuel_left across WRF's fuel weights.

Every dumped field at every dumped step is compared by raw float32 word, on
the host reference and on the device, and the device against the host.  No
output is held out (:data:`HELD_OUT`).
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "data" / "oracles" / "fire_levelset"
HARNESS = ROOT / "tools" / "fire_levelset_wrf471_oracle"

CASES = ("point_iso", "line_wind", "substep", "options", "cap", "asym",
         "weight_7", "weight_30", "weight_180", "weight_900")
#: Every array the oracle dumps per step, all of them compared.
FIELDS = ("time", "tbound", "lfn", "tign", "ros", "fire_area", "fuel_frac",
          "fuel_frac_burnt", "fgrnhfx", "fgrnqfx", "GRNHFX", "GRNQFX",
          "CANHFX", "CANQFX", "AVG_FUEL_FRAC", "mean_GRNHFX", "mean_GRNQFX")
#: Outputs that cannot be bit-exact against WRF, with the reason in
#: PROVENANCE.md.  There are none.
HELD_OUT: frozenset = frozenset()
_INPUTS = ("vx", "vy", "dzdxf", "dzdyf", "r_0", "bbb", "phiwc", "betafl",
           "fgip", "fuel_time", "fmc_g")


def _case(name: str) -> dict:
    from woof.verify.urban_oracle import load_case
    from woof.verify.wrf471_fixtures import require_fixture_dir

    return load_case(require_fixture_dir(FIXTURES, "fire level-set") / name)


def _int(case, key) -> int:
    return int(np.asarray(case[key]).reshape(()))


def _replay(name: str, xp) -> tuple[dict, dict]:
    """``(port, oracle)``: ``{(step, field): host array}`` at every dump."""
    from woof.fire.levelset import (FireMesh, FireOptions, FireParams,
                                    Ignition, advance, initial_state)

    case = _case(name)
    dxa = float(np.asarray(case["dxa"]).reshape(()))
    mesh = FireMesh(_int(case, "nxa"), _int(case, "nya"), dxa, dxa,
                    _int(case, "ir"), _int(case, "jr"))
    # Fortran (i, j) -> the port's (j, i).
    inputs = {key: xp.asarray(np.ascontiguousarray(
        np.asarray(case["in_" + key]).T)) for key in _INPUTS}
    params = FireParams.build(mesh, xp=xp, **inputs)
    table = np.asarray(case["ignitions"]).reshape(8, -1, order="F")
    ignitions = [Ignition(start=(float(r[0]), float(r[1])),
                          end=(float(r[2]), float(r[3])),
                          start_time=float(r[4]), end_time=float(r[5]),
                          radius=float(r[6]), ros=float(r[7]))
                 for r in table.T]
    o = [float(v) for v in np.asarray(case["options"], dtype=np.float32)]
    options = FireOptions(
        fire_advection=_int(case, "advection"),
        fire_lsm_reinit=bool(_int(case, "reinit")),
        fire_viscosity=o[0], fire_viscosity_bg=o[1],
        fire_viscosity_band=o[2], fire_viscosity_ngp=int(o[3]),
        fire_lsm_band_ngp=int(o[4]), fire_slope_factor=o[5],
        fire_atm_feedback=o[6], fire_lsm_reinit_iter=int(o[7]),
        fire_lfn_ext_up=o[8])
    dt = np.float32(np.asarray(case["dt"]).reshape(()))
    dumps = {int(d) for d in np.atleast_1d(case["dumps"])}
    host = getattr(xp, "asnumpy", np.asarray)
    state = initial_state(mesh, xp=xp)
    port, oracle = {}, {}
    for step in range(1, _int(case, "nsteps") + 1):
        state, last, mean = advance(state, mesh, params, dt,
                                    ignitions=ignitions, options=options,
                                    substeps_count=_int(case, "nsub"))
        if step not in dumps:
            continue
        got = {"time": np.float32(state.time), "lfn": state.lfn,
               "tign": state.tign, "fuel_frac": state.fuel_frac,
               "mean_GRNHFX": mean["GRNHFX"], "mean_GRNQFX": mean["GRNQFX"]}
        for field in FIELDS:
            value = got[field] if field in got else getattr(last, field)
            port[step, field] = np.asarray(host(value), dtype=np.float32)
            want = np.asarray(case[f"s{step}_{field}"], dtype=np.float32)
            oracle[step, field] = want.T if want.ndim == 2 else want
    return port, oracle


def _mismatches(port: dict, oracle: dict) -> dict:
    """``{(step, field): (words differing, max ULP)}`` for every miss."""
    from woof.core.fp32_ulp import fp32_ulp_distance

    out = {}
    for key, want in oracle.items():
        got = np.ascontiguousarray(port[key], dtype=np.float32)
        want = np.ascontiguousarray(want, dtype=np.float32)
        assert got.shape == want.shape, (key, got.shape, want.shape)
        differ = int(np.count_nonzero(got.view(np.uint32)
                                      != want.view(np.uint32)))
        if differ:
            ulp = int(np.max(np.asarray(fp32_ulp_distance(got, want))))
            out[key] = (differ, ulp)
    return out


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# the fixture is what the pinned harness wrote
# ---------------------------------------------------------------------------

def test_fixture_files_match_their_recorded_hashes():
    lines = (FIXTURES / "FIXTURES.sha256").read_text().splitlines()
    assert len(lines) > 100
    for line in lines:
        digest, rel = line.split(maxsplit=1)
        assert _sha256(FIXTURES / rel) == digest, rel


def test_fixture_was_written_by_this_harness_from_the_pinned_sources():
    pins = [line for line in (HARNESS / "SOURCES.sha256").read_text()
            .splitlines() if line and not line.startswith("#")]
    assert (FIXTURES / "SOURCES.sha256").read_text().splitlines() == pins
    for line in (FIXTURES / "HARNESS.sha256").read_text().splitlines():
        digest, rel = line.split(maxsplit=1)
        assert _sha256(HARNESS / rel) == digest, (
            f"{rel} changed after the fixture was generated: rebuild it "
            "with tools/fire_levelset_wrf471_oracle/build.sh")


def test_every_dumped_field_is_compared_and_none_is_held_out():
    case = _case("point_iso")
    dumped = {key.split("_", 1)[1] for key in case if key.startswith("s1_")}
    assert dumped == set(FIELDS)
    assert HELD_OUT == frozenset()


def test_the_cases_reach_the_arms_they_are_for():
    """A case that stopped reaching its arm would pass vacuously."""
    line = _case("line_wind")
    lfn = np.asarray(line["s140_lfn"])
    r0 = np.asarray(line["in_r_0"])
    assert np.count_nonzero(lfn[:10, :] <= 0) > 0       # the ENO1 edge band
    assert np.count_nonzero(np.abs(lfn) < 40) > 100     # the WENO5 band
    near = (r0 == 0) & (lfn < 20)                        # at the patch ...
    assert np.count_nonzero(near) > 0
    assert np.count_nonzero((r0 == 0) & (lfn <= 0)) == 0  # ... which holds
    cap = _case("cap")
    assert float(np.max(cap["s25_ros"])) == pytest.approx(6.0, abs=1e-5)
    weights = [float(np.min(_case(f"weight_{w}")["s40_fuel_frac"]))
               for w in (7, 30, 180, 900)]
    assert weights == sorted(weights) and weights[0] < 1e-3 < weights[-1]
    options = _case("options")
    assert np.count_nonzero(np.asarray(options["s14_lfn"]) <= 0) < \
        np.count_nonzero(np.asarray(options["s17_lfn"]) <= 0)
    asym = _case("asym")
    assert (_int(asym, "nx"), _int(asym, "ny")) == (48, 40)
    assert _int(asym, "ir") != _int(asym, "jr")
    assert np.asarray(asym["options"]).tolist() != np.asarray(
        line["options"]).tolist()
    final = np.asarray(asym["s120_lfn"])
    assert np.all(np.isfinite(final)) and np.count_nonzero(final <= 0) > 100


def test_the_recorded_frames_are_bound_to_the_unit_that_compiles():
    from woof.core import kernel_frame_recordings as recordings
    from woof.fire import levelset

    source = levelset.module_source()
    assert hashlib.sha256(source.encode()).hexdigest() == \
        recordings.FIRE_LEVELSET_COMPOSED_SOURCE_SHA256
    assert levelset.MODULE_OPTIONS == \
        recordings.FIRE_LEVELSET_COMPOSED_OPTIONS
    exports = set(re.findall(r"__global__ void (fl_\w+)", source))
    for row in recordings.FIRE_LEVELSET_COMPOSED_FRAME_READINGS.values():
        assert set(row) == exports


@pytest.mark.gpu
@requires_gpu
def test_fire_unit_frames_match_the_recorded_unit():
    import cupy as cp

    from woof.certify.compile_platform import compile_platform_fingerprint
    from woof.core import kernel_frame_recordings as recordings
    from woof.fire import levelset

    rows = recordings.FIRE_LEVELSET_COMPOSED_FRAME_READINGS
    module = levelset._module(cp.cuda.Device().id)
    names = next(iter(rows.values()))
    got = {name: int(cp.cuda.driver.funcGetAttribute(
        cp.cuda.driver.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
        module.get_function(name).ptr)) for name in names}
    fingerprint = compile_platform_fingerprint()
    key = (fingerprint["device_compute_capability"],
           fingerprint["nvrtc_build"])
    if key in rows:
        assert got == dict(rows[key]), (key, got)
    else:
        ceiling = max(max(row.values()) for row in rows.values())
        assert max(got.values()) <= ceiling, (key, got, ceiling)


# ---------------------------------------------------------------------------
# the host reference and the device, word for word
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", CASES)
def test_host_reference_equals_wrf_word_for_word(name):
    port, oracle = _replay(name, np)
    assert len(oracle) >= 2 * len(FIELDS)
    assert _mismatches(port, oracle) == {}


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("name", CASES)
def test_device_equals_wrf_and_the_host_word_for_word(name):
    import cupy as cp

    device, oracle = _replay(name, cp)
    assert _mismatches(device, oracle) == {}
    host, _ = _replay(name, np)
    assert _mismatches(device, host) == {}
