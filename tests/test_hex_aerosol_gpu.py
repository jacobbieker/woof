"""aer_opt = 1 and 2 on the hex generic seam ARE WRF's aerosol kernels.

The seam hands the radiation the cell latitude/longitude (aer_opt = 1, WRF's
Tegen climatology kernels) or the auxinput15 records remapped to the cells
(aer_opt = 2, WRF's user-aerosol optics).  Both are woof.core.rrtmg_aerosol_
optics / woof.core.aer_opt2, the functions the WRF path runs; this test steps
the seam and replays the same inputs through them.
"""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

import mpas_column_physics_cases as cases  # noqa: E402
from woof.core import mpas_column_suite as suite  # noqa: E402

NZ, NCOL, DT = cases.NZ, cases.NCOL, cases.DT
LAT = np.linspace(-60.0, 60.0, NCOL).astype(np.float32)
LON = np.linspace(-170.0, 170.0, NCOL).astype(np.float32)


def _mapping(physics, surface_inputs=None):
    plan = suite.column_physics_plan(
        physics, n_levels=NZ, dt=DT, radiation_seconds=120.0,
        surface_pbl_seconds=DT)
    surface = cases.surface(plan)
    rng = np.random.default_rng(2)
    values = {
        "n_levels": NZ, "n_columns": NCOL, "dt": DT,
        "radiation_seconds": 120.0, "surface_pbl_seconds": DT,
        "cumulus_seconds": None, "cumulus_scheme": None,
        "microphysics_scheme": "wsm6", "start_time": cases.START,
        "p_top_pa": cases.P_TOP, "dx_m": 12000.0, "gf_ishallow": 0,
        "wsm6_hail_opt": 0, "fractional_seaice": None,
        "orographic_statistics": None, "xice_threshold": 0.5,
        "radiation_scheme": "rrtmg_legacy", "surface_layer_scheme": "mm5_revised",
        "land_surface_scheme": "noahmp", "pbl_scheme": "ysu", "drag_scheme": "off",
        "latitude_deg": LAT, "longitude_deg": LON,
        "terrain_height_m": rng.uniform(170.0, 200.0, NCOL).astype(np.float32),
        "z_interface_nominal_m": np.linspace(0.0, cases.ZTOP, NZ + 1),
        "dx_column_m": np.full(NCOL, 12000.0, np.float32),
        "physics": physics,
        **{name: np.asarray(value) for name, value in surface.items()},
    }
    if surface_inputs is not None:
        values["surface_inputs"] = surface_inputs
    return values


def _seal(values):
    from woof.hex.cuda_suite_physics_v841 import SealedSuiteConstructorV841

    return SealedSuiteConstructorV841.from_mapping(values)


def _host(value):
    return cp.asnumpy(cp.asarray(value)).copy()


def _seam(physics, surface_inputs=None):
    return suite.run_mpas_column_physics(
        **_seal(_mapping(physics, surface_inputs)).arwen_kwargs())


def _swdown(seam):
    return _host(seam._driver.fields["swdown"])


def test_hex_aer_opt_1_columns_run_wrf_tegen_climatology_on_the_cell_centres():
    from woof.core import rrtmg_aerosol_optics as ao

    calls = []
    real = ao.aer1_time_pressure_device

    def capture(monthly, julian, p, pf, tables):
        out = real(monthly, julian, p, pf, tables)
        calls.append((_host(monthly), float(julian), _host(out[1])))
        return out

    ao.aer1_time_pressure_device = capture
    try:
        seam = _seam(dict(suite.MPAS_DEFAULT_PHYSICS, aer_opt=1))
        base = _seam(dict(suite.MPAS_DEFAULT_PHYSICS))
        cases.run(cp, seam, 4)
        cases.run(cp, base, 4)
    finally:
        ao.aer1_time_pressure_device = real
    assert calls
    leaf = next(item for item in __import__(
        "woof.core.radiation_composition", fromlist=["x"]).radiation_adapters(
            seam._driver.radiation_callable) if getattr(item, "aer_opt", 0) == 1
            and getattr(item, "shortwave", False))
    direct = ao.aer1_climatology_device(
        cp.asarray(LAT), cp.asarray(LON), leaf._sw_tables.aer1_climatology)
    for monthly, _, _ in calls:
        np.testing.assert_array_equal(
            monthly.view(np.uint32), _host(direct).view(np.uint32))
    np.testing.assert_array_equal(
        _host(seam._driver.aodtot).reshape(-1).view(np.uint32),
        calls[-1][2].reshape(-1).view(np.uint32))
    assert np.all(_host(seam._driver.aodtot) > 0)
    assert np.any(_swdown(seam) != _swdown(base))


USER = dict(aer_opt=2, aer_aod550_opt=2, aer_angexp_opt=2, aer_ssa_opt=2,
            aer_asy_opt=2, aer_auxinput15="unused-<date>.nc",
            aer_auxinput15_interval=3600)


def _frames():
    rng = np.random.default_rng(11)
    frame = np.stack([rng.uniform(0.05, 0.6, NCOL), rng.uniform(0.8, 1.8, NCOL),
                      rng.uniform(0.8, 0.98, NCOL), rng.uniform(0.6, 0.9, NCOL)])
    return frame[None].astype(np.float32), np.array([0.0])


def test_hex_aer_opt_2_columns_are_wrf_user_aerosol_optics_on_the_same_records():
    from woof.core import aer_opt2

    frames, offsets = _frames()
    physics = dict(suite.MPAS_DEFAULT_PHYSICS, **USER)
    hex_seam = _seam(physics, {"aerosol_frames": frames,
                               "aerosol_frame_offsets": offsets})
    cases.run(cp, hex_seam, 4)

    # WRF's path: the same records handed through auxiliary_inputs.
    fields = aer_opt2.INPUT_FIELDS
    real = aer_opt2.auxiliary_inputs
    aer_opt2.auxiliary_inputs = lambda cfg, start, elapsed, shape: {
        key: frames[0, j].reshape(shape) for j, key in enumerate(fields)}
    try:
        wrf_seam = _seam(physics, {"aerosol_frames": frames,
                                   "aerosol_frame_offsets": offsets})
        for leaf in __import__("woof.core.radiation_composition", fromlist=["x"]
                               ).radiation_adapters(wrf_seam._driver.radiation_callable):
            leaf.user_aerosol_frames = None
        cases.run(cp, wrf_seam, 4)
    finally:
        aer_opt2.auxiliary_inputs = real
    np.testing.assert_array_equal(
        _swdown(hex_seam).view(np.uint32), _swdown(wrf_seam).view(np.uint32))
    aod = _host(hex_seam._driver.aerosol2d["AOD5502D"]).reshape(-1)
    np.testing.assert_array_equal(aod.view(np.uint32), frames[0, 0].view(np.uint32))
    base = _seam(dict(suite.MPAS_DEFAULT_PHYSICS))
    cases.run(cp, base, 4)
    assert np.any(_swdown(hex_seam) != _swdown(base))


def test_hex_aer_opt_2_refuses_an_unstaged_record_and_missing_frames():
    frames, offsets = _frames()
    physics = dict(suite.MPAS_DEFAULT_PHYSICS, **USER)
    with pytest.raises(ValueError, match="missing .*aerosol_frames"):
        _seal(_mapping(physics))
    seam = _seam(physics, {"aerosol_frames": frames,
                           "aerosol_frame_offsets": offsets + 3600.0})
    with pytest.raises(ValueError, match="no auxinput15 record was staged"):
        cases.run(cp, seam, 4)
