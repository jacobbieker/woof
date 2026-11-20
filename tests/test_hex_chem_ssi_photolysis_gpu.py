"""solar_ssi_opt on hex's RADM2 chemistry: Fast-TUV's flux factor, word for word the WRF path's.

The hex column runtime (woof.hex.chem_wrf) runs the WRF path's own photolysis.ftuv process, so with
``solar_ssi_opt = 1`` its photolysis rates take the day's per-bin factor (woof.spacewx.solar_spectrum) the way
a WRF domain's do.  On tests/test_hex_chem_radm2_gpu.py's harness (the same columns on hex and scattered into
a WRF grid), every ph_* array and every row matches byte for byte after each step, and the rates differ from
the run without the option.
"""
from dataclasses import replace
from datetime import datetime, timedelta

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

import test_hex_chem_wrf_gpu as base  # noqa: E402
from test_hex_chem_radm2_gpu import _NoFrames  # noqa: E402

NZ, NCOL, DT = base.NZ, base.NCOL, base.DT
STEPS = 2

pytestmark = pytest.mark.gpu


def _spectrum_file(path):
    """A NOAA CDR-layout SSI file over the run's days: the cycle-24 mean spectrum, with 337-420 nm (NO2's
    and HCHO's band) 2 % and 290-320 nm (O3 -> O1D's) 3 % above it."""
    netCDF4 = pytest.importorskip("netCDF4")
    from woof.spacewx import solar_spectrum as ss
    wl, dw, frac = ss.reference_spectrum()
    tsi, days = 1361.0, 3
    first = (base.START - timedelta(days=1)).replace(hour=0)
    start = (first - datetime(1610, 1, 1)).days
    shape = np.where((wl > 337.5) & (wl < 420.0), 1.02, np.where((wl > 290.0) & (wl < 320.0), 1.03, 1.0))
    with netCDF4.Dataset(path, "w") as data:
        data.createDimension("time", days)
        data.createDimension("wavelength", wl.size)
        data.createDimension("bounds", 2)
        t = data.createVariable("time", "f8", ("time",))
        t.units = "days since 1610-01-01 00:00:00"
        t[:] = start + np.arange(days)
        b = data.createVariable("time_bnds", "f8", ("time", "bounds"))
        b[:] = np.stack([start + np.arange(days), start + 1 + np.arange(days)], axis=1)
        data.createVariable("wavelength", "f8", ("wavelength",))[:] = wl
        data.createVariable("Wavelength_Band_Width", "f8", ("wavelength",))[:] = dw
        data.createVariable("SSI", "f8", ("time", "wavelength"))[:] = np.tile(tsi * frac * shape, (days, 1))
        data.createVariable("TSI", "f8", ("time",))[:] = np.full(days, tsi)
    return path


def _run(overrides):
    from woof.core.chem_context import chem_processes
    from woof.core.chem_driver import _cold_start, chem_step
    raw = {"schema": "gpuwm-hex.chemistry/v1", "wrf_chem": {"chem_sets": "radm2"}}
    seam = base._seam()
    seam._cfg = replace(seam._cfg, **overrides)
    inputs = base._profile()
    state2 = {n: inputs[n].copy() for n in
              ("theta", "qv", "qc", "qr", "qi", "qs", "qg", "pressure", "rho_dry", "z_interface")}
    runtime = base._hex_runtime(seam, raw, np.full(NCOL, 2.25e8))
    runtime.bind_seam(seam)
    assert all(getattr(runtime.cfg, k) == v for k, v in overrides.items())
    scalars = cp.zeros((len(runtime.rows), NZ, NCOL), dtype=cp.float32)
    wcfg = replace(runtime.cfg, ny=base.NY, nx=base.NX)
    wrf = base._wrf_state(seam, wcfg)
    wrf.physics.noahmp_params = seam._driver.noahmp_params
    runtime.state.chem.emission_frames = wrf.chem.emission_frames = _NoFrames()
    start = base._columns(inputs, inputs)
    runtime.cold_start(scalars, start)
    base._wrf_load(wrf, start, {**seam._driver.fields, **runtime.geography}, None)
    wrf.elapsed_seconds = 0.0
    _cold_start(wrf, wcfg, chem_processes(wrf.chem.table))
    keys = [a for a in runtime.serialized if a.startswith("chemdiag_ph_")]
    assert "chemdiag_ph_no2" in keys, keys[:5]
    out = []
    for step in range(1, STEPS + 1):
        seam.run_phase1(dt=DT, **inputs)
        seam.run_phase2(**state2)
        driver = seam._driver
        columns = base._columns(inputs, state2)
        runtime.step(scalars, columns, dt=DT, elapsed_seconds=step * DT, physics_fields=driver.fields,
                     microphysics=driver.microphysics)
        runtime.commit()
        base._wrf_load(wrf, columns, {**driver.fields, **runtime.geography}, driver.microphysics)
        wrf.elapsed_seconds = (step - 1) * DT
        chem_step(wrf, wcfg, DT)
        rows = [(cp.asnumpy(scalars[i]), cp.asnumpy(base._to_hex(getattr(wrf, row.state_attr))))
                for i, row in enumerate(runtime.rows)]
        ph = {a: (cp.asnumpy(getattr(runtime.state, a)), cp.asnumpy(base._to_hex(getattr(wrf, a)))) for a in keys}
        out.append((rows, ph))
    return runtime, wrf, out


def test_hex_photolysis_takes_the_factor_word_for_word(tmp_path):
    path = _spectrum_file(tmp_path / "ssi.nc")
    runtime, wrf, on = _run({"solar_ssi_opt": 1, "solar_ssi_source": str(path)})
    hex_spec = runtime.state.chem.ftuv_solar_spectrum
    wrf_spec = wrf.chem.ftuv_solar_spectrum
    np.testing.assert_array_equal(hex_spec.start_factor, wrf_spec.start_factor)
    f = hex_spec.start_factor
    assert 1.019 < f[13] <= 1.02 and f[11] == pytest.approx(1.03, rel=1e-9)
    assert f[14] == pytest.approx(1.0, rel=1e-9)
    for step, (rows, ph) in enumerate(on, start=1):
        for i, (hexed, wrfed) in enumerate(rows):
            assert hexed.tobytes() == wrfed.reshape(hexed.shape).tobytes(), (step, runtime.rows[i].name)
        for attr, (hexed, wrfed) in ph.items():
            assert hexed.reshape(wrfed.shape).tobytes() == wrfed.tobytes(), (step, attr)

    _runtime, _wrf, off = _run({})
    assert not hasattr(_runtime.state.chem, "ftuv_solar_spectrum")
    for name in ("chemdiag_ph_no2", "chemdiag_ph_o31d", "chemdiag_ph_ch2or"):
        a, b = on[0][1][name][0], off[0][1][name][0]
        lit = b > 0
        assert lit.any(), name
        ratio = a[lit] / b[lit]
        assert np.all(ratio >= 1.0) and ratio.max() > 1.005 and ratio.max() < 1.031, (name, ratio.max())
