"""WRF-Chem aerosol-radiation feedback (aer_ra_feedback = 1) on hex columns.

The hex column seam's radiation is the WRF legacy RRTMG; with feedback it takes the chemistry's aerosol
optics through woof.core.chem_radiation.prepare exactly as a WRF domain does.  These tests drive the seam
with the hex chem runtime bound to it and require the optics the radiation consumed at every radiation
call to be bitwise, column by column, those the WRF path computes from the same rows and phase-one columns
in the interior of a WRF grid (helpers shared with test_hex_chem_wrf_gpu), and the shortwave heating to
respond to the aerosol.
"""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

import test_hex_chem_wrf_gpu as base  # noqa: E402
from woof.core import mpas_column_batch as mcb  # noqa: E402

NZ, NCOL, DT, START = base.NZ, base.NCOL, base.DT, base.START
STEPS = 7
RADIATION_SECONDS = 2 * DT
#: the optical properties chem_radiation hands the legacy RRTMG (RadiationAerosols.legacy)
BANDS = ("tauaer", "waer", "gaer", "tauaerlw")


def _seam():
    return mcb.run_mpas_column_batch(
        n_levels=NZ, n_columns=NCOL, dt=DT, radiation_seconds=RADIATION_SECONDS, surface_pbl_seconds=DT,
        start_time=START, latitude_deg=np.linspace(20.0, 34.0, NCOL),
        longitude_deg=np.linspace(-10.0, 5.0, NCOL), terrain_height_m=np.zeros(NCOL),
        z_interface_nominal_m=np.linspace(0.0, 16000.0, NZ + 1),
        p_top_pa=float(101325.0 * np.exp(-2.0)), dx_m=15000.0,
        soil_moisture=0.05, vegfra=5.0, isltyp=1, ivgtyp=16)


def _raw(feedback):
    return {"schema": "gpuwm-hex.chemistry/v1",
            "wrf_chem": {"chem_opt": 401, "dust_opt": 1, "aer_ra_feedback": feedback}}


def _phase1_columns(inputs):
    return {name: inputs[name] for name in
            ("pressure", "theta", "rho_dry", "z_interface", "qv", "qc", "qr", "u", "v", "w")}


def _drive(feedback):
    seam = _seam()
    inputs = base._profile()
    state2 = {name: inputs[name].copy() for name in
              ("theta", "qv", "qc", "qr", "qi", "qs", "qg", "pressure", "rho_dry", "z_interface")}
    runtime = base._hex_runtime(seam, _raw(feedback), np.full(NCOL, 2.25e8))
    runtime.bind_seam(seam)
    scalars = cp.zeros((len(runtime.rows), NZ, NCOL), dtype=cp.float32)
    calls = []
    for step in range(1, STEPS + 1):
        start = (step - 1) * DT
        rows_at_start = scalars.copy()
        runtime.prepare_radiation(scalars, _phase1_columns(inputs), elapsed_seconds=start)
        result = seam.run_phase1(dt=DT, **inputs)
        if result.radiation_ran:
            cached = getattr(seam._state, "chem", None)
            optics = None
            if feedback:
                aerosols = cached.radiation_aerosols
                assert cached.radiation_aerosols_time == start
                optics = {k: cp.asnumpy(aerosols.legacy_optics[k]).copy() for k in BANDS}
            calls.append({"step": step, "rows": rows_at_start, "optics": optics,
                          "sw": cp.asnumpy(seam._driver.rthratensw).copy()})
        seam.run_phase2(**state2)
        runtime.step(scalars, base._columns(inputs, state2), dt=DT, elapsed_seconds=step * DT,
                     physics_fields=seam._driver.fields, microphysics=seam._driver.microphysics)
        runtime.commit()
    return seam, runtime, inputs, calls


@pytest.fixture(scope="module")
def runs():
    return {"on": _drive(1), "off": _drive(0)}


def test_radiation_optics_are_bitwise_the_wrf_path_per_column(runs):
    from dataclasses import replace
    from woof.core.chem_driver import _context
    from woof.core.chem_radiation import _legacy_optics
    seam, runtime, inputs, calls = runs["on"]
    wcfg = replace(runtime.cfg, ny=base.NY, nx=base.NX)
    wrf = base._wrf_state(seam, wcfg)
    assert len(calls) >= 3
    for call in calls:
        base._wrf_load(wrf, _phase1_columns(inputs), seam._driver.fields, None)
        for i, row in enumerate(runtime.rows):
            getattr(wrf, row.state_attr)[...] = base._to_wrf(call["rows"][i])
        wrf.elapsed_seconds = (call["step"] - 1) * DT
        ctx = _context(wrf, wcfg, call["step"])
        ctx.prep.refresh(wrf)
        reference = _legacy_optics(ctx)
        for name in BANDS:
            back = cp.asnumpy(base._to_hex(reference[name]))
            assert call["optics"][name].reshape(back.shape).tobytes() == back.tobytes(), (call["step"], name)


def test_feedback_is_not_vacuous(runs):
    on, off = runs["on"][3], runs["off"][3]
    assert [c["step"] for c in on] == [c["step"] for c in off]
    assert on[0]["sw"].tobytes() == off[0]["sw"].tobytes(), "no aerosol yet at the first call"
    tau = on[-1]["optics"]["tauaer"]
    assert float(tau.max()) > 0.0
    assert on[-1]["sw"].tobytes() != off[-1]["sw"].tobytes(), "the aerosol did not reach the shortwave"


def test_feedback_refused_without_optics_rows(tmp_path):
    from woof.hex.chem_wrf import HexWrfChemistry
    seam = _seam()
    raw = {"schema": "gpuwm-hex.chemistry/v1",
           "wrf_chem": {"chem_sets": "tracer_mixing", "aer_ra_feedback": 1}}
    with pytest.raises(Exception, match="optics.gocart"):
        HexWrfChemistry("c.json", raw, "0" * 64, cfg=seam._cfg, area=np.ones(NCOL), nlev=NZ, ncells=NCOL,
                        offset=0, start_time=START, fnm=seam._state.fnm, fnp=seam._state.fnp)
