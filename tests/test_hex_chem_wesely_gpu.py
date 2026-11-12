"""Wesely gas dry deposition on hex columns: the land-use dataset and GSW the WRF driver hands its chem.

drydep.wesely (woof.core.chem_drydep_gas) maps land classes through the run's land-use dataset (WRF MMINLU,
read from the surface scheme's parameter tables on ``state.physics``) and reads radiation-time GSW, which the
WRF driver allocates only for a chem run.  The hex column seam names no chemistry in its own config, so
woof.hex.chem_wrf.HexWrfChemistry.bind_seam takes both from the seam.  Without them every set with Wesely
deposition (cams_aq here; RADM2 later) refused at its first step.  The cams_aq rows must then match the WRF
path's chem_step word for word, column by column, every step (tests/test_hex_chem_wrf_gpu.py's harness).
"""
from dataclasses import replace

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

import test_hex_chem_wrf_gpu as base  # noqa: E402

NZ, NCOL, DT = base.NZ, base.NCOL, base.DT
RAW = {"schema": "gpuwm-hex.chemistry/v1", "wrf_chem": {"chem_sets": "cams_aq"}}
#: initial ppmv of o3, no2, co, so2 (deposition needs something to deposit)
INITIAL = (0.04, 0.003, 0.12, 0.002)


class _NoFrames:
    """No anthropogenic inventory frames: emission.inventory emits nothing."""

    def available_hours(self, source, field):
        return frozenset()

    def at(self, source, field, when):
        raise AssertionError("no frame is available")


@pytest.fixture(scope="module")
def run():
    from woof.core.chem_driver import chem_step
    seam = base._seam()
    inputs = base._profile()
    state2 = {n: inputs[n].copy() for n in
              ("theta", "qv", "qc", "qr", "qi", "qs", "qg", "pressure", "rho_dry", "z_interface")}
    area = np.full(NCOL, 2.25e8)
    runtime = base._hex_runtime(seam, RAW, area)
    assert "gsw" not in seam._driver.fields
    runtime.bind_seam(seam)
    scalars = cp.zeros((len(runtime.rows), NZ, NCOL), dtype=cp.float32)
    for i, value in enumerate(INITIAL):
        scalars[i] = np.float32(value)
    wcfg = replace(runtime.cfg, ny=base.NY, nx=base.NX)
    wrf = base._wrf_state(seam, wcfg)
    wrf.physics.noahmp_params = seam._driver.noahmp_params
    for i, row in enumerate(runtime.rows):
        getattr(wrf, row.state_attr)[...] = base._to_wrf(scalars[i].reshape(NZ, 1, NCOL))
    runtime.state.chem.emission_frames = wrf.chem.emission_frames = _NoFrames()
    history = []
    for step in range(1, 6):
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
        history.append([(cp.asnumpy(scalars[i]), cp.asnumpy(getattr(wrf, row.state_attr)))
                        for i, row in enumerate(runtime.rows)])
    return runtime, seam, history


def test_the_seam_hands_the_land_use_tables_and_gsw(run):
    runtime, seam, _ = run
    assert runtime.state.physics.noahmp_params is seam._driver.noahmp_params
    assert runtime.state.physics.noahmp_params.dataset_identifier
    gsw = cp.asnumpy(seam._driver.fields["gsw"])
    assert gsw.shape == (1, NCOL) and np.all(np.isfinite(gsw)) and gsw.max() > 0.0


def test_rows_are_the_wrf_paths_every_step(run):
    _, _, history = run
    for step, rows in enumerate(history, 1):
        for i, (hexed, wrf) in enumerate(rows):
            back = cp.asnumpy(base._to_hex(cp.asarray(wrf)))
            assert hexed.tobytes() == back.tobytes(), (step, i)


def test_deposition_acted(run):
    runtime, _, history = run
    first = history[0][0][0]
    last = history[-1][0][0]
    assert last[0].sum() < first[0].sum(), "no ozone was deposited from the lowest level"
    assert runtime.ledger()["rows"]["o3"]["deposited_kg"] < 0.0
