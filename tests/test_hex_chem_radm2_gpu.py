"""RADM2 and RADM2SORG (chem_opt 1/2) on hex columns, word for word against the WRF path.

The hex column runtime (woof.hex.chem_wrf) runs the WRF path's own RADM2 package modules on ``ny = 1``
columns: Wesely gas deposition, vertical mixing, the RADM2 anthropogenic emissions (emiss_opt 3), Fast-TUV
photolysis (phot_opt 3), the RADM2 gas phase and, for radm2sorg, SORGAM's deposition and aerosol.  A fresh run
starts from WRF's own cold start (the chem_in_opt = 0 background gas profile, the SORGAM initial aerosol) on the
start's columns, as woof.core.chem_driver._cold_start does before the first step.  The same columns, scattered
into the interior of a WRF grid (tests/test_hex_chem_wrf_gpu.py's harness), go through woof.core.chem_driver's
cold start and chem_step: every row and every serialized process array must match byte for byte, column by
column, every step.
"""
from dataclasses import replace

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

import test_hex_chem_wrf_gpu as base  # noqa: E402

NZ, NCOL, DT = base.NZ, base.NCOL, base.DT
STEPS = 4


class _NoFrames:
    """No anthropogenic inventory frames: emission.radm2 emits nothing."""

    def available_hours(self, source, field):
        return frozenset()

    def at(self, source, field, when):
        raise AssertionError("no frame is available")


@pytest.fixture(scope="module", params=["radm2", "radm2sorg"])
def run(request):
    from woof.core.chem_context import chem_processes
    from woof.core.chem_driver import _cold_start, chem_step
    raw = {"schema": "gpuwm-hex.chemistry/v1", "wrf_chem": {"chem_sets": request.param}}
    seam = base._seam()
    inputs = base._profile()
    state2 = {n: inputs[n].copy() for n in
              ("theta", "qv", "qc", "qr", "qi", "qs", "qg", "pressure", "rho_dry", "z_interface")}
    area = np.full(NCOL, 2.25e8)
    runtime = base._hex_runtime(seam, raw, area)
    runtime.bind_seam(seam)
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
    keys = [a for a in runtime.serialized if not a.startswith("chemledger")]
    history = []

    def record(step):
        rows = [(cp.asnumpy(scalars[i]), cp.asnumpy(base._to_hex(getattr(wrf, row.state_attr))))
                for i, row in enumerate(runtime.rows)]
        arrays = {a: (cp.asnumpy(getattr(runtime.state, a)), cp.asnumpy(base._to_hex(getattr(wrf, a))))
                  for a in keys if getattr(wrf, a, None) is not None and getattr(wrf, a).shape[-2:] == (base.NY, base.NX)}
        history.append((step, rows, arrays))

    record(0)
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
        record(step)
    return request.param, runtime, history


def test_the_cold_start_is_the_wrf_paths(run):
    _, runtime, history = run
    step, rows, _ = history[0]
    for i, (hexed, wrf) in enumerate(rows):
        assert hexed.tobytes() == wrf.reshape(hexed.shape).tobytes(), runtime.rows[i].name
    assert sum(float(h.sum()) for h, _ in rows) > 0.0, "the cold start wrote nothing"


def test_rows_and_process_arrays_are_the_wrf_paths_every_step(run):
    name, runtime, history = run
    compared = set(history[-1][2])
    assert {"chemproc_ph_no2", "chemproc_radm2_last_chem_secs"} <= compared or any("ph_no2" in a for a in compared), \
        sorted(compared)[:10]
    if name == "radm2sorg":
        assert any("sorgam_" in a for a in compared), sorted(compared)[:10]
    for step, rows, arrays in history[1:]:
        for i, (hexed, wrf) in enumerate(rows):
            assert hexed.tobytes() == wrf.reshape(hexed.shape).tobytes(), (step, runtime.rows[i].name)
        for attr, (hexed, wrf) in arrays.items():
            assert hexed.reshape(wrf.shape).tobytes() == wrf.tobytes(), (step, attr)


def test_the_chemistry_acted(run):
    name, runtime, history = run
    index = {row.name: i for i, row in enumerate(runtime.rows)}
    first, last = history[0][1], history[-1][1]
    assert not np.array_equal(first[index["o3"]][0], last[index["o3"]][0]), "ozone did not change"
    if name == "radm2sorg":
        assert any(not np.array_equal(first[i][0], last[i][0]) for row, i in index.items()
                   if row.endswith(("aj", "ai"))), "no aerosol mass changed"


def test_emiss_opt_enables_what_the_wrf_door_enables():
    from woof.hex import chem_wrf
    from woof.namelist_chem import _sources_for
    for name in ("radm2", "radm2sorg"):
        assert chem_wrf.chem_sources(name, {"emiss_opt": 3}) == _sources_for(name, {"emiss_opt": 3})
        assert "edgar-v81" in chem_wrf.chem_sources(name, {"emiss_opt": 3})
        assert chem_wrf.chem_sources(name, {}) == ()
        assert chem_wrf.chem_sources(name, {"chem_sources": "synthetic-anthro"}) == ("synthetic-anthro",)
        with pytest.raises(ValueError, match="emiss_opt = 1"):
            chem_wrf.chem_sources(name, {"emiss_opt": 1})


def test_inventory_emissions_are_the_wrf_paths():
    """radm2 emitting from the synthetic-anthro inventory (InventoryFrames on the cell centres, each cell's
    own area through the area-equivalent map factor): rows byte-identical to the WRF path every step."""
    from woof.core.chem_context import chem_processes
    from woof.core.chem_driver import _cold_start, chem_step
    raw = {"schema": "gpuwm-hex.chemistry/v1",
           "wrf_chem": {"chem_sets": "radm2", "emiss_opt": 3, "chem_sources": "synthetic-anthro"}}
    seam = base._seam()
    inputs = base._profile()
    state2 = {n: inputs[n].copy() for n in
              ("theta", "qv", "qc", "qr", "qi", "qs", "qg", "pressure", "rho_dry", "z_interface")}
    runtime = base._hex_runtime(seam, raw, base.SMOKE_AREA)
    runtime.bind_seam(seam)
    assert runtime.cfg.chem_sources == "synthetic-anthro"
    scalars = cp.zeros((len(runtime.rows), NZ, NCOL), dtype=cp.float32)
    wcfg = replace(runtime.cfg, ny=base.NY, nx=base.NX)
    wrf = base._wrf_state(seam, wcfg)
    wrf.physics.noahmp_params = seam._driver.noahmp_params
    wrf.msft = base._to_wrf(runtime.state.msft)
    start = base._columns(inputs, inputs)
    runtime.cold_start(scalars, start)
    base._wrf_load(wrf, start, {**seam._driver.fields, **runtime.geography}, None)
    wrf.elapsed_seconds = 0.0
    _cold_start(wrf, wcfg, chem_processes(wrf.chem.table))
    for step in range(1, 4):
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
        for i, row in enumerate(runtime.rows):
            want = cp.asnumpy(base._to_hex(getattr(wrf, row.state_attr)))
            assert cp.asnumpy(scalars[i]).tobytes() == want.reshape(NZ, NCOL).tobytes(), (step, row.name)
    assert runtime.state.chem.radm2_frames_receipt is not None
    assert float(cp.asnumpy(runtime.state.chem.radm2_frames["e_co"]).max()) > 0.0, "the inventory gave no CO"
    assert runtime.ledger()["rows"]["co"]["emitted_kg"] > 0.0, "no CO was emitted"
