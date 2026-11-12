"""The RADM2 / RADM2SORG process chain as the chem driver runs it.

The kernels are graded word for word against WRF v4.7.1 by their own parity
tests (test_chem_radm2_wrf471_parity, test_chem_sorgam_*_wrf471_parity,
test_chem_ftuv_wrf471_parity, test_chem_radm2_emiss_wrf471_parity).  This
file grades the WIRING: the process modules in CHEM_STEP_ORDER on a small
synthetic domain (the cold starts, then several steps of emission.radm2,
photolysis.ftuv, drydep.sorgam, chem.radm2 and aerosol.sorgam plus the PM
sums), and the CPU-only parts: the sets, the shared rows' gating, the chem
clock and the door's refusals.
"""
from __future__ import annotations

import hashlib
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import requires_gpu
from woof import chem_table
from woof.core import chem_radm2

NZ, NY, NX = 24, 2, 3


def test_sets_rows_and_gating():
    radm2 = chem_table.load_sets(["radm2"])
    sorg = chem_table.load_sets(["radm2sorg"])
    assert len(radm2.rows) == 42 and len(sorg.rows) == 80
    assert [r.name for r in radm2.rows_for("photolysis.ftuv")] == ["o3"]
    assert len(sorg.rows_for("chem.radm2")) == 42
    assert [r.name for r in sorg.rows_for("aerosol.sorgam")][:4] == ["sulf", "hno3", "nh3", "hcl"]
    assert len(sorg.rows_for("drydep.sorgam")) == 38
    assert not radm2.rows_for("aerosol.sorgam")
    for table in (radm2, sorg):
        assert not {"chem.sulfur", "emission.inventory", "drydep.gocart",
                    "wetdep.ls", "optics.gocart"} & set(table.processes)
        assert not table.diagnostics
    numbers = [r for r in sorg.rows if r.units == "kg-1"]
    assert [r.name for r in numbers] == ["nu0", "ac0", "corn"]
    # The shared rows keep every pre-RADM2 selection exactly.
    gocart = chem_table.load_sets(["gocart_simple"])
    assert "chem.radm2" not in gocart.processes
    assert [r.name for r in gocart.rows_for("chem.sulfur")] == ["so2", "sulf", "dms", "msa"]
    cams = chem_table.load_sets(["cams_aq"])
    assert set(cams.processes) == {"emission.inventory", "drydep.wesely", "mixing.vertmx"}


def test_number_rows_ledger_counts_particles():
    from woof.core.chem_driver import _kg_per_unit

    row = chem_table.catalog().species["nu0"]
    assert _kg_per_unit(row) == 1.0


def test_radm2_cannot_share_a_run_with_gocart_sulfur():
    table = chem_table.load_sets(["radm2sorg", "gocart_simple"])
    reason = chem_radm2._table_refusal(table, SimpleNamespace())
    assert "one mechanism" in reason


def test_chem_clock_is_wrfs():
    """chem_driver.F:366-417 with last_chem_time = start - dt
    (chemics_init.F:2070-2073): ktau 1 runs over dt, ktau 2 over 2 dt (the
    adaptive branch never recorded ktau 1), then dt; with chemdt = 2 min
    and dt = 45 s the mechanism runs where the next 120 s boundary falls in
    the step, over the time since the last such step."""
    def run(chemdt, dt, steps):
        diag = {"radm2_last_chem_secs": np.full((1, 1), -dt)}
        chem = SimpleNamespace(radm2_last_chem_host=-dt)
        out = []
        for ktau in range(1, steps + 1):
            ctx = SimpleNamespace(
                state=SimpleNamespace(chem=chem), diag=diag,
                cfg=SimpleNamespace(chemdt=chemdt),
                clock=SimpleNamespace(curr_secs=(ktau - 1) * dt))
            go, dtc = chem_radm2.chem_step_clock(ctx, dt, ktau)
            out.append((go, dtc) if go else None)
        return out
    assert run(0.0, 60.0, 4) == [(True, 60.0), (True, 120.0), (True, 60.0), (True, 60.0)]
    got = run(2.0, 45.0, 8)
    assert [g is not None for g in got] == [True, False, True, False, False, True, False, True]
    assert [g[1] for g in got if g] == [45.0, 135.0, 135.0, 90.0]


def _column():
    """A mid-latitude summer column, noon UTC at 0 E: float32 (nz, ny, nx)."""
    z_w = np.linspace(0.0, 16000.0, NZ + 1, dtype=np.float32)
    z = 0.5 * (z_w[1:] + z_w[:-1])
    t = (295.0 - 6.5e-3 * np.minimum(z, 11000.0)).astype(np.float32)
    p = (101325.0 * np.exp(-z / 7600.0)).astype(np.float32)
    p_w = (101325.0 * np.exp(-z_w / 7600.0)).astype(np.float32)
    t_w = np.interp(z_w, z, t).astype(np.float32)
    qv = (0.012 * np.exp(-z / 2500.0)).astype(np.float32)
    dry = p / (287.04 * t * (1.0 + 0.608 * qv))
    def grid(a):
        return np.ascontiguousarray(np.broadcast_to(a[:, None, None], (a.size, NY, NX)), np.float32)
    return {"z_at_w": grid(z_w), "z": grid(z), "t_phy": grid(t), "p_phy": grid(p),
            "p8w": grid(p_w), "t8w": grid(t_w), "qv": grid(qv),
            "qc": grid(np.where((z > 1000) & (z < 1500), 2e-4, 0.0).astype(np.float32)),
            "alt": grid((1.0 / dry).astype(np.float32)),
            "rho": grid((dry * (1.0 + qv)).astype(np.float32)),
            "dz8w": grid(np.diff(z_w).astype(np.float32))}


#: Typical rural-to-polluted mixing ratios (ppmv) for the gases the test
#: starts above epsilc; the rest start at the floor.
GAS_START = {"o3": 0.04, "no2": 0.008, "no": 0.004, "co": 0.15, "so2": 0.003,
             "hcho": 0.002, "ald": 0.001, "hc3": 0.004, "hc5": 0.002, "eth": 0.002,
             "olt": 5e-4, "oli": 3e-4, "tol": 1e-3, "xyl": 7e-4, "iso": 5e-4,
             "h2o2": 1e-3, "hno3": 1e-3, "nh3": 0.004, "pan": 3e-4}


def _run(steps):
    import cupy as cp
    from woof.core.chem_context import (ChemAllocation, ChemClock, ChemContext,
                                        allocation_shape, chem_processes)
    from woof.core.chem_state import process_attr

    table = chem_table.load_sets(["radm2sorg"])
    met = {k: cp.asarray(v) for k, v in _column().items()}
    met.update({name: cp.full((NY, NX), value, dtype=cp.float32) for name, value in
                (("ust", 0.35), ("rmol", -0.01), ("znt", 0.1), ("pblh", 1200.0),
                 ("xlat", 45.0), ("xlong", 0.0))})
    state = SimpleNamespace()
    for row in table.rows:
        setattr(state, row.state_attr, cp.full((NZ, NY, NX), np.float32(1e-16)))
    for name, value in GAS_START.items():
        getattr(state, table.row(name).state_attr)[...] = np.float32(value)
    prep = SimpleNamespace(get=met.get)
    state.chem = SimpleNamespace(table=table, prep=prep)
    processes = [(k, m) for k, m in chem_processes(table)
                 if k not in ("drydep.wesely", "mixing.vertmx")]
    diag = {}
    for _key, module in processes:
        for alloc in module.ALLOCATES:
            array = cp.zeros(allocation_shape(alloc, len(module.rows(table)), NZ, NY, NX),
                             dtype=alloc.dtype)
            diag[alloc.name] = array
            setattr(state, process_attr(alloc), array)
    # drydep.wesely's aerodynamic resistance from 2 m, which sorgam_depdriver reads.
    diag["wesely_aer_res_def"] = cp.full((NY, NX), np.float32(40.0))
    ddvel = cp.zeros((len(table.rows), NY, NX), dtype=cp.float32)
    dt = 60.0
    cfg = SimpleNamespace(dt=dt, chemdt=0.0, kemit=1, vertmix_onoff=1, photdt=0.0,
                          aer_ra_feedback=0, cu_diag=0, chem_sources="",
                          start_time=datetime(2026, 6, 21, 12))

    def ctx_for(ktau):
        return ChemContext(state, cfg, table, prep=prep, physics_fields=met, diag=diag,
                           ddvel=ddvel, clock=ChemClock(gmt=12.0, julday=172,
                                                        curr_secs=(ktau - 1) * dt,
                                                        ktau=ktau, dt=dt))
    ctx = ctx_for(1)
    import woof.core.chem_sorgam as sorgam
    sorgam.cold_start(ctx)
    diag["radm2_last_chem_secs"][...] = -dt
    state.chem.radm2_last_chem_host = -dt
    for _key, module in processes:
        module.init(ctx)
    o3_start = cp.asnumpy(getattr(state, table.row("o3").state_attr)).copy()
    for ktau in range(1, steps + 1):
        ctx = ctx_for(ktau)
        ddvel[...] = 0
        for _key, module in processes:
            module.step(ctx, dt, ktau)
        sorgam.refresh_output(state)
    fields = {row.name: cp.asnumpy(getattr(state, row.state_attr)) for row in table.rows}
    fields.update({name: cp.asnumpy(diag[name]) for name in diag})
    fields["ddvel"] = cp.asnumpy(ddvel)
    return table, fields, o3_start


@pytest.mark.gpu
@requires_gpu
def test_radm2sorg_chain_runs_and_repeats():
    table, a, o3_start = _run(4)
    _table, b, _ = _run(4)
    for name in a:
        assert np.array_equal(a[name], b[name], equal_nan=True), name
    for row in table.rows:
        assert np.isfinite(a[row.name]).all() and (a[row.name] >= 0).all(), row.name
    assert not np.array_equal(a["o3"], o3_start)
    assert (a["nu0"] >= 1e7).all()
    pm = a["sorgam_pm2_5_dry"]
    assert (pm > 0).all() and np.isfinite(pm).all()
    assert (a["sorgam_pm10"] >= pm).all()
    aer = [table.row(n) for n in chem_table.load_sets(["radm2sorg"]).names
           if table.row(n).phase == "aerosol" and n != "sulf"]
    planes = a["ddvel"][[table.names.index(r.name) for r in aer]]
    assert (planes > 0).all() and (planes <= 0.5).all()
    assert np.isfinite(a["ph_no2"]).all() and (a["ph_no2"] > 0).all()
    digest = hashlib.sha256()
    for name in sorted(a):
        digest.update(name.encode() + np.ascontiguousarray(a[name]).tobytes())
    print("radm2sorg chain sha256", digest.hexdigest())
