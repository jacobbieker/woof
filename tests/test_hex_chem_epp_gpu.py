"""spacewx_epp_opt on hex's RADM2 chemistry: the solar-proton NO, OH and HO2, word for word the WRF path's.

The hex column runtime (woof.hex.chem_wrf) runs the WRF path's own emission.radm2, whose solar-proton hook
(woof.core.chem_radm2_emiss._solar_protons, woof.spacewx.epp_chem) reads the cells' own xlat/xlong
(set_geography, the mesh's cell centres), chem_prep's p8w and the case start.  The key sits in the chemistry
config's wrf_chem block, so both hex routes take it from the same place:

* the hex WOOF route, on tests/test_hex_chem_radm2_gpu.py's harness (the same columns on hex and scattered
  into a WRF grid): every row and every photolysis rate matches byte for byte after each step, on equal
  cells and on a variable-resolution set of cell areas;
* the MPAS v8.4.2 suite route (woof.hex.mpasphys.chemistry_v842, whose column config has no WRF radiation),
  on tests/test_hex_chem_suite_v842_gpu.py's stand-in session: its rows are the WOOF route's byte for byte.

On both, the cells inside a polar cap gain NO against the same run without the source and every other cell
is untouched to the bit; the NO the source adds each step is positive at every cap cell's top and the same
from the same pressures on hex and on the WRF grid.  (Below a 137 hPa model top that hard event's increment
is near float32's resolution of the RADM2 background, so it is checked as the source adds it, not as a row
difference; the published-scale check, about 1 ppbv NO a day at 1 hPa, is tests/test_spacewx_epp_chem.py's,
on the same woof.spacewx.epp_chem code.)  The proton flux is a fixed hold (a hard event, about 4e4 pfu above 10 MeV, the
size of October 1989's peak), the geomagnetic field the synthetic dipole of tests/test_spacewx_ionosphere.py.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

import test_hex_chem_suite_v842_gpu as sv842  # noqa: E402
import test_hex_chem_wrf_gpu as base  # noqa: E402
from test_hex_chem_radm2_gpu import _NoFrames  # noqa: E402
from test_spacewx_epp_chem import JAN2012, _held  # noqa: E402
from test_spacewx_ionosphere import _dipole  # noqa: E402
from woof.hex import chem_wrf  # noqa: E402

NZ, NCOL, DT = base.NZ, base.NCOL, base.DT
STEPS = 3
pytestmark = pytest.mark.gpu

#: cell centres from the auroral zone into both polar caps and out to mid latitudes
LAT = np.array([82.0, 50.0, 74.0, 35.0, -79.0, 63.0, 88.0, -45.0])
LON = np.linspace(-10.0, 5.0, NCOL)
#: a hard event: January 2012's spectrum times 15 (about 4e4 pfu above 10 MeV)
HARD = [v * 15.0 for v in JAN2012]
EPP = {"spacewx_epp_opt": 1, "spacewx_epp_source": "swpc"}
AREAS = {"uniform": np.full(NCOL, 2.25e8), "variable": np.geomspace(9.0e6, 4.0e8, NCOL)[base.PERM]}


@pytest.fixture
def held(monkeypatch):
    """The fixed hold, and a record of every step's NO increment the source adds (ppmv, per apply call)."""
    from woof.spacewx import epp_chem, igrf, particles
    monkeypatch.setattr(igrf, "load", lambda *a, **k: _dipole())
    monkeypatch.setattr(epp_chem, "held_flux", lambda source, start, **k: _held(HARD))
    calls, apply = [], epp_chem.EppSource.apply

    def spy(self, rows, p8w, dt):
        q = epp_chem.ion_pairs(p8w, self.log_rate, cp) * self.cap
        per_pair = particles.vmr_tendency(1.0, 1.0) * 1.0e6 * float(dt) * dict(epp_chem.YIELDS)["no"]
        calls.append((cp.asnumpy(q).reshape(q.shape[0], -1) * per_pair, cp.asnumpy(p8w).reshape(p8w.shape[0], -1)))
        return apply(self, rows, p8w, dt)

    monkeypatch.setattr(epp_chem.EppSource, "apply", spy)
    return calls


def _source_acts_in_the_caps_only(calls, cap):
    """Each step's NO increment: positive at the top of every cap cell, zero outside the caps."""
    for inc, _p8w in calls:
        assert inc.shape[-1] == NCOL
        assert np.all(inc[:, ~cap] == 0.0)
        assert np.all(inc[-1, cap] > 0.0), inc[-1]


def _cap():
    from woof.spacewx import aurora, particles
    frame = aurora.dipole_frame(_dipole(), base.START)
    return particles.polar_cap(aurora.dipole_latitude(frame, LAT, LON)).astype(bool)


def _raw(epp):
    return {"schema": "gpuwm-hex.chemistry/v1", "wrf_chem": {"chem_sets": "radm2", **(EPP if epp else {})}}


def _woof_route(epp, area):
    """STEPS steps of the hex runtime and the WRF path's chem_step on the same columns."""
    from woof.core.chem_context import chem_processes
    from woof.core.chem_driver import _cold_start, chem_step
    seam = base._seam()
    inputs = base._profile()
    state2 = {n: inputs[n].copy() for n in
              ("theta", "qv", "qc", "qr", "qi", "qs", "qg", "pressure", "rho_dry", "z_interface")}
    runtime = base._hex_runtime(seam, _raw(epp), area)
    runtime.set_geography(LAT, LON)
    runtime.bind_seam(seam)
    assert runtime.cfg.spacewx_epp_opt == int(epp)
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


@pytest.mark.parametrize("mesh", sorted(AREAS))
def test_the_woof_route_is_the_wrf_paths_source_word_for_word(held, capsys, mesh):
    runtime, wrf, on = _woof_route(True, AREAS[mesh])
    assert "[spacewx epp]" in capsys.readouterr().out
    cap = _cap()
    assert cap.any() and not cap.all(), cap
    hex_cap = cp.asnumpy(runtime.state.chem.epp_source.cap).reshape(-1).astype(bool)
    assert hex_cap.tolist() == cap.tolist()
    assert runtime.receipt()["spacewx_epp"]["spacewx_epp_spe"] is True
    assert runtime.receipt()["knobs"]["spacewx_epp_opt"] == 1
    np.testing.assert_array_equal(cp.asnumpy(base._to_hex(wrf.chem.epp_source.cap)).reshape(-1), hex_cap)
    for step, (rows, ph) in enumerate(on, start=1):
        for i, (hexed, wrfed) in enumerate(rows):
            assert hexed.tobytes() == wrfed.reshape(hexed.shape).tobytes(), (step, runtime.rows[i].name)
        for attr, (hexed, wrfed) in ph.items():
            assert hexed.reshape(wrfed.shape).tobytes() == wrfed.tobytes(), (step, attr)

    off_runtime, _wrf, off = _woof_route(False, AREAS[mesh])
    assert getattr(off_runtime.state.chem, "epp_source", None) is None
    assert "spacewx_epp" not in off_runtime.receipt()
    for (rows_on, _p), (rows_off, _q) in zip(on, off):
        for (a, _w), (b, _v), row in zip(rows_on, rows_off, runtime.rows):
            assert a[:, ~cap].tobytes() == b[:, ~cap].tobytes(), row.name
    # hex's source and the WRF path's, from the same p8w, every step
    assert len(held) == 2 * STEPS
    for (inc_hex, p_hex), (inc_wrf, p_wrf) in zip(held[0::2], held[1::2]):
        assert p_hex.tobytes() == cp.asnumpy(base._to_hex(cp.asarray(p_wrf.reshape(-1, base.NY, base.NX)))).tobytes()
        np.testing.assert_array_equal(inc_hex, cp.asnumpy(base._to_hex(cp.asarray(
            inc_wrf.reshape(-1, base.NY, base.NX)))).reshape(inc_hex.shape))
    _source_acts_in_the_caps_only(held[0::2], cap)


def _suite(tmp_path, raw, area, start, dataset):
    """tests/test_hex_chem_suite_v842_gpu.py's _suite with the start's columns in place at bind (the RADM2
    cold start reads them) and no inventory frames."""
    import json
    from types import SimpleNamespace
    from woof.hex.mpasphys import chemistry_v842 as suite
    path = tmp_path / "chemistry.json"
    path.write_text(json.dumps(raw))
    mesh = SimpleNamespace(n_cells=NCOL, attrs={}, arrays={"areaCell": np.asarray(area, np.float64),
                                                           "latCell": np.radians(LAT), "lonCell": np.radians(LON)})
    zw = np.linspace(0.0, 16000.0, NZ + 1)
    vertical = SimpleNamespace(zw=zw, zgrid=np.repeat(zw[None, :], NCOL, axis=0))
    chem = suite.SuiteChemistry(path, registry_names=sv842.REGISTRY, mesh=mesh, nlev=NZ, grid_path=path,
                                start_text=sv842.CONFIGS["config_start_time"], vertical=vertical,
                                configs=dict(sv842.CONFIGS), dt=DT)
    rng = np.random.default_rng(7)
    host = SimpleNamespace(scalars=rng.uniform(1e-6, 1e-3, (len(sv842.REGISTRY), NZ, NCOL)).astype(np.float32))
    chem.extend_state(host)
    dycore = SimpleNamespace(atmosphere=SimpleNamespace(state=SimpleNamespace(scalars=cp.asarray(host.scalars))))
    session = sv842._Session()
    # Wesely reads MPAS's sfc_input.mminlu: the seam's own land-use dataset (as the suite test's cams_aq case)
    session.native = SimpleNamespace(
        get_text_field=lambda pool, name: dataset if (pool, name) == ("sfc_input", "mminlu") else None)
    session.pools["sfc_input", "xland"] = cp.asarray(np.ones(NCOL), dtype=cp.float32)
    holder = SimpleNamespace(columns=start)
    chem._columns = lambda atmosphere: sv842._prepared(holder.columns)
    real = chem_wrf.HexWrfChemistry.__init__

    def no_frames(self, *a, **k):
        real(self, *a, **k)
        self.state.chem.emission_frames = _NoFrames()

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(chem_wrf.HexWrfChemistry, "__init__", no_frames)
        chem.bind(dycore=dycore, cache=None, coupling=SimpleNamespace(session=session, names=sv842.REGISTRY))
    return chem, dycore, session, holder


def _suite_route(tmp_path, monkeypatch, epp, area):
    """The suite route and the WOOF route's runtime on its config, on the same columns (tests/
    test_hex_chem_suite_v842_gpu.py's wrf_case, RADM2)."""
    tmp_path.mkdir()
    original = chem_wrf.attach_statics
    monkeypatch.setattr(chem_wrf, "attach_statics",
                        lambda runtime, **kw: original(runtime, lat_deg=None, lon_deg=None, fields=base._statics()))
    seam = base._seam()
    inputs = base._profile()
    state2 = {n: inputs[n].copy() for n in
              ("theta", "qv", "qc", "qr", "qi", "qs", "qg", "pressure", "rho_dry", "z_interface")}
    raw = _raw(epp)
    chem, dycore, session, holder = _suite(tmp_path, raw, area, base._columns(inputs, inputs),
                                         seam._driver.noahmp_params.dataset_identifier)
    ref = chem_wrf.HexWrfChemistry("chemistry.json", raw, chem.digest, cfg=seam._cfg, area=area, nlev=NZ,
                                   ncells=NCOL, offset=0, start_time=base.START, fnm=seam._state.fnm,
                                   fnp=seam._state.fnp, terrain=seam._state.ht)
    ref.bind_seam(seam)
    original(ref, lat_deg=None, lon_deg=None, fields=base._statics())
    ref.set_geography(LAT, LON)
    ref.state.chem.emission_frames = _NoFrames()
    ref_scalars = cp.zeros((len(ref.rows), NZ, NCOL), dtype=cp.float32)
    # the RADM2 cold start the suite's attach ran (chem_wrf.attach), on the same start columns
    ref.cold_start(ref_scalars, chem_wrf.columns_from_prep(sv842._prepared(holder.columns)))
    albedo = cp.asarray(np.linspace(0.08, 0.3, NCOL), dtype=cp.float32)
    rng = np.random.default_rng(13)
    out = []
    for step in range(1, STEPS + 1):
        seam.run_phase1(dt=DT, **inputs)
        seam.run_phase2(**state2)
        driver = seam._driver
        columns = base._columns(inputs, state2)
        holder.columns = columns
        fields = dict(driver.fields)
        fields["gsw"] = cp.asarray(rng.uniform(0.0, 800.0, (1, NCOL)), dtype=cp.float32)
        fields["swdown"] = fields["gsw"] / (np.float32(1.0) - albedo.reshape(1, NCOL))
        session.publish(fields, albedo=albedo)
        if driver.microphysics is not None:
            session.pools["diag_physics", "rainncv"] = driver.microphysics.rainncv.reshape(NCOL)
        ref.step(ref_scalars, ref.columns(sv842._prepared(columns), "post_rk"), dt=DT, elapsed_seconds=step * DT,
                 physics_fields=fields, microphysics=driver.microphysics)
        ref.commit()
        chem.step(session=session, dt=DT)
        out.append((cp.asnumpy(dycore.atmosphere.state.scalars)[len(sv842.REGISTRY):].copy(),
                    cp.asnumpy(ref_scalars).copy()))
    return chem, out


@pytest.mark.parametrize("mesh", sorted(AREAS))
def test_the_suite_route_takes_it_from_the_chemistry_config(held, tmp_path, monkeypatch, mesh):
    chem, on = _suite_route(tmp_path / "on", monkeypatch, True, AREAS[mesh])
    from woof.config import radiation_enabled
    assert chem.runtime.cfg.spacewx_epp_opt == 1 and not radiation_enabled(chem.runtime.cfg)
    cap = _cap()
    assert cp.asnumpy(chem.runtime.state.chem.epp_source.cap).reshape(-1).astype(bool).tolist() == cap.tolist()
    for step, (suite_rows, woof_rows) in enumerate(on, start=1):
        assert suite_rows.tobytes() == woof_rows.tobytes(), step
    # the suite runtime's source and its WOOF-route reference's, every step
    assert len(held) == 2 * STEPS
    for (inc_ref, _p), (inc_suite, _q) in zip(held[0::2], held[1::2]):
        assert inc_ref.tobytes() == inc_suite.tobytes()
    _source_acts_in_the_caps_only(held, cap)
    _chem, off = _suite_route(tmp_path / "off", monkeypatch, False, AREAS[mesh])
    a, b = on[-1][0], off[-1][0]
    assert a[:, :, ~cap].tobytes() == b[:, :, ~cap].tobytes()
