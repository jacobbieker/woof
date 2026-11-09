"""WRF-Chem column processes on hex columns, bitwise per column against the WRF path.

The hex physics is the WRF PhysicsDriver on ``ny = 1`` columns (woof.core.mpas_column_batch).  These tests
run that seam on synthetic moist columns, then the hex chem step (woof.hex.chem_wrf) on its columns and,
from the same columns scattered in a shuffled order into the interior of a ``(nz, 4, 6)`` WRF grid (its
edge ring padded with copies, since WRF's GOCART deposition skips the domain's first row and column), the
WRF path's own ``woof.core.chem_driver.chem_step``.  Every row and every process array must match word for word, column
by column, at every step; the hex ledger must close; the restart must continue bit-identically.
"""

import datetime
import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from woof.core import mpas_column_batch as mcb  # noqa: E402

NZ, NCOL, NY, NX = 20, 8, 4, 6
DT = 60.0
START = datetime.datetime(2021, 6, 1, 18, 0)
STEPS = 6
#: hex column c sits at WRF interior point (j, i) = (1 + PERM[c] // 4, 1 + PERM[c] % 4), shuffled; the
#: edge ring holds copies of hex columns (SOURCE).
PERM = np.array([5, 2, 7, 0, 3, 6, 1, 4])
POS = (1 + PERM // 4) * NX + 1 + PERM % 4
SOURCE = np.zeros(NY * NX, dtype=np.int64)
SOURCE[POS] = np.arange(NCOL)
_ring = np.setdiff1d(np.arange(NY * NX), POS)
SOURCE[_ring] = np.arange(_ring.size) % NCOL


def _profile(seed=3):
    rng = np.random.default_rng(seed)
    z_iface = np.linspace(0.0, 16000.0, NZ + 1)
    z_mid = 0.5 * (z_iface[:-1] + z_iface[1:])
    p_iface = 101325.0 * np.exp(-z_iface / 8000.0)
    p_mid = 101325.0 * np.exp(-z_mid / 8000.0)
    theta = 300.0 + 25.0 * z_mid / 16000.0
    temp = theta * (p_mid / 1.0e5) ** (287.0 / 1004.0)
    rho_dry = p_mid / (287.0 * temp)
    qv = 0.017 * np.exp(-z_mid / 3000.0)

    def cols(profile, jitter=1.0e-3):
        base = np.repeat(profile[:, None], NCOL, axis=1)
        base *= 1.0 + jitter * rng.standard_normal(base.shape)
        return np.ascontiguousarray(base, dtype=np.float32)

    fields = {
        # Winds equal across columns on each level: a C-grid face then destaggers to exactly the cell
        # wind, so the WRF grid's u_phy is the hex column's (the one documented hex difference).
        "u": cols(np.linspace(14.0, 30.0, NZ), jitter=0.0),
        "v": cols(np.linspace(-6.0, 2.0, NZ), jitter=0.0),
        "theta": cols(theta),
        "pressure": cols(p_mid, jitter=0.0),
        "pressure_interface": cols(p_iface, jitter=0.0),
        "z_interface": np.ascontiguousarray(np.repeat(z_iface[:, None], NCOL, axis=1), dtype=np.float32),
        "w": cols(0.05 * np.sin(np.linspace(0, np.pi, NZ + 1)), jitter=0.0),
        "rho_dry": cols(rho_dry, jitter=0.0),
        "qv": cols(qv),
        # Cloud and rain low down: WSM6 rains, so the large-scale wet deposition acts.
        "qc": cols(np.where(z_mid < 3000.0, 1.5e-3, 0.0)),
        "qr": cols(np.where(z_mid < 3000.0, 4.0e-4, 0.0)),
        "qi": np.zeros((NZ, NCOL), dtype=np.float32),
        "qs": np.zeros((NZ, NCOL), dtype=np.float32),
        "qg": np.zeros((NZ, NCOL), dtype=np.float32),
    }
    return {name: cp.asarray(value) for name, value in fields.items()}


#: hex column c is water (xland = 2) in the sea-salt case
WATER = np.array([0, 1, 0, 0, 1, 0, 1, 0], dtype=bool)
#: (wrf_chem block, water columns): WRF-Chem's DUST package; GOCART primary aerosol with dust and sea salt
CASES = {
    "dust": ({"chem_opt": 401, "dust_opt": 1, "wetscav_onoff": -1}, False),
    "gocart_primary": ({"chem_sets": "gocart_primary", "dust_opt": 1, "seas_opt": 1,
                        "wetscav_onoff": -1}, True),
    # Fire smoke with plume rise (RAVE frames, synthetic) on cells of unequal area.
    "smoke": ({"chem_sets": "smoke", "biomass_burn_opt": 1}, False),
}
#: Voronoi cell areas of the smoke case (m2): unequal, so the fire's per-cell mass becomes a flux through
#: each cell's own area (the area-equivalent map factor).
SMOKE_AREA = np.array([2.25e8, 1.9e8, 2.6e8, 2.1e8, 2.4e8, 1.75e8, 2.3e8, 2.0e8])
#: hex columns that burn, the fire's PM2.5 (kg per cell per hour) and FRP (MW)
FIRE_PM25 = np.array([0.0, 4.0e3, 0.0, 9.0e2, 0.0, 0.0, 2.2e4, 0.0])
FIRE_FRP = np.array([0.0, 60.0, 0.0, 15.0, 0.0, 0.0, 420.0, 0.0])


class _Frames:
    """A SourceFrames (woof.core.chem_fire): the trailing 24 posted hours of one fire, on a grid."""

    def __init__(self, to_grid):
        self.reference_time = START.replace(tzinfo=datetime.timezone.utc)
        self.hours = frozenset(self.reference_time - datetime.timedelta(hours=h) for h in range(1, 25))
        self.to_grid = to_grid

    def available_hours(self, source, field):
        return self.hours

    def at(self, source, field, when):
        assert when in self.hours, when
        scale = 0.5 + 0.5 * ((when.hour % 6) / 5.0)
        values = {"PM25": FIRE_PM25, "FRP_MEAN": FIRE_FRP}[field] * scale
        return self.to_grid(values)


def _seam(water=False):
    extra = {"xland": np.where(WATER, 2.0, 1.0), "landmask": np.where(WATER, 0.0, 1.0)} if water else {}
    return mcb.run_mpas_column_batch(**extra,
        n_levels=NZ, n_columns=NCOL, dt=DT, radiation_seconds=600.0, surface_pbl_seconds=DT,
        start_time=START, latitude_deg=np.linspace(20.0, 34.0, NCOL),
        longitude_deg=np.linspace(-10.0, 5.0, NCOL), terrain_height_m=np.zeros(NCOL),
        z_interface_nominal_m=np.linspace(0.0, 16000.0, NZ + 1),
        p_top_pa=float(101325.0 * np.exp(-2.0)), dx_m=15000.0,
        soil_moisture=0.05, vegfra=5.0, isltyp=1, ivgtyp=16)


def _statics():
    rng = np.random.default_rng(11)
    return {"EROD": rng.uniform(0.05, 0.4, (3, NCOL)), "CLAYFRAC": rng.uniform(0.0, 0.3, NCOL),
            "SANDFRAC": rng.uniform(0.4, 0.9, NCOL)}


def _raw(case="dust"):
    return {"schema": "gpuwm-hex.chemistry/v1", "wrf_chem": dict(CASES[case][0])}


def _hex_runtime(seam, raw, area):
    from woof.hex.chem_wrf import HexWrfChemistry, attach_statics
    runtime = HexWrfChemistry("chemistry.json", raw, "0" * 64, cfg=seam._cfg, area=area, nlev=NZ,
                              ncells=NCOL, offset=0, start_time=START, fnm=seam._state.fnm,
                              fnp=seam._state.fnp, terrain=seam._state.ht)
    attach_statics(runtime, lat_deg=None, lon_deg=None, fields=_statics())
    runtime.set_geography(np.linspace(20.0, 34.0, NCOL), np.linspace(-10.0, 5.0, NCOL))
    return runtime


def _to_wrf(array):
    """(..., 1, NCOL) or (..., NCOL) hex layout -> (..., NY, NX) WRF layout with the columns shuffled."""
    flat = array.reshape(*array.shape[:-2], NCOL) if array.shape[-2:] == (1, NCOL) else array
    out = flat[..., cp.asarray(SOURCE)]
    return cp.ascontiguousarray(out.reshape(*flat.shape[:-1], NY, NX))


def _to_hex(array):
    flat = array.reshape(*array.shape[:-2], NY * NX)
    return cp.ascontiguousarray(flat[..., cp.asarray(POS)])


def _wrf_state(seam, cfg):
    """A WRF-shaped grid the WRF chem driver steps, columns shuffled (identity ARW mass coupling)."""
    from woof.core.chem_state import attach_chem_state
    from woof.core.chem_statics import attach_dust_static_fields
    z = lambda *shape: cp.zeros(shape, dtype=cp.float32)
    s = SimpleNamespace()
    s.p, s.thp, s.alt = z(NZ, NY, NX), z(NZ, NY, NX), z(NZ, NY, NX)
    s.qv, s.qc, s.qr = z(NZ, NY, NX), z(NZ, NY, NX), z(NZ, NY, NX)
    s.thb, s.phb = z(NZ), z(NZ + 1)
    s.php, s.w = z(NZ + 1, NY, NX), z(NZ + 1, NY, NX)
    s.u, s.v = z(NZ, NY, NX + 1), z(NZ, NY + 1, NX)
    s.fnm, s.fnp = seam._state.fnm, seam._state.fnp
    s.use_theta_m = 0
    s.ht = z(NY, NX)
    s.msft = cp.ones((NY, NX), dtype=cp.float32)
    s.has_msf = False
    s.mub2d = cp.ones((NY, NX), dtype=cp.float32)
    s.mup = s.mup0 = z(NY, NX)
    s.c1h, s.c2h = cp.ones(NZ, dtype=cp.float32), z(NZ)
    s.dnw = cp.full(NZ, -1.0 / NZ, dtype=cp.float32)
    s.elapsed_seconds = 0.0
    s.nwfa = s.nifa = None
    s.physics = SimpleNamespace(fields={}, microphysics=None)
    attach_chem_state(s, cfg, cp)
    s.chem.start_time = START
    fields = _statics()
    attach_dust_static_fields(s, {k: cp.asnumpy(_to_wrf(cp.asarray(v, dtype=cp.float32)))
                                  for k, v in fields.items()})
    return s


def _wrf_load(s, columns, physics_fields, microphysics):
    nz = NZ
    s.p[...] = _to_wrf(columns["pressure"])
    s.thp[...] = _to_wrf(columns["theta"])
    cp.divide(np.float32(1.0), _to_wrf(columns["rho_dry"]), out=s.alt)
    cp.multiply(_to_wrf(columns["z_interface"]), np.float32(9.81), out=s.php)
    s.w[...] = _to_wrf(columns["w"])
    for name in ("qv", "qc", "qr"):
        getattr(s, name)[...] = _to_wrf(columns[name])
    s.u[...] = columns["u"][:, :1].reshape(nz, 1, 1)
    s.v[...] = columns["v"][:, :1].reshape(nz, 1, 1)
    s.physics.fields = {k: (_to_wrf(v) if isinstance(v, cp.ndarray) and v.shape[-2:] == (1, NCOL) else v)
                        for k, v in physics_fields.items()}
    s.physics.microphysics = (None if microphysics is None else
                              SimpleNamespace(rainncv=_to_wrf(microphysics.rainncv)))


def _columns(inputs, state2):
    return {"pressure": state2["pressure"], "theta": state2["theta"], "rho_dry": state2["rho_dry"],
            "z_interface": state2["z_interface"], "qv": state2["qv"], "qc": state2["qc"],
            "qr": state2["qr"], "u": inputs["u"], "v": inputs["v"], "w": inputs["w"]}


@pytest.fixture(scope="module", params=sorted(CASES))
def run(request):
    """STEPS seam steps; the hex chem step and the WRF chem step on the same columns after each."""
    from woof.core.chem_driver import chem_step
    from woof.core.chem_state import process_attr
    case = request.param
    seam = _seam(CASES[case][1])
    inputs = _profile()
    state2 = {name: inputs[name].copy() for name in
              ("theta", "qv", "qc", "qr", "qi", "qs", "qg", "pressure", "rho_dry", "z_interface")}
    area = SMOKE_AREA if case == "smoke" else np.full(NCOL, 2.25e8)
    runtime = _hex_runtime(seam, _raw(case), area)
    scalars = cp.zeros((len(runtime.rows), NZ, NCOL), dtype=cp.float32)
    wcfg = replace(runtime.cfg, ny=NY, nx=NX)
    wrf = _wrf_state(seam, wcfg)
    if case == "smoke":
        runtime.state.chem.emission_frames = _Frames(lambda v: np.asarray(v, np.float64).reshape(1, NCOL))
        wrf.chem.emission_frames = _Frames(lambda v: np.asarray(v, np.float64)[SOURCE].reshape(NY, NX))
        # The WRF grid's map factor carries each column's cell area, as the hex column's does.
        wrf.msft = _to_wrf(runtime.state.msft)
    history = []
    for step in range(1, STEPS + 1):
        seam.run_phase1(dt=DT, **inputs)
        seam.run_phase2(**state2)
        driver = seam._driver
        columns = _columns(inputs, state2)
        micro = getattr(driver, "microphysics", None)
        runtime.step(scalars, columns, dt=DT, elapsed_seconds=step * DT, physics_fields=driver.fields,
                     microphysics=micro)
        runtime.commit()
        _wrf_load(wrf, columns, {**driver.fields, **runtime.geography}, micro)
        wrf.elapsed_seconds = (step - 1) * DT
        chem_step(wrf, wcfg, DT)
        arrays = {}
        for _key, module in runtime.processes:
            for alloc in getattr(module, "ALLOCATES", ()):
                attr = process_attr(alloc)
                arrays[attr] = (cp.asnumpy(getattr(runtime.state, attr)), cp.asnumpy(getattr(wrf, attr)))
        history.append({
            "rows": [(cp.asnumpy(scalars[i]), cp.asnumpy(getattr(wrf, row.state_attr)))
                     for i, row in enumerate(runtime.rows)],
            "arrays": arrays})
    return SimpleNamespace(case=case, seam=seam, runtime=runtime, scalars=scalars, history=history, inputs=inputs,
                           state2=state2, area=area)


def test_rows_are_bitwise_the_wrf_path_per_column_every_step(run):
    for step, record in enumerate(run.history, 1):
        for i, (hexed, wrf) in enumerate(record["rows"]):
            back = cp.asnumpy(_to_hex(cp.asarray(wrf)))
            assert hexed.tobytes() == back.tobytes(), (step, run.runtime.rows[i].name)


def test_process_arrays_are_bitwise_the_wrf_path_per_column(run):
    for step, record in enumerate(run.history, 1):
        for attr, (hexed, wrf) in record["arrays"].items():
            if attr == "chemwork_wetdep_alpha":
                continue  # a one-word parameter carrier at flat index 0, not a per-column field
            back = cp.asnumpy(_to_hex(cp.asarray(wrf)))
            assert hexed.reshape(back.shape).tobytes() == back.tobytes(), (step, attr)


def test_the_comparison_is_not_vacuous(run):
    last = run.history[-1]
    assert sum(float(h.sum()) for h, _ in last["rows"]) > 0.0, "nothing was emitted"
    if run.case == "smoke":
        smoke = last["rows"][0][0]
        burning = FIRE_PM25 > 0
        # (unburnt columns hold only the 1e-16 floor the fire kernel keeps)
        assert (smoke[:, burning].sum(axis=0) > 1e-3).all() and smoke[:, ~burning].max() < 1e-15
        kmax = [v for k, v in last["arrays"].items() if k.endswith("plume_k_max")][0][0]
        assert (kmax.reshape(-1)[burning] > 2).any(), "no plume rose above the lowest levels"
        return
    edust = [v for k, v in last["arrays"].items() if k.endswith("edust")][0][0]
    assert (edust > 0).any()
    lifted = np.stack([h for h, _ in last["rows"]])
    assert (lifted[:, 1:] > 0).any(), "nothing was mixed above the first level"
    if run.case == "gocart_primary":
        seas = [i for i, row in enumerate(run.runtime.rows) if row.name.startswith("seas_")]
        assert all(lifted[i][:, WATER].sum() > 0 for i in seas), "no sea salt over the water columns"


def test_hex_ledger_closes(run):
    ledger = run.runtime.ledger()
    assert ledger["max_relative_closure"] < 1e-12
    buckets = ("emitted", "scavenged") if run.case == "smoke" else ("emitted", "deposited", "settled", "scavenged")
    for bucket in buckets:
        total = sum(abs(r[bucket + "_kg"]) for r in ledger["rows"].values())
        assert total > 0.0, f"no mass was {bucket}: that process went untested"


def test_products_are_finite_and_named(run):
    columns = _columns(run.inputs, run.state2)
    products = run.runtime.products(run.scalars, columns)
    from woof.hex.chem_wrf import product_names
    names, _units = product_names(run.runtime.table)
    assert set(names) <= set(products)
    for name in names:
        value = cp.asnumpy(products[name])
        assert value.shape == (NCOL,) and np.isfinite(value).all(), name
    first = "smoke_column_mass" if run.case == "smoke" else "dust_1_column_mass"
    assert float(cp.asnumpy(products[first]).sum()) > 0.0


def test_abort_restores_the_process_arrays_and_ledger(run):
    from woof.core.chem_state import ledger_attr
    runtime = run.runtime
    before = {attr: cp.asnumpy(getattr(runtime.state, attr)).tobytes() for attr in runtime.serialized}
    seam, inputs, state2 = run.seam, run.inputs, run.state2
    saved = run.scalars.copy()
    seam_state = seam.export_state()
    seam.run_phase1(dt=DT, **inputs)
    seam.run_phase2(**{k: v.copy() for k, v in state2.items()})
    runtime.step(run.scalars, _columns(inputs, state2), dt=DT, elapsed_seconds=(STEPS + 1) * DT,
                 physics_fields=seam._driver.fields, microphysics=seam._driver.microphysics)
    assert cp.asnumpy(getattr(runtime.state, ledger_attr("emitted"))).tobytes() != before[ledger_attr("emitted")]
    runtime.abort()
    run.scalars[...] = saved
    for attr in runtime.serialized:
        assert cp.asnumpy(getattr(runtime.state, attr)).tobytes() == before[attr], attr
    assert runtime.steps == STEPS
    del seam_state


def test_restart_continues_bit_identically():
    """Two steps, restart payload into a fresh runtime, two more: the same bytes as four straight."""
    def drive(runtime, scalars, seam, inputs, state2, first, last):
        for step in range(first, last + 1):
            seam.run_phase1(dt=DT, **inputs)
            seam.run_phase2(**state2)
            runtime.step(scalars, _columns(inputs, state2), dt=DT, elapsed_seconds=step * DT,
                         physics_fields=seam._driver.fields, microphysics=seam._driver.microphysics)
            runtime.commit()

    def fresh():
        seam = _seam()
        inputs = _profile()
        state2 = {n: inputs[n].copy() for n in
                  ("theta", "qv", "qc", "qr", "qi", "qs", "qg", "pressure", "rho_dry", "z_interface")}
        return seam, inputs, state2

    area = np.full(NCOL, 2.25e8)
    seam, inputs, state2 = fresh()
    straight = _hex_runtime(seam, _raw(), area)
    scalars = cp.zeros((len(straight.rows), NZ, NCOL), dtype=cp.float32)
    drive(straight, scalars, seam, inputs, state2, 1, 4)

    seam2, inputs2, state22 = fresh()
    first = _hex_runtime(seam2, _raw(), area)
    scalars2 = cp.zeros_like(scalars)
    drive(first, scalars2, seam2, inputs2, state22, 1, 2)
    payload = json.loads(json.dumps(first.restart_payload(), default=lambda a: a.tolist()))
    payload["arrays"] = {k: np.asarray(v, dtype=getattr(first.state, k).dtype)
                         for k, v in payload["arrays"].items()}
    resumed = _hex_runtime(seam2, _raw(), area)
    resumed.restore_payload(payload)
    drive(resumed, scalars2, seam2, inputs2, state22, 3, 4)
    assert cp.asnumpy(scalars2).tobytes() == cp.asnumpy(scalars).tobytes()
    for attr in straight.serialized:
        assert (cp.asnumpy(getattr(resumed.state, attr)).tobytes()
                == cp.asnumpy(getattr(straight.state, attr)).tobytes()), attr
