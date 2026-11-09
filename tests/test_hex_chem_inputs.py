"""Input chemistry on the hex route: ``chem_in_opt`` initial composition and ``have_bcs_chem`` boundaries.

The hex wrf_chem block takes WRF's two switches; they enable the rows' declared boundary sources exactly as
the WRF namelist door does (woof.namelist_chem), and the WRF path's own remap
(woof.chem_source_init.fill_boundary_sources) fills the rows on the mesh cells as a ``(1, ncells)`` target.
The CAMS fixture is the synthetic ECMWF-encoded pair the WRF tests use (tests/data/chem_cams/).

What is asserted:

* the switches parse as WRF's and select the sources the namelist door selects; bad values are refused;
* the initial fill writes the hex scalar rows in place and equals the WRF path's fill on the same cells
  word for word;
* the boundary reader adds ``lbc_<row>`` to each boundary time, equal to the WRF path's fill at that time
  on the file's own equation-of-state pressure, and leaves the file's met words alone; the pool admits the
  rows only once a run declares them, and the rows must follow the whole driven moisture block.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from woof.hex import chem_wrf

DATA = Path(__file__).resolve().parent / "data" / "chem_cams"
CYCLE = datetime(2025, 7, 30, 0, tzinfo=timezone.utc)
NZ, NCOL = 20, 8


def _raw(**switches):
    return {"schema": "gpuwm-hex.chemistry/v1", "wrf_chem": {"chem_sets": "cams_aq", **switches}}


def test_the_switches_select_the_namelist_doors_sources():
    from woof.namelist_chem import _sources_for
    assert chem_wrf.chem_sources("cams_aq", {"chem_sets": "cams_aq"}) == ()
    on = chem_wrf.chem_sources("cams_aq", {"chem_in_opt": 1})
    assert on == _sources_for("cams_aq", {"chem_in_opt": 1}) and on[0] == "cams-global"
    assert chem_wrf.chem_sources("cams_aq", {"have_bcs_chem": True}) == on
    name, table, _ = chem_wrf.load_table(_raw(chem_in_opt=1))
    assert name == "cams_aq" and "cams-global" in table.sources
    meta = json.loads(chem_wrf.history_metadata(_raw(chem_in_opt=1, have_bcs_chem=1), "0" * 64))
    assert meta["input_chemistry"] == {"chem_in_opt": 1, "have_bcs_chem": 1}


@pytest.mark.parametrize("key,value", [("chem_in_opt", 2), ("chem_in_opt", True), ("have_bcs_chem", 3),
                                       ("have_bcs_chem", "yes")])
def test_a_switch_takes_only_wrfs_off_or_on(key, value):
    with pytest.raises(ValueError, match=key):
        chem_wrf.load_table(_raw(**{key: value}))


def test_the_boundary_names_follow_the_rows_and_only_with_have_bcs_chem():
    assert chem_wrf.lbc_scalar_names(_raw(chem_in_opt=1)) == ()
    assert chem_wrf.lbc_scalar_names(_raw(have_bcs_chem=1)) == ("lbc_o3", "lbc_no2", "lbc_co", "lbc_so2")


def test_the_rows_must_follow_the_whole_driven_block(monkeypatch):
    from woof.hex import chem_transport
    monkeypatch.setitem(chem_transport._CONFIGS, "chemistry_x", (None, _raw(have_bcs_chem=1), None, None, None))
    wsm6 = ("qv", "qc", "qr", "qi", "qs", "qg")
    species = (*wsm6, "o3", "no2", "co", "so2")
    full = tuple("lbc_" + n for n in wsm6)
    assert chem_transport.drive_chemistry_rows("chemistry_x", full, wsm6, species) == (
        *full, "lbc_o3", "lbc_no2", "lbc_co", "lbc_so2")
    # A stream carrying only qv, qc and qr would leave the rows behind undriven species.
    with pytest.raises(ValueError, match="whole driven block"):
        chem_transport.drive_chemistry_rows("chemistry_x", full[:3], wsm6, species)
    assert chem_transport.drive_chemistry_rows("wsm6_column", full[:3], wsm6, wsm6) == full[:3]


def test_the_pool_admits_supplied_rows_once_declared():
    from woof.hex.lbc import LbcFileError, LbcInventory, LbcPool, declare_supplied_scalars
    inventory = LbcInventory.__new__(LbcInventory)
    with pytest.raises(LbcFileError, match="undeclared"):
        LbcPool(inventory, scalar_names=("lbc_qv", "lbc_zz_test_row"))
    declare_supplied_scalars(("lbc_zz_test_row",))
    LbcPool(inventory, scalar_names=("lbc_qv", "lbc_zz_test_row"))
    with pytest.raises(LbcFileError, match="stream variables"):
        declare_supplied_scalars(("lbc_qc",))


def test_the_boundary_pressure_is_mpas_equation_of_state():
    rho, theta, qv = 1.1, 300.0, 0.01
    fields = {"lbc_rho": np.full((3, 2), rho, np.float32), "lbc_theta": np.full((3, 2), theta, np.float32),
              "lbc_qv": np.full((3, 2), qv, np.float32)}
    p = chem_wrf.lbc_pressure(fields)
    assert p.shape == (2, 3)
    rgas, rv, cp = 287.0, 461.6, 1004.5
    r, t, q = (float(np.float32(v)) for v in (rho, theta, qv))
    expect = 1.0e5 * (rgas * r * t * (1 + rv / rgas * q) / 1.0e5) ** (cp / (cp - rgas))
    np.testing.assert_allclose(p, expect, rtol=1e-12)


# -- the CAMS fixture ------------------------------------------------------------------------------------

def _backend():
    from woof.ingest.cpu_backend import CpuPreprocessBackend
    try:
        backend = CpuPreprocessBackend()
    except Exception as error:  # the library is not built on this box
        pytest.skip(f"CPU preprocessing bridge unavailable: {error}")
    for name in ("weighted_combination", "hybrid_full_pressure"):
        if not callable(getattr(backend, name, None)):
            pytest.skip(f"bridge predates {name}")
    return backend


@pytest.fixture(scope="module")
def cache(tmp_path_factory):
    from woof import chem_table, data_store_fetch as dsf
    _backend()
    try:
        from woof import grib2_stack  # noqa: F401
    except ImportError:
        pytest.skip("grib2_stack front absent")
    ml, sl = sorted(DATA.glob("SYNTHETIC-cams-ml-*.grib2")), sorted(DATA.glob("SYNTHETIC-cams-sl-*.grib2"))
    if not ml or not sl:
        pytest.skip("synthetic CAMS fixture absent")
    north, west, south, east = json.loads(next(DATA.glob("SYNTHETIC-cams-*.manifest.json")).read_text())[
        "area_nwse"]
    source = chem_table.catalog().sources["cams-global"]
    plan = dsf.resolve_acquisition(source, area=f"{south + 0.05},{west + 0.05},{north - 0.05},{east - 0.05}",
                                   start=CYCLE, end=CYCLE + timedelta(hours=1), now=CYCLE + timedelta(days=30))
    groups = {r.group: r.sha256 for r in plan.requests}
    root = tmp_path_factory.mktemp("cams") / "cache"
    dsf.fetch(plan, licence="CC-BY-4.0", cache_root=root, progress=lambda *_: None,
              files={groups["model_level"]: ml[0], groups["single_level"]: sl[0]},
              files_note="SYNTHETIC test fixture")
    return root


#: hex cells inside the fixture's area, in no grid order
LAT = np.array([30.62, 30.95, 30.71, 31.03, 30.55, 30.88, 30.79, 30.66])
LON = np.array([-95.31, -95.88, -95.52, -95.44, -95.73, -95.36, -95.61, -95.80])


def _pressure(n=NZ):
    column = np.linspace(99000.0, 15000.0, n)
    jitter = 1.0 + 1.0e-3 * np.sin(np.arange(NCOL))[None, :]
    return (column[:, None] * jitter).astype(np.float32)


def _wrf_fill(table, sources, pressure, valid, cache, frame):
    """The WRF path's fill on the same cells as a (1, ncells) grid."""
    from woof.chem_source_init import fill_boundary_sources
    state = SimpleNamespace(chem=SimpleNamespace(table=table))
    for row in table.rows:
        setattr(state, row.state_attr, np.zeros((pressure.shape[0], 1, NCOL), np.float32))
    fill_boundary_sources(state, SimpleNamespace(chem_sources=",".join(sources)), valid_time=valid,
                          latlon=(LAT.reshape(1, NCOL), LON.reshape(1, NCOL)),
                          pressure=pressure.reshape(-1, 1, NCOL), cache_root=cache, backend=_backend(),
                          frame=frame)
    return state


class _BoundaryRuntime:
    coupling = "none"

    def __init__(self, raw):
        self.raw = raw
        self.set_name, self.table, _ = chem_wrf.load_table(raw)
        self.rows = self.table.transported
        sources = chem_wrf.chem_sources(self.set_name, raw["wrf_chem"])
        self.cfg = SimpleNamespace(chem_sources=",".join(sources))


def test_the_boundary_reader_adds_the_rows_and_keeps_the_met_words(cache):
    from woof.hex.lbc import LbcFile
    rng = np.random.default_rng(5)
    met = {"lbc_rho": rng.uniform(0.3, 1.2, (NCOL, NZ)).astype(np.float32),
           "lbc_theta": rng.uniform(290.0, 340.0, (NCOL, NZ)).astype(np.float32),
           "lbc_qv": rng.uniform(1e-4, 1.5e-2, (NCOL, NZ)).astype(np.float32),
           "lbc_u": rng.standard_normal((3 * NCOL, NZ)).astype(np.float32),
           "lbc_w": np.zeros((NCOL, NZ + 1), np.float32)}
    met["lbc_rho"].sort(axis=1)
    met["lbc_rho"] = np.ascontiguousarray(met["lbc_rho"][:, ::-1])
    valid = (CYCLE + timedelta(minutes=30)).replace(tzinfo=None)
    asked = {}

    def base(path, *, extra_scalars=()):
        asked["extra"] = tuple(extra_scalars)
        return LbcFile(Path(path), valid, dict(met))

    raw = _raw(have_bcs_chem=1)
    runtime = _BoundaryRuntime(raw)
    reader = chem_wrf.BoundaryReader(runtime, latitude=LAT, longitude=LON, base=base, cache_root=cache,
                                     backend=_backend())
    admitted = reader("lbc.2025-07-30_00.30.00.nc", extra_scalars=("lbc_qc", *reader.names))
    assert asked["extra"] == ("lbc_qc",)
    for name, value in met.items():
        assert admitted.fields[name] is value
    pressure = chem_wrf.lbc_pressure(met).astype(np.float32)
    wrf = _wrf_fill(runtime.table, runtime.cfg.chem_sources.split(","), pressure, valid, cache, "boundary")
    for name, row in zip(reader.names, runtime.rows):
        got = admitted.fields[name]
        assert got.shape == (NCOL, NZ) and got.dtype == np.float32
        expect = getattr(wrf, row.state_attr).reshape(NZ, NCOL).T
        assert np.array_equal(got, expect), name
        assert np.all(got > 0), name
    receipt = reader.receipt()
    assert receipt["rows"] == list(reader.names) and len(receipt["frames"]) == 1


def test_the_initial_fill_writes_the_hex_rows_as_the_wrf_path_fills_its_grid(cache):
    cp = pytest.importorskip("cupy")
    from woof.core import mpas_column_batch as mcb
    from woof.hex.chem_wrf import HexWrfChemistry
    start = datetime(2025, 7, 30, 1, 0)
    seam = mcb.run_mpas_column_batch(
        n_levels=NZ, n_columns=NCOL, dt=60.0, radiation_seconds=600.0, surface_pbl_seconds=60.0,
        start_time=start, latitude_deg=LAT, longitude_deg=LON, terrain_height_m=np.zeros(NCOL),
        z_interface_nominal_m=np.linspace(0.0, 16000.0, NZ + 1), p_top_pa=float(101325.0 * np.exp(-2.0)),
        dx_m=15000.0, soil_moisture=0.2, vegfra=50.0, isltyp=4, ivgtyp=10)
    raw = _raw(chem_in_opt=1)
    runtime = HexWrfChemistry("chemistry.json", raw, "0" * 64, cfg=seam._cfg, area=np.full(NCOL, 2.0e8),
                              nlev=NZ, ncells=NCOL, offset=2, start_time=start, fnm=seam._state.fnm,
                              fnp=seam._state.fnp, terrain=seam._state.ht)
    scalars = cp.zeros((2 + len(runtime.rows), NZ, NCOL), cp.float32)
    scalars[0] = 7.0
    pressure = _pressure()
    valid = start.replace(tzinfo=timezone.utc)
    receipts = runtime.fill_initial(scalars, latitude=LAT, longitude=LON, pressure=cp.asarray(pressure),
                                    valid_time=valid, cache_root=cache, backend=_backend())
    assert [r["source"] for r in receipts] == ["cams-global"]
    assert bool(cp.all(scalars[0] == 7.0)) and bool(cp.all(scalars[1] == 0.0))
    wrf = _wrf_fill(runtime.table, runtime.sources, pressure, valid, cache, "initial")
    for i, row in enumerate(runtime.rows):
        got = cp.asnumpy(scalars[2 + i])
        assert np.array_equal(got, getattr(wrf, row.state_attr).reshape(NZ, NCOL)), row.name
        assert np.all(got > 0), row.name
    assert runtime.receipt()["initial_composition"] == receipts


def test_the_initial_fill_refuses_without_a_covering_file(tmp_path):
    cp = pytest.importorskip("cupy")
    from woof.chem_source_init import ChemSourceUnavailable
    from woof.core import mpas_column_batch as mcb
    from woof.hex.chem_wrf import HexWrfChemistry
    _backend()
    start = datetime(2025, 7, 30, 1, 0)
    seam = mcb.run_mpas_column_batch(
        n_levels=NZ, n_columns=NCOL, dt=60.0, radiation_seconds=600.0, surface_pbl_seconds=60.0,
        start_time=start, latitude_deg=LAT, longitude_deg=LON, terrain_height_m=np.zeros(NCOL),
        z_interface_nominal_m=np.linspace(0.0, 16000.0, NZ + 1), p_top_pa=float(101325.0 * np.exp(-2.0)),
        dx_m=15000.0, soil_moisture=0.2, vegfra=50.0, isltyp=4, ivgtyp=10)
    runtime = HexWrfChemistry("chemistry.json", _raw(chem_in_opt=1), "0" * 64, cfg=seam._cfg,
                              area=np.full(NCOL, 2.0e8), nlev=NZ, ncells=NCOL, offset=0, start_time=start,
                              fnm=seam._state.fnm, fnp=seam._state.fnp, terrain=seam._state.ht)
    scalars = cp.zeros((len(runtime.rows), NZ, NCOL), cp.float32)
    with pytest.raises(ChemSourceUnavailable, match="woof.data_store_fetch fetch"):
        runtime.fill_initial(scalars, latitude=LAT, longitude=LON, pressure=_pressure(),
                             valid_time=start.replace(tzinfo=timezone.utc), cache_root=tmp_path / "empty",
                             backend=_backend())


def test_the_pool_admits_the_readers_rows_beside_the_stream():
    """The pool's own admission hands the reader's rows on under their pool names (the device driving state
    stacks ``admitted.fields[name]`` for every pool scalar)."""
    from woof.hex.lbc import LbcFile, LbcPool, declare_supplied_scalars
    declare_supplied_scalars(("lbc_zz_pool_row",))
    when = datetime(2025, 7, 30, 0)
    met = {name: np.full((4, 3), 1.0, np.float32) for name in ("lbc_qv", "lbc_rho", "lbc_theta")}
    row = np.arange(12, dtype=np.float32).reshape(4, 3)

    def reader(path, *, extra_scalars=()):
        assert "lbc_zz_pool_row" in extra_scalars
        return LbcFile(Path(path), when, {**met, "lbc_zz_pool_row": row})

    inventory = SimpleNamespace(latest_before=lambda moment: Path("lbc.nc"))
    pool = LbcPool(inventory, scalar_names=("lbc_qv", "lbc_zz_pool_row"), reader=reader)
    admitted = pool.start(when)
    assert admitted.fields["lbc_zz_pool_row"] is row


# -- aerosol-aware Thompson coupling (aerosol_mp_coupling = "diagnose") ---------------------------------

def test_a_coupling_block_runs_beside_the_thompson_aerosol_row(tmp_path):
    from woof.hex import chem_transport
    from woof.hex.species_row import THOMPSON_AEROSOL_SPECIES_ROW
    raw = {"schema": "gpuwm-hex.chemistry/v1",
           "wrf_chem": {"chem_opt": 401, "chem_in_opt": 1, "aerosol_mp_coupling": "diagnose"}}
    assert chem_wrf.physics_row(raw) == "thompson_aero_column"
    assert chem_wrf.physics_row(_raw()) == "wsm6_column"
    path = tmp_path / "chemistry.json"
    path.write_text(json.dumps(raw))
    row = chem_transport.configure(path)
    rows = tuple(r.name for r in chem_wrf.declare(raw))
    assert row.appended_scalar_names == ("nwfa", "nifa", *rows)
    assert row.driven_scalar_names == ("qv", "qc", "qr", "qi", "qs", "qg", "ni", "nr", "nc")
    species = chem_transport._CONFIGS[row.name][3]
    assert species.names() == (*THOMPSON_AEROSOL_SPECIES_ROW.names(), *rows)
    args = SimpleNamespace(chemistry_config=path, physics_backend="wsm6_column", physics_suite=None,
                           physics=None)
    with pytest.raises(ValueError, match="would replace it"):
        chem_transport.admit_arguments(args)
    args.physics_backend = "thompson_aero_column"
    chem_transport.admit_arguments(args)
    # Under have_bcs_chem the numbers ride the pool before the rows, as they sit in the species.
    bcs = {**raw, "wrf_chem": {**raw["wrf_chem"], "have_bcs_chem": 1}}
    assert chem_wrf.lbc_scalar_names(bcs)[:3] == ("lbc_nwfa", "lbc_nifa", "lbc_" + rows[0])
    with pytest.raises(ValueError, match="aerosol_mp_coupling"):
        chem_wrf.load_table({"wrf_chem": {"chem_opt": 401, "aerosol_mp_coupling": "auto"}})


def _synthetic_source(monkeypatch):
    """The data-store fill with a synthetic remap (the CAMS fixture carries no aerosol): every row the
    source fills gets a smooth positive profile; everything else in fill_boundary_sources runs."""
    import woof.chem_source_init as init

    def fields(table, source_row, frames, target, valid, **kwargs):
        from woof.chem_boundary_remap import boundary_rows
        rows = boundary_rows(table, source_row.name, available=kwargs.get("available"),
                             frame=kwargs.get("frame"))
        nz = target.pressure.shape[0]
        shape = target.pressure.shape
        out = {}
        for n, row in enumerate(rows):
            profile = (1.0 + n) * np.exp(-np.arange(nz) / 6.0)[:, None, None]
            out[row.name] = (profile * (1.0 + 0.1 * np.cos(target.longitude))[None]).astype(np.float32
                                                                                          ).reshape(shape)
        return out, {"source": source_row.name, "rows": [row.name for row in rows]}

    monkeypatch.setattr(init, "covering_files", lambda *a, **k: {"model_level": Path("synthetic.grib2")})
    monkeypatch.setattr(init, "_frames", lambda *a, **k: None)
    monkeypatch.setattr(init, "files_cycle", lambda *a, **k: None)
    monkeypatch.setattr(init, "boundary_fields_for_grid", fields)


def test_the_coupled_start_diagnoses_the_numbers_from_the_source_composition(monkeypatch):
    cp = pytest.importorskip("cupy")
    from woof.core import mpas_column_batch as mcb
    from woof.core.chem_mp_coupling import launch_niwfa, pack_terms, rows as coupled_rows
    from woof.hex.chem_wrf import HexWrfChemistry
    from woof.hex.species_row import AEROSOL_MASS, SpeciesDeclaration, THOMPSON_AEROSOL_SPECIES_ROW
    _synthetic_source(monkeypatch)
    start = datetime(2025, 7, 30, 1, 0)
    seam = mcb.run_mpas_column_batch(
        n_levels=NZ, n_columns=NCOL, dt=60.0, radiation_seconds=600.0, surface_pbl_seconds=60.0,
        start_time=start, latitude_deg=LAT, longitude_deg=LON, terrain_height_m=np.zeros(NCOL),
        z_interface_nominal_m=np.linspace(0.0, 16000.0, NZ + 1), p_top_pa=float(101325.0 * np.exp(-2.0)),
        dx_m=15000.0, soil_moisture=0.2, vegfra=50.0, isltyp=4, ivgtyp=10, microphysics_scheme="thompson_aero")
    raw = {"schema": "gpuwm-hex.chemistry/v1",
           "wrf_chem": {"chem_opt": 401, "chem_in_opt": 1, "aerosol_mp_coupling": "diagnose"}}
    runtime = HexWrfChemistry("chemistry.json", raw, "0" * 64, cfg=seam._cfg, area=np.full(NCOL, 2.25e8),
                              nlev=NZ, ncells=NCOL, offset=11, start_time=start, fnm=seam._state.fnm,
                              fnp=seam._state.fnp, terrain=seam._state.ht)
    runtime.species_row = THOMPSON_AEROSOL_SPECIES_ROW.with_extras(tuple(
        SpeciesDeclaration(r.name, AEROSOL_MASS, True, r.name.upper(), "row") for r in runtime.rows))
    scalars = cp.zeros((11 + len(runtime.rows), NZ, NCOL), cp.float32)
    receipts = runtime.fill_initial(scalars, latitude=LAT, longitude=LON, pressure=_pressure(),
                                    valid_time=start.replace(tzinfo=timezone.utc))
    assert receipts[0]["thompson_initialization"]["initialized_fields"] == ["nifa"]
    index = THOMPSON_AEROSOL_SPECIES_ROW.index_of
    nifa = cp.asnumpy(scalars[index("nifa")])
    assert (nifa > 0).all(), "dust feeds the ice-friendly number"
    assert not cp.asnumpy(scalars[index("nwfa")]).any(), "no dust row feeds the water-friendly number"
    for i, row in enumerate(runtime.rows):
        assert (cp.asnumpy(scalars[11 + i]) > 0).all(), row.name
    # The GSL operator itself on the rows the fill wrote: the numbers are its words.
    terms = pack_terms(coupled_rows(runtime.table))
    expect_nifa, expect_nwfa = cp.zeros((NZ, 1, NCOL), cp.float32), cp.zeros((NZ, 1, NCOL), cp.float32)
    launch_niwfa({t.name: cp.ascontiguousarray(scalars[11 + [r.name for r in runtime.rows].index(t.name)]
                                               ).reshape(NZ, 1, NCOL) for t in terms}, terms,
                 expect_nifa, expect_nwfa)
    assert np.array_equal(cp.asnumpy(expect_nifa).reshape(NZ, NCOL), nifa)


def test_the_coupled_boundary_drives_numbers_diagnosed_from_its_composition(monkeypatch):
    from woof.hex.lbc import LbcFile
    _synthetic_source(monkeypatch)
    _backend()
    raw = {"schema": "gpuwm-hex.chemistry/v1",
           "wrf_chem": {"chem_opt": 401, "have_bcs_chem": 1, "aerosol_mp_coupling": "diagnose"}}

    class Runtime(_BoundaryRuntime):
        coupling = "diagnose"

    runtime = Runtime(raw)
    runtime.cfg = SimpleNamespace(chem_sources=runtime.cfg.chem_sources, aerosol_mp_coupling="diagnose",
                                  mp_physics=28, chem_sets="dust")
    met = {"lbc_rho": np.full((NCOL, NZ), 1.0, np.float32), "lbc_theta": np.full((NCOL, NZ), 300.0, np.float32),
           "lbc_qv": np.full((NCOL, NZ), 0.01, np.float32), "lbc_nwfa": np.full((NCOL, NZ), 5.0, np.float32)}
    valid = datetime(2025, 7, 30, 3)
    reader = chem_wrf.BoundaryReader(runtime, latitude=LAT, longitude=LON,
                                     base=lambda path, extra_scalars=(): LbcFile(Path(path), valid, dict(met)))
    admitted = reader("lbc.nc", extra_scalars=reader.names)
    assert reader.names[:2] == ("lbc_nwfa", "lbc_nifa")
    assert (admitted.fields["lbc_nifa"] > 0).all()
    # No dust row feeds nwfa, so the operator leaves it to the scheme and the stream's own field drives it.
    assert np.array_equal(admitted.fields["lbc_nwfa"], met["lbc_nwfa"])
    del met["lbc_nwfa"]
    with pytest.raises(ValueError, match="carries no lbc_nwfa"):
        reader("lbc.nc", extra_scalars=reader.names)

def test_a_regional_attach_pads_the_garbage_column_and_fills_the_start(monkeypatch):
    """The regional physics runs one garbage column past the mesh cells: the geography and the initial fill
    take the donor column's copy (woof.hex.cuda_regional_forecast_v841.pad_regional_physics_column)."""
    cp = pytest.importorskip("cupy")
    ncells = 5
    seen = {}

    class Runtime:
        raw = _raw(chem_in_opt=1, have_bcs_chem=1)
        switches = {"chem_in_opt": 1, "have_bcs_chem": 1}
        restart_provider_receipt = None
        steps, offset, rows = 0, 0, (1, 2)
        species_row = SimpleNamespace(names=lambda: ("qv",))

        def __init__(self):
            self.ncells = ncells + 1

        def set_geography(self, lat, lon):
            seen["geography"] = (np.asarray(lat), np.asarray(lon))

        def fill_initial(self, scalars, **kwargs):
            seen["fill"] = kwargs

    runtime = Runtime()
    monkeypatch.setattr(chem_wrf, "attach_statics", lambda *a, **k: None)
    monkeypatch.setattr(chem_wrf, "install_boundary_reader",
                        lambda rt, driver, latitude, longitude: seen.setdefault("reader", latitude.size))
    lat = np.radians(np.linspace(30.0, 31.0, ncells))
    mesh = SimpleNamespace(arrays={"latCell": lat, "lonCell": -lat}, attrs={"regional_garbage_donor_cell": 1})
    backend = SimpleNamespace(
        chemistry=runtime, _seam=SimpleNamespace(_driver=SimpleNamespace(fields={"xland": cp.ones(ncells + 1)})),
        _prepare=lambda atmosphere, names, post_rk: SimpleNamespace(pres_hyd_p="p"))
    driver = SimpleNamespace(atmosphere=SimpleNamespace(state=SimpleNamespace(scalars=cp.zeros((2, 3, ncells + 1)))))
    host = {"regional": {}, "prepared": SimpleNamespace(mesh=mesh), "start_time_text": "2025-07-30_01:00:00"}
    chem_wrf.attach(backend, host, driver)
    filled = seen["fill"]["latitude"]
    assert filled.size == ncells + 1 and filled[-1] == filled[1] and seen["fill"]["pressure"] == "p"
    assert seen["reader"] == ncells
