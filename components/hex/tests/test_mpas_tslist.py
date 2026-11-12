"""WRF's per-step station time series on the MPAS suite route (woof.hex.mpasdiag.tslist_v842).

&diagnostics config_tslist names a WRF station file; every completed step each station's nearest cell is sampled
by the WRF route's own sampler (``TslistSampler`` / ``tslist_gather``) and written by its Rust writer.  Here:

* the options resolve (off by default, nothing built), are refused by name when set without the file, and are
  refused on the RRFS fork route and for a checkpoint continuation;
* placement is the nearest cell centre, an outside-the-mesh station is refused, and the plan the Rust writer is
  opened with carries the cell's own latitude, longitude and height;
* the samples equal the pools' own columns at the station's cell (E3: the same words a history frame and the
  ``.wcol`` columns hold for that cell), through the real kernel, with t = 0 never sampled and the hour WRF's;
* the real Rust writer writes WRF's eight files for each station.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from woof.hex.mpasdiag import tslist_v842 as ts
from woof.hex.mpasphys import resolve_namelist

F32 = np.float32


def _namelist(**diagnostics):
    lines = ["&physics", " config_physics_suite='convection_permitting'", "/"]
    if diagnostics:
        lines.append("&diagnostics")
        lines += [f" {key}={value}" for key, value in diagnostics.items()]
        lines.append("/")
    return "\n".join(lines) + "\n"


def _resolve(**diagnostics):
    return resolve_namelist(_namelist(**diagnostics), default_backend="hex-cuda")


def test_options_default_off_and_resolve():
    off = _resolve()
    assert off.tslist is None and "tslist" not in off.receipt()
    config = _resolve(config_tslist="'/data/tslist'", config_max_ts_level=8)
    assert config.tslist == ts.TslistConfig("/data/tslist", 5, 8, 200)
    assert config.receipt()["tslist"] == dict(path="/data/tslist", max_locs=5, max_level=8, buf_size=200)
    assert _resolve(config_nwp_diagnostics=".true.", config_tslist="'/x'").nwp_diagnostics
    with pytest.raises(ValueError, match="has no option"):
        _resolve(config_tslist="'/x'", config_ts_buffer=3)


@pytest.mark.parametrize("diagnostics,match", [
    (dict(config_max_ts_level=4), "without config_tslist"),
    (dict(config_ts_buf_size=10), "without config_tslist"),
    (dict(config_tslist=3), "names the WRF station file"),
    (dict(config_tslist="'/x'", config_max_ts_level=0), "config_max_ts_level is an integer from 1 through 999"),
    (dict(config_tslist="'/x'", config_max_ts_locs=-1), "config_max_ts_locs"),
    (dict(config_tslist="'/x'", config_ts_buf_size=2.5), "config_ts_buf_size"),
])
def test_refusals_are_named(diagnostics, match):
    with pytest.raises(ValueError, match=match):
        _resolve(**diagnostics)


def test_rrfs_fork_route_refuses_the_option():
    from woof.hex.mpasphys.rrfs.driver_rrfs import resolve_namelist as resolve_rrfs
    text = "&physics\n config_physics_suite='hrrrv5'\n/\n&diagnostics\n config_tslist='/x'\n/\n"
    with pytest.raises(ValueError, match="not available on the RRFS fork route"):
        resolve_rrfs(text)


def test_the_off_state_receipt_is_unchanged():
    """Digest recorded on origin/main (6487b6aa3), before the option existed."""
    import hashlib
    import json
    receipt = _resolve().receipt()
    assert hashlib.sha256(json.dumps(receipt, sort_keys=True, default=str).encode()).hexdigest()[:16] == "bb186b77c0898a2c"


# ---------------------------------------------------------------------------
# placement
# ---------------------------------------------------------------------------
def _ring_mesh(nc=36):
    """A ring of cell centres along the equator (neighbours: the two beside it), longitude 0..350."""
    lon = np.radians(np.arange(nc) * (360.0 / nc))
    lat = np.zeros(nc)
    around = np.zeros((6, nc), dtype=np.int32)
    for c in range(nc):
        around[0, c] = (c - 1) % nc + 1
        around[1, c] = (c + 1) % nc + 1
    return lat, lon, around


def test_nearest_cell_and_the_outside_test():
    lat, lon, around = _ring_mesh()
    index, inside = ts.nearest_cells(lat, lon, around, [0.0, 3.0, 40.0, 0.2], [24.0, -3.0, 100.0, 359.0])
    assert index.tolist() == [2, 0, 10, 0] or index.tolist() == [2, 35, 10, 0]
    assert inside.tolist() == [True, True, False, True]
    # a sentinel neighbour (nCells + 1, as the pools hold a missing one) is ignored, not indexed
    around[2, 5] = lat.size + 1
    ts.nearest_cells(lat, lon, around, [0.0], [50.0])


def _station_file(path: Path, rows):
    head = ["# station file", "%-32s%7s %8s" % ("", "LAT", "LON"), "# fixed format A25,1X,A5,1X,F7.3,1X,F8.3"]
    lines = [f"{name:<25} {prefix:<5} {lat:7.3f} {lon:8.3f}" for name, prefix, lat, lon in rows]
    path.write_text("\n".join(head + lines) + "\n")
    return str(path)


def _bridge():
    from woof.tslist_bridge import resolve
    try:
        return resolve()
    except RuntimeError:
        pytest.skip("the Rust tslist library is not built (cargo build --release --offline in tools/tslist_bridge)")


def test_the_plan_carries_the_cells_own_coordinates(tmp_path):
    _bridge()
    lat, lon, around = _ring_mesh()
    height = np.arange(lat.size, dtype=np.float32) * 10.0
    path = _station_file(tmp_path / "tslist", [("Station one", "one", 0.4, 24.7), ("Station two", "two", -0.3, 301.0)])
    plan, cells = ts.placement_plan(path, 5, lat, lon, height, around)
    assert cells.tolist() == [2, 30]
    rows = [row.split("\t") for row in plan.splitlines()]
    assert [(int(r[1]), int(r[2])) for r in rows] == [(3, 1), (31, 1)]
    word = lambda text: np.uint32(int(text, 16)).view(np.float32)
    assert [float(word(r[3])) for r in rows] == pytest.approx([0.0, 0.0], abs=1e-6)
    assert [float(word(r[4])) for r in rows] == pytest.approx([20.0, 300.0], abs=1e-4)
    assert [float(word(r[5])) for r in rows] == [20.0, 300.0]
    with pytest.raises(ValueError, match="lie outside this mesh"):
        ts.placement_plan(_station_file(tmp_path / "far", [("Far", "far", 60.0, 10.0)]), 5, lat, lon, height, around)
    grid = tmp_path / "ij"
    grid.write_text("# x\n" + "%-32s%7s %8s\n" % ("", "I", "J") + "# y\n" + f"{'Grid station':<25} {'g':<5} {3:7d} {4:8d}\n")
    with pytest.raises(ValueError, match="needs geographic"):
        ts.placement_plan(str(grid), 5, lat, lon, height, around)


# ---------------------------------------------------------------------------
# the module: a stand-in owner holding the pools as (level, cell) arrays
# ---------------------------------------------------------------------------
NC, NL = 36, 12


def _pools(seed=3):
    rng = np.random.default_rng(seed)
    lat, lon, around = _ring_mesh(NC)
    zgrid = (np.linspace(0, 12000, NL + 1, dtype=F32)[:, None] + rng.uniform(0, 300, NC).astype(F32)[None, :])
    pb = (np.linspace(98000, 22000, NL, dtype=F32)[:, None] * rng.uniform(0.98, 1.02, NC).astype(F32)[None, :]).astype(F32)
    eta = np.linspace(0, 1, NL + 1)
    fzm = np.zeros(NL, F32)
    fzm[1:] = np.linspace(0.45, 0.55, NL - 1)
    mass = lambda lo, hi: rng.uniform(lo, hi, (NL, NC)).astype(F32)
    scalars = {name: mass(0, 1e-3) for name in ("qv", "qc", "qr", "qi", "qs", "qg")}
    scalars["qv"] = mass(1e-4, 1.2e-2)
    pad = lambda a: np.concatenate([a, a[..., :1]], axis=-1)
    fields = {
        ("mesh", "latCell"): pad(lat.astype(F32)), ("mesh", "lonCell"): pad(lon.astype(F32)),
        ("mesh", "cellsOnCell"): pad(around), ("mesh", "zgrid"): pad(zgrid), ("mesh", "fzm"): fzm,
        ("mesh", "fzp"): (F32(1) - fzm).astype(F32),
        ("diag", "uReconstructZonal"): pad(mass(-20, 30)), ("diag", "uReconstructMeridional"): pad(mass(-20, 20)),
        ("diag", "pressure_base"): pad(pb), ("diag", "pressure_p"): pad(mass(-300, 300)),
        ("state", "w"): pad(rng.uniform(-3, 3, (NL + 1, NC)).astype(F32)),
        ("state", "theta_m"): pad(mass(280, 340)),
        ("state", "scalars"): pad(np.stack([scalars[n] for n in ("qv", "qc", "qr", "qi", "qs", "qg")])),
    }
    surface = {}
    for name, (lo, hi) in dict(t2m=(250, 310), q2=(1e-3, 1e-2), u10=(-15, 15), v10=(-15, 15), surface_pressure=(8e4, 1.02e5),
                               glw=(200, 400), gsw=(0, 900), hfx=(-100, 300), lh=(-50, 500), skintemp=(250, 320),
                               rainc=(0, 5), rainnc=(0, 5)).items():
        surface[name] = pad(rng.uniform(lo, hi, NC).astype(F32))
        fields["diag_physics", name] = surface[name]
    fields["sfc_input", "tslb"] = pad(rng.uniform(260, 300, (4, NC)).astype(F32))
    return dict(fields=fields, scalars=scalars, zgrid=zgrid, pb=pb)


class _CaptureWriter:
    def __init__(self, directory, plan, levels, start):
        self.directory, self.plan, self.levels, self.start = directory, plan, levels, start
        self.batches, self.closed = [], False

    def write_batch(self, values, samples):
        self.batches.append(np.array(values[:samples]))

    def checkpoint(self):
        return "receipt"

    def close(self):
        self.closed = True


def _owner(cp, pools, tmp_path, with_qg=True):
    fields = {key: cp.asarray(value) for key, value in pools["fields"].items()}
    rows = [dict(pool=p, name=n, pool_name=n) for (p, n) in fields if p in ("diag_physics", "diag", "sfc_input")]
    names = ["qv", "qc", "qr", "qi", "qs", "qg"]
    indices = {("state", "index_" + n): i + 1 for i, n in enumerate(names) if with_qg or n != "qg"}
    session = SimpleNamespace(layout=SimpleNamespace(dimensions=dict(nCellsSolve=NC, nVertLevels=NL), fields=rows,
                                                     indices=indices), config=None)
    writer = SimpleNamespace(start=datetime(2024, 3, 1, 12, 0, 0), output=tmp_path)
    return SimpleNamespace(cp=cp, session=session, authority=lambda pool, name: fields[pool, name], writer=writer,
                           fields=fields)


@pytest.fixture
def cp():
    return pytest.importorskip("cupy")


def _module(cp, tmp_path, levels=4, stations=((0.3, 24.0), (0.1, 301.0), (-0.2, 150.0)), buf=2):
    _bridge()
    pools = _pools()
    owner = _owner(cp, pools, tmp_path)
    path = _station_file(tmp_path / "tslist", [(f"Station {i}", f"s{i}", la, lo) for i, (la, lo) in enumerate(stations)])
    writers = []

    def factory(directory, plan, nlevel, start):
        writers.append(_CaptureWriter(directory, plan, nlevel, start))
        return writers[-1]

    module = ts.TslistModule(owner, ts.TslistConfig(path, 5, levels, buf), writer_factory=factory)
    module.setup()
    return module, owner, pools, writers[0]


def test_stations_sit_on_their_nearest_cells(cp, tmp_path):
    module, owner, pools, writer = _module(cp, tmp_path)
    assert module.cells.tolist() == [2, 30, 15]
    assert writer.start == "2024-03-01_12:00:00" and writer.levels == 4 and writer.directory == str(tmp_path)


def _reference(pools, cell, levels, hour):
    """The row the WRF kernel makes of one cell's pool column (wrf_timeseries.F calc_ts), host arithmetic."""
    f = pools["fields"]
    col = lambda pool, name, n=NL: f[pool, name][:n, cell]
    p = col("diag", "pressure_base") + col("diag", "pressure_p")
    zg = col("mesh", "zgrid", NL + 1)
    phb = zg * F32(9.81)
    qv = pools["scalars"]["qv"][:, cell]
    theta = col("state", "theta_m") / (F32(1.0) + (F32(461.6) / F32(287.0)) * qv)
    w = col("state", "w", NL + 1)
    row = np.zeros(15 + 7 * levels, dtype=F32)
    row[0] = hour
    surface = [f["diag_physics", n][cell] for n in ("t2m", "q2", "u10", "v10", "surface_pressure", "glw", "gsw", "hfx", "lh", "skintemp")]
    row[1:11] = surface
    row[11] = f["sfc_input", "tslb"][0, cell]
    row[12], row[13] = f["diag_physics", "rainc"][cell], f["diag_physics", "rainnc"][cell]
    k = np.arange(levels)
    row[15:15 + levels] = col("diag", "uReconstructZonal")[:levels]
    row[15 + levels:15 + 2 * levels] = col("diag", "uReconstructMeridional")[:levels]
    row[15 + 2 * levels:15 + 3 * levels] = ((w[:-1] + w[1:]) / F32(2.0))[:levels]
    row[15 + 3 * levels:15 + 4 * levels] = (F32(0.5) * ((phb + 0)[:-1] + (phb + 0)[1:]) / F32(9.81))[:levels]
    row[15 + 4 * levels:15 + 5 * levels] = theta[:levels]
    row[15 + 5 * levels:15 + 6 * levels] = qv[:levels]
    row[15 + 6 * levels:15 + 7 * levels] = p[:levels]
    return row


def test_samples_equal_the_pools_columns_at_the_stations_cells(cp, tmp_path):
    module, owner, pools, writer = _module(cp, tmp_path, levels=4, buf=2)
    when = owner.writer.start
    for step in (0, 1, 2, 3):
        module.begin(step, when + timedelta(seconds=30 * step))
        module.update()
    assert module.samples == 3  # t = 0 is never sampled
    module.sampler.checkpoint()
    batches = np.concatenate(writer.batches)
    assert batches.shape == (3, 3, 15 + 7 * 4)
    for sample, step in enumerate((1, 2, 3)):
        minutes = F32(0.5 * step)
        hour = F32(minutes / F32(60.0))
        for station, cell in enumerate(module.cells):
            ref = _reference(pools, int(cell), 4, hour)
            got = batches[sample, station]
            assert got[0] == hour
            # the profile words (profiles are pure copies/blends of the pool column) and the surface words
            np.testing.assert_allclose(got[1:14], ref[1:14], rtol=0, atol=0)
            np.testing.assert_allclose(got[15:], ref[15:], rtol=2e-6, atol=0)
    # the column water of the five condensate species, from calc_p8w's interface pressures (host mirror)
    cell = int(module.cells[0])
    p = (pools["fields"]["diag", "pressure_base"] + pools["fields"]["diag", "pressure_p"])[:NL, cell].astype(np.float64)
    fnm, fnp = pools["fields"]["mesh", "fzm"].astype(np.float64), pools["fields"]["mesh", "fzp"].astype(np.float64)
    z = pools["zgrid"][:, cell].astype(np.float64)
    p8w = np.zeros(NL + 1)
    p8w[1:NL] = fnm[1:NL] * p[1:NL] + fnp[1:NL] * p[:NL - 1]
    z0, z1, z2 = z[0], 0.5 * (z[0] + z[1]), 0.5 * (z[1] + z[2])
    w1 = (z0 - z2) / (z1 - z2)
    p8w[0] = w1 * p[0] + (1 - w1) * p[1]
    z0, z1, z2 = z[NL], 0.5 * (z[NL - 1] + z[NL]), 0.5 * (z[NL - 2] + z[NL - 1])
    w1 = (z0 - z2) / (z1 - z2)
    p8w[NL] = np.exp(w1 * np.log(p[NL - 1]) + (1 - w1) * np.log(p[NL - 2]))
    clw = sum(float((pools["scalars"][n][:, cell] * (p8w[:-1] - p8w[1:])).sum()) for n in ("qc", "qr", "qi", "qs", "qg")) / 9.81
    np.testing.assert_allclose(batches[0, 0, 14], clw, rtol=2e-5)


def test_a_checkpoint_continuation_is_refused_by_name():
    from woof.hex.mpasphys.resident_output_diagnostics_v842 import ResidentOutputDiagnostics
    owner = ResidentOutputDiagnostics.__new__(ResidentOutputDiagnostics)
    owner.tslist = object()
    owner.nwp = None
    with pytest.raises(ValueError, match="checkpoint continuation of a run with &diagnostics config_tslist"):
        owner.resume_accumulators({})


def test_the_rust_writer_writes_wrfs_eight_files_per_station(cp, tmp_path):
    _bridge()
    pools = _pools()
    owner = _owner(cp, pools, tmp_path)
    path = _station_file(tmp_path / "tslist", [("Station A", "sta", 0.3, 24.0), ("Station B", "stb", 0.1, 301.0)])
    module = ts.TslistModule(owner, ts.TslistConfig(path, 5, 3, 200))
    module.setup()
    for step in (0, 1, 2):
        module.begin(step, owner.writer.start + timedelta(seconds=60 * step))
        module.update()
    module.close()
    names = sorted(p.name for p in tmp_path.iterdir() if p.name.startswith(("sta.", "stb.")))
    assert names == sorted(f"{prefix}.d01.{ext}" for prefix in ("sta", "stb")
                           for ext in ("TS", "UU", "VV", "WW", "PH", "TH", "QV", "PR"))
    lines = (tmp_path / "sta.d01.TS").read_text().splitlines()
    assert lines[0].startswith("Station A") and "(  0.300,  24.000)" in lines[0] and "2024-03-01_12:00:00" in lines[0]
    rows = [line.split() for line in lines[1:] if line.strip()]
    assert [float(r[1]) for r in rows] == pytest.approx([1 / 60, 2 / 60], abs=1e-6)  # ts_hour of steps 1 and 2 (60 s)
    assert len(rows) == 2 and all(len(r) == 5 + 14 for r in rows)  # id, hour, station, i, j, then the 14 surface words
