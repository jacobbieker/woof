"""Hex observation nudging (woof.hex.obs_nudging): controls, refusals, the cell KD-tree and the reader.

Device-free.  The kernels are checked against WRF's in tests/test_hex_obs_nudging_gpu.py.
Every observation here is synthetic (tests/hex_obs_synthetic.py).
"""
from datetime import datetime
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from woof.hex import obs_nudging as obs

import hex_obs_synthetic as synthetic

TOML = """[fdda]
obs_nudge_opt = 1
max_obs = 500
fdda_start = 0.0
fdda_end = 120.0
obs_nudge_wind = 1
obs_coef_wind = 6.0e-4
obs_nudge_temp = 1
obs_coef_temp = 6.0e-4
obs_rinxy = 40.0
obs_twindo = 0.6666667
obs_ionf = 2
"""


def _toml(tmp_path, text=TOML):
    path = tmp_path / "obs.toml"
    path.write_text(text)
    return path


def test_toml_fdda_table_reads_wrf_keys_and_defaults(tmp_path):
    cfg = obs.load_config(_toml(tmp_path))
    assert cfg.obs_nudge_opt == 1 and cfg.obs_rinxy == 40.0 and cfg.obs_ionf == 2
    assert cfg.obs_nudge_mois == 0 and cfg.obs_sfcfacr == 1.0 and cfg.obs_dpsmx == 7.5  # RunConfig defaults
    assert cfg.obs_inputfile == str((tmp_path / "OBS_DOMAIN101").resolve())
    assert obs.refusals(cfg) == []


def test_toml_run_table_and_input_override(tmp_path):
    cfg = obs.load_config(_toml(tmp_path, TOML.replace("[fdda]", "[run]")), input_file=tmp_path / "x" / "OBS")
    assert cfg.obs_coef_temp == 6.0e-4
    assert cfg.obs_inputfile == str(tmp_path / "x" / "OBS")


def test_unknown_key_is_refused_by_name(tmp_path):
    with pytest.raises(obs.HexObsNudgingRefusal, match="HEX_OBS_NUDGING_KEYS"):
        obs.load_config(_toml(tmp_path, TOML + "obs_rinxyy = 3.0\n"))


def test_namelist_fdda_group(tmp_path):
    path = tmp_path / "namelist.input"
    path.write_text("&fdda\n obs_nudge_opt = 1, 0\n max_obs = 300, 300\n fdda_start = 0., 0.\n"
                    " fdda_end = 60., 60.\n obs_nudge_wind = 1, 1\n obs_coef_wind = 6.E-4, 6.E-4\n"
                    " obs_rinxy = 50., 50.\n obs_twindo = 0.5, 0.5\n obs_ionf = 1, 1\n"
                    " obs_inputfile = 'OBS_A'\n/\n")
    cfg = obs.load_config(path)
    assert (cfg.obs_nudge_opt, cfg.max_obs, cfg.obs_rinxy, cfg.obs_twindo) == (1, 300, 50.0, 0.5)
    assert cfg.obs_inputfile == str((tmp_path / "OBS_A").resolve())
    path.write_text(path.read_text().replace("obs_nudge_opt = 1, 0", "obs_nudge_opt = 1, 1"))
    with pytest.raises(obs.HexObsNudgingRefusal, match="HEX_OBS_NUDGING_DOMAIN"):
        obs.load_config(path)


@pytest.mark.parametrize("extra, code", [
    ("obs_scl_neg_qv_innov = 1\n", "HEX_OBS_NUDGING_NEG_QV"),
    ("obs_rinxy = 0.0\n", "HEX_OBS_NUDGING_CONFIG"),
    ("obs_idynin = 1\n", "HEX_OBS_NUDGING_CONFIG"),
])
def test_refusals_by_name(tmp_path, extra, code):
    text = TOML.replace("obs_rinxy = 40.0\n", "") if "rinxy" in extra else TOML
    cfg = obs.load_config(_toml(tmp_path, text + extra))
    assert code in [problem.code for problem in obs.refusals(cfg)]


def test_nothing_nudged_is_refused(tmp_path):
    text = TOML.replace("obs_nudge_wind = 1", "obs_nudge_wind = 0").replace("obs_nudge_temp = 1", "obs_nudge_temp = 0")
    assert [p.code for p in obs.refusals(obs.load_config(_toml(tmp_path, text)))] == ["HEX_OBS_NUDGING_NO_FIELD"]


# ----------------------------------------------------------------------------
# geometry on a small synthetic Delaunay mesh
# ----------------------------------------------------------------------------

def _mesh(center=(40.0, -100.0), half=1.5, step=0.25, bdy=False):
    """Cells on a lat/lon lattice, triangulated (scipy Delaunay), MPAS-style zero-based connectivity."""
    from scipy.spatial import Delaunay

    lat = np.arange(center[0] - half, center[0] + half + 1e-9, step)
    lon = np.arange(center[1] - half, center[1] + half + 1e-9, step)
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    la, lo = la.ravel(), lo.ravel()
    jitter = np.random.default_rng(0).uniform(-0.03, 0.03, (2, la.size))
    la, lo = la + jitter[0], lo + jitter[1]
    triangles = Delaunay(np.stack([lo, la], axis=1)).simplices
    n = la.size
    vertices: list[list[int]] = [[] for _ in range(n)]
    for v, tri in enumerate(triangles):
        for c in tri:
            vertices[c].append(v)
    width = max(len(v) for v in vertices)
    voc = np.full((n, width), -1, dtype=np.int64)
    for c, row in enumerate(vertices):
        voc[c, :len(row)] = row
    arrays = {"latCell": np.radians(la), "lonCell": np.radians(lo), "verticesOnCell": voc,
              "nEdgesOnCell": np.array([len(v) for v in vertices]), "cellsOnVertex": triangles.astype(np.int64)}
    if bdy:
        edge = (np.abs(la - center[0]) > half - 0.3) | (np.abs(lo - center[1]) > half - 0.3)
        arrays["bdyMaskCell"] = np.where(edge, 7, 0)
    return SimpleNamespace(n_cells=n, arrays=arrays, attrs={"sphere_radius": 6371229.0})


def test_barycentric_weights_reproduce_a_linear_field_and_place_the_report():
    mesh = _mesh()
    locator = obs.MeshLocator(mesh)
    rng = np.random.default_rng(1)
    lat = 40.0 + rng.uniform(-1.0, 1.0, 50)
    lon = -100.0 + rng.uniform(-1.0, 1.0, 50)
    near, corners, weights, admitted = locator.locate(lat, lon)
    assert admitted.all()
    assert np.allclose(weights.sum(axis=1), 1.0, atol=1e-6) and (weights >= 0).all()
    # The containing cell is the nearest centre (WRF's nearest grid point for kpbl).
    xyz = obs._unit(np.radians(lat), np.radians(lon))
    assert np.array_equal(near, np.argmax(xyz @ locator.xyz.T, axis=1))
    # A field linear in (x, y, z) is reproduced at the report's gnomonic projection onto the triangle's
    # plane, which is the report itself to O(edge^2) (about 1e-5 here).
    field = locator.xyz @ np.array([0.3, -1.1, 2.0])
    target = xyz @ np.array([0.3, -1.1, 2.0])
    interpolated = (weights * field[corners]).sum(axis=1)
    assert np.allclose(interpolated, target, atol=1e-4)
    # At a cell centre the weight is one on that cell: WRF's bilinear weights at a grid point.
    c_lat, c_lon = np.degrees(mesh.arrays["latCell"][77]), np.degrees(mesh.arrays["lonCell"][77])
    near, corners, weights, _ = locator.locate(np.array([c_lat]), np.array([c_lon]))
    assert near[0] == 77
    assert weights[0][list(corners[0]).index(77)] == pytest.approx(1.0, abs=1e-6)


def test_specified_zone_reports_are_not_admitted():
    mesh = _mesh(bdy=True)
    locator = obs.MeshLocator(mesh)
    _, _, _, admitted = locator.locate(np.array([40.0, 41.45]), np.array([-100.0, -100.0]))
    assert admitted.tolist() == [True, False]


def test_reach_lists_are_great_circle_distances_within_the_radius():
    mesh = _mesh()
    locator = obs.MeshLocator(mesh)
    lat, lon = np.array([40.1, 39.6]), np.array([-100.2, -99.5])
    radius = 60e3
    start, report, rsq = locator.reach(lat, lon, radius)
    assert start[0] == 0 and start[-1] == report.size and np.all(np.diff(start) >= 0)
    cells = np.repeat(np.arange(mesh.n_cells), np.diff(start))
    d = obs.great_circle_m(locator.radius, np.radians(lat[report]), np.radians(lon[report]),
                           mesh.arrays["latCell"][cells], mesh.arrays["lonCell"][cells])
    assert np.allclose(np.sqrt(rsq.astype(np.float64)) * obs.DISTANCE_UNIT_M, d, rtol=1e-6)
    assert d.max() <= radius * (1 + 1e-5)
    # Every cell inside the radius is listed, ascending report order within a cell (WRF's order).
    all_d = obs.great_circle_m(locator.radius, np.radians(lat)[:, None], np.radians(lon)[:, None],
                               mesh.arrays["latCell"][None, :], mesh.arrays["lonCell"][None, :])
    listed = set(zip(report.tolist(), cells.tolist()))
    inside = set(zip(*np.nonzero(all_d < radius * (1 - 1e-6))))
    assert inside <= listed and len(inside) > 20
    for c in range(mesh.n_cells):
        assert np.all(np.diff(report[start[c]:start[c + 1]]) > 0)


def test_planar_limit_of_the_spherical_distance():
    """On a small patch at the equator the great-circle distance is the grid distance WRF uses."""
    radius = 6371229.0
    dx = 3000.0
    dlon = dx / radius
    for i, j in ((0, 0), (3, 4), (-7, 2), (12, -5)):
        d = obs.great_circle_m(radius, 0.0, 0.0, j * dlon, i * dlon)
        assert d / dx == pytest.approx(math.hypot(i, j), rel=2e-5)


# ----------------------------------------------------------------------------
# the WRF route's Rust reader in geographic mode
# ----------------------------------------------------------------------------

def _reader():
    from woof import bridges

    binary = bridges.find_artifact("GPUWM_RW_FDDAOBS", bridges.executable_name("rw_fddaobs"),
                                   crate_relative=bridges.RUSTWX_CRATE_RELATIVE)
    if binary is None:
        pytest.skip("rw_fddaobs is not built")
    return binary


def test_reader_geographic_mode_carries_longitude_latitude(tmp_path):
    _reader()
    path = synthetic.write(tmp_path / "OBS_DOMAIN101", [
        synthetic.Surface("20240101000000", 40.25, -99.75, 300.0, temperature=290.0, u=3.0, v=-1.0, rh=50.0,
                          psfc=97000.0),
        synthetic.Sounding("20240101003000", 39.5, 260.5, 300.0,
                           [(85000., 1500., 285., 10., 1., 40.), (50000., 5500., 260., 20., 2., 30.)]),
    ])
    cfg = obs.load_config(_toml(tmp_path), input_file=path)
    try:
        rows = obs.read_observations(cfg, original_start=datetime(2024, 1, 1), dt=30.0)
    except obs.HexObsNudgingRefusal as error:
        if "predates the geographic mode" in str(error):
            pytest.skip(str(error))
        raise
    assert rows["rko"].tolist() == [1.0, -99.0, -99.0]
    assert rows["latitude"].tolist() == [40.25, 39.5, 39.5]
    assert rows["longitude"].tolist() == [-99.75, -99.5, -99.5]
    assert rows["time"].tolist() == [0.0, 0.5, 0.5]
    assert rows["nlev"].tolist() == [1.0, 2.0, 2.0] and rows["lev"].tolist() == [1.0, 1.0, 2.0]
    # Earth-relative winds stay as reported; pressures in kPa (WRF's carrier units).
    assert rows["varobs"][0, :2].tolist() == [3.0, -1.0]
    assert rows["varobs"][0, 4] == pytest.approx(97.0)
    assert rows["varobs"][1, 4] == pytest.approx(85.0)


# ----------------------------------------------------------------------------
# door refusals
# ----------------------------------------------------------------------------

def test_adaptive_step_refuses_obs_nudging_by_name():
    from woof.hex import adaptive_dt

    names = [row[1] for row in adaptive_dt.INCOMPATIBLE]
    assert "--obs-nudging" in names


def test_door_flags_and_partitioned_refusal():
    import argparse

    from woof.hex import forecast_door

    parser = argparse.ArgumentParser()
    forecast_door.add_forecast_arguments(parser)
    args = parser.parse_known_args(["--obs-nudging", "a.toml", "--obs-nudging-input", "OBS"])[0]
    assert args.obs_nudging == Path("a.toml") and args.obs_nudging_input == Path("OBS")
    request = SimpleNamespace(nudging=None, obs_nudging=Path("a.toml"), tc_climate=None, w_damping=None,
                              inflow_perturbation=None)
    assert forecast_door._partitioned_samehost_refusal(request).startswith("HEX_OBS_NUDGING_PARTITIONED")
