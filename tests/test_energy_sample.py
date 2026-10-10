"""wrfout site sampling (``woof.energy.sample``) against a numpy oracle.

The sampler's arithmetic is the Rust ``rw-sitesample`` crate; these tests
write a small synthetic wrfout-like file (Lambert attributes, staggered
dimensions, ``Times``) and grade ``sample_wrfout`` against an independent
numpy transcription of the documented contract that lives only here:
destagger, rotate to earth-relative at mass points, bilinear at the site,
linear in height above model terrain, NaN outside.

Tests that call the library skip with the reason when it is not built;
the refusals that happen before the library is reached run everywhere.
"""

from __future__ import annotations

from pathlib import Path

import netCDF4
import numpy as np
import pytest

from woof import bridge_assets, bridges
from woof.energy import sample, sample_bridge
from woof.energy.sample import (PROFILE_VARS, SURFACE_VARS, SampleUnavailable,
                                available_variables, sample_wrfout)
from woof.static.projection import projection_class

REPO_ROOT = Path(__file__).resolve().parents[1]

_REASON = sample_bridge.unavailable_reason()
needs_library = pytest.mark.skipif(
    _REASON is not None,
    reason=f"the rw-sitesample library is not built here ({_REASON}); "
           "cd tools/rustwx && cargo build --release -p rw-sitesample "
           "--offline")

G = 9.81
NX, NY, NZ = 12, 10, 6
# float32-exact reference so the attributes the file stores are the grid
GRID = dict(ref_lat=float(np.float32(51.6)), ref_lon=float(np.float32(-3.4)),
            truelat1=50.0, truelat2=55.0,
            stand_lon=-3.5, dx=100.0, dy=100.0, e_we=NX + 1, e_sn=NY + 1)
Z_W = np.array([0.0, 10.0, 25.0, 50.0, 90.0, 140.0, 200.0])
HEIGHTS = [2.0, 10.0, 60.0, 150.0, 250.0]


def _grid(**overrides):
    params = {**GRID, **overrides}
    return projection_class("lambert")(**params)


def _fields(seed: int) -> dict[str, np.ndarray]:
    """One record of every field, shapes without the Time axis."""
    rng = np.random.default_rng(seed)
    j, i = np.mgrid[0:NY, 0:NX].astype(np.float64)
    hgt = 100.0 + 5.0 * i + 3.0 * j
    z_w = hgt[None] + Z_W[:, None, None] * (1.0 + 0.01 * j[None])
    phb = G * (z_w - 1.0)
    ph = G * (1.0 + rng.uniform(-0.5, 0.5, z_w.shape))
    alpha = 0.2 + 0.05 * i + 0.02 * j        # large, varying rotation
    fields = {
        "HGT": hgt, "PH": ph, "PHB": phb,
        "U": rng.uniform(-10, 10, (NZ, NY, NX + 1)),
        "V": rng.uniform(-10, 10, (NZ, NY + 1, NX)),
        "W": rng.uniform(-1, 1, (NZ + 1, NY, NX)),
        "T": rng.uniform(-5, 5, (NZ, NY, NX)),
        "P": rng.uniform(-200, 200, (NZ, NY, NX)),
        "PB": 100000.0 - 12.0 * Z_W[:NZ, None, None] + 0.0 * j[None],
        "SINALPHA": np.sin(alpha), "COSALPHA": np.cos(alpha),
    }
    fields["PB"] = np.broadcast_to(fields["PB"], (NZ, NY, NX)).copy()
    for name in ("QVAPOR", "QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP"):
        fields[name] = rng.uniform(0, 1e-3, (NZ, NY, NX))
    for name in SURFACE_VARS:
        fields[name] = rng.uniform(-5, 5, (NY, NX))
    return fields


def write_wrfout(path: Path, times: list[str], *, seed: int = 0,
                 drop: tuple[str, ...] = (), attrs: dict | None = None,
                 grid=None) -> list[dict[str, np.ndarray]]:
    """A wrfout-like file with ``len(times)`` records; returns the fields."""
    grid = grid or _grid()
    lat, lon = grid.latlon_mass()
    records = [_fields(seed + n) for n in range(len(times))]
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("Time", None)
        ds.createDimension("DateStrLen", 19)
        ds.createDimension("west_east", NX)
        ds.createDimension("south_north", NY)
        ds.createDimension("bottom_top", NZ)
        ds.createDimension("west_east_stag", NX + 1)
        ds.createDimension("south_north_stag", NY + 1)
        ds.createDimension("bottom_top_stag", NZ + 1)
        values = {"MAP_PROJ": np.int32(1), "TRUELAT1": np.float32(50.0),
                  "TRUELAT2": np.float32(55.0), "STAND_LON": np.float32(-3.5),
                  "CEN_LAT": np.float32(51.6), "CEN_LON": np.float32(-3.4),
                  "MOAD_CEN_LAT": np.float32(51.6), "DX": np.float32(100.0),
                  "DY": np.float32(100.0),
                  "WEST-EAST_GRID_DIMENSION": np.int32(NX + 1),
                  "SOUTH-NORTH_GRID_DIMENSION": np.int32(NY + 1)}
        values.update(attrs or {})
        for key, value in values.items():
            ds.setncattr(key, value)
        t = ds.createVariable("Times", "S1", ("Time", "DateStrLen"))
        for n, text in enumerate(times):
            t[n] = np.array(list(text), dtype="S1")
        dims = sample._SOURCE_DIMS
        for name in ("XLAT", "XLONG"):
            if name in drop:
                continue
            v = ds.createVariable(name, "f4", dims[name])
            for n in range(len(times)):
                v[n] = lat if name == "XLAT" else lon
        for name in records[0]:
            if name in drop:
                continue
            v = ds.createVariable(name, "f4", dims[name])
            for n, rec in enumerate(records):
                v[n] = rec[name]
    # what the file holds, at file precision
    return [{k: v.astype(np.float32).astype(np.float64)
             for k, v in rec.items()} for rec in records]


# ---------------------------------------------------------------------------
# The numpy oracle (tests only)
# ---------------------------------------------------------------------------

def _bilinear(plane: np.ndarray, fi: float, fj: float) -> float:
    i0 = min(int(np.floor(fi)), plane.shape[-1] - 2)
    j0 = min(int(np.floor(fj)), plane.shape[-2] - 2)
    wx, wy = fi - i0, fj - j0
    return ((1 - wy) * ((1 - wx) * plane[..., j0, i0] + wx * plane[..., j0, i0 + 1])
            + wy * ((1 - wx) * plane[..., j0 + 1, i0]
                    + wx * plane[..., j0 + 1, i0 + 1]))


def _vertical(column: np.ndarray, zagl: np.ndarray, heights) -> np.ndarray:
    out = np.interp(heights, zagl, column)
    out[(np.asarray(heights) < zagl[0]) | (np.asarray(heights) > zagl[-1])] = np.nan
    return out


def oracle(rec: dict[str, np.ndarray], fi: float, fj: float, heights):
    inside = 0 <= fi <= NX - 1 and 0 <= fj <= NY - 1
    if not inside:
        return None
    um = 0.5 * (rec["U"][..., :-1] + rec["U"][..., 1:])
    vm = 0.5 * (rec["V"][..., :-1, :] + rec["V"][..., 1:, :])
    s, c = rec["SINALPHA"], rec["COSALPHA"]
    z = (rec["PH"] + rec["PHB"]) / G
    zagl = _bilinear(0.5 * (z[:-1] + z[1:]), fi, fj) - _bilinear(rec["HGT"], fi, fj)
    columns = {
        "U": _bilinear(um * c - vm * s, fi, fj),
        "V": _bilinear(vm * c + um * s, fi, fj),
        "W": _bilinear(0.5 * (rec["W"][:-1] + rec["W"][1:]), fi, fj),
        "THETA": _bilinear(rec["T"] + 300.0, fi, fj),
        "PRES": _bilinear(rec["P"] + rec["PB"], fi, fj),
    }
    for name in ("QVAPOR", "QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP"):
        columns[name] = _bilinear(rec[name], fi, fj)
    profile = {k: _vertical(col, zagl, heights) for k, col in columns.items()}
    surface = {k: _bilinear(rec[k], fi, fj) for k in SURFACE_VARS}
    surface["U10"] = _bilinear(rec["U10"] * c - rec["V10"] * s, fi, fj)
    surface["V10"] = _bilinear(rec["V10"] * c + rec["U10"] * s, fi, fj)
    return profile, surface, _bilinear(rec["HGT"], fi, fj)


def _sites(fractional: list[tuple[float, float]]):
    lat, lon = _grid().ij_to_latlon(np.array([f[0] + 1 for f in fractional]),
                                    np.array([f[1] + 1 for f in fractional]))
    return np.asarray(lat), np.asarray(lon)


_FRACTIONAL = [(0.02, 0.03), (3.3, 4.7), (NX - 1.01, NY - 1.02), (5.5, 0.25),
               (7.01, 6.99), (NX - 1.0 + 0.4, 3.0), (-0.6, 2.0)]


# ---------------------------------------------------------------------------
# Library-backed behaviour
# ---------------------------------------------------------------------------

@needs_library
def test_sampling_matches_the_numpy_oracle_in_time_order(tmp_path):
    later = write_wrfout(tmp_path / "b.nc",
                         ["2026-01-01_00:30:00", "2026-01-01_00:45:00"], seed=10)
    earlier = write_wrfout(tmp_path / "a.nc", ["2026-01-01_00:15:00"], seed=1)
    lat, lon = _sites(_FRACTIONAL)
    result = sample_wrfout([tmp_path / "b.nc", tmp_path / "a.nc"], lat, lon,
                           HEIGHTS)
    assert result.times.tolist() == [
        np.datetime64("2026-01-01T00:15:00"), np.datetime64("2026-01-01T00:30:00"),
        np.datetime64("2026-01-01T00:45:00")]
    assert result.inside.tolist() == [True] * 5 + [False] * 2
    assert set(result.profile) == set(PROFILE_VARS)
    assert set(result.surface) == set(SURFACE_VARS)
    assert result.dx_m == 100.0
    records = earlier + later
    for t, rec in enumerate(records):
        for s, (fi, fj) in enumerate(_FRACTIONAL):
            expected = oracle(rec, fi, fj, HEIGHTS)
            if expected is None:
                for key in PROFILE_VARS:
                    assert np.isnan(result.profile[key][t, s]).all()
                for key in SURFACE_VARS:
                    assert np.isnan(result.surface[key][t, s])
                assert np.isnan(result.terrain_m[s])
                continue
            profile, surface, terrain = expected
            for key in PROFILE_VARS:
                np.testing.assert_allclose(result.profile[key][t, s],
                                           profile[key], rtol=2e-6,
                                           atol=1e-9, err_msg=key)
            for key in SURFACE_VARS:
                np.testing.assert_allclose(result.surface[key][t, s],
                                           surface[key], rtol=2e-6,
                                           atol=1e-6, err_msg=key)
            if t == 0:
                np.testing.assert_allclose(result.terrain_m[s], terrain,
                                           rtol=1e-6)
    # 2 m is below the lowest mass level, 250 m above the highest
    assert np.isnan(result.profile["U"][:, :, 0]).all()
    assert np.isnan(result.profile["U"][:, :, -1]).all()
    assert np.isfinite(result.profile["U"][:, :5, 1:4]).all()
    assert any("2 m is outside the mass-level column" in n for n in result.notes)
    assert any("2 of 7 site(s) outside" in n for n in result.notes)


@needs_library
def test_rotation_turns_grid_wind_to_earth_relative(tmp_path):
    """A pure grid-x wind under a 90 degree rotation is earth-northward."""
    write_wrfout(tmp_path / "w.nc", ["2026-01-01_00:00:00"])
    with netCDF4.Dataset(tmp_path / "w.nc", "a") as ds:
        ds["U"][0] = np.full((NZ, NY, NX + 1), 5.0)
        ds["V"][0] = np.zeros((NZ, NY + 1, NX))
        ds["SINALPHA"][0] = np.ones((NY, NX))
        ds["COSALPHA"][0] = np.zeros((NY, NX))
    lat, lon = _sites([(4.5, 4.5)])
    result = sample_wrfout([tmp_path / "w.nc"], lat, lon, [60.0],
                           profile_vars=("U", "V"), surface_vars=())
    np.testing.assert_allclose(result.profile["U"][0, 0, 0], 0.0, atol=1e-6)
    np.testing.assert_allclose(result.profile["V"][0, 0, 0], 5.0, rtol=1e-6)


@needs_library
def test_missing_variables_are_omitted_and_named(tmp_path):
    write_wrfout(tmp_path / "m.nc", ["2026-01-01_00:00:00"],
                 drop=("SWDDNI", "COSZEN", "QGRAUP"))
    available = available_variables([tmp_path / "m.nc"])
    assert {"SWDDNI", "COSZEN", "QGRAUP"}.isdisjoint(available)
    assert {"U", "THETA", "U10", "SWDOWN"} <= available
    lat, lon = _sites([(2.0, 2.0)])
    result = sample_wrfout([tmp_path / "m.nc"], lat, lon, [60.0])
    assert "QGRAUP" not in result.profile and "SWDDNI" not in result.surface
    assert "QVAPOR" in result.profile and "SWDOWN" in result.surface
    assert "omitted SWDDNI: the history lacks SWDDNI" in result.notes
    assert "omitted QGRAUP: the history lacks QGRAUP" in result.notes


@needs_library
def test_no_site_inside_samples_nothing_and_says_so(tmp_path):
    write_wrfout(tmp_path / "o.nc", ["2026-01-01_00:00:00"])
    result = sample_wrfout([tmp_path / "o.nc"], np.array([10.0]),
                           np.array([20.0]), [60.0])
    assert result.inside.tolist() == [False]
    assert np.isnan(result.profile["U"]).all()
    assert any("1 of 1 site(s) outside" in n for n in result.notes)


@needs_library
@pytest.mark.parametrize("cells", [0.5, 1.0])
def test_projection_that_disagrees_with_xlat_is_refused(tmp_path, cells):
    """Coordinates written for a grid shifted east of what the attributes
    describe; half a cell is the stagger slip the check exists to catch."""
    ref_lat, ref_lon = _grid().ij_to_latlon(GRID["e_we"] / 2.0 - cells,
                                            GRID["e_sn"] / 2.0)
    shifted = _grid(ref_lat=float(ref_lat), ref_lon=float(ref_lon))
    write_wrfout(tmp_path / "x.nc", ["2026-01-01_00:00:00"], grid=shifted)
    lat, lon = _sites([(5.0, 5.0)])
    with pytest.raises(SampleUnavailable, match="cells away"):
        sample_wrfout([tmp_path / "x.nc"], lat, lon, [60.0])


@needs_library
def test_non_finite_coordinates_at_a_site_are_refused(tmp_path):
    write_wrfout(tmp_path / "n.nc", ["2026-01-01_00:00:00"])
    with netCDF4.Dataset(tmp_path / "n.nc", "a") as ds:
        ds["XLAT"][0, 5, 5] = np.nan
    lat, lon = _sites([(5.1, 4.9)])
    with pytest.raises(SampleUnavailable, match="not a finite position"):
        sample_wrfout([tmp_path / "n.nc"], lat, lon, [60.0])


@needs_library
def test_reading_only_the_levels_the_heights_need_changes_nothing(tmp_path):
    write_wrfout(tmp_path / "a.nc", ["2026-01-01_00:00:00",
                                     "2026-01-01_00:15:00"], seed=3)
    lat, lon = _sites(_FRACTIONAL)
    full = sample_wrfout([tmp_path / "a.nc"], lat, lon, [10.0, 60.0, 250.0])
    low = sample_wrfout([tmp_path / "a.nc"], lat, lon, [10.0, 60.0])
    assert any(f"lowest {NZ} of {NZ} levels" in n for n in full.notes)
    assert any("lowest 4 of 6 levels" in n for n in low.notes), low.notes
    for key in PROFILE_VARS:
        np.testing.assert_array_equal(low.profile[key],
                                      full.profile[key][:, :, :2], err_msg=key)


@needs_library
def test_bridge_refuses_inconsistent_shapes():
    with pytest.raises(sample_bridge.SiteSampleBridgeError, match="site heights"):
        sample_bridge.profile(np.zeros((3, 4, 4)), [1.0], [1.0],
                              np.zeros((1, 2)), [10.0])
    with pytest.raises(sample_bridge.SiteSampleBridgeError, match="not one grid"):
        sample_bridge.wind_profile(np.zeros((3, 4, 5)), np.zeros((3, 4, 4)),
                                   np.zeros((4, 4)), np.ones((4, 4)), [1.0],
                                   [1.0], np.zeros((1, 3)), [10.0])
    with pytest.raises(sample_bridge.SiteSampleBridgeError, match="unknown stagger"):
        sample_bridge.profile(np.zeros((3, 4, 4)), [1.0], [1.0],
                              np.zeros((1, 3)), [10.0], stagger=9)


# ---------------------------------------------------------------------------
# Refusals before the library is reached
# ---------------------------------------------------------------------------

def test_duplicate_times_are_refused(tmp_path):
    write_wrfout(tmp_path / "a.nc", ["2026-01-01_00:00:00"])
    write_wrfout(tmp_path / "b.nc", ["2026-01-01_00:00:00", "2026-01-01_00:15:00"])
    lat, lon = _sites([(2.0, 2.0)])
    with pytest.raises(SampleUnavailable, match="appears twice"):
        sample_wrfout([tmp_path / "a.nc", tmp_path / "b.nc"], lat, lon, [10.0])


def test_files_from_two_domains_are_refused(tmp_path):
    write_wrfout(tmp_path / "a.nc", ["2026-01-01_00:00:00"])
    write_wrfout(tmp_path / "b.nc", ["2026-01-01_00:15:00"],
                 attrs={"DX": np.float32(200.0), "DY": np.float32(200.0)})
    lat, lon = _sites([(2.0, 2.0)])
    with pytest.raises(SampleUnavailable, match="not the same domain"):
        sample_wrfout([tmp_path / "a.nc", tmp_path / "b.nc"], lat, lon, [10.0])


def test_an_unimplemented_projection_is_refused(tmp_path):
    write_wrfout(tmp_path / "ll.nc", ["2026-01-01_00:00:00"],
                 attrs={"MAP_PROJ": np.int32(6)})
    lat, lon = _sites([(2.0, 2.0)])
    with pytest.raises(SampleUnavailable, match="MAP_PROJ=6"):
        sample_wrfout([tmp_path / "ll.nc"], lat, lon, [10.0])


def test_history_without_coordinates_supplies_nothing(tmp_path):
    write_wrfout(tmp_path / "n.nc", ["2026-01-01_00:00:00"], drop=("XLAT",))
    assert available_variables([tmp_path / "n.nc"]) == set()
    lat, lon = _sites([(2.0, 2.0)])
    with pytest.raises(SampleUnavailable, match="lacks XLAT"):
        sample_wrfout([tmp_path / "n.nc"], lat, lon, [10.0])


def test_nothing_requested_can_be_supplied(tmp_path):
    write_wrfout(tmp_path / "s.nc", ["2026-01-01_00:00:00"],
                 drop=("SWDDNI", "SWDDIF"))
    lat, lon = _sites([(2.0, 2.0)])
    with pytest.raises(SampleUnavailable, match="none of the requested keys"):
        sample_wrfout([tmp_path / "s.nc"], lat, lon, [10.0],
                      profile_vars=(), surface_vars=("SWDDNI", "SWDDIF"))


def test_available_variables_needs_every_file(tmp_path):
    write_wrfout(tmp_path / "a.nc", ["2026-01-01_00:00:00"])
    write_wrfout(tmp_path / "b.nc", ["2026-01-01_00:15:00"], drop=("COSZEN",))
    assert "COSZEN" in available_variables([tmp_path / "a.nc"])
    assert "COSZEN" not in available_variables([tmp_path / "a.nc",
                                                tmp_path / "b.nc"])
    with pytest.raises(SampleUnavailable, match="not found"):
        available_variables([tmp_path / "missing.nc"])
    with pytest.raises(SampleUnavailable, match="no wrfout files"):
        available_variables([])


@pytest.mark.parametrize("kwargs, match", [
    (dict(profile_vars=("WSPD",)), "not profile output key"),
    (dict(surface_vars=("U",)), "not surface output key"),
    (dict(heights_m=[-1.0]), "finite metres"),
    (dict(heights_m=[float("nan")]), "finite metres"),
    (dict(lat=np.array([1.0, 2.0])), "one length"),
    (dict(lat=np.array([np.nan])), "finite"),
])
def test_malformed_arguments_are_value_errors(tmp_path, kwargs, match):
    write_wrfout(tmp_path / "a.nc", ["2026-01-01_00:00:00"])
    lat, lon = _sites([(2.0, 2.0)])
    call = dict(paths=[tmp_path / "a.nc"], lat=lat, lon=lon, heights_m=[10.0])
    call.update(kwargs)
    with pytest.raises(ValueError, match=match):
        sample_wrfout(**call)


def test_a_missing_library_names_how_to_build_it(tmp_path, monkeypatch):
    monkeypatch.setenv(sample_bridge.SITESAMPLE_BRIDGE_ENV,
                       str(tmp_path / "librw_sitesample.so"))
    monkeypatch.setattr(sample_bridge, "_LIBRARY", None)
    with pytest.raises(sample_bridge.SiteSampleBridgeMissing) as caught:
        sample_bridge.load()
    message = str(caught.value)
    assert sample_bridge.SITESAMPLE_BRIDGE_ENV in message
    assert "pixi run build-native" in message
    assert "cargo build --release --locked --offline -p rw-sitesample" in message
    assert sample_bridge.unavailable_reason().startswith(
        "SiteSampleBridgeMissing")


def test_a_library_with_another_abi_is_refused(tmp_path, monkeypatch):
    """The version probe is asked before any signature is bound."""

    class Probe:
        argtypes = None
        restype = None

        def __call__(self):
            return 99

    class Fake:
        gpuwm_sitesample_abi_version = Probe()

    fake = tmp_path / "librw_sitesample.so"
    fake.write_bytes(b"")
    monkeypatch.setattr(sample_bridge, "_LIBRARY", None)
    monkeypatch.setattr(sample_bridge, "resolve_sitesample_bridge", lambda: fake)
    monkeypatch.setattr(sample_bridge.ctypes, "CDLL", lambda path: Fake())
    with pytest.raises(sample_bridge.SiteSampleBridgeError, match="ABI 99"):
        sample_bridge.load()


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def test_the_bundle_entry_and_handshake_match_the_module():
    artifact, = [a for a in bridge_assets.BUNDLED_ARTIFACTS
                 if a.name == "rw_sitesample"]
    assert artifact.env_var == sample_bridge.SITESAMPLE_BRIDGE_ENV
    assert artifact.kind == "library"
    assert artifact.crate == bridges.RUSTWX_CRATE_RELATIVE
    assert bridge_assets.artifact_filename(artifact, "linux-x86_64") == (
        "librw_sitesample.so")
    assert bridge_assets.library_abi_for("rw_sitesample") == (
        "gpuwm_sitesample_abi_version", sample_bridge.SITESAMPLE_ABI)
    assert bridges.BRIDGE_ABI_MARKERS["rw_sitesample"] == sample_bridge.ABI_MARKER


def test_the_crate_is_a_workspace_member_that_stamps_its_revision():
    rustwx = REPO_ROOT / "tools" / "rustwx"
    assert '"crates/rw-sitesample"' in (rustwx / "Cargo.toml").read_text(
        encoding="utf-8")
    crate = rustwx / "crates" / "rw-sitesample"
    manifest = (crate / "Cargo.toml").read_text(encoding="utf-8")
    assert 'crate-type = ["cdylib", "rlib"]' in manifest
    assert "rustc-env=GPUWM_BRIDGE_SOURCE_REV" in (crate / "build.rs").read_text(
        encoding="utf-8")
    capi = (crate / "src" / "capi.rs").read_text(encoding="utf-8")
    assert "SOURCE_REV_STAMP" in capi
    assert sample_bridge.ABI_MARKER.decode() in capi
    assert f"SITESAMPLE_ABI_VERSION: u32 = {sample_bridge.SITESAMPLE_ABI}" in capi


def test_the_doctor_reports_the_sampler():
    from woof import doctor

    assert "rw_sitesample" in doctor._CHECKED_ARTIFACTS
    check = doctor._sitesample_check()
    assert check.name == "energy site sampler (rw-sitesample)"
    assert check.status == ("verified" if _REASON is None else "missing")
