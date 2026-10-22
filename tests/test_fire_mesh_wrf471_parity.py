"""The fire mesh against WRF v4.7.1 itself, word for word, on CPU and GPU.

The fixture is WRF's own Fortran, byte-unmodified and pinned by sha256,
compiled at -O0 in REAL(4) by ``tools/fire_mesh_wrf471_oracle/build.sh``
(TOOLCHAIN.txt): interpolate_z2fire (module_fr_fire_util.F, whole) refines
the atmospheric terrain onto the fire mesh, and module_fr_fire_model.F's
continuation and centred differences give the slope, at
module_fr_fire_driver.F's ``dxf = dx/ir``.  Seven cases reach every branch:
even ratios, odd ratios (edge nodes written twice, the later write kept),
unit and unequal ratios and spacings, sea inside the domain, a shore on the
domain edge (WRF's continuation goes below zero past it), and terrain below
sea level.

The LANDFIRE pixel each fire cell takes is held to two independent
placements, for every projection woof grids have (Lambert secant and
tangent, Mercator, polar stereographic), in both hemispheres, and for
nests:

* the cell centres through WRF v4.7.1 share/module_llxy.F (byte-identical
  to the v4.6.1 source tools/llxy_wrf461_oracle builds; fixture by
  tools/fire_mesh_wrf471_oracle/build_llxy_fire.py).  The latitude and
  longitude words must be WRF's at 0 ULP, and the pixel must be the one they
  fall in;
* the ``[static.highres]`` overlay's own nearest-neighbour reprojection,
  which must take the same pixel, except for a pinned cell 5 mm from a
  pixel edge (PROVENANCE.md).

Pass is max ULP 0 for every float word and exact equality for every index.
"""
from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import requires_gpu

REPO = Path(__file__).resolve().parents[1]
TOOL = REPO / "tools" / "fire_mesh_wrf471_oracle"
ORACLE = REPO / "woof" / "data" / "fire" / "oracle"
LLXY = ORACLE / "llxy_fire_mesh"
CASES = ("below_sea", "coast", "coast_edge", "fine_ratio", "hills_even",
         "odd_ratio", "unit_ratio")
FIELDS = ("zsf", "dzdxf", "dzdyf")


def _case(name: str) -> dict:
    from woof.verify.urban_oracle import load_case
    raw = load_case(ORACLE / name)
    out = {key: (np.asarray(value).T if np.ndim(value) == 2 else value)
           for key, value in raw.items()}
    for key in ("ir", "jr"):
        out[key] = int(out[key])
    return out


def _fire_mesh(case: dict, xp):
    from woof.static import fuel
    z = fuel.refine_zsf(np.asarray(case["zs"], np.float64), case["ir"],
                        case["jr"], xp=xp)
    dzdx, dzdy = fuel.fire_mesh_slope(
        z, fuel.fire_mesh_spacing(float(case["dx"]), case["ir"]),
        fuel.fire_mesh_spacing(float(case["dy"]), case["jr"]))
    return {"zsf": fuel._host(z), "dzdxf": fuel._host(dzdx),
            "dzdyf": fuel._host(dzdy)}


def _assert_words(got: dict, case: dict, label: str) -> None:
    from woof.core.fp32_ulp import assert_bit_exact, max_ulp
    for name in FIELDS:
        want = np.asarray(case[name], np.float32)
        assert got[name].dtype == np.float32, name
        assert got[name].shape == want.shape, (label, name)
        assert max_ulp(got[name], want) == 0, (label, name)
        assert_bit_exact(got[name], want, f"{label} {name}")


def test_fixture_is_the_pinned_wrf_and_its_own_receipts():
    """Every published word is the one build.sh wrote, from pinned sources."""
    pins = [line.split() for line in
            (TOOL / "SOURCES.sha256").read_text().splitlines()
            if line and not line.startswith("#")]
    recorded = [line.split() for line in
                (ORACLE / "SOURCES.sha256").read_text().splitlines() if line]
    assert recorded == pins
    for line in (ORACLE / "FIXTURES.sha256").read_text().splitlines():
        digest, rel = line.split()
        got = hashlib.sha256((ORACLE / rel).read_bytes()).hexdigest()
        assert got == digest, rel
    harness = dict(reversed(line.split()) for line in
                   (ORACLE / "HARNESS.sha256").read_text().splitlines())
    for rel, digest in harness.items():
        got = hashlib.sha256((TOOL / rel).read_bytes()).hexdigest()
        assert got == digest, rel
    report = (ORACLE / "libmvec-report.txt").read_text()
    head, _, control = report.partition("# positive control")
    assert "_ZGV" not in head and "_ZGV" in control
    assert sorted(path.name for path in ORACLE.iterdir() if path.is_dir()
                  and path.name != LLXY.name) == sorted(CASES)


@pytest.mark.parametrize("name", CASES)
def test_cpu_fire_mesh_is_wrfs_word_for_word(name):
    case = _case(name)
    from woof.static import fuel
    assert fuel.fire_mesh_spacing(float(case["dx"]), case["ir"]) == \
        np.float32(case["dxf"])
    assert fuel.fire_mesh_spacing(float(case["dy"]), case["jr"]) == \
        np.float32(case["dyf"])
    _assert_words(_fire_mesh(case, np), case, f"cpu {name}")


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("name", CASES)
def test_gpu_fire_mesh_is_wrfs_word_for_word(name):
    import cupy as cp
    case = _case(name)
    gpu = _fire_mesh(case, cp)
    _assert_words(gpu, case, f"gpu {name}")
    cpu = _fire_mesh(case, np)
    for field in FIELDS:
        np.testing.assert_array_equal(gpu[field].view(np.uint32),
                                      cpu[field].view(np.uint32))


def test_the_shore_on_the_domain_edge_continues_below_sea_level():
    """WRF's continuation past the domain is linear and unclamped for the
    terrain (interpolate_z2fire's flag_z0 = 0), so land rising from a shore
    on the domain edge continues below zero past it.  woof keeps WRF's."""
    case = _case("coast_edge")
    assert float(np.min(case["zsf"])) < 0.0
    assert float(np.min(case["zs"])) == 0.0
    got = _fire_mesh(case, np)
    assert float(got["zsf"].min()) == float(np.min(case["zsf"]))


# --- the LANDFIRE pixel ------------------------------------------------------

def _llxy_cases() -> dict:
    deck = json.loads((LLXY / "deck.json").read_text())["cases"]
    rows: dict[str, list] = {}

    def word(text: str) -> float:
        return struct.unpack(">d", bytes.fromhex(text))[0]

    for line in (LLXY / "fixture.csv").read_text().splitlines():
        if line.startswith("IJLL"):
            _, cid, i, j, lat, lon = line.split()
            rows.setdefault(cid, []).append(
                (word(i), word(j), word(lat), word(lon)))
    for cid, case in deck.items():
        shape = (case["ny"] * case["sr_y"], case["nx"] * case["sr_x"])
        table = np.array(rows[cid])
        case.update(i=table[:, 0].reshape(shape), j=table[:, 1].reshape(shape),
                    lat=table[:, 2].reshape(shape),
                    lon=table[:, 3].reshape(shape))
    return deck


LLXY_CASES = tuple(sorted(json.loads(
    (LLXY / "deck.json").read_text())["cases"]))


def _grid(case: dict, cases: dict | None = None):
    """The woof grid of one deck case: a root, or a nest of its parent."""
    from woof.static.lambert import LambertGrid
    from woof.static.projection import MercatorGrid, PolarStereoGrid
    if "parent" in case:
        parent = _grid((cases or _llxy_cases())[case["parent"]])
        return parent.nest(case["i_parent_start"], case["j_parent_start"],
                           case["parent_grid_ratio"], case["nx"] + 1,
                           case["ny"] + 1)
    cls = {1: LambertGrid, 2: PolarStereoGrid, 3: MercatorGrid}[case["code"]]
    return cls(case["ref_lat"], case["ref_lon"], case["truelat1"],
               case["truelat2"], case["stand_lon"], case["dx"], case["dx"],
               case["nx"] + 1, case["ny"] + 1)


def _refined(grid, sr: int):
    """A grid whose mass points are ``grid``'s fire cells at ratio ``sr``."""
    return type(grid)(grid.ref_lat, grid.ref_lon, grid.truelat1,
                      grid.truelat2, grid.stand_lon, grid.dx / sr,
                      grid.dy / sr, (grid.e_we - 1) * sr + 1,
                      (grid.e_sn - 1) * sr + 1,
                      known_x=sr * (grid.known_x - 0.5) + 0.5,
                      known_y=sr * (grid.known_y - 0.5) + 0.5)


#: How near (in pixels) to a pixel edge a cell's WRF position must lie for
#: the overlay comparison to hold it out (PROVENANCE.md).
OVERLAY_EDGE_PIXELS = 1.0e-3
#: The cells held out, by case: (row, col) of the fire mesh.  One cell, at
#: 2.5e-4 pixel (5 mm) from a row edge, where the overlay takes the next row.
OVERLAY_HELD_OUT = {"lc_nh_conus": [(10, 10)]}


def _ulp64(got, want) -> int:
    def key(values):
        bits = np.asarray(values, np.float64).view(np.int64)
        return np.where(bits < 0, np.int64(-2 ** 63) - bits, bits)
    return int(np.abs(key(got) - key(want)).max())


def test_the_llxy_fixture_is_the_pinned_source():
    from tools.llxy_wrf461_oracle.build_llxy import SOURCE_SHA256
    pins = dict(reversed(line.split()) for line in
                (TOOL / "SOURCES.sha256").read_text().splitlines()
                if line and not line.startswith("#"))
    assert pins["share/module_llxy.F"] == SOURCE_SHA256
    for name in ("FIXTURES.sha256", "HARNESS.sha256"):
        base = LLXY if name == "FIXTURES.sha256" else TOOL
        for line in (LLXY / name).read_text().splitlines():
            digest, rel = line.split()
            got = hashlib.sha256((base / rel).read_bytes()).hexdigest()
            assert got == digest, rel
    header = [line for line in (LLXY / "fixture.csv").read_text()
              .splitlines() if line.startswith("#")]
    assert f"# source_sha256: {SOURCE_SHA256} (byte-identical in v4.6.1)" \
        in header
    assert ("# wrf_commit: f52c197ed39d12e087d02c50f412d90d418f6186 "
            "(tag v4.7.1)") in header


def test_the_deck_covers_every_woof_projection_and_both_hemispheres():
    from woof.static.projection import implemented_projections
    cases = _llxy_cases()
    codes = {case["code"] for case in cases.values()}
    assert codes == {1, 2, 3}  # lambert, polar, mercator
    assert set(implemented_projections()) == {"lambert", "mercator", "polar"}
    for code in codes:
        lats = [float(case["lat"].mean()) for case in cases.values()
                if case["code"] == code]
        assert min(lats) < 0.0 < max(lats), code
    assert any("parent" in case for case in cases.values())


def test_the_llxy_queries_are_the_fire_cell_centres():
    from woof.static.fuel import fire_mesh_axis
    for case in _llxy_cases().values():
        np.testing.assert_array_equal(
            case["i"][0], fire_mesh_axis(case["nx"], case["sr_x"]))
        np.testing.assert_array_equal(
            case["j"][:, 0], fire_mesh_axis(case["ny"], case["sr_y"]))


@pytest.mark.parametrize("cid", LLXY_CASES)
def test_fire_cell_latlon_is_wrfs_llxy_word_for_word(cid):
    """The fire-cell centres the LANDFIRE reader places are WRF v4.7.1
    module_llxy's latitude and longitude, binary64 word for word."""
    from woof.static.fuel import fire_mesh_latlon
    cases = _llxy_cases()
    case = cases[cid]
    lat, lon = fire_mesh_latlon(_grid(case, cases), case["sr_x"],
                                case["sr_y"])
    assert lat.shape == case["lat"].shape
    assert _ulp64(lat, case["lat"]) == 0, cid
    assert _ulp64(lon, case["lon"]) == 0, cid


@pytest.mark.parametrize("cid", LLXY_CASES)
def test_landfire_pixels_are_the_ones_wrfs_latlon_fall_in(tmp_path, cid):
    """A geographic raster: the pixel WRF's own module_llxy latitude and
    longitude fall in is the pixel the fuel reader takes, for every cell."""
    rasterio = pytest.importorskip("rasterio")
    pytest.importorskip("pyproj")
    from rasterio.transform import from_origin
    from woof.static.fuel import landfire_pixel_indices

    cases = _llxy_cases()
    case = cases[cid]
    lat, lon = case["lat"], case["lon"]
    res = 0.000713  # off any round number, so no centre sits on an edge
    west, north = lon.min() - 0.0123, lat.max() + 0.0117
    width = int((lon.max() + 0.02 - west) / res)
    height = int((north - lat.min() + 0.02) / res)
    path = tmp_path / "geographic.tif"
    with rasterio.open(path, "w", driver="GTiff", width=width, height=height,
                       count=1, dtype="uint8", crs="EPSG:4326",
                       transform=from_origin(west, north, res, res)) as out:
        out.write(np.ones((height, width), np.uint8), 1)
    with rasterio.open(path) as dataset:
        row, col = landfire_pixel_indices(dataset, _grid(case, cases),
                                          case["sr_x"], case["sr_y"])
        inverse = ~dataset.transform
    want_col = np.floor(inverse.a * lon + inverse.b * lat + inverse.c)
    want_row = np.floor(inverse.d * lon + inverse.e * lat + inverse.f)
    assert (row >= 0).all()
    np.testing.assert_array_equal(col, want_col.astype(np.int64))
    np.testing.assert_array_equal(row, want_row.astype(np.int64))


@pytest.mark.parametrize("cid", [cid for cid in LLXY_CASES])
def test_landfire_pixels_are_the_highres_overlays_nearest_words(tmp_path,
                                                                cid):
    """A geographic raster whose words are their pixel's index: the
    ``[static.highres]`` overlay's default reprojection (the native
    static-fields library, its own projection code), nearest neighbour onto
    a grid whose mass points are the fire cells, returns exactly the index
    the fuel reader takes."""
    rasterio = pytest.importorskip("rasterio")
    pytest.importorskip("pyproj")
    from rasterio.transform import from_origin
    from woof.static import rust_bridge
    from woof.static.fuel import landfire_pixel_indices
    from woof.static.highres import (BoundRaster, resample_continuous,
                                     sha256_file)

    if rust_bridge.route("resample_continuous") is None:
        pytest.skip("the overlay's default route is the native "
                    "static-fields bridge, which is not built here")
    cases = _llxy_cases()
    case = cases[cid]
    if case["sr_x"] != case["sr_y"]:
        pytest.skip("woof grids have dx == dy, so an anisotropic fire mesh "
                    "is no grid the overlay can warp onto")
    grid = _grid(case, cases)
    fire_grid = _refined(grid, case["sr_x"])
    lat, lon = case["lat"], case["lon"]
    res = 0.000173
    west, north = lon.min() - 0.0123, lat.max() + 0.0117
    width = int((lon.max() + 0.02 - west) / res)
    height = int((north - lat.min() + 0.02) / res)
    assert width * height < 2 ** 24  # every index is a float32 word
    path = tmp_path / "geographic.tif"
    with rasterio.open(path, "w", driver="GTiff", width=width, height=height,
                       count=1, dtype="float32", crs="EPSG:4326",
                       transform=from_origin(west, north, res, res)) as out:
        out.write(np.arange(width * height, dtype=np.float32)
                  .reshape(height, width), 1)
    bound = BoundRaster(path=path, sha256=sha256_file(path),
                        source_id="synthetic", role="parity",
                        source_url="", license_id="", license_url="",
                        nominal_resolution="0.000173 deg")
    words = resample_continuous(bound, fire_grid, method="nearest")
    with rasterio.open(path) as dataset:
        row, col = landfire_pixel_indices(dataset, grid, case["sr_x"],
                                          case["sr_y"])
    assert (row >= 0).all()
    # The overlay places cells by its own projection code, within about
    # 5 mm of WRF's module_llxy words, so it may take the neighbouring pixel
    # of a cell whose WRF position lies within OVERLAY_EDGE_PIXELS of a pixel
    # edge.  There the pixel is the one WRF's words fall in (the llxy test
    # above).  The cells where the two differ are pinned, and each must be
    # such an edge cell, one pixel off.
    ours = (row * width + col).astype(float)
    held = words != ours
    assert sorted(map(tuple, np.argwhere(held).tolist())) == \
        OVERLAY_HELD_OUT.get(cid, [])
    col_f, row_f = (lon - west) / res, (north - lat) / res
    edge = np.minimum(np.abs(col_f - np.round(col_f)),
                      np.abs(row_f - np.round(row_f)))
    assert (edge[held] < OVERLAY_EDGE_PIXELS).all()
    assert np.isin(np.abs(words[held] - ours[held]),
                   (1.0, float(width))).all()


@pytest.mark.gpu
@requires_gpu
def test_gpu_gather_takes_the_cpu_pixels(tmp_path):
    """The device gather reads the pixels the host indices name."""
    import cupy as cp
    rasterio = pytest.importorskip("rasterio")
    pytest.importorskip("pyproj")
    from rasterio.transform import from_origin
    from woof.static.fuel import sample_landfire

    cases = _llxy_cases()
    case = cases["lc_nh_conus"]
    lat, lon = case["lat"], case["lon"]
    res = 0.0011
    west, north = lon.min() - 0.004, lat.max() + 0.006
    width = int((lon.max() - west) / res)  # the east edge falls short
    height = int((north - lat.min() + 0.01) / res)
    data = (np.arange(width * height) % 211 - 9).astype(np.int16)
    path = tmp_path / "codes.tif"
    with rasterio.open(path, "w", driver="GTiff", width=width, height=height,
                       count=1, dtype="int16", crs="EPSG:4326", nodata=-9,
                       transform=from_origin(west, north, res, res)) as out:
        out.write(data.reshape(height, width), 1)
    grid = _grid(case, cases)
    cpu = sample_landfire(path, grid, case["sr_x"], case["sr_y"])
    gpu = sample_landfire(path, grid, case["sr_x"], case["sr_y"], xp=cp)
    assert (cpu == np.iinfo(np.int32).min).any()  # outside and no data
    np.testing.assert_array_equal(gpu.get(), cpu)
