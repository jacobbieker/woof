"""``woof track splice``: a fine low child and a coarse full-depth parent.

Synthetic wrfout pairs on one Lambert projection, with fields that are
LINEAR in height and in the horizontal, so every interpolation the
splice does is exact and the tests can name the answer.
"""

from __future__ import annotations

from datetime import datetime, timedelta
import json
from pathlib import Path

import numpy as np
import pytest

from woof.io.netcdf_serialization import netCDF4
from woof.track.splice import (
    GRAVITY, SPLICE_SCHEMA, WEIGHT_VARIABLE, DomainHistory, SpliceRefused,
    Tier, child_weight, discover_frames, interp_columns, merge_fields,
    parse_tier, path_document, sample_path, write_path_csv)

START = datetime(1974, 4, 3, 12)
PARENT_DX = 3000.0
CHILD_DX = 1000.0
#: The parent's and child's column tops (m): a deeper parent.
PARENT_TOP = 20000.0
CHILD_TOP = 20000.0


def _parent_grid():
    from woof.static.lambert import LambertGrid

    return LambertGrid(40.0, -100.0, 30.0, 60.0, -100.0, PARENT_DX, PARENT_DX,
                       21, 21, known_x=1.0, known_y=1.0)


def _profile(z, x, y, *, offset):
    """theta-like field linear in height and position."""

    return 290.0 + 0.004 * z + 0.01 * x + 0.02 * y + offset


def _write_frame(path, when, *, nx, ny, nz, top, offset, positions,
                 dx):
    """``positions(i, j) -> (lat, lon)`` for 0-based (possibly half) mass
    indices of THIS grid."""

    jj, ii = np.mgrid[0:ny, 0:nx].astype(float)
    lat, lon = positions(ii, jj)
    ju, iu = np.mgrid[0:ny, 0:nx + 1].astype(float)
    lat_u, lon_u = positions(iu - 0.5, ju)
    jv, iv = np.mgrid[0:ny + 1, 0:nx].astype(float)
    lat_v, lon_v = positions(iv, jv - 0.5)
    z_full = np.linspace(0.0, top, nz + 1)
    z_mass = 0.5 * (z_full[:-1] + z_full[1:])
    with netCDF4.Dataset(path, "w") as nc:
        for name, size in (("Time", 1), ("DateStrLen", 19),
                           ("west_east", nx), ("south_north", ny),
                           ("bottom_top", nz), ("west_east_stag", nx + 1),
                           ("south_north_stag", ny + 1),
                           ("bottom_top_stag", nz + 1)):
            nc.createDimension(name, size)
        nc.MAP_PROJ = 1
        nc.TRUELAT1 = 30.0
        nc.TRUELAT2 = 60.0
        nc.STAND_LON = -100.0
        nc.DX = dx
        nc.DY = dx
        times = nc.createVariable("Times", "S1", ("Time", "DateStrLen"))
        times[0] = np.frombuffer(
            when.strftime("%Y-%m-%d_%H:%M:%S").encode(), dtype="S1")
        mass3 = ("Time", "bottom_top", "south_north", "west_east")
        w3 = ("Time", "bottom_top_stag", "south_north", "west_east")
        u3 = ("Time", "bottom_top", "south_north", "west_east_stag")
        v3 = ("Time", "bottom_top", "south_north_stag", "west_east")
        mass2 = ("Time", "south_north", "west_east")

        def put(name, dims, values, dtype="f8"):
            var = nc.createVariable(name, dtype, dims)
            var[...] = np.asarray(values)[None]

        put("XLAT", mass2, lat)
        put("XLONG", mass2, lon)
        put("XLAT_U", ("Time", "south_north", "west_east_stag"), lat_u)
        put("XLONG_U", ("Time", "south_north", "west_east_stag"), lon_u)
        put("XLAT_V", ("Time", "south_north_stag", "west_east"), lat_v)
        put("XLONG_V", ("Time", "south_north_stag", "west_east"), lon_v)
        put("HGT", mass2, np.zeros((ny, nx)))
        put("SINALPHA", mass2, np.zeros((ny, nx)))
        put("COSALPHA", mass2, np.ones((ny, nx)))
        phb = np.broadcast_to((z_full * GRAVITY)[:, None, None],
                              (nz + 1, ny, nx))
        put("PHB", w3, phb)
        put("PH", w3, np.zeros_like(phb))
        zm = np.broadcast_to(z_mass[:, None, None], (nz, ny, nx))
        put("T", mass3, _profile(zm, lat, lon, offset=offset) - 300.0)
        put("PB", mass3, 1.0e5 - 4.0 * zm)
        put("P", mass3, np.full((nz, ny, nx), offset))
        put("QVAPOR", mass3, np.full((nz, ny, nx), 0.01 + 1e-3 * offset))
        zu = np.broadcast_to(z_mass[:, None, None], (nz, ny, nx + 1))
        put("U", u3, 5.0 + 0.001 * zu + offset)
        zv = np.broadcast_to(z_mass[:, None, None], (nz, ny + 1, nx))
        put("V", v3, -2.0 + 0.0005 * zv + offset)
        zw = np.broadcast_to(z_full[:, None, None], (nz + 1, ny, nx))
        put("W", w3, 0.0001 * zw + offset)


def _pair(tmp_path, *, frames=3, child_offset=1.0):
    """A 3 km parent (20 x 20, 40 levels) and a 1 km child (30 x 30,
    60 levels) inside it, ``frames`` frames 15 min apart."""

    grid = _parent_grid()

    def parent_pos(i, j):
        return grid.ij_to_latlon(i + 1.0, j + 1.0)

    def child_pos(i, j):
        # child mass (0, 0) sits at parent mass index (4.0, 5.0)
        return grid.ij_to_latlon(4.0 + i / 3.0 + 1.0, 5.0 + j / 3.0 + 1.0)

    pdir, cdir = tmp_path / "parent", tmp_path / "child"
    pdir.mkdir()
    cdir.mkdir()
    for k in range(frames):
        when = START + timedelta(minutes=15 * k)
        stamp = when.strftime("%Y-%m-%d_%H:%M:%S")
        _write_frame(pdir / f"wrfout_d01_{stamp}", when, nx=20, ny=20, nz=40,
                     top=PARENT_TOP, offset=0.0, positions=parent_pos,
                     dx=PARENT_DX)
        _write_frame(cdir / f"wrfout_d02_{stamp}", when, nx=30, ny=30, nz=60,
                     top=CHILD_TOP, offset=child_offset, positions=child_pos,
                     dx=CHILD_DX)
    return pdir, cdir, grid, child_pos


def _tiers(pdir, cdir, *, cutoff=5000.0, edge=0, blend=0):
    parent = DomainHistory(discover_frames([pdir], None, what="parent"),
                           label="parent")
    child = DomainHistory(discover_frames([cdir], None, what="child"),
                          label="child")
    return [Tier(parent), Tier(child, cutoff_m=cutoff, edge_cells=edge,
                               lateral_blend_cells=blend)]


def test_the_child_weight_is_a_hard_switch_or_a_centred_ramp():
    z = np.array([0.0, 4999.0, 5000.0, 5001.0, 9000.0])
    assert child_weight(z, 5000.0).tolist() == [1.0, 1.0, 0.0, 0.0, 0.0]
    ramp = child_weight(np.array([3999.0, 4000.0, 5000.0, 6000.0, 6001.0]),
                        5000.0, 2000.0)
    assert ramp.tolist() == [1.0, 1.0, 0.5, 0.0, 0.0]
    with pytest.raises(SpliceRefused):
        child_weight(z, 5000.0, -1.0)


def test_column_interpolation_is_exact_on_linear_profiles():
    z = np.tile(np.linspace(0.0, 1000.0, 11)[:, None], (1, 3))
    v = 2.0 * z + 7.0
    target = np.array([[-5.0, 15.0, 2000.0], [995.0, 500.0, 1000.0]])
    values, below = interp_columns(z, v, target)
    assert values[0, 0] == 7.0 and below[0, 0]          # held below
    assert values[0, 1] == pytest.approx(37.0)
    assert np.isnan(values[0, 2])                         # above the top
    assert values[1].tolist() == pytest.approx([1997.0, 1007.0, 2007.0])


def test_the_domain_locates_its_own_points(tmp_path):
    pdir, cdir, grid, child_pos = _pair(tmp_path, frames=1)
    child = DomainHistory(discover_frames([cdir], None, what="child"),
                          label="child")
    lat, lon = child_pos(np.array([0.0, 12.5, 29.0]),
                         np.array([0.0, 7.25, 29.0]))
    x, y = child.mass_index(lat, lon)
    assert x == pytest.approx([0.0, 12.5, 29.0], abs=1e-6)
    assert y == pytest.approx([0.0, 7.25, 29.0], abs=1e-6)


def test_a_path_takes_the_child_below_and_the_parent_above(tmp_path):
    pdir, cdir, grid, child_pos = _pair(tmp_path)
    tiers = _tiers(pdir, cdir)
    lat, lon = child_pos(np.array([10.0, 10.0, 10.0, 40.0]),
                         np.array([12.0, 12.0, 12.0, 12.0]))
    when = np.datetime64(START + timedelta(minutes=7, seconds=30), "s")
    sample = sample_path(tiers, times=[when] * 4, lat=lat, lon=lon,
                         height=[1000.0, 4999.0, 7000.0, 1000.0])
    # child offset is +1 on U (5 + 0.001 z + offset), earth == grid here
    u = sample.spliced["u"]
    assert u[0] == pytest.approx(5.0 + 1.0 + 1.0)
    assert u[1] == pytest.approx(5.0 + 4.999 + 1.0)
    assert u[2] == pytest.approx(5.0 + 7.0)              # parent above
    # The fourth fix lies outside the child: the parent alone answers.
    assert not sample.child_inside[3]
    assert u[3] == pytest.approx(5.0 + 1.0)
    # Continuity at the splice: child minus parent is the offset, exactly.
    vertical = sample.continuity["joins"][0]["vertical"]
    assert vertical["u"]["mean"] == pytest.approx(1.0)
    assert vertical["theta"]["mean"] == pytest.approx(1.0)
    assert vertical["u"]["n"] == 3
    assert sample.source[0] == "child" and sample.source[2] == "parent"


def test_a_blend_layer_mixes_linearly(tmp_path):
    pdir, cdir, grid, child_pos = _pair(tmp_path)
    tiers = _tiers(pdir, cdir)
    lat, lon = child_pos(np.array([15.0]), np.array([15.0]))
    when = np.datetime64(START, "s")
    sample = sample_path(tiers, times=[when], lat=lat, lon=lon,
                         height=[5250.0], blend_m=1000.0)
    assert sample.child_weight[0] == pytest.approx(0.25)
    assert sample.spliced["u"][0] == pytest.approx(5.0 + 5.25 + 0.25)


def test_a_path_outside_the_history_is_missing_not_invented(tmp_path):
    pdir, cdir, grid, child_pos = _pair(tmp_path, frames=2)
    tiers = _tiers(pdir, cdir)
    lat, lon = child_pos(np.array([15.0]), np.array([15.0]))
    late = np.datetime64(START + timedelta(hours=2), "s")
    sample = sample_path(tiers, times=[late], lat=lat, lon=lon,
                         height=[1000.0])
    assert np.isnan(sample.spliced["u"][0])
    document = path_document(sample, name="synthetic", sources={})
    json.dumps(document, allow_nan=False)
    assert document["samples"][0]["u"] is None


def test_the_csv_and_json_outputs_carry_every_column(tmp_path):
    pdir, cdir, grid, child_pos = _pair(tmp_path, frames=2)
    tiers = _tiers(pdir, cdir)
    lat, lon = child_pos(np.array([5.0, 6.0]), np.array([5.0, 6.0]))
    when = np.datetime64(START, "s")
    sample = sample_path(tiers, times=[when, when], lat=lat,
                         lon=lon, height=[500.0, 9000.0])
    path = write_path_csv(sample, tmp_path / "out.csv")
    header = path.read_text().splitlines()[0].split(",")
    for column in ("u", "child_u", "parent_u", "child_weight", "theta",
                   "source"):
        assert column in header
    document = path_document(sample, name="synthetic", sources={"a": 1})
    assert document["schema"] == SPLICE_SCHEMA
    assert document["fixes"] == 2


def test_the_merged_field_set_keeps_the_child_below_and_the_parent_above(
        tmp_path):
    pdir, cdir, grid, child_pos = _pair(tmp_path, frames=2)
    tiers = _tiers(pdir, cdir)
    child = tiers[1].history
    written = merge_fields(tiers, tmp_path / "merged")["child"]
    assert [p.name for p in written] == [p.name for p in child.paths]
    with netCDF4.Dataset(written[0]) as nc, \
            netCDF4.Dataset(child.paths[0]) as src:
        weight = np.asarray(nc.variables[WEIGHT_VARIABLE][0])
        z_full = (np.asarray(src.variables["PHB"][0])) / GRAVITY
        z_mass = 0.5 * (z_full[:-1] + z_full[1:])
        below = z_mass[:, 0, 0] < 5000.0
        assert np.all(weight[below] == 1.0)
        assert np.all(weight[~below] == 0.0)
        u = np.asarray(nc.variables["U"][0])
        u_src = np.asarray(src.variables["U"][0])
        np.testing.assert_array_equal(u[below], u_src[below])
        # above: the parent's profile (offset 0), exact for a linear field
        k = int(np.flatnonzero(~below)[0])
        assert u[k, 10, 10] == pytest.approx(5.0 + 0.001 * z_mass[k, 0, 0])
        p = np.asarray(nc.variables["P"][0])
        assert p[k, 10, 10] == pytest.approx(0.0, abs=1e-6)
        assert p[0, 10, 10] == pytest.approx(1.0)
        assert nc.getncattr("SPLICE_SCHEMA") == SPLICE_SCHEMA
    # The child's own history is untouched.
    with netCDF4.Dataset(child.paths[0]) as src:
        assert WEIGHT_VARIABLE not in src.variables


def test_the_merged_field_set_refuses_two_projections(tmp_path):
    pdir, cdir, grid, child_pos = _pair(tmp_path, frames=1)
    for path in cdir.iterdir():
        with netCDF4.Dataset(path, "r+") as nc:
            nc.STAND_LON = -90.0
    with pytest.raises(SpliceRefused):
        merge_fields(_tiers(pdir, cdir), tmp_path / "merged")


def test_a_folder_of_several_domains_needs_a_domain(tmp_path):
    pdir, cdir, grid, child_pos = _pair(tmp_path, frames=1)
    for path in cdir.iterdir():
        (pdir / path.name).write_bytes(path.read_bytes())
    with pytest.raises(SpliceRefused, match="--parent-domain"):
        discover_frames([pdir], None, what="parent")
    assert len(discover_frames([pdir], 2, what="parent")) == 1


def test_the_cli_samples_a_path_file(tmp_path, capsys):
    from woof.cli import main as cli_main

    pdir, cdir, grid, child_pos = _pair(tmp_path, frames=2)
    lat, lon = child_pos(np.array([10.0, 11.0]), np.array([10.0, 11.0]))
    track = tmp_path / "path.csv"
    rows = ["time,lat,lon,height"]
    for k in range(2):
        when = START + timedelta(minutes=5 * k)
        rows.append(f"{when.isoformat()}Z,{lat[k]},{lon[k]},"
                    f"{1000.0 + 6000.0 * k}")
    track.write_text("\n".join(rows) + "\n")
    out = tmp_path / "sample.json"
    assert cli_main(["track", "splice", "--child", str(cdir), "--parent",
                     str(pdir), "--splice-height-m", "5000", "--path",
                     str(track), "--out", str(out), "--edge-cells", "0",
                     "--lateral-blend-cells", "0"]) == 0
    document = json.loads(out.read_text())
    assert document["fixes"] == 2
    join = document["continuity"]["joins"][0]
    assert join["vertical"]["u"]["mean"] == pytest.approx(1.0)
    assert "answered by" in capsys.readouterr().out


def test_the_path_and_field_set_reach_icechunk(tmp_path):
    pytest.importorskip("icechunk")
    from woof.track.splice import write_path_icechunk
    from woof.energy import icechunk_store as ics

    pdir, cdir, grid, child_pos = _pair(tmp_path, frames=2)
    tiers = _tiers(pdir, cdir)
    child = tiers[1].history
    lat, lon = child_pos(np.array([10.0]), np.array([10.0]))
    sample = sample_path(tiers, times=[np.datetime64(START, "s")],
                         lat=lat, lon=lon, height=[1000.0])
    store = tmp_path / "repo"
    info = write_path_icechunk(sample, str(store), group="splice/synthetic",
                               name="synthetic", sources={})
    assert info["committed"]
    repo = ics.open_repository(str(store), create=False)
    session = repo.readonly_session(branch="main")
    zarr = ics._zarr()
    group = zarr.open_group(session.store, path="splice/synthetic", mode="r")
    assert group.attrs["schema"] == SPLICE_SCHEMA
    assert float(group["u"][0]) == pytest.approx(5.0 + 1.0 + 1.0)
    written = merge_fields(tiers, tmp_path / "merged")["child"]
    axis = ics.TimeAxis(str(child.times[0]), str(child.times[-1]), 900)
    ics.write_gridded_segment(str(store), written, group="spliced",
                              axis=axis, on_existing="overwrite")
    session = repo.readonly_session(branch="main")
    spliced = zarr.open_group(session.store, path="spliced", mode="r")
    assert WEIGHT_VARIABLE in {name for name, _ in spliced.members()}


def _fine(tmp_path, grid, *, frames=2, offset=2.0):
    """A 333 m third tier (24 x 24, 50 levels) inside the 1 km child."""

    def fine_pos(i, j):
        return grid.ij_to_latlon(8.0 + i / 9.0 + 1.0, 9.0 + j / 9.0 + 1.0)

    fdir = tmp_path / "fine"
    fdir.mkdir()
    for k in range(frames):
        when = START + timedelta(minutes=15 * k)
        stamp = when.strftime("%Y-%m-%d_%H:%M:%S")
        _write_frame(fdir / f"wrfout_d03_{stamp}", when, nx=24, ny=24, nz=50,
                     top=20000.0, offset=offset, positions=fine_pos,
                     dx=1000.0 / 3.0)
    return fdir, fine_pos


def test_a_tier_fades_in_across_its_edge_and_never_uses_its_edge_zone(
        tmp_path):
    pdir, cdir, grid, child_pos = _pair(tmp_path)
    tiers = _tiers(pdir, cdir, edge=5, blend=4)
    x = np.array([2.0, 5.0, 7.0, 9.0, 15.0])
    lat, lon = child_pos(x, np.full(x.size, 15.0))
    when = np.datetime64(START, "s")
    sample = sample_path(tiers, times=[when] * x.size, lat=lat, lon=lon,
                         height=[1000.0] * x.size)
    assert sample.child_weight.tolist() == pytest.approx(
        [0.0, 0.0, 0.5, 1.0, 1.0])
    u_parent = 5.0 + 1.0
    assert sample.spliced["u"][0] == pytest.approx(u_parent)   # relax zone
    assert sample.spliced["u"][2] == pytest.approx(u_parent + 0.5)
    assert sample.spliced["u"][4] == pytest.approx(u_parent + 1.0)
    lateral = sample.continuity["joins"][0]["lateral"]["u"]
    assert lateral["n"] >= 2 and lateral["mean"] == pytest.approx(1.0)


def test_three_tiers_answer_by_precedence_and_fall_back_in_order(tmp_path):
    pdir, cdir, grid, child_pos = _pair(tmp_path, frames=2)
    fdir, fine_pos = _fine(tmp_path, grid)
    parent = DomainHistory(discover_frames([pdir], None, what="p"), label="p")
    child = DomainHistory(discover_frames([cdir], None, what="c"), label="c")
    fine = DomainHistory(discover_frames([fdir], None, what="f"), label="f")
    tiers = [Tier(parent), Tier(child, cutoff_m=8000.0, edge_cells=0,
                                lateral_blend_cells=0),
             Tier(fine, cutoff_m=3000.0, edge_cells=0,
                  lateral_blend_cells=0)]
    # inside fine, low / inside fine above 3 km / inside child only, low /
    # above 8 km / outside the child
    la, lo = fine_pos(np.array([12.0, 12.0]), np.array([12.0, 12.0]))
    lb, lob = child_pos(np.array([25.0, 25.0]), np.array([25.0, 25.0]))
    lc, loc = child_pos(np.array([40.0]), np.array([10.0]))
    lat = np.concatenate([la, lb, lc])
    lon = np.concatenate([lo, lob, loc])
    height = [1000.0, 5000.0, 1000.0, 9000.0, 1000.0]
    when = np.datetime64(START, "s")
    sample = sample_path(tiers, times=[when] * 5, lat=lat, lon=lon,
                         height=height)
    assert list(sample.source) == ["f", "c", "c", "p", "p"]
    u = sample.spliced["u"]
    assert u[0] == pytest.approx(5.0 + 1.0 + 2.0)
    assert u[1] == pytest.approx(5.0 + 5.0 + 1.0)
    assert u[3] == pytest.approx(5.0 + 9.0)
    joins = sample.continuity["joins"]
    assert [j["fine"] for j in joins] == ["c", "f"]
    assert joins[1]["vertical"]["u"]["mean"] == pytest.approx(1.0)


def test_the_stack_must_run_coarse_to_fine_with_one_full_depth_grid(
        tmp_path):
    pdir, cdir, grid, child_pos = _pair(tmp_path, frames=1)
    parent, child = (t.history for t in _tiers(pdir, cdir))
    with pytest.raises(SpliceRefused, match="coarse to fine"):
        sample_path([Tier(child), Tier(parent, cutoff_m=1.0)], times=[],
                    lat=[], lon=[], height=[])
    with pytest.raises(SpliceRefused, match="full depth"):
        sample_path([Tier(parent, cutoff_m=1.0)], times=[], lat=[], lon=[],
                    height=[])
    with pytest.raises(SpliceRefused, match="cutoff"):
        sample_path([Tier(parent), Tier(child)], times=[], lat=[], lon=[],
                    height=[])


def test_a_path_leaving_the_footprint_passes_smoothly_to_the_parent(
        tmp_path):
    """A straight path out of the child: the spliced value ramps across
    the blend band, with no step larger than the band's slope."""

    pdir, cdir, grid, child_pos = _pair(tmp_path)
    tiers = _tiers(pdir, cdir, edge=5, blend=6)
    x = np.linspace(15.0, 35.0, 81)
    lat, lon = child_pos(x, np.full(x.size, 15.0))
    when = np.datetime64(START, "s")
    sample = sample_path(tiers, times=[when] * x.size, lat=lat, lon=lon,
                         height=[1000.0] * x.size)
    steps = np.abs(np.diff(sample.spliced["u"]))
    # the offset (1 m/s) spread over 6 cells at 4 samples per cell, plus
    # the parent's own gradient: never a jump of the whole offset
    assert steps.max() < 0.1
    assert sample.source[0] == "child" and sample.source[-1] == "parent"


def test_three_tiers_merge_against_the_merged_tier_below(tmp_path):
    pdir, cdir, grid, child_pos = _pair(tmp_path, frames=2)
    fdir, fine_pos = _fine(tmp_path, grid)
    tiers = [Tier(DomainHistory(discover_frames([pdir], None, what="p"),
                                label="p")),
             Tier(DomainHistory(discover_frames([cdir], None, what="c"),
                                label="c"), cutoff_m=8000.0, edge_cells=0,
                  lateral_blend_cells=0),
             Tier(DomainHistory(discover_frames([fdir], None, what="f"),
                                label="f"), cutoff_m=3000.0, edge_cells=0,
                  lateral_blend_cells=0)]
    written = merge_fields(tiers, tmp_path / "merged")
    assert set(written) == {"c", "f"}
    with netCDF4.Dataset(written["f"][0]) as nc:
        u = np.asarray(nc.variables["U"][0])
        phb = np.asarray(nc.variables["PHB"][0]) / GRAVITY
        z_mass = 0.5 * (phb[:-1] + phb[1:])[:, 0, 0]
        k_mid = int(np.flatnonzero((z_mass > 3000.0) & (z_mass < 8000.0))[0])
        k_top = int(np.flatnonzero(z_mass > 8000.0)[0])
        # between the cutoffs the fine grid takes the middle tier
        assert u[k_mid, 12, 12] == pytest.approx(
            5.0 + 0.001 * z_mass[k_mid] + 1.0)
        # above both, the coarsest
        assert u[k_top, 12, 12] == pytest.approx(5.0 + 0.001 * z_mass[k_top])


def test_a_tier_spec_reads_path_cutoff_and_domain():
    assert parse_tier("run/") == (Path("run/"), None, None)
    assert parse_tier("run/,5000,d02") == (Path("run/"), 5000.0, 2)
    # read from the right: a path that holds a comma survives
    assert parse_tier("runs/a,b/out,3000") == (Path("runs/a,b/out"), 3000.0,
                                               None)


def test_a_file_of_another_domain_is_refused(tmp_path):
    pdir, cdir, grid, child_pos = _pair(tmp_path, frames=1)
    path = next(cdir.iterdir())
    with pytest.raises(SpliceRefused, match="not the d03"):
        discover_frames([path], 3, what="tier")


def test_a_field_a_tier_lacks_is_handed_down_not_blended_as_zero(tmp_path):
    pdir, cdir, grid, child_pos = _pair(tmp_path, frames=1)
    for path in cdir.iterdir():
        with netCDF4.Dataset(path, "r+") as nc:
            nc.renameVariable("QVAPOR", "QVAPOR_DROPPED")
    tiers = _tiers(pdir, cdir, edge=5, blend=4)
    lat, lon = child_pos(np.array([7.0, 15.0]), np.array([15.0, 15.0]))
    when = np.datetime64(START, "s")
    sample = sample_path(tiers, times=[when] * 2, lat=lat, lon=lon,
                         height=[1000.0, 1000.0])
    assert sample.child_weight.tolist() == pytest.approx([0.5, 1.0])
    assert sample.spliced["qvapor"].tolist() == pytest.approx([0.01, 0.01])


def test_an_agl_path_asks_every_tier_for_one_height(tmp_path):
    pdir, cdir, grid, child_pos = _pair(tmp_path, frames=1)
    for path in cdir.iterdir():
        with netCDF4.Dataset(path, "r+") as nc:
            nc.variables["HGT"][...] = 300.0
    tiers = _tiers(pdir, cdir)
    lat, lon = child_pos(np.array([15.0]), np.array([15.0]))
    when = np.datetime64(START, "s")
    sample = sample_path(tiers, times=[when], lat=lat, lon=lon,
                         height=[1000.0], height_datum="agl")
    # the coarsest tier's ground is 0 m, so both are read at 1000 m MSL
    assert sample.child["u"][0] == pytest.approx(5.0 + 1.0 + 1.0)
    assert sample.parent["u"][0] == pytest.approx(5.0 + 1.0)


def test_a_fine_point_outside_the_middle_tier_falls_back_to_the_coarsest(
        tmp_path):
    """The finest grid sticks out of the middle one: above its own cutoff
    those points take the coarsest grid, never their own value."""

    pdir, cdir, grid, child_pos = _pair(tmp_path, frames=2)

    def fine_pos(i, j):  # straddles the child's east edge (child x 26..)
        return grid.ij_to_latlon(12.0 + i / 9.0 + 1.0, 9.0 + j / 9.0 + 1.0)

    fdir = tmp_path / "fine"
    fdir.mkdir()
    for k in range(2):
        when = START + timedelta(minutes=15 * k)
        stamp = when.strftime("%Y-%m-%d_%H:%M:%S")
        _write_frame(fdir / f"wrfout_d03_{stamp}", when, nx=24, ny=24, nz=50,
                     top=20000.0, offset=2.0, positions=fine_pos,
                     dx=1000.0 / 3.0)
    tiers = [Tier(DomainHistory(discover_frames([pdir], None, what="p"),
                                label="p")),
             Tier(DomainHistory(discover_frames([cdir], None, what="c"),
                                label="c"), cutoff_m=8000.0, edge_cells=0,
                  lateral_blend_cells=0),
             Tier(DomainHistory(discover_frames([fdir], None, what="f"),
                                label="f"), cutoff_m=3000.0, edge_cells=0,
                  lateral_blend_cells=0)]
    written = merge_fields(tiers, tmp_path / "merged")
    with netCDF4.Dataset(written["f"][0]) as nc:
        u = np.asarray(nc.variables["U"][0])
        phb = np.asarray(nc.variables["PHB"][0]) / GRAVITY
        z_mass = 0.5 * (phb[:-1] + phb[1:])[:, 0, 0]
        k = int(np.flatnonzero((z_mass > 3000.0) & (z_mass < 8000.0))[0])
        # column i = 23 of the fine grid lies at child x = 12*3+23/3-4*3 > 29
        assert u[k, 12, 23] == pytest.approx(5.0 + 0.001 * z_mass[k])
        assert u[k, 12, 2] == pytest.approx(5.0 + 0.001 * z_mass[k] + 1.0)
