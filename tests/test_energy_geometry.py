"""Property tests for :mod:`woof.energy.geometry` (fixed seeds, CPU only)."""

from __future__ import annotations

import math
import time

import numpy as np
import pytest
from scipy.spatial import cKDTree

from woof.energy import geometry as geo
from woof.energy.geometry import (
    GeometryError,
    LocalProjection,
    Rect,
    buffer_polyline,
    cover_with_rectangles,
    densify_polyline,
    local_projection,
    point_in_polygon,
    segment_bearings,
)
from woof.static.projection import EARTH_RADIUS_M


def _great_circle_m(lon1, lat1, lon2, lat2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(np.asarray(lon2) - np.asarray(lon1))
    h = (np.sin(0.5 * (p2 - p1)) ** 2
         + np.cos(p1) * np.cos(p2) * np.sin(0.5 * dl) ** 2)
    return 2.0 * EARTH_RADIUS_M * np.arcsin(np.minimum(1.0, np.sqrt(h)))


def _distance_to_line_m(proj, coords, qx, qy):
    """Distance in ``proj`` metres from query points to the polyline,
    through a k-d tree on a 2 m densification (accurate to millimetres at
    the test half-widths)."""

    lon, lat, _ = densify_polyline(coords, 2.0)
    lx, ly = proj.forward(lon, lat)
    tree = cKDTree(np.stack((lx, ly), axis=1))
    dist, _ = tree.query(np.stack((qx, qy), axis=1))
    return dist


# ---------------------------------------------------------------------------
# projection


@pytest.mark.parametrize("lat0, lon0", [(52.0, -3.5), (0.0, 0.0), (72.5, -40.0),
                                        (-33.9, 151.2), (10.0, 179.9)])
def test_projection_round_trip_within_a_millimetre_to_500_km(lat0, lon0):
    proj = local_projection(lat0, lon0)
    rng = np.random.default_rng(1)
    azimuth = rng.uniform(0.0, 2.0 * np.pi, 20000)
    radius = rng.uniform(0.0, 500e3, 20000)
    x, y = radius * np.sin(azimuth), radius * np.cos(azimuth)
    lon, lat = proj.inverse(x, y)
    x2, y2 = proj.forward(lon, lat)
    assert np.max(np.hypot(x2 - x, y2 - y)) < 1e-3
    # Azimuthal equidistant: radius is the great-circle distance to the centre.
    d = _great_circle_m(lon0, lat0, lon, lat)
    assert np.max(np.abs(d - radius)) < 1e-3


def test_projection_centre_axes_and_shapes():
    proj = LocalProjection(51.6, -3.9)
    x, y = proj.forward(-3.9, 51.6)
    assert abs(float(x)) < 1e-6 and abs(float(y)) < 1e-6
    x, y = proj.forward(-3.9, 51.7)          # due north
    assert abs(float(x)) < 1e-6 and float(y) > 11000
    x, y = proj.forward(-3.8, 51.6)          # east (slightly south of +x)
    assert float(x) > 6000
    grid_lon, grid_lat = np.meshgrid([-4.0, -3.9, -3.8], [51.5, 51.6])
    gx, gy = proj.forward(grid_lon, grid_lat)
    assert gx.shape == (2, 3) and gy.shape == (2, 3)
    lon, lat = proj.inverse(gx, gy)
    assert np.allclose(lon, grid_lon, atol=1e-10)
    assert np.allclose(lat, grid_lat, atol=1e-10)


def test_projection_refuses_bad_centre():
    with pytest.raises(GeometryError):
        local_projection(91.0, 0.0)
    with pytest.raises(GeometryError):
        local_projection(float("nan"), 0.0)


# ---------------------------------------------------------------------------
# densify and bearings


def test_densify_uniform_spacing_and_exact_ends():
    coords = [(-4.0, 51.6), (-3.7, 51.65), (-3.7, 51.65), (-3.5, 51.5),
              (-3.2, 51.7)]
    lon, lat, ch = densify_polyline(coords, 100.0)
    assert (lon[0], lat[0]) == coords[0]
    assert (lon[-1], lat[-1]) == coords[-1]
    steps = np.diff(ch)
    assert np.allclose(steps[:-1], 100.0)
    assert 0.0 < steps[-1] <= 100.0
    total = sum(_great_circle_m(a[0], a[1], b[0], b[1])
                for a, b in zip(coords, coords[1:]))
    assert ch[-1] == pytest.approx(total, abs=1e-6)
    # Consecutive samples are spacing apart along the great circle, except
    # across a vertex where the chord is shorter.
    chord = _great_circle_m(lon[:-1], lat[:-1], lon[1:], lat[1:])
    assert np.all(chord <= steps + 1e-6)
    assert np.median(chord[:-1]) == pytest.approx(100.0, abs=1e-6)


def test_densify_single_segment_and_degenerate():
    lon, lat, ch = densify_polyline([(0.0, 0.0), (0.0, 0.01)], 250.0)
    length = 0.01 * math.pi / 180.0 * EARTH_RADIUS_M
    assert ch[-1] == pytest.approx(length)
    assert len(ch) == math.ceil(length / 250.0) + 1
    assert np.allclose(lon, 0.0)
    assert np.all(np.diff(lat) > 0)
    # Exact multiple: no zero-length tail step.
    lon, lat, ch = densify_polyline([(0.0, 0.0), (0.0, 0.01)], length / 4)
    assert len(ch) == 5
    # Zero-length line: the single point back.
    lon, lat, ch = densify_polyline([(1.0, 2.0), (1.0, 2.0)], 10.0)
    assert lon.tolist() == [1.0] and lat.tolist() == [2.0] and ch.tolist() == [0.0]


@pytest.mark.parametrize("spacing", [0.0, -5.0, float("nan"), float("inf")])
def test_densify_refuses_bad_spacing(spacing):
    with pytest.raises(GeometryError):
        densify_polyline([(0.0, 0.0), (0.1, 0.0)], spacing)


def test_densify_keeps_a_chain_of_submillimetre_steps():
    # 200 000 steps of 0.56 mm: each is below the duplicate tolerance but
    # together they are 111 m of line.
    lat = np.arange(200_000) * (0.00056 / EARTH_RADIUS_M) * 180.0 / math.pi
    coords = np.stack((np.zeros_like(lat), lat), axis=1)
    lon, lat_out, ch = densify_polyline(coords, 10.0)
    assert ch[-1] == pytest.approx(111.9, abs=0.2)
    assert len(ch) == 13
    bearings = segment_bearings(coords[::1000, 0], coords[::1000, 1])
    assert np.allclose(bearings, 0.0)


def test_densify_folds_longitudes_consistently_across_antimeridian():
    lon, lat, ch = densify_polyline([(179.9, 0.0), (180.1, 0.0)], 5000.0)
    assert np.all((lon >= -180.0) & (lon < 180.0))
    assert lon[0] == pytest.approx(179.9) and lon[-1] == pytest.approx(-179.9)
    assert ch[-1] == pytest.approx(0.2 * math.pi / 180.0 * EARTH_RADIUS_M)


def test_densify_refuses_antipodal_leg():
    with pytest.raises(GeometryError, match="antipodal"):
        densify_polyline([(0.0, 0.0), (180.0, 0.0)], 1e6)
    with pytest.raises(GeometryError, match="antipodal"):
        buffer_polyline([(0.0, 0.0), (180.0, 0.0)], 1000.0)


def test_densify_refuses_bad_coordinates():
    with pytest.raises(GeometryError):
        densify_polyline([], 10.0)
    with pytest.raises(GeometryError):
        densify_polyline([(0.0, 95.0), (0.0, 0.0)], 10.0)
    with pytest.raises(GeometryError):
        densify_polyline([(0.0, float("nan")), (0.0, 0.0)], 10.0)


def test_known_bearings():
    east = segment_bearings(np.array([0.0, 0.5, 1.0]), np.zeros(3))
    assert np.allclose(east, 90.0)
    north = segment_bearings(np.zeros(3), np.array([10.0, 10.5, 11.0]))
    assert np.allclose(north, 0.0)
    assert np.all(north < 360.0)
    south = segment_bearings(np.zeros(2), np.array([1.0, 0.0]))
    assert np.allclose(south, 180.0)
    west = segment_bearings(np.array([0.0, -1.0]), np.zeros(2))
    assert np.allclose(west, 270.0)
    # Slightly west of north must stay in [0, 360), not go negative.
    nnw = segment_bearings(np.array([0.0, -1e-4]), np.array([0.0, 1.0]))
    assert np.all((nnw >= 0.0) & (nnw < 360.0)) and nnw[0] > 359.0


def test_bearings_centred_inside_one_sided_at_ends_and_great_circle():
    # A right-angle corner: east then north.  The corner's centred bearing
    # is about north-east; the ends take their own segment.
    b = segment_bearings(np.array([0.0, 0.01, 0.01]), np.array([0.0, 0.0, 0.01]))
    assert b[0] == pytest.approx(90.0, abs=1e-3)
    assert b[1] == pytest.approx(45.0, abs=0.01)
    assert b[2] == pytest.approx(0.0, abs=1e-3)
    # Great-circle azimuth: London -> New York departs west-north-west.
    gc = segment_bearings(np.array([-0.1, -74.0]), np.array([51.5, 40.7]))
    assert gc[0] == pytest.approx(288.3, abs=0.3)
    # Duplicates share their run's bearing.
    dup = segment_bearings(np.array([0.0, 0.0, 1.0]), np.zeros(3))
    assert np.allclose(dup, 90.0)
    # Doubling straight back uses the forward difference.
    back = segment_bearings(np.array([0.0, 1.0, 0.0]), np.zeros(3))
    assert np.allclose(back, [90.0, 270.0, 270.0])


def test_bearings_refuse_no_direction():
    with pytest.raises(GeometryError):
        segment_bearings(np.array([1.0, 1.0]), np.array([2.0, 2.0]))
    with pytest.raises(GeometryError):
        segment_bearings(np.array([1.0]), np.array([2.0, 3.0]))
    with pytest.raises(GeometryError):
        segment_bearings(np.array([]), np.array([]))


# ---------------------------------------------------------------------------
# buffer


def _check_buffer(coords, half_width, *, seed=3, n=6000, spq=4):
    polygon = buffer_polyline(coords, half_width, segments_per_quarter=spq)
    for ring in polygon:
        assert ring[0] == ring[-1] and len(ring) >= 4
    arr = np.asarray(coords, dtype=float)
    proj = local_projection(float(arr[:, 1].mean()), float(arr[:, 0].mean()))
    lon, lat, _ = densify_polyline(coords, 20.0)
    lx, ly = proj.forward(lon, lat)
    rng = np.random.default_rng(seed)
    # Points scattered round the line out to 1.5 half-widths.
    pick = rng.integers(0, len(lx), n)
    ang = rng.uniform(0.0, 2.0 * np.pi, n)
    rad = rng.uniform(0.0, 1.5 * half_width, n)
    qx = lx[pick] + rad * np.cos(ang)
    qy = ly[pick] + rad * np.sin(ang)
    dist = _distance_to_line_m(proj, coords, qx, qy)
    qlon, qlat = proj.inverse(qx, qy)
    inside = point_in_polygon(qlon, qlat, [polygon])
    near = dist <= half_width
    far = dist > 1.1 * half_width
    assert near.sum() > 100 and far.sum() > 100
    assert np.all(inside[near]), "a point within the half-width was excluded"
    assert not np.any(inside[far]), "a point beyond 1.1x the half-width was included"
    # The boundary itself (vertices and edge midpoints) stays within the
    # band (half-width, 1.1 half-width].
    for ring in polygon:
        r = np.asarray(ring)
        bx, by = proj.forward(r[:, 0], r[:, 1])
        mx, my = 0.5 * (bx[:-1] + bx[1:]), 0.5 * (by[:-1] + by[1:])
        d = _distance_to_line_m(proj, coords, np.concatenate((bx, mx)),
                                np.concatenate((by, my)))
        assert d.min() >= half_width
        assert d.max() <= 1.1 * half_width
    return polygon


def test_buffer_straight_line_contains_and_excludes():
    polygon = _check_buffer([(-3.9, 51.6), (-3.75, 51.62)], 250.0)
    assert len(polygon) == 1
    ring = np.asarray(polygon[0])
    # Outer ring counter-clockwise (RFC 7946).
    area = 0.5 * np.sum(ring[:-1, 0] * ring[1:, 1] - ring[1:, 0] * ring[:-1, 1])
    assert area > 0


def test_buffer_long_straight_run_at_high_latitude():
    # A 40 km straight run: an output edge straight in the projection would
    # bow tens of metres away from the same edge straight in (lon, lat),
    # which is how GeoJSON and point_in_polygon read it.
    polygon = _check_buffer([(-45.0, 66.0), (-44.1, 66.0), (-44.4, 66.15)],
                            1000.0, seed=9)
    ring = np.asarray(polygon[0])
    edge = _great_circle_m(ring[:-1, 0], ring[:-1, 1], ring[1:, 0], ring[1:, 1])
    assert edge.max() <= 2.0 * 1000.0 * 1.01


def test_buffer_sharp_hairpin():
    # Out 5 km east, back west 80 m to the north: the legs sit well inside
    # one another's corridor and the inside of the bend folds.
    coords = [(-3.5, 52.0), (-3.4266, 52.0), (-3.5, 52.0007)]
    polygon = _check_buffer(coords, 300.0)
    assert len(polygon) == 1


def test_buffer_zigzag_and_self_crossing():
    zigzag = [(-3.50, 52.00), (-3.49, 52.01), (-3.48, 52.00), (-3.47, 52.01),
              (-3.46, 52.0), (-3.4605, 52.002)]
    _check_buffer(zigzag, 200.0, seed=5)
    figure8 = [(0.0, 0.0), (0.02, 0.02), (0.02, 0.0), (0.0, 0.02), (0.0, 0.0)]
    _check_buffer(figure8, 150.0, seed=6)


def test_buffer_closed_loop_has_hole():
    square = [(0.0, 0.0), (0.05, 0.0), (0.05, 0.05), (0.0, 0.05), (0.0, 0.0)]
    polygon = _check_buffer(square, 400.0, seed=7)
    assert len(polygon) == 2
    hole = np.asarray(polygon[1])
    area = 0.5 * np.sum(hole[:-1, 0] * hole[1:, 1] - hole[1:, 0] * hole[:-1, 1])
    assert area < 0  # holes clockwise
    centre = point_in_polygon(np.array([0.025]), np.array([0.025]), [polygon])
    assert not centre[0]


def test_buffer_single_point_is_a_disc():
    polygon = _check_buffer([(10.0, 45.0), (10.0, 45.0)], 100.0)
    ring = np.asarray(polygon[0])
    d = _great_circle_m(10.0, 45.0, ring[:, 0], ring[:, 1])
    assert np.all((d >= 100.0) & (d <= 110.0))
    # About segments_per_quarter chords per quarter circle.
    assert 12 <= len(ring) - 1 <= 24


def test_buffer_segments_per_quarter_sets_density():
    coarse = buffer_polyline([(0.0, 0.0), (0.0, 0.0)], 500.0,
                             segments_per_quarter=3)
    fine = buffer_polyline([(0.0, 0.0), (0.0, 0.0)], 500.0,
                           segments_per_quarter=8)
    assert len(fine[0]) > len(coarse[0])


@pytest.mark.parametrize("half_width", [0.0, -1.0, float("nan"), True])
def test_buffer_refuses_bad_half_width(half_width):
    with pytest.raises(GeometryError):
        buffer_polyline([(0.0, 0.0), (0.01, 0.0)], half_width)


@pytest.mark.parametrize("spq", [0, -2, 2.5, True])
def test_buffer_refuses_bad_segments_per_quarter(spq):
    with pytest.raises(GeometryError):
        buffer_polyline([(0.0, 0.0), (0.01, 0.0)], 50.0, segments_per_quarter=spq)


def test_buffer_refuses_a_corridor_over_a_pole():
    with pytest.raises(GeometryError, match="pole"):
        buffer_polyline([(0.0, 89.95), (180.0, 89.95)], 2000.0)
    with pytest.raises(GeometryError, match="pole"):
        buffer_polyline([(10.0, -89.99), (10.0, -89.9)], 2000.0)


def test_buffer_refuses_continental_extent_and_lattice_blowup():
    with pytest.raises(GeometryError, match="projection centre"):
        buffer_polyline([(-10.0, 40.0), (40.0, 60.0)], 1000.0)
    with pytest.raises(GeometryError, match="lattice nodes"):
        buffer_polyline([(-3.0, 50.0), (3.0, 56.0)], 2.0)


# ---------------------------------------------------------------------------
# point in polygon


def test_point_in_polygon_with_hole_and_multiple_polygons():
    outer = [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0), (0.0, 0.0)]
    hole = [(1.0, 1.0), (1.0, 3.0), (3.0, 3.0), (3.0, 1.0), (1.0, 1.0)]
    other = [(10.0, 10.0), (11.0, 10.0), (10.5, 11.0)]  # open ring
    lon = np.array([0.5, 2.0, 3.5, 5.0, 10.5, 10.9, -1.0])
    lat = np.array([0.5, 2.0, 3.5, 2.0, 10.4, 10.9, 2.0])
    got = point_in_polygon(lon, lat, [[outer, hole], [other]])
    assert got.tolist() == [True, False, True, False, True, False, False]
    assert point_in_polygon(lon, lat, []).tolist() == [False] * 7


def test_point_in_polygon_matches_brute_force_on_random_star():
    rng = np.random.default_rng(11)
    theta = np.sort(rng.uniform(0, 2 * np.pi, 300))
    radius = rng.uniform(0.3, 1.0, 300)
    ring = list(zip(radius * np.cos(theta), radius * np.sin(theta)))
    hole = [(0.1 * math.cos(t), 0.1 * math.sin(t))
            for t in np.linspace(0, 2 * np.pi, 40, endpoint=False)]
    qx = rng.uniform(-1.1, 1.1, 20000)
    qy = rng.uniform(-1.1, 1.1, 20000)
    got = point_in_polygon(qx, qy, [[ring, hole]])
    # Brute-force even-odd over all edges.
    want = np.zeros(qx.size, dtype=bool)
    for r in (ring, hole):
        a = np.asarray(r)
        b = np.roll(a, -1, axis=0)
        for (x0, y0), (x1, y1) in zip(a, b):
            if y0 == y1:
                continue
            straddle = (y0 > qy) != (y1 > qy)
            xc = x0 + (qy - y0) * (x1 - x0) / (y1 - y0)
            want ^= straddle & (qx < xc)
    assert np.array_equal(got, want)
    assert got.reshape(100, 200).shape == (100, 200)


def test_point_in_polygon_across_antimeridian():
    ring = [(179.0, -1.0), (-179.0, -1.0), (-179.0, 1.0), (179.0, 1.0)]
    got = point_in_polygon(np.array([179.5, -179.5, 180.0, 0.0, 170.0]),
                           np.zeros(5), [[ring]])
    assert got.tolist() == [True, True, True, False, False]


def test_point_in_polygon_refuses_degenerate_ring():
    with pytest.raises(GeometryError):
        point_in_polygon(np.zeros(1), np.zeros(1), [[[(0.0, 0.0), (1.0, 1.0)]]])
    with pytest.raises(GeometryError):
        point_in_polygon(np.zeros(2), np.zeros(1), [])


# ---------------------------------------------------------------------------
# rectangle cover


def _check_cover(rects, x, y, *, margin, dx, max_nx, max_ny, align=None):
    member_count = np.zeros(x.size, dtype=int)
    for r in rects:
        assert isinstance(r, Rect)
        assert 1 <= r.nx <= max_nx and 1 <= r.ny <= max_ny
        assert r.x_max == pytest.approx(r.x_min + (r.nx - 1) * dx, abs=1e-6)
        assert r.y_max == pytest.approx(r.y_min + (r.ny - 1) * dx, abs=1e-6)
        m = r.members
        assert np.all(np.diff(m) > 0)
        member_count[m] += 1
        assert np.all(x[m] - r.x_min >= margin - 1e-6)
        assert np.all(r.x_max - x[m] >= margin - 1e-6)
        assert np.all(y[m] - r.y_min >= margin - 1e-6)
        assert np.all(r.y_max - y[m] >= margin - 1e-6)
        if align is not None:
            for v in (r.x_min, r.x_max, r.y_min, r.y_max):
                assert abs(v / align - round(v / align)) < 1e-9
        # Shrunk to members: no wasted cell row beyond the margin (or the
        # snapping) on the lower edges.
        slack = (align if align is not None else 0.0) + 1e-6
        assert x[m].min() - r.x_min <= margin + slack
        assert y[m].min() - r.y_min <= margin + slack
    assert np.all(member_count == 1)


def _line_points(length_m, spacing_m, *, angle_deg=30.0, wiggle_m=0.0, seed=0):
    s = np.arange(0.0, length_m + spacing_m / 2, spacing_m)
    a = math.radians(angle_deg)
    rng = np.random.default_rng(seed)
    w = wiggle_m * np.sin(s / 7000.0) + rng.normal(0.0, 5.0, s.size)
    return (s * math.cos(a) - w * math.sin(a) + 12345.0,
            s * math.sin(a) + w * math.cos(a) - 6789.0)


@pytest.mark.parametrize("align", [None, 300.0])
def test_cover_invariants_on_a_line(align):
    x, y = _line_points(200e3, 100.0, wiggle_m=3000.0)
    kwargs = dict(margin_m=2000.0, dx_m=100.0, max_nx=600, max_ny=600)
    rects = cover_with_rectangles(x, y, align_m=align, **kwargs)
    _check_cover(rects, x, y, margin=2000.0, dx=100.0, max_nx=600, max_ny=600,
                 align=align)
    # A 200 km line under 56 km tiles needs few rectangles.
    assert len(rects) <= 6
    again = cover_with_rectangles(x, y, align_m=align, **kwargs)
    assert [(r.x_min, r.y_min, r.nx, r.ny) for r in rects] == \
           [(r.x_min, r.y_min, r.nx, r.ny) for r in again]
    assert all(np.array_equal(a.members, b.members) for a, b in zip(rects, again))


def test_cover_invariants_on_clusters_and_network():
    rng = np.random.default_rng(21)
    centres = rng.uniform(-150e3, 150e3, (25, 2))
    pts = np.concatenate([c + rng.normal(0, 4000, (400, 2)) for c in centres])
    lx, ly = _line_points(120e3, 50.0, angle_deg=-70.0, seed=2)
    x = np.concatenate((pts[:, 0], lx))
    y = np.concatenate((pts[:, 1], ly))
    for kwargs in (dict(margin_m=1000.0, dx_m=50.0, max_nx=400, max_ny=300),
                   dict(margin_m=500.0, dx_m=100.0, max_nx=101, max_ny=151,
                        align_m=500.0)):
        rects = cover_with_rectangles(x, y, **kwargs)
        _check_cover(rects, x, y, margin=kwargs["margin_m"],
                     dx=kwargs["dx_m"], max_nx=kwargs["max_nx"],
                     max_ny=kwargs["max_ny"], align=kwargs.get("align_m"))


def test_cover_single_point_and_tight_fit():
    rects = cover_with_rectangles(np.array([0.0]), np.array([0.0]),
                                  margin_m=1000.0, dx_m=100.0, max_nx=21,
                                  max_ny=21)
    assert len(rects) == 1 and rects[0].nx == 21 and rects[0].ny == 21
    # Points exactly one max extent apart (less the margins) share one rect.
    x = np.array([0.0, 3000.0 - 1e-3])
    rects = cover_with_rectangles(x, np.zeros(2), margin_m=1000.0, dx_m=100.0,
                                  max_nx=51, max_ny=51)
    assert len(rects) == 1 and rects[0].nx <= 51
    assert cover_with_rectangles(np.array([]), np.array([]), margin_m=1.0,
                                 dx_m=1.0, max_nx=5, max_ny=5) == []


def test_cover_merge_pass_joins_small_groups():
    # Strip greedy along x cuts {A, B} | {C}; C fits with B, so the merge
    # pass (or the y-first order) must produce two rectangles, not three.
    x = np.array([0.0, 900.0, 1500.0])
    y = np.array([0.0, 5000.0, 5000.0])
    rects = cover_with_rectangles(x, y, margin_m=0.0, dx_m=100.0, max_nx=11,
                                  max_ny=11)
    _check_cover(rects, x, y, margin=0.0, dx=100.0, max_nx=11, max_ny=11)
    assert len(rects) == 2


def test_cover_timing_250k_points():
    rng = np.random.default_rng(31)
    x, y = _line_points(250e3 * 1.0, 1.0, angle_deg=20.0, wiggle_m=8000.0)
    x = x + rng.normal(0.0, 300.0, x.size)
    assert x.size >= 250_000
    start = time.perf_counter()
    rects = cover_with_rectangles(x, y, margin_m=2000.0, dx_m=100.0,
                                  max_nx=600, max_ny=600, align_m=300.0)
    elapsed = time.perf_counter() - start
    assert elapsed < 20.0  # about 0.1 s on an idle workstation
    counts = np.zeros(x.size, dtype=int)
    for r in rects:
        counts[r.members] += 1
    assert np.all(counts == 1)


@pytest.mark.parametrize("kwargs, match", [
    (dict(margin_m=1000.0, dx_m=0.0, max_nx=10, max_ny=10), "dx_m"),
    (dict(margin_m=1000.0, dx_m=-5.0, max_nx=10, max_ny=10), "dx_m"),
    (dict(margin_m=-1.0, dx_m=100.0, max_nx=10, max_ny=10), "margin_m"),
    (dict(margin_m=600.0, dx_m=100.0, max_nx=11, max_ny=100), "twice the margin"),
    (dict(margin_m=300.0, dx_m=100.0, max_nx=100, max_ny=6), "largest y extent"),
    (dict(margin_m=400.0, dx_m=100.0, max_nx=13, max_ny=13, align_m=500.0),
     "align_m"),
    (dict(margin_m=10.0, dx_m=100.0, max_nx=1, max_ny=10), "max_nx"),
    (dict(margin_m=10.0, dx_m=100.0, max_nx=10, max_ny=10.5), "max_ny"),
    (dict(margin_m=10.0, dx_m=100.0, max_nx=10, max_ny=10, align_m=250.0),
     "whole multiple"),
    (dict(margin_m=10.0, dx_m=100.0, max_nx=10, max_ny=10, align_m=-300.0),
     "align_m"),
    (dict(margin_m=float("nan"), dx_m=100.0, max_nx=10, max_ny=10), "margin_m"),
    (dict(margin_m=True, dx_m=100.0, max_nx=10, max_ny=10), "margin_m"),
    (dict(margin_m=10.0, dx_m=True, max_nx=10, max_ny=10), "dx_m"),
    (dict(margin_m=10.0, dx_m=100.0, max_nx=10, max_ny=10, align_m=True),
     "align_m"),
])
def test_cover_refusals(kwargs, match):
    with pytest.raises(GeometryError, match=match):
        cover_with_rectangles(np.zeros(3), np.zeros(3), **kwargs)


def test_cover_refuses_bad_points():
    with pytest.raises(GeometryError):
        cover_with_rectangles(np.array([0.0, np.nan]), np.zeros(2), margin_m=1.0,
                              dx_m=1.0, max_nx=10, max_ny=10)
    with pytest.raises(GeometryError):
        cover_with_rectangles(np.zeros(3), np.zeros(2), margin_m=1.0,
                              dx_m=1.0, max_nx=10, max_ny=10)
