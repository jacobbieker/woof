"""The steep-ground terrain pass (woof.static.terrain_slope), on CPU.

Synthetic ground only: a canyon wall at 100 m as steep as the one that
took a nest non-finite in four steps, rolling hills that must keep every
byte, and the footprint independence a moving nest's corridor needs.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from woof.acoustic_adaptation import (STABLE_SLOPE_BY_OFFCENTERING,
                                      steepest_slope)
from woof.static.terrain_slope import (ROUND_REACH, STEEP_GROUND_REACH_M,
                                       cell_slopes, held_slope,
                                       smooth_steep_ground,
                                       steep_ground_reach,
                                       steep_ground_rounds)

DX = 100.0


def _canyon(ny=96, nx=120, seed=7):
    """Rolling ground cut by a narrow V canyon whose walls drop 400 m over
    two to three cells (slope near 1.6, 58 degrees), running obliquely."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:ny, 0:nx].astype(np.float64)
    hills = 900.0 + 40.0 * np.sin(x / 9.0) * np.cos(y / 13.0)
    across = (x - 0.6 * y - 40.0) / 1.2
    canyon = -400.0 * np.exp(-0.5 * (across / 1.5) ** 2)
    return hills + canyon + rng.normal(0.0, 0.5, size=(ny, nx))


def _hills(ny=64, nx=80):
    y, x = np.mgrid[0:ny, 0:nx].astype(np.float64)
    return 300.0 + 30.0 * np.sin(x / 7.0) + 20.0 * np.cos(y / 5.0)


def _laplacian(h):
    out = np.zeros_like(h)
    out[1:-1, 1:-1] = (h[2:, 1:-1] + h[:-2, 1:-1] + h[1:-1, 2:]
                       + h[1:-1, :-2] - 4.0 * h[1:-1, 1:-1]) / DX
    return out


def test_bound_is_the_steepest_measured_stable_slope():
    assert held_slope() == max(row[2] for row in STABLE_SLOPE_BY_OFFCENTERING)
    assert held_slope() == pytest.approx(0.85)


@pytest.mark.parametrize("dx,rounds", [(50.0, 4), (100.0, 4), (300.0, 1),
                                       (500.0, 1), (600.0, 0), (3000.0, 0)])
def test_rounds_fit_in_a_fixed_ground_distance(dx, rounds):
    assert steep_ground_rounds(dx) == rounds
    assert steep_ground_reach(dx) == rounds * ROUND_REACH
    assert steep_ground_reach(dx) * dx <= STEEP_GROUND_REACH_M


def test_cell_slopes_read_faces_like_the_acoustic_rule():
    ground = _canyon()
    assert cell_slopes(ground, DX, DX).max() == pytest.approx(
        steepest_slope(ground, DX, DX).slope)


def test_gentle_ground_is_returned_untouched():
    ground = _hills()
    assert steepest_slope(ground, DX, DX).slope < held_slope()
    eased, record = smooth_steep_ground(ground, DX, DX)
    assert record is None
    assert eased is ground


def test_a_canyon_is_smoothed_to_the_bound_and_nothing_else_moves():
    ground = _canyon()
    assert steepest_slope(ground, DX, DX).slope > 1.4
    eased, record = smooth_steep_ground(ground, DX, DX)
    assert steepest_slope(eased, DX, DX).slope <= held_slope()
    assert record["bound"] == held_slope()
    assert record["rounds"] == steep_ground_rounds(DX)
    assert 1 <= record["rounds_used"] <= record["rounds"]
    moved = eased != ground
    assert record["cells_changed"] == int(moved.sum())
    # Smoothed, not cut: the canyon floor is no sharper than it was.
    assert (np.abs(_laplacian(eased))[moved].max()
            < np.abs(_laplacian(ground))[moved].max())
    # Ground away from the canyon keeps every byte.
    y, x = np.mgrid[0:ground.shape[0], 0:ground.shape[1]]
    far = np.abs((x - 0.6 * y - 40.0) / np.hypot(1.0, 0.6)) > 30
    assert not moved[far].any()
    assert moved.mean() < 0.4


def test_a_cell_depends_only_on_ground_within_the_reach():
    """A footprint smoothed with ``reach`` cells of extra ground equals the
    same cells of a larger domain smoothed whole: a moving nest's corridor
    crop must equal the footprint built directly."""
    ground = _canyon(ny=140, nx=170)
    reach = steep_ground_reach(DX)
    whole, _ = smooth_steep_ground(ground, DX, DX)
    for j0, i0, ny, nx in ((30, 25, 50, 60), (25, 70, 40, 45),
                           (60, 40, 55, 70), (22, 22, 20, 20)):
        window = ground[j0 - reach:j0 + ny + reach,
                        i0 - reach:i0 + nx + reach]
        part, _ = smooth_steep_ground(window, DX, DX)
        inner = part[reach:reach + ny, reach:reach + nx]
        expected = whole[j0:j0 + ny, i0:i0 + nx]
        assert np.array_equal(inner.view(np.uint64),
                              expected.view(np.uint64))


def test_non_finite_ground_is_refused():
    ground = _canyon()
    ground[3, 4] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        smooth_steep_ground(ground, DX, DX)


# --- the high-resolution terrain path --------------------------------------

def _grid(ny, nx, dx=DX):
    return SimpleNamespace(e_sn=ny + 1, e_we=nx + 1, dx=dx, dy=dx)


def _windows(ground, ny, nx, halo, reach, j0, i0):
    """The halo-extended and the reach-wider windows a footprint samples."""
    extended = ground[j0 - halo:j0 + ny + halo, i0 - halo:i0 + nx + halo]
    wide = ground[j0 - halo - reach:j0 + ny + halo + reach,
                  i0 - halo - reach:i0 + nx + halo + reach]
    return extended.copy(), wide.copy()


@pytest.mark.parametrize("smoothing", [None, ("1-2-1", 3), ("1-2-1", 6),
                                       ("smth-desmth", 2)])
def test_gentle_overlay_bytes_are_unchanged(monkeypatch, smoothing):
    """Whatever the smoother -- including one reaching past the halo, whose
    edge rows depend on the frozen ring -- gentle ground keeps its bytes."""
    from woof.static.highres import _smoothed_steep_ground
    from woof.static.terrain_smoothing import (WPS_DEFAULT, TerrainSmoothing,
                                               smooth_terrain_reference)
    monkeypatch.setenv("WOOF_STATIC_PYTHON", "1")
    setting = (WPS_DEFAULT if smoothing is None
               else TerrainSmoothing(*smoothing))
    ny, nx, halo = 30, 40, 3
    reach = steep_ground_reach(DX)
    ground = _hills(ny + 2 * (halo + reach), nx + 2 * (halo + reach))
    extended, wide = _windows(ground, ny, nx, halo, reach,
                              halo + reach, halo + reach)
    got, record = _smoothed_steep_ground(extended, wide, reach,
                                         _grid(ny, nx), halo=halo,
                                         terrain_smoothing=setting)
    assert record is None
    expected = smooth_terrain_reference(extended, setting)[
        halo:halo + ny, halo:halo + nx]
    assert np.array_equal(got.view(np.uint64), expected.view(np.uint64))


def test_overlay_smooths_a_canyon_alike_in_every_footprint(monkeypatch):
    from woof.static import highres
    from woof.static.highres import _smoothed_steep_ground
    from woof.static.terrain_smoothing import WPS_DEFAULT
    monkeypatch.setenv("WOOF_STATIC_PYTHON", "1")
    # Off in production since the 1.1.0 merge (upstream's terrain
    # auto-smoothing owns steep ground); the pass itself is still held to
    # its footprint invariance here.
    monkeypatch.setattr(highres, "STEEP_GROUND_PASS", True)
    halo, reach = 3, steep_ground_reach(DX)
    ground = _canyon(ny=160, nx=190)
    ny, nx = 60, 70
    built = {}
    for name, (j0, i0) in {"a": (40, 40), "b": (52, 61)}.items():
        extended, wide = _windows(ground, ny, nx, halo, reach, j0, i0)
        got, record = _smoothed_steep_ground(
            extended, wide, reach, _grid(ny, nx), halo=halo,
            terrain_smoothing=WPS_DEFAULT)
        assert record is not None and record["cells_changed"] > 0
        assert steepest_slope(got, DX, DX).slope <= held_slope()
        built[name] = (j0, i0, got)
    (ja, ia, a), (jb, ib, b) = built["a"], built["b"]
    dj, di = jb - ja, ib - ia
    assert np.array_equal(a[dj:, di:].view(np.uint64),
                          b[:ny - dj, :nx - di].view(np.uint64))


def test_steep_ground_outside_the_domain_records_nothing(monkeypatch):
    """A wall in the ring past the halo may move ring cells, but the
    domain's own cells keep their bytes and no record is written."""
    from woof.static.highres import _smoothed_steep_ground
    from woof.static.terrain_smoothing import (WPS_DEFAULT,
                                               smooth_terrain_reference)
    monkeypatch.setenv("WOOF_STATIC_PYTHON", "1")
    ny, nx, halo = 30, 40, 3
    reach = steep_ground_reach(DX)
    total = (ny + 2 * (halo + reach), nx + 2 * (halo + reach))
    ground = _hills(*total)
    ground[:, :3] += 600.0          # a cliff at the far west of the ring
    extended, wide = _windows(ground, ny, nx, halo, reach,
                              halo + reach, halo + reach)
    assert steepest_slope(wide, DX, DX).slope > 2.0
    got, record = _smoothed_steep_ground(extended, wide, reach,
                                         _grid(ny, nx), halo=halo,
                                         terrain_smoothing=WPS_DEFAULT)
    assert record is None
    expected = smooth_terrain_reference(extended, WPS_DEFAULT)[
        halo:halo + ny, halo:halo + nx]
    assert np.array_equal(got.view(np.uint64), expected.view(np.uint64))


def test_overlay_without_a_wide_window_is_the_old_path(monkeypatch):
    from woof.static.highres import _smoothed_steep_ground
    from woof.static.terrain_smoothing import (WPS_DEFAULT,
                                               smooth_terrain_reference)
    monkeypatch.setenv("WOOF_STATIC_PYTHON", "1")
    ground = _canyon(ny=20, nx=24)
    got, record = _smoothed_steep_ground(ground, None, 0, _grid(14, 18),
                                         halo=3,
                                         terrain_smoothing=WPS_DEFAULT)
    assert record is None
    np.testing.assert_array_equal(
        got, smooth_terrain_reference(ground, WPS_DEFAULT)[3:17, 3:21])


@pytest.mark.parametrize("dx,widens", [(100.0, True), (3000.0, False)])
def test_terrain_window_covers_the_ring_the_pass_reads(dx, widens):
    from woof.static.build import HALO
    from woof.static.highres_fetch import domain_footprint
    from woof.static.highres_production import _terrain_bbox
    from woof.static.lambert import LambertGrid
    grid = LambertGrid(ref_lat=35.0, ref_lon=-105.0, truelat1=30.0,
                       truelat2=60.0, stand_lon=-105.0, dx=dx, dy=dx,
                       e_we=41, e_sn=33)
    narrow = domain_footprint(grid, HALO)
    window = _terrain_bbox(narrow, grid)
    if not widens:
        assert window == narrow
        return
    ring = domain_footprint(grid, HALO + steep_ground_reach(dx))
    assert window.lat_min <= ring.lat_min and window.lat_max >= ring.lat_max
    assert window.lon_min <= ring.lon_min and window.lon_max >= ring.lon_max
    assert window.lat_min < narrow.lat_min and window.lon_max > narrow.lon_max
    assert _terrain_bbox(narrow, None) == narrow
