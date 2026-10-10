"""Binary64 local-plane deformation geometry for fine hex meshes.

The v8.4.1 deformation weights were built in binary32 from Earth-centred
coordinates divided by the sphere radius, as the single-precision reference
build does.  binary32 holds those coordinates to 0.5 m whatever the cell size,
so at 50 m the weights were wrong by ~18 % (measured below).  ``local64``
evaluates the same formula in binary64 about each cell centre and casts once;
``auto`` keeps the native bytes on every mesh at or above 500 m spacing (every
minted class, every anchor, every coarse proof) and switches below it.
"""

from __future__ import annotations

import numpy as np
import pytest

from woof.hex.errors import ConfigurationRefusal
from woof.hex.mixing_v841 import (
    LOCAL64_GEOMETRY_QUANTUM_RATIO,
    finest_cell_spacing_m,
    initialize_deformation_weights_v841,
    select_deformation_geometry,
)
from woof.hex.precision_probe import (
    analytic_deformation_weights,
    deformation_geometry_error,
    synthetic_hex_patch,
)

NAMES = ("coef_c2", "coef_s2", "coef_cs")


def _max_rel(actual, reference, rows) -> float:
    a = np.asarray(actual, dtype=np.float64)[rows]
    r = np.asarray(reference, dtype=np.float64)[rows]
    return float(np.max(np.abs(a - r)) / np.max(np.abs(r)))


@pytest.mark.parametrize("spacing", [50.0, 100.0])
def test_local64_weights_match_the_analytic_hexagon_at_fine_spacing(spacing):
    patch = synthetic_hex_patch(spacing, rings=3)
    reference = analytic_deformation_weights(patch)
    rows = patch.interior_cells
    assert rows.size == 19

    local = initialize_deformation_weights_v841(patch, dtype=np.float32, geometry="local64")
    native = initialize_deformation_weights_v841(patch, dtype=np.float32, geometry="native")
    local_err = max(_max_rel(getattr(local, n), ref, rows) for n, ref in zip(NAMES, reference))
    native_err = max(_max_rel(getattr(native, n), ref, rows) for n, ref in zip(NAMES, reference))

    assert local.geometry == "local64" and local.coef_c2.dtype == np.float32
    # binary32 storage of a binary64-exact value: a few 1e-8.
    assert local_err < 1.0e-6
    # The old binary32 Earth-centred evaluation: percent-level at 50-100 m.
    assert native_err > 5.0e-2
    assert native_err / local_err > 1.0e5


def test_local64_matches_the_binary64_mirror_off_the_equator():
    """At 52 N meridian convergence rotates each cell's frame away from the
    patch frame, so the analytic patch reference no longer applies; the
    binary64 Earth-centred mirror is the reference there."""

    patch = synthetic_hex_patch(50.0, rings=3, lat_deg=52.3, lon_deg=-1.7)
    native64 = initialize_deformation_weights_v841(patch, dtype=np.float64, geometry="native")
    local64 = initialize_deformation_weights_v841(patch, dtype=np.float64, geometry="local64")
    local32 = initialize_deformation_weights_v841(patch, dtype=np.float32, geometry="local64")
    rows = patch.interior_cells
    for name in NAMES:
        assert _max_rel(getattr(local64, name), getattr(native64, name), rows) < 1.0e-8
        assert _max_rel(getattr(local32, name), getattr(native64, name), rows) < 1.0e-6


def test_local64_agrees_with_the_mirror_everywhere_including_signs_and_halo():
    """Whole patch, every slot: the cellsOnEdge sign flips and the halo
    guard's zero rows are the native mirror's."""

    patch = synthetic_hex_patch(3000.0, rings=3, lat_deg=40.0, lon_deg=-97.0)
    native64 = initialize_deformation_weights_v841(patch, dtype=np.float64, geometry="native")
    local64 = initialize_deformation_weights_v841(patch, dtype=np.float64, geometry="local64")
    boundary = np.setdiff1d(np.arange(patch.arrays["nEdgesOnCell"].size), patch.interior_cells)
    assert boundary.size > 0
    for name in NAMES:
        mirror = getattr(native64, name)
        ours = getattr(local64, name)
        assert np.all(ours[boundary] == 0.0) and np.all(mirror[boundary] == 0.0)
        np.testing.assert_allclose(ours, mirror, rtol=0.0, atol=1.0e-9 * np.max(np.abs(mirror)))
        assert np.array_equal(np.sign(ours), np.sign(mirror))


def test_local64_handles_a_cell_at_the_pole():
    patch = synthetic_hex_patch(100.0, rings=2, lat_deg=90.0, lon_deg=0.0)
    native64 = initialize_deformation_weights_v841(patch, dtype=np.float64, geometry="native")
    local64 = initialize_deformation_weights_v841(patch, dtype=np.float64, geometry="local64")
    rows = patch.interior_cells
    for name in NAMES:
        assert _max_rel(getattr(local64, name), getattr(native64, name), rows) < 1.0e-6


def test_auto_keeps_native_bytes_at_and_above_500_m():
    for spacing in (711.0, 3000.0, 15000.0):
        patch = synthetic_hex_patch(spacing, rings=2, lat_deg=35.0, lon_deg=-97.0)
        auto = initialize_deformation_weights_v841(patch, dtype=np.float32)
        native = initialize_deformation_weights_v841(patch, dtype=np.float32, geometry="native")
        assert auto.geometry == "native"
        assert auto.finest_spacing_m == pytest.approx(spacing, rel=1.0e-4)
        for name in NAMES:
            assert getattr(auto, name).tobytes() == getattr(native, name).tobytes()


def test_auto_switches_to_local64_below_500_m():
    for spacing in (50.0, 100.0, 250.0, 380.0):
        patch = synthetic_hex_patch(spacing, rings=2)
        auto = initialize_deformation_weights_v841(patch, dtype=np.float32)
        local = initialize_deformation_weights_v841(patch, dtype=np.float32, geometry="local64")
        assert auto.geometry == "local64"
        for name in NAMES:
            assert getattr(auto, name).tobytes() == getattr(local, name).tobytes()


def test_auto_keeps_the_binary64_scaffold_native_at_any_spacing():
    patch = synthetic_hex_patch(50.0, rings=2)
    auto = initialize_deformation_weights_v841(patch, dtype=np.float64)
    native = initialize_deformation_weights_v841(patch, dtype=np.float64, geometry="native")
    assert auto.geometry == "native"
    for name in NAMES:
        assert getattr(auto, name).tobytes() == getattr(native, name).tobytes()


def test_the_gate_is_the_documented_ratio():
    radius = 6_371_229.0
    quantum = float(np.spacing(np.float32(radius)))
    assert quantum == 0.5
    boundary = quantum / LOCAL64_GEOMETRY_QUANTUM_RATIO
    assert boundary == pytest.approx(500.0)
    assert select_deformation_geometry(np.float32, radius, 499.0) == "local64"
    assert select_deformation_geometry(np.float32, radius, 501.0) == "native"
    assert select_deformation_geometry(np.float64, radius, 1.0) == "native"
    # A mesh with no measurable neighbour pair cannot be judged fine.
    assert select_deformation_geometry(np.float32, radius, float("inf")) == "native"


def test_finest_spacing_is_the_class_keys_min_dcEdge():
    """The gate reads the regional class key's own measurement, so the two
    cannot classify one cull two ways."""

    patch = synthetic_hex_patch(600.0, rings=2)
    patch.arrays["dcEdge"] = patch.arrays["dcEdge"].copy()
    patch.arrays["dcEdge"][3] = 499.0
    assert finest_cell_spacing_m(patch) == 499.0
    weights = initialize_deformation_weights_v841(patch)
    assert weights.geometry == "local64" and weights.finest_spacing_m == 499.0


def test_finest_spacing_falls_back_to_coordinates_without_dcEdge():
    patch = synthetic_hex_patch(100.0, rings=2)
    del patch.arrays["dcEdge"]
    assert finest_cell_spacing_m(patch) == pytest.approx(100.0, rel=1.0e-6)
    assert initialize_deformation_weights_v841(patch).geometry == "local64"


def test_every_mode_records_the_spacing_it_read():
    patch = synthetic_hex_patch(100.0, rings=1)
    for mode in ("auto", "native", "local64"):
        assert initialize_deformation_weights_v841(patch, geometry=mode).finest_spacing_m == 100.0


def test_an_unknown_geometry_mode_is_refused():
    patch = synthetic_hex_patch(100.0, rings=1)
    with pytest.raises(ConfigurationRefusal, match="deformation_geometry"):
        initialize_deformation_weights_v841(patch, geometry="float64")


def test_local64_refuses_a_planar_mesh_like_the_mirror():
    patch = synthetic_hex_patch(100.0, rings=1)
    patch.attrs["on_a_sphere"] = "NO"
    for mode in ("auto", "native", "local64"):
        with pytest.raises(ConfigurationRefusal, match="on_a_sphere"):
            initialize_deformation_weights_v841(patch, geometry=mode)


def test_local64_refuses_out_of_range_vertices_on_an_active_cell():
    patch = synthetic_hex_patch(100.0, rings=2)
    cell = int(patch.interior_cells[0])
    patch.arrays["verticesOnCell"] = patch.arrays["verticesOnCell"].copy()
    patch.arrays["verticesOnCell"][cell, 2] = 10_000
    with pytest.raises(ValueError, match="verticesOnCell reaches outside"):
        initialize_deformation_weights_v841(patch, geometry="local64")


def test_local64_refuses_a_degenerate_cell_count():
    patch = synthetic_hex_patch(100.0, rings=1)
    patch.arrays["nEdgesOnCell"] = patch.arrays["nEdgesOnCell"].copy()
    patch.arrays["nEdgesOnCell"][0] = 2
    with pytest.raises(ValueError, match="invalid nEdgesOnCell"):
        initialize_deformation_weights_v841(patch, geometry="local64")


def test_the_probe_reports_the_old_and_new_error():
    report = deformation_geometry_error(50.0)
    assert report["auto32_geometry"] == "local64"
    assert report["local64_32"] < 1.0e-6
    assert report["native32"] > 5.0e-2
    assert report["local64_32_vs_native64"] < 1.0e-6
