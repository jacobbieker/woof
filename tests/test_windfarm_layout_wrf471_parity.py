"""Wind-farm layout against unmodified WRF v4.7.1: max_ulp 0, every word.

The fixtures under ``tests/data/oracles/windfarm_layout`` come from
``tools/windfarm_layout_wrf471_oracle`` (see its ``PROVENANCE.md``): WRF's
own ``module_wind_fitch.F`` placement, table read and ``dragcof`` with
``share/module_llxy.F``, compiled byte-unmodified at -O0 in real(4), run on
four synthetic farms that woof laid out and wrote as WRF inputs.  Each case
carries the woof inputs (``deck/deck.json``) it was made from, so the
layout is rebuilt here from scratch and graded:

* the WRF inputs woof writes are byte-identical to the deck WRF read;
* placement: the float32 lat/lon WRF read, ``proj_info`` (cone, rebydx,
  rsw, polei, polej, hemi), ``ts_rx``/``ts_ry`` at max_ulp 0, and
  ``ival``/``jval``/``nkind`` exact;
* tables: ``turbws``/``turbtc``/``turbpw`` and every header word exact;
* ``dragcof`` + ``power1`` on the CPU and (GPU tests) the device at
  max_ulp 0 against WRF and against each other;
* the device upload is raw-word equal to the host layout.

Nothing is held out.  Two negative controls prove the gates can fail: the
float64 WPS-convention transform woof uses for its static fields, rounded
to float32, is NOT WRF's ``ts_rx``; and ``dragcof``'s thrust interpolation
evaluated in binary64 and rounded once is not WRF's real(4) answer.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu
from woof.energy.turbine_library import load_turbine_library
from woof.verify.urban_oracle import load_case, ulp_table
from woof.windfarm import layout as wl
from woof.windfarm.fitch_tables import dragcof_cpu
from tools.windfarm_layout_wrf471_oracle.make_deck import grid_from_spec

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "tests" / "data" / "oracles" / "windfarm_layout"
HARNESS = REPO / "tools" / "windfarm_layout_wrf471_oracle"
#: The committed cases: every fixture directory with a deck.
CASES = tuple(sorted(p.name for p in ROOT.iterdir()
                     if (p / "deck" / "deck.json").is_file()))
DRAG_FIELDS = ("thrcof", "powcof", "tkecof", "power1")


def _words(a) -> np.ndarray:
    a = np.ascontiguousarray(a)
    return a.view(np.uint32) if a.dtype == np.float32 else a


def _assert_words(port, ref, what):
    port, ref = np.asarray(port), np.asarray(ref)
    assert port.dtype == ref.dtype, f"{what}: {port.dtype} vs {ref.dtype}"
    table = ulp_table(port, ref)
    assert table["max_ulp"] == 0 and np.array_equal(_words(port),
                                                    _words(ref)), \
        f"{what}: {table}"


@lru_cache(maxsize=None)
def _case(name: str):
    deck = json.loads((ROOT / name / "deck" / "deck.json").read_text())
    grid = grid_from_spec(deck["grid"], [tuple(n) for n in deck["nests"]])
    turbines = [wl.turbine_from_row(r, n + 1)
                for n, r in enumerate(deck["turbines"])]
    library = load_turbine_library()
    assert library.sha256 == deck["turbine_library_sha256"]
    layout = wl.build_layout(turbines, grid, library=library)
    oracle = {part: load_case(ROOT / name / part)
              for part in ("placement", "tables", "dragcof")}
    return deck, grid, layout, oracle


def _queries(layout, oracle):
    q = oracle["dragcof"]
    kt = q["kt"] - 1
    group = layout.turbine_group[kt]
    return (layout.group_curve[group], q["speedhub"],
            layout.group_rotor_diameter_m[group],
            q["correction_factor"])


# -- provenance --------------------------------------------------------------


def test_fixtures_are_the_ones_the_harness_recorded():
    listed = {}
    for line in (ROOT / "FIXTURES.sha256").read_text().splitlines():
        digest, path = line.split(maxsplit=1)
        listed[path.removeprefix("./")] = digest
    on_disk = {str(p.relative_to(ROOT)) for p in ROOT.rglob("*")
               if p.is_file()
               and p.name not in ("FIXTURES.sha256", "PROVENANCE.md")}
    assert set(listed) == on_disk
    for path, digest in listed.items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == \
            digest, path
    pinned = [line for line in (HARNESS / "SOURCES.sha256").read_text()
              .splitlines() if line and not line.startswith("#")]
    assert (ROOT / "SOURCES.sha256").read_text().splitlines() == pinned
    harness = {}
    for line in (ROOT / "HARNESS.sha256").read_text().splitlines():
        digest, path = line.split(maxsplit=1)
        harness[path] = digest
    for path, digest in harness.items():
        assert hashlib.sha256((HARNESS / path).read_bytes()).hexdigest() \
            == digest, f"{path} changed since the fixtures were made"
    assert "glibc" in (ROOT / "TOOLCHAIN.txt").read_text()
    assert "_ZGV" in (ROOT / "libmvec-report.txt").read_text()   # control


@pytest.mark.parametrize("name", CASES)
def test_woof_writes_the_inputs_wrf_read(name, tmp_path):
    _, _, layout, _ = _case(name)
    layout.write_wrf_inputs(tmp_path)
    deck = ROOT / name / "deck"
    written = sorted(p.name for p in tmp_path.iterdir())
    assert written == sorted(p.name for p in deck.iterdir()
                             if p.name == "windturbines.txt"
                             or p.name.startswith("wind-turbine-"))
    for file in written:
        assert (tmp_path / file).read_bytes() == (deck / file).read_bytes()


# -- placement ----------------------------------------------------------------


@pytest.mark.parametrize("name", CASES)
def test_placement_is_wrfs_to_the_word(name):
    _, _, layout, oracle = _case(name)
    p = oracle["placement"]
    _assert_words(layout.turbine_lat, p["lat"], "lat")
    _assert_words(layout.turbine_lon, p["lon"], "lon")
    proj = layout.projection
    _assert_words(np.array([proj.lat1, proj.lon1], np.float32),
                  p["lat_lon_known"], "XLAT/XLONG(its, jts)")
    _assert_words(proj.members(), p["proj"], "proj_info")
    _assert_words(layout.turbine_x, p["ts_rx"], "ts_rx")
    _assert_words(layout.turbine_y, p["ts_ry"], "ts_ry")
    _assert_words(layout.turbine_i + 1, p["ival"], "ival")
    _assert_words(layout.turbine_j + 1, p["jval"], "jval")
    of_group, _ = layout.wrf_types()
    _assert_words(of_group[layout.turbine_group] + 1, p["nkind"], "nkind")


def test_half_cell_turbines_reach_nints_rounding_edge():
    """The decks put turbines on exact half-cell coordinates: some of them
    land within a float32 ulp or two of .5, where NINT decides."""

    edge = 0
    for name in CASES:
        _, _, layout, _ = _case(name)
        frac = np.abs(np.mod(layout.turbine_x.astype(np.float64), 1.0) - 0.5)
        edge += int(np.count_nonzero(frac < 1e-4))
    assert edge >= 8


def test_negative_control_float64_placement_is_not_wrfs():
    differs = 0
    for name in CASES:
        _, grid, layout, oracle = _case(name)
        x, _ = grid.latlon_to_ij(layout.turbine_lat.astype(np.float64),
                                 layout.turbine_lon.astype(np.float64))
        differs += int(np.count_nonzero(
            _words(np.asarray(x, np.float32))
            != _words(oracle["placement"]["ts_rx"])))
    assert differs > 0


# -- tables -------------------------------------------------------------------


@pytest.mark.parametrize("name", CASES)
def test_tables_are_what_wrf_read(name):
    _, _, layout, oracle = _case(name)
    t = oracle["tables"]
    of_group, types = layout.wrf_types()
    group = layout.turbine_group
    curve = layout.group_curve[group]
    tables = layout.tables
    _assert_words(layout.group_hub_height_m[group], t["hubheight"],
                  "hubheight")
    _assert_words(layout.group_rotor_diameter_m[group], t["diameter"],
                  "diameter")
    _assert_words(tables.standing_thrust[curve], t["stc"], "stc")
    _assert_words(tables.standing_thrust_cutout[curve], t["stc2"], "stc2")
    _assert_words(tables.cut_in_ms[curve], t["cutin"], "cutin")
    _assert_words(tables.cut_out_ms[curve], t["cutout"], "cutout")
    _assert_words((layout.rated_kw[curve] / np.float32(1000.0))
                  .astype(np.float32), t["npower"], "npower")
    for mine, theirs in (("speeds_ms", "turbws"),
                         ("thrust_coefficient", "turbtc"),
                         ("power_kw", "turbpw")):
        full = np.zeros((len(types), 100), np.float32)
        for k, (c, _, _) in enumerate(types):
            full[k] = getattr(tables, mine)[c]
        _assert_words(full, t[theirs], theirs)


# -- dragcof ------------------------------------------------------------------


@pytest.mark.parametrize("name", CASES)
def test_dragcof_cpu_is_wrfs(name):
    _, _, layout, oracle = _case(name)
    out = dragcof_cpu(layout.tables, *_queries(layout, oracle))
    for field in DRAG_FIELDS:
        _assert_words(out[field], oracle["dragcof"][field], field)


def test_axis_tables_are_dragcof_at_each_groups_curve_and_rotor():
    _, _, layout, _ = _case("lc_tangent_4km")
    for g in range(0, layout.n_groups, 37):
        out = dragcof_cpu(layout.tables,
                          np.full(layout.speeds_ms.size,
                                  layout.group_curve[g]),
                          layout.speeds_ms, layout.group_rotor_diameter_m[g])
        _assert_words(layout.group_thrust_coefficient[g], out["thrcof"],
                      "axis Ct")
        _assert_words(layout.group_power_coefficient[g], out["powcof"],
                      "axis Cp")
        _assert_words(layout.group_power_w[g], out["power1"], "axis power")


def test_negative_control_binary64_thrust_is_not_wrfs():
    """The same interpolation in binary64, rounded once, is not real(4)."""

    _, _, layout, oracle = _case("lc_nh_12km")
    curve, speed, _, _ = _queries(layout, oracle)
    t = layout.tables
    ws = t.speeds_ms[curve].astype(np.float64)
    tc = t.thrust_coefficient[curve].astype(np.float64)
    s = speed.astype(np.float64)
    above = ws > s[:, None]
    nu = above.argmax(axis=1)
    nb = np.maximum(nu - 1, 0)
    rows = np.arange(speed.size)
    mid = (speed > t.cut_in_ms[curve]) & (speed < t.cut_out_ms[curve])
    with np.errstate(all="ignore"):
        mutated = tc[rows, nb] + (tc[rows, nu] - tc[rows, nb]) / (
            ws[rows, nu] - ws[rows, nb]) * (s - ws[rows, nb])
    ref = oracle["dragcof"]["thrcof"]
    assert np.count_nonzero(_words(mutated[mid].astype(np.float32))
                            != _words(ref[mid])) > 0


# -- the device ---------------------------------------------------------------


@pytest.mark.gpu
@requires_gpu
def test_upload_is_raw_word_equal_both_ways():
    import cupy as cp

    for name in CASES:
        _, _, layout, _ = _case(name)
        device = layout.to_device(device_id=0)
        for key, array in device.arrays().items():
            host = (getattr(layout.tables, key.split(".", 1)[1])
                    if key.startswith("tables.") else getattr(layout, key))
            assert array.flags.c_contiguous, key
            assert array.dtype == host.dtype, key
            assert np.array_equal(_words(cp.asnumpy(array)), _words(host)), key


def test_the_decks_cover_every_projection_and_a_three_level_ladder():
    codes = set()
    depth = 0
    for name in CASES:
        deck, _, layout, _ = _case(name)
        codes.add(layout.projection.code)
        depth = max(depth, len(deck["nests"]))
        lat = layout.turbine_lat
        assert lat.size == 331
    assert codes == {1, 2, 3}
    assert depth == 2                      # d01 -> d02 -> d03
    lats = np.concatenate([_case(n)[2].turbine_lat for n in CASES])
    lons = np.concatenate([_case(n)[2].turbine_lon for n in CASES])
    assert lats.min() < -75 and lats.max() > 88 and np.any(np.abs(lats) < 5)
    assert lons.min() < -179.9 and lons.max() > 179.7    # across the dateline


@pytest.mark.parametrize("name", CASES)
def test_each_cells_groups_are_wrfs_runs_in_wrfs_order(name):
    """Walk WRF's own turbines k = 1..nt (its ival/jval/nkind): in every
    cell, the runs of consecutive equal types are the layout's groups, in
    order, with their counts -- what a kernel must add, in WRF's order."""

    _, _, layout, oracle = _case(name)
    p = oracle["placement"]
    of_group, _ = layout.wrf_types()
    runs: dict[tuple[int, int], list[list[int]]] = {}
    for kind, i, j in zip(p["nkind"], p["ival"], p["jval"]):
        cell = runs.setdefault((int(j) - 1, int(i) - 1), [])
        if cell and cell[-1][0] == kind:
            cell[-1][1] += 1
        else:
            cell.append([int(kind), 1])
    assert sum(len(r) for r in runs.values()) == layout.n_groups
    for (j, i), expected in runs.items():
        got = [[int(of_group[g]) + 1, int(layout.group_turbines[g])]
               for g in layout.groups_of(j, i)]
        assert got == expected, (j, i)
