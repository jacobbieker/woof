"""WRF's windfarm_opt = 1 as a setting: config, files, identity, pricing.

The kernel is graded against WRF v4.7.1 in
tests/test_windfarm_fitch_wrf471_parity.py and the driver seam on the
device in tests/test_windfarm_coupling_gpu.py; this file holds what decides
whether and with which turbines the scheme runs.  Off is the default, and a
run without it must resolve, fingerprint and checkpoint exactly as before
the four fields existed.  Synthetic farms only.
"""
from __future__ import annotations

import ast
import csv
import dataclasses
import math
import tomllib
from decimal import Decimal, getcontext
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from woof.config import (WINDFARM_RUN_DEFAULTS, RunConfig, validate_run_config,
                          windfarm_refusal)
from woof.core import windfarm_coupling as wc
from woof.static.lambert import LambertGrid
from woof.windfarm import layout as wl

REPO = Path(__file__).resolve().parents[1]
CURVE = "V112/3000"
LAT0, LON0 = 35.20, -97.90


def _cfg(**over) -> RunConfig:
    base = dict(nx=41, ny=41, nz=10, dx=1000.0, dy=1000.0, ztop=20000.0,
                dt=6.0, run_seconds=600.0, bl_pbl_physics=5,
                sf_sfclay_physics=5)
    base.update(over)
    return RunConfig(**base)


def _grid(n=41, dx=1000.0):
    return LambertGrid(ref_lat=LAT0 + 0.01, ref_lon=LON0 + 0.01,
                       truelat1=30.0, truelat2=60.0, stand_lon=LON0,
                       dx=dx, dy=dx, e_we=n + 1, e_sn=n + 1)


def _farm(spacing=600.0, size=3):
    rows = []
    for a in range(size):
        for b in range(size):
            lat = LAT0 + a * spacing / 111195.0
            lon = LON0 + b * spacing / (111195.0
                                        * math.cos(math.radians(LAT0)))
            rows.append({"turbine_id": f"wt{a}{b}", "lat": lat, "lon": lon,
                         "hub_height_m": 94.0, "curve": CURVE})
    return rows


def _write_csv(path, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _words(a):
    return np.ascontiguousarray(a).view(np.uint32 if a.dtype == np.float32
                                        else np.int32)


def _assert_same_layout(a, b):
    """Two FitchLayouts that add the same thing in the same order."""
    assert a.shape == b.shape
    assert np.array_equal(a.cell_offsets, b.cell_offsets)
    for name in ("cell", "count"):
        assert np.array_equal(getattr(a, name), getattr(b, name)), name
    for name in ("hub_height_m", "rotor_diameter_m"):
        assert np.array_equal(_words(getattr(a, name)),
                              _words(getattr(b, name))), name
    # The curve numbering may differ (WRF numbers machines, the library
    # curves); each group's own table must not.
    for g in range(a.cell.size):
        ca, cb = int(a.curve[g]), int(b.curve[g])
        n = int(a.nval[ca])
        assert n == int(b.nval[cb])
        for name in ("speeds_ms", "thrust_coefficient", "power_kw"):
            assert np.array_equal(_words(getattr(a, name)[ca, :n]),
                                  _words(getattr(b, name)[cb, :n])), name
        assert _words(a.standing_thrust[ca:ca + 1]) == \
            _words(b.standing_thrust[cb:cb + 1])


# ---------------------------------------------------------------------------
# RunConfig
# ---------------------------------------------------------------------------

def test_the_fields_are_appended_last_at_wrfs_defaults():
    names = [f.name for f in dataclasses.fields(RunConfig)]
    # appended after them: the output-only rrtmg_spectral_output, WRF's
    # couple_farms, the output-only woof.radtran diagnostic's eight keys,
    # then the shallow-cumulus selector and cumulus-radiation feedback,
    # the output-only solar_diagnostics and radtran_aerosol, WRF's
    # use_theta_m and the output-only do_avgflx_em
    assert names[-20:] == ["windfarm_opt", "windfarm_ij",
                           "windfarm_tke_factor", "windfarm_turbine_file",
                           "rrtmg_spectral_output", "couple_farms",
                           "radtran_wavelengths", "radtran_interval_s", "radtran_umu",
                           "radtran_phi", "radtran_mol_abs_param", "radtran_solver",
                           "radtran_streams", "radtran_pseudospherical",
                           "shcu_physics", "cu_rad_feedback",
                           "solar_diagnostics", "radtran_aerosol",
                           "use_theta_m", "do_avgflx_em"]
    for name, default in WINDFARM_RUN_DEFAULTS.items():
        assert RunConfig.__dataclass_fields__[name].default == default
    # Registry.EM_COMMON:3003-3005.
    assert wc.WINDFARM_RUN_DEFAULTS["windfarm_tke_factor"] == 0.25


@pytest.mark.parametrize("kwargs, needle", [
    (dict(windfarm_opt=2), "MAV"),
    (dict(windfarm_opt=True), "windfarm_opt = True"),
    (dict(windfarm_opt=0, windfarm_ij=2), "windfarm_ij = 2"),
    (dict(windfarm_opt=0, windfarm_tke_factor=-0.1), "finite number >= 0"),
    (dict(windfarm_opt=0, windfarm_tke_factor=float("nan")), ">= 0"),
    (dict(windfarm_opt=1, bl_pbl_physics=5), "needs windfarm_turbine_file"),
    (dict(windfarm_opt=1, windfarm_turbine_file="f.csv", bl_pbl_physics=1),
     "needs MYNN"),
    (dict(windfarm_opt=1, windfarm_turbine_file="f.csv", bl_pbl_physics=0),
     "km_opt = 2"),
    (dict(windfarm_opt=1, windfarm_ij=1, windfarm_turbine_file="f.csv",
          bl_pbl_physics=5), "windfarm_ij = 1"),
])
def test_each_refusal_names_what_would_break(kwargs, needle):
    why = windfarm_refusal(**kwargs)
    assert why is not None and needle in why


@pytest.mark.parametrize("kwargs", [
    dict(windfarm_opt=0),
    # A shared file with the switch on other domains only: inert here.
    dict(windfarm_opt=0, windfarm_turbine_file="farm.csv", bl_pbl_physics=1),
    dict(windfarm_opt=1, windfarm_turbine_file="farm.csv", bl_pbl_physics=5),
    dict(windfarm_opt=1, windfarm_ij=1, windfarm_tke_factor=1.0,
         windfarm_turbine_file="windturbines-ij.txt", bl_pbl_physics=5),
])
def test_the_ported_combinations_pass(kwargs):
    assert windfarm_refusal(**kwargs) is None


def test_validate_run_config_refuses_the_farm_without_mynn():
    cfg = _cfg(bl_pbl_physics=1, sf_sfclay_physics=1, windfarm_opt=1,
               windfarm_turbine_file="farm.csv")
    with pytest.raises(ValueError, match="needs MYNN"):
        validate_run_config(cfg)


def test_direct_physics_initialization_has_the_same_refusal():
    from woof.core.physics import initialize_physics

    cfg = _cfg(bl_pbl_physics=1, sf_sfclay_physics=1, windfarm_opt=1,
               windfarm_turbine_file="farm.csv")
    with pytest.raises(ValueError, match="needs MYNN"):
        initialize_physics(SimpleNamespace(), cfg)


def test_a_route_without_a_grid_or_with_a_slab_is_refused(tmp_path):
    from woof.core.physics import _attach_windfarm

    path = _write_csv(tmp_path / "farm.csv", _farm())
    cfg = _cfg(windfarm_opt=1, windfarm_turbine_file=str(path))
    state = SimpleNamespace(mup=np.zeros((41, 41), np.float32))
    with pytest.raises(ValueError, match="no projected grid"):
        _attach_windfarm(state, cfg, {}, None)
    slab = SimpleNamespace(mup=np.zeros((16, 41), np.float32))
    with pytest.raises(ValueError, match="streamed slab"):
        _attach_windfarm(slab, cfg, {}, _grid())


def test_moving_and_spawned_nests_are_refused_by_name():
    from woof.experiment import _refuse_moving_windfarm

    on = SimpleNamespace(windfarm_opt=1)
    domains = [SimpleNamespace(grid_id=1, run=on, follow=None, spawn=None),
               SimpleNamespace(grid_id=2, run=on, follow=object(),
                               spawn=None)]
    with pytest.raises(ValueError, match="grid_id = 2 .* moves"):
        _refuse_moving_windfarm(domains, None, "test")
    domains[1].follow = None
    domains[1].spawn = object()
    with pytest.raises(ValueError, match="windfarm_opt = 1"):
        _refuse_moving_windfarm(domains, None, "test")
    domains[1].spawn = None
    _refuse_moving_windfarm(domains, None, "test")
    relocation = SimpleNamespace(enabled=True, moves=(1,), follow=None,
                                 grid_id=2)
    with pytest.raises(ValueError, match="grid_id = 2"):
        _refuse_moving_windfarm(domains, relocation, "test")


def test_per_domain_keys_and_preparation_inert():
    from woof.experiment import _DOMAIN_RUN_OVERRIDES
    from woof.ingest.prepared_cache import PREPARATION_INERT_RUN_FIELDS

    assert {"windfarm_opt", "windfarm_turbine_file"} <= set(
        _DOMAIN_RUN_OVERRIDES)
    # WRF scope 1: one value for the whole run.
    assert not {"windfarm_ij", "windfarm_tke_factor"} & set(
        _DOMAIN_RUN_OVERRIDES)
    assert {f"run.{name}" for name in WINDFARM_RUN_DEFAULTS} <= \
        PREPARATION_INERT_RUN_FIELDS


def test_the_packed_ensemble_is_routed_to_the_member_driver():
    from woof.ensemble.suite_capabilities import plan_suite

    plan = plan_suite(_cfg(windfarm_opt=1, windfarm_turbine_file="farm.csv"),
                      members=2)
    assert plan.mode == "member_local"
    assert any("windfarm_opt=1" in reason
               for reason in plan.native_fallback_reasons)


# ---------------------------------------------------------------------------
# Turbine files
# ---------------------------------------------------------------------------

def test_a_woof_table_and_its_wrf_inputs_place_the_same_turbines(tmp_path):
    """CSV -> layout -> write_wrf_inputs -> WRF list: the same groups and
    the same table words, so WRF and woof run the same farm."""
    grid = _grid()
    path = _write_csv(tmp_path / "farm.csv", _farm())
    built = wl.build_layout(wl.read_turbines(path), grid)
    from_table = wc.fitch_layout_for_file(path, shape=(41, 41), grid=grid)
    _assert_same_layout(from_table, wc.fitch_layout_from_layout(built))
    assert int(from_table.count.sum()) == 9
    wrf_dir = built.write_wrf_inputs(tmp_path / "wrf")
    assert wc.turbine_file_kind(wrf_dir / "windturbines.txt") == "wrf"
    from_wrf = wc.fitch_layout_for_file(wrf_dir / "windturbines.txt",
                                        shape=(41, 41), grid=grid)
    _assert_same_layout(from_wrf, from_table)
    # windfarm_ij = 1: the same cells as 1-based i j rows.
    rows = [f"{int(i) + 1} {int(j) + 1} 1" for i, j in
            zip(built.turbine_i, built.turbine_j)]
    (wrf_dir / "windturbines-ij.txt").write_text("\n".join(rows) + "\n")
    from_ij = wc.fitch_layout_for_file(wrf_dir / "windturbines-ij.txt",
                                       ij=1, shape=(41, 41), grid=grid)
    _assert_same_layout(from_ij, from_table)


def test_a_cell_listing_a_b_a_is_three_groups(tmp_path):
    table = "3\n90. 100. 0.13 2.0\n3. 0.8 10.\n10. 0.7 1500.\n25. 0.2 2000.\n"
    (tmp_path / "wind-turbine-1.tbl").write_text(table)
    (tmp_path / "wind-turbine-2.tbl").write_text(
        table.replace("90. 100.", "120. 140."))
    (tmp_path / "windturbines-ij.txt").write_text(
        "5 6 1\n5 6 1\n5 6 2\n5 6 1\n9 9 2\n")
    layout, tables = wc.wrf_fitch_layout(tmp_path / "windturbines-ij.txt",
                                         ij=1, shape=(20, 20))
    cell = (6 - 1) * 20 + (5 - 1)
    groups = list(layout.groups_in(5, 4))
    assert [int(layout.count[g]) for g in groups] == [2, 1, 1]
    assert [int(layout.curve[g]) for g in groups] == [0, 1, 0]
    assert [float(layout.hub_height_m[g]) for g in groups] == [90., 120.,
                                                               90.]
    assert int(layout.cell[groups[0]]) == cell
    assert [p.name for p in tables] == ["wind-turbine-1.tbl",
                                        "wind-turbine-2.tbl"]


@pytest.mark.parametrize("listing, tables, needle", [
    ("5 6 1\n\n5 6 1\n", True, "blank line"),
    ("5 6 101\n", True, "outside 1..100"),
    ("5 6 3\n", True, "wind-turbine-3.tbl does not exist"),
    ("1 6 1\n", True, "outermost row or column"),
    ("25 6 1\n", True, "outside this 20 x 20 domain"),
    ("5 6\n", True, "three values"),
    ("5 6 1.5\n", True, "not an integer"),
])
def test_wrf_list_refusals_are_named(tmp_path, listing, tables, needle):
    (tmp_path / "wind-turbine-1.tbl").write_text(
        "2\n90. 100. 0.13 2.0\n3. 0.8 10.\n25. 0.2 2000.\n")
    (tmp_path / "windturbines-ij.txt").write_text(listing)
    with pytest.raises(wc.WindFarmRefused, match=needle):
        wc.wrf_fitch_layout(tmp_path / "windturbines-ij.txt", ij=1,
                            shape=(20, 20))


@pytest.mark.parametrize("table, needle", [
    ("1\n90. 100. 0.13 2.0\n3. 0.8 10.\n", "nval = 1"),
    ("2\n40. 100. 0.13 2.0\n3. 0.8 10.\n25. 0.2 2000.\n", "below the ground"),
    ("2\n90. 100. 0.13 2.0\n25. 0.8 10.\n3. 0.2 2000.\n", "ascend"),
    ("2\n90. 100. 0.13 2.0\n3. 0.8 10.\n", "ends before row 2"),
])
def test_wrf_table_refusals_are_named(tmp_path, table, needle):
    (tmp_path / "wind-turbine-1.tbl").write_text(table)
    with pytest.raises(wc.WindFarmRefused, match=needle):
        wc.read_wrf_table(tmp_path / "wind-turbine-1.tbl", 1)


def test_a_table_read_continues_across_records_as_fortran_does(tmp_path):
    # List-directed: the header may span records, and the rest of a
    # READ's last record is skipped.
    (tmp_path / "t.tbl").write_text(
        "2 ignored\n90.\n100. 0.13\n2.0 extra\n3.,0.8,10.\n25 0.2 2d3\n")
    t = wc.read_wrf_table(tmp_path / "t.tbl", 1)
    assert (t.hub_height_m, t.diameter_m, t.standing_thrust) == (
        np.float32(90.), np.float32(100.), np.float32(0.13))
    assert t.power_kw.tolist() == [10.0, 2000.0]


def test_what_fortran_skips_is_never_read(tmp_path):
    # A trailing note after the values a READ consumes is skipped by
    # Fortran, even with a slash or star in it; so it is here.
    (tmp_path / "wind-turbine-1.tbl").write_text(
        "2  ! rows\n90. 100. 0.13 2.0  ! hub(m) D(m) stc P(MW)\n"
        "3. 0.8 10.  m/s, -, kW\n25. 0.2 2000. 2*0\n")
    (tmp_path / "windturbines-ij.txt").write_text(
        "5 6 1 ! first\n7 6 1 farm/a\n")
    assert wc.turbine_file_kind(tmp_path / "windturbines-ij.txt") == "wrf"
    layout, _ = wc.wrf_fitch_layout(tmp_path / "windturbines-ij.txt", ij=1,
                                    shape=(20, 20))
    assert int(layout.count.sum()) == 2
    assert float(layout.power_kw[0, 1]) == 2000.0


@pytest.mark.parametrize("token, needle", [
    ("2*0.8", "repeat count"), ("0.8/", "slash"), ("1_0", "digit"),
    ("1e400", "overflows"), ("abc", "not a number"),
])
def test_consumed_tokens_wrf_would_not_read_are_refused(tmp_path, token,
                                                        needle):
    (tmp_path / "t.tbl").write_text(
        f"2\n90. 100. 0.13 2.0\n3. {token} 10.\n25. 0.2 2000.\n")
    with pytest.raises(wc.WindFarmRefused, match=needle):
        wc.read_wrf_table(tmp_path / "t.tbl", 1)


def test_the_curves_digest_reads_the_tables_the_reader_reads(tmp_path):
    table = "2\n90. 100. 0.13 2.0\n3. 0.8 10.\n25. 0.2 2000.\n"
    (tmp_path / "wind-turbine-1.tbl").write_text(table)
    (tmp_path / "wind-turbine-7.tbl").write_text(table)
    (tmp_path / "windturbines-ij.txt").write_text("5 6 1 7\n")
    assert wc.wrf_table_paths(tmp_path / "windturbines-ij.txt") == {
        1: tmp_path / "wind-turbine-1.tbl"}
    first = wc.windfarm_input_digests(tmp_path / "windturbines-ij.txt")
    (tmp_path / "wind-turbine-7.tbl").write_text(table + "\n")
    assert wc.windfarm_input_digests(tmp_path / "windturbines-ij.txt") \
        == first
    (tmp_path / "wind-turbine-1.tbl").write_text(table.replace("0.2", "0.3"))
    assert wc.windfarm_input_digests(
        tmp_path / "windturbines-ij.txt")[1] != first[1]


def test_a_checkpoint_names_the_bytes_the_model_was_built_from(tmp_path):
    from woof.io.restart import configuration_echo

    path = _write_csv(tmp_path / "farm.csv", _farm())
    cfg = _cfg(windfarm_opt=1, windfarm_turbine_file=str(path))
    # What build_windfarm records as it uploads the layout (a GPU step;
    # tests/test_windfarm_coupling_gpu.py builds it for real).
    wc._BUILT[str(path.resolve())] = wc.windfarm_input_digests(path)
    at_build = configuration_echo(cfg)[wc.TURBINE_FILE_IDENTITY_KEY]
    rows = _farm()
    rows[0]["hub_height_m"] = 95.0
    _write_csv(path, rows)
    assert configuration_echo(cfg)[wc.TURBINE_FILE_IDENTITY_KEY] == at_build
    del wc._BUILT[str(path.resolve())]
    assert configuration_echo(cfg)[wc.TURBINE_FILE_IDENTITY_KEY] != at_build


def test_a_streamed_domain_is_refused_before_preparation():
    from woof.core.streaming import StreamingOptions
    from woof.experiment import _refuse_streamed_windfarm

    on = SimpleNamespace(windfarm_opt=1)
    off = SimpleNamespace(windfarm_opt=0)
    tiles = StreamingOptions(mode="on")
    with pytest.raises(ValueError, match="streams its columns"):
        _refuse_streamed_windfarm(tiles, [SimpleNamespace(
            grid_id=1, run=on, tiles=None)])
    _refuse_streamed_windfarm(tiles, [SimpleNamespace(grid_id=1, run=off,
                                                      tiles=None)])
    _refuse_streamed_windfarm(StreamingOptions(mode="auto"), [
        SimpleNamespace(grid_id=1, run=on, tiles=None)])
    _refuse_streamed_windfarm(StreamingOptions(), [SimpleNamespace(
        grid_id=1, run=on, tiles=None)])


def test_a_real_is_read_correctly_rounded_not_double_rounded():
    # 1 + 2**-24 is the midpoint between 1 and the next float32; a decimal
    # a hair above it rounds to binary64 exactly ON the midpoint, and then
    # to float32 1.0 by ties-to-even.  strtof (gfortran's read) rounds the
    # decimal itself, up.
    getcontext().prec = 80
    value = Fraction(1) + Fraction(1, 2 ** 24) + Fraction(1, 2 ** 70)
    text = str(Decimal(value.numerator) / Decimal(value.denominator))
    assert np.float32(float(text)) == np.float32(1.0)
    assert wc._f32_decimal(text, "t") == np.nextafter(np.float32(1.0),
                                                      np.float32(2.0))
    for text in ("0.1", "3.45", "117.", "1e-3", "2.5D2"):
        assert wc._f32_decimal(text, "t") == np.float32(
            float(text.replace("D", "e")))


def test_a_woof_csv_with_a_txt_suffix_is_still_a_table(tmp_path):
    path = _write_csv(tmp_path / "farm.txt", _farm())
    assert wc.turbine_file_kind(path) == "table"
    assert wc.turbine_file_kind(tmp_path / "farm.geojson") == "table"


def test_layout_refusals_reach_the_seam_by_name(tmp_path):
    rows = _farm()
    rows[0] = {**rows[0], "lat": 50.0}
    path = _write_csv(tmp_path / "farm.csv", rows)
    with pytest.raises(wc.WindFarmRefused, match="outside d01"):
        wc.fitch_layout_for_file(path, shape=(41, 41), grid=_grid())
    with pytest.raises(wc.WindFarmRefused, match="does not exist"):
        wc.fitch_layout_for_file(tmp_path / "nope.csv", shape=(41, 41),
                                 grid=_grid())


# ---------------------------------------------------------------------------
# Identity: off is invisible, on binds the bytes
# ---------------------------------------------------------------------------

def test_off_leaves_the_checkpoint_echo_and_digest_alone(tmp_path):
    from woof.io.restart import (_configuration_digest_values,
                                  configuration_echo)

    plain = configuration_echo(_cfg())
    # A shared file with the switch off on this domain is inert too.
    shared = configuration_echo(_cfg(windfarm_turbine_file="x.csv",
                                     windfarm_tke_factor=1.0))
    assert plain == shared
    for name in (*WINDFARM_RUN_DEFAULTS, wc.TURBINE_FILE_IDENTITY_KEY,
                 wc.CURVES_IDENTITY_KEY):
        assert name not in plain
    digest = _configuration_digest_values(dataclasses.asdict(_cfg()))
    assert not set(WINDFARM_RUN_DEFAULTS) & set(digest)


def test_on_binds_the_file_bytes_and_a_saved_digest_stays(tmp_path):
    from woof.io.restart import configuration_echo

    path = _write_csv(tmp_path / "farm.csv", _farm())
    cfg = _cfg(windfarm_opt=1, windfarm_turbine_file=str(path))
    echo = configuration_echo(cfg)
    assert echo["windfarm_opt"] == 1
    assert echo[wc.TURBINE_FILE_IDENTITY_KEY] == wc._sha256(path)
    from woof.energy.turbine_library import load_turbine_library
    assert echo[wc.CURVES_IDENTITY_KEY] == load_turbine_library().sha256
    stored = dict(echo)
    rows = _farm()
    rows[0]["hub_height_m"] = 95.0
    _write_csv(path, rows)
    assert configuration_echo(cfg)[wc.TURBINE_FILE_IDENTITY_KEY] != \
        stored[wc.TURBINE_FILE_IDENTITY_KEY]
    assert wc.bind_windfarm_identity(dict(stored)) == stored


def test_wrf_tables_bind_beside_the_list(tmp_path):
    (tmp_path / "wind-turbine-1.tbl").write_text(
        "2\n90. 100. 0.13 2.0\n3. 0.8 10.\n25. 0.2 2000.\n")
    (tmp_path / "windturbines-ij.txt").write_text("5 6 1\n")
    first = wc.windfarm_input_digests(tmp_path / "windturbines-ij.txt")
    (tmp_path / "wind-turbine-1.tbl").write_text(
        "2\n90. 100. 0.13 2.0\n3. 0.8 10.\n25. 0.25 2000.\n")
    second = wc.windfarm_input_digests(tmp_path / "windturbines-ij.txt")
    assert first[0] == second[0] and first[1] != second[1]


def test_restart_identity_payload_drops_off_and_binds_on(tmp_path):
    from datetime import datetime, timezone

    from woof.core.model import restart_identity_payload
    from woof.experiment import experiment_from_run_config

    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    off = restart_identity_payload(experiment_from_run_config(_cfg(), start))
    for domain in off["domains"]:
        assert not set(WINDFARM_RUN_DEFAULTS) & set(domain["run"])
    path = _write_csv(tmp_path / "farm.csv", _farm())
    on = restart_identity_payload(experiment_from_run_config(
        _cfg(windfarm_opt=1, windfarm_turbine_file=str(path)), start))
    run = on["domains"][0]["run"]
    assert run["windfarm_opt"] == 1
    assert run[wc.TURBINE_FILE_IDENTITY_KEY] == wc._sha256(path)


def test_a_resume_refuses_a_flipped_farm_and_tolerates_absent_keys():
    from woof.io.restart import _require_config_match, configuration_echo

    live = _cfg()
    stored = configuration_echo(live)
    _require_config_match(stored, live, "ck.npz")
    on = dataclasses.replace(live, windfarm_opt=1,
                             windfarm_turbine_file="missing.csv")
    with pytest.raises(Exception, match="windfarm"):
        _require_config_match(stored, on, "ck.npz")


def test_the_driver_carrier_is_classified_as_rebuilt():
    from woof.io.restart import DRIVER_REBUILT_ATTRS

    assert "windfarm" in DRIVER_REBUILT_ATTRS


# ---------------------------------------------------------------------------
# Pricing, history and the registry
# ---------------------------------------------------------------------------

def test_off_prices_and_writes_nothing():
    from woof.core.preflight import physics_array_shapes
    from woof.io.history_layout import produced_history_shapes

    assert wc.windfarm_array_shapes(_cfg()) == {}
    assert not any(k.startswith("windfarm/") or k == "fields/power"
                   for k in physics_array_shapes(_cfg()))
    assert "POWER" not in produced_history_shapes(_cfg())


def test_on_prices_the_layout_and_writes_power(tmp_path):
    from datetime import datetime, timezone

    from woof.core import preflight as pf
    from woof.experiment import experiment_from_run_config
    from woof.io.history_layout import produced_history_shapes

    path = _write_csv(tmp_path / "farm.csv", _farm())
    cfg = _cfg(windfarm_opt=1, windfarm_turbine_file=str(path))
    shapes = wc.windfarm_array_shapes(cfg)
    assert shapes == {"fields/power": (41, 41),
                      "windfarm/cell_offsets": (41 * 41 + 1,),
                      "windfarm/groups": (5, 9),
                      "windfarm/curves": (1, 302)}
    assert produced_history_shapes(cfg)["POWER"] == (41, 41)
    exp = experiment_from_run_config(cfg, datetime(2026, 1, 1,
                                                  tzinfo=timezone.utc))
    dc = exp.domains[0]
    items = {item.name: item for item in pf.estimate_domain(dc).items}
    for name, shape in shapes.items():
        assert items[name].category == "physics"
        assert items[name].shape == shape
    assert "windfarm_fitch_composed" in pf.domain_kernel_modules(
        dc, prices_refl=False)
    # The bound when the file cannot be read: one group per column.
    bound = wc.windfarm_array_shapes(dataclasses.replace(
        cfg, windfarm_turbine_file=str(tmp_path / "gone.csv")))
    assert bound["windfarm/groups"] == (5, 41 * 41)


def test_power_is_wrfs_registry_row_and_in_the_windfarm_preset():
    from woof.io.history_selection import HISTORY_PRESETS
    from woof.io.wrf_output_schema import REGISTRY_VAR_META

    assert REGISTRY_VAR_META["POWER"] == ("Power production", "W")
    assert "POWER" in HISTORY_PRESETS["windfarm"]


def test_the_registry_lists_the_four_keys_and_the_mynn_rule():
    import json

    registry = json.loads((REPO / "woof" / "physics_registry_v2.json")
                          .read_text(encoding="utf-8"))
    rows = registry["parameters"]
    assert rows["windfarm_opt"]["enum"] == [0, 1]
    assert rows["windfarm_ij"]["enum"] == [0, 1]
    assert rows["windfarm_tke_factor"]["default"] == 0.25
    assert rows["windfarm_turbine_file"]["default"] == ""
    text = json.dumps(registry)
    assert "Set windfarm_opt=0." in text


# ---------------------------------------------------------------------------
# The namelist importer
# ---------------------------------------------------------------------------

def _fixture_texts():
    """test_namelist_import's two namelists, read without importing it (its
    module imports a case bundle this tree does not ship)."""
    tree = ast.parse((REPO / "tests" / "test_namelist_import.py").read_text(
        encoding="utf-8"))
    found = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and \
                getattr(node.targets[0], "id", None) in ("WPS_TEXT",
                                                         "INPUT_TEXT"):
            found[node.targets[0].id] = ast.literal_eval(node.value)
    return found["WPS_TEXT"], found["INPUT_TEXT"]


def _import(tmp_path, physics=""):
    from woof.namelist_import import import_namelists

    wps, inp = _fixture_texts()
    inp = inp.replace(" sf_sfclay_physics = 91, 91,\n",
                      " sf_sfclay_physics = 5, 5,\n")
    inp = inp.replace(" bl_pbl_physics = 11, 11,\n",
                      " bl_pbl_physics = 5, 5,\n")
    inp = inp.replace("&physics\n", "&physics\n" + physics)
    (tmp_path / "namelist.wps").write_text(wps)
    (tmp_path / "namelist.input").write_text(inp)
    return import_namelists(tmp_path / "namelist.wps",
                            tmp_path / "namelist.input")


def test_windfarm_keys_translate(tmp_path):
    text, _ = _import(tmp_path, " windfarm_opt = 1, 0,\n windfarm_ij = 1,\n"
                                " windfarm_tke_factor = 1.0,\n")
    raw = tomllib.loads(text)
    shared = raw["shared"]
    assert shared["windfarm_opt"] == 1
    assert shared["windfarm_ij"] == 1
    assert shared["windfarm_tke_factor"] == 1.0
    assert shared["windfarm_turbine_file"] == str(
        (tmp_path / "windturbines-ij.txt").resolve())
    assert raw["domain"][1]["windfarm_opt"] == 0


def test_lat_lon_list_is_the_default_and_omitted_keys_change_nothing(
        tmp_path):
    plain, _ = _import(tmp_path)
    zero, _ = _import(tmp_path, " windfarm_opt = 0, 0,\n")
    assert zero == plain and "windfarm" not in plain
    text, _ = _import(tmp_path, " windfarm_opt = 1, 1,\n")
    shared = tomllib.loads(text)["shared"]
    assert shared["windfarm_turbine_file"].endswith("windturbines.txt")
    assert "windfarm_ij" not in shared and "windfarm_tke_factor" not in shared


@pytest.mark.parametrize("physics, needle", [
    (" windfarm_opt = 2, 0,\n", "MAV"),
    (" windfarm_opt = 1, 0,\n windfarm_ij = 3,\n", "windfarm_ij"),
    (" windfarm_opt = 1, 0,\n windfarm_tke_factor = -1.0,\n", ">= 0"),
    (" windfarm_opt = 1, 0,\n windfarm_wake_model = 2,\n",
     "windfarm_wake_model"),
])
def test_unported_windfarm_values_and_keys_are_refused(tmp_path, physics,
                                                       needle):
    with pytest.raises(ValueError, match=needle):
        _import(tmp_path, physics)


def test_the_farm_under_another_pbl_is_refused_at_import(tmp_path):
    from woof.namelist_import import import_namelists

    wps, inp = _fixture_texts()
    inp = inp.replace("&physics\n", "&physics\n windfarm_opt = 1, 0,\n")
    (tmp_path / "namelist.wps").write_text(wps)
    (tmp_path / "namelist.input").write_text(inp)
    with pytest.raises(ValueError, match="bl_pbl_physics = 5"):
        import_namelists(tmp_path / "namelist.wps",
                         tmp_path / "namelist.input")
