"""``woof hex ice run``: MALI's forward mode from a namelist and an MPAS input file.

A forward-mode oracle case written out as an MPAS land-ice input file and a
namelist runs through the door to the v8.4.2 oracle's words
(tests/data/oracles/mali_forward), and a run stopped and continued from its
restart ends on the straight run's words.
"""
from __future__ import annotations

import numpy as np
import pytest

from conftest import requires_gpu
from woof.verify import mali_forward_oracle as fo


def _case(name: str, precision: str = "r4") -> dict:
    return next(c for c in fo.cases(precision) if c["name"] == name)


def _value(v) -> str:
    if isinstance(v, bool):
        return ".true." if v else ".false."
    if isinstance(v, str):
        return "'" + v + "'"
    return repr(v)


def write_namelist(path, config: dict) -> None:
    lines = ["&landice"]
    lines += [f"    {k} = {_value(v)}" for k, v in config.items()
              if k not in ("on_a_sphere", "sphere_radius") and not k.startswith("stream:")]
    path.write_text("\n".join(lines + ["/", ""]), encoding="utf-8")


def write_input(path, case: dict, skip: tuple[str, ...] = ()) -> None:
    """The case as an MPAS land-ice input file: one Time record, Registry names."""
    import netCDF4
    dims = dict(case["dims"])
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("Time", None)
        ds.createDimension("StrLen", 64)
        for name in ("nCells", "nEdges", "nVertices", "nVertLevels", "nVertInterfaces", "maxEdges", "maxEdges2",
                     "vertexDegree", "nRegions"):
            if name in dims:
                ds.createDimension(name, dims[name])
        ds.createDimension("TWO", 2)
        ds.createDimension("R3", 3)
        ds.setncattr("on_a_sphere", "YES" if case["config"].get("on_a_sphere") else "NO")
        ds.setncattr("sphere_radius", float(case["config"].get("sphere_radius", 0.0)))
        for pool, name, kind, shape in fo.fields(case):
            if kind == "d" or pool in ("scratch", "globalStatsAM", "regionalStatsAM") or "@" in name or name in skip:
                continue
            value = case["pools"][pool][name]
            if kind == "c":
                var = ds.createVariable(name, "S1", ("Time", "StrLen"))
                var[0] = np.frombuffer(value.ljust(64)[:64].encode("ascii"), "S1")
                continue
            axes = tuple(s if isinstance(s, str) else {2: "TWO", 3: "R3"}[s] for s in shape)
            if pool == "mesh" and name in fo.MESH_INPUTS:
                var = ds.createVariable(name, np.asarray(value).dtype, axes)
                var[...] = value
            else:
                var = ds.createVariable(name, np.asarray(value).dtype, ("Time",) + axes)
                var[0] = value


def _outputs(path) -> dict[str, np.ndarray]:
    import netCDF4
    out = {}
    with netCDF4.Dataset(path) as ds:
        ds.set_auto_mask(False)
        for name, var in ds.variables.items():
            if var.dtype == np.dtype("S1"):
                out[name] = b"".join(np.asarray(var[0]).tolist()).decode("ascii").rstrip()
            else:
                out[name] = np.asarray(var[0])
        out["clock/now"] = ds.getncattr("clock_now")
    return out


def _run(tmp_path, case: dict, *extra: str, skip=()) -> int:
    from woof.hex.cli import main
    write_namelist(tmp_path / "namelist.landice", case["config"])
    write_input(tmp_path / "input.nc", case, skip)
    return main(["ice", "run", "--namelist", str(tmp_path / "namelist.landice"), "--input",
                 str(tmp_path / "input.nc"), "--out", str(tmp_path / "out"), "--precision", case["precision"],
                 *extra])


@requires_gpu
@pytest.mark.parametrize("name, precision", [
    ("dome_uniform", "r4"), ("hydro_radial_uniform", "r4"), ("eismint2_a_temperature_varres", "r8"),
    ("dome_adaptive_varres", "r8"), ("simple_radial_shelf_varres", "r4"), ("simple_rim_uniform", "r8"),
    ("stats_dome_varres", "r4"), ("stats_simple_shelf_uniform", "r8"), ("am_marine_varres", "r4"),
    ("am_marine_uniform", "r8"),
])
def test_the_door_runs_to_the_oracles_words(tmp_path, name, precision):
    case = _case(name, precision)
    assert _run(tmp_path, case) == 0
    got = _outputs(tmp_path / "out" / "output.nc")
    pinned = fo.load_fixture(precision)[case["name"]]
    assert got["clock/now"] == str(pinned["clock/now"])
    compared = 0
    for pool, field in fo.outputs(case):
        key = f"{pool}/{field}"
        if key not in pinned or "@" in field or pool in fo.AM_POOLS:
            continue   # a second time level lives in restart.npz only; the members' words in their files
        if pinned[key].dtype.kind in "SU":
            assert got[field] == str(pinned[key]).rstrip(), field
        else:
            assert fo.word_mismatches(got[field].reshape(pinned[key].shape), pinned[key]) == 0, field
        compared += 1
    assert compared > 100
    # the statistics' reports, as the model logs them
    log = tmp_path / "out" / "statistics.log"
    assert log.exists() == fo.stats_on(case["config"])
    if log.exists():
        assert fo.stats_log(log.read_text().splitlines()) == str(pinned["stats/log"])
    # the analysis members: each compute a record of its stream's file
    if fo.am_on(case["config"]):
        import netCDF4
        from woof.hex.mpas_ice import analysis_v842 as am
        words, stamps = [], []
        files = {m: netCDF4.Dataset(tmp_path / "out" / f"{m}Output.nc") for m in am.enabled(case["config"])}
        try:
            counts = dict.fromkeys(files, 0)
            for code, stamp in zip(pinned["am/member"], pinned["am/xtime"]):
                member = am.MEMBERS[int(code) - 1]
                ds, t = files[member], counts[member]
                counts[member] += 1
                stamps.append(b"".join(np.asarray(ds["xtime"][t]).tolist()).decode("ascii").rstrip())
                words.append(np.concatenate([np.asarray(ds[n][t]).ravel() for n in am.FIELDS[member]]))
            assert all(len(ds["xtime"]) == counts[m] for m, ds in files.items())
        finally:
            for ds in files.values():
                ds.close()
        assert stamps == [str(x) for x in pinned["am/xtime"]]
        assert fo.word_mismatches(np.concatenate(words), pinned["am/words"]) == 0


@requires_gpu
def test_a_run_continued_from_its_restart_ends_on_the_straight_runs_words(tmp_path):
    from woof.hex.cli import main
    case = _case("dome_adaptive_uniform", "r8")
    (tmp_path / "straight").mkdir()
    assert _run(tmp_path / "straight", case) == 0
    (tmp_path / "split").mkdir()
    assert _run(tmp_path / "split", case, "--steps", "2") == 0
    first = tmp_path / "split" / "out"
    assert main(["ice", "run", "--namelist", str(tmp_path / "split" / "namelist.landice"), "--restart",
                 str(first / "restart.npz"), "--out", str(tmp_path / "split" / "rest")]) == 0
    straight = _outputs(tmp_path / "straight" / "out" / "output.nc")
    resumed = _outputs(tmp_path / "split" / "rest" / "output.nc")
    assert straight.keys() == resumed.keys()
    for key, value in straight.items():
        if isinstance(value, str):
            assert resumed[key] == value, key
        else:
            assert fo.word_mismatches(resumed[key], value) == 0, key


@requires_gpu
def test_the_door_refuses_by_name(tmp_path, capsys):
    from woof.hex.cli import main
    case = _case("dome_uniform")
    for folder, variant, skip, words in (
            ("a", case, ("coeffs_reconstruct",), "coeffs_reconstruct"),
            ("b", dict(case, config=dict(case["config"], config_not_an_option=1)), (), "config_not_an_option"),
            ("c", dict(case, config=dict(case["config"], config_velocity_solver="FO")), (), "config_velocity_solver"),
            ("e", _case("simple_uniform_uniform"), ("xEdge",), "xEdge")):
        (tmp_path / folder).mkdir()
        capsys.readouterr()
        assert _run(tmp_path / folder, variant, skip=skip) == 2
        assert words in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(["ice", "run", "--namelist", str(tmp_path / "a" / "namelist.landice"), "--out", str(tmp_path / "d")])


def test_the_registry_defaults_cover_every_option_the_forward_mode_reads():
    from woof.hex.mpas_ice import run_v842
    defaults = run_v842.default_config()
    missing = [k for k in fo.BASE_CONFIG if k not in defaults and k not in ("on_a_sphere", "sphere_radius")]
    assert not missing
    assert defaults["config_velocity_solver"] == "sia" and defaults["config_SGH"] is False
    assert defaults["config_dt"] == "0001-00-00_00:00:00"


@requires_gpu
def test_the_statistics_log_is_this_runs_alone(tmp_path):
    # statistics on but none in the run: an empty log; statistics off: an
    # earlier run's log in the same out directory is removed
    case = _case("stats_dome_uniform", "r4")
    quiet = dict(case, config=dict(case["config"], config_stats_interval=100, config_write_stats_on_startup=False))
    (tmp_path / "out").mkdir()
    log = tmp_path / "out" / "statistics.log"
    log.write_text("a stale report\n")
    assert _run(tmp_path, quiet) == 0
    assert log.read_text() == ""
    log.write_text("a stale report\n")
    off = dict(case, config=dict(case["config"], config_stats_interval=0, config_write_stats_on_startup=False))
    assert _run(tmp_path, off) == 0
    assert not log.exists()


@requires_gpu
def test_an_analysis_member_that_did_not_compute_leaves_no_file(tmp_path):
    # globalStats at 'output_interval' with no startup compute never computes:
    # an earlier run's globalStatsOutput.nc in the same out directory is removed
    case = _case("am_marine_uniform", "r4")
    never = dict(case, config=dict(case["config"], config_AM_globalStats_compute_on_startup=False))
    (tmp_path / "out").mkdir()
    stale = tmp_path / "out" / "globalStatsOutput.nc"
    stale.write_text("a stale file\n")
    assert _run(tmp_path, never) == 0
    assert not stale.exists()
    assert (tmp_path / "out" / "regionalStatsOutput.nc").exists()
