"""The live Icechunk export ``woof go`` runs beside a forecast
(``woof/icechunk_export.py``) and its resume door, ``woof icechunk-export``.

A fake forecast publishes tiny wrfout frames the way the history writer
does -- a hidden temporary renamed onto its name, then a ``ready/`` marker
-- while the exporter follows it.  Every store is ``file://`` under
``tmp_path`` or ``memory://``; nothing touches the network.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

ic = pytest.importorskip("icechunk")
zarr = pytest.importorskip("zarr")
netCDF4 = pytest.importorskip("netCDF4")

from woof import icechunk_export as ie  # noqa: E402
from woof.energy import icechunk_store as st  # noqa: E402

T0 = datetime(2025, 1, 1, tzinfo=timezone.utc)
NZ, NY, NX = 2, 3, 4


# --------------------------------------------------------------------------
# a fake forecast


def _write_frame(path: Path, valid: datetime, value: float, *,
                 with_refl: bool, shape=(NZ, NY, NX)) -> None:
    # Under the process-wide HDF5 lock, as the history writer writes:
    # the exporter may be reading in another thread of this process.
    from woof.io.netcdf_serialization import netcdf4_session

    with netcdf4_session():
        _write_frame_unlocked(path, valid, value, with_refl=with_refl,
                              shape=shape)


def _write_frame_unlocked(path: Path, valid: datetime, value: float, *,
                          with_refl: bool, shape) -> None:
    nz, ny, nx = shape
    with netCDF4.Dataset(path, "w", format="NETCDF4") as nc:
        nc.createDimension("Time", None)
        nc.createDimension("DateStrLen", 19)
        nc.createDimension("bottom_top", nz)
        nc.createDimension("south_north", ny)
        nc.createDimension("west_east", nx)
        nc.DX = 100.0
        times = nc.createVariable("Times", "S1", ("Time", "DateStrLen"))
        times.set_auto_chartostring(False)
        stamp = valid.strftime("%Y-%m-%d_%H:%M:%S")
        times[0] = np.frombuffer(stamp.encode("ascii"), dtype="S1")
        nc.createVariable("XTIME", "f4", ("Time",))[0] = 0.0
        u = nc.createVariable("U", "f4", ("Time", "bottom_top",
                                          "south_north", "west_east"))
        u.units = "m s-1"
        u[0] = np.full((nz, ny, nx), value, np.float32)
        nc.createVariable("T2", "f4", ("Time", "south_north",
                                       "west_east"))[0] = \
            np.full((ny, nx), value, np.float32)
        nc.createVariable("ITIMESTEP", "i4", ("Time",))[0] = int(value)
        nc.createVariable("HGT", "f4", ("south_north", "west_east"))[:] = \
            np.arange(ny * nx, dtype=np.float32).reshape(ny, nx)
        if with_refl:
            nc.createVariable("REFL_10CM", "f4", (
                "Time", "bottom_top", "south_north", "west_east"))[0] = \
                np.full((nz, ny, nx), -value, np.float32)


def publish(run: Path, domain: int, index: int, *, step_s: int = 900,
            offset: float = 0.0, marker: bool = True, shape=(NZ, NY, NX),
            refl_from: int = 1) -> Path:
    """Publish frame ``index`` of ``domain`` as the history writer does:
    temporary, rename, then the ready marker.  ``U`` is ``index +
    offset``; ``REFL_10CM`` is first written at ``refl_from``."""

    from woof.progress_log import write_frame_marker

    valid = T0 + timedelta(seconds=index * step_s)
    folder = run / "wrfout"
    folder.mkdir(parents=True, exist_ok=True)
    name = f"wrfout_d{domain:02d}_{valid:%Y-%m-%d_%H_%M_%S}"
    temporary = folder / f".{name}.tmp.{os.getpid()}"
    _write_frame(temporary, valid, float(index) + offset,
                 with_refl=index >= refl_from, shape=shape)
    final = folder / name
    os.replace(temporary, final)
    if marker:
        write_frame_marker(run / "ready", domain=domain,
                           valid_time=valid.replace(tzinfo=None), path=final)
    return final


def _options(url: str, **extra) -> ie.ExportOptions:
    return ie.ExportOptions.from_mapping({"url": url, **extra},
                                         allow_memory=True)


def _file_url(path: Path) -> str:
    return "file://" + str(path)


def _cli(paths, **flags) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    ie.add_arguments(parser)
    argv = [str(p) for p in paths]
    for key, value in flags.items():
        option = "--" + key.replace("_", "-")
        if value is True:
            argv.append(option)
        else:
            argv += [option, str(value)]
    return parser.parse_args(argv)


def _snapshot(group) -> dict:
    """Every array of a group (values, dtype, dims, attrs) and its attrs,
    less the creation instant."""

    out = {"attrs": {k: v for k, v in dict(group.attrs).items()
                     if k != "created_utc"}}
    for name, array in group.arrays():
        out[name] = (np.asarray(array[...]), str(array.dtype),
                     tuple(array.metadata.dimension_names or ()),
                     dict(array.attrs))
    return out


def _open_group(url: str, group: str):
    repo = st.open_repository(url, create=False)
    return zarr.open_group(repo.readonly_session("main").store, path=group,
                           mode="r")


def assert_same_store(a_url: str, b_url: str, groups, *,
                      ignore_attrs=()) -> None:
    for group in groups:
        a, b = _snapshot(_open_group(a_url, group)), \
            _snapshot(_open_group(b_url, group))
        for key in ignore_attrs:
            a["attrs"].pop(key, None)
            b["attrs"].pop(key, None)
        assert a["attrs"] == b["attrs"], group
        assert set(a) == set(b), (group, set(a) ^ set(b))
        for name in a:
            if name == "attrs":
                continue
            (va, da, ma, aa), (vb, db, mb, ab) = a[name], b[name]
            assert (da, ma, aa) == (db, mb, ab), (group, name)
            np.testing.assert_array_equal(va, vb, err_msg=f"{group}/{name}")


def _drain(paths, url: str, tmp_path: Path, **flags) -> dict:
    state = tmp_path / f"state-{uuid.uuid4().hex[:8]}"
    args = _cli(paths, url=url, state_dir=state, json=True, **flags)
    code = ie.main(args)
    status = json.loads((state / ie.STATUS_NAME).read_text())
    status["exit_code"] = code
    return status


# --------------------------------------------------------------------------
# options


def test_options_from_a_url_or_a_table(tmp_path):
    options = ie.ExportOptions.from_mapping("s3://bucket/uk")
    assert options.url == "s3://bucket/uk" and options.compress
    assert options.zstd_clevel == 5 and options.group(2) == "d02"
    table = ie.ExportOptions.from_mapping(
        {"url": "store", "zstd_clevel": 3, "group_prefix": "uk/",
         "threads": 2}, base_dir=tmp_path)
    assert table.url == str((tmp_path / "store").resolve())
    assert table.group(1) == "uk/d01" and table.threads == 2
    assert ie.ExportOptions.from_mapping(None) is None


@pytest.mark.parametrize("value, match", [
    ({"url": "ftp://x/y"}, "not one of"),
    ({"url": "s3://key:secret@bucket/x"}, "credentials"),
    ({"url": "memory://x"}, "in-process"),
    ({"url": "s3://b/x", "zstd": 3}, "unknown key"),
    ({"url": "s3://b/x", "zstd_clevel": 12}, "1-9"),
    ({"url": "s3://b/x", "compress": "yes"}, "true or false"),
    ({"branch": "main"}, "needs url"),
    (["s3://b/x"], "URL or a table"),
])
def test_options_refusals_are_named(value, match):
    with pytest.raises(ie.IcechunkExportRefused, match=match):
        ie.ExportOptions.from_mapping(value)


def test_the_output_table_takes_icechunk_and_a_domain_table_refuses_it():
    from woof.io.history_selection import HistorySelection

    HistorySelection.from_mapping({"icechunk": "gs://b/run"})
    with pytest.raises(ValueError, match="zstd_clevel"):
        HistorySelection.from_mapping({"icechunk": {"url": "gs://b/run",
                                                    "zstd_clevel": 0}})
    with pytest.raises(ValueError, match="tree-wide"):
        HistorySelection.from_mapping({"icechunk": "gs://b/run"},
                                      tree_wide=False)


def test_go_reads_the_config_table_and_the_flag_wins(tmp_path):
    config = tmp_path / "case.toml"
    config.write_text('[output]\nicechunk = { url = "repo", zstd_clevel = 7 }\n'
                      "[[domain]]\ngrid_id = 1\nhistory_interval_s = 600\n"
                      "[[domain]]\ngrid_id = 2\nhistory_interval_s = 300.0\n")
    args = argparse.Namespace(icechunk=None)
    options = ie.go_export_options(args, config)
    assert options.url == str(tmp_path / "repo") and options.zstd_clevel == 7
    args.icechunk = "s3://bucket/run"
    options = ie.go_export_options(args, config)
    assert options.url == "s3://bucket/run" and options.zstd_clevel == 7
    assert ie.history_intervals(config) == {1: 600, 2: 300}
    assert ie.go_export_options(argparse.Namespace(icechunk=None),
                                tmp_path / "absent.toml") is None


def test_go_refuses_the_flag_on_prepare_only(tmp_path):
    from woof import go_cli

    config = tmp_path / "case.toml"
    config.write_text('[output]\nicechunk = "s3://b/x"\n')
    args = argparse.Namespace(icechunk=None, prepare_only=True)
    assert go_cli._icechunk_for_chain(args, config) is None
    args.icechunk = "s3://b/y"
    with pytest.raises(go_cli.GoRefusal, match="prepare-only"):
        go_cli._icechunk_for_chain(args, config)
    bad = tmp_path / "bad.toml"
    bad.write_text('[output]\nicechunk = { url = "s3://b/x", level = 1 }\n')
    with pytest.raises(go_cli.GoRefusal, match="unknown key"):
        go_cli.icechunk_request(argparse.Namespace(icechunk=None), bad)


def test_go_flag_is_registered():
    from woof import cli

    parser = cli.build_parser()
    args = parser.parse_args(["go", "case.toml", "--icechunk", "s3://b/x"])
    assert args.icechunk == "s3://b/x"
    args = parser.parse_args(["icechunk-export", "run", "--url", "s3://b/x"])
    assert args.func is ie.main


# --------------------------------------------------------------------------
# discovery


def test_discovery_reads_markers_and_finds_frames_without_one(tmp_path):
    run = tmp_path / "run"
    a = publish(run, 1, 0)
    b = publish(run, 1, 1, marker=False)
    (run / "wrfout" / ".wrfout_d01_2025-01-01_00_30_00.tmp.9").write_text("x")
    frames = ie.discover_frames([run])
    assert [f.path for f in frames] == [a, b]
    assert frames[1].valid == T0 + timedelta(minutes=15)
    assert ie.parse_frame_name("wrfout_d03_2026-05-20_18_00_00") == (
        3, datetime(2026, 5, 20, 18, tzinfo=timezone.utc))
    assert ie.discover_frames([tmp_path / "absent"]) == []


# --------------------------------------------------------------------------
# live export


def test_live_export_follows_a_forecast_and_equals_an_after_the_fact_one(
        tmp_path):
    """Frames land in the store while the forecast is still writing; at
    its end the store holds every frame of both domains (one group each),
    REFL_10CM -- first written at t=1 -- is NaN at t=0, and the store
    equals an export of the same files made afterwards, and the batch
    writer's (``tools/wrfout_to_netcdf.py --icechunk-group``) data."""

    run = tmp_path / "run"
    url = _file_url(tmp_path / "live.icechunk")
    state = tmp_path / "export"
    exporter = ie.Exporter([run], _options(url, threads=2), state_dir=state,
                           intervals={1: 900, 2: 1800}, poll_seconds=0.05,
                           finish_file=state / ie.FINISH_NAME)
    result: dict = {}
    worker = threading.Thread(
        target=lambda: result.update(exporter.run(follow=True)))
    worker.start()
    seen_during: list[int] = []
    for index in range(8):
        publish(run, 1, index)
        if index % 2 == 0:
            publish(run, 2, index // 2, step_s=1800)
        time.sleep(0.15)
        status = ie._read_json(state / ie.STATUS_NAME) or {}
        seen_during.append(int(status.get("frames_exported") or 0))
    (state / ie.FINISH_NAME).write_text("{}")
    worker.join(timeout=120)
    assert not worker.is_alive()
    assert result["state"] == "complete"
    assert result["frames_exported"] == 12 and result["frames_pending"] == 0
    assert max(seen_during[:-1]) > 0, seen_during   # landed during the run

    d01 = _open_group(url, "d01")
    assert d01.attrs["time_interval_s"] == 900 and d01["U"].shape[0] == 8
    assert d01.attrs["axis_grows"] is True
    assert np.isnan(d01["REFL_10CM"][0]).all()
    assert np.all(d01["REFL_10CM"][5] == -5.0)
    np.testing.assert_array_equal(d01["frame_written"][:], np.ones(8))
    assert "zstd" in repr(d01["U"].compressors)
    d02 = _open_group(url, "d02")
    assert d02.attrs["time_interval_s"] == 1800 and d02["U"].shape[0] == 4

    after = _file_url(tmp_path / "after.icechunk")
    status = _drain([run], after, tmp_path, threads=1)
    assert status["state"] == "complete" and status["exit_code"] == 0
    assert_same_store(url, after, ["d01", "d02"])

    batch = _file_url(tmp_path / "batch.icechunk")
    for domain, step in ((1, 900), (2, 1800)):
        files = sorted((run / "wrfout").glob(f"wrfout_d{domain:02d}_*"))
        count = len(files)
        st.write_gridded_segment(
            batch, files, group=f"d{domain:02d}",
            axis=st.TimeAxis(np.datetime64("2025-01-01T00:00"),
                             np.datetime64("2025-01-01T00:00")
                             + np.timedelta64((count - 1) * step, "s"), step))
    assert_same_store(url, batch, ["d01", "d02"], ignore_attrs=("axis_grows",))


def test_a_rerun_commits_nothing(tmp_path):
    run = tmp_path / "run"
    for index in range(3):
        publish(run, 1, index)
    url = _file_url(tmp_path / "repo")
    first = _drain([run], url, tmp_path)
    again = _drain([run], url, tmp_path)
    assert again["state"] == "complete"
    assert again["last_snapshot"] == first["last_snapshot"]


def test_a_resumed_run_folder_continues_the_store_and_wins_its_overlap(
        tmp_path):
    """Run 1 died after frame 5, run 2 resumed from frame 4's checkpoint
    into a new folder: one store, ten frames, no duplicate valid time,
    frames 4-5 from run 2.  Exporting run 1 again changes nothing."""

    url = _file_url(tmp_path / "repo")
    first, second = tmp_path / "run1", tmp_path / "run2"
    for index in range(6):
        publish(first, 1, index)
    assert _drain([first], url, tmp_path)["frames_exported"] == 6
    time.sleep(0.01)
    for index in range(4, 10):
        publish(second, 1, index, offset=100.0)
    status = _drain([second], url, tmp_path)
    assert status["state"] == "complete"
    group = _open_group(url, "d01")
    assert group["U"].shape[0] == 10
    u = group["U"][:, 0, 0, 0]
    np.testing.assert_array_equal(u, [0, 1, 2, 3, 104, 105, 106, 107, 108,
                                      109])
    tip = status["last_snapshot"]
    again = _drain([first], url, tmp_path)
    assert again["last_snapshot"] == tip
    np.testing.assert_array_equal(_open_group(url, "d01")["U"][:, 0, 0, 0],
                                  u)
    # Both folders at once, either order: the same store.
    both = _file_url(tmp_path / "both")
    _drain([second, first], both, tmp_path)
    assert_same_store(url, both, ["d01"])


def test_a_rewritten_frame_replaces_the_one_exported(tmp_path):
    """A history rewind deletes frames past the rewind point and the run
    writes them again: the newer file wins."""

    run = tmp_path / "run"
    url = _file_url(tmp_path / "repo")
    for index in range(3):
        publish(run, 1, index)
    _drain([run], url, tmp_path)
    time.sleep(0.01)
    publish(run, 1, 2, offset=50.0)
    _drain([run], url, tmp_path)
    assert _open_group(url, "d01")["U"][2, 0, 0, 0] == 52.0


def test_a_refused_frame_is_named_and_the_export_is_incomplete(tmp_path):
    run = tmp_path / "run"
    url = _file_url(tmp_path / "repo")
    publish(run, 1, 0)
    publish(run, 1, 1, shape=(NZ, NY + 1, NX))
    publish(run, 1, 2)
    status = _drain([run], url, tmp_path)
    assert status["state"] == "incomplete" and status["exit_code"] == 2
    assert status["frames_refused"] == 1
    reason = status["refused"]["wrfout_d01_2025-01-01_00_15_00"]
    assert "shape" in reason
    np.testing.assert_array_equal(_open_group(url, "d01")["frame_written"][:],
                                  [1, 0, 1])


def test_a_store_failure_is_retried_then_fails_and_resumes(tmp_path):
    run = tmp_path / "run"
    for index in range(3):
        publish(run, 1, index)
    url = _file_url(tmp_path / "repo")
    calls = []

    def flaky(*args, **kwargs):
        calls.append(kwargs["segment_id"])
        if kwargs["segment_id"].endswith("00_15_00"):
            raise OSError("connection reset")
        return st.write_gridded_segment(*args, **kwargs)

    state = tmp_path / "state"
    exporter = ie.Exporter([run], _options(url), state_dir=state,
                           write=flaky, sleep=lambda _s: None,
                           say=lambda _line: None)
    status = exporter.run()
    assert status["state"] == "failed" and "connection reset" in status["error"]
    assert calls.count("wrfout_d01_2025-01-01_00_15_00") == \
        ie.TRANSIENT_ATTEMPTS
    resumed = ie.Exporter([run], _options(url), state_dir=state).run()
    assert resumed["state"] == "complete" and resumed["frames_exported"] == 3


def test_the_resume_door_without_an_export_folder_needs_a_url(tmp_path,
                                                              capsys):
    run = tmp_path / "run"
    publish(run, 1, 0)
    assert ie.main(_cli([run])) == 2
    assert "--url" in capsys.readouterr().err
    assert ie.main(_cli([tmp_path / "absent"], url="s3://b/x")) == 2


def test_an_exporter_crash_leaves_the_forecast_running_and_resume_completes(
        tmp_path):
    """The exporter process is killed mid-run; the fake forecast publishes
    every frame on schedule regardless; go's handle reports the export
    INCOMPLETE with the resume command; that command completes the store,
    which then equals an after-the-fact export."""

    run = tmp_path / "go-run" / "run"
    state = tmp_path / "go-run" / ie.EXPORT_DIRNAME
    url = _file_url(tmp_path / "repo")
    said: list[str] = []
    live = ie.LiveExport(_options(url), sources=[run], state_dir=state,
                         intervals={1: 900}, say=said.append,
                         poll_seconds=0.05)
    assert live.start()
    published: list[float] = []
    killed = threading.Event()

    def forecast():
        for index in range(10):
            publish(run, 1, index)
            published.append(time.monotonic())
            time.sleep(0.2)

    producer = threading.Thread(target=forecast)
    producer.start()
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        status = live.status() or {}
        if int(status.get("frames_exported") or 0) >= 2:
            os.kill(live.process.pid, signal.SIGKILL)
            killed.set()
            break
        time.sleep(0.05)
    assert killed.is_set(), live.status()
    producer.join(timeout=120)
    assert len(published) == 10                     # the forecast finished
    assert max(np.diff(published)) < 5.0            # and never stalled
    live.finish()
    result = live.wait()
    assert not result["ok"] and result["exit_code"] == -signal.SIGKILL
    warning = [line for line in said if "INCOMPLETE" in line]
    assert warning and "woof icechunk-export" in warning[0]

    assert ie.main(_cli([state], json=True)) == 0
    group = _open_group(url, "d01")
    np.testing.assert_array_equal(group["frame_written"][:], np.ones(10))
    after = _file_url(tmp_path / "after")
    _drain([run], after, tmp_path)
    assert_same_store(url, after, ["d01"])
    # The run folder itself also names the export (go's layout).
    assert ie.main(_cli([run], json=True)) == 0


def test_live_export_through_go_handle_completes(tmp_path):
    run = tmp_path / "go-run" / "run"
    state = tmp_path / "go-run" / ie.EXPORT_DIRNAME
    url = _file_url(tmp_path / "repo")
    said: list[str] = []
    live = ie.LiveExport(_options(url), sources=[run], state_dir=state,
                         say=said.append, poll_seconds=0.05)
    assert live.start()
    for index in range(4):
        publish(run, 1, index)
        time.sleep(0.05)
    live.finish()
    result = live.wait()
    assert result["ok"] and result["frames_exported"] == 4, (result, said)
    assert any("icechunk export complete" in line for line in said)
    spec = json.loads((state / ie.SPEC_NAME).read_text())
    assert spec["schema"] == ie.SPEC_SCHEMA and spec["sources"] == [str(run)]
    assert (state / ie.LOG_NAME).is_file()


def test_progress_line_reports_lag():
    line = ie.progress_line({"frames_exported": 3, "frames_found": 5,
                             "frames_pending": 2, "lag_model_seconds": 1800,
                             "oldest_pending_wait_seconds": 75})
    assert line == ("icechunk export: 3/5 frames committed, 2 pending "
                    "(30 min 00 s of model time behind, oldest waiting "
                    "1 min 15 s)")


def test_a_deferred_start_waits_for_the_run_to_claim_its_folder(tmp_path):
    """``woof run-plan``'s prepared launch refuses an output folder holding
    anything it did not write, so go starts the exporter only once the
    plan has made ``<output>/chain``."""

    output = tmp_path / "out"
    run = output / "chain" / "run"
    state = output / ie.EXPORT_DIRNAME
    url = _file_url(tmp_path / "repo")
    live = ie.LiveExport(_options(url), sources=[run], state_dir=state,
                         say=lambda _line: None, poll_seconds=0.05)
    assert live.start(after=output / "chain")
    time.sleep(0.3)
    assert not state.exists() and live.process is None
    publish(run, 1, 0)
    publish(run, 1, 1)
    live.finish()
    result = live.wait()
    assert result["ok"] and result["frames_exported"] == 2


def test_a_run_that_ends_before_its_folder_starts_no_exporter(tmp_path):
    output = tmp_path / "out"
    live = ie.LiveExport(_options(_file_url(tmp_path / "repo")),
                         sources=[output / "chain" / "run"],
                         state_dir=output / ie.EXPORT_DIRNAME,
                         say=lambda _line: None, poll_seconds=0.05)
    live.start(after=output / "chain")
    live.finish()
    result = live.wait()
    assert not result["ok"] and result["state"] == "not-started"
    assert "ended before" in result["error"]
    assert not output.exists()


def test_an_export_that_finds_no_frame_is_not_complete(tmp_path):
    (tmp_path / "run").mkdir()
    status = _drain([tmp_path / "run"], _file_url(tmp_path / "repo"),
                    tmp_path)
    assert status["state"] == "empty" and status["exit_code"] == 2
    assert "no history frames" in status["error"]


def test_the_state_file_is_checked_against_the_store(tmp_path):
    """A repository deleted since the last export is exported again, not
    reported complete from a stale state.json."""

    import shutil

    run = tmp_path / "run"
    for index in range(3):
        publish(run, 1, index)
    repo = tmp_path / "repo"
    state = tmp_path / "state"
    args = _cli([run], url=_file_url(repo), state_dir=state, json=True)
    assert ie.main(args) == 0
    shutil.rmtree(repo)
    assert ie.main(args) == 0
    np.testing.assert_array_equal(
        _open_group(_file_url(repo), "d01")["frame_written"][:], [1, 1, 1])


def test_a_frame_reached_through_a_symlink_is_one_frame(tmp_path):
    real = tmp_path / "real"
    publish(real / "run", 1, 0)
    publish(real / "run", 1, 1)
    link = tmp_path / "link"
    link.symlink_to(real)
    frames = ie.discover_frames([link / "run"])
    assert len(frames) == 2
    assert all(str(f.path).startswith(str(link)) for f in frames)


def test_concurrent_writes_never_share_a_chunk(tmp_path):
    """A group made with two frames per time chunk is written one array
    at a time, however many threads are asked for."""

    import netCDF4 as nc4

    url = _file_url(tmp_path / "repo")
    first = publish(tmp_path / "a", 1, 0)
    st.write_gridded_segment(
        url, [first], group="d01", time_chunk=2,
        axis=st.TimeAxis(np.datetime64("2025-01-01T00:00"),
                         np.datetime64("2025-01-01T02:00"), 900))
    # One file holding frames 2 and 3, which share a time chunk.
    two = tmp_path / "b" / "wrfout_d01_2025-01-01_00_30_00"
    two.parent.mkdir()
    with nc4.Dataset(two, "w") as handle:
        handle.createDimension("Time", None)
        handle.createDimension("DateStrLen", 19)
        handle.createDimension("bottom_top", NZ)
        handle.createDimension("south_north", NY)
        handle.createDimension("west_east", NX)
        times = handle.createVariable("Times", "S1", ("Time", "DateStrLen"))
        times.set_auto_chartostring(False)
        u = handle.createVariable("U", "f4", ("Time", "bottom_top",
                                              "south_north", "west_east"))
        t2 = handle.createVariable("T2", "f4", ("Time", "south_north",
                                                "west_east"))
        for n, minute in enumerate((30, 45)):
            stamp = f"2025-01-01_00:{minute}:00"
            times[n] = np.frombuffer(stamp.encode("ascii"), dtype="S1")
            u[n] = np.full((NZ, NY, NX), float(minute), np.float32)
            t2[n] = np.full((NY, NX), float(minute), np.float32)
    st.write_gridded_segment(url, [two], group="d01", write_threads=8)
    group = _open_group(url, "d01")
    np.testing.assert_array_equal(group["U"][2:4, 0, 0, 0], [30.0, 45.0])
    np.testing.assert_array_equal(group["T2"][2:4, 0, 0], [30.0, 45.0])


def test_go_stops_an_exporter_that_stalls(tmp_path):
    class Stalled(ie.LiveExport):
        STALL_SECONDS = 1.0

        def command(self):
            import sys
            return [sys.executable, "-c", "import time; time.sleep(120)"]

    said: list[str] = []
    live = Stalled(_options(_file_url(tmp_path / "repo")),
                   sources=[tmp_path / "run"], state_dir=tmp_path / "state",
                   say=said.append)
    assert live.start()
    started = time.monotonic()
    result = live.wait()
    assert time.monotonic() - started < 60
    assert not result["ok"]
    assert any("no progress" in line and "INCOMPLETE" in line
               for line in said)
