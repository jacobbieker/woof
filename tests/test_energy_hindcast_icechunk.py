"""Icechunk stores for the energy hindcast: storage URLs, the site store,
the gridded wrfout writer, conflicts between concurrent writers, and the
``woof energy icechunk`` / ``extract --icechunk-mode`` doors.

Every store here is ``memory://`` or ``file://`` under ``tmp_path``.  The S3
round trip runs only when ``WOOF_S3_TEST_URL`` names a bucket prefix to use
(credentials from the environment); otherwise it is skipped.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import pickle
import sys
import types
import uuid
from pathlib import Path

import numpy as np
import pytest

ic = pytest.importorskip("icechunk")
xr = pytest.importorskip("xarray")
zarr = pytest.importorskip("zarr")
pytest.importorskip("netCDF4")

from woof.energy import extract  # noqa: E402
from woof.energy import icechunk_store as st  # noqa: E402
from woof.energy.contracts import load_sites  # noqa: E402
from woof.energy.extract import ForecastData  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import energy_icechunk_workers as workers  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures" / "energy"
SITES = FIXTURES / "sites_wales.json"
DAY = "2025-01-01T00:00,2025-01-01T23:45,15min"


def _memory() -> str:
    return f"memory://{uuid.uuid4().hex}"


def _axis(text: str = DAY) -> st.TimeAxis:
    return st.TimeAxis.parse(text)


def _forecast(site_ids, times, heights, *, u=3.0, t2=280.0, domain="tile0"):
    count, nt, nh = len(site_ids), len(times), len(heights)
    return ForecastData(
        times=np.asarray(times, dtype="datetime64[s]"),
        heights_m=np.asarray(heights, dtype=np.float64),
        coords={"site_id": np.asarray(site_ids, dtype=object),
                "terrain_height": np.full(count, 12.0),
                "domain_id": np.asarray([domain] * count, dtype=object),
                "dx_m": np.full(count, 100.0),
                "inside": np.ones(count, dtype=np.int8)},
        variables={"u": np.full((nt, count, nh), u),
                   "t2": np.full((nt, count), t2)},
        variable_notes={"u": ["a note"]}, notes=["global"],
        attrs={"domains": domain})


def _open(store: str, group: str):
    repo = st.open_repository(store, create=False)
    return xr.open_zarr(repo.readonly_session("main").store, group=group,
                        consolidated=False)


@pytest.fixture(scope="module")
def sites():
    return load_sites(SITES)


@pytest.fixture
def site_store(sites):
    store = _memory()
    st.init_site_store(store, sites=SITES, axis=_axis(), site_chunk=4,
                       variables=["u", "t2"])
    return store


# ------------------------------------------------------------------ storage


def test_storage_from_url_schemes(tmp_path):
    for url in (f"file://{tmp_path}/a", str(tmp_path / "b"),
                "s3://bucket/some/prefix", "gs://bucket/p", "gcs://bucket"):
        assert isinstance(st.storage_from_url(url), ic.Storage), url
    minio = st.storage_from_url("s3://bucket/x",
                                endpoint_url="http://127.0.0.1:9000",
                                region="us-east-1")
    assert isinstance(minio, ic.Storage)
    assert st.storage_from_url("s3://public/x", anonymous=True)
    assert st.storage_from_url("memory://same") \
        is st.storage_from_url("memory://same")
    assert st.storage_from_url("memory://one") \
        is not st.storage_from_url("memory://two")


@pytest.mark.parametrize("url, match", [
    ("ftp://host/x", "not one of"),
    ("s3://key:secret@bucket/x", "credentials in the URL"),
    ("s3:///nobucket", "names no bucket"),
    ("file://", "names no path"),
])
def test_storage_from_url_refusals(url, match):
    with pytest.raises(st.IcechunkStoreRefused, match=match):
        st.storage_from_url(url)


def test_is_store_url():
    assert st.is_store_url("s3://b/p") and st.is_store_url("memory://x")
    assert not st.is_store_url("/data/repo")
    assert not st.is_store_url("relative/repo")


def test_filesystem_commits_are_locked_beside_the_repo(tmp_path):
    from contextlib import nullcontext

    assert st.commit_lock("s3://b/p") is nullcontext
    assert st.commit_lock("memory://x") is nullcontext
    assert st.local_path(f"file://{tmp_path}/r") == tmp_path / "r"
    with st.commit_lock(f"file://{tmp_path}/r")():
        assert (tmp_path / ("r" + st.COMMIT_LOCK_SUFFIX)).exists()
    st.open_repository(str(tmp_path / "r"))
    assert ic.Repository.exists(ic.local_filesystem_storage(
        str(tmp_path / "r")))


def test_open_repository_refuses_a_file(tmp_path):
    target = tmp_path / "f"
    target.write_text("x")
    with pytest.raises(st.IcechunkStoreRefused, match="not an Icechunk"):
        st.open_repository(str(target))
    with pytest.raises(st.IcechunkStoreRefused, match="no Icechunk"):
        st.open_repository(_memory(), create=False)


# ---------------------------------------------------------------- time axis


def test_time_axis():
    axis = _axis()
    assert axis.size == 96
    assert axis.times()[1] == np.datetime64("2025-01-01T00:15:00")
    np.testing.assert_array_equal(
        axis.index_of(["2025-01-01T00:30", "2025-01-01T23:45"]), [2, 95])
    with pytest.raises(st.IcechunkStoreRefused, match="not on the store"):
        axis.index_of(["2025-01-01T00:20"])
    with pytest.raises(st.IcechunkStoreRefused, match="not on the store"):
        axis.index_of(["2025-01-02T00:00"])
    with pytest.raises(st.IcechunkStoreRefused, match="whole number"):
        st.TimeAxis("2025-01-01T00:00", "2025-01-01T00:20", 900)
    assert st.TimeAxis.from_attrs(axis.to_attrs()) == axis
    assert st.parse_interval("15min") == 900
    assert st.parse_interval("1h") == 3600
    assert st.parse_interval("60") == 60
    with pytest.raises(st.IcechunkStoreRefused):
        st.parse_interval("0.5s")


# --------------------------------------------------------------- site store


def test_init_site_store_is_idempotent_and_refuses_a_different_one(sites):
    store = _memory()
    first = st.init_site_store(store, sites=SITES, axis=_axis(),
                               variables=["u", "t2"])
    assert first["created"] and first["times"] == 96
    assert first["sites"] == len(sites) and first["time_chunk"] == 96
    again = st.init_site_store(store, sites=SITES, axis=_axis(),
                               variables=["u", "t2"])
    assert not again["created"] and again["snapshot"] == first["snapshot"]
    with pytest.raises(st.IcechunkStoreRefused, match="differs in"):
        st.init_site_store(store, sites=SITES, axis=_axis(),
                           variables=["u"])
    with _open(store, "forecast") as ds:
        assert ds.sizes == {"time": 96, "site": len(sites),
                            "height": len(sites.heights_m)}
        assert ds["u"].dtype == np.float32
        assert bool(ds["u"].isnull().all())
        assert str(ds["site_id"].values[0]) == sites.sites[0].site_id
        assert "lat" in ds.coords and "domain_id" in ds.coords
        assert ds.attrs["schema"] == st.SITE_STORE_SCHEMA
        codecs = ds["u"].encoding["compressors"]
        assert "zstd" in repr(codecs) and "bitshuffle" in repr(codecs)


def test_site_region_write_reads_back(site_store, sites):
    ids = [s.site_id for s in sites.sites][5:9]
    times = _axis().times()[10:14]
    out = st.write_site_region(site_store, _forecast(ids, times,
                                                     sites.heights_m),
                               job_id="job-1", segment_id="seg-1")
    assert out["committed"] and out["times_written"] == 4
    with _open(site_store, "forecast") as ds:
        u = ds["u"].values
        assert np.all(u[10:14, 5:9] == 3.0)
        assert np.isnan(u[:10]).all() and np.isnan(u[14:]).all()
        assert np.isnan(u[:, :5]).all() and np.isnan(u[:, 9:]).all()
        assert int(ds["written"].sum()) == 16
        assert [str(v) for v in ds["domain_id"].values[5:9]] == ["tile0"] * 4
        assert np.all(ds["inside"].values[5:9] == 1)
    repo = st.open_repository(site_store, create=False)
    tip = next(iter(repo.ancestry(branch="main")))
    assert tip.metadata["job_id"] == "job-1"
    assert tip.metadata["segment_id"] == "seg-1"
    assert tip.metadata["schema"] == st.WRITE_SCHEMA
    assert tip.metadata["notes"] == ["global"]


def test_site_region_rerun_is_idempotent(site_store, sites):
    ids = [s.site_id for s in sites.sites][:3]
    data = _forecast(ids, _axis().times()[:4], sites.heights_m)
    first = st.write_site_region(site_store, data, job_id="j")
    skipped = st.write_site_region(site_store, data, job_id="j",
                                   on_existing="skip")
    assert not skipped["committed"] and skipped["times_skipped"] == 4
    assert skipped["snapshot"] == first["snapshot"]
    again = st.write_site_region(site_store, data, job_id="j")
    with _open(site_store, "forecast") as ds:
        assert np.all(ds["u"].values[:4, :3] == 3.0)
        assert int(ds["written"].sum()) == 12
    assert again["committed"]


def test_site_region_unordered_sites(site_store, sites):
    ids = [s.site_id for s in sites.sites]
    chosen = [ids[7], ids[2], ids[20]]
    data = _forecast(chosen, _axis().times()[:2], sites.heights_m)
    data.variables["t2"] = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    st.write_site_region(site_store, data)
    with _open(site_store, "forecast") as ds:
        t2 = ds["t2"].values
    np.testing.assert_array_equal(t2[:2, [7, 2, 20]],
                                  [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])


@pytest.mark.parametrize("change, match", [
    ("off_axis", "not on the store"),
    ("site", "not in the store"),
    ("heights", "differ from the store"),
    ("variable", "initialised without"),
])
def test_site_region_refusals(site_store, sites, change, match):
    ids = [s.site_id for s in sites.sites][:2]
    times = _axis().times()[:2]
    heights = list(sites.heights_m)
    if change == "off_axis":
        times = times + np.timedelta64(60, "s")
    if change == "site":
        ids = ids[:1] + ["nope"]
    if change == "heights":
        heights = heights[:-1]
    data = _forecast(ids, times, heights)
    if change == "variable":
        data.variables["v"] = data.variables["u"]
    with pytest.raises(st.IcechunkStoreRefused, match=match):
        st.write_site_region(site_store, data)


def test_site_region_needs_init():
    store = _memory()
    st.open_repository(store)
    data = _forecast(["a"], _axis().times()[:1], [10.0])
    with pytest.raises(st.IcechunkStoreRefused, match="init"):
        st.write_site_region(store, data)


# ---------------------------------------------------------------- conflicts


def _colliding(store, sites, first_ids, second_ids):
    """Write ``second`` with ``first`` committed in between, so ``second``
    always commits against a moved tip."""

    times = _axis().times()[:4]
    first = _forecast(first_ids, times, sites.heights_m, u=1.0)
    second = _forecast(second_ids, times, sites.heights_m, u=2.0)

    def hook(attempt):
        if attempt == 1:
            st.write_site_region(store, first, job_id="first")

    return st.write_site_region(store, second, job_id="second",
                                before_commit=hook,
                                policy=st.RetryPolicy(base_delay_s=0.001))


def test_disjoint_chunks_rebase(sites):
    store = _memory()
    st.init_site_store(store, sites=SITES, axis=_axis(), site_chunk=4,
                       variables=["u", "t2"])
    ids = [s.site_id for s in sites.sites]
    out = _colliding(store, sites, ids[0:4], ids[4:8])
    assert out["rebases"] == 1 and out["rewrites"] == 0
    with _open(store, "forecast") as ds:
        assert np.all(ds["u"].values[:4, 0:4] == 1.0)
        assert np.all(ds["u"].values[:4, 4:8] == 2.0)


def test_shared_chunk_is_rewritten_not_lost(sites):
    store = _memory()
    st.init_site_store(store, sites=SITES, axis=_axis(), site_chunk=64,
                       variables=["u", "t2"])
    ids = [s.site_id for s in sites.sites]
    out = _colliding(store, sites, ids[0:3], ids[3:6])
    assert out["rewrites"] == 1
    with _open(store, "forecast") as ds:
        assert np.all(ds["u"].values[:4, 0:3] == 1.0)
        assert np.all(ds["u"].values[:4, 3:6] == 2.0)
        assert int(ds["written"].values[:4, :6].sum()) == 24


def test_retries_are_bounded():
    repo = st.open_repository(_memory())
    sleeps = []

    def apply(session):
        zarr.open_group(session.store, mode="a").attrs["k"] = uuid.uuid4().hex

    def hook(attempt):
        other = repo.writable_session("main")
        zarr.open_group(other.store, mode="a").attrs["k"] = uuid.uuid4().hex
        other.commit("interloper")

    with pytest.raises(st.IcechunkStoreRefused, match="after 3 attempts"):
        st.commit_with_retry(repo, branch="main", apply=apply, message="m",
                             metadata={}, before_commit=hook,
                             policy=st.RetryPolicy(max_attempts=3),
                             sleep=sleeps.append)
    assert len(sleeps) == 2
    assert all(0.0 <= s <= 0.5 for s in sleeps)


# ------------------------------------------------------------ gridded store


def test_gridded_segment_round_trip(tmp_path):
    store = _memory()
    paths = workers.write_segment_files(tmp_path / "seg", 4, 4)
    out = st.write_gridded_segment(store, paths, group="leaf7/d02",
                                   axis=_axis(), job_id="job-7",
                                   segment_id="s0")
    assert out["created"] and out["frames_written"] == 4
    assert out["dropped_variables"] == [] and out["chunk_aligned"]
    assert out["compression"]["shuffle"] == "bitshuffle"
    with _open(store, "leaf7/d02") as ds:
        assert ds.sizes["Time"] == 96
        assert ds["Time"].values[4] == np.datetime64("2025-01-01T01:00")
        assert "XTIME" not in ds.variables and "Times" not in ds.variables
        u = ds["U"].values
        for t in range(4, 8):
            assert np.all(u[t] == t)
        assert np.isnan(u[:4]).all() and np.isnan(u[8:]).all()
        refl = ds["REFL_10CM"].values
        assert np.isnan(refl[4]).all()           # seed frame: NaN-filled
        assert np.all(refl[5:8] == -np.arange(5, 8)[:, None, None, None])
        assert ds["ITIMESTEP"].values[6] == 6
        np.testing.assert_array_equal(ds["HGT_STATIC"].values,
                                      np.arange(12).reshape(3, 4))
        np.testing.assert_array_equal(np.nonzero(ds["frame_written"].values)[0],
                                      [4, 5, 6, 7])
        assert ds.attrs["wrf_DX"] == 100.0
        assert "zstd" in repr(ds["U"].encoding["compressors"])
    assert out["absent_variables"] == ["REFL_10CM"]


def test_gridded_rerun_skips_or_overwrites(tmp_path):
    store = _memory()
    paths = workers.write_segment_files(tmp_path / "seg", 0, 3)
    first = st.write_gridded_segment(store, paths, group="d01", axis=_axis())
    again = st.write_gridded_segment(store, paths, group="d01", axis=_axis())
    assert not again["committed"] and again["frames_skipped"] == 3
    assert again["snapshot"] == first["snapshot"]
    forced = st.write_gridded_segment(store, paths, group="d01",
                                      on_existing="overwrite")
    assert forced["committed"] and forced["frames_written"] == 3


def test_gridded_window_and_refusals(tmp_path):
    store = _memory()
    paths = workers.write_segment_files(tmp_path / "seg", 0, 6, per_file=2)
    out = st.write_gridded_segment(
        store, paths, group="d01", axis=_axis(),
        window=("2025-01-01T00:30", "2025-01-01T01:15"))
    assert out["frames"] == 3 and out["frames_windowed_out"] == 3
    with _open(store, "d01") as ds:
        np.testing.assert_array_equal(
            np.nonzero(ds["frame_written"].values)[0], [2, 3, 4])
    with pytest.raises(st.IcechunkStoreRefused, match="time axis"):
        st.write_gridded_segment(store, paths, group="d01",
                                 axis=_axis("2025-01-01T00:00,"
                                            "2025-01-02T00:00,15min"))
    with pytest.raises(st.IcechunkStoreRefused, match="give its time axis"):
        st.write_gridded_segment(store, paths, group="d02")
    late = workers.write_segment_files(tmp_path / "late", 200, 1)
    with pytest.raises(st.IcechunkStoreRefused, match="not on the store"):
        st.write_gridded_segment(store, late, group="d01")
    with pytest.raises(st.IcechunkStoreRefused, match="in both"):
        st.write_gridded_segment(store, paths + workers.write_segment_files(
            tmp_path / "dup", 0, 1), group="d01")


def test_gridded_time_chunk_alignment_is_reported(tmp_path):
    store = _memory()
    paths = workers.write_segment_files(tmp_path / "seg", 2, 4)
    out = st.write_gridded_segment(store, paths, group="d01", axis=_axis(),
                                   time_chunk=4)
    assert not out["chunk_aligned"]
    with _open(store, "d01") as ds:
        assert ds["U"].encoding["chunks"][0] == 4


def test_gridded_window_drops_restart_overlap_and_chunk_from_group(tmp_path):
    store = _memory()
    first = workers.write_segment_files(tmp_path / "a", 0, 5)
    restart = workers.write_segment_files(tmp_path / "b", 4, 4)
    with pytest.raises(st.IcechunkStoreRefused, match="in both"):
        st.write_gridded_segment(store, first + restart, group="d01",
                                 axis=_axis())
    out = st.write_gridded_segment(
        store, first + restart, group="d01", axis=_axis(), time_chunk=4,
        window=("2025-01-01T00:00", "2025-01-01T01:00"))
    assert out["frames"] == 4 and out["frames_windowed_out"] == 5
    assert out["chunk_aligned"]
    later = st.write_gridded_segment(store, restart, group="d01",
                                     window=("2025-01-01T01:00",
                                             "2025-01-01T01:30"))
    assert not later["chunk_aligned"]       # the group's chunk is 4 frames
    with pytest.raises(st.IcechunkStoreRefused, match="at least 1"):
        st.write_gridded_segment(store, restart, group="d01", time_chunk=0)


def test_site_digest_is_the_same_for_a_path_and_loaded_sites(sites):
    store = _memory()
    st.init_site_store(store, sites=SITES, axis=_axis(), variables=["u"])
    again = st.init_site_store(store, sites=load_sites(SITES), axis=_axis(),
                               variables=["u"])
    assert not again["created"]


# -------------------------------------------------- concurrent processes


def _run(context, target, argsets):
    results = context.Queue()
    gate = context.Barrier(len(argsets))
    procs = [context.Process(target=target, args=(*args, gate, results))
             for args in argsets]
    for proc in procs:
        proc.start()
    out = [results.get(timeout=300) for _ in procs]
    for proc in procs:
        proc.join(timeout=60)
    errors = [o["error"] for o in out if "error" in o]
    assert not errors, "\n".join(errors)
    assert all(proc.exitcode == 0 for proc in procs)
    return out


def test_two_processes_write_two_segments_into_a_file_repo(tmp_path):
    """Two segments of one domain, two processes, one uninitialised group:
    both race to create it, both commit, and the axis reads back
    continuous with every frame present."""

    store = f"file://{tmp_path / 'repo'}"
    seg_a = workers.write_segment_files(tmp_path / "a", 0, 6, per_file=2)
    seg_b = workers.write_segment_files(tmp_path / "b", 6, 6, per_file=2)
    context = multiprocessing.get_context("spawn")
    out = _run(context, workers.gridded_writer, [
        (store, [str(p) for p in seg_a], "leaf1/d01", DAY, "job-a"),
        (store, [str(p) for p in seg_b], "leaf1/d01", DAY, "job-b"),
    ])
    assert sum(o["created"] for o in out) == 1
    assert all(o["committed"] for o in out)
    with _open(store, "leaf1/d01") as ds:
        written = np.nonzero(ds["frame_written"].values)[0]
        np.testing.assert_array_equal(written, np.arange(12))
        times = ds["Time"].values[written]
        assert np.all(np.diff(times) == np.timedelta64(900, "s"))
        u = ds["U"].values
        for t in range(12):
            assert np.all(u[t] == t), t
    report = st.verify_repository(store)
    group = report["groups"]["leaf1/d01"]
    assert group["steps_written"] == 12
    assert group["first_written"] == "2025-01-01T00:00:00Z"
    assert group["last_written"] == "2025-01-01T02:45:00Z"
    rerun = st.write_gridded_segment(store, seg_a, group="leaf1/d01")
    assert not rerun["committed"] and rerun["frames_skipped"] == 6


def test_two_processes_share_site_chunks_without_lost_writes(tmp_path,
                                                             sites):
    """Two jobs, one site chunk, commits forced to collide: one rebases
    or rewrites, and both jobs' values are in the store."""

    store = f"file://{tmp_path / 'sites'}"
    st.init_site_store(store, sites=SITES, axis=_axis(), site_chunk=64,
                       variables=["u", "t2"])
    ids = [s.site_id for s in sites.sites]
    times = _axis().times()
    jobs = []
    for n, (chosen, window, u) in enumerate([
            (ids[0:10], times[0:8], 1.0), (ids[10:20], times[4:12], 2.0)]):
        path = tmp_path / f"job{n}.pkl"
        path.write_bytes(pickle.dumps(_forecast(chosen, window,
                                                sites.heights_m, u=u)))
        jobs.append(path)
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(2)
    out = _run(context, workers.site_writer, [
        (store, str(jobs[0]), "job-0", barrier),
        (store, str(jobs[1]), "job-1", barrier),
    ])
    assert all(o["committed"] for o in out)
    assert sum(o["rewrites"] + o["rebases"] for o in out) >= 1
    with _open(store, "forecast") as ds:
        u = ds["u"].values
        assert np.all(u[0:8, 0:10] == 1.0)
        assert np.all(u[4:12, 10:20] == 2.0)
        assert int(ds["written"].sum()) == 8 * 10 + 8 * 10
    repo = st.open_repository(store, create=False)
    jobs_seen = {snap.metadata.get("job_id")
                 for snap in repo.ancestry(branch="main")}
    assert {"job-0", "job-1"} <= jobs_seen


# ------------------------------------------------------------------- verify


def test_verify_repository(site_store, sites):
    ids = [s.site_id for s in sites.sites][:2]
    st.write_site_region(site_store, _forecast(ids, _axis().times()[3:5],
                                               sites.heights_m),
                         job_id="v")
    report = st.verify_repository(site_store)
    assert report["schema"] == st.VERIFY_SCHEMA
    assert report["snapshot"] and report["last_commit"]["metadata"]["job_id"] \
        == "v"
    group = report["groups"]["forecast"]
    assert group["steps_written"] == 2 and group["sites"] == len(sites)
    assert group["first_written"] == "2025-01-01T00:45:00Z"
    quick = st.verify_repository(site_store, group="forecast", coverage=False)
    assert "steps_written" not in quick["groups"]["forecast"]
    with pytest.raises(st.IcechunkStoreRefused, match="no group"):
        st.verify_repository(site_store, group="absent")
    with pytest.raises(st.IcechunkStoreRefused, match="no branch"):
        st.verify_repository(site_store, branch="nope")
    with pytest.raises(st.IcechunkStoreRefused, match="no Icechunk"):
        st.verify_repository(_memory())


# -------------------------------------------------------------------- doors


def test_cli_init_and_verify(tmp_path, capsys):
    from woof.cli import build_parser

    repo = f"file://{tmp_path / 'r'}"
    parser = build_parser()
    args = parser.parse_args([
        "energy", "icechunk", "init", repo, "--sites", str(SITES),
        "--start", "2025-01-01T00:00", "--end", "2025-01-01T23:45",
        "--interval", "15min", "--vars", "u,t2", "--site-chunk", "8",
        "--job-id", "init-1"])
    assert args.func(args) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["created"] and summary["site_chunk"] == 8
    args = parser.parse_args(["energy", "icechunk", "verify", repo])
    assert args.func(args) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["groups"]["forecast"]["steps_written"] == 0
    args = parser.parse_args(["energy", "icechunk", "verify",
                              f"file://{tmp_path / 'missing'}"])
    assert args.func(args) == 2
    assert "refused" in capsys.readouterr().err


def _extract_args(output, **extra):
    record = dict(plan="plan.json", sites=None, heights_m=None, vars=None,
                  format="icechunk", output=str(output))
    record.update(extra)
    return types.SimpleNamespace(**record)


def test_extract_region_mode(monkeypatch, sites, capsys):
    store = _memory()
    st.init_site_store(store, sites=SITES, axis=_axis(),
                       variables=["u", "t2"])
    ids = [s.site_id for s in sites.sites][:4]
    data = _forecast(ids, _axis().times()[8:12], sites.heights_m)
    monkeypatch.setattr(extract, "build_forecast", lambda *a, **k: data)
    args = _extract_args(store, icechunk_mode="region", job_id="ex-1")
    assert extract.main(args) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["output"] == store and summary["snapshot"]
    assert summary["icechunk_mode"] == "region"
    with _open(store, "forecast") as ds:
        assert np.all(ds["u"].values[8:12, :4] == 3.0)


def test_extract_replace_mode_to_a_url(monkeypatch, tmp_path, sites, capsys):
    data = _forecast([s.site_id for s in sites.sites][:2],
                     _axis().times()[:3], sites.heights_m)
    for name, value in (("lat", 51.5), ("lon", -3.2), ("asset_id", "a"),
                        ("kind", "turbine"), ("bearing_deg", np.nan),
                        ("voltage_kv", np.nan), ("hub_height_m", 90.0),
                        ("capacity_mw", 2.0)):
        data.coords[name] = np.asarray([value] * 2,
                                       dtype=object if isinstance(value, str)
                                       else float)
    monkeypatch.setattr(extract, "build_forecast", lambda *a, **k: data)
    url = f"file://{tmp_path / 'repl'}"
    assert extract.main(_extract_args(url, job_id="r")) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["output"] == url
    repo = ic.Repository.open(ic.local_filesystem_storage(
        str(tmp_path / "repl")))
    with xr.open_zarr(repo.readonly_session("main").store,
                      consolidated=False) as ds:
        assert np.all(ds["u"].values == 3.0)
    assert extract.main(_extract_args(url, format="netcdf")) == 2
    assert "needs --format icechunk" in capsys.readouterr().err
    assert extract.main(_extract_args(tmp_path / "x.nc", format="netcdf",
                                      job_id="j")) == 2
    assert "apply only to --format icechunk" in capsys.readouterr().err


def test_extract_region_without_init_is_refused(monkeypatch, sites, capsys):
    store = _memory()
    data = _forecast([sites.sites[0].site_id], _axis().times()[:1],
                     sites.heights_m)
    monkeypatch.setattr(extract, "build_forecast", lambda *a, **k: data)
    assert extract.main(_extract_args(store, icechunk_mode="region")) == 2
    assert "refused" in capsys.readouterr().err


def test_wrfout_tool_segment_mode(tmp_path, monkeypatch, capsys):
    tools = Path(__file__).resolve().parents[1] / "tools"
    monkeypatch.syspath_prepend(str(tools))
    import wrfout_to_netcdf as tool

    source = tmp_path / "src"
    workers.write_segment_files(source, 0, 3)
    url = f"file://{tmp_path / 'grid'}"
    argv = ["wrfout_to_netcdf.py", str(source), "--icechunk-repo", url,
            "--icechunk-group", "leaf/d01", "--time-axis", DAY,
            "--job-id", "tool-1", "--segment-id", "s1",
            "--compress-zstd-bitshuffle", "--zstd-clevel", "3"]
    monkeypatch.setattr(sys, "argv", argv)
    assert tool.main() == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["frames_written"] == 3 and summary["job_id"] == "tool-1"
    assert summary["compression"]["clevel"] == 3
    monkeypatch.setattr(sys, "argv", argv)
    assert tool.main() == 0
    assert json.loads(capsys.readouterr().out)["frames_skipped"] == 3
    # the original whole-directory export takes a URL as well
    out_dir = tmp_path / "nc"
    monkeypatch.setattr(sys, "argv", [
        "wrfout_to_netcdf.py", str(source), str(out_dir), "--icechunk-repo",
        f"file://{tmp_path / 'whole'}", "--compress-zstd-bitshuffle"])
    assert tool.main() == 0
    assert "Committed Icechunk snapshot" in capsys.readouterr().out
    assert len(list(out_dir.glob("wrfout_d01_*.nc"))) == 3


# ----------------------------------------------------------------------- S3


@pytest.mark.network
@pytest.mark.skipif(not os.environ.get("WOOF_S3_TEST_URL"),
                    reason="set WOOF_S3_TEST_URL=s3://bucket/prefix to run")
def test_s3_round_trip(sites):
    base = os.environ["WOOF_S3_TEST_URL"].rstrip("/")
    store = f"{base}/woof-test-{uuid.uuid4().hex}"
    st.init_site_store(store, sites=SITES, axis=_axis(),
                       variables=["u", "t2"])
    ids = [s.site_id for s in sites.sites][:2]
    out = st.write_site_region(store, _forecast(ids, _axis().times()[:2],
                                                sites.heights_m),
                               job_id="s3")
    report = st.verify_repository(store)
    assert report["snapshot"] == out["snapshot"]
    assert report["groups"]["forecast"]["steps_written"] == 2
