"""WRF's lightning diagnostics on the MPAS suite route (woof.hex.mpasdiag.lightning_diag_v842).

&diagnostics config_lightning_option 1/2/3 (and WRF's lightning knobs) add WRF's flash rates/counts and LPI as ``diag``
fields.  Here:

* the options resolve (off by default, WRF's defaults), and option 11, iccg_method 4, cldtop_adjustment, PR92 without
  a radar operator, LPI without graupel, and every option on the RRFS fork route are refused by name; the Registry,
  default streams and driver receipt carry the fields only with the option (the off state is pinned by digest);
* the mesh columns handed to the WRF route's launcher are WRF's columns (theta*exner, mid-layer heights, layer
  depths, the top memory level, xland, lat/lon in degrees), built here independently with numpy;
* over several steps, with WRF's activation clock (lightning_start_seconds, lightning_dt), every published word
  equals woof.core.lightning.launch_lightning (the kernels graded against WRF's Fortran by
  tests/test_lightning_cuda.py) on those columns: PR92 w (1), PR92 cloud top (2) and LPI (3);
* the flash counts and the clock ride a checkpoint: a continuation on local steps plus the straight run's step
  offset is bitwise the straight run.
"""
from __future__ import annotations

import datetime as dt
from types import SimpleNamespace

import numpy as np
import pytest

from woof.hex.mpasdiag import lightning_diag_v842 as ltng
from woof.hex.mpasphys import resolve_namelist

NL, NC = 20, 12
DT = 20.0


def _namelist(extra=None, physics=()):
    lines = ["&physics", " config_physics_suite='convection_permitting'", *physics, "/"]
    if extra is not None:
        lines += ["&diagnostics", *[f" {k}={v}" for k, v in extra.items()], "/"]
    return "\n".join(lines) + "\n"


def _resolve(extra=None, physics=()):
    return resolve_namelist(_namelist(extra, physics), default_backend="hex-cuda")


def test_options_default_off_and_resolve():
    assert _resolve().lightning_diagnostics is None and "lightning_diagnostics" not in _resolve().receipt()
    assert _resolve({"config_lightning_option": 0}).lightning_diagnostics is None
    config = _resolve({"config_lightning_option": 3, "config_lightning_dt": 60.0})
    assert config.lightning_diagnostics == ltng.LightningConfig(option=3, lightning_dt=60.0)
    assert ltng.packages(config) == {"ltng_lpi"}
    assert config.receipt()["lightning_diagnostics"]["option"] == 3
    for key in ltng.OPTIONS:
        assert key not in config.options
    pr92 = _resolve({"config_lightning_option": 2, "config_iccg_method": 3, "config_flashrate_factor": 0.5})
    assert ltng.packages(pr92) == {"ltng_crm_PR92z"}
    assert set(pr92.lightning_diagnostics.names) == set(ltng.FLASH_NAMES)


@pytest.mark.parametrize("extra, physics, message", [
    ({"config_lightning_option": 11}, (), "= 11 is not available.*ktop_deep"),
    ({"config_lightning_option": 4}, (), "has no flash or LPI producer"),
    ({"config_lightning_option": 1, "config_iccg_method": 4}, (), "iccg_method=4 is held"),
    ({"config_lightning_option": 1, "config_cldtop_adjustment": 1.0}, (), "read only by config_lightning_option = 11"),
    ({"config_lightning_dt": 60.0}, (), "while config_lightning_option is 0"),
    ({"config_lightning_option": 1, "config_iccg_prescribed_num": 0.0, "config_iccg_prescribed_den": 0.0}, (), "0/0"),
    ({"config_lightning_option": ".true."}, (), "is an integer"),
    ({"config_lightning_option": 1}, (" config_microp_scheme='mp_wsm6'",), "PR92.*'mp_wsm6'"),
    ({"config_lightning_option": 3}, (" config_microp_scheme='mp_kessler'",), "needs a graupel.*'mp_kessler'"),
])
def test_unavailable_options_are_refused(extra, physics, message):
    with pytest.raises(ValueError, match=message):
        _resolve(extra, physics)


def test_lpi_resolves_for_wsm6():
    assert _resolve({"config_lightning_option": 3}, (" config_microp_scheme='mp_wsm6'",)).lightning_diagnostics.option == 3


def test_rrfs_fork_route_refuses_the_options():
    from woof.hex.mpasphys.rrfs.driver_rrfs import resolve_namelist as resolve_rrfs
    text = "&physics\n config_physics_suite='hrrrv5'\n/\n&diagnostics\n config_lightning_option=3\n/\n"
    with pytest.raises(ValueError, match="not available on the RRFS fork route"):
        resolve_rrfs(text)


def _layout(extra):
    from woof.hex.mpasphys.framework_backend_v842 import registry_source
    from woof.hex.mpasphys.pool_v842 import RegistryLayout
    text = _namelist(extra)
    config = resolve_namelist(text, default_backend="hex-cuda")
    dims = dict(nCells=10, nEdges=30, nVertices=20, nVertLevels=12, maxEdges=6, maxEdges2=12,
                vertexDegree=3, nSoilLevels=4, nOznLevels=59, nAerLevels=1, nMonths=12)
    return RegistryLayout(registry_source(), config, text, dims)


def test_pools_and_streams_only_with_the_option(tmp_path):
    from woof.hex.mpasphys.forecast_v842 import read_streams, write_default_streams
    from woof.hex.streams_v842 import _all_packages, field_definition
    off, lpi, pr92 = _layout(None), _layout({"config_lightning_option": 3}), _layout({"config_lightning_option": 1})
    rows = lambda layout: {r["name"]: r for r in layout.fields if r["pool"] == "diag"}
    assert not set(ltng.NAMES) & set(rows(off))
    assert set(ltng.NAMES) <= set(rows(lpi))
    assert set(ltng.FLASH_NAMES) <= set(rows(pr92)) and "lpi" not in rows(pr92)
    assert [r for r in lpi.fields if r["name"] not in ltng.NAMES] == list(off.fields)
    assert "ltng_lpi" in lpi.packages and not set(ltng.PACKAGES.values()) & off.packages
    streams = {}
    for label, layout in (("off", off), ("lpi", lpi)):
        write_default_streams(layout, tmp_path / f"streams.{label}")
        streams[label] = {s.name: s for s in read_streams(tmp_path / f"streams.{label}", layout)}
    assert set(ltng.NAMES) <= set(streams["lpi"]["diagnostics"].fields)
    assert [f for f in streams["lpi"]["diagnostics"].fields if f not in ltng.NAMES] == \
        list(streams["off"]["diagnostics"].fields)
    assert field_definition("lpi")["gates"] == ["ltng_lpi"]
    assert field_definition("ic_flashcount")["attrs"]["units"] == "#"
    assert set(ltng.PACKAGES.values()) <= _all_packages()


def test_the_off_state_registry_streams_and_receipt_are_unchanged():
    """The digests test_mpas_afwa_diagnostics pins for the off state (recorded on origin/main fe554cdb2)."""
    import hashlib
    import json
    import tempfile
    from pathlib import Path
    from woof.hex.mpasphys.forecast_v842 import write_default_streams
    layout = _layout(None)
    digest = hashlib.sha256(json.dumps([layout.fields, sorted(layout.packages), sorted(layout.dimensions.items())],
                                       sort_keys=True, default=str).encode()).hexdigest()[:16]
    assert digest == "91afd1d2f4968e3f"
    with tempfile.TemporaryDirectory() as directory:
        write_default_streams(layout, Path(directory) / "streams")
        assert hashlib.sha256((Path(directory) / "streams").read_bytes()).hexdigest()[:16] == "b868bf60f2a107f7"
    assert hashlib.sha256(json.dumps(_resolve().receipt(), sort_keys=True, default=str).encode()).hexdigest()[:16] == "bb186b77c0898a2c"


# ---------------------------------------------------------------------------
# synthetic convective mesh columns
# ---------------------------------------------------------------------------
@pytest.fixture
def cp():
    return pytest.importorskip("cupy")


def _columns(seed=5, nl=NL, nc=NC):
    rng = np.random.default_rng(seed)
    f = np.float32
    ht = rng.uniform(0, 600, nc).astype(f)
    depth = np.linspace(0, 16000, nl + 1)[:, None] * rng.uniform(0.95, 1.05, nc)[None, :]
    zgrid = (ht[None, :] + depth).astype(f)
    zmid = 0.5 * (zgrid[1:] + zgrid[:-1]) - ht[None, :]
    temperature = 300.0 - 0.0068 * zmid + rng.normal(0, 0.3, (nl, nc))
    exner = np.exp(-zmid / 26000.0).astype(f)
    theta = (temperature / exner).astype(f)
    storm = np.zeros(nc, bool)
    storm[[2, 3, 4, 8]] = True
    profile = np.sin(np.pi * np.clip((np.vstack([zgrid[:1], zgrid[1:]]) - ht) / 13000.0, 0, 1))
    w = (np.where(storm, rng.uniform(15, 35, nc), rng.uniform(0, 2, nc))[None, :] * profile).astype(f)
    mid = np.exp(-((zmid - 6000.0) / 2500.0) ** 2)
    q = lambda scale: (np.where(storm, scale, scale * 0.05)[None, :] * mid * rng.uniform(0.6, 1.4, (nl, nc))).astype(f)
    species = dict(qv=(0.016 * np.exp(-zmid / 2500.0)).astype(f), qc=q(1.5e-3), qr=q(1.0e-3), qi=q(5e-4),
                   qs=q(1.2e-3), qg=q(3e-3), nr=q(1e4))
    refl = np.where(storm[None, :], 10 + 45 * mid, -10 + 20 * mid).astype(f)
    lat = np.deg2rad(rng.uniform(30, 48, nc)).astype(f)
    lon = np.deg2rad(np.where(np.arange(nc) % 2, rng.uniform(250, 280, nc), rng.uniform(-110, -95, nc))).astype(f)
    return dict(zgrid=zgrid, theta=theta, exner=exner, w=w, species=species, refl=refl, lat=lat, lon=lon,
                xland=np.where(np.arange(nc) % 5 == 0, 2, 1).astype(f), area=rng.uniform(1e7, 1.5e7, nc).astype(f))


def _owner(cp, cols, scheme="mp_thompson", nl=NL, nc=NC):
    pad = lambda a: cp.asarray(np.concatenate([a, a[..., :1]], axis=-1))
    fields = {("mesh", "zgrid"): pad(cols["zgrid"]), ("diag", "theta"): pad(cols["theta"]),
              ("diag", "exner"): pad(cols["exner"]), ("state", "w"): pad(cols["w"]),
              ("sfc_input", "xland"): pad(cols["xland"]), ("mesh", "latCell"): pad(cols["lat"]),
              ("mesh", "lonCell"): pad(cols["lon"]), ("mesh", "areaCell"): pad(cols["area"]),
              ("diag", "pressure_base"): pad(np.full((nl, nc), 5e4, np.float32)),
              ("diag", "pressure_p"): pad(np.zeros((nl, nc), np.float32))}
    names = list(cols["species"])
    fields["state", "scalars"] = cp.asarray(np.stack([np.concatenate([cols["species"][n], cols["species"][n][:, :1]],
                                                                     axis=-1) for n in names]))
    indices = {("state", "index_" + n): i + 1 for i, n in enumerate(names)}
    session = SimpleNamespace(layout=SimpleNamespace(dimensions=dict(nCellsSolve=nc, nVertLevels=nl), indices=indices),
                              config=SimpleNamespace(execution_options={"config_microp_scheme": scheme}))
    buffers = {}

    def buffer(pool, name):
        return buffers.setdefault((pool, name), cp.zeros(nc + 1, dtype=cp.float32))

    return SimpleNamespace(cp=cp, session=session, authority=lambda pool, name: fields[pool, name], buffer=buffer,
                           buffers=buffers, dycore=SimpleNamespace(config=SimpleNamespace(config_dt=DT)), halo=None)


class _Utils:
    def __init__(self, written=()):
        self.written = set(written)

    def stream_inclusion_count(self, name, direction=None, **kw):
        return 1

    def field_will_be_written(self, name):
        return name in self.written


def _wrf_columns(cols, refl_on):
    """WRF's lightning column (woof.core.lightning.update_lightning) of the same cells, built with numpy."""
    f = np.float32
    nl, nc = cols["theta"].shape

    def extend(value, top=None):
        last = value[-1:] if top is None else np.full((1, nc), top, f)
        return np.ascontiguousarray(np.concatenate([value, last]).reshape(nl + 1, 1, nc), dtype=f)

    z = cols["zgrid"]
    lon = np.degrees(cols["lon"].astype(np.float64))
    lon = np.where(lon > 180.0, lon - 360.0, lon)
    out = dict(t=extend(cols["theta"] * cols["exner"]), theta=extend(cols["theta"]), pi=extend(cols["exner"]),
               z=extend((f(0.5) * (z[:-1] + z[1:])).astype(f)), dz=extend((z[1:] - z[:-1]).astype(f), 0.0),
               w=cols["w"].reshape(nl + 1, 1, nc), ht=z[0].reshape(1, nc), land=cols["xland"].reshape(1, nc),
               lat=np.degrees(cols["lat"].astype(np.float64)).astype(f).reshape(1, nc),
               lon=lon.astype(f).reshape(1, nc), top=np.ones((1, nc), np.int32),
               inputnum=np.zeros((12, 1, nc), f), inputden=np.zeros((12, 1, nc), f),
               refl=extend(cols["refl"], -35.0) if refl_on else np.zeros((nl + 1, 1, nc), f))
    for name in ltng.SPECIES:
        out[name] = extend(cols["species"][name], 0.0)
    return out


def test_the_mesh_columns_are_wrfs_columns(cp):
    cols = _columns()
    module = ltng.LightningModule(_owner(cp, cols), ltng.LightningConfig(option=2))
    module.setup(_Utils())
    module.reflectivity = lambda theta, zgrid: cp.asarray(cols["refl"])
    got = module.columns()
    want = _wrf_columns(cols, True)
    assert set(got) == set(want)
    for name, value in want.items():
        np.testing.assert_array_equal(cp.asnumpy(got[name]), value, err_msg=name)


@pytest.mark.parametrize("config", [
    ltng.LightningConfig(option=1),
    ltng.LightningConfig(option=2, iccg_method=3, lightning_dt=40.0, start_seconds=20.0),
    ltng.LightningConfig(option=1, iccg_method=1, iccg_prescribed_num=2.0, iccg_prescribed_den=1.0,
                         flashrate_factor=0.5, cellcount_method=2),
    ltng.LightningConfig(option=3, lightning_dt=60.0),
])
def test_every_word_equals_the_wrf_routes_launcher_over_the_steps(cp, config):
    """Five steps of the module against launch_lightning + next_activation (the WRF route's driver) on WRF's
    columns of the same cells, word for word, including the steps the clock skips."""
    from woof.core.lightning import launch_lightning, next_activation
    cols = _columns()
    module = ltng.LightningModule(_owner(cp, cols), config)
    utils = _Utils(ltng.NAMES)
    module.setup(utils)
    module.reflectivity = lambda theta, zgrid: cp.asarray(cols["refl"])
    names = config.names
    want = {name: cp.zeros((1, NC), dtype=cp.float32) for name in ltng.NAMES}
    fields = {k: cp.asarray(v) for k, v in _wrf_columns(cols, config.option in (1, 2)).items()}
    interval = config.lightning_dt if config.lightning_dt > 0 else DT
    activation, fired = float(np.float32(config.start_seconds)), 0
    spacing = float(np.sqrt(np.mean(cols["area"].astype(np.float64))))
    for step in range(1, 6):
        module.begin(step, None)
        module.update()
        following = next_activation(activation, (step - 1) * DT, DT, interval)
        if following is not None:
            launch_lightning(fields, want, option=config.option, method=config.iccg_method,
                             cellcount_method=config.cellcount_method, dx=spacing, dy=spacing, interval=interval,
                             factor=config.flashrate_factor, num=config.iccg_prescribed_num,
                             den=config.iccg_prescribed_den, adjust=0.0, month=1)
            activation, fired = following, fired + 1
        assert module.activation == activation
        for name in names:
            np.testing.assert_array_equal(cp.asnumpy(module.out[name]).view(np.uint32),
                                          cp.asnumpy(want[name]).view(np.uint32), err_msg=f"{name} step {step}")
    assert module.activations == fired and 1 <= fired <= 5
    module.compute()
    for name in names:
        np.testing.assert_array_equal(cp.asnumpy(module.owner.buffers["diag", name][:NC]), cp.asnumpy(want[name][0]))
    if config.option == 3:
        assert (cp.asnumpy(want["lpi"]) > 0).any()
    else:
        assert (cp.asnumpy(want["ic_flashcount"]) > 0).any() and (cp.asnumpy(want["cg_flashcount"]) > 0).any()
        assert not cp.asnumpy(want["lpi"]).any()


@pytest.mark.parametrize("config, checkpoint", [
    (ltng.LightningConfig(option=1, lightning_dt=40.0), 3),
    (ltng.LightningConfig(option=2, lightning_dt=60.0, start_seconds=20.0), 4),
    (ltng.LightningConfig(option=3), 5),
])
def test_a_checkpoint_resumes_the_counts_and_the_clock(cp, config, checkpoint):
    """A continuation counts its loop steps from 0 again (runtime_v842) with the straight run's steps before it as
    the owner's step_offset, as physics_clock_offset hands physics: every activation and word equals the straight
    run's.  THE BREAKAGE: the window was formed from the local step, so the restored activation (the straight run's
    clock) lay past it and lightning stopped updating after a resume."""
    cols = _columns()

    def module(offset=0):
        owner = _owner(cp, cols)
        owner.step_offset = offset
        m = ltng.LightningModule(owner, config)
        m.setup(_Utils(ltng.NAMES))
        m.reflectivity = lambda theta, zgrid: cp.asarray(cols["refl"])
        return m

    straight, first = module(), module()
    for step in range(1, checkpoint + 1):
        for m in (straight, first):
            m.begin(step, None)
            m.update()
    resumed = module(offset=checkpoint)
    resumed.begin(0, None)
    resumed.resume(first.state())
    before = straight.activations
    for local in range(1, 7):
        straight.begin(checkpoint + local, None)
        resumed.begin(local, None)
        for m in (straight, resumed):
            m.update()
        assert resumed.activation == straight.activation, local
        for name in config.names:
            np.testing.assert_array_equal(cp.asnumpy(resumed.out[name]).view(np.uint32),
                                          cp.asnumpy(straight.out[name]).view(np.uint32), f"{name} step {local}")
    assert resumed.activations == straight.activations - before > 0


def test_the_runtime_hands_the_continuation_offset_to_the_diagnostics():
    """runtime_v842 sets the diagnostics' step_offset from the same physics_clock_offset as physics' begin_step;
    a fresh run's stays 0."""
    import inspect
    from woof.hex.mpasphys import runtime_v842
    from woof.hex.mpasphys.resident_output_diagnostics_v842 import ResidentOutputDiagnostics
    assert ResidentOutputDiagnostics.step_offset == 0
    source = inspect.getsource(runtime_v842._run_native_dynamics)
    assert "diagnostics.step_offset = physics_step_offset" in source
    assert source.index("physics_step_offset,physics_seconds_offset=clock_offset(dt)") < \
        source.index("diagnostics.step_offset = physics_step_offset")


def test_lightning_is_refused_on_a_multi_rank_mesh(cp):
    owner = _owner(cp, _columns())
    owner.halo = object()
    with pytest.raises(ValueError, match="multi-rank"):
        ltng.LightningModule(owner, ltng.LightningConfig(option=3)).setup(_Utils())
