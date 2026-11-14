"""WRF-HAILCAST on the MPAS suite route (woof.hex.mpasdiag.hailcast_diag_v842).

&diagnostics config_hailcast_opt (with config_haildt and config_hailcast_version) adds the WRF route's HAILCAST
fields as ``diag`` fields.  Here:

* the options resolve (off by default) and are refused when inconsistent and on the RRFS fork route; the Registry,
  default streams and driver receipt carry the fields only with the option (the off state is pinned by digest);
* over 48 steps of real convective columns (tests/data/oracles/hailcast/real_columns.npz, the columns the WRF route's
  stone kernels are graded on against WRF's and HRRR's Fortran) on a ring of cells, every word the module keeps (the
  updraft mask and duration, the five embryos, the max/mean/std and the run maximum) equals what the WRF route's own
  woof.core.hail.update keeps on a periodic one-row grid of the same columns (whose 3x3 box is that ring), for both
  stone models, and stones land;
* the WRF route's duration kernel on a one-row, non-periodic grid zeroes every duration (why HAILCAST was inert on the
  WOOF column route), and the mesh duration does not;
* the interval fields restart after a written frame, and everything rides a checkpoint.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from woof.hex.mpasdiag import hailcast_diag_v842 as hc
from woof.hex.mpasphys import resolve_namelist

DT = 20.0
ROOT = Path(__file__).resolve().parents[3]


def _namelist(extra=None, physics=()):
    lines = ["&physics", " config_physics_suite='convection_permitting'", *physics, "/"]
    if extra is not None:
        lines += ["&diagnostics", *[f" {k}={v}" for k, v in extra.items()], "/"]
    return "\n".join(lines) + "\n"


def _resolve(extra=None):
    return resolve_namelist(_namelist(extra), default_backend="hex-cuda")


def test_options_default_off_and_resolve():
    assert _resolve().hailcast_diagnostics is None and "hailcast_diagnostics" not in _resolve().receipt()
    config = _resolve({"config_hailcast_opt": ".true."})
    assert config.hailcast_diagnostics == hc.HailcastConfig(0.0, "wrf_461")
    assert hc.packages(config) == {"hailcast"}
    assert config.receipt()["hailcast_diagnostics"] == {"haildt": 0.0, "version": "wrf_461"}
    other = _resolve({"config_hailcast_opt": 1, "config_haildt": 60, "config_hailcast_version": "'wrf_39_noaa'"})
    assert other.hailcast_diagnostics == hc.HailcastConfig(60.0, "wrf_39_noaa")
    for key in hc.OPTIONS:
        assert key not in other.options


@pytest.mark.parametrize("extra, message", [
    ({"config_haildt": 30.0}, "while config_hailcast_opt is off"),
    ({"config_hailcast_opt": ".true.", "config_haildt": 90.0}, "between 0 \\(every step\\) and 60"),
    ({"config_hailcast_opt": ".true.", "config_hailcast_version": "'wrf_99'"}, "is one of"),
    ({"config_hailcast_opt": 2}, "0/1"),
])
def test_inconsistent_options_are_refused(extra, message):
    with pytest.raises(ValueError, match=message):
        _resolve(extra)


def test_rrfs_fork_route_refuses_the_options():
    from woof.hex.mpasphys.rrfs.driver_rrfs import resolve_namelist as resolve_rrfs
    text = "&physics\n config_physics_suite='hrrrv5'\n/\n&diagnostics\n config_hailcast_opt=.true.\n/\n"
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
    off, on = _layout(None), _layout({"config_hailcast_opt": ".true."})
    rows = lambda layout: {r["name"]: r for r in layout.fields if r["pool"] == "diag"}
    assert not set(hc.NAMES) & set(rows(off)) and set(hc.NAMES) <= set(rows(on))
    assert [r for r in on.fields if r["name"] not in hc.NAMES] == list(off.fields)
    streams = {}
    for label, layout in (("off", off), ("on", on)):
        write_default_streams(layout, tmp_path / f"streams.{label}")
        streams[label] = {s.name: s for s in read_streams(tmp_path / f"streams.{label}", layout)}
    assert set(hc.NAMES) <= set(streams["on"]["diagnostics"].fields)
    assert [f for f in streams["on"]["diagnostics"].fields if f not in hc.NAMES] == \
        list(streams["off"]["diagnostics"].fields)
    d = field_definition("hailcast_diam_max")
    assert d["attrs"]["units"] == "mm" and d["attrs"]["wrf_name"] == "HAILCAST_DIAM_MAX" and d["gates"] == ["hailcast"]
    assert "hailcast" in _all_packages()


def test_the_off_state_registry_streams_and_receipt_are_unchanged():
    """The digests test_mpas_afwa_diagnostics pins for the off state (recorded on origin/main fe554cdb2)."""
    import hashlib
    import json
    import tempfile
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
# real convective columns on a ring of cells
# ---------------------------------------------------------------------------
@pytest.fixture
def cp():
    return pytest.importorskip("cupy")


def _real_columns(nc=16):
    """The first ``nc`` real columns as the MPAS pools hold them: theta, zgrid faces, rho_zz (zz = 1), pressure."""
    data = np.load(ROOT / "tests/data/oracles/hailcast/real_columns.npz")
    prof = data["profiles"][:, :nc].astype(np.float32)          # [field, column, level]
    t, z, p, rho, qv, qi, qc, qr, qs, qg, w = (np.ascontiguousarray(prof[i].T) for i in range(11))
    ht = data["ht"][:nc].astype(np.float32)
    nl = t.shape[0] - 1                                           # w is on the lower 50 faces: 49 layers, 50 faces
    t, z, p, rho, qv, qi, qc, qr, qs, qg = (a[:nl] for a in (t, z, p, rho, qv, qi, qc, qr, qs, qg))
    zgrid = np.empty((nl + 1, nc), np.float32)
    zgrid[0] = ht
    for k in range(nl):
        zgrid[k + 1] = 2 * z[k] - zgrid[k]
    theta = (t / (p / np.float32(1e5)) ** np.float32(287.0 / 1004.0)).astype(np.float32)
    return dict(theta=theta, zgrid=zgrid, p=p, rho_zz=(rho / (1 + qv)).astype(np.float32), w=w[:nl + 1],
                species=dict(qv=qv, qi=qi, qc=qc, qr=qr, qs=qs, qg=qg), nl=nl, nc=nc)


def _ring(nc):
    index = np.arange(nc)
    cells = np.full((3, nc + 1), nc + 1, np.int32)                # a third slot beyond nEdgesOnCell
    cells[0, :nc], cells[1, :nc] = (index - 1) % nc + 1, (index + 1) % nc + 1
    return cells, np.full(nc + 1, 2, np.int32)


def _owner(cp, cols):
    nl, nc = cols["nl"], cols["nc"]
    pad = lambda a: cp.asarray(np.concatenate([a, a[..., :1]], axis=-1))
    cells, edges = _ring(nc)
    fields = {("mesh", "zgrid"): pad(cols["zgrid"]), ("mesh", "zz"): pad(np.ones((nl, nc), np.float32)),
              ("state", "rho_zz"): pad(cols["rho_zz"]), ("diag", "theta"): pad(cols["theta"]),
              ("diag", "pressure_base"): pad(cols["p"]), ("diag", "pressure_p"): pad(np.zeros((nl, nc), np.float32)),
              ("state", "w"): pad(cols["w"]), ("mesh", "cellsOnCell"): cp.asarray(cells),
              ("mesh", "nEdgesOnCell"): cp.asarray(edges)}
    names = list(cols["species"])
    fields["state", "scalars"] = cp.asarray(np.stack([np.concatenate([cols["species"][n], cols["species"][n][:, :1]],
                                                                     axis=-1) for n in names]))
    indices = {("state", "index_" + n): i + 1 for i, n in enumerate(names)}
    session = SimpleNamespace(layout=SimpleNamespace(dimensions=dict(nCellsSolve=nc, nVertLevels=nl), indices=indices))
    buffers = {}
    return SimpleNamespace(cp=cp, session=session, authority=lambda pool, name: fields[pool, name], buffers=buffers,
                           buffer=lambda pool, name: buffers.setdefault((pool, name), cp.zeros(nc + 1, cp.float32)),
                           dycore=SimpleNamespace(config=SimpleNamespace(config_dt=DT)), halo=None)


class _Utils:
    def __init__(self, written=()):
        self.written = set(written)

    def stream_inclusion_count(self, name, direction=None, **kw):
        return 1

    def field_will_be_written(self, name):
        return name in self.written


class _WrfState:
    """The WRF route's DomainState as woof.core.hail.update reads it: (nz, 1, nCells) arrays, PHB = 9.81 * zgrid."""

    def __init__(self, cp, cols):
        nl, nc = cols["nl"], cols["nc"]
        r = lambda a: cp.asarray(np.ascontiguousarray(a.reshape(a.shape[0], 1, nc)))
        self.slots = {}
        self.cp = cp
        self.p = r(cols["p"])
        self._theta = r(cols["theta"])
        rho_dry = cols["rho_zz"] * np.float32(1)
        self.alt = r((np.float32(1) / rho_dry).astype(np.float32))
        self.phb = r((cols["zgrid"] * np.float32(9.81)).astype(np.float32))
        self.php = cp.zeros_like(self.phb)
        self.w = r(cols["w"])
        self.ht = cp.asarray(cols["zgrid"][0].reshape(1, nc))
        for name, value in cols["species"].items():
            setattr(self, name, r(value))
        self.elapsed_seconds = 0.0

    def total_theta(self):
        return self._theta

    def scratch(self, shape, slot):
        if slot not in self.slots:
            self.slots[slot] = self.cp.zeros(shape, dtype=self.cp.float32)
        return self.slots[slot]


def _cfg(cols, version, haildt):
    return SimpleNamespace(gsd_diagnostics=0, hailcast_opt=1, dx=3000.0, dy=3000.0, mp_physics=1, nz=cols["nl"], ny=1,
                           nx=cols["nc"], dt=DT, open_x=False, open_y=False, specified=False, nested=False,
                           haildt=haildt, hailcast_version=version)


CHECKED = (*hc.NAMES, "hailcast_wup_mask", "hailcast_wdur")


@pytest.mark.parametrize("version, haildt", [("wrf_461", 0.0), ("wrf_39_noaa", 60.0)])
def test_every_word_equals_the_wrf_routes_hail_update_on_a_ring(cp, version, haildt):
    from woof.core import hail
    cols = _real_columns()
    module = hc.HailcastModule(_owner(cp, cols), hc.HailcastConfig(haildt, version))
    module.setup(_Utils())
    state, cfg = _WrfState(cp, cols), _cfg(cols, version, haildt)
    hail.initialize(state, cfg)
    for step in range(1, 49):
        module.begin(step, None)
        module.update()
        hail.update(state, cfg)
        state.elapsed_seconds += DT
        for name in CHECKED:
            np.testing.assert_array_equal(cp.asnumpy(module.w[name]).view(np.uint32),
                                          cp.asnumpy(state.slots[name]).reshape(-1).view(np.uint32),
                                          err_msg=f"{name} step {step}")
    assert module.calls == (48 if haildt == 0 else 17)      # the first step, then every 60 s crossing
    assert (cp.asnumpy(module.w["hailcast_wdur"]) > hc.DURATION_S).any()
    assert (cp.asnumpy(module.w["hailcast_diam_max"]) > 5).any()      # stones land


def test_the_wrf_routes_duration_kernel_is_inert_on_a_one_row_grid_and_the_mesh_one_is_not(cp):
    from woof.core.kernels import get_kernel
    nc = 8
    mask = cp.ones(nc, dtype=cp.float32)
    dur = cp.zeros(nc, dtype=cp.float32)
    out = cp.zeros(nc, dtype=cp.float32)
    for _ in range(3):
        get_kernel("hail", "hail_duration")(((nc + 127) // 128,), (128,),
                                            (mask, mask, dur, out, np.int32(nc), np.int32(1), np.float32(DT),
                                             np.int32(0), np.int32(0)))
        dur = out.copy()
    assert not cp.asnumpy(dur).any()
    cols = _real_columns(nc)
    module = hc.HailcastModule(_owner(cp, cols), hc.HailcastConfig())
    module.setup(_Utils())
    for _ in range(3):
        module.w["hailcast_wdur"][...] = module.duration(mask)
        module.w["hailcast_wup_mask"][...] = mask
    assert (cp.asnumpy(module.w["hailcast_wdur"]) == np.float32(3 * DT)).all()


def test_the_interval_fields_restart_after_a_frame_and_ride_a_checkpoint(cp):
    cols = _real_columns()
    config = hc.HailcastConfig(0.0, "wrf_461")

    def module(written=()):
        m = hc.HailcastModule(_owner(cp, cols), config)
        m.setup(_Utils(written))
        return m

    straight = module(hc.NAMES)
    for step in range(1, 48):
        straight.begin(step, None)
        straight.update()
    assert (cp.asnumpy(straight.w["hailcast_diam_max"]) > 0).any()
    straight.compute()
    np.testing.assert_array_equal(cp.asnumpy(straight.owner.buffers["diag", "hailcast_diam_max"][:cols["nc"]]),
                                  cp.asnumpy(straight.w["hailcast_diam_max"]))
    straight.reset()
    checkpoint = straight.state()
    resumed = module(hc.NAMES)
    resumed.resume(checkpoint)
    runmax = cp.asnumpy(straight.w["hailcast_runmax"]).copy()
    for m in (straight, resumed):
        m.begin(48, None)
        m.update()
    for name in CHECKED:
        np.testing.assert_array_equal(cp.asnumpy(resumed.w[name]).view(np.uint32),
                                      cp.asnumpy(straight.w[name]).view(np.uint32), name)
    # the window restarted (this step's own stones only); the run maximum did not
    assert (cp.asnumpy(straight.w["hailcast_runmax"]) >= runmax).all()


def test_hailcast_is_refused_on_a_multi_rank_mesh(cp):
    owner = _owner(cp, _real_columns(4))
    owner.halo = object()
    with pytest.raises(ValueError, match="multi-rank"):
        hc.HailcastModule(owner, hc.HailcastConfig()).setup(_Utils())
