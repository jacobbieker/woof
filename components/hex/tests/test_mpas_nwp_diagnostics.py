"""WRF's nwp_diagnostics maxima on the MPAS suite route (woof.hex.mpasdiag.nwp_diag_v842).

&diagnostics config_nwp_diagnostics = .true. adds T2MAX/T2MIN, WSPD10MAX, U10MAX/V10MAX, GUST, W_UP_MAX/W_DN_MAX,
GRPL_MAX, REFD_MAX and the PSD hail maxima as ``diag`` fields.  Here:

* the option resolves (off by default), is refused on the RRFS fork route, and the Registry and default streams
  carry the fields only with it, leaving every other row untouched;
* per mesh column, the module's words equal the WRF route's kernels on identical columns: the NWP windows
  (nwp_windows.cu replay entry, bitwise), the gust and 10 m wind peak (the NumPy mirror pinned to wind_gust_diag.cu),
  the PSD hail leaf (nwp_hail.py) and the CLWRF T2 extremes (clwrf.cu);
* a maximum is zeroed after the frame that carries it and its next interval starts clean, a maximum no due
  stream carries is left alone, and a checkpoint's accumulators resume.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from woof.hex.mpasdiag import nwp_diag_v842 as nwp
from woof.hex.mpasphys import resolve_namelist

NL, NC = 24, 15


def _namelist(on=True):
    lines = ["&physics", " config_physics_suite='convection_permitting'", "/"]
    if on is not None:
        lines += ["&diagnostics", f" config_nwp_diagnostics={'.true.' if on else '.false.'}", "/"]
    return "\n".join(lines) + "\n"


def test_option_defaults_off_and_resolves():
    assert not resolve_namelist(_namelist(None), default_backend="hex-cuda").nwp_diagnostics
    assert not resolve_namelist(_namelist(False), default_backend="hex-cuda").nwp_diagnostics
    config = resolve_namelist(_namelist(True), default_backend="hex-cuda")
    assert config.nwp_diagnostics and config.receipt()["nwp_diagnostics"] is True
    assert nwp.OPTION not in config.options
    assert nwp.packages(config) == {nwp.PACKAGE}
    assert "nwp_diagnostics" not in resolve_namelist(_namelist(None), default_backend="hex-cuda").receipt()
    with pytest.raises(ValueError, match="has no option"):
        resolve_namelist(_namelist(None) + "&diagnostics\n config_nwp_diags=1\n/\n", default_backend="hex-cuda")
    # both WOOF options live in the one group
    both = _namelist(True).replace("config_nwp_diagnostics=.true.",
                                   "config_nwp_diagnostics=.true.\n config_solar_diagnostics=.false.")
    assert resolve_namelist(both, default_backend="hex-cuda").nwp_diagnostics


def test_rrfs_fork_route_refuses_the_option():
    from woof.hex.mpasphys.rrfs.driver_rrfs import resolve_namelist as resolve_rrfs
    text = "&physics\n config_physics_suite='hrrrv5'\n/\n&diagnostics\n config_nwp_diagnostics=.true.\n/\n"
    with pytest.raises(ValueError, match="not available on the RRFS fork route"):
        resolve_rrfs(text)


def _layout(on):
    from woof.hex.mpasphys.framework_backend_v842 import registry_source
    from woof.hex.mpasphys.pool_v842 import RegistryLayout
    text = _namelist(on)
    config = resolve_namelist(text, default_backend="hex-cuda")
    dims = dict(nCells=10, nEdges=30, nVertices=20, nVertLevels=12, maxEdges=6, maxEdges2=12,
                vertexDegree=3, nSoilLevels=4, nOznLevels=59, nAerLevels=1, nMonths=12)
    return RegistryLayout(registry_source(), config, text, dims)


def test_pools_and_streams_only_with_the_option(tmp_path):
    from woof.hex.mpasphys.forecast_v842 import read_streams, write_default_streams
    from woof.hex.streams_v842 import field_definition, _all_packages
    off, on = _layout(False), _layout(True)
    rows = lambda layout: {r["name"]: r for r in layout.fields if r["pool"] == "diag"}
    assert not set(nwp.NAMES) & set(rows(off))
    assert set(nwp.NAMES) <= set(rows(on))
    assert [r for r in on.fields if r["name"] not in nwp.NAMES] == list(off.fields)
    assert nwp.PACKAGE in on.packages and nwp.PACKAGE not in off.packages
    for name in nwp.NAMES:
        assert rows(on)[name]["dimensions"] == ["nCells"] and rows(on)[name]["shape"] == [11]
    streams = {}
    for label, layout in (("off", off), ("on", on)):
        write_default_streams(layout, tmp_path / f"streams.{label}")
        streams[label] = {s.name: s for s in read_streams(tmp_path / f"streams.{label}", layout)}
    assert set(nwp.NAMES) <= set(streams["on"]["diagnostics"].fields)
    assert not set(nwp.NAMES) & set(streams["off"]["diagnostics"].fields)
    # every stream other than diagnostics, and the diagnostics stream's other words, are unchanged
    assert {n: s.fields for n, s in streams["on"].items() if n != "diagnostics"} == \
           {n: s.fields for n, s in streams["off"].items() if n != "diagnostics"}
    assert [f for f in streams["on"]["diagnostics"].fields if f not in nwp.NAMES] == \
           list(streams["off"]["diagnostics"].fields)
    d = field_definition("gust")
    assert d["attrs"]["units"] == "m s-1" and d["attrs"]["wrf_name"] == "GUST"
    assert nwp.PACKAGE in _all_packages()


# ---------------------------------------------------------------------------
# stand-in owner: the pools the module reads, as (level, cell) device arrays
# ---------------------------------------------------------------------------
def _columns(seed=7, nl=NL, nc=NC):
    rng = np.random.default_rng(seed)
    f = np.float32
    z = (np.linspace(0, 16000, nl + 1, dtype=f)[:, None]
         + rng.uniform(0, 600, nc).astype(f)[None, :] * np.linspace(1, 0, nl + 1, dtype=f)[:, None])
    p = np.linspace(97000, 9000, nl, dtype=f)[:, None] * rng.uniform(0.97, 1.03, nc).astype(f)[None, :]
    zm = 0.5 * (z[:-1] + z[1:])
    eta = np.linspace(0, 1, nl + 1, dtype=f)
    fzm = np.zeros(nl, f)
    fzm[1:] = (eta[1:nl] - 0.5 * (eta[:-1][:nl - 1] + eta[1:][:nl - 1])) / (
        0.5 * (eta[1:nl] + eta[2:nl + 1]) - 0.5 * (eta[:nl - 1] + eta[1:nl]))
    fzp = (f(1) - fzm).astype(f)
    return dict(
        zgrid=z.astype(f), p=p.astype(f), fzm=fzm.astype(f), fzp=fzp,
        rho_zz=(p / f(287.0) / f(270.0)).astype(f), zz=np.full((nl, nc), 1.0, f),
        qv=rng.uniform(0, 0.015, (nl, nc)).astype(f), qg=rng.uniform(0, 2e-3, (nl, nc)).astype(f),
        qr=rng.uniform(0, 1e-3, (nl, nc)).astype(f), nr=rng.uniform(1e3, 1e6, (nl, nc)).astype(f),
        zm=zm.astype(f))


def _step_inputs(rng, nl=NL, nc=NC):
    f = np.float32
    return dict(
        w=rng.uniform(-6, 25, (nl + 1, nc)).astype(f),
        u=rng.uniform(-20, 30, (nl, nc)).astype(f), v=rng.uniform(-25, 25, (nl, nc)).astype(f),
        u10=rng.uniform(-15, 15, nc).astype(f), v10=rng.uniform(-15, 15, nc).astype(f),
        t2m=rng.uniform(260, 310, nc).astype(f), hpbl=rng.uniform(0, 4000, nc).astype(f),
        refl=rng.uniform(-30, 60, (nl, nc)).astype(f))


class _Utils:
    def __init__(self):
        self.written = set()

    def stream_inclusion_count(self, name, direction=None, **kw):
        return 1

    def field_will_be_written(self, name):
        return name in self.written


def _owner(cp, col, mp="mp_thompson", scalars=("qv", "qr", "qg", "nr")):
    fields = {}
    buffers = {}

    def put(pool, name, value):
        fields[pool, name] = cp.asarray(value)

    for name in ("zgrid", "fzm", "fzp", "zz"):
        put("mesh", name, col[name])
    put("state", "rho_zz", col["rho_zz"])
    put("diag", "pressure_base", col["p"] - np.float32(37.0))
    put("diag", "pressure_p", np.full_like(col["p"], 37.0))
    layout_fields = [dict(pool=p, pool_name=n) for p, n in
                     (("diag_physics", "u10"), ("diag_physics", "v10"), ("diag_physics", "t2m"),
                      ("diag_physics", "hpbl"), ("diag_physics", "refl10cm"))]
    names = list(scalars)
    stack = cp.asarray(np.stack([col[n] for n in names]))
    fields["state", "scalars"] = stack
    indices = {("state", "index_" + n): i + 1 for i, n in enumerate(names)}
    session = SimpleNamespace(
        layout=SimpleNamespace(dimensions=dict(nCellsSolve=NC, nVertLevels=NL), fields=layout_fields,
                               indices=indices),
        config=SimpleNamespace(execution_options={"config_microp_scheme": mp}))

    def buffer(pool, name):
        return buffers.setdefault((pool, name), cp.zeros(NC + 1, dtype=cp.float32))

    owner = SimpleNamespace(cp=cp, session=session, authority=lambda pool, name: fields[pool, name],
                            buffer=buffer, fields=fields, buffers=buffers, put=put)
    return owner


def _feed(owner, inputs):
    cp = owner.cp
    # a (level, cell) native array padded one cell, as the pools hold it
    pad = lambda a: np.concatenate([a, a[..., :1]], axis=-1)
    owner.put("state", "w", pad(inputs["w"]))
    owner.put("diag", "uReconstructZonal", pad(inputs["u"]))
    owner.put("diag", "uReconstructMeridional", pad(inputs["v"]))
    for name in ("u10", "v10", "t2m", "hpbl"):
        owner.put("diag_physics", name, pad(inputs[name]))
    owner.put("diag_physics", "refl10cm", pad(inputs["refl"]))


@pytest.fixture
def cp():
    return pytest.importorskip("cupy")


def _module(cp, mp="mp_thompson", scalars=("qv", "qr", "qg", "nr"), seed=7):
    col = _columns(seed)
    owner = _owner(cp, col, mp, scalars)
    module = nwp.NwpDiagnosticsModule(owner)
    utils = _Utils()
    module.setup(utils)
    return module, owner, utils, col


def _structured(col, inputs, cp):
    """The same columns as a (ny, nx) WRF grid, through the WRF replay entry's own inputs."""
    ny, nx = 3, NC // 3
    f = np.float32
    sh = lambda a: np.ascontiguousarray(a.reshape(a.shape[:-1] + (ny, nx)))
    alt = f(1) / (col["rho_zz"] * col["zz"])
    return dict(
        ny=ny, nx=nx, sh=sh, alt=sh(alt), p=sh(col["p"]), qv=sh(col["qv"]), qg=sh(col["qg"]),
        php=np.zeros((NL + 1, ny, nx), f), phb=sh((col["zgrid"] * f(9.81)).astype(f)),
        fnm=col["fzm"], fnp=col["fzp"])


def test_windows_equal_the_wrf_kernels_on_identical_columns(cp):
    from woof.core.kernels import get_kernel
    from woof.core.nwp_windows import update as wrf_windows, FIELDS
    module, owner, utils, col = _module(cp)
    st = _structured(col, None, cp)
    ny, nx, sh = st["ny"], st["nx"], st["sh"]
    D = cp.asarray
    f = np.float32
    # WRF's own phy_prep words through the replay entry of nwp_prepare
    rho, pw, full_phb = (cp.zeros((NL, ny, nx), cp.float32), cp.zeros((NL + 1, ny, nx), cp.float32),
                         cp.zeros((NL + 1, ny, nx), cp.float32))
    get_kernel("nwp_windows", "nwp_prepare")(((ny * nx + 127) // 128,), (128,),
        (D(st["p"]), D(st["alt"]), D(st["qv"]), D(st["php"]), D(st["phb"]), D(st["fnm"]), D(st["fnp"]),
         rho, pw, full_phb, np.int32(NL), np.int32(ny * nx), np.int32(1)))
    out = cp.zeros((len(FIELDS), ny, nx), cp.float32)
    rng = np.random.default_rng(11)
    utils.written = set()
    for step in range(2, 6):
        inputs = _step_inputs(rng)
        _feed(owner, inputs)
        module.update()
        wrf_windows(D(sh(inputs["w"])), pw, D(np.zeros(NL + 1, f)), D(st["qg"]), rho, D(st["php"]), D(st["phb"]),
                    D(sh(inputs["u10"])), D(sh(inputs["v10"])), D(sh(inputs["refl"])), out,
                    dt=1.0, xtime=0.0, history_minutes=nwp._NO_RESET_MINUTES, step=2)
        for field, name in (("WSPD10MAX", "wspd10max"), ("W_UP_MAX", "w_up_max"), ("W_DN_MAX", "w_dn_max"),
                            ("GRPL_MAX", "grpl_max"), ("REFD_MAX", "refd_max")):
            want = cp.asnumpy(out[FIELDS.index(field)]).reshape(-1)
            got = cp.asnumpy(module.acc[name])
            np.testing.assert_array_equal(got.view(np.uint32), want.view(np.uint32), err_msg=f"{name} step {step}")
    # the interval really accumulated: the extremes are not the last step's words alone
    assert float(cp.asnumpy(module.acc["w_up_max"]).max()) > 0 > float(cp.asnumpy(module.acc["w_dn_max"]).min())
    assert float(cp.asnumpy(module.acc["grpl_max"]).max()) > 0


def test_gust_and_10m_peak_equal_the_numpy_mirror_of_the_wrf_kernel(cp):
    from woof.core import wind_gust_diag as wg
    module, owner, utils, col = _module(cp)
    f = np.float32
    rng = np.random.default_rng(5)
    mirror = {k: [np.zeros((1, 1), f) for _ in range(NC)] for k in
              ("wspd10max", "u10max", "v10max", "w_up_max", "w_dn_max", "gust_max")}
    for step in range(4):
        inputs = _step_inputs(rng)
        _feed(owner, inputs)
        module.update()
        for c in range(NC):
            # one mesh cell is one 1x1 WRF column: its mass-point wind is both faces of the staggered grid
            u = np.repeat(inputs["u"][:, c][:, None, None], 2, axis=2)
            v = np.repeat(inputs["v"][:, c][:, None, None], 2, axis=1)
            ph = np.zeros((NL + 1, 1, 1), f)
            phb = (col["zgrid"][:, c] * f(9.81)).astype(f)[:, None, None]
            wg.mirror_wind_maxima_step_np(
                u, v, inputs["w"][:, c][:, None, None], col["p"][:, c][:, None, None], ph, phb,
                col["fzm"], col["fzp"], inputs["u10"][c].reshape(1, 1), inputs["v10"][c].reshape(1, 1),
                inputs["hpbl"][c].reshape(1, 1), {k: v_[c] for k, v_ in mirror.items()})
    for name, slot in (("u10max", "u10max"), ("v10max", "v10max"), ("gust", "gust_max")):
        want = np.concatenate([a.reshape(-1) for a in mirror[slot]])
        got = cp.asnumpy(module.acc[name])
        np.testing.assert_array_equal(got.view(np.uint32), want.view(np.uint32), err_msg=name)
    assert float(cp.asnumpy(module.acc["gust"]).min()) > 0
    # the gust is at least the 10 m wind it starts from
    assert np.all(cp.asnumpy(module.acc["gust"]) >= cp.asnumpy(module.acc["wspd10max"]) - 1e-4)


@pytest.mark.parametrize("mp,scalars", [("mp_thompson", ("qv", "qr", "qg", "nr")), ("mp_wsm6", ("qv", "qg"))])
def test_hail_equals_the_psd_leaf_on_the_same_columns(cp, mp, scalars):
    from woof.core.nwp_hail import update as psd
    module, owner, utils, col = _module(cp, mp, scalars)
    rng = np.random.default_rng(3)
    inputs = _step_inputs(rng)
    _feed(owner, inputs)
    module.update()
    D = cp.asarray
    hail1, hail2 = cp.zeros((1, NC), cp.float32), cp.zeros((1, NC), cp.float32)
    rho = cp.asnumpy(module.rho).reshape(NL, 1, NC)
    kw = {}
    if mp == "mp_thompson":
        kw = dict(qr=D(col["qr"].reshape(NL, 1, NC)), nr=D(col["nr"].reshape(NL, 1, NC)))
    psd(D(col["qg"].reshape(NL, 1, NC)), D(rho), hail1, hail2, mp_physics=nwp.HAIL_MP[mp], **kw)
    for name, want in (("hail_maxk1", hail1), ("hail_max2d", hail2)):
        np.testing.assert_array_equal(cp.asnumpy(module.acc[name]).view(np.uint32),
                                      cp.asnumpy(want).reshape(-1).view(np.uint32), err_msg=name)
    assert float(cp.asnumpy(module.acc["hail_max2d"]).max()) > 0
    # a scheme without the PSD leaf leaves the hail words at 0
    other, o_owner, _, _ = _module(cp, "mp_kessler", ("qv",))
    _feed(o_owner, inputs)
    other.update()
    assert not cp.asnumpy(other.acc["hail_max2d"]).any()


def test_t2_extremes_equal_the_clwrf_kernel(cp):
    from woof.core import clwrf
    module, owner, utils, col = _module(cp)
    rng = np.random.default_rng(9)
    dt, period = 60.0, 3
    D = lambda a: cp.asarray(np.asarray(a, np.float32))
    out = {n: cp.zeros(NC, cp.float32) for n in clwrf.HISTORY}
    counter = cp.zeros(NC, cp.int32)
    zero = cp.zeros(NC, cp.float32)
    for k in range(period):
        inputs = _step_inputs(rng)
        _feed(owner, inputs)
        module.update()
        clwrf.launch({"t2": D(inputs["t2m"]), "q2": zero, "tsk": D(inputs["t2m"]), "u10": zero, "v10": zero,
                      "raincv": zero, "rainncv": zero}, out, counter, dt=dt, xtime=k * dt / 60.,
                     seconds=k * dt, interval=period * dt / 60.)
    utils.written = {"t2max", "t2min"}
    module.compute()
    for name, slot in (("t2max", "clwrf_t2clmax"), ("t2min", "clwrf_t2clmin")):
        np.testing.assert_array_equal(cp.asnumpy(owner.buffers["diag", name])[:NC].view(np.uint32),
                                      cp.asnumpy(out[slot]).view(np.uint32), err_msg=name)


def test_reset_after_the_carrying_frame_and_resume(cp):
    module, owner, utils, col = _module(cp)
    rng = np.random.default_rng(21)
    first = [_step_inputs(rng) for _ in range(3)]
    second = [_step_inputs(rng) for _ in range(2)]
    run = lambda steps: [(_feed(owner, s), module.update()) for s in steps]
    run(first)
    utils.written = {"wspd10max", "w_up_max", "t2max", "t2min", "gust"}
    module.compute()
    pub = lambda n: cp.asnumpy(owner.buffers["diag", n])[:NC].copy()
    peak = np.max([np.hypot(s["u10"], s["v10"]) for s in first], axis=0)
    np.testing.assert_allclose(pub("wspd10max"), peak, rtol=1e-6)
    np.testing.assert_allclose(pub("t2max"), np.max([s["t2m"] for s in first], axis=0), atol=1e-4)
    np.testing.assert_allclose(pub("t2min"), np.min([s["t2m"] for s in first], axis=0), atol=1e-4)
    keep = cp.asnumpy(module.acc["grpl_max"]).copy()
    assert keep.max() > 0
    checkpoint = module.state()
    module.reset()                       # after_snapshot: only what the due stream carried
    assert not cp.asnumpy(module.acc["wspd10max"]).any()
    assert not cp.asnumpy(module.acc["w_up_max"]).any()
    assert np.isinf(cp.asnumpy(module.acc["t2max"])).all()
    np.testing.assert_array_equal(cp.asnumpy(module.acc["grpl_max"]), keep)      # not carried: left alone
    run(second)
    utils.written = {"wspd10max", "w_up_max", "t2max", "t2min"}
    module.compute()
    peak2 = np.max([np.hypot(s["u10"], s["v10"]) for s in second], axis=0)
    np.testing.assert_allclose(pub("wspd10max"), peak2, rtol=1e-6)
    np.testing.assert_allclose(pub("t2max"), np.max([s["t2m"] for s in second], axis=0), atol=1e-4)
    assert pub("w_up_max").max() <= max(s["w"].max() for s in second)
    # a continuation resumes the interval where the checkpoint left it
    fresh, f_owner, f_utils, _ = _module(cp)
    fresh.resume(checkpoint)
    for name in nwp.NAMES:
        np.testing.assert_array_equal(cp.asnumpy(fresh.acc[name]).view(np.uint32),
                                      checkpoint[nwp.STATE_PREFIX + name].view(np.uint32), err_msg=name)


def test_resident_provider_folds_resets_and_carries_the_module(cp):
    from woof.hex.mpasdiag.convective_v842 import MAXIMA
    from woof.hex.mpasphys.resident_output_diagnostics_v842 import PRODUCED, ResidentOutputDiagnostics as R
    module, owner, utils, col = _module(cp)
    calls = []
    provider = R.__new__(R)
    provider.cp, provider.nwp, provider.buffers, provider.acoustic = cp, module, {}, {}
    provider._event_views, provider._resume, provider.updates = None, None, 0
    provider.convective_maxima = MAXIMA
    provider.convective = SimpleNamespace(update=lambda: calls.append("convective"),
                                          reset=lambda: calls.append("reset"))
    # off: the produced set is the class's own (a receipt that names it is unchanged)
    assert R.produced is PRODUCED and not set(nwp.NAMES) & set(PRODUCED)
    _feed(owner, _step_inputs(np.random.default_rng(1)))
    provider._update_convective()
    assert calls == ["convective"] and cp.asnumpy(module.acc["wspd10max"]).any()
    state = provider.accumulator_state()
    assert {nwp.STATE_PREFIX + n for n in nwp.NAMES} <= set(state)
    provider.resume_accumulators(state)
    assert set(provider._resume) == {nwp.STATE_PREFIX + n for n in (*nwp.NAMES, 'g_wspd')}
    utils.written = {"wspd10max"}
    provider.after_snapshot()
    assert calls == ["convective", "reset"] and not cp.asnumpy(module.acc["wspd10max"]).any()
    assert cp.asnumpy(module.acc["gust"]).any()               # not carried by the due stream


def test_absent_scalars_are_not_read_as_the_last_row(cp):
    """RegistryLayout's index_qg is -1 without graupel; slicing it would read the last scalar row."""
    module, owner, utils, col = _module(cp)
    owner.session.layout.indices["state", "index_qg"] = -1
    assert module._scalar("qg") is None
    _feed(owner, _step_inputs(np.random.default_rng(2)))
    module.update()
    assert not cp.asnumpy(module.acc["grpl_max"]).any()
    assert cp.asnumpy(module.acc["wspd10max"]).any()


def test_refl10cm_is_read_from_diag_physics_only(cp):
    module, owner, utils, col = _module(cp)
    inputs = _step_inputs(np.random.default_rng(4))
    _feed(owner, inputs)
    module.update()
    got = cp.asnumpy(module.acc["refd_max"])
    np.testing.assert_array_equal(got, np.maximum(inputs["refl"][0], np.float32(0)))   # maxima start at 0
    assert got.any()
    # a layout whose only refl10cm is in the diag pool (not where the Registry puts it) reads as absent
    other, o_owner, _, _ = _module(cp)
    o_owner.session.layout.fields[:] = [r for r in o_owner.session.layout.fields if r["pool_name"] != "refl10cm"]
    _feed(o_owner, inputs)
    other.update()
    assert not cp.asnumpy(other.acc["refd_max"]).any()
