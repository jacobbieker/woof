"""diff_opt 2 vertical-diffusion rows TL/AD (P3a item 2b-2,
woof.grad.wrfplus.diffusion.h2_tl/h2_ad, rows vu, vv, vw, vt, vs): WRF
vertical_diffusion_2's vertical_diffusion_u_2, _v_2, _w_2 and
vertical_diffusion_s with the surface fluxes of its vflux, hflux and qflux
cases, on woof's strict wrf_smag_vd_u / _v / _w / _s and the surface
kernels dycore.launch_wrf_smag2d_vertical launches after them.  The
stresses tau13 / tau23 and tau33, the metrics (rdz, rdzw, zx, zy of ph),
the density (1/alt, times 1 + qv when moist), the vertical K and the
winds, w or the field are active; the surface fields USTM, HFX and QFX,
the drag coefficient, the prescribed heat flux and the base theta are
passive inputs.

Breakage this prevents:
- an adjoint that is not the transpose of its TL (dot-product test);
- an adjoint that overwrites instead of adding into its inputs, or that
  touches the tendency's adjoint (the rows accumulate);
- a TL that is not the derivative of the base-state update
  (TL vs finite differences);
- a base state, TL or AD that drifts from WRF 4.7.1's nonlinear
  vertical_diffusion_2 (tests/data/wrfplus/vdiffusion2.npz, made by
  tools/wrfplus_oracle from the WRF routines in WRFPLUS 4.7.1's
  libwrflib.a, with the TL/AD from fourth-order differences of them);
- a TL/AD base state that drifts from woof's own strict primal (the
  float build must give dycore.launch_wrf_smag2d_vertical's words).

Each probe runs in a child process, as in test_grad_wrfplus_hdiff2.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/data/wrfplus/vdiffusion2.npz"


def _has_cuda() -> bool:
    try:
        import cupy  # noqa: F401
        return bool(getattr(cupy, "__woof_facade__", False)) is False and \
            cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def _has_metal() -> bool:
    try:
        import mlx.core as mx
        return bool(mx.metal.is_available())
    except Exception:
        return False


HAVE_CXX = any(shutil.which(c) for c in ("clang++", "g++", "c++"))
BACKENDS = [pytest.param("host", marks=pytest.mark.skipif(not HAVE_CXX, reason="no C++ compiler")),
            pytest.param("cuda", marks=pytest.mark.skipif(not _has_cuda(), reason="no CUDA card")),
            pytest.param("mlx", marks=pytest.mark.skipif(not _has_metal(), reason="no Metal"))]

# Each row with its surface options (vu / vv: none, USTM, cd0; vt: none,
# the prescribed flux, HFX, HFX with full_theta; vs: none and the qv row's
# QFX, moist).
VARIANTS = [("vu", {}), ("vu", dict(surface="ustm")), ("vu", dict(surface="cd0", cd0=2e-3)),
            ("vv", {}), ("vv", dict(surface="ustm")), ("vv", dict(surface="cd0", cd0=2e-3)),
            ("vw", {}),
            ("vt", {}), ("vt", dict(heat="const", heat_flux=0.1)), ("vt", dict(heat="hfx")),
            ("vt", dict(heat="hfx", full_theta=1)),
            ("vs", {}), ("vs", dict(qflux=1, moist=1))]
# Periodic, bounded (both axes, one), the shortest column, single-cell and
# two-cell lines, moist density and bounded length-2 axes.
CONFIGS = [dict(nz=5, ny=4, nx=5), dict(nz=5, ny=4, nx=5, boundary_x=1, boundary_y=1),
           dict(nz=4, ny=4, nx=5, boundary_x=1), dict(nz=3, ny=1, nx=2),
           dict(nz=3, ny=3, nx=1, boundary_y=1),
           dict(nz=4, ny=3, nx=4, boundary_x=1, boundary_y=1, moist=1),
           dict(nz=3, ny=2, nx=2, boundary_x=1, boundary_y=1)]


def _geoms():
    """(key, geometry overrides) over VARIANTS x CONFIGS."""
    out = []
    for v, (row, var) in enumerate(VARIANTS):
        for i, over in enumerate(CONFIGS):
            out.append((f"{row}{v}/{i}", dict(over, row=row, **var)))
    return out


GEOMS = _geoms()


STRICT_ENV = {"GPUWM_WRF_EXACT": "1", "WOOF_WRF_EXACT_DIAGNOSTICS": "1",
              "WOOF_WRF_EXACT_ADVECTION": "1", "GPUWM_WRF_EXACT_DIFFUSION": "1",
              "GPUWM_WRF_EXACT_BIGSTEP": "1"}


def _run(backend, body, tmp_path, extra_env=None):
    env = dict(os.environ, WOOF_BACKEND=backend, PYTHONPATH=str(ROOT),
               WOOF_HOST_CACHE=str(tmp_path / "hostcache"), WOOF_HOST_THREADS="4")
    env.update(extra_env or {})
    prelude = (
        "import json, numpy as np, woof\nimport cupy as cp\n"
        "from woof.grad.wrfplus import diffusion as D, cases as C, _checks as K\n"
        "out = {}\n"
        "def dev(d):\n"
        "    return {k: cp.asarray(v) for k, v in d.items()}\n"
        "def run_tl(g, tr, pe, nl, tend=None):\n"
        "    st = {'tend': tr['tend'].copy() if tend is None else tend.copy()}\n"
        "    sd = {'tend': cp.zeros_like(tr['tend'])}\n"
        "    D.h2_tl(st, sd, tr, pe, g, nl=nl)\n"
        "    return st, sd\n"
        "def run_ad(g, tr, sa, adj=None):\n"
        "    sa = {'tend': sa['tend'].copy()}\n"
        "    if adj is None:\n"
        "        adj = {n: cp.zeros_like(tr[n]) for n in D.h2_active(g)}\n"
        "    D.h2_ad(sa, tr, adj, g)\n"
        "    return sa, adj\n"
        "def seed(tr, s, dt):\n"
        "    rng = np.random.default_rng(s)\n"
        "    return {'tend': cp.asarray(rng.standard_normal(tr['tend'].shape).astype(dt))}\n")
    script = prelude + body + "\nprint('RESULT ' + json.dumps(out))\n"
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                          env=env, timeout=3600)
    assert done.returncode == 0, done.stderr[-4000:]
    line = [ln for ln in done.stdout.splitlines() if ln.startswith("RESULT ")][-1]
    return json.loads(line[len("RESULT "):])


@pytest.mark.parametrize("backend", BACKENDS)
def test_dot_product(backend, tmp_path):
    """<TL dx, dy> == <dx, AD dy>: 1e-12 in binary64; 1e-5 in binary32,
    relative to the term magnitudes."""
    dtypes = ["float32"] if backend == "mlx" else ["float64", "float32"]
    out = _run(backend, f"""
for dtn in {dtypes!r}:
    dt = np.dtype(dtn)
    for n, (key, over) in enumerate({GEOMS!r}):
        g = C.h2_geometry(**over)
        X = D.h2_active(g)
        tr = dev(C.h2_trajectory(g, 11 + n, dt, hybrid=bool(n % 2)))
        p = dev(C.h2_perturbation(g, 100 + n, dt))
        dy = seed(tr, 200 + n, dt)
        def tl(dx):
            return [run_tl(g, tr, dict(zip(X, dx)), False)[1]["tend"]]
        def ad(dys):
            return [run_ad(g, tr, {{"tend": dys[0]}})[1][m] for m in X]
        out[f"{{dtn}}/{{key}}"] = K.dot_product(
            tl, ad, [p[m] for m in X], [dy["tend"]], terms=dtn == "float32")
""", tmp_path)
    assert len(out) == (1 if backend == "mlx" else 2) * len(GEOMS)
    for key, rel in out.items():
        limit = 1e-12 if key.startswith("float64") else 1e-5
        assert rel < limit, (key, rel)


@pytest.mark.parametrize("backend", BACKENDS)
def test_adjoint_accumulates(backend, tmp_path):
    """The AD adds into ``adj`` and leaves the tendency's adjoint as it is
    (the primal accumulates into the tendency)."""
    dtn = "float32" if backend == "mlx" else "float64"
    out = _run(backend, f"""
dt = np.dtype({dtn!r})
for n, (key, over) in enumerate({GEOMS!r}):
    g = C.h2_geometry(**over)
    X = D.h2_active(g)
    tr = dev(C.h2_trajectory(g, 51 + n, dt))
    q = seed(tr, 500 + n, dt)
    s = dev(C.h2_perturbation(g, 600 + n, dt))
    sa, zero = run_ad(g, tr, q)
    _, adj = run_ad(g, tr, q, {{m: s[m].copy() for m in X}})
    out[f"{{key}}/st"] = float(cp.abs(sa["tend"] - q["tend"]).max())
    for m in X:
        want = cp.asnumpy(s[m]).astype(np.float64) + cp.asnumpy(zero[m])
        got = cp.asnumpy(adj[m]).astype(np.float64)
        scale = max(float(np.abs(want).max()), 1e-300)
        out[f"{{key}}/{{m}}"] = float(np.abs(got - want).max()) / scale
""", tmp_path)
    limit = 1e-5 if backend == "mlx" else 1e-13
    bad = {k: v for k, v in out.items()
           if (v != 0.0 if k.endswith("/st") else not v < limit)}
    assert not bad, bad


@pytest.mark.parametrize("backend", [b for b in BACKENDS if b.values[0] != "mlx"])
def test_tl_matches_finite_differences(backend, tmp_path):
    """||NL(x + a dx) - NL(x) - a TL dx|| / ||a TL dx|| -> 0 as O(a)."""
    out = _run(backend, f"""
for n, (key, over) in enumerate({GEOMS!r}):
    g = C.h2_geometry(**over)
    X = D.h2_active(g)
    tr = dev(C.h2_trajectory(g, 31 + n, np.float64))
    p = dev(C.h2_perturbation(g, 300 + n, np.float64))
    def model(xs):
        t = dict(tr); t.update(dict(zip(X, xs)))
        st, _ = run_tl(g, t, {{m: cp.zeros_like(t[m]) for m in X}}, True)
        return [st["tend"]]
    def tl(dx):
        return [run_tl(g, tr, dict(zip(X, dx)), False)[1]["tend"]]
    out[key] = K.tl_residual(model, tl, [tr[m] for m in X], [p[m] for m in X],
                             [1e-1, 1e-3, 1e-5])
""", tmp_path)
    assert len(out) == len(GEOMS)
    for key, rows in out.items():
        misfits = [r for _, r in rows]
        assert misfits[-1] < 1e-6, (key, rows)
        assert misfits[0] < 0.1, (key, rows)


@pytest.mark.parametrize("backend", BACKENDS)
def test_wrf_oracle_agreement(backend, tmp_path):
    """woof's base state, TL and AD reproduce WRF 4.7.1's nonlinear
    vertical_diffusion_2 (with compute_diff_metrics, rho and
    cal_deform_and_div) on WRF's loop region: the base tendency to 1e-13 in
    binary64, the TL and AD to 1e-8 of their scale (the fixture
    differentiates WRF by fourth-order central differences).  The
    tendency adjoint is seeded on the loop region only."""
    if not FIXTURE.exists():
        pytest.skip("tests/data/wrfplus/vdiffusion2.npz not built")
    dtn = "float32" if backend == "mlx" else "float64"
    out = _run(backend, f"""
import json as J
dt = np.dtype({dtn!r})
z = np.load({str(FIXTURE)!r})
for case in sorted({{k.split("/")[0] for k in z.files if "/" in k}}):
    pre = case + "/"
    g = D.H2Geometry(**J.loads(str(z[pre + "geometry"])))
    X = D.h2_active(g)
    pick = lambda kind: {{k.split("/", 2)[2]: cp.asarray(z[k].astype(dt))
                          for k in z.files if k.startswith(pre + kind + "/")}}
    tr, pe, sd = pick("traj"), pick("pert"), pick("seed")
    m = z[pre + "wrfplus/mask"]
    t0 = z[pre + "traj/tend"]
    st, dd = run_tl(g, tr, pe, True)
    _, adj = run_ad(g, tr, sd)
    got = {{"nl_tend": cp.asnumpy(st["tend"]).astype(np.float64) - t0,
            "tl_d_tend": cp.asnumpy(dd["tend"]).astype(np.float64)}}
    for n in X:
        got["ad_" + n] = cp.asnumpy(adj[n]).astype(np.float64)
    for name, w in got.items():
        e = z[pre + "wrfplus/" + name]
        if name == "nl_tend":
            e = e - t0
        if name.endswith("_tend"):
            e, w = e[m], w[m]
        scale = max(float(np.abs(e).max()), 1e-300)
        out[pre + name] = float(np.abs(w - e).max()) / scale
""", tmp_path)
    for key, v in out.items():
        if backend == "mlx":
            limit = 1e-4
        else:
            limit = 1e-13 if key.endswith("nl_tend") else 1e-8
        assert v < limit, (key, v)
    cases = {k.split("/")[0] for k in out}
    assert len(cases) >= 16
    assert {k.split("/")[0].split("_")[1] for k in out} == {"vu", "vv", "vw", "vt", "vs"}


@pytest.mark.parametrize("backend", [b for b in BACKENDS if b.values[0] != "mlx"])
def test_float_base_state_is_the_strict_primal(backend, tmp_path):
    """The float build's base state gives woof's strict diff_opt 2 vertical
    words through dycore.launch_wrf_smag2d_vertical as the model launches
    it (the interior row, then the surface kernels of its isfflx case:
    USTM, cd0, the prescribed heat flux, HFX, QFX), so the adjoint is
    bound to the model woof runs: a primal edit fails here until the TL/AD
    is re-derived.  The rows add every point; the model zeroes the open
    strips afterwards.  Host and CUDA; Metal is not checked here."""
    out = _run(backend, f"""
from types import SimpleNamespace as NS
from woof.core import dycore
f32 = np.float32
def diff(a, b):
    return int((cp.asnumpy(a).view(np.uint32) != cp.asnumpy(b).view(np.uint32)).sum())
dx, dy = 3000.0, 2500.0
def primal(g, tr):
    nz, ny, nx = g.nz, g.ny, g.nx
    zero = lambda *s: cp.zeros(s, dtype=f32)
    fields = {{n: tr[n] for n in ("ustm", "hfx", "qfx") if n in tr}}
    for n in ("ustm", "hfx", "qfx"):
        fields.setdefault(n, zero(ny, nx))
    st = NS(u=tr.get("u", zero(nz, ny, nx + 1)), v=tr.get("v", zero(nz, ny + 1, nx)),
            w=tr.get("w", zero(nz + 1, ny, nx)), php=tr["php"], phb=tr["phb"],
            alt=tr["alt"], qv=tr["qv"] if g.moist else None, msft=tr["msft"],
            msfu=tr["msfu"], msfv=tr["msfv"], fnm=tr["fnm"], fnp=tr["fnp"], dn=tr["dn"],
            dnw=tr["dnw"], cf1=g.cf1, cf2=g.cf2, cf3=g.cf3,
            thb=tr.get("thb", zero(nz, ny, nx)), thp=zero(nz, ny, nx), mup0=zero(ny, nx),
            physics=NS(fields=fields))
    # the isfflx case and surface scheme that select the row's surface terms
    if g.surface == "cd0":
        isfflx, active = 0, 0
    elif g.heat == "const":
        isfflx, active = 0, 0
    elif g.surface == "ustm" or g.heat == "hfx" or g.qflux:
        isfflx, active = 1, 1
    else:
        isfflx, active = 1, 0
    cfg = NS(dx=dx, dy=dy, open_x=bool(g.boundary_x), open_y=bool(g.boundary_y),
             specified=False, nested=False, isfflx=isfflx, sf_sfclay_physics=active,
             tke_drag_coefficient=g.cd0, tke_heat_flux=g.heat_flux, use_theta_m=0)
    tend = tr["tend"].copy()
    bufs = {{"ru": zero(nz, ny, nx + 1), "rv": zero(nz, ny + 1, nx)}}
    bufs["rw"] = zero(nz + 1, ny, nx)
    bufs["rth"], bufs["rqv"] = zero(nz, ny, nx), zero(nz, ny, nx)
    slot = {{"vu": "ru", "vv": "rv", "vw": "rw", "vt": "rth", "vs": "rqv" if g.qflux else "rth"}}
    bufs[slot[g.row]] = tend
    rows = None
    khv = None
    if g.row in ("vt", "vs"):
        rows, khv = [(tr["field"], tend, bool(g.full_theta))], tr["xk"]
    # every row's buffer is live (the surface kernels write ru / rv / rth
    # whatever the row); only the row under test is the trajectory's
    dycore.launch_wrf_smag2d_vertical(
        st, cfg, tr["xk"], ru=bufs["ru"], rv=bufs["rv"], rw=bufs["rw"], rth=bufs["rth"],
        rqv=bufs["rqv"], time_t=False, kmv=tr["xk"], khv=khv, scalar_rows=rows)
    return tend
for n, (key, over) in enumerate({GEOMS!r}):
    g = C.h2_geometry(rdx=float(f32(1 / dx)), rdy=float(f32(1 / dy)), **over)
    X = D.h2_active(g)
    tr = dev(C.h2_trajectory(g, 41 + n, f32, hybrid=True))
    st, _ = run_tl(g, tr, {{m: cp.zeros_like(tr[m]) for m in X}}, True)
    out[key] = diff(st["tend"], primal(g, tr))
    out[key + "/moves"] = int(diff(st["tend"], tr["tend"]) == 0)
""", tmp_path, extra_env=STRICT_ENV)
    assert len(out) == 2 * len(GEOMS)
    bad = {k: v for k, v in out.items() if v}
    assert not bad, bad


def test_geometry_rejects_unported_options():
    """Surface options on rows that have none, qflux off the moist vs row,
    unknown surface or heat kinds and flags other than 0/1 are refused, not
    run."""
    from woof.grad.wrfplus import cases as C
    for over in (dict(row="vw", surface="ustm"), dict(row="vs", heat="hfx"),
                 dict(row="vt", qflux=1, moist=1), dict(row="vs", qflux=1),
                 dict(row="vu", surface="drag"), dict(row="vt", heat="flux"),
                 dict(row="vt", full_theta=2), dict(row="u", surface="cd0"),
                 dict(row="vt", nz=2)):
        with pytest.raises(ValueError):
            C.h2_geometry(**over).validate()
