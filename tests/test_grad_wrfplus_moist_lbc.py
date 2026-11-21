"""Moist lateral-boundary rows TL/AD (P3a 8k, woof.grad.wrfplus.lbc):
WRF's flow_dep_bdy on a hydrometeor without a table (``flow_tl/_ad``),
and qv's rk_step-1 capture checked against the primal's own single-row
launch (the multi-row comparison is in test_grad_wrfplus_lbc).

Breakage this prevents:
- a flow_dep_bdy adjoint that is not the transpose of its TL, that does
  not consume the ring it overwrites, or that sends a copied cell's
  adjoint to the wrong interior cell (dot-product test, every ring width,
  corners owned by the y sides);
- a TL that is not the derivative (TL vs finite differences);
- a base state that drifts from woof's strict primal: flow_dep_bdy
  against lateral_bc.apply_flow_dependent_boundaries, and qv's capture
  row against lateral_bc._launch_state_relaxation with qv's time-t source
  and mass, as moist.advance_scalars_stage launches it.

Each probe runs in a child process, as in test_grad_wrfplus_lbc.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


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

# ring widths 1-3, the smallest interiors, one level, a long thin domain
FLOW = [dict(), dict(spec_zone=2), dict(spec_zone=3, ny=8, nx=9), dict(nz=1, ny=3, nx=3),
        dict(nz=2, ny=5, nx=14, spec_zone=2)]
# qv's capture row: hybrid and not, map factors, a wider zone
QV = [dict(), dict(has_msf=1), dict(spec_zone=2, relax_zone=5, width=7, ny=14, nx=15)]

STRICT_ENV = {"GPUWM_WRF_EXACT": "1", "WOOF_WRF_EXACT_DIAGNOSTICS": "1",
              "WOOF_WRF_EXACT_ADVECTION": "1", "GPUWM_WRF_EXACT_DIFFUSION": "1",
              "GPUWM_WRF_EXACT_BIGSTEP": "1"}


def _run(backend, body, tmp_path, extra_env=None):
    env = dict(os.environ, WOOF_BACKEND=backend, PYTHONPATH=str(ROOT),
               WOOF_HOST_CACHE=str(tmp_path / "hostcache"), WOOF_HOST_THREADS="4")
    env.update(extra_env or {})
    prelude = (
        "import json, numpy as np, woof\nimport cupy as cp\n"
        "from woof.grad.wrfplus import lbc as L, cases as C, _checks as K\n"
        "out = {}\n"
        "def dev(d):\n"
        "    return {k: cp.asarray(v) for k, v in d.items()}\n"
        "def tl(g, tr, dq, nl=False):\n"
        "    st, sd = {'q': tr['q'].copy()}, {'q': dq.copy()}\n"
        "    L.flow_tl(st, sd, tr, {}, g, nl=nl)\n"
        "    return st['q'], sd['q']\n"
        "def ad(g, tr, aq):\n"
        "    sa = {'q': aq.copy()}\n"
        "    L.flow_ad(sa, tr, {}, g)\n"
        "    return sa['q']\n")
    script = prelude + body + "\nprint('RESULT ' + json.dumps(out))\n"
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                          env=env, timeout=1800)
    assert done.returncode == 0, done.stderr[-4000:]
    line = [ln for ln in done.stdout.splitlines() if ln.startswith("RESULT ")][-1]
    return json.loads(line[len("RESULT "):])


@pytest.mark.parametrize("backend", BACKENDS)
def test_flow_dot_product(backend, tmp_path):
    """<TL dq, a> == <dq, AD a>, with outflow and inflow on every side:
    1e-12 in binary64, 1e-5 in binary32."""
    dtypes = ["float32"] if backend == "mlx" else ["float64", "float32"]
    out = _run(backend, f'''
for dtn in {dtypes!r}:
    dt = np.dtype(dtn)
    for i, over in enumerate({FLOW!r}):
        g = C.flow_geometry(**over)
        tr = dev(C.flow_trajectory(g, 11 + i, dt))
        rng = np.random.default_rng(100 + i)
        dq = cp.asarray(rng.standard_normal(tr["q"].shape).astype(dt))
        a = cp.asarray(rng.standard_normal(tr["q"].shape).astype(dt))
        out[f"{{dtn}}/{{i}}"] = K.dot_product(lambda x: [tl(g, tr, x[0])[1]],
                                              lambda y: [ad(g, tr, y[0])], [dq], [a],
                                              terms=dtn == "float32")
''', tmp_path)
    assert len(out) == (1 if backend == "mlx" else 2) * len(FLOW)
    for key, rel in out.items():
        assert rel < (1e-12 if key.startswith("float64") else 1e-5), (key, rel)


@pytest.mark.parametrize("backend", [b for b in BACKENDS if b.values[0] != "mlx"])
def test_flow_tl_matches_finite_differences(backend, tmp_path):
    """||NL(q + a dq) - NL(q) - a TL dq|| / ||a TL dq|| -> 0 (the map is
    linear in q at fixed fluxes, so it is 0 to roundoff).  NL is the same
    function's base update, so this only ties the base and tangent
    branches together; the primal words test ties the base to woof."""
    out = _run(backend, f'''
for i, over in enumerate({FLOW!r}):
    g = C.flow_geometry(**over)
    tr = dev(C.flow_trajectory(g, 31 + i, np.float64))
    dq = cp.asarray(np.random.default_rng(300 + i).standard_normal(tr["q"].shape))
    def model(xs):
        return [tl(g, {{**tr, "q": xs[0]}}, cp.zeros_like(xs[0]), nl=True)[0]]
    out[f"{{i}}"] = K.tl_residual(model, lambda d: [tl(g, tr, d[0])[1]], [tr["q"]], [dq],
                                  [1e-3, 1e-5])
''', tmp_path)
    for key, rows in out.items():
        assert max(r for _, r in rows) < 1e-9, (key, rows)


@pytest.mark.parametrize("backend", [b for b in BACKENDS if b.values[0] != "mlx"])
def test_float_base_state_is_the_strict_primal(backend, tmp_path):
    """The float build gives woof's strict words: flow_dep_bdy against
    apply_flow_dependent_boundaries (every ring cell moves), and qv's
    capture row against _launch_state_relaxation with qv's time-t source
    and mass.  Host and CUDA; Metal is not checked here."""
    out = _run(backend, f'''
from types import SimpleNamespace as NS
from woof.ingest import lateral_bc as LB
f32 = np.float32
def diff(a, b):
    return int((cp.asnumpy(a).view(np.uint32) != cp.asnumpy(b).view(np.uint32)).sum())
for i, over in enumerate({FLOW!r}):
    g = C.flow_geometry(**over)
    tr = dev(C.flow_trajectory(g, 41 + i, f32))
    st, _ = tl(g, tr, cp.zeros_like(tr["q"]), nl=True)
    ref = tr["q"].copy()
    LB.apply_flow_dependent_boundaries((ref,), tr["ru"], tr["rv"], g.spec_zone)
    out[f"flow/{{i}}"] = diff(st, ref)
    out[f"flow/{{i}}/moves"] = diff(st, tr["q"])
for i, over in enumerate({QV!r}):
    g = C.lbc_geometry(kind="qv", apply_relax=1, **over)
    tr = dev(C.lbc_trajectory(g, 61 + i, f32, hybrid=bool(i % 2)))
    st = {{n: tr[n].copy() for n in L.LB_STATE}}
    L.lbc_tl(st, {{n: cp.zeros_like(tr[n]) for n in L.LB_STATE}}, tr,
             {{n: cp.zeros_like(tr[n]) for n in L.LB_ACTIVE}}, g, nl=True)
    ref = tr["tend"].copy()
    z = cp.zeros_like(tr["mub"])
    state = NS(mub2d=tr["mub"], mup=z, u=tr["src"], v=tr["src"], w=tr["src"], thp=tr["src"],
               thb=cp.zeros(g.nz, dtype=f32), php=tr["src"], c1h=tr["c1h"], c2h=tr["c2h"],
               c1f=tr["c1f"], c2f=tr["c2f"], msft=tr["msft"], msfu=tr["msfu"],
               msfv=tr["msfv"], msfv_inv=tr["msfv"], has_msf=bool(g.has_msf),
               lateral_boundaries=None)
    bound = NS(**{{s: NS(value=tr[s], tendency=tr[s + "_t"])
                  for s in ("west", "east", "south", "north")}})
    LB._launch_state_relaxation(
        state, "qv", ref, bound, dtbc=g.dtbc, dt=20.0, spec_zone=g.spec_zone,
        relax_zone=g.relax_zone, spec_exp=0.0, apply_relax=True,
        weights=(tr["fcx"], tr["gcx"]), clear_specified=False, add_held=None,
        divide_msf=False, source_field=tr["src"], source_mup=tr["mup"])
    out[f"qv/{{i}}"] = diff(st["tend"], ref)
    out[f"qv/{{i}}/moves"] = diff(st["tend"], tr["tend"])
''', tmp_path, extra_env=STRICT_ENV)
    assert len(out) == 2 * (len(FLOW) + len(QV))
    bad = {k: v for k, v in out.items() if (v == 0 if k.endswith("/moves") else v)}
    assert not bad, bad


def test_flow_geometry_rejects():
    """spec_zone 0 and a ring with no interior to copy are refused."""
    from woof.grad.wrfplus import cases as C
    for over in (dict(spec_zone=0), dict(ny=4, nx=9, spec_zone=2), dict(nz=0)):
        with pytest.raises(ValueError):
            C.flow_geometry(**over).validate()
