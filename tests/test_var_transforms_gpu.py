"""woof.da.var.transforms and control: U, U^T and the inner products on the
device (plan V0.2-V0.4; gates O2, O3 and O8).

What is asserted, and why it is enough:

- **Oracle tier = the reference, word for word.**  The binary64 kernels
  spell every operation as a correctly rounded intrinsic in the reference's
  order, so every stage of ``U v`` and ``U^T x`` equals
  :mod:`woof.da.var.reference`'s bit for bit, on CUDA and on Metal, in fast
  and in strict numerics (child processes, one per mode).  The reference
  equals WRFDA 4.7.1's serial build word for word (O2,
  ``test_var_transforms_reference``), and so, on the tutorial case, does
  the device (``test_o2_device_*``).
- **Production tier: one set of words on every backend.**  Binary32 in the
  same order: within 2e-6 of the binary64 reference, and the SHA-256 of
  its outputs is a constant of this file; CUDA and Metal under strict
  numerics both reproduce it.
- **Inner products are a function of the input alone.**  ``wrfda-serial``
  equals a host left-to-right sum, ``fixed-tree`` a host emulation of the
  chunk-and-tree order, exactly; that is what makes them backend-independent
  by construction, and the constants below pin the device words too.
- **Adjoints (O3):** ``<U v, x> = <v, U^T x>`` to 1e-13 (binary64) and 1e-6
  (binary32).
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu

from woof.da.var import be_dat as bd
from woof.da.var import reference as ref
from woof.da.var.transforms import XA_KEYS, psichi_adj_plan

from test_var_be_dat import _synthetic
from test_var_transforms_reference import _oracle, _xa, _xb

NX, NY, NZ = 13, 11, 6
REPO = Path(__file__).resolve().parents[1]

#: SHA-256 of the production tier's xa (U v) and cv (U^T x) words for the
#: synthetic case below, cv_options 5 and 7, and the fixed-tree dot of the
#: float32 vectors; measured on Metal (strict) and CUDA (A6000).
PRODUCTION_DIGESTS = {
    "production5": "ed648c5813d886a05692eb568bd94f20c0f0b990d33e6442c683f6c5f85a1db5",
    "production7": "4d08d34a8e5bbb04196342f38e51819340d08760128438e83967ee8950c8a038",
    "dot32": "-0x1.b60a7b2853e0bp+8",
}


def _case(cv_options):
    be = _synthetic(ni=NX, nj=NY, nk=NZ, nb2d=2, cv_options=cv_options)
    b = bd.setup_be_regional(be, ix=NX, jy=NY, kz=NZ, ds=3000.0, config=bd.BeConfig(cv_options=cv_options))
    rng = np.random.default_rng(5)
    xb = _xb(rng, NX, NY, NZ)
    cv = rng.standard_normal(NX * NY * sum(b.cv_mz))
    xa = _xa(rng, NX, NY, NZ)
    return b, xb, cv, xa


# ------------------------------------------------------- host-only checks


def _gather(plan, u, v, cx, cy, nz):
    row, src, op = plan
    out = np.zeros((cx.size, nz))
    uf, vf = u.reshape(-1, nz), v.reshape(-1, nz)
    for r in range(cx.size):
        acc = out[r].copy()
        for e in range(row[r], row[r + 1]):
            s, o = src[e], op[e]
            c = (cy if o & 1 else cx).reshape(-1)[s]
            p = c * (vf if o & 2 else uf)[s]
            acc = acc - p if o & 4 else acc + p
        out[r] = acc
    return out.reshape(u.shape)


def test_psichi_gather_plan_is_the_scatter_in_its_order():
    rng = np.random.default_rng(2)
    nx, ny, nz = 9, 7, 3
    u, v = rng.standard_normal((2, nx, ny, nz))
    cx, cy = rng.uniform(0.5, 2.0, (2, nx, ny))
    want_psi, want_chi = ref.psichi_to_uv_adj(u, v, cx, cy)
    # the corner statements first, as the device does on its copies
    uc, vc = u.copy(), v.copy()
    ie, je = nx - 1, ny - 1
    for (a0, a1, b0, b1, c0, c1) in ((1, 0, 0, 1, 0, 0), (ie - 1, 0, ie, 1, ie, 0), (1, je, 0, je - 1, 0, je),
                                     (ie - 1, je, ie, je - 1, ie, je)):
        for a in (uc, vc):
            a[a0, a1] = a[a0, a1] + 0.5 * a[c0, c1]
            a[b0, b1] = a[b0, b1] + 0.5 * a[c0, c1]
    p_plan, c_plan = psichi_adj_plan(nx, ny)
    np.testing.assert_array_equal(_gather(p_plan, uc, vc, cx, cy, nz), want_psi)
    np.testing.assert_array_equal(_gather(c_plan, uc, vc, cx, cy, nz), want_chi)


def _tree_dot(x, y):
    """Host emulation of the fixed-tree order (kernels/var_u.cu)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n, chunk = x.size, 4096

    def tree(sh):
        sh = list(sh)
        s = 128
        while s:
            for t in range(s):
                sh[t] = sh[t] + sh[t + s]
            s //= 2
        return sh[0]

    partial = []
    for c in range(max(1, (n + chunk - 1) // chunk)):
        sh = []
        for t in range(256):
            acc = 0.0
            for q in range(16):
                i = c * chunk + t * 16 + q
                if i < n:
                    acc = acc + float(x[i]) * float(y[i])
            sh.append(acc)
        partial.append(tree(sh))
    sh = []
    for t in range(256):
        acc = 0.0
        for q in range(t, len(partial), 256):
            acc = acc + partial[q]
        sh.append(acc)
    return tree(sh)


def _serial_dot(x, y):
    acc = 0.0
    for a, b in zip(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)):
        acc = acc + float(a) * float(b)
    return acc


# ------------------------------------------------------- device, in-process


@pytest.fixture(scope="module")
def cp():
    import cupy

    return cupy


def _stages(cv_options, precision):
    """Every stage of U v and U^T x on the device, against the reference."""
    from woof.da.var.transforms import DeviceU

    b, xb, cv, xa = _case(cv_options)
    r, ra = ref.transform_u(cv, b, xb), ref.transform_u_adj(xa, b, xb)
    dev = DeviceU(b, xb, precision)
    g, ga = dev.apply(cv, stages=True), dev.adjoint(xa, stages=True)
    pairs = []
    for st in ("vv", "vp", "vp_bal"):
        pairs += [(f"U.{st}.{k}", g[st][k], r[st][k]) for k in r[st]]
    pairs += [(f"U.xa.{k}", g["xa"][k], r["xa"][k]) for k in XA_KEYS]
    for st in ("vp", "vv_eof", "vv"):
        pairs += [(f"UT.{st}.{k}", ga[st][k], ra[st][k]) for k in ra[st]]
    pairs += [(f"UT.xa.{k}", ga["xa"][k], ra["xa"][k]) for k in XA_KEYS]
    pairs.append(("UT.cv", ga["cv"], ra["cv"]))
    report, words = {}, hashlib.sha256()
    for name, a, want in pairs:
        a = np.asarray(a.get())
        want = np.asarray(want)
        if want.size:
            d = np.abs(a.astype(np.float64).reshape(want.shape) - want)
            report[name] = float(d.max()) / max(float(np.abs(want).max()), 1e-300)
        if name.startswith("U.xa.") or name == "UT.cv":
            words.update(np.ascontiguousarray(a, dtype=a.dtype.newbyteorder("<")).tobytes())
    return report, words.hexdigest()


@requires_gpu
@pytest.mark.parametrize("cv_options", [5, 7])
def test_oracle_tier_is_the_reference_word_for_word(cv_options):
    report, _ = _stages(cv_options, "oracle")
    bad = {k: v for k, v in report.items() if v != 0.0}
    assert not bad, bad


@requires_gpu
@pytest.mark.parametrize("cv_options", [5, 7])
def test_production_tier_is_within_binary32_of_the_reference(cv_options):
    report, _ = _stages(cv_options, "production")
    bad = {k: v for k, v in report.items() if not v <= 2e-6}
    assert not bad, bad


@requires_gpu
@pytest.mark.parametrize("precision,rtol", [("oracle", 1e-13), ("production", 1e-6)])
@pytest.mark.parametrize("cv_options", [5, 7])
def test_u_adjoint_on_device(cp, cv_options, precision, rtol):
    from woof.da.var.control import dot
    from woof.da.var.transforms import DeviceU

    b, xb, cv, xa = _case(cv_options)
    dev = DeviceU(b, xb, precision)
    cvd = cp.asarray(cv, dtype=dev.dtype)
    xad = {k: cp.asarray(v, dtype=dev.dtype) for k, v in xa.items()}
    fwd = dev.apply(cvd)["xa"]
    adj = dev.adjoint(xad)["cv"]
    lhs = sum(dot(fwd[k].ravel(), xad[k].ravel()) for k in XA_KEYS)
    rhs = dot(cvd, adj)
    assert lhs == pytest.approx(rhs, rel=rtol)


@requires_gpu
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("n", [1, 4095, 4096, 4097, 3 * 4096 * 256 + 17])
def test_dots_are_their_host_orders_exactly(cp, dtype, n):
    from woof.da.var.control import dot

    rng = np.random.default_rng(n)
    x = rng.standard_normal(n).astype(dtype)
    y = rng.standard_normal(n).astype(dtype)
    xd, yd = cp.asarray(x), cp.asarray(y)
    if n < 100000:
        assert dot(xd, yd, "wrfda-serial") == _serial_dot(x, y)
        assert dot(xd, yd, "fixed-tree") == _tree_dot(x, y)
    else:
        # the tree on a host loop is slow at this size: its partial count
        # (769) exercises the combine stage's strided sums
        assert dot(xd, yd, "fixed-tree") == pytest.approx(float(np.dot(x.astype(np.float64), y)), rel=1e-12)


@requires_gpu
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_axpy_rounds_each_operation(cp, dtype):
    from woof.da.var.control import axpy

    rng = np.random.default_rng(3)
    x = rng.standard_normal(5000).astype(dtype)
    y = rng.standard_normal(5000).astype(dtype)
    a = dtype(0.3125001)
    yd = cp.asarray(y)
    axpy(a, cp.asarray(x), yd)
    np.testing.assert_array_equal(yd.get(), y + a * x)


# ----------------------------------------- identity across modes (children)

_CHILD = r"""
import json, sys
sys.path.insert(0, sys.argv[1])
import numpy as np
from test_var_transforms_gpu import _stages, _case
from woof.da.var.control import dot
import cupy as cp
out = {}
for cvo in (5, 7):
    r, _ = _stages(cvo, "oracle")
    out[f"oracle{cvo}"] = max(r.values())
    r, d = _stages(cvo, "production")
    out[f"production{cvo}"] = d
rng = np.random.default_rng(9)
x = rng.standard_normal(70001).astype(np.float32)
y = rng.standard_normal(70001).astype(np.float32)
out["dot32"] = dot(cp.asarray(x), cp.asarray(y)).hex()
print("RESULT " + json.dumps(out))
"""


@requires_gpu
@pytest.mark.parametrize("mode", ["fast", "strict"])
def test_words_in_fast_and_strict_child_processes(mode):
    env = dict(os.environ, GPUWM_NUMERICS=mode)
    proc = subprocess.run([sys.executable, "-c", _CHILD, str(Path(__file__).parent)], env=env,
                          capture_output=True, text=True, timeout=1800, cwd=REPO)
    assert proc.returncode == 0, proc.stderr[-4000:]
    line = [s for s in proc.stdout.splitlines() if s.startswith("RESULT ")][-1]
    got = json.loads(line[len("RESULT "):])
    assert got["oracle5"] == 0.0 and got["oracle7"] == 0.0, got
    from woof import backend

    if mode == "strict" or backend.active() == "cuda":
        # CUDA: intrinsics fix every rounding in both modes; Metal: strict
        assert {k: got[k] for k in PRODUCTION_DIGESTS} == PRODUCTION_DIGESTS, got


# --------------------------------------------------------------- gate O2


def _o2_device():
    from woof.da.var.transforms import DeviceU

    b, xb, du = _oracle()
    return DeviceU(b, xb, "oracle"), du


@requires_gpu
def test_o2_device_forward_is_wrfda_word_for_word():
    dev, du = _o2_device()
    st = dev.apply(du["cv1"], stages=True)
    bad = {}
    for stage in ("vv", "vp", "vp_bal"):
        for n in range(1, 6):
            a = st[stage][f"v{n}"].get()
            want = du[f"{stage}%v{n}"][:, :, : a.shape[2]]
            if not np.array_equal(a, want):
                bad[f"{stage}%v{n}"] = float(np.max(np.abs(a - want)))
    for k in XA_KEYS:
        a = st["xa"][k].get()
        if not np.array_equal(a, du[f"xa%{k}"]):
            bad[f"xa%{k}"] = float(np.max(np.abs(a - du[f"xa%{k}"])))
    assert not bad, bad


@requires_gpu
def test_o2_device_adjoint_is_wrfda_word_for_word():
    dev, du = _o2_device()
    st = dev.adjoint({k: du[f"xa%{k}#2"] for k in XA_KEYS}, stages=True)
    bad = {}
    for k in XA_KEYS:
        a = st["xa"][k].get()
        if not np.array_equal(a, du[f"xa%{k}#3"]):
            bad[f"xa_adj%{k}"] = float(np.max(np.abs(a - du[f"xa%{k}#3"])))
    for n in range(1, 6):
        for stage, tag in (("vp", "vp_adj"), ("vv_eof", "vv_adj")):
            a = st[stage][f"v{n}"].get()
            want = du[f"{tag}%v{n}"][:, :, : a.shape[2]]
            if not np.array_equal(a, want):
                bad[f"{tag}%v{n}"] = float(np.max(np.abs(a - want)))
    if not np.array_equal(st["cv"].get(), du["cv2"]):
        bad["cv2"] = float(np.max(np.abs(st["cv"].get() - du["cv2"])))
    assert not bad, bad
