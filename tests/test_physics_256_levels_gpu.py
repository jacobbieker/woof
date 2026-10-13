"""YSU, Shin-Hong, Kain-Fritsch and terrain drag on 256-level columns (GPU).

Four ceilings went 128 -> 256 together, to the dynamical core's own
``ACOUSTIC_VERTICAL_LEVEL_BOUNDS``: ``YSU_KMAX`` (regional and global),
``SHINHONG_KMAX``, ``KF_KMAX`` and ``TD_KMAX``.  Two claims, each per scheme:

1. **A 256-level column runs.**  The column is the shape the user's
   end-to-end run has -- 256 levels to a 1 hPa model top -- through the
   SHIPPED launcher, and every output is finite and non-trivial.  The
   kernels' own guards (``nz > *_KMAX``) return without writing, which a
   launcher fed ``cp.empty`` outputs would hand back as garbage, so "finite
   and moved" is also the check that the guard did not fire.

2. **Nothing moved below the old ceiling.**  At ``nz <= 128`` every output
   word of the shipped kernel equals the same source compiled with the old
   ``#define *_KMAX 128``, compared as raw bits.  For YSU and Shin-Hong the
   define is only a refusal guard; for KF it also sizes the two arrays kept
   on the stack for bit-identity (docs/kernel_local_memory_bounds.md section
   9) and for terrain drag the per-thread stress profiles, so for those two
   the frames are asserted DIFFERENT first -- proof the 128 arm is a
   different image and not the shipped one by another route.

Small columns counts throughout: the card is shared.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

pytestmark = pytest.mark.gpu

DEEP = 256
OLD_KMAX = 128
P_TOP_PA = 100.0


# ---------------------------------------------------------------------------
# One hydrostatic atmosphere to a 1 hPa top, sampled at any nz
# ---------------------------------------------------------------------------

def _temperature(z):
    """A US-standard-like profile: troposphere, tropopause, stratosphere."""
    z = np.asarray(z, dtype=np.float64)
    t = np.where(z < 11000.0, 300.0 - 0.0065 * z, 228.5)
    t = np.where(z >= 20000.0, 228.5 + 0.001 * (z - 20000.0), t)
    t = np.where(z >= 32000.0, 240.5 + 0.0028 * (z - 32000.0), t)
    return np.where(z >= 47000.0, 282.5, t)


def _atmosphere(nz, ncol, *, seed=11):
    """``(nz, 1, ncol)`` float64 fields, p_interface from 1000 hPa to 1 hPa.

    Interfaces are stretched in height (fine near the ground, as a real
    256-level grid is), pressure is integrated hydrostatically from the
    temperature profile, and each column gets its own small perturbation
    so neighbouring lanes take different branches.
    """
    rng = np.random.default_rng(seed + nz)
    zf = np.linspace(0.0, 60000.0, 60001)
    lnp = np.log(1.0e5) - np.concatenate((
        [0.0], np.cumsum(9.81 / (287.0 * _temperature(0.5 * (zf[1:] + zf[:-1])))
                         * np.diff(zf))))
    z_top = float(np.interp(-np.log(P_TOP_PA), -lnp, zf))
    s = np.linspace(0.0, 1.0, nz + 1)
    z_ifc = z_top * (0.35 * s + 0.65 * s ** 2.2)
    p_ifc = np.exp(np.interp(z_ifc, zf, lnp))
    p_ifc[-1] = P_TOP_PA
    z = 0.5 * (z_ifc[1:] + z_ifc[:-1])
    p = 0.5 * (p_ifc[1:] + p_ifc[:-1])
    t = _temperature(z)
    exner = (p / 1.0e5) ** (287.0 / 1004.5)

    def cols(profile, jitter=0.0):
        base = np.repeat(np.asarray(profile, np.float64)[:, None], ncol, 1)
        if jitter:
            base = base + jitter * rng.standard_normal(base.shape)
        return base[:, None, :]

    qv = (0.016 * np.exp(-z / 2500.0) + 3.0e-6)
    out = dict(
        z_interface=cols(z_ifc), dz=cols(np.diff(z_ifc)), z=cols(z),
        p_interface=cols(p_ifc), pressure=cols(p),
        temperature=cols(t, 0.6), exner=cols(exner),
        qv=np.maximum(cols(qv) * (1.0 + 0.15 * rng.standard_normal(
            (nz, 1, ncol))), 1.0e-7),
        u=cols(6.0 + 0.0015 * np.minimum(z, 20000.0), 1.5),
        v=cols(np.full(nz, -2.0), 1.5),
        w=cols(np.where(z < 2500.0, 0.02, 0.0)),
    )
    out["theta"] = out["temperature"] / out["exner"]
    return out


def _f32(cp, a):
    return cp.ascontiguousarray(cp.asarray(np.asarray(a, np.float32)))


def _surface(cp, ncol, *, seed=5, shinhong=False):
    rng = np.random.default_rng(seed)

    def f2(lo, hi):
        return _f32(cp, rng.uniform(lo, hi, (1, ncol)))

    surf = dict(psfc=_f32(cp, np.full((1, ncol), 1.0e5)), znt=f2(0.01, 0.9),
                ust=f2(0.05, 0.85), hfx=f2(-30, 250), qfx=f2(-1e-5, 2e-4),
                wspd=f2(0.7, 15), br=f2(-2.0, 0.6), psim=f2(0.4, 4.0),
                psih=f2(0.4, 4.0), xland=f2(1.0, 2.0), u10=f2(-8, 8),
                v10=f2(-8, 8))
    if shinhong:
        surf["corf"] = f2(8e-5, 1.2e-4)
    return surf


def _pbl_columns(cp, nz, ncol, *, shinhong=False):
    atm = _atmosphere(nz, ncol)
    rng = np.random.default_rng(nz)
    small = lambda lo, hi: _f32(cp, rng.uniform(lo, hi, (nz, 1, ncol)))  # noqa: E731
    col = dict(u=_f32(cp, atm["u"]), v=_f32(cp, atm["v"]),
               theta=_f32(cp, atm["theta"]), qv=_f32(cp, atm["qv"]),
               qc=small(0, 3e-5), qi=small(0, 2e-5),
               p=_f32(cp, atm["pressure"]),
               p_interface=_f32(cp, atm["p_interface"]),
               exner=_f32(cp, atm["exner"]), dz=_f32(cp, atm["dz"]))
    if shinhong:
        col["tke"] = small(0.005, 0.5)
    else:
        col["rthraten"] = small(-8e-4, 2e-4)
    return col


def _bits(out):
    import cupy as cp

    return {k: np.ascontiguousarray(cp.asnumpy(v)).copy()
            for k, v in out.items() if hasattr(v, "shape")}


def _assert_bitwise(got, want, why):
    assert got.keys() == want.keys()
    for name in sorted(got):
        assert got[name].dtype == want[name].dtype, name
        assert got[name].tobytes() == want[name].tobytes(), (
            f"{name} moved: {why}")


def _assert_ran(out, moving):
    """Finite everywhere; the named tendencies are not all zero."""
    for name, value in out.items():
        if value.dtype.kind == "f":
            assert np.isfinite(value).all(), f"{name} has non-finite words"
    for name in moving:
        assert np.any(out[name] != 0), f"{name} is identically zero"


# ---------------------------------------------------------------------------
# The old-ceiling arm: the shipped source with only the define put back
# ---------------------------------------------------------------------------

def _old_ceiling_source(source: str, define: str) -> str:
    shipped = f"#define {define} {DEEP}"
    assert source.count(shipped) == 1, f"{define} is not {DEEP} in the source"
    return source.replace(shipped, f"#define {define} {OLD_KMAX}")


def _raw_module(source: str):
    """``load_module``'s own compile call, on another source string."""
    import cupy as cp

    return cp.RawModule(code=source, options=("-std=c++17",),
                        name_expressions=None)


def _routing(module_obj, stem, old_module, served):
    """A get_kernel that serves ``stem`` from ``old_module``.

    ``served`` counts the kernels it handed out, so a test can prove the
    old-ceiling arm actually launched rather than the shipped one.
    """
    real = module_obj

    def get_kernel(name, func):
        if name == stem:
            served.append(func)
            return old_module.get_function(func)
        return real(name, func)

    return get_kernel


# ---------------------------------------------------------------------------
# YSU, regional and global
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("package", ["woof.core", "woof.globe.core"])
def test_ysu_runs_a_256_level_column_to_a_1_hpa_top(package):
    import importlib

    import cupy as cp

    Y = importlib.import_module(f"{package}.ysu")
    assert Y.VERTICAL_LEVEL_BOUNDS == (4, DEEP)
    col = _pbl_columns(cp, DEEP, 64)
    out = _bits(Y.launch_ysu(**col, **_surface(cp, 64), dt=30.0))
    assert out["du"].shape == (DEEP, 1, 64)
    _assert_ran(out, ("du", "dv", "dtheta", "exch_h", "hpbl"))
    assert (out["hpbl"] > 0).all()
    assert ((out["kpbl"] >= 1) & (out["kpbl"] < DEEP)).all()


@pytest.mark.parametrize("package", ["woof.core", "woof.globe.core"])
@pytest.mark.parametrize("nz", [49, OLD_KMAX])
def test_ysu_is_bit_identical_to_the_128_level_kernel(package, nz, monkeypatch):
    import importlib

    import cupy as cp

    Y = importlib.import_module(f"{package}.ysu")
    kernels = importlib.import_module(f"{package}.kernels")
    old = _raw_module(_old_ceiling_source(kernels.module_source("ysu"),
                                          "YSU_KMAX"))
    col = _pbl_columns(cp, nz, 96)
    surf = _surface(cp, 96)
    shipped = _bits(Y.launch_ysu(**col, **surf, dt=30.0))
    served = []
    monkeypatch.setattr(Y, "get_kernel",
                        _routing(Y.get_kernel, "ysu", old, served))
    before = _bits(Y.launch_ysu(**col, **surf, dt=30.0))
    assert served == ["ysu_column"]
    _assert_ran(shipped, ("du", "dtheta", "exch_h"))
    _assert_bitwise(shipped, before, f"YSU_KMAX 128 -> 256 at nz={nz}")


# ---------------------------------------------------------------------------
# Shin-Hong
# ---------------------------------------------------------------------------

def test_shinhong_runs_a_256_level_column_to_a_1_hpa_top():
    import cupy as cp

    from woof.core import shinhong as S

    assert S.VERTICAL_LEVEL_BOUNDS == (4, DEEP)
    col = _pbl_columns(cp, DEEP, 64, shinhong=True)
    out = _bits(S.launch_shinhong(**col, **_surface(cp, 64, shinhong=True),
                                  dt=30.0, dx=1000.0, dy=1000.0))
    assert out["du"].shape == (DEEP, 1, 64)
    _assert_ran(out, ("du", "dv", "dtheta", "exch_h", "hpbl"))
    assert ((out["kpbl"] >= 1) & (out["kpbl"] < DEEP)).all()


@pytest.mark.parametrize("nz", [49, OLD_KMAX])
def test_shinhong_is_bit_identical_to_the_128_level_kernel(nz, monkeypatch):
    import cupy as cp

    from woof.core import shinhong as S
    from woof.core.kernels import module_source

    old = _raw_module(_old_ceiling_source(module_source("shinhong"),
                                          "SHINHONG_KMAX"))
    col = _pbl_columns(cp, nz, 96, shinhong=True)
    surf = _surface(cp, 96, shinhong=True)
    kw = dict(dt=30.0, dx=1000.0, dy=1000.0)
    shipped = _bits(S.launch_shinhong(**col, **surf, **kw))
    served = []
    monkeypatch.setattr(S, "get_kernel",
                        _routing(S.get_kernel, "shinhong", old, served))
    before = _bits(S.launch_shinhong(**col, **surf, **kw))
    assert served == ["shinhong_column"]
    _assert_ran(shipped, ("du", "dtheta", "exch_h"))
    _assert_bitwise(shipped, before, f"SHINHONG_KMAX 128 -> 256 at nz={nz}")


# ---------------------------------------------------------------------------
# Kain-Fritsch
# ---------------------------------------------------------------------------

def _kf_inputs(cp, nz, ncol):
    atm = _atmosphere(nz, ncol, seed=23)
    # Grid-scale ascent through the lowest 3 km, so KF's w-dependent trigger
    # fires on most columns at every level count (it scales w by dx/25 km).
    atm["w"] = np.where(atm["z"] < 3000.0, 0.15, 0.0)
    return [_f32(cp, atm[name]) for name in (
        "u", "v", "temperature", "qv")] + [
        _f32(cp, np.zeros_like(atm["qv"]))] + [
        _f32(cp, atm[name]) for name in ("pressure", "exner", "dz", "w")]


def test_kf_runs_a_256_level_column_to_a_1_hpa_top():
    import cupy as cp

    from woof.core import kf

    assert kf.VERTICAL_LEVEL_BOUNDS == (8, DEEP)
    out = _bits(kf.launch_kf(*_kf_inputs(cp, DEEP, 64),
                             dx=12000.0, dt=60.0, cudt=300.0))
    _assert_ran(out, ("rthcuten", "rqvcuten"))
    assert set(np.unique(out["triggered"])) <= {0, 1}
    assert out["triggered"].sum() > 0, "no column convected"
    assert (out["rainc"] >= 0).all() and (out["rainc"] > 0).any()


@pytest.mark.parametrize("nz", [49, OLD_KMAX])
def test_kf_is_bit_identical_to_the_128_level_kernel(nz, monkeypatch):
    import cupy as cp

    from woof.core import kernels
    from woof.core import kf
    from woof.core.kf import KFPhaseMode

    old_fn = kernels.get_kernel_int_defines("kf", "kf_column",
                                            (("KF_KMAX", OLD_KMAX),))
    new_fn = kernels.get_kernel("kf", "kf_column")
    # The 128 arm is a different image: the stack pair is half the size.
    assert (int(old_fn.attributes["local_size_bytes"])
            < int(new_fn.attributes["local_size_bytes"]))
    real = kernels.get_kernel
    served = []

    def routed(name, func):
        if (name, func) == ("kf", "kf_column"):
            served.append(func)
            return old_fn
        return real(name, func)

    inputs = _kf_inputs(cp, nz, 96)
    triggered = 0
    for phase in KFPhaseMode:
        kw = dict(dx=12000.0, dt=60.0, cudt=300.0, phase_mode=phase)
        shipped = _bits(kf.launch_kf(*inputs, **kw))
        with monkeypatch.context() as m:
            m.setattr(kernels, "get_kernel", routed)
            before = _bits(kf.launch_kf(*inputs, **kw))
        _assert_bitwise(shipped, before,
                        f"KF_KMAX 128 -> 256 at nz={nz}, {phase.name}")
        triggered += int(shipped["triggered"].sum())
    assert len(served) == len(KFPhaseMode)
    assert triggered > 0, "no column convected, so the A/B graded nothing"


# ---------------------------------------------------------------------------
# Terrain drag (gwd_opt 1 = KIM, 3 = GSL)
# ---------------------------------------------------------------------------

def _drag_case(cp, nz, ncol, *, seed=31):
    atm = _atmosphere(nz, ncol, seed=seed)
    rng = np.random.default_rng(seed)
    atmosphere = {name: _f32(cp, atm[name]) for name in (
        "u", "v", "temperature", "qv", "pressure", "exner", "p_interface",
        "z_interface", "dz")}

    def plane(lo, hi):
        return _f32(cp, rng.uniform(lo, hi, (1, ncol)))

    def stats():
        return {"var": plane(150.0, 600.0), "con": plane(0.5, 8.0),
                "oa": [plane(-0.8, 0.8) for _ in range(4)],
                "ol": [plane(0.1, 0.9) for _ in range(4)]}

    znu = (atm["pressure"][:, 0, 0] - P_TOP_PA) / (1.0e5 - P_TOP_PA)
    return dict(atmosphere=atmosphere, ls=stats(), ss=stats(),
                sina=plane(-0.3, 0.3), cosa=plane(0.9, 1.0),
                xland=_f32(cp, np.ones((1, ncol))), br=plane(-0.5, 0.5),
                pblh=plane(300.0, 1500.0),
                kpbl=cp.asarray(rng.integers(3, 9, (1, ncol)).astype(np.int32)),
                znu=znu, du=_f32(cp, np.zeros((nz, 1, ncol))))


def _run_drag(td, case, gwd_opt):
    import cupy as cp

    du = case["du"].copy()
    dv = case["du"].copy()
    diag = {}
    if gwd_opt == 1:
        ls = case["ls"]
        td.launch_gwdo(case["atmosphere"], du, dv, var=ls["var"],
                       con=ls["con"], oa=ls["oa"], ol=ls["ol"],
                       sina=case["sina"], cosa=case["cosa"], dx=12000.0,
                       dt=60.0, diagnostics=diag)
    else:
        td.launch_gwdo_gsl(case["atmosphere"], du, dv, ls=case["ls"],
                           ss=case["ss"], sina=case["sina"],
                           cosa=case["cosa"], xland=case["xland"],
                           br=case["br"], pblh=case["pblh"],
                           kpbl=case["kpbl"],
                           kpblmax=td.gsl_kpblmax(case["znu"]), dx=15000.0,
                           dt=60.0, diagnostics=diag)
    cp.cuda.Device().synchronize()
    return _bits({"du": du, "dv": dv, **diag})


@pytest.mark.parametrize("gwd_opt", [1, 3])
def test_terrain_drag_runs_a_256_level_column_to_a_1_hpa_top(gwd_opt):
    import cupy as cp

    from woof.core import terrain_drag as td

    assert td.VERTICAL_LEVEL_BOUNDS == (None, DEEP)
    out = _run_drag(td, _drag_case(cp, DEEP, 64), gwd_opt)
    assert out["du"].shape == (DEEP, 1, 64)
    _assert_ran(out, ("du", "dv"))


def _old_drag_module():
    import cupy as cp

    from woof import nvrtc_ptx_cache as compiler
    from woof.core import terrain_drag as td

    source = _old_ceiling_source(td.module_source(), "TD_KMAX")
    ptx, _ = compiler.compile_using_nvrtc(source, td.MODULE_OPTIONS, None,
                                          "terrain_drag.cu")
    module = cp.cuda.function.Module()
    module.load(ptx.encode() if isinstance(ptx, str) else ptx)
    return module


@pytest.mark.parametrize("gwd_opt", [1, 3])
@pytest.mark.parametrize("nz", [49, OLD_KMAX])
def test_terrain_drag_is_bit_identical_to_the_128_level_kernel(
        gwd_opt, nz, monkeypatch):
    import cupy as cp

    from woof.core import terrain_drag as td

    old = _old_drag_module()
    name = "gwdo_column" if gwd_opt == 1 else "gwdo_gsl_column"

    def frame(fn):
        return int(cp.cuda.driver.funcGetAttribute(
            cp.cuda.driver.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, fn.ptr))

    # The 128 arm is a different image: its stress profiles are half size.
    assert frame(old.get_function(name)) < frame(td._kernel(name))
    case = _drag_case(cp, nz, 96)
    shipped = _run_drag(td, case, gwd_opt)
    served = []

    def routed(func):
        served.append(func)
        return old.get_function(func)

    monkeypatch.setattr(td, "_kernel", routed)
    before = _run_drag(td, case, gwd_opt)
    assert served == [name]
    _assert_ran(shipped, ("du", "dv"))
    _assert_bitwise(shipped, before,
                    f"TD_KMAX 128 -> 256 at nz={nz}, gwd_opt={gwd_opt}")
