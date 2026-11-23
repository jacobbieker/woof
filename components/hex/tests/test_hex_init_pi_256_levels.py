"""Hex member init and the TC potential-intensity diagnostic on 256-level columns.

``woof/hex/init_gpu/physics.cu`` sizes its per-thread columns with ``MAX_FG``
and ``MAX_Z``, and ``woof/hex/tc_climate/potential_intensity.cu`` sizes its
sounding with ``PI_MAX_LEV``.  All three default to 128, the shipped tier.
Extents of at most 128 compile the unchanged text.  Deeper extents prepend
one ``#define`` per bound for the 256 tier.  Claims:

1. The tier selection: no define at or below 128, one define per deeper
   extent, refusal above 256 (no device).
2. The PI source with the ``PI_MAX_LEV`` macro preprocesses to the same text
   as the literal-128 source it replaced (no device).
3. CPU algebra: the deep PI build reproduces the shipped build's words on
   the reference sounding, and a 200-level resampling of that sounding gives
   the reference PI to within the resampling error.
4. On a card: the deep init build reproduces the shipped build's words at a
   64-level column; a 256-level column is finite and hydrostatic; the PI
   kernel matches the CPU algebra at 200 levels.
"""
from __future__ import annotations

import ctypes
import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from woof.hex.init_gpu import physics
from woof.hex.tc_climate import session

PI_DIR = Path(session.__file__).parent
DEEP = 256


@pytest.mark.parametrize("extents,expected", [
    ((49, 55, 4), []),
    ((128, 128, 128), []),
    ((49, 129, 4), ["#define MAX_Z 256"]),
    ((129, 55, 4), ["#define MAX_FG 256"]),
    ((49, 55, 129), ["#define MAX_FG 256"]),
    ((200, 256, 4), ["#define MAX_FG 256", "#define MAX_Z 256"]),
])
def test_the_init_tier_defines(extents, expected):
    assert physics.tier_defines(*extents) == expected


@pytest.mark.parametrize("extents", [(49, 257, 4), (257, 55, 4), (49, 55, 257), (1, 55, 4), (49, 1, 4)])
def test_the_init_refuses_outside_its_tiers(extents):
    with pytest.raises(ValueError, match="unsupported column extent"):
        physics.tier_defines(*extents)


def test_the_shipped_init_source_is_the_unchanged_text(tmp_path):
    kernels = tmp_path / "woof/core/kernels"
    kernels.mkdir(parents=True)
    for name in ("glibc_flt32.cuh", "glibc_trig_flt32.cuh"):
        (kernels / name).write_text(f"// {name}\n")
    text = Path(physics.__file__).with_name("physics.cu").read_text()
    headers = "// glibc_flt32.cuh\n\n// glibc_trig_flt32.cuh\n"
    source, _ = physics.physics_source(49, 128, 4, engine_root=tmp_path)
    assert source == headers + "\n" + text
    deep, _ = physics.physics_source(49, DEEP, 4, engine_root=tmp_path)
    assert deep == headers + "\n" + "#define MAX_Z 256\n" + text


def test_the_pi_tier_ladder():
    assert session.pi_level_tier(55) == session.pi_level_tier(128) == 128
    assert session.pi_level_tier(129) == session.pi_level_tier(DEEP) == DEEP
    with pytest.raises(ValueError, match="at most 256 levels"):
        session.pi_level_tier(DEEP + 1)
    text = (PI_DIR / "potential_intensity.cu").read_text()
    assert session.pi_source(128) == text
    assert session.pi_source(DEEP) == "#define PI_MAX_LEV 256\n" + text


def _literal_source():
    """The PI source as it was before the macro: 128 written out."""
    text = (PI_DIR / "potential_intensity.cu").read_text()
    text = re.sub(r"#ifndef PI_MAX_LEV\n#define PI_MAX_LEV 128\n#endif\n", "", text)
    assert "#define PI_MAX_LEV" not in text
    return text.replace("PI_MAX_LEV", "128")


def test_the_pi_macro_preprocesses_to_the_literal_source(tmp_path):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("a C++ preprocessor is needed")
    out = {}
    for label, text in (("macro", (PI_DIR / "potential_intensity.cu").read_text()),
                        ("literal", _literal_source())):
        path = tmp_path / f"{label}.cu"
        path.write_text(text)
        out[label] = subprocess.run(
            [compiler, "-E", "-P", "-x", "c++", "-D__CUDACC_RTC__", "-Dnanf(x)=0", str(path)],
            check=True, capture_output=True, text=True).stdout
    assert out["macro"] == out["literal"]


@pytest.fixture(scope="module")
def algebra(tmp_path_factory):
    """The PI kernel compiled for the host at both tiers (CPU algebra only)."""
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ needed for CPU algebra check")
    root = tmp_path_factory.mktemp("pi-tiers")
    libs = {}
    for tier in (128, DEEP):
        cpp = root / f"pi{tier}.cpp"
        cpp.write_text(f"#define PI_MAX_LEV {tier}\n" + '''#include <cmath>
#define __device__
#define __global__
struct Index { int x=0; };
Index blockDim{1}, blockIdx, threadIdx;
#include "''' + str(PI_DIR / "potential_intensity.cu") + '''"
extern "C" void pi_one(int nz,const float* p,const float* t,const float* r,
 float sst,float slp,int ocean,float* out,int* status) {
 blockIdx.x=threadIdx.x=0;
 potential_intensity(1,nz,p,t,r,&sst,&slp,&ocean,.9f,.8f,50.f,out,status);
}
''')
        lib = root / f"pi{tier}.so"
        subprocess.run([compiler, "-std=c++17", "-O0", "-ffp-contract=off", "-shared", "-fPIC",
                        str(cpp), "-o", str(lib)], check=True, capture_output=True, text=True)
        dll = ctypes.CDLL(str(lib))
        f = ctypes.POINTER(ctypes.c_float)
        i = ctypes.POINTER(ctypes.c_int)
        dll.pi_one.argtypes = [ctypes.c_int, f, f, f, ctypes.c_float, ctypes.c_float, ctypes.c_int, f, i]
        libs[tier] = dll
    return libs


def _ptr(a):
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_float if a.dtype == np.float32 else ctypes.c_int))


def _reference():
    ref = json.loads((PI_DIR / "pi-reference.json").read_text())
    return ref, [np.array(ref[k], dtype=np.float32) for k in ("pressure_pa", "temperature_k", "vapor_kgkg")]


def _resampled(levels):
    """The reference sounding linearly resampled in log pressure onto ``levels`` levels."""
    _, (p, t, r) = _reference()
    lnp = np.log(p.astype(np.float64))
    target = np.linspace(lnp[0], lnp[-1], levels)
    # np.interp needs ascending abscissae; pressure falls with height.
    tt = np.interp(-target, -lnp, t.astype(np.float64))
    rr = np.interp(-target, -lnp, r.astype(np.float64))
    return [np.exp(target).astype(np.float32), tt.astype(np.float32), rr.astype(np.float32)]


def _pi(dll, p, t, r, sst):
    pi = np.zeros(1, np.float32)
    status = np.zeros(1, np.int32)
    dll.pi_one(len(p), _ptr(p), _ptr(t), _ptr(r), sst, 101000, 1, _ptr(pi), _ptr(status))
    return pi[0], int(status[0])


def test_the_deep_pi_build_reproduces_the_shipped_words(algebra):
    ref, (p, t, r) = _reference()
    for row in ref["expected"]:
        shipped = _pi(algebra[128], p, t, r, row["sst_k"])
        deep = _pi(algebra[DEEP], p, t, r, row["sst_k"])
        assert shipped[1] == deep[1] == row["status"]
        assert np.float32(shipped[0]).tobytes() == np.float32(deep[0]).tobytes()


def test_a_200_level_sounding_gives_the_reference_pi(algebra):
    ref, _ = _reference()
    p, t, r = _resampled(200)
    for row in ref["expected"]:
        pi, status = _pi(algebra[DEEP], p, t, r, row["sst_k"])
        assert status == 1
        # Finer levels move CAPE by the trapezoid error only.
        assert pi == pytest.approx(row["pi_ms"], abs=0.5), (row, pi)  # measured 0.16 to 0.38
        # The shipped build refuses it at its level guard (PI left NaN,
        # status 0) although the same SST and ocean point give status 1 above.
        shipped_pi, shipped_status = _pi(algebra[128], p, t, r, row["sst_k"])
        assert shipped_status == 0 and np.isnan(shipped_pi), "the shipped build refuses it"


# ---------------------------------------------------------------------------
# On a card
# ---------------------------------------------------------------------------

def _column(nz, ncell=4):
    """A 30 km column on a uniform height grid, close to hydrostatic balance."""
    z = np.linspace(0.0, 30000.0, nz + 1)
    zmid = 0.5 * (z[1:] + z[:-1])
    t = np.maximum(288.15 - 0.0065 * zmid, 216.65)
    p = 101325.0 * np.exp(-9.80616 * zmid / (287.0 * 250.0))
    dz = np.diff(z)
    dzu = np.concatenate(([dz[0]], 0.5 * (dz[1:] + dz[:-1])))
    fz = np.full(nz, 0.5)
    rep = lambda a: np.ascontiguousarray(np.broadcast_to(a, (ncell,) + a.shape), "f4")
    return dict(t=rep(t)[None], p=rep(p)[None], rh=np.full((1, ncell, nz), 50.0, "f4"),
                sh=np.zeros((1, ncell, nz), "f4"), zgrid=rep(z), zz=np.ones((ncell, nz), "f4"),
                fzm=fz.astype("f4"), fzp=fz.astype("f4"), dzu=dzu.astype("f4"), rdzw0=np.float32(1 / dz[0]))


def _dynamics(cp, module, col, nz, ncell=4):
    args = {k: cp.asarray(v) for k, v in col.items() if k != "rdzw0"}
    shape = (1, ncell, nz)
    out = {n: cp.empty(shape, "f4") for n in ("qv", "theta", "rho", "rho_base", "theta_base", "rho_zz", "relhum")}
    out.update({n: cp.empty((1, ncell), "f4") for n in ("precipw", "surface_pressure")})
    iterations = cp.empty(shape, "i4")
    module.get_function("dynamics_cells")(((ncell + 127) // 128,), (128,), (
        args["t"], args["p"], args["rh"], args["sh"], args["zgrid"], args["zz"], args["fzm"], args["fzp"],
        args["dzu"], col["rdzw0"], cp.zeros(1, "i4"), np.int32(ncell), np.int32(1), np.int32(nz),
        np.float32(1.61), *[out[n] for n in ("qv", "theta", "rho", "rho_base", "theta_base", "rho_zz",
                                             "precipw", "surface_pressure", "relhum")], iterations))
    res = {k: cp.asnumpy(v) for k, v in out.items()}
    res["iterations"] = cp.asnumpy(iterations)
    return res


@pytest.mark.gpu
def test_the_deep_init_build_reproduces_the_shipped_words_and_runs_256_levels():
    cp = pytest.importorskip("cupy")
    from woof.hex import init_gpu

    kernels = init_gpu.Kernels()
    shipped_source, _ = physics.physics_source(49, 64, 4)
    deep_source, _ = physics.physics_source(49, DEEP, 4)
    shipped, deep = kernels.compile(shipped_source), kernels.compile(deep_source)
    col = _column(64)
    a, b = _dynamics(cp, shipped, col, 64), _dynamics(cp, deep, col, 64)
    for name in sorted(a):
        assert a[name].tobytes() == b[name].tobytes(), f"{name} moved: MAX_Z 128 -> 256"

    out = _dynamics(cp, deep, _column(DEEP), DEEP)
    for name, value in out.items():
        assert np.isfinite(value).all(), name
    assert (out["iterations"] < 30).all(), "hydrostatic adjustment did not converge"
    rho = out["rho"][0, 0]
    assert (rho > 0).all() and (np.diff(rho) < 0).all(), "density must fall with height"
    assert 9.0e4 < float(out["surface_pressure"].min()) < 1.1e5


@pytest.mark.gpu
def test_the_pi_kernel_matches_the_cpu_algebra_at_200_levels(algebra):
    cp = pytest.importorskip("cupy")
    from woof import backend
    if backend.active() != "cuda":
        # The PI module is binary64 throughout (and calls remainder()); Metal
        # has no double, so the module does not build there at any tier
        # (test_tc_climate's resident PI test fails the same way on main).
        pytest.skip("the TC PI kernel is binary64 and CUDA-only")
    ref, _ = _reference()
    p, t, r = _resampled(200)
    module = cp.RawModule(code=session.pi_source(session.pi_level_tier(200)), options=("--std=c++17",))
    fields = [cp.asarray(a) for a in (p, t, r)]
    row = ref["expected"][1]
    pi = cp.empty(1, cp.float32)
    status = cp.empty(1, cp.int32)
    module.get_function("potential_intensity")((1,), (1,), (
        1, len(p), *fields, cp.asarray([row["sst_k"]], dtype=cp.float32),
        cp.asarray([101000], dtype=cp.float32), cp.ones(1, cp.int32),
        cp.float32(.9), cp.float32(.8), cp.float32(50), pi, status))
    assert int(cp.asnumpy(status)[0]) == 1
    expected, _ = _pi(algebra[DEEP], p, t, r, row["sst_k"])
    assert float(cp.asnumpy(pi)[0]) == pytest.approx(float(expected), abs=.003)
