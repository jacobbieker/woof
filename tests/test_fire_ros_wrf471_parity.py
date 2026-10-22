"""WRF v4.7.1 fire spread: the CPU and GPU ports against WRF's own Fortran.

The fixture is ``phys/module_fr_fire_phys.F`` (with ``module_fr_fire_util.F``
and ``share/module_model_constants.F``), byte-unmodified and pinned by
sha256, compiled at -O0 by ``tools/fire_ros_wrf471_oracle/build.sh``
(``TOOLCHAIN.txt``) and run by ``run_fire_ros.F90`` in two configurations:

* ``namelist`` -- beside the pinned ``test/em_fire/namelist.fire``: the 13
  Anderson categories, no-fuel 14 and two numbers ksb maps to no fuel;
* ``defaults`` -- no namelist.fire: the compiled DATA, which add the Scott &
  Burgan 40 as one-particle categories (SB4 excepted: WRF rejects it).

Each crosses its categories with 28 surface moistures (0, low, high, and
every moisture of extinction with both float32 neighbours) for
``set_fire_params``, and every cell with a sweep of wind (calm, along,
against, across, beyond the 30 m/s cap), slope (flat, up, down, cross) and
fire-line normal (four unit normals and a non-unit one) for ``fire_ros``,
under ``fire_advection`` 0 and 1.

Both ports must equal every recorded word (max ULP 0).  One word is held
out: ``ros_base`` on chaparral cells, which WRF's ``fire_ros`` never
assigns (the driver passes 0.0 in and the port returns 0.0; PROVENANCE.md).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu
from woof.fire import wrf_sfire as W

ROOT = Path(__file__).resolve().parent / "data" / "oracles" / "fire_ros"
TOOLS = Path(__file__).resolve().parents[1] / "tools" / "fire_ros_wrf471_oracle"
CONFIGS = ("namelist", "defaults")
PARAMS = ("fuel_time", "fgip", "ischap", "betafl", "bbb", "phiwc", "r_0",
          "iboros")
ROS = ("ros_base", "ros_wind", "ros_slope")


def _load(config):
    from woof.verify.urban_oracle import load_case
    return load_case(ROOT / config)


def _grid(o):
    cats, fm = o["nfuel_cat"], o["fmc_g"]
    cat = np.repeat(cats[:, None], fm.size, axis=1).astype(np.float32)
    fmc = np.repeat(fm[None, :], cats.size, axis=0)
    return cat, fmc


def _words(a):
    return np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)


def _ulp(a, b):
    from woof.core.fp32_ulp import fp32_ulp_distance
    return np.asarray(fp32_ulp_distance(np.asarray(a, np.float32),
                                        np.asarray(b, np.float32)))


def _held_out(o, name):
    """The words compared for ``name``: all, except ros_base on chaparral."""
    if name != "ros_base":
        return np.ones(o[name].shape, dtype=bool)
    chap = o["ischap"] > 0
    chap = chap.reshape(chap.shape + (1,) * (o[name].ndim - 2))
    return ~np.broadcast_to(chap, o[name].shape)


def _ros(p, o, adv, xp=np):
    """fire_ros over every (cell, case) pair at once: (nc, nm, ncase)."""
    c = xp.asarray(o["cases"])
    vec = [c[q][None, None, :] for q in range(6)]
    expand = W.FireParams(**{k: getattr(p, k)[:, :, None] for k in PARAMS})
    return W.fire_ros(expand, *vec, fire_advection=adv)


def _report(o, name, got):
    keep = _held_out(o, name)
    ref = o[name]
    return (int(keep.sum()), int(_ulp(got, ref)[keep].max()),
            int(np.count_nonzero(_words(got)[keep] != _words(ref)[keep])))


# --------------------------------------------------------------------------
# the fixture is what the harness says it is
# --------------------------------------------------------------------------

def test_fixture_hashes_and_pins():
    lines = (ROOT / "FIXTURES.sha256").read_text().split("\n")
    seen = 0
    for line in filter(None, lines):
        digest, rel = line.split(maxsplit=1)
        assert hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == digest, rel
        seen += 1
    # per configuration: nfuel_cat, fmc_g, cases, the outputs, MANIFEST.txt
    assert seen == 2 * (3 + len(PARAMS) + len(ROS) + 1)
    pinned = [line for line in (TOOLS / "SOURCES.sha256").read_text().splitlines()
              if line and not line.startswith("#")]
    assert (ROOT / "SOURCES.sha256").read_text().splitlines() == pinned
    for line in (ROOT / "HARNESS.sha256").read_text().splitlines():
        digest, rel = line.split(maxsplit=1)
        assert hashlib.sha256((TOOLS / rel).read_bytes()).hexdigest() == digest, (
            f"{rel} changed since the fixture was built; rerun build.sh")
    toolchain = (ROOT / "TOOLCHAIN.txt").read_text()
    assert "gfortran" in toolchain and "glibc" in toolchain
    report = (ROOT / "libmvec-report.txt").read_text()
    assert "_ZGV" in report.split("# positive control")[1]
    assert "_ZGV" not in report.split("# phys/module_fr_fire_phys.F at")[0]


def test_sweep_reaches_every_branch():
    """The fixture exercises what the port must get right."""
    nl, df = _load("namelist"), _load("defaults")
    for o in (nl, df):
        cases = o["cases"]
        # backing: wind against at least one normal; calm; beyond 30 m/s
        dot = cases[0] * cases[4] + cases[1] * cases[5]
        assert (dot < 0).any() and (dot > 0).any()
        assert ((cases[0] == 0) & (cases[1] == 0)).any()
        # slope up and down the normal
        sdot = cases[2] * cases[4] + cases[3] * cases[5]
        assert (sdot > 0).any() and (sdot < 0).any()
        # moisture above extinction: r_0 < 0; chaparral; no fuel
        assert (o["r_0"] < 0).any() and (o["r_0"] > 0).any()
        assert (o["ischap"] > 0).any()
        assert ((o["r_0"] == 0) & (o["betafl"] == 1)).any()
        # the 6 m/s cap's sequential trim, including a negative slope term
        total = o["ros_base"] + o["ros_wind"] + o["ros_slope"]
        assert (np.abs(total - 6.0) < 1e-3).any()
        assert ((o["ros_slope"] < 0) & (o["ros_base"] > 0)).any()
    # gfortran MAX's tie: wind . n == -0.0, so max(0., -0.) decides the
    # sign of a zero ros_wind under fire_advection = 1
    v = nl["cases"].astype(np.float32)
    dot = v[0] * v[4] + v[1] * v[5]
    assert ((dot == 0) & np.signbit(dot)).any()
    # wind beyond fire_ros' 30 m/s cap
    assert (np.hypot(nl["cases"][0], nl["cases"][1]) > 30).any()
    # a non-unit normal (WRF v4.7.1 does not normalise propx/propy)
    assert (np.abs(np.hypot(df["cases"][4], df["cases"][5]) - 1) > 0.1).any()
    # the Scott & Burgan categories
    assert set(range(101, 110)) <= set(df["nfuel_cat"].tolist())


def test_held_out_set_is_exactly_chaparral_ros_base():
    for config in CONFIGS:
        o = _load(config)
        chap_rows = np.unique(np.argwhere(o["ischap"] > 0)[:, 0])
        assert o["nfuel_cat"][chap_rows].tolist() == [4]
        held = ~_held_out(o, "ros_base")
        assert held.sum() == (o["ischap"] > 0).sum() * np.prod(
            o["ros_base"].shape[2:])
        # The driver passed 0.0 there, and that is what WRF left.
        np.testing.assert_array_equal(o["ros_base"][held], 0.0)


def test_vectorised_expf_equals_the_scalar_glibc_transcription():
    """wrf_sfire._expf is noahmp_libm.expf's core over arrays; word for word
    on a dense sample, both special-case bands and the edges."""
    from woof.core import noahmp_libm
    rng = np.random.default_rng(5)
    x = np.concatenate([
        rng.uniform(-110.0, 110.0, 200_000).astype(np.float32),
        rng.uniform(-1.0, 1.0, 50_000).astype(np.float32),
        (rng.integers(0, 2**32, 50_000, dtype=np.uint64)
         .astype(np.uint32).view(np.float32)),
        np.float32([0.0, -0.0, np.inf, -np.inf, np.nan, 88.0, -88.0,
                    88.72283, 88.72284, -103.97208, -103.97209, 1e-45]),
    ])
    with np.errstate(invalid="ignore", over="ignore"):
        want = np.fromiter((noahmp_libm.expf(v) for v in x), np.float32,
                           x.size)
    got = W._expf(x)
    nan = np.isnan(want)
    assert np.array_equal(np.isnan(got), nan)
    assert np.array_equal(_words(got)[~nan], _words(want)[~nan])


def test_kernel_double_literals_are_the_fortran_real4_literals():
    """fire_spread.cu's fire_ros template spells WRF's literals in double;
    each rounds to the same float32 as gfortran's real(4) literal."""
    for text in ("30.", "196.850", "5.275", "-0.3", ".00508", ".03333",
                 "1.2974", "1.41", "6.0"):
        assert np.float32(float(text)).view(np.uint32) == W.r4(text).view(
            np.uint32), text


# --------------------------------------------------------------------------
# CPU and GPU against the oracle
# --------------------------------------------------------------------------

@pytest.mark.parametrize("config", CONFIGS)
def test_cpu_set_fire_params_equals_wrf(config):
    o = _load(config)
    cat, fmc = _grid(o)
    p = W.set_fire_params(cat, fmc, table=config)
    for name in PARAMS:
        words, max_ulp, differ = _report(o, name, getattr(p, name))
        assert (max_ulp, differ) == (0, 0), (config, name, max_ulp, differ)


@pytest.mark.parametrize("config", CONFIGS)
@pytest.mark.parametrize("adv", (0, 1))
def test_cpu_fire_ros_equals_wrf(config, adv):
    o = _load(config)
    cat, fmc = _grid(o)
    p = W.set_fire_params(cat, fmc, table=config)
    for name, got in zip(ROS, _ros(p, o, adv)):
        sub = {name: o[name][..., adv], "ischap": o["ischap"]}
        words, max_ulp, differ = _report(sub, name, got)
        assert (max_ulp, differ) == (0, 0), (config, adv, name, max_ulp, differ)


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("config", CONFIGS)
@pytest.mark.parametrize("adv", (0, 1))
def test_gpu_equals_wrf(config, adv):
    import cupy as cp
    o = _load(config)
    cat, fmc = _grid(o)
    p = W.set_fire_params(cp.asarray(cat), cp.asarray(fmc), table=config)
    for name in PARAMS:
        words, max_ulp, differ = _report(o, name, cp.asnumpy(getattr(p, name)))
        assert (max_ulp, differ) == (0, 0), (config, name, max_ulp, differ)
    for name, got in zip(ROS, _ros(p, o, adv, xp=cp)):
        sub = {name: o[name][..., adv], "ischap": o["ischap"]}
        words, max_ulp, differ = _report(sub, name, cp.asnumpy(got))
        assert (max_ulp, differ) == (0, 0), (config, adv, name, max_ulp, differ)
