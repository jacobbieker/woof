"""WRF-Hydro subsurface routing (the opt-in), bitwise against WRF 4.7.1's hydro/.

The port lives in ``optional/wrf_hydro`` under UCAR's WRF-Hydro licence and
loads only through :func:`woof.optin.load` with ``WOOF_ENABLE_WRF_HYDRO=1``;
every test here skips, naming that rule, when the opt-in is off.  The fixture
is ``optional/wrf_hydro/oracle/build.sh``: the pinned WRF-Hydro routines
compiled unchanged and driven for six routing calls over a 20 x 16 grid.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu
from woof import optin

REASON = optin.unavailable_reason("wrf_hydro")
pytestmark = pytest.mark.skipif(REASON is not None, reason=f"opt-in rule: {REASON}")

FIXTURES = optin.optional_root() / "wrf_hydro" / "oracle" / "fixtures"
XX, YY, NSOIL, NSTEP, DT = 20, 16, 4, 6, 900.0
SLDPTH = (0.1, 0.3, 0.6, 1.0)
ZSOIL = (-0.1, -0.4, -1.0, -2.0)


def _rows(name):
    with (FIXTURES / name).open(newline="", encoding="ascii") as fh:
        return list(csv.DictReader(fh))


def _plane(rows, key, dtype=np.float32):
    a = np.zeros((YY, XX), dtype)
    for r in rows:
        a[int(r["j"]) - 1, int(r["i"]) - 1] = dtype(r[key]) if dtype is np.float32 else int(r[key])
    return a


def _stack(rows, keys, dtype=np.float32):
    return np.ascontiguousarray(np.stack([_plane(rows, k, dtype) for k in keys]))


def _bits(a):
    return np.ascontiguousarray(a, np.float32).view(np.uint32)


def test_fixtures_are_the_ones_build_sh_hashed():
    rows = (FIXTURES / "oracle-sha256sums.txt").read_text().splitlines()
    pinned = {line.split()[1]: line.split()[0] for line in rows}
    for name in ("subsurface-inputs.csv", "subsurface-steps.csv",
                 "subsurface-totals.csv"):
        assert pinned[name] == hashlib.sha256((FIXTURES / name).read_bytes()).hexdigest()


def test_fixture_reaches_outflow_drainage_and_the_spill():
    inp = _rows("subsurface-inputs.csv")
    last = [r for r in _rows("subsurface-steps.csv") if r["step"] == str(NSTEP)]
    q = np.array([float(r["qsubrt"]) for r in last])
    assert (q > 0).any() and (q < 0).any()
    assert any(float(r["qsubbdryrt"]) != 0 for r in last)
    infx0 = {(r["i"], r["j"]): float(r["infxs"]) for r in inp}
    assert any(float(r["infxs"]) > infx0[(r["i"], r["j"])] for r in last)


def test_rt_option_2_is_refused_by_name():
    mod = optin.load("wrf_hydro")
    with pytest.raises(NotImplementedError, match="ROUTE_SUBSURFACE2"):
        mod.SubsurfaceRouting(dt=DT, sldpth=SLDPTH, zsoil=ZSOIL, lksat=None,
                              nexp=None, soldep=None, slope=None, sdir=None,
                              dist=None, rt_option=2)


@pytest.mark.gpu
@requires_gpu
def test_subsurface_routing_is_bitwise_against_wrf_hydro():
    import cupy as cp

    mod = optin.load("wrf_hydro")
    inp = _rows("subsurface-inputs.csv")
    dev = lambda a: cp.asarray(a)  # noqa: E731
    routing = mod.SubsurfaceRouting(
        dt=DT, sldpth=SLDPTH, zsoil=ZSOIL,
        lksat=dev(_plane(inp, "lksat")), nexp=dev(_plane(inp, "nexp")),
        soldep=dev(_plane(inp, "soldep")),
        slope=dev(_stack(inp, [f"slope{k}" for k in range(1, 9)])),
        sdir=dev(_stack(inp, ["sdir_i", "sdir_j", "sdir_k"], np.int32)),
        dist=dev(_stack(inp, [f"dist{k}" for k in range(1, 10)])))
    smc = dev(_stack(inp, [f"smc{k}" for k in range(1, 5)]))
    soil = {name: dev(_stack(inp, [f"{name}{k}" for k in range(1, 5)]))
            for name in ("smcmax", "smcref", "smcwlt")}
    infx = dev(_plane(inp, "infxs"))
    state = mod.SubsurfaceState.zeros(XX, YY)
    steps = _rows("subsurface-steps.csv")
    totals = {int(r["step"]): np.float32(r["qsubbdrytrt"]) for r in _rows("subsurface-totals.csv")}
    for n in range(1, NSTEP + 1):
        routing.route(state, smc=smc, infiltration_excess=infx, **soil)
        want = [r for r in steps if r["step"] == str(n)]
        got = {"zwattablrt": state.zwattablrt, "qsubrt": state.qsubrt,
               "qsubbdryrt": state.qsubbdryrt, "infxs": infx}
        for key, arr in got.items():
            assert np.array_equal(_bits(cp.asnumpy(arr)), _bits(_plane(want, key))), (n, key)
        assert np.array_equal(_bits(cp.asnumpy(smc)),
                              _bits(_stack(want, [f"smc{k}" for k in range(1, 5)]))), (n, "smc")
        assert _bits(cp.asnumpy(state.qsubbdrytrt))[0] == _bits(np.array([totals[n]]))[0], n
        state.end_hydro_step()
