"""WRF-Hydro overland routing (the opt-in), bitwise against WRF 4.6.1's hydro/.

``OverlandRouting`` with rt_option = 1, run by ``woof_wrf_hydro.overland``.
The fixture is ``optional/wrf_hydro/oracle/build_overland.sh``: the pinned
routines compiled unchanged, three calls of six 20 s substeps over a 20 x 16
grid with a channel line, a lake, a dry patch, a deep pit and a head mound
that sends GETMAX8DIR looking for another outlet (an instrumented build found
both outcomes: 50 found, 18 none).  Every test skips, naming the opt-in rule,
unless ``WOOF_ENABLE_WRF_HYDRO=1``.
"""

from __future__ import annotations

import csv
import hashlib

import numpy as np
import pytest

from conftest import requires_gpu
from woof import optin

REASON = optin.unavailable_reason("wrf_hydro")
pytestmark = pytest.mark.skipif(REASON is not None, reason=f"opt-in rule: {REASON}")

FIX = optin.optional_root() / "wrf_hydro" / "oracle" / "fixtures" / "overland"
XX, YY, DT, DTRT = 20, 16, 120.0, 20.0


def _rows(name):
    with (FIX / name).open(newline="") as fh:
        return list(csv.DictReader(fh))


def _plane(rows, key, dtype=np.float32):
    a = np.zeros((YY, XX), dtype)
    for r in rows:
        a[int(r["j"]) - 1, int(r["i"]) - 1] = np.float32(r[key]) if dtype is np.float32 else int(r[key])
    return a


def _stack(rows, keys, dtype=np.float32):
    return np.ascontiguousarray(np.stack([_plane(rows, k, dtype) for k in keys]))


def _bits(a):
    return np.ascontiguousarray(a, np.float32).view(np.uint32)


def test_fixtures_are_the_ones_the_build_hashed():
    rows = (FIX / "overland-sha256sums.txt").read_text().splitlines()
    pinned = {line.split()[1]: line.split()[0] for line in rows}
    for name in ("overland-inputs.csv", "overland-steps.csv", "overland-totals.csv"):
        assert pinned[name] == hashlib.sha256((FIX / name).read_bytes()).hexdigest()


def test_rt_option_2_is_refused_by_name():
    mod = optin.load("wrf_hydro")
    with pytest.raises(NotImplementedError, match="ROUTE_OVERLAND2"):
        mod.OverlandRouting(dist=None, slope=None, sdir=None, roughness=None,
                            ch_netrt=None, lake_mask=None, rt_option=2)


@pytest.mark.gpu
@requires_gpu
def test_overland_routing_is_bitwise_against_wrf_hydro():
    import cupy as cp

    mod = optin.load("wrf_hydro")
    inp = _rows("overland-inputs.csv")
    d = cp.asarray
    ov = mod.OverlandRouting(
        dist=d(_stack(inp, [f"dist{k}" for k in range(1, 10)])),
        slope=d(_stack(inp, [f"slope{k}" for k in range(1, 9)])),
        sdir=d(_stack(inp, ["sdir_i", "sdir_j", "sdir_k"], np.int32)),
        roughness=d(_plane(inp, "roughness")), ch_netrt=d(_plane(inp, "ch_netrt", np.int32)),
        lake_mask=d(_plane(inp, "lake", np.int32)))
    z = lambda: cp.zeros((YY, XX), cp.float32)  # noqa: E731
    state = mod.OverlandState(head=z(), retention=d(_plane(inp, "retention")), to_channel=z(),
                              to_lake=z(), boundary_flux=z(), totals=cp.zeros((3,), cp.float32))
    qx, qy = z(), z()
    steps = _rows("overland-steps.csv")
    totals = _rows("overland-totals.csv")
    for n in (1, 2, 3):
        rows = [r for r in steps if r["call"] == str(n)]
        ov.route(state, d(_plane(rows, "infxs")), qx, qy, dt=DT, dtrt_ter=DTRT)
        for key, arr in (("head", state.head), ("retention", state.retention), ("qx", qx),
                         ("qy", qy), ("to_channel", state.to_channel), ("to_lake", state.to_lake),
                         ("boundary_flux", state.boundary_flux)):
            assert np.array_equal(_bits(cp.asnumpy(arr)), _bits(_plane(rows, key))), (n, key)
        t = totals[n - 1]
        want = np.array([t["acc_channel"], t["acc_lake"], t["boundary_flux_total"]], np.float32)
        assert np.array_equal(_bits(cp.asnumpy(state.totals)), _bits(want)), (n, "totals")
