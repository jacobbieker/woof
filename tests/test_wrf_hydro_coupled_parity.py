"""The coupled WRF/WRF-Hydro exchange (the opt-in), bitwise against WRF 4.7.1.

The fixture is ``optional/wrf_hydro/oracle/build_coupled.sh``: WRF 4.7.1's
own ``wrf_cpl_HYDRO`` (hydro/CPL/WRF_cpl/module_wrf_HYDRO.F90) over the
whole hydro/ tree -- HYDRO_ini reading the synthetic Fulldom_hires.nc,
hydro2dtbl.nc, LAKEPARM.nc, GWBASINS.nc, GWBUCKPARM.nc and CHANPARM.TBL
(tests/wrf_hydro_synthetic.py), then HYDRO_exe -- compiled whole with
MPP_LAND on one process and driven over four coupling cycles with the LSM
fields of ``gen_coupled_inputs.cycles``, once with the lake a level pool
once (variant "da") as lake_option 3's USGS persistence/level-pool
hybrid reading its timeslices, and once (variant "gw2d") with GWBASESWCRT = 3's
2-D aquifer from gwhires.nc in place of the bucket, coupled to the soil and
the channel (AGGFACTRT = 1; the harness stands in the zero-length bucket
arrays WRF 4.7.1's HYDRO_ini otherwise faults on).  Graded: the network, slopes,
steepest-descent indices, retention depths and lake mask HYDRO_ini built;
after every cycle the SMOIS, SH2O and SFCHEADRT handed back to WRF and the
routing state (routing-grid head, water table, QSTRMVOLRT, QINFLOWBASE,
lake inflow, channel flow, head and volume, lake elevation and outflow,
bucket level and outflow, DTCT; for gw2d the aquifer head, surface excess,
soil and channel fluxes and the routing grid's soil moisture).  Every test skips, naming the opt-in rule,
unless ``WOOF_ENABLE_WRF_HYDRO=1``.
"""

from __future__ import annotations

import csv
import datetime
import hashlib
import sys
from collections import defaultdict

import numpy as np
import pytest

from conftest import requires_gpu
from woof import optin

REASON = optin.unavailable_reason("wrf_hydro")
pytestmark = pytest.mark.skipif(REASON is not None, reason=f"opt-in rule: {REASON}")

ORACLE = optin.optional_root() / "wrf_hydro" / "oracle"
FIX = ORACLE / "fixtures" / "coupled"


def _gen():
    if str(ORACLE) not in sys.path:
        sys.path.insert(0, str(ORACLE))
    import gen_coupled_inputs

    return gen_coupled_inputs


#: the run as built (level-pool lake), lake_option 3's variant (the lake a
#: USGS persistence/level-pool hybrid) and GWBASESWCRT = 3's (the 2-D
#: aquifer): steps fixture, sums file, static fixture
VARIANTS = {"": ("coupled-steps.csv", "coupled-sha256sums.txt", "coupled-static.csv"),
            "da": ("coupled-da-steps.csv", "coupled-da-sha256sums.txt", "coupled-static.csv"),
            "gw2d": ("coupled-gw2d-steps.csv", "coupled-gw2d-sha256sums.txt",
                     "coupled-gw2d-static.csv")}


def _sums(variant=""):
    rows = (FIX / VARIANTS[variant][1]).read_text().splitlines()
    return {line.split()[1].rsplit("/", 1)[-1]: line.split()[0] for line in rows}


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_fixtures_are_the_ones_the_build_hashed(tmp_path, variant):
    sums = _sums(variant)
    assert sums["coupled-steps.csv"] == hashlib.sha256(
        (FIX / VARIANTS[variant][0]).read_bytes()).hexdigest()
    assert sums["coupled-static.csv"] == hashlib.sha256(
        (FIX / VARIANTS[variant][2]).read_bytes()).hexdigest()
    for name in ("coupled_harness.F90", "coupled_wrf_stubs.F90", "build_coupled.sh",
                 "gen_coupled_inputs.py"):
        assert sums[name] == hashlib.sha256((ORACLE / name).read_bytes()).hexdigest(), name
    # the inputs regenerate byte for byte
    _gen().write(tmp_path, variant)
    for name in ("coupled-inputs.txt", "hydro.namelist"):
        assert sums[name] == hashlib.sha256((tmp_path / name).read_bytes()).hexdigest(), name


def _coupled(tmp_path_factory, variant):
    from importlib import import_module

    from woof.wrf_hydro_route import soil_centres

    g = _gen()
    d = tmp_path_factory.mktemp("coupled")
    g.write(d, variant)
    veg, sol = g.vegsoil(variant)
    nx, ny, _ = g.GRIDS[variant]
    mod = import_module(optin.load("wrf_hydro").__name__ + ".coupled")
    return g, mod.WrfHydroCoupled(d / "hydro.namelist", ix=nx, jx=ny,
                                  zs=soil_centres(g.DZS), vegtyp=veg, soltyp=sol,
                                  hydro_dt=g.HYDRO_DT, start=datetime.datetime(2020, 1, 1))


def _static(variant):
    out = defaultdict(dict)
    with (FIX / VARIANTS[variant][2]).open(newline="") as fh:
        for r in csv.DictReader(fh):
            out[r["field"]][int(r["index"])] = int(r["value"])
    return {k: np.array([v[i] for i in range(len(v))], np.int64) for k, v in out.items()}


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_hydro_ini_is_bitwise_against_wrf_hydro(tmp_path_factory, variant):
    _, c = _coupled(tmp_path_factory, variant)
    want = _static(variant)
    net = c.dom.network
    for name in ("chanxi", "chanyj", "to_node", "typel", "lakenode", "order"):
        assert np.array_equal(net[name], want[name]), name
    for name in ("chanlen", "node_area", "bw", "zelev"):
        assert np.array_equal(np.asarray(net[name], np.float32).view(np.int32),
                              want[name].astype(np.int32)), name
    bits = lambda a: np.ascontiguousarray(a, np.float32).ravel().view(np.int32)  # noqa: E731
    assert np.array_equal(bits(c.dom.retention), want["retention"].astype(np.int32))
    assert np.array_equal(bits(c.dom.slope), want["slope"].astype(np.int32))
    assert np.array_equal(c.dom.sdir.ravel(), want["sdir"])
    assert np.array_equal(c.lake_mask.get().ravel(), want["lake_mask"])


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_coupling_cycles_are_bitwise_against_wrf_hydro(tmp_path_factory, variant):
    import cupy as cp

    g, c = _coupled(tmp_path_factory, variant)
    assert (c.reservoir_da is not None) == (variant == "da")
    assert (c.gw2d is not None) == (variant == "gw2d")
    want = defaultdict(dict)
    with (FIX / VARIANTS[variant][0]).open(newline="") as fh:
        for r in csv.DictReader(fh):
            want[(int(r["cycle"]), r["field"])][int(r["index"])] = int(r["bits"])
    nx, ny, _ = g.GRIDS[variant]
    smois = cp.zeros((g.NSOIL, ny, nx), cp.float32)
    sh2o = cp.zeros_like(smois)
    head = cp.zeros((ny, nx), cp.float32)
    for n, (sm, sh, infx, sold) in enumerate(g.cycles(variant), start=1):
        smois[...] = cp.asarray(sm)
        sh2o[...] = cp.asarray(sh)
        c.exchange(smois=smois, sh2o=sh2o, infxsrt=cp.asarray(infx), soldrain=cp.asarray(sold),
                   sfcheadrt=head)
        got = dict(smois=smois, sh2o=sh2o, sfcheadrt=head, infxsrt_rt=c.infxsrt,
                   head_rt=c.ostate.head, zwattablrt=c.sstate.zwattablrt,
                   qstrmvolrt_acc=c.qstrm_acc, qinflowbase=c.qinflowbase,
                   to_lake=c.ostate.to_lake, qlink1=c.qlink1, hlink=c.hlink, cvol=c.cvol,
                   resht=c.channel.lakes.resht, qlakeo=c.channel.lakes.qlakeo,
                   dtct=c.channel.dtct)
        if c.bucket is not None:
            got.update(z_gwsubbas=c.z_gw, qout_gwsubbas=c.qout_gw)
        if c.gw2d is not None:
            got.update(gw_head=c.gw2d.ho, gw_excess=c.gw2d.excess, qsgwrt=c.gw2d.qsgwrt,
                       qgw_chanrt=c.gw2d.qgw_chanrt, smcrt=c.gt_out["smcrt"])
        assert {k for (m, k) in want if m == n} == set(got), n
        for name, arr in got.items():
            w = want[(n, name)]
            expect = np.array([w[i] for i in range(len(w))], np.int64).astype(np.int32)
            have = np.ascontiguousarray(arr.get(), np.float32).ravel().view(np.int32)
            assert np.array_equal(have[:len(expect)], expect), (n, name)
