"""WRF's side of its NEMO coupling, against WRF v4.6.1's frame/module_cpl.F.

``woof/ocean/wrf_cpl.cu`` runs ``cpl_snd2`` (the six fields WRF hands an
OASIS-coupled ocean every step) and ``cpl_rcv`` (how WRF merges the SST and
surface current the ocean hands back under ``CPLMASK``).
``tests/data/oracles/wrf_cpl/wrf-cpl.csv`` is the two routines' own text, cut
byte for byte from the pinned source and compiled by
``tools/wrf_cpl_wrf461_oracle/build.sh`` (see ``PROVENANCE.md`` there).

The coupling-interval average WOOF puts between them (``FluxAverager``) is not
WRF's or OASIS's code and is held to conservation instead: the integral the
ocean receives equals the one the atmosphere emitted.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu

ORACLE = Path(__file__).resolve().parent / "data" / "oracles" / "wrf_cpl"
FIXTURE = ORACLE / "wrf-cpl.csv"

INPUTS = ("qfx", "raincv", "rainncv", "gsw", "glw", "emiss", "sst", "lh",
          "hfx", "u_phy", "v_phy", "uoce", "voce", "rho", "ust")
SENT = ("evap_precip", "surf_net_solar", "surf_net_non_solar", "taux",
        "tauy", "taumod")


def _rows():
    with FIXTURE.open(newline="", encoding="ascii") as fh:
        return list(csv.DictReader(fh))


def _col(rows, key):
    return np.asarray([np.float32(r[key]) for r in rows], np.float32)


def _bits(a):
    return np.ascontiguousarray(a, np.float32).view(np.uint32)


# --------------------------------------------------------------------------
# CPU
# --------------------------------------------------------------------------

def test_fixture_is_the_one_build_sh_hashed():
    rows = (ORACLE / "oracle-sha256sums.txt").read_text().splitlines()
    pinned = {line.split()[1]: line.split()[0] for line in rows}
    assert pinned["wrf-cpl.csv"] == hashlib.sha256(FIXTURE.read_bytes()).hexdigest()


def test_fixture_reaches_the_edges_it_names():
    rows = _rows()
    taux, tauy = _col(rows, "taux"), _col(rows, "tauy")
    # no relative wind and a relative wind under the 1.e-7 floor
    assert taux[0] == 0.0 and tauy[0] == 0.0 and taux[1] == 0.0
    assert _col(rows, "taumod")[3] == 0.0
    assert _col(rows, "evap_precip")[5] < -1.0
    m1, m2 = _col(rows, "cplmask1"), _col(rows, "cplmask2")
    assert ((m1 == 0) & (m2 == 0)).any() and ((m1 > 0) & (m2 > 0)).any()
    assert np.all(m1 + m2 <= 1.0)
    # no coupling: SST is the input SST, the current is zero
    none = (m1 == 0) & (m2 == 0)
    assert np.array_equal(_col(rows, "sst_out")[none], _col(rows, "sst_input")[none])
    assert np.all(_col(rows, "uoce_out")[none] == 0.0)


def test_unported_exchange_fields_are_refused_by_name():
    from woof.ocean import wrf_cpl

    for name in wrf_cpl.NOT_IN_WRF_461:
        with pytest.raises(NotImplementedError, match=name):
            wrf_cpl.receive(name, None, None, None)
    with pytest.raises(NotImplementedError, match="remapping"):
        wrf_cpl.require_shared_grid((10, 12), (20, 24))
    with pytest.raises(NotImplementedError, match="remapping"):
        wrf_cpl.require_shared_grid((10, 12), (10, 12), remap={"type": "conserv"})
    wrf_cpl.require_shared_grid((10, 12), (10, 12))


def test_stbolt_is_wrfs():
    from woof.ocean.wrf_cpl import STBOLT

    assert np.float32(STBOLT) == np.float32(5.67051e-8)


def test_send_and_receive_fields_are_wrfs():
    from woof.ocean.wrf_cpl import RECEIVE_FIELDS, SEND_FIELDS

    assert SEND_FIELDS == ("EVAP-PRECIP", "SURF_NET_SOLAR",
                           "SURF_NET_NON-SOLAR", "TAUX", "TAUY", "TAUMOD")
    assert RECEIVE_FIELDS == ("SST", "UOCE", "VOCE")


# --------------------------------------------------------------------------
# GPU
# --------------------------------------------------------------------------

def _device_surface(rows):
    import cupy as cp

    from woof.ocean.wrf_cpl import AtmosphereSurface

    return AtmosphereSurface(**{name: cp.asarray(_col(rows, name).reshape(1, -1))
                                for name in INPUTS})


def _zeros_out(shape):
    import cupy as cp

    from woof.ocean.wrf_cpl import AtmosphereToOcean

    return AtmosphereToOcean(**{name: cp.zeros(shape, cp.float32) for name in SENT})


@pytest.mark.gpu
@requires_gpu
def test_send_is_bitwise_against_cpl_snd2():
    import cupy as cp

    from woof.ocean import wrf_cpl

    rows = _rows()
    out = _zeros_out((1, len(rows)))
    wrf_cpl.send(_device_surface(rows), out, dt=float(np.float32(rows[0]["dt"])))
    for name in SENT:
        got = cp.asnumpy(getattr(out, name)).reshape(-1)
        assert np.array_equal(_bits(got), _bits(_col(rows, name))), name


@pytest.mark.gpu
@requires_gpu
def test_receive_is_bitwise_against_cpl_rcv():
    import cupy as cp

    from woof.ocean import wrf_cpl

    rows = _rows()
    n = len(rows)
    mask = cp.asarray(np.stack([_col(rows, "cplmask1"),
                                _col(rows, "cplmask2")]).reshape(2, 1, n))
    wrf_cpl.validate_cplmask(mask)
    for field, key, obs in (("SST", "sst", "sst_input"),
                            ("UOCE", "uoce", None), ("VOCE", "voce", None)):
        received = cp.asarray(np.stack([_col(rows, f"rcv_{key}1"),
                                        _col(rows, f"rcv_{key}2")]).reshape(2, 1, n))
        out = cp.full((1, n), -999.0, cp.float32)
        o = cp.asarray(_col(rows, obs).reshape(1, n)) if obs else None
        wrf_cpl.receive(field, mask, received, out, obs=o)
        assert np.array_equal(_bits(cp.asnumpy(out).reshape(-1)),
                              _bits(_col(rows, f"{key}_out"))), field


@pytest.mark.gpu
@requires_gpu
def test_flux_average_conserves_the_emitted_integral():
    """Over 37 steps of an uneven dt and three coupling intervals, the
    integral delivered to the ocean (sum of mean*interval) matches the
    float64 integral of what was sent, per cell, to a few float32 ULP of the
    integral -- not to the per-step rounding a plain float32 sum would
    accumulate -- and a constant field comes back exactly."""
    import cupy as cp

    from woof.ocean import wrf_cpl

    rng = np.random.default_rng(461)
    shape = (8, 16)
    dts = [30.0, 45.0, 15.0, 60.0, 30.0, 30.0, 90.0, 60.0, 30.0, 10.0, 50.0]
    interval = sum(dts)
    avg = wrf_cpl.FluxAverager(shape, interval=interval)
    out = _zeros_out(shape)
    want = {name: np.zeros(shape) for name in SENT}
    got = {name: np.zeros(shape) for name in SENT}
    for _ in range(3):
        for dt in dts:
            planes = {name: rng.normal(0, 300, shape).astype(np.float32) for name in SENT}
            planes["taumod"] = np.full(shape, 0.125, np.float32)  # constant
            sent = wrf_cpl.AtmosphereToOcean(**{k: cp.asarray(v) for k, v in planes.items()})
            avg.accumulate(sent, dt)
            for name in SENT:
                want[name] += planes[name].astype(np.float64) * dt
        assert avg.due
        avg.deliver(out)
        for name in SENT:
            got[name] += cp.asnumpy(getattr(out, name)).astype(np.float64) * interval
        assert np.all(cp.asnumpy(out.taumod) == np.float32(0.125))
    for name in SENT:
        scale = np.abs(want[name]).max()
        assert np.abs(got[name] - want[name]).max() <= 8 * np.spacing(np.float32(scale)), name


@pytest.mark.gpu
@requires_gpu
def test_averager_refuses_a_step_across_the_coupling_time():
    from woof.ocean import wrf_cpl

    avg = wrf_cpl.FluxAverager((2, 2), interval=60.0)
    out = _zeros_out((2, 2))
    avg.accumulate(out, 45.0)
    with pytest.raises(ValueError, match="partition"):
        avg.accumulate(out, 30.0)
