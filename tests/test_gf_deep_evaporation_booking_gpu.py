"""The opt-in GF deep evaporation booking (RunConfig.gf_deep_rain_evaporation_booking).

module_cu_gf_deep.F cup_output_ens_3d accumulates the downdraft evaporation that
detrained condensate cannot cover as ``-xmb*E_unc`` and then negates the
accumulator (:3348), so deep GF reports ``xmb*(sum pwo + E_unc)`` as surface rain
while its vapour tendency already holds ``E_unc``.  On the 216 GFDRV oracle columns
the stock deep ``pret`` is 1.29-1.66 times the column water the deep tendencies
remove.  The booking (gf.cu ``GF_DEEP_EVAP_BOOKING``) subtracts ``E_unc`` instead.

Graded here, on the same columns and launch as tests/test_gf_gfdrv_cuda.py:
every tendency and every shallow output is bitwise unchanged, the deep rain never
grows, and the deep column budget closes (rain = column vapour + condensate loss).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TOOLS = os.path.join(_ROOT, "tools", "gf_wrf461_oracle")
for _p in (_ROOT, _TOOLS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from woof.core.gf import gf_kernel_capacity                         # noqa: E402
from woof.core.kernels import load_module, load_module_int_defines  # noqa: E402
from woof.verify.gf_oracle import GF_NZ, load_gf_oracle              # noqa: E402
from gf_field_lists import DRV_ISCA_FIELDS, DRV_LEV_FIELDS, DRV_SCA_FIELDS  # noqa: E402

from test_gf_gfdrv_cuda import _launch                               # noqa: E402

G = 9.81


@pytest.fixture(scope="module")
def fixture():
    return load_gf_oracle()


@pytest.fixture(scope="module")
def runs(fixture):
    stock = _launch(load_module("gf"), fixture, 1)
    booked = _launch(load_module_int_defines(
        "gf", (("GF_KMAX", gf_kernel_capacity(GF_NZ)), ("GF_DEEP_EVAP_BOOKING", 1))), fixture, 1)
    return stock, booked


def _sca(run, name):
    return run["sca"][:, DRV_SCA_FIELDS.index(name)]


def _lev(run, name):
    return run["lev"][:, DRV_LEV_FIELDS.index(name), :]


def _deep_loss(run, fixture):
    """Column vapour + condensate the deep tendencies remove, kg m-2 s-1."""
    p8w = np.asarray(fixture.levels["p8w"], dtype=np.float64)
    dp = np.zeros_like(p8w)
    dp[:, :-1] = p8w[:, :-1] - p8w[:, 1:]
    rate = _lev(run, "outq").astype(np.float64) + _lev(run, "outqc").astype(np.float64)
    return -(rate * dp).sum(axis=1) / G


@pytest.mark.gpu
def test_tendencies_and_shallow_outputs_are_bitwise_unchanged(runs):
    stock, booked = runs
    for name in DRV_LEV_FIELDS:
        np.testing.assert_array_equal(_lev(booked, name).view(np.uint32),
                                      _lev(stock, name).view(np.uint32), err_msg=name)
    for name in ("prets", "xmb_shallow", "htop", "hbot", "cutens"):
        np.testing.assert_array_equal(_sca(booked, name).view(np.uint32),
                                      _sca(stock, name).view(np.uint32), err_msg=name)
    np.testing.assert_array_equal(booked["isc"], stock["isc"])


@pytest.mark.gpu
def test_deep_rain_never_grows_and_closes_the_column_budget(runs, fixture):
    stock, booked = runs
    p_stock, p_booked = _sca(stock, "pret"), _sca(booked, "pret")
    assert np.all(p_booked <= p_stock)
    deep = (stock["isc"][:, DRV_ISCA_FIELDS.index("ktop_deep")] > 0) & (p_stock > 0)
    assert deep.sum() >= 30
    loss = _deep_loss(stock, fixture)[deep]
    stock_ratio = p_stock[deep] / loss
    booked_ratio = p_booked[deep] / loss
    # WRF's arithmetic: rain 1.29-1.66x the water the column loses.
    assert stock_ratio.min() > 1.2 and stock_ratio.max() < 1.7
    # Booked: the residual is the scheme's cloud-top/base flux terms and the
    # p_cup-vs-p8w layer masses, about a per cent.
    assert np.abs(booked_ratio - 1.0).max() < 0.03, booked_ratio
