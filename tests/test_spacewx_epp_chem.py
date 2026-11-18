"""The solar-proton NOx/HOx source in RADM2 chemistry (spacewx_epp_opt).

The device rate is checked against the host's layer means
(woof.spacewx.particles), the hold against a fake feed, the admission against
RunConfig, and the source against a stand-in chem context: the polar caps
gain NO, OH and HO2 in WRF's ppmv, the rest of the domain nothing, and an
hour below the S1 threshold adds nothing anywhere.
"""
from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pytest

from woof.config import ConfigurationRefused, RunConfig, validate_run_config
from woof.spacewx import epp_chem as E
from woof.spacewx import particles as P

from conftest import requires_gpu  # noqa: E402
from test_spacewx_ionosphere import _dipole  # noqa: E402

START = datetime(2024, 5, 11, 6, 30)
JAN2012 = [np.nan, np.nan, np.nan, 2693.35, 265.57, 8.39]


def _cfg(**overrides):
    values = dict(nx=6, ny=4, nz=5, dx=3000.0, dy=3000.0, ztop=20000.0, dt=20.0, run_seconds=3600.0)
    values.update(overrides)
    return RunConfig(**values)


def _record(flux):
    hours = np.array([np.datetime64("2024-05-11T04", "h"), np.datetime64("2024-05-11T05", "h")])
    return P.ProtonRecord(hours, P.CHANNELS_MEV, np.array([flux, flux], dtype=np.float64), "fake", "0" * 64)


# --- admission ----------------------------------------------------------------------

def test_defaults_are_off_and_admitted():
    cfg = _cfg()
    assert (cfg.spacewx_epp_opt, cfg.spacewx_epp_source) == (0, "")
    validate_run_config(cfg)
    assert E.device_shapes(cfg) == {} and E.transient_shapes(cfg) == {}


@pytest.mark.parametrize("overrides, match", [
    (dict(spacewx_epp_opt=2), "spacewx_epp_opt"),
    (dict(spacewx_epp_opt=True), "spacewx_epp_opt"),
    (dict(spacewx_epp_source="omni"), "would do nothing"),
    (dict(spacewx_epp_opt=1), "needs spacewx_epp_source"),
    (dict(spacewx_epp_opt=1, spacewx_epp_source="/nonexistent/omni2_2012.dat"), "neither"),
    (dict(spacewx_epp_opt=1, spacewx_epp_source="omni"), "RADM2"),
])
def test_refusals_name_the_breakage(overrides, match):
    with pytest.raises(ConfigurationRefused, match=match):
        validate_run_config(_cfg(**overrides))


def test_radm2_admits_the_source_and_it_reaches_the_identity():
    from woof.io import restart
    cfg = _cfg(chem_sets="radm2", bl_pbl_physics=1, sf_sfclay_physics=1, spacewx_epp_opt=1, spacewx_epp_source="swpc")
    validate_run_config(cfg)
    assert E.device_shapes(cfg) == {"spacewx_epp/cap": (4, 6)}
    assert E.transient_shapes(cfg)["spacewx_epp/p8w"] == (6, 4, 6)
    on = restart.configuration_echo(cfg)
    assert on["spacewx_epp_opt"] == 1 and on["spacewx_epp_source"] == "swpc"
    off = restart.configuration_echo(_cfg(chem_sets="radm2", bl_pbl_physics=1, sf_sfclay_physics=1))
    assert "spacewx_epp_opt" not in off and "spacewx_epp_source" not in off


def test_hex_refuses_it():
    from woof.hex.physics_suites import PhysicsSelectionRefusal, resolve_physics
    with pytest.raises(PhysicsSelectionRefusal, match="spacewx_epp_opt"):
        resolve_physics("arwen_mpas_column", {"spacewx_epp_opt": 1}, mesh="x1.40962")


# --- the held flux ---------------------------------------------------------------------

def test_the_flux_of_the_last_ended_hour_is_held_and_memoised(tmp_path):
    seen = []

    def loader(kind, start, end):
        seen.append((kind, start, end))
        return _record(JAN2012)

    held = E.held_flux("swpc", START, loader=loader, memo=tmp_path)
    assert held["spe"] and held["record"]["spacewx_epp_hour"] == "2024-05-11T05:00Z"
    np.testing.assert_allclose(held["flux"], JAN2012)
    # A resume (or another domain) reads the memo, not the feed that has moved on.
    again = E.held_flux("swpc", START, loader=lambda *a: (_ for _ in ()).throw(OSError("gone")), memo=tmp_path)
    np.testing.assert_allclose(again["flux"], JAN2012)
    assert "spacewx_epp_held_from" in again["record"] and len(seen) == 1
    with pytest.raises(E.EppChemError, match="no proton flux"):
        E.held_flux("omni", datetime(2024, 5, 11, 9), loader=loader)


# --- the rate --------------------------------------------------------------------------

def _p8w():
    # 31 w levels from 1000 hPa to 0.05 hPa, two columns (one per ny x nx cell).
    p = np.geomspace(1.0e5, 5.0, 31)
    return np.repeat(p[:, None, None], 2, axis=2).reshape(31, 1, 2)


def test_the_tabulated_rate_matches_the_layer_means():
    p = _p8w()
    table = E.log_rate_table(JAN2012, P.CHANNELS_MEV)
    q = E.ion_pairs(p, table)
    f = P.absorbed(JAN2012, P.DEPTH_GRID)
    exact = P.ion_pairs_per_gram(f, p[:, 0, 0])
    # Within 3 % wherever the rate matters; the lowest layers get about 1e-3
    # ion pairs g-1 s-1 against 1e10 aloft.
    np.testing.assert_allclose(q[:, 0, 0], exact, rtol=0.03, atol=1.0e-9 * exact.max())
    assert q.dtype == np.float32 and np.all(q[:, 0, 0] == q[:, 0, 1])


@pytest.mark.gpu
@requires_gpu
def test_the_device_rate_matches_the_host():
    import cupy as cp
    p = _p8w()
    table = E.log_rate_table(JAN2012, P.CHANNELS_MEV)
    host = E.ion_pairs(p.astype(np.float32), table)
    dev = cp.asnumpy(E.ion_pairs(cp.asarray(p, dtype=cp.float32), table, cp))
    np.testing.assert_allclose(dev, host, rtol=2e-4)


# --- the source in a chem step ---------------------------------------------------------

class _Table:
    def __init__(self, names):
        self.rows = [SimpleNamespace(name=n) for n in names]

    def row(self, name):
        return next(r for r in self.rows if r.name == name)


def _held(flux):
    rec = _record(flux)
    f = rec.at(datetime(2024, 5, 11, 5))
    return {"channels": P.CHANNELS_MEV, "flux": f, "spe": rec.active(f), "record": {"spacewx_epp_source": "fake"}}


def _cap(xlat):
    from woof.spacewx import aurora
    return P.polar_cap(aurora.dipole_latitude(aurora.dipole_frame(_dipole(), START), xlat, np.zeros_like(xlat)))


def _ctx(flux, xlat, monkeypatch):
    import cupy as cp
    from woof.spacewx import igrf
    monkeypatch.setattr(igrf, "load", lambda *a, **k: _dipole())
    monkeypatch.setattr(E, "held_flux", lambda source, start, **k: _held(flux))
    p8w = cp.asarray(np.repeat(np.geomspace(1.0e5, 5.0, 31)[:, None], xlat.size, axis=1).reshape(31, 1, -1),
                     dtype=cp.float32)
    fields = {n: cp.full((30, 1, xlat.size), 1.0e-3, dtype=cp.float32) for n in ("no", "ho", "ho2")}
    state = SimpleNamespace(chem=SimpleNamespace(start_time=START), **{f"chem_{n}": a for n, a in fields.items()})
    met = {"p8w": p8w, "xlat": cp.asarray(xlat.reshape(1, -1), dtype=cp.float32),
           "xlong": cp.zeros((1, xlat.size), dtype=cp.float32)}
    ctx = SimpleNamespace(cfg=SimpleNamespace(spacewx_epp_opt=1, spacewx_epp_source="swpc", start_time=START),
                          state=state, table=_Table(("no", "ho", "ho2")), met=lambda name: met[name],
                          field=lambda row: fields[row.name])
    return ctx, fields, p8w


@pytest.mark.gpu
@requires_gpu
def test_the_caps_gain_no_oh_and_ho2_and_the_rest_nothing(monkeypatch, capsys):
    import cupy as cp
    from woof.core import chem_radm2_emiss as R
    xlat = np.array([85.0, 75.0, 40.0, 0.0, -80.0])
    ctx, fields, p8w = _ctx(JAN2012, xlat, monkeypatch)
    before = {n: cp.asnumpy(a).copy() for n, a in fields.items()}
    dt = 600.0
    R._solar_protons(ctx, dt)
    assert "[spacewx epp]" in capsys.readouterr().out
    cap = _cap(xlat)
    assert cap.tolist() == [1, 1, 0, 0, 1]
    q = E.ion_pairs(cp.asnumpy(p8w), E.log_rate_table(JAN2012, P.CHANNELS_MEV))[:, 0, 0]
    per_pair = P.vmr_tendency(1.0, 1.0) * 1.0e6 * dt
    for name, y in E.YIELDS:
        gain = cp.asnumpy(fields[name]) - before[name]
        for c in range(xlat.size):
            if cap[c]:
                np.testing.assert_allclose(gain[:, 0, c], q * per_pair * y, rtol=1e-3, atol=1e-12)
            else:
                assert np.all(gain[:, 0, c] == 0.0)
    # 1 hPa in this soft event: about 1 ppbv NO per day (docs/spacewx.md).
    edges = np.geomspace(1.0e5, 5.0, 31)
    k = int(np.argmin(np.abs(np.sqrt(edges[1:] * edges[:-1]) - 100.0)))
    no = cp.asnumpy(fields["no"])[k, 0, 0] - before["no"][k, 0, 0]
    assert 0.3 < no * 1.0e3 * 86400.0 / dt < 3.0


@pytest.mark.gpu
@requires_gpu
def test_an_hour_below_s1_adds_nothing(monkeypatch):
    import cupy as cp
    from woof.core import chem_radm2_emiss as R
    quiet = [np.nan, np.nan, np.nan, 2.0, 0.5, 0.2]
    ctx, fields, _p = _ctx(quiet, np.array([85.0, 0.0]), monkeypatch)
    before = {n: cp.asnumpy(a).copy() for n, a in fields.items()}
    R._solar_protons(ctx, 600.0)
    for name in fields:
        np.testing.assert_array_equal(cp.asnumpy(fields[name]), before[name])
    assert ctx.state.chem.epp_source.receipt()["spacewx_epp_spe"] is False
