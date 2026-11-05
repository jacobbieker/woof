"""Morrison with CESM aerosol (mp_physics=40) restarts: its advected
droplet number crosses a restart, and its algorithm identity is its own."""
from __future__ import annotations

import re

import numpy as np
import pytest

from woof.io import restart
from test_restart import _cfg, _fill_setup, _shim_state

SPECIES = ("qv", "qc", "qr", "qi", "qs", "qg", "nc", "nr", "ni", "ns", "ng")


@pytest.mark.parametrize("mode", (0, 1))
def test_morrison_aero_state_round_trips(mode, tmp_path, monkeypatch):
    cfg = _cfg(moist=True, mp_physics=40, morr_rimed_ice=mode)
    source = _shim_state(cfg, monkeypatch)
    _fill_setup(source)
    rng = np.random.default_rng(40 + mode)
    for name in SPECIES:
        field = getattr(source, name)
        field[...] = rng.uniform(0.0, 1.0e-3, field.shape).astype(np.float32)
    # Droplet number is a prognostic of mp=40 (a transported scalar), not
    # mp=10's diagnosed constant: its words must cross unchanged.
    source.nc[...] = rng.uniform(1.0e6, 5.0e8, source.nc.shape).astype(
        np.float32)
    path = restart.write_restart(tmp_path / "aero.npz", source, cfg)
    fresh = _shim_state(cfg, monkeypatch)
    _fill_setup(fresh)
    restart.restore_restart(path, fresh, cfg)
    for name in SPECIES:
        np.testing.assert_array_equal(getattr(fresh, name),
                                      getattr(source, name))


def test_morrison_aero_identity_is_not_morrisons():
    ids = restart.MICROPHYSICS_ALGORITHM_IDENTITIES
    assert ids[40] != ids[10]
    assert "aero" in ids[40]


def test_prior_morrison_aero_algorithm_is_rejected(tmp_path, monkeypatch):
    cfg = _cfg(moist=True, mp_physics=40, morr_rimed_ice=1)
    source = _shim_state(cfg, monkeypatch)
    _fill_setup(source)
    previous = "morrison-tm-aero-wrf-v4.7.1-v0"
    with monkeypatch.context() as old:
        old.setitem(restart.MICROPHYSICS_ALGORITHM_IDENTITIES, 40, previous)
        path = restart.write_restart(tmp_path / "previous.npz", source, cfg)
    fresh = _shim_state(cfg, monkeypatch)
    _fill_setup(fresh)
    fresh.qv.fill(np.float32(.0123))
    before = fresh.qv.tobytes()
    moved = re.escape(
        f"mp_physics=40 (microphysics): {previous} -> "
        f"{restart.MICROPHYSICS_ALGORITHM_IDENTITIES[40]}")
    with pytest.raises(restart.RestartMismatchError, match=moved):
        restart.restore_restart(path, fresh, cfg)
    assert fresh.qv.tobytes() == before
