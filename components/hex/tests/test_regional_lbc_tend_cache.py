"""PaddedRegionalDrivingState.tendency keeps its full-extent expansion per LBC interval.

The expansion (zero fill plus a scatter of the boundary support) is the same
for every stage of one interval; ``start`` and ``advance`` drop the held
expansions.  WOOF_HEX_LBC_TEND_CACHE_MB caps the bytes held (0, the default,
expands on every call).
"""

import numpy as np
import pytest


@pytest.fixture
def cp():
    return pytest.importorskip("cupy")


def _pool(cp, cap_bytes):
    from woof.hex.cuda_regional_forecast_v841 import PaddedRegionalDrivingState
    pool = PaddedRegionalDrivingState.__new__(PaddedRegionalDrivingState)
    pool.ncells, pool.nedges, pool.nlev = 6, 9, 2
    pool._scalar_names = ("qv",)
    pool._support = {"cell": cp.asarray([0, 2, 5]), "interface": cp.asarray([0, 2, 5]),
                     "edge": cp.asarray([1, 4, 8])}
    pool._tend_expanded, pool._tend_expanded_bytes = {}, 0
    pool._tend_cache_bytes = cap_bytes
    pool._tend_scalars = cp.zeros((1, 2, 3), cp.float32)
    pool._tend = _interval(cp, 1.0)
    return pool


def _interval(cp, scale):
    rng = np.random.default_rng(int(scale))
    return {name: cp.asarray((scale * rng.standard_normal((2, 3))).astype(np.float32))
            for name in ("rho_zz", "rtheta_m", "ru", "u", "w", "rho", "theta", "rho_edge")}


def _want(pool, name):
    extent = pool.nedges if name in ("ru", "u", "rho_edge") else pool.ncells
    kind = "edge" if extent == pool.nedges else "cell"
    full = np.zeros((2, extent), np.float32)
    full[:, pool._support[kind].get()] = pool._tend[name].get()
    return full.tobytes()


def test_expansion_is_held_for_the_interval_and_dropped_on_advance(cp, monkeypatch):
    from woof.hex.cuda_regional_v841 import DeviceRegionalDrivingState
    monkeypatch.setattr(DeviceRegionalDrivingState, "advance",
                        lambda self, when=None: setattr(self, "_tend", _interval(cp, 2.0)))
    pool = _pool(cp, 1 << 20)
    first = {name: pool.tendency(name) for name in ("rho_zz", "rtheta_m", "ru")}
    for name, array in first.items():
        assert array.get().tobytes() == _want(pool, name)
        assert pool.tendency(name) is array            # same interval: no new expansion
    pool.advance()                                      # installs the next interval's tendencies
    assert pool._tend_expanded == {} and pool._tend_expanded_bytes == 0
    for name, array in first.items():
        again = pool.tendency(name)
        assert again is not array and again.get().tobytes() == _want(pool, name)


def test_cap_bounds_what_is_held(cp):
    off = _pool(cp, 0)
    assert off.tendency("ru") is not off.tendency("ru")
    small = _pool(cp, 2 * 6 * 4)                        # room for one cell field, not an edge field
    assert small.tendency("rho_zz") is small.tendency("rho_zz")
    assert small.tendency("ru") is not small.tendency("ru")
    assert small.tendency("ru").get().tobytes() == _want(small, "ru")


def test_cap_switch(monkeypatch):
    from woof.hex.cuda_regional_forecast_v841 import _tend_cache_bytes
    monkeypatch.delenv("WOOF_HEX_LBC_TEND_CACHE_MB", raising=False)
    assert _tend_cache_bytes() == 0
    monkeypatch.setenv("WOOF_HEX_LBC_TEND_CACHE_MB", "64")
    assert _tend_cache_bytes() == 64 * 2**20
    for bad in ("lots", "inf", "nan", "-1"):
        monkeypatch.setenv("WOOF_HEX_LBC_TEND_CACHE_MB", bad)
        with pytest.raises(ValueError):
            _tend_cache_bytes()
