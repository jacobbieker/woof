"""solar_ssi_opt on hex's generic column seam (legacy RRTMG and RRTMG-K).

The seam runs WRF's own ``PhysicsDriver`` on the cell columns, so the day's
band factor (woof.spacewx.solar_spectrum) reaches the seam's shortwave the
way it reaches a WRF domain's.  At every radiation call the seam's result is
bit for bit the one a WRF-route adapter computes on the same atmosphere when
it carries the factor the same series gives at that time; the factor is not
all ones, and it changes the shortwave heating.
"""
from __future__ import annotations

from datetime import timedelta

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

import mpas_column_physics_cases as cases  # noqa: E402
from test_hex_rrtmgk_gpu import NCOL, START, _build, _cap, _run, _spy  # noqa: E402
from test_spacewx_solar_ssi import REF, _ssi_file  # noqa: E402
from woof.spacewx import solar_spectrum as ss  # noqa: E402

pytestmark = pytest.mark.gpu

#: The series' 200-263 nm band (RRTMG band 28) 30 % above the cycle mean on
#: every day, so the factor is the same at every call.
UV = 1.3


def _fresh(label, cfg):
    lat, lon = _cap(NCOL)
    args = (START, lat.reshape(1, -1), lon.reshape(1, -1))
    if label == "k":
        from woof.core.rrtmgk import RRTMGKRadiation
        return RRTMGKRadiation(*args, o3input=cfg.o3input, icloud=cfg.icloud,
                               column_chunk=None, p_top=cases.P_TOP)
    from woof.core.rrtmg_legacy import RRTMGLegacyRadiation
    return RRTMGLegacyRadiation(*args, p_top=cases.P_TOP, o3input=cfg.o3input)


def _physics(label):
    name = "radiation.wrf-rrtmg-k" if label == "k" else "radiation.rrtmg-legacy"
    return dict(cases.selections()[name])


@pytest.mark.parametrize("label", ["legacy", "k"])
def test_the_seam_takes_the_band_factor_bit_for_bit(monkeypatch, tmp_path, label):
    path = _ssi_file(tmp_path / "ssi.nc", first=(START - timedelta(days=1)).replace(hour=0),
                     uv=(UV, UV, UV))
    want = tuple(float(v) for v in ss.band_factor(np.where(np.arange(14) == 12, REF * UV, REF)))
    assert want[12] > 1.29 and max(abs(v - 1.0) for i, v in enumerate(want) if i != 12) < 0.01

    seam = _build(dict(_physics(label), solar_ssi_opt=1, solar_ssi_source=str(path)))
    driver, cfg = seam._driver, seam._cfg
    assert (cfg.solar_ssi_opt, cfg.solar_ssi_source) == (1, str(path))
    assert driver.solar_spectrum is not None
    target = driver.radiation_callable
    assert ss.SpectralVariability.shortwave_targets(target) == (target,)
    fresh = _fresh(label, cfg)
    assert type(fresh) is type(target)
    # The WRF route's driver with the same solar_ssi_opt: its own
    # SpectralVariability from the same configuration and series (every day
    # alike, so one factor for every call).
    route = ss.SpectralVariability.for_run(cfg, START, fresh)
    factor = tuple(float(v) for v in route.factor(0.0))
    np.testing.assert_allclose(factor, want, rtol=1e-6)     # the file holds float32 SSI

    def reference(**kwargs):
        assert target.sw_band_factor == factor
        route.apply(fresh, 0.0)
        assert fresh.sw_band_factor == factor
        return fresh(**kwargs)

    calls = _spy(monkeypatch, seam, also=reference)
    _run(seam, steps=2)
    assert calls, "the seam never radiated"
    for step, (got, ref) in enumerate(calls):
        assert set(got) == set(ref)
        for name in got:
            np.testing.assert_array_equal(got[name], ref[name], f"{label} call {step} {name}")

    base = _build(_physics(label))
    assert base._driver.solar_spectrum is None
    with monkeypatch.context() as patch:
        base_calls = _spy(patch, base)
        _run(base, steps=1)
    assert base_calls
    off, on = base_calls[0][0], calls[0][0]
    assert getattr(base._driver.radiation_callable, "sw_band_factor", None) is None
    np.testing.assert_array_equal(off["rthratenlw"], on["rthratenlw"])
    assert not np.array_equal(off["rthratensw"], on["rthratensw"])
