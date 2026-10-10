"""SWDDNI/SWDDIF/COSZEN on the device: the radiation plumbing half.

``tests/test_energy_output_preset.py`` pins the preset, the schema rows and
the CPU publication rule.  This file pins what only a device can show:

* RTE+RRTMGP hands back its surface direct beam and diffuse flux when the
  driver asks (``surface_direct_requested``), their sum is the surface
  downward shortwave it already published, night columns are zero, and
  the request moves no other output bit;
* the driver turns one radiation result into SWDDNI = SWDDIR / COSZEN on
  daylit columns, zero at night, and fails closed when a scheme that
  declared the outputs returns none;
* a spectrum composition carries the requested beam through without
  changing its BEP+BEM split.
"""
from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import requires_gpu

pytestmark = [pytest.mark.gpu, requires_gpu]


def _rrtmgp_call(cp, *, requested: bool):
    from woof.core import rrtmgp as rr

    nz, ny, nx = 12, 1, 13
    plev = np.broadcast_to(
        np.geomspace(100000.0, 5718.0, nz + 1)[:, None, None],
        (nz + 1, ny, nx)).copy()
    play = np.sqrt(plev[:-1] * plev[1:])
    temperature = np.broadcast_to(
        np.linspace(290.0, 215.0, nz)[:, None, None], play.shape).copy()
    exner = (play / 100000.0) ** (287.0 / 1004.0)
    qv = np.broadcast_to(np.geomspace(8e-3, 1e-5, nz)[:, None, None],
                         play.shape).copy()
    qc = np.zeros_like(play)
    qi = np.zeros_like(play)
    qc[3:7, :, 5:9] = 2e-4
    atmosphere = {name: cp.asarray(value, dtype=cp.float32)
                  for name, value in {
                      "pressure": play, "p_interface": plev,
                      "temperature": temperature, "theta": temperature / exner,
                      "exner": exner, "qv": qv, "qc": qc, "qi": qi}.items()}
    fields = {"tsk": cp.full((ny, nx), 288.0, cp.float32),
              "albedo": cp.asarray(np.linspace(0.1, 0.3, nx)[None],
                                    dtype=cp.float32),
              "emiss": cp.full((ny, nx), 0.96, cp.float32),
              "glw": cp.full((ny, nx), 300.0, cp.float32)}
    state = SimpleNamespace(elapsed_seconds=0.0, qc=atmosphere["qc"],
                            qr=cp.zeros_like(atmosphere["qc"]))
    cfg = SimpleNamespace(mp_physics=1, dt=60.0, radt=12.0, radt_minutes=12.0)
    radiation = rr.RRTMGPRadiation(
        datetime(2011, 4, 27, 18), cp.full((ny, nx), 35.0),
        cp.full((ny, nx), -97.5), column_chunk=4)
    mu = [0.7, -0.2, 0.0, 0.2, -0.5, 0.6, 0.1, -0.1, 0.8, -0.3, 0.4, -0.9,
          0.3]
    radiation._cosine_zenith = lambda *a, **kw: cp.asarray([mu], cp.float32)
    assert radiation.supplies_surface_direct is True
    if requested:
        radiation.surface_direct_requested = True
    return radiation(atmosphere=atmosphere, fields=fields, state=state,
                     cfg=cfg), np.asarray(mu, np.float32)


def test_rrtmgp_returns_the_direct_diffuse_pair_only_on_request():
    cp = pytest.importorskip("cupy")
    plain, _ = _rrtmgp_call(cp, requested=False)
    asked, mu = _rrtmgp_call(cp, requested=True)
    assert plain.swddir is None and plain.swddif is None
    assert asked.swddir is not None and asked.swddif is not None
    direct = cp.asnumpy(asked.swddir)[0]
    diffuse = cp.asnumpy(asked.swddif)[0]
    swdown = cp.asnumpy(asked.swdown)[0]
    night = mu <= 0.0
    assert np.all(direct[night] == 0.0) and np.all(diffuse[night] == 0.0)
    assert np.all(direct >= 0.0) and np.all(diffuse >= 0.0)
    assert np.all(direct[~night] > 0.0) and np.all(diffuse[~night] > 0.0)
    np.testing.assert_allclose(direct + diffuse, swdown, rtol=2e-6, atol=1e-3)
    # The request reads a plane the solve already had: nothing else moves.
    for name in ("swdown", "gsw", "rthratensw", "rthratenlw", "glw",
                 "coszen"):
        np.testing.assert_array_equal(
            cp.asnumpy(getattr(asked, name)), cp.asnumpy(getattr(plain, name)),
            err_msg=name)


def test_legacy_rrtmg_declares_the_pair_only_with_a_shortwave():
    from woof.core.rrtmg_legacy import RRTMGLegacyRadiation
    from woof.core.dudhia import DudhiaShortwaveRadiation
    from woof.core.rrtm_lw import RRTMLongwaveRadiation

    assert RRTMGLegacyRadiation.supplies_surface_direct is True
    assert RRTMGLegacyRadiation.publishes_coszen is True
    assert not getattr(DudhiaShortwaveRadiation, "supplies_surface_direct",
                       False)
    assert DudhiaShortwaveRadiation.publishes_coszen is True
    # The classic longwave returns a placeholder COSZEN of zeros; it must
    # never be published as one.
    assert not getattr(RRTMLongwaveRadiation, "publishes_coszen", False)


def _bare_driver(cp, *, direct: bool, coszen: bool):
    from woof.core.physics import PhysicsDriver

    driver = object.__new__(PhysicsDriver)
    shape = (2, 3)
    driver.surface_dni = cp.zeros(shape, cp.float32) if direct else None
    driver.surface_dif = cp.zeros(shape, cp.float32) if direct else None
    driver.radiation_coszen = cp.zeros(shape, cp.float32) if coszen else None
    return driver


def _result(cp, **overrides):
    from woof.core.physics import RadiationResult

    shape = (2, 3)
    base = dict(
        rthratenlw=cp.zeros((1,) + shape, cp.float32),
        rthratensw=cp.zeros((1,) + shape, cp.float32),
        swdown=cp.zeros(shape, cp.float32), glw=cp.zeros(shape, cp.float32),
        coszen=cp.asarray([[0.5, 0.25, -0.1], [1.0, 0.0, 0.8]], cp.float32),
        swddir=cp.asarray([[400.0, 100.0, 0.0], [900.0, 0.0, 640.0]],
                          cp.float32),
        swddif=cp.asarray([[80.0, 60.0, 0.0], [50.0, 0.0, 70.0]],
                          cp.float32))
    base.update(overrides)
    return RadiationResult(**base)


def test_driver_turns_the_beam_into_direct_normal_irradiance():
    cp = pytest.importorskip("cupy")
    driver = _bare_driver(cp, direct=True, coszen=True)
    result = _result(cp)
    driver._capture_surface_solar(result, 2, 3)
    dni = cp.asnumpy(driver.surface_dni)
    np.testing.assert_allclose(
        dni, [[800.0, 400.0, 0.0], [900.0, 0.0, 800.0]], rtol=1e-6)
    np.testing.assert_array_equal(cp.asnumpy(driver.surface_dif),
                                  cp.asnumpy(result.swddif))
    np.testing.assert_array_equal(cp.asnumpy(driver.radiation_coszen),
                                  cp.asnumpy(result.coszen))
    # DNI . cosZ + DIF closes on the global flux where the sun is up.
    cosine = cp.asnumpy(result.coszen)
    day = cosine > 0
    np.testing.assert_allclose(
        (dni * cosine + cp.asnumpy(driver.surface_dif))[day],
        (cp.asnumpy(result.swddir) + cp.asnumpy(result.swddif))[day],
        rtol=1e-6)


def test_driver_without_a_producer_captures_nothing():
    cp = pytest.importorskip("cupy")
    driver = _bare_driver(cp, direct=False, coszen=False)
    # No buffers: a result with no beam and no COSZEN is fine.
    driver._capture_surface_solar(
        _result(cp, swddir=None, swddif=None, coszen=None), 2, 3)
    assert driver.surface_dni is None and driver.radiation_coszen is None


@pytest.mark.parametrize("missing", ["swddir", "swddif", "coszen"])
def test_driver_fails_closed_when_a_declared_output_is_missing(missing):
    cp = pytest.importorskip("cupy")
    driver = _bare_driver(cp, direct=True, coszen=False)
    with pytest.raises(ValueError, match=missing.upper()):
        driver._capture_surface_solar(_result(cp, **{missing: None}), 2, 3)


def test_driver_fails_closed_when_declared_coszen_is_missing():
    cp = pytest.importorskip("cupy")
    driver = _bare_driver(cp, direct=False, coszen=True)
    with pytest.raises(ValueError, match="publishes_coszen"):
        driver._capture_surface_solar(_result(cp, coszen=None), 2, 3)


def test_composition_carries_the_requested_beam():
    cp = pytest.importorskip("cupy")
    from woof.core.radiation_composition import ComposedRadiation

    beam = _result(cp)

    class Shortwave:
        supplies_surface_direct = True

        def __call__(self, **_):
            return beam

    class Longwave:
        publishes_olr = False

        def __call__(self, **_):
            return _result(cp, swddir=None, swddif=None)

    composed = ComposedRadiation(
        datetime(2011, 4, 27, 18), cp.zeros((2, 3)), cp.zeros((2, 3)),
        longwave_adapter=Longwave(), shortwave_adapter=Shortwave())
    assert composed.supplies_surface_direct is True
    assert composed.publishes_coszen is True
    arguments = dict(atmosphere={}, fields={"glw": cp.zeros((2, 3))},
                     state=SimpleNamespace(elapsed_seconds=0.0),
                     cfg=SimpleNamespace(swint_opt=0))
    assert composed(**arguments).swddir is None
    composed.surface_direct_requested = True
    assert composed(**arguments).swddir is beam.swddir
