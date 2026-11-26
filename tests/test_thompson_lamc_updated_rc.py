"""Thompson's cloud slope from the updated cloud mass (quirk pass, correct-mode FQ7).

THE QUIRK
---------
WRF 4.7.1's Thompson (module_mp_thompson.F:2996-3019) balances the cloud
water mass and number after every source has been accumulated: it forms the
updated mass xrc = (qc1d + qcten*dt)*rho and number xnc, then diagnoses the
cloud slope from them.  :3003 divides by the ENTRY mass rc(k), not xrc
(a copy of :1833, where rc is the current mass).  The clamp decision on the
mean size, and so the rediagnosed number, then answers a mass the cloud no
longer holds; the clamp branches (:3006, :3010) already use xrc.  The
sources only remove cloud water, so xrc <= rc and the stale mass gives a mean
size that is too large, never too small.  The HRRR fork's own file
(module_mp_thompson.F, v4.1.21 WRFV3.9, sha 4d600111...) holds the same line
at :2839 (xrc at :2830).  RunConfig.thompson_lamc_updated_rc forms lamc from xrc.

WHAT THIS FILE HOLDS
--------------------
* The key is correct-mode finding FQ7: on by default, off in exact mode,
  read only under mp_physics = 28 or 38 (the aerosol-aware schemes whose
  cloud number leaves the call; the classic and generated columns of mp = 8
  hold a fixed droplet number that no output carries).
* Exact mode is WRF's program word for word: the two units that carry the
  balance hold the correction only inside ``#ifdef THOMPSON_LAMC_UPDATED_RC``
  blocks, and each unit with those blocks removed is byte-identical to the
  source frozen before the correction.
* Every launcher compiles the define exactly when the key reaches the run,
  and it composes with the FQ5 define.
* On the device, the corrected balance answers the updated mass.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

from woof import correct_mode
from woof.config import RunConfig, validate_run_config

sys.path.insert(0, str(Path(__file__).parent))
from test_thompson_rain_snow_closure import _split_blocks  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
KERNELS = ROOT / "woof" / "core" / "kernels"
KEY = "thompson_lamc_updated_rc"
DEFINE = "THOMPSON_LAMC_UPDATED_RC"

#: Each unit: (blocks, sha256 of the unit with them removed = origin/main
#: 2618b3ade's source, before the correction).
UNITS = {
    "thompson_aerosol_warm.cu": (1, "900e26aa8314c8687173078353ef413052c2f888c2638234600bb9c10e70e765"),
    "thompson_fork.cu": (1, "a916ede73fd7cf39ff585b6889e7e90f04338f6c2b097b638d63a09a11cd9ef4"),
}


def _run(**kw):
    return validate_run_config(RunConfig(
        nx=8, ny=8, nz=12, dx=3000.0, dy=3000.0, ztop=12000.0, dt=10.0,
        run_seconds=60.0, moist=True, **kw))


def test_the_key_is_correct_mode_finding_fq7():
    assert correct_mode.FINDING_BY_KEY[KEY] == "FQ7"
    assert correct_mode.KEY_SCOPE[KEY] == ("mp_physics", (28, 38))
    assert RunConfig.__dataclass_fields__[KEY].default is correct_mode.CORRECT_KEYS_DEFAULT
    assert correct_mode.resolve("exact").settings[KEY] is False
    assert correct_mode.resolve("correct").settings[KEY] is True


@pytest.mark.parametrize("mp", [28, 38])
def test_the_launchers_read_it_where_the_aerosol_aware_thompson_runs(mp):
    from woof.core.thompson import lamc_updated_rc_due
    assert lamc_updated_rc_due(_run(mp_physics=mp)) is correct_mode.CORRECT_KEYS_DEFAULT
    assert lamc_updated_rc_due(_run(mp_physics=mp, **{KEY: True})) is True
    assert lamc_updated_rc_due(_run(mp_physics=mp, **{KEY: False})) is False
    exact = _run(mp_physics=mp, **correct_mode.resolve("exact").settings)
    assert lamc_updated_rc_due(exact) is False


@pytest.mark.parametrize("mp", [6, 8, 10, 29])
def test_it_is_inert_where_the_cloud_number_does_not_leave_the_call(mp):
    """Classic Thompson (8) holds a fixed droplet number; RCON, WSM6 and
    Morrison never read the key."""
    from woof.core.thompson import lamc_updated_rc_due
    assert lamc_updated_rc_due(_run(mp_physics=mp, **{KEY: True})) is False


@pytest.mark.parametrize("unit", sorted(UNITS))
def test_without_the_define_every_unit_is_the_program_before_the_correction(unit):
    text = (KERNELS / unit).read_text(encoding="utf-8")
    blocks, digest = UNITS[unit]
    found, stripped = _split_blocks(text, DEFINE)
    assert len(found) == blocks, unit
    for body in found:
        depth = 0
        for line in body.splitlines():
            word = line.strip()
            if word.startswith("#if"):
                depth += 1
            elif word.startswith("#endif"):
                depth -= 1
            elif word.startswith(("#else", "#elif")):
                assert depth > 0, unit
        assert "xrc" in body and "lamc" in body, unit
    assert hashlib.sha256(stripped.encode("utf-8")).hexdigest() == digest, unit


def test_the_default_compile_never_sets_the_define():
    from woof.core import kernels
    assert DEFINE not in repr(getattr(kernels, "CUDA_DEFINES", ""))
    from woof.core.thompson import LAMC_UPDATED_RC_DEFINES
    assert LAMC_UPDATED_RC_DEFINES == ((DEFINE, 1),)


@pytest.mark.parametrize("closure", [False, True])
@pytest.mark.parametrize("lamc", [False, True])
@pytest.mark.parametrize("hail", [False, True])
def test_the_aerosol_launch_composes_the_defines(monkeypatch, closure, lamc, hail):
    """The warm unit takes FQ5's and FQ7's defines together, in the order the
    prewarm names; the cold unit takes FQ5's alone."""
    from woof.core import thompson_aerosol_launch as launch
    from woof.core.thompson import LAMC_UPDATED_RC_DEFINES, RAIN_SNOW_CLOSURE_DEFINES
    seen = []
    monkeypatch.setattr(launch, "get_kernel_int_defines",
                        lambda module, func, defines: seen.append((module, defines)))
    monkeypatch.setattr(launch, "get_kernel",
                        lambda module, func: seen.append((module, ())))
    hail_calls = []

    class _Module:
        def get_function(self, func):
            return None

    monkeypatch.setattr(launch, "_hail_module",
                        lambda name, rain_snow_closure=False, lamc_updated_rc=False:
                        hail_calls.append((name, rain_snow_closure, lamc_updated_rc)) or _Module())
    with launch.hail_aware_scope(hail), launch.rain_snow_closure_scope(closure), \
            launch.lamc_updated_rc_scope(lamc):
        launch.aerosol_kernel("thompson_aerosol_warm", "f")
        launch.aerosol_kernel("thompson_aerosol_cold", "f")
    if hail:
        # one canonical cache key per variant: three positional arguments
        assert hail_calls == [("thompson_aerosol_warm", closure, lamc),
                              ("thompson_aerosol_cold", closure, False)]
        return
    warm = (RAIN_SNOW_CLOSURE_DEFINES if closure else ()) \
        + (LAMC_UPDATED_RC_DEFINES if lamc else ())
    cold = RAIN_SNOW_CLOSURE_DEFINES if closure else ()
    # with no define at all the stock kernel loads (get_kernel)
    assert seen == [("thompson_aerosol_warm", warm), ("thompson_aerosol_cold", cold)]


# --- the device: the corrected balance answers the updated mass ------------

_AM_R = np.pi * 1000.0 / 6.0
_BM_R = 3.0
_D0C = 1.0e-6
_D0R = 50.0e-6


def _gamma_moments(nu):
    from scipy.special import gamma
    return gamma(nu + 1.0), gamma(_BM_R + nu + 1.0)


def _balance_reference(qc_entry, qc_after, nc_entry, rho, dt, updated):
    """WRF :2996-3019 in float64 with ncten = 0 on entry; returns (ncten,
    the mean size the balance's own lamc gives the final number and the
    updated mass)."""
    rc = max(1.0e-12, qc_entry * rho)
    xrc = max(1.0e-12, qc_after * rho)
    xnc = max(2.0, nc_entry * rho)
    nu = min(15, int(np.floor(1000.0e6 / xnc + 0.5)) + 2)
    ccg1, ccg2 = _gamma_moments(nu)
    lamc = (xnc * _AM_R * ccg2 / ccg1 / (xrc if updated else rc)) ** (1.0 / 3.0)
    xdc = (_BM_R + nu + 1.0) / lamc
    ncten = 0.0
    if xdc < _D0C or xdc > 2.0 * _D0R:
        lamc = (_BM_R + nu + 1.0) / (_D0C if xdc < _D0C else 2.0 * _D0R)
        xnc = ccg1 / ccg2 * xrc / _AM_R * lamc ** _BM_R
        ncten = (xnc - nc_entry * rho) / dt / rho
    final = (nc_entry + ncten * dt) * rho
    final_lamc = (final * _AM_R * ccg2 / ccg1 / xrc) ** (1.0 / 3.0)
    return ncten, (_BM_R + nu + 1.0) / final_lamc


def _device_balance(cp, updated, qc_entry, qc_after, nc_entry, rho, dt):
    from woof.core.thompson_aerosol_launch import (
        lamc_updated_rc_scope, rain_snow_closure_scope)
    from woof.core.thompson_aerosol_warm import launch_ncten_balance
    arrays = [cp.asarray(np.asarray(v, np.float32))
              for v in (qc_entry, qc_after, nc_entry, rho)]
    ncten = cp.zeros_like(arrays[0])
    with lamc_updated_rc_scope(updated), rain_snow_closure_scope(False):
        launch_ncten_balance(*arrays, ncten, dt)
    cp.cuda.Stream.null.synchronize()
    return cp.asnumpy(ncten).astype(np.float64)


@pytest.mark.gpu
def test_the_corrected_balance_answers_the_updated_cloud_mass():
    cp = pytest.importorskip("cupy")
    if cp.cuda.runtime.getDeviceCount() == 0:
        pytest.skip("device verification needs CUDA")
    dt = 10.0
    # (entry qc, updated qc, entry nc per kg).  qcten holds only sinks, so the
    # updated mass xrc never exceeds the entry mass rc and the stale rc gives
    # the mean size xDc (which goes as mass**(1/3)) too LARGE, never too small.
    #   0  shrinks 5x at 5e5 /kg: stale xDc 139 um is past 2*D0r = 100 um and
    #      clamps; the updated cloud is 81 um and does not (spurious clamp).
    #   1  shrinks 5x at 1e9 /kg, small mass: stale xDc 1.40 um clears D0c =
    #      1 um; the updated cloud is 0.82 um and must clamp (missed clamp).
    #   2  holds its mass: the correction is inert.
    #   3  shrinks 5x, no clamp either way: inert.
    qc_entry = np.asarray([5.0e-4, 5.0e-7, 1.0e-4, 5.0e-4], np.float64)
    qc_after = np.asarray([1.0e-4, 1.0e-7, 1.0e-4, 1.0e-4], np.float64)
    nc_entry = np.asarray([5.0e5, 1.0e9, 1.0e8, 1.0e8], np.float64)
    assert (qc_after <= qc_entry).all()
    rho = np.ones(4)
    exact = _device_balance(cp, False, qc_entry, qc_after, nc_entry, rho, dt)
    fixed = _device_balance(cp, True, qc_entry, qc_after, nc_entry, rho, dt)
    for i in range(4):
        for got, updated in ((exact, False), (fixed, True)):
            want, _ = _balance_reference(qc_entry[i], qc_after[i], nc_entry[i], 1.0, dt, updated)
            assert got[i] == pytest.approx(want, rel=1e-4, abs=1e-3), (i, updated)
    # the word differs on both flipped clamps (one number: exact vs corrected)
    assert exact[0] != 0.0 and fixed[0] == 0.0
    assert exact[1] == 0.0 and fixed[1] != 0.0
    # and not where the clamp decision is the same
    assert exact[2] == fixed[2] and exact[3] == fixed[3]
    # the physical claim: the corrected balance leaves every updated cloud
    # inside [D0c, 2*D0r]; the stale one clamps the 81 um cloud out to 100 um
    # (column 0) and leaves the 0.82 um cloud below D0c (column 1)
    tol = 1.0 + 1.0e-9
    for i in range(4):
        size = _balance_reference(qc_entry[i], qc_after[i], nc_entry[i], 1.0, dt, True)[1]
        assert _D0C / tol <= size <= 2.0 * _D0R * tol, i
    stale0 = _balance_reference(qc_entry[0], qc_after[0], nc_entry[0], 1.0, dt, False)[1]
    stale1 = _balance_reference(qc_entry[1], qc_after[1], nc_entry[1], 1.0, dt, False)[1]
    assert stale0 == pytest.approx(2.0 * _D0R, rel=1e-6)
    assert stale1 < _D0C
