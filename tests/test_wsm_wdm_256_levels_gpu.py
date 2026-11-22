"""WSM3, WSM5, WSM7, WDM5 and WDM7 on 256-level columns (CUDA and Metal).

These five single-thread-per-column microphysics units size their per-thread
level arrays by a compile-time ``*_KMAX`` tier (WSM3_KMAX, WSM5_KMAX,
WSM7_KMAX, WDM5_KMAX, and WDM6_KMAX for the separate WDM7 unit).  Their
ladders ended at 80; they now carry 128 and 256 as well.  Three claims, per
scheme:

1. **A 256-level column runs.**  A 256-level synthetic sounding to a 1 hPa
   top, seeded with cloud, rain, ice and (where the scheme has them)
   graupel and hail, goes through the SHIPPED launcher.  Every output is
   finite, the mixing ratios stay non-negative, the column moves, and the
   column's water (vapour + condensate + the surface precipitation it
   booked) is conserved.  The kernels' own guards (``nz > *_KMAX`` returns
   without writing) would hand back the untouched inputs, which "the column
   moved" catches.

2. **Nothing moved at or below the old ceiling.**  The 64 and 80 tiers are
   the unchanged source (and are what the shipped launcher compiles there),
   so at nz = 49 and nz = 80 every output word of the shipped launcher
   equals the same source compiled at the 128 and 256 tiers, compared as raw
   bits: the tier is an allocation size and no arithmetic depends on it.

3. **The admitted bound and the priced frame follow.**  The contract bound,
   the registry row and preflight's per-thread local frame are the 256-level
   values (the frames are re-read off the driver in
   ``test_the_priced_frames_are_what_the_driver_reports`` where the backend
   exposes them).

Small column counts throughout: the card is shared.  The same file runs on
the Metal backend (``WOOF_BACKEND=mlx``).
"""

from __future__ import annotations

import dataclasses
import os
import sys

import numpy as np
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

pytestmark = pytest.mark.gpu

DEEP = 256
NCOL = 24
DT = 60.0
SCHEMES = ("wsm3", "wsm5", "wsm7", "wdm5", "wdm7")


# ---------------------------------------------------------------------------
# One 256-level atmosphere to a 1 hPa top, with weather in it
# ---------------------------------------------------------------------------

def _fields(nz, ncol=NCOL, *, seed=3):
    """float32 ``(nz, 1, ncol)`` state: thermodynamics plus every species."""
    from test_physics_256_levels_gpu import _atmosphere

    atm = _atmosphere(nz, ncol, seed=seed)
    z = atm["z"]
    t = atm["temperature"]
    rng = np.random.default_rng(seed + nz)

    def band(lo, hi, amount):
        inside = (z >= lo) & (z <= hi)
        jitter = 1.0 + 0.3 * rng.standard_normal(z.shape)
        return np.where(inside, amount * np.maximum(jitter, 0.2), 0.0)

    p = atm["pressure"]
    rho = p / (287.0 * t * (1.0 + 0.608 * atm["qv"]))
    # Moisten the lower troposphere past saturation so cloud forms and the
    # existing condensate has something to interact with.
    qv = np.where((z > 300.0) & (z < 5000.0), atm["qv"] * 1.18, atm["qv"])
    out = dict(
        theta=atm["theta"], pii=atm["exner"], rho=rho, pressure=p,
        dz=atm["dz"], qv=qv,
        qc=band(800.0, 4000.0, 3.0e-4) * (t > 258.0),
        qr=band(0.0, 3500.0, 4.0e-4),
        qi=band(5000.0, 11000.0, 1.0e-4) * (t < 268.0),
        qs=band(3000.0, 9000.0, 3.0e-4) * (t < 276.0),
        qg=band(3000.0, 6000.0, 2.0e-4),
        qh=band(4000.0, 6000.0, 1.0e-4),
        w=atm["w"],
        xland=np.broadcast_to(
            np.where(np.arange(ncol) % 2, 1.0, 2.0), (1, ncol)).copy(),
    )
    out["nn"] = np.full(z.shape, 2.0e8)
    out["nc"] = np.where(out["qc"] > 0, 1.0e8, 1.0)
    out["nr"] = np.where(out["qr"] > 0, 1.0e4, 1.0)
    return out


def _device(cp, fields, names):
    return {n: cp.ascontiguousarray(
        cp.asarray(np.asarray(fields[n], np.float32))) for n in names}


_VOLUME = ("theta", "pii", "rho", "pressure", "dz", "qv", "qc", "qr", "qi",
           "qs", "qg", "qh", "nn", "nc", "nr", "w")
_SPECIES = ("qv", "qc", "qr", "qi", "qs", "qg", "qh")


def _surface(cp, ncol):
    z = lambda: cp.zeros((1, ncol), cp.float32)  # noqa: E731
    return {n: z() for n in (
        "rainnc", "rainncv", "snownc", "snowncv", "graupelnc", "graupelncv",
        "hailnc", "hailncv", "sr")}


def _effs(cp, shape):
    return {n: cp.zeros(shape, cp.float32) for n in ("effc", "effi", "effs")}


# ---------------------------------------------------------------------------
# One call of each scheme through its shipped launcher
# ---------------------------------------------------------------------------

def _call_wsm3(cp, d, s, e):
    from woof.core import wsm3

    wsm3.launch_wsm3(d["theta"], d["qv"], d["qc"], d["qr"], d["w"], d["rho"],
                     d["pii"], d["pressure"], d["dz"], s["rainnc"],
                     s["rainncv"], s["snownc"], s["snowncv"], s["sr"], DT, **e)


def _call_wsm5(cp, d, s, e):
    from woof.core.column_schemes import COLUMN_SCHEMES
    from woof.core.microphysics import launch_column_scheme

    row = COLUMN_SCHEMES[4]
    launch_column_scheme(
        row, d["theta"], tuple(d[n] for n in row.species), d["rho"],
        d["pressure"], d["pii"], d["dz"], tuple(s[n] for n in (
            "rainnc", "rainncv", "snownc", "snowncv", "sr")),
        e["effc"], e["effi"], e["effs"], DT)


def _call_wsm7(cp, d, s, e):
    from woof.core import wsm_family

    wsm_family.launch_column(
        24, theta=d["theta"], qv=d["qv"],
        species={n: d[n] for n in ("qc", "qi", "qr", "qs", "qg", "qh")},
        rho=d["rho"], pii=d["pii"], pressure=d["pressure"], dz=d["dz"],
        surface=s, dt=DT, **e)


def _call_wdm5(cp, d, s, e):
    from woof.core import wdm6

    wdm6.launch_wdm5(
        d["theta"], d["qv"], d["qc"], d["qr"], d["qi"], d["qs"], d["nn"],
        d["nc"], d["nr"], d["rho"], d["pii"], d["pressure"], d["dz"],
        s["rainnc"], s["rainncv"], s["snownc"], s["snowncv"], s["sr"], DT,
        **e)


def _call_wdm7(cp, d, s, e):
    from woof.core import wdm6

    wdm6.launch_wdm7(
        d["theta"], d["qv"], d["qc"], d["qr"], d["qi"], d["qs"], d["qg"],
        d["qh"], d["nn"], d["nc"], d["nr"], d["rho"], d["pii"],
        d["pressure"], d["dz"], d["xland"], s["rainnc"], s["rainncv"],
        s["snownc"], s["snowncv"], s["graupelnc"], s["graupelncv"],
        s["hailnc"], s["hailncv"], s["sr"], DT, **e)


_CALL = dict(wsm3=_call_wsm3, wsm5=_call_wsm5, wsm7=_call_wsm7,
             wdm5=_call_wdm5, wdm7=_call_wdm7)
#: Species each scheme carries: the others are left out of its column.
_CARRIES = dict(wsm3=("qv", "qc", "qr"),
                wsm5=("qv", "qc", "qr", "qi", "qs"),
                wsm7=_SPECIES, wdm5=("qv", "qc", "qr", "qi", "qs"),
                wdm7=_SPECIES)


def _run(scheme, nz, *, steps=3, seed=3):
    """``steps`` calls; returns the post-state, accumulators and the pre-state."""
    import cupy as cp

    fields = _fields(nz, seed=seed)
    d = _device(cp, fields, _VOLUME)
    for name in _SPECIES:
        if name not in _CARRIES[scheme]:
            d[name] = cp.zeros_like(d[name])
    d["xland"] = cp.asarray(np.asarray(fields["xland"], np.float32))
    s = _surface(cp, NCOL)
    e = _effs(cp, d["theta"].shape)
    before = {n: cp.asnumpy(v).copy() for n, v in d.items()}
    for _ in range(steps):
        _CALL[scheme](cp, d, s, e)
    cp.cuda.Device().synchronize()
    out = {n: cp.asnumpy(v) for n, v in d.items()}
    out.update({f"s_{n}": cp.asnumpy(v) for n, v in s.items()})
    out.update({n: cp.asnumpy(v) for n, v in e.items()})
    return out, before


def _column_water(d, species):
    """kg m-2 per column: sum over levels of q * rho * dz."""
    mass = d["rho"].astype(np.float64) * d["dz"].astype(np.float64)
    return sum((d[n].astype(np.float64) * mass).sum(axis=0)
               for n in species)[0]


# ---------------------------------------------------------------------------
# 1. A 256-level column runs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scheme", SCHEMES)
def test_a_256_level_column_runs_conserves_water_and_moves(scheme):
    after, before = _run(scheme, DEEP)
    for name, value in after.items():
        if value.dtype.kind == "f":
            assert np.isfinite(value).all(), f"{scheme}: {name} not finite"
    for name in _CARRIES[scheme]:
        assert (after[name] >= 0).all(), f"{scheme}: {name} went negative"
    # The column moved: the guard (nz > KMAX) would return untouched inputs.
    assert (after["theta"] != before["theta"]).any(), scheme
    assert (after["qv"] != before["qv"]).any(), scheme
    moved = [n for n in _CARRIES[scheme]
             if n != "qv" and (after[n] != before[n]).any()]
    assert len(moved) >= 2, (scheme, moved)
    # Surface precipitation was booked (rain falls out of a 3.5 km column).
    assert (after["s_rainnc"] > 0).any(), scheme
    assert (after["s_rainnc"] >= 0).all()
    # Water: vapour + condensate before == after + surface precipitation
    # (rainnc is the total, snow/graupel/hail are parts of it).  Float32
    # microphysics, so a tolerance rather than a word.
    w0 = _column_water(before, _CARRIES[scheme])
    w1 = _column_water(after, _CARRIES[scheme]) + after["s_rainnc"][0]
    rel = np.abs(w1 - w0) / w0
    assert rel.max() < 5.0e-4, (scheme, float(rel.max()))


# ---------------------------------------------------------------------------
# 2. Nothing moved at or below the old ceiling
# ---------------------------------------------------------------------------

def _force_tier(monkeypatch, scheme, tier):
    """Make the shipped launcher compile ``tier`` whatever nz is."""
    if scheme == "wsm3":
        from woof.core import wsm3
        monkeypatch.setattr(wsm3, "_kernel_capacity", lambda nz: tier)
    elif scheme == "wsm5":
        from woof.core import column_schemes
        row = column_schemes.COLUMN_SCHEMES[4]
        monkeypatch.setitem(
            column_schemes.COLUMN_SCHEMES, 4,
            dataclasses.replace(row, level_tier=lambda nz: tier))
    elif scheme == "wsm7":
        from woof.core import wsm_family
        monkeypatch.setattr(wsm_family, "_FORCED_TIER", tier)
    elif scheme == "wdm5":
        from woof.core import wdm6
        monkeypatch.setattr(wdm6, "wdm5_level_tier", lambda nz: tier)
    else:
        from woof.core import wdm6
        monkeypatch.setattr(wdm6, "_kernel_capacity", lambda nz: tier)


def _words(out):
    return {n: np.ascontiguousarray(v).tobytes() for n, v in out.items()}


@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize("nz", [49, 80])
def test_the_deeper_tiers_move_no_bits_at_or_below_the_old_ceiling(
        scheme, nz, monkeypatch):
    """The shipped route (the unchanged 64 / 80 source) against the same
    source compiled at the 128 and 256 tiers: the tier only sizes arrays."""
    shipped, _ = _run(scheme, nz)
    assert (shipped["s_rainnc"] > 0).any(), "the A/B graded nothing"
    shipped = _words(shipped)
    for tier in (128, DEEP):
        with monkeypatch.context() as m:
            _force_tier(m, scheme, tier)
            deeper = _words(_run(scheme, nz)[0])
        for name in sorted(shipped):
            assert shipped[name] == deeper[name], (
                f"{scheme} {name} moved at nz={nz}: tier {tier}")


# ---------------------------------------------------------------------------
# 3. Bounds, registry, preflight
# ---------------------------------------------------------------------------

def test_the_contract_registry_and_ladders_say_256():
    import json
    from pathlib import Path

    import woof
    from woof.core import wdm6_constants as wdm
    from woof.core import wsm3_constants as wsm3
    from woof.core import wsm6_constants as wsm6
    from woof.core import wsm_family
    from woof import physics_vertical_contract as contract

    assert wsm3.WSM3_VERTICAL_LEVEL_BOUNDS == (2, DEEP)
    assert wsm6.WSM5_VERTICAL_LEVEL_BOUNDS == (2, DEEP)
    assert wdm.WDM5_VERTICAL_LEVEL_BOUNDS == (2, DEEP)
    assert wdm.WDM7_VERTICAL_LEVEL_BOUNDS == (2, DEEP)
    assert contract.WSM7_VERTICAL_LEVEL_BOUNDS == (2, DEEP)
    assert wdm.WDM_SCHEMES[26].vertical_level_bounds == (2, DEEP)
    for ladder in (wsm3.WSM3_KERNEL_LEVEL_TIERS,
                   wsm6.WSM5_KERNEL_LEVEL_TIERS,
                   wdm.WDM5_KERNEL_LEVEL_TIERS,
                   wdm.WDM6_KERNEL_LEVEL_TIERS,
                   wsm_family.row(24).level_tiers):
        assert ladder == (64, 80, 128, 256)
    assert wsm3.wsm3_level_tier(80) == 80 and wsm3.wsm3_level_tier(81) == 128
    assert wsm6.wsm5_level_tier(257 - 1) == 256
    assert wdm.wdm5_level_tier(129) == 256 and wdm.wdm6_level_tier(128) == 128
    for fn in (wsm3.wsm3_level_tier, wsm6.wsm5_level_tier,
               wdm.wdm5_level_tier, wdm.wdm6_level_tier):
        with pytest.raises(ValueError):
            fn(DEEP + 1)
    registry = json.loads((Path(woof.__file__).parent
                           / "physics_registry_v2.json").read_text())
    options = registry["components"]["microphysics"]["options"]
    for key in ("wsm3-mp3", "wsm5-mp4", "wsm7-mp24", "wdm5-mp14",
                "wdm7-mp26"):
        assert options[key]["consumers"]["vertical_level_bounds"][
            "maximum"] == DEEP, key


def test_hex_follows_the_new_bounds():
    """Hex runs these through the generic column seam (``engine_validated``):
    its own ceiling term is the structural 256 and the seam's resolved-physics
    vertical preflight, which reads the contract rows, admits 256 levels and
    refuses 257 for each scheme."""
    from woof.hex.vertical_spec import (
        hex_column_level_ceiling, hex_column_level_terms)
    from woof.physics_compat import validate_resolved_physics_vertical_levels
    from woof.physics_vertical_contract import PhysicsVerticalPreflightError

    assert hex_column_level_ceiling(microphysics=None, pbl=None) == DEEP
    assert hex_column_level_terms(
        microphysics="wsm5-mp4", pbl=None, engine_validated=True) == (
            ("the hex column (dycore and YSU-GWDO)", DEEP),)
    for mp in (3, 4, 14, 24, 26):
        validate_resolved_physics_vertical_levels({"mp_physics": mp, "nz": DEEP})
        with pytest.raises(PhysicsVerticalPreflightError):
            validate_resolved_physics_vertical_levels(
                {"mp_physics": mp, "nz": DEEP + 1})


def test_preflight_prices_the_deeper_frames():
    from woof.core import preflight as pf
    from woof.core import wsm_family

    frame, tier = pf.CONDITIONAL_COLUMN_TIERED_FRAMES["wsm3"]
    assert [frame.frame_bytes(tier(n)) for n in (49, 80, 128, DEEP)] == [
        5664, 7072, 11296, 22560]
    frame, tier = pf.CONDITIONAL_COLUMN_TIERED_FRAMES["wsm5"]
    assert [frame.frame_bytes(tier(n)) for n in (49, 80, 128, DEEP)] == [
        7072, 7072, 11296, 22560]
    assert [wsm_family.local_frame_bytes(24, n) for n in (49, 80, 128, DEEP)
            ] == [7216, 9008, 14384, 28720]
    assert [pf.WDM5_TIER_FRAME.frame_bytes(t) for t in (64, 80, 128, DEEP)
            ] == [9248, 11552, 18464, 36896]
    assert [pf.WDM7_TIER_FRAME.frame_bytes(t) for t in (64, 80, 128, DEEP)
            ] == [9776, 12208, 19504, 38960]


def test_the_priced_frames_are_what_the_driver_reports():
    """The pricing rows above, read back off the driver where it exposes the
    attribute (CUDA; the Metal backend has no local-size attribute)."""
    import re
    from pathlib import Path

    import cupy as cp

    from woof.core import preflight as pf
    from woof.core import kernels
    from woof.core.kernels import get_kernel_int_defines

    if not hasattr(cp.cuda.driver, "funcGetAttribute"):
        pytest.skip("backend exposes no CUfunction local-size attribute")

    def local_bytes(fn):
        if hasattr(fn, "attributes"):
            return int(fn.attributes["local_size_bytes"])
        return int(cp.cuda.driver.funcGetAttribute(3, fn.ptr))

    cases = {
        "wsm3": ("WSM3_KMAX", (), lambda t: pf.CONDITIONAL_COLUMN_TIERED_FRAMES[
            "wsm3"][0].frame_bytes(t)),
        "wsm5": ("WSM5_KMAX", (), lambda t: pf.CONDITIONAL_COLUMN_TIERED_FRAMES[
            "wsm5"][0].frame_bytes(max(t, 80))),
        "wdm5": ("WDM5_KMAX", (), pf.WDM5_TIER_FRAME.frame_bytes),
        "wdm7": ("WDM6_KMAX", (("WDM_HAIL", 1),), pf.WDM7_TIER_FRAME.frame_bytes),
    }
    from woof.core import wsm_family
    cases["wsm7"] = ("WSM7_KMAX", (), lambda t: dict(
        wsm_family.row(24).local_frame_bytes)[t])
    for unit, (define, extra, priced) in cases.items():
        source = (Path(kernels.__file__).parent / f"{unit}.cu").read_text()
        names = sorted(set(re.findall(
            r'extern\s+"C"\s+__global__\s+void\s+(\w+)', source)))
        if unit == "wdm7":
            names = ["wdm7_column"]
        for tier in (64, 80, 128, DEEP):
            widest = max(local_bytes(get_kernel_int_defines(
                unit, n, ((define, tier),) + extra)) for n in names)
            # The pricing is a ceiling: never below what the driver reports.
            assert widest <= priced(tier), (unit, tier, widest)
