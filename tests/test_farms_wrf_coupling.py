"""WRF swint_opt = 2 (FARMS) in a running WRF domain.

The kernel is held to compiled WRF word for word by
tests/test_farms_wrf471_parity.py.  This file runs short moist forecasts
(Thompson, legacy RRTMG, radiation every third step, a June afternoon sun)
and checks what WRF's radiation driver does around FARMS:

* every FARMS output is finite and the sun is up in it;
* with couple_farms off, swint_opt = 2 changes no prognostic word and no
  carried field except FARMS's own six and the two the driver's
  Solar_step rewrites on every step under swint_opt = 2 -- the history
  CLDFRA and COSZEN (module_radiation_driver.F:1176-1609; between radiation
  calls WRF otherwise holds both at the last call's values);
* on a radiation step, the per-step CLDFRA FARMS reads is the radiation
  call's own CLDFRA word for word, and the Solar_step COSZEN the radiation
  call's COSZEN;
* couple_farms copies SWDOWN2 into SWDOWN, which feeds the land surface.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

FARMS = ("swdown2", "swddir2", "swddni2", "swddif2", "swdownc2", "swddnic2")
NSTEPS = 8


def _cfg(swint, couple, rung):
    from tilestream import harness
    from woof.config import validate_run_config
    physics = dict(bl_pbl_physics=1, sf_sfclay_physics=91, sf_surface_physics=2)
    if rung == "mynn":
        physics = dict(bl_pbl_physics=5, sf_sfclay_physics=5,
                       sf_surface_physics=3, num_soil_layers=9)
    return validate_run_config(harness.make_config(
        16, 12, 20, moist=True, mp_physics=8, ra_physics=4,
        ra_rrtmg_variant="rrtmg_legacy", dt=1.0, time_step_sound=4,
        ztop=20000.0, radt_minutes=0.05, swint_opt=swint,
        couple_farms=couple, **physics))


def _run(swint, couple, rung="ysu"):
    import cupy as cp
    from tilestream.physics_inventory import default_builder
    from woof.core import dycore

    if rung == "mynn":
        from woof import backend
        if backend.active() != "cuda":
            pytest.skip("MYNN/RUC units do not translate to Metal yet")
    cfg = _cfg(swint, couple, rung)
    state, drv = default_builder(
        cfg, 4242, start_time=datetime(2024, 6, 21, 17, tzinfo=timezone.utc))
    checks = {"cldfra": [], "coszen": [], "couple": [], "farms": []}
    if swint == 2:
        step = drv.farms.step
        captured = []

        def spy_step(*args, **kw):
            cf = step(*args, **kw)
            captured.append(cf.copy())
            return cf
        drv.farms.step = spy_step
        run_farms = drv._run_farms

        def spy_run(atmosphere, st, cfg_, *, itimestep, radiation_due):
            if radiation_due:
                mine = drv.farms.radiation_coszen_at(
                    float(st.elapsed_seconds), np.float32(drv.radt_minutes))
                checks["coszen"].append(int(
                    (cp.asnumpy(mine).view(np.uint32)
                     != cp.asnumpy(drv.fields["coszen"]).view(np.uint32)).sum()))
            return run_farms(atmosphere, st, cfg_, itimestep=itimestep,
                             radiation_due=radiation_due)
        drv._run_farms = spy_run
    for _ in range(NSTEPS):
        before = drv.call_counts.get("radiation", 0)
        dycore.step(state, cfg)
        if swint != 2:
            continue
        checks["farms"].append({k: cp.asnumpy(drv.fields[k]) for k in FARMS})
        if drv.call_counts.get("radiation", 0) > before:
            checks["cldfra"].append(int(
                (cp.asnumpy(drv.cldfra).view(np.uint32)
                 != cp.asnumpy(captured[-1]).view(np.uint32)).sum()))
        if couple:
            checks["couple"].append(bool(
                (cp.asnumpy(drv.fields["swdown"]).view(np.uint32)
                 == cp.asnumpy(drv.fields["swdown2"]).view(np.uint32)).all()))
    words = {}
    for name in ("u", "v", "w", "ph", "t", "mu", "qv", "qc", "qr", "qi",
                 "qs", "qg"):
        a = getattr(state, name, None)
        if a is not None:
            words["state/" + name] = cp.asnumpy(a)
    for key, a in drv.fields.items():
        if hasattr(a, "shape"):
            words["fields/" + key] = cp.asnumpy(a)
    for name in ("cldfra", "radiation_coszen", "rthraten"):
        a = getattr(drv, name, None)
        if a is not None:
            words["driver/" + name] = cp.asnumpy(a)
    return words, checks


def _differing(a, b):
    return sorted(k for k in set(a) & set(b)
                  if a[k].shape != b[k].shape or
                  (np.ascontiguousarray(a[k]).view(np.uint8)
                   != np.ascontiguousarray(b[k]).view(np.uint8)).any())


@pytest.mark.parametrize("rung", ["ysu", "mynn"])
def test_farms_changes_only_its_own_words_and_the_solar_step(rung):
    off, _ = _run(0, False, rung)
    on, checks = _run(2, False, rung)
    assert set(on) - set(off) == {"fields/" + k for k in FARMS}
    # WRF's Solar_step rewrites CLDFRA and COSZEN every step under
    # swint_opt = 2; nothing else moves, the prognostic state included.
    expected = {"driver/cldfra", "driver/radiation_coszen", "fields/coszen"}
    assert set(_differing(off, on)) <= expected, _differing(off, on)
    assert not any(k.startswith("state/") for k in _differing(off, on))
    # every FARMS word finite, the June afternoon sun up in every column
    for row in checks["farms"]:
        for name, value in row.items():
            assert np.isfinite(value).all(), name
        assert (row["swdown2"] > 0).all() and (row["swdownc2"] > 0).all()
        assert (row["swdown2"] <= row["swdownc2"] * 1.5).all()
    # radiation steps: FARMS read the radiation call's own CLDFRA, and the
    # Solar_step COSZEN is the radiation call's COSZEN
    assert len(checks["cldfra"]) >= 2 and set(checks["cldfra"]) == {0}
    assert len(checks["coszen"]) >= 2 and set(checks["coszen"]) == {0}


def test_couple_farms_feeds_swdown_to_the_land_surface():
    on, _ = _run(2, False)
    coupled, checks = _run(2, True)
    assert checks["couple"] and all(checks["couple"])
    moved = _differing(on, coupled)
    # SWDOWN feeds Noah (SOLDN = SWDOWN), so the surface and then the
    # atmosphere move; GSW stays the radiation call's on every step.
    assert "fields/swdown" in moved and "fields/tsk" in moved
    assert any(k.startswith("state/") for k in moved)
    for name, value in coupled.items():
        if value.dtype.kind == "f":
            assert np.isfinite(value).all(), name
