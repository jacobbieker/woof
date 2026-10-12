"""The PBL column kernels' vertical bounds, and where a user meets them.

Two surfaces, one pair of numbers each.  ``woof check`` must refuse an
impossible vertical grid BEFORE a user pays for fetch and preparation, and
the launcher must refuse it by name if anything ever reaches it anyway.  The
defect these pin: a 130-level YSU configuration passed ``woof check`` and
both preparation stages, then died mid-run on a bare
``ValueError: nz=130 exceeds YSU_KMAX=128``.  YSU, Shin-Hong, Kain-Fritsch
and terrain drag now sit at the dycore's 256-level ceiling, so the
over-the-ceiling cases here are 257 levels.

No device is opened here.  Every launcher rejects the level count before it
touches an array, so a host array of the right shape reaches the refusal.
"""

from __future__ import annotations

import numpy as np
import pytest

from woof.physics_vertical_contract import (
    ACOUSTIC_VERTICAL_LEVEL_BOUNDS,
    KF_VERTICAL_LEVEL_BOUNDS,
    MYJ_VERTICAL_LEVEL_BOUNDS,
    PhysicsVerticalPreflightError,
    SASE_VERTICAL_LEVEL_BOUNDS,
    SHINHONG_VERTICAL_LEVEL_BOUNDS,
    TERRAIN_DRAG_VERTICAL_LEVEL_BOUNDS,
    YSU_VERTICAL_LEVEL_BOUNDS,
)


def test_the_contract_bounds_are_the_launchers_own_bounds():
    """The standalone contract restates each launcher's number; pin them.

    ``woof/physics_vertical_contract.py`` deliberately imports no forecast
    executor, so it cannot read these off the runtime modules.  This is what
    stops the restatement drifting from the kernel that owns it.
    """
    from woof.core import myjpbl, sase_limits, shinhong, ysu

    assert YSU_VERTICAL_LEVEL_BOUNDS == ysu.VERTICAL_LEVEL_BOUNDS
    assert SHINHONG_VERTICAL_LEVEL_BOUNDS == shinhong.VERTICAL_LEVEL_BOUNDS
    assert MYJ_VERTICAL_LEVEL_BOUNDS == (myjpbl.MYJ_MIN_COLUMN_LEVELS,
                                         myjpbl.MYJ_MAX_COLUMN_LEVELS)
    assert SASE_VERTICAL_LEVEL_BOUNDS[1] == sase_limits.MAX_COLUMN_LEVELS


def test_the_guard_only_ceilings_are_the_launchers_own_bounds():
    """YSU (regional, global, ensemble batch), KF and terrain drag.

    Each of these was a 128-level guard over a column the kernel no longer
    sizes at 128 (or, for terrain drag, a per-thread profile it now sizes at
    256); they sit at the dycore's own 256-level ceiling.
    """
    from woof.core import kf, terrain_drag
    from woof.globe.core import ysu as globe_ysu

    assert KF_VERTICAL_LEVEL_BOUNDS == kf.VERTICAL_LEVEL_BOUNDS
    assert TERRAIN_DRAG_VERTICAL_LEVEL_BOUNDS == terrain_drag.VERTICAL_LEVEL_BOUNDS
    assert YSU_VERTICAL_LEVEL_BOUNDS == globe_ysu.VERTICAL_LEVEL_BOUNDS
    dycore_ceiling = ACOUSTIC_VERTICAL_LEVEL_BOUNDS[1]
    for bounds in (YSU_VERTICAL_LEVEL_BOUNDS, SHINHONG_VERTICAL_LEVEL_BOUNDS,
                   KF_VERTICAL_LEVEL_BOUNDS,
                   TERRAIN_DRAG_VERTICAL_LEVEL_BOUNDS):
        assert bounds[1] == dycore_ceiling == 256


def test_the_ensemble_batch_ysu_plan_follows_the_contract():
    """The batch planner once carried its own literal 128."""
    from woof.ensemble.batch_physics import ysu_output_plan

    low, high = YSU_VERTICAL_LEVEL_BOUNDS
    ysu_output_plan(nz=high, ny=2, nx=2, members=2)
    for nz in (low - 1, high + 1):
        with pytest.raises(ValueError, match=f"{low}..{high} levels"):
            ysu_output_plan(nz=nz, ny=2, nx=2, members=2)


@pytest.mark.parametrize("define, module_name, bounds", (
    ("YSU_KMAX", "ysu", YSU_VERTICAL_LEVEL_BOUNDS),
    ("SHINHONG_KMAX", "shinhong", SHINHONG_VERTICAL_LEVEL_BOUNDS),
    ("MYJ_KMAX", "myjpbl", MYJ_VERTICAL_LEVEL_BOUNDS),
    ("SASE_KMAX", "sase", SASE_VERTICAL_LEVEL_BOUNDS),
    ("KF_KMAX", "kf", KF_VERTICAL_LEVEL_BOUNDS),
    ("TD_KMAX", "terrain_drag", TERRAIN_DRAG_VERTICAL_LEVEL_BOUNDS),
    ("YSU_KMAX", "globe:ysu", YSU_VERTICAL_LEVEL_BOUNDS),
))
def test_the_declared_ceiling_is_the_compiled_kernel_ceiling(
        define, module_name, bounds):
    """The ceiling is a compile-time array extent, so read it from the .cu."""
    from pathlib import Path

    import woof.core.kernels as kernels
    import woof.globe.core.kernels as globe_kernels

    package = kernels
    if module_name.startswith("globe:"):
        package, module_name = globe_kernels, module_name.split(":", 1)[1]
    source = (Path(package.__file__).with_name(f"{module_name}.cu")
              .read_text(encoding="utf-8"))
    line = next(ln for ln in source.splitlines()
                if ln.startswith(f"#define {define} "))
    assert int(line.split()[-1]) == bounds[1]


def _column(nz: int) -> np.ndarray:
    return np.zeros((nz, 2, 2), dtype=np.float32)


def test_ysu_refuses_too_many_levels_by_name():
    from woof.core.ysu import launch_ysu

    with pytest.raises(PhysicsVerticalPreflightError) as caught:
        launch_ysu(*(_column(257) for _ in range(6)), None, None, None, None,
                   psfc=None, znt=None, ust=None, hfx=None, qfx=None,
                   wspd=None, br=None, psim=None, psih=None, xland=None,
                   u10=None, v10=None, dt=60.0)
    message = str(caught.value)
    assert "YSU PBL requires 4 <= nz <= 256, got nz=257" in message
    assert "woof check" in message


def test_ysu_refuses_too_few_levels_by_name():
    from woof.core.ysu import launch_ysu

    with pytest.raises(PhysicsVerticalPreflightError) as caught:
        launch_ysu(*(_column(3) for _ in range(6)), None, None, None, None,
                   psfc=None, znt=None, ust=None, hfx=None, qfx=None,
                   wspd=None, br=None, psim=None, psih=None, xland=None,
                   u10=None, v10=None, dt=60.0)
    assert "YSU PBL requires 4 <= nz <= 256, got nz=3" in str(caught.value)


def test_shinhong_refuses_out_of_range_levels_by_name():
    from woof.core.shinhong import launch_shinhong

    with pytest.raises(PhysicsVerticalPreflightError) as caught:
        launch_shinhong(*(_column(257) for _ in range(6)), None, None, None,
                        None, None, psfc=None, znt=None, ust=None, hfx=None,
                        qfx=None, wspd=None, br=None, psim=None, psih=None,
                        xland=None, u10=None, v10=None, corf=None,
                        dt=60.0, dx=1000.0, dy=1000.0)
    assert ("Shin-Hong PBL requires 4 <= nz <= 256, got nz=257"
            in str(caught.value))


def test_global_ysu_refuses_too_many_levels_by_name():
    from woof.globe.core.ysu import launch_ysu

    with pytest.raises(PhysicsVerticalPreflightError) as caught:
        launch_ysu(*(_column(257) for _ in range(6)), None, None, None, None,
                   psfc=None, znt=None, ust=None, hfx=None, qfx=None,
                   wspd=None, br=None, psim=None, psih=None, xland=None,
                   u10=None, v10=None, dt=60.0)
    message = str(caught.value)
    assert "YSU PBL requires 4 <= nz <= 256, got nz=257" in message
    assert "workspace" in message and "local memory" not in message


def test_kf_refuses_out_of_range_levels():
    from woof.core.kf import kernel_capacity, launch_kf

    assert kernel_capacity(256) == 256
    for nz in (7, 257):
        with pytest.raises(ValueError,
                           match=f"KF requires 8 <= nz <= 256, got {nz}"):
            launch_kf(*(_column(nz) for _ in range(9)),
                      dx=12000.0, dt=60.0, cudt=5.0)


@pytest.mark.parametrize("gwd_opt", (1, 3))
def test_terrain_drag_refuses_too_many_levels_by_name(gwd_opt):
    from woof.core import terrain_drag

    launch = (terrain_drag.launch_gwdo if gwd_opt == 1
              else terrain_drag.launch_gwdo_gsl)
    with pytest.raises(PhysicsVerticalPreflightError) as caught:
        launch({}, _column(257), _column(257), **{
            name: None for name in (
                ("var", "con", "oa", "ol", "sina", "cosa") if gwd_opt == 1
                else ("ls", "ss", "sina", "cosa", "xland", "br", "pblh",
                      "kpbl"))},
            **({"kpblmax": 10} if gwd_opt == 3 else {}),
            dx=3000.0, dt=60.0)
    message = str(caught.value)
    assert (f"Terrain drag gwd_opt={gwd_opt} requires nz <= 256, "
            "got nz=257") in message
    assert "woof check" in message


@pytest.mark.parametrize("gwd_opt", (1, 3))
def test_the_preflight_refuses_a_too_deep_terrain_drag_run(gwd_opt):
    """Terrain drag used to meet its ceiling only at the first PBL call."""
    from woof.physics_compat import validate_resolved_physics_vertical_levels

    selection = {
        "cu_physics": 0, "sf_surface_physics": 0, "mp_physics": 0,
        "bl_pbl_physics": 0, "ra_lw_physics": 0, "ra_sw_physics": 0,
        "sf_sfclay_physics": 0, "ra_rrtmg_variant": "rte-rrtmgp",
        "gwd_opt": gwd_opt,
    }
    label = f"Terrain drag gwd_opt={gwd_opt}"
    receipt = validate_resolved_physics_vertical_levels(
        dict(selection, nz=256))
    row = next(c for c in receipt["checks"] if c["component"] == label)
    assert (row["minimum"], row["maximum"]) == (None, 256)
    with pytest.raises(PhysicsVerticalPreflightError) as caught:
        validate_resolved_physics_vertical_levels(dict(selection, nz=257))
    assert f"{label} requires nz <= 256, got nz=257" in str(caught.value)
    # Off, it is not asked at all.
    receipt = validate_resolved_physics_vertical_levels(
        dict(selection, nz=257, gwd_opt=0))
    assert not [c for c in receipt["checks"]
                if c["component"].startswith("Terrain drag")]


def test_myj_refuses_out_of_range_levels_by_name():
    from woof.core.myjpbl import launch_myj_pbl

    with pytest.raises(PhysicsVerticalPreflightError) as caught:
        launch_myj_pbl({"dz": _column(130)}, {"psfc": np.zeros((2, 2))},
                       None, None, None, dtturbl=60.0, flqi=False)
    assert ("MYJ PBL requires 4 <= nz <= 128, got nz=130"
            in str(caught.value))


def test_a_named_refusal_is_still_a_value_error():
    """Existing callers catch ValueError; the named class must not break them."""
    assert issubclass(PhysicsVerticalPreflightError, ValueError)


def test_the_preflight_and_the_launcher_name_the_same_thing():
    """One grammar, so a user meets the same sentence at check and at run."""
    from woof.physics_compat import (
        PhysicsVerticalPreflightError as CompatError,
        validate_resolved_physics_vertical_levels,
    )

    assert CompatError is PhysicsVerticalPreflightError
    with pytest.raises(PhysicsVerticalPreflightError) as caught:
        validate_resolved_physics_vertical_levels({
            "nz": 257, "cu_physics": 0, "sf_surface_physics": 0,
            "mp_physics": 8, "bl_pbl_physics": 1, "ra_lw_physics": 0,
            "ra_sw_physics": 0, "sf_sfclay_physics": 0,
            "ra_rrtmg_variant": "rte-rrtmgp",
        })
    assert "YSU PBL requires 4 <= nz <= 256, got nz=257" in str(caught.value)
