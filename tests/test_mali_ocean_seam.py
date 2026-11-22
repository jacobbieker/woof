"""MALI's side of the land-ice/ocean seam (woof.hex.mpas_ice.ocean_seam_v842).

There is no oracle for the seam (MALI v8.4.2 standalone has no coupler), so it
is graded by conservation and by the model's own relations: the solid runoff
carries exactly the step's calved ice, the liquid runoff exactly the
hydrology's discharge across the grounding line into ocean cells, the shelf
pressure is the flotation pressure, and an imported floating basal mass
balance reaches the model's field word for word.
"""
from __future__ import annotations

import numpy as np
import pytest

from conftest import requires_gpu
from woof.verify import mali_forward_oracle as fo

TOLERANCE = {"r4": 1.0e-5, "r8": 1.0e-12}


def _case(name: str, precision: str) -> dict:
    return next(c for c in fo.cases(precision) if c["name"] == name)


def _steps(case: dict, config: dict | None = None):
    """Each step's pools, after a cold start."""
    from woof.hex.mpas_ice import forward_v842 as fw
    config = dict(case["config"], **(config or {}))
    pools = fo.device_pools(case)
    clock = fw.cold_start(pools=pools, config=config)
    while not clock.stopped:
        fw.advance(pools=pools, config=config, clock=clock, n_steps=1)
        yield pools, config


def _host(pools: dict, pool: str, name: str) -> np.ndarray:
    import cupy as cp
    return cp.asnumpy(pools[pool][name])


def _close(a: float, b: float, precision: str, scale: float) -> bool:
    return abs(a - b) <= TOLERANCE[precision] * max(scale, 1e-300)


@requires_gpu
@pytest.mark.parametrize("precision", ["r4", "r8"])
@pytest.mark.parametrize("mesh", ["uniform", "varres"])
def test_the_solid_runoff_carries_the_calved_ice(precision, mesh):
    import cupy as cp
    from woof.hex.mpas_ice import ocean_seam_v842 as seam
    case = _case(f"dome_calving_{mesh}", precision)
    calved_any = False
    for pools, config in _steps(case):
        out = {k: cp.asnumpy(v) for k, v in seam.export_fields(pools=pools, config=config).items()}
        area = _host(pools, "mesh", "areaCell").astype(np.float64)
        deltat = float(_host(pools, "mesh", "deltat")[0])
        calved = config["config_ice_density"] * float(np.sum(_host(pools, "geometry", "calvingThickness") * area))
        sent = float(np.sum(out["li2ocn_solid_runoff"].astype(np.float64) * area)) * deltat
        assert _close(sent, calved, precision, abs(calved)), (sent, calved)
        assert (out["li2ocn_solid_runoff"] >= 0).all()
        calved_any |= calved > 0
        # no hydrology: no liquid runoff
        assert not out["li2ocn_liquid_runoff"].any()
    assert calved_any


@requires_gpu
@pytest.mark.parametrize("precision", ["r4", "r8"])
@pytest.mark.parametrize("mesh", ["uniform", "varres"])
def test_the_liquid_runoff_carries_the_grounding_line_discharge(precision, mesh):
    import cupy as cp
    from woof.hex.mpas_ice import ocean_seam_v842 as seam
    case = _case(f"hydro_marine_{mesh}", precision)
    edges = 0
    for pools, config in _steps(case):
        out = {k: cp.asnumpy(v) for k, v in seam.export_fields(pools=pools, config=config).items()}
        rate, receiver = seam.grounding_line_discharge(pools, config)
        mask = _host(pools, "geometry", "cellMask")
        is_ocean = seam.ocean(mask, _host(pools, "geometry", "bedTopography"), config["config_sea_level"])
        cells_on_edge = _host(pools, "mesh", "cellsOnEdge").reshape(-1, 2) - 1
        on_line = receiver >= 0
        edges += int(on_line.sum())
        for e in np.nonzero(on_line)[0]:
            c = receiver[e]
            other = cells_on_edge[e, 0] if cells_on_edge[e, 1] == c else cells_on_edge[e, 1]
            assert is_ocean[c] and seam.grounded(mask[other:other + 1])[0]
        area = _host(pools, "mesh", "areaCell").astype(np.float64)
        sent = float(np.sum(out["li2ocn_liquid_runoff"].astype(np.float64) * area)) / seam.RHO_WATER
        discharged = float(np.sum(rate[on_line].astype(np.float64)))
        assert _close(sent, discharged, precision, float(np.sum(np.abs(rate[on_line])))), (sent, discharged)
        # only ocean cells receive it
        assert not out["li2ocn_liquid_runoff"][~is_ocean].any()
    assert edges > 0


@requires_gpu
@pytest.mark.parametrize("precision", ["r4", "r8"])
def test_the_shelf_fields_follow_the_floating_mask_and_flotation(precision):
    import cupy as cp
    from woof.hex.mpas_ice import ocean_seam_v842 as seam
    case = _case("hydro_marine_uniform", precision)
    pools, config = next(_steps(case))
    out = {k: cp.asnumpy(v) for k, v in seam.export_fields(pools=pools, config=config, diagnostics=True).items()}
    shelf = seam.floating(_host(pools, "geometry", "cellMask"))
    assert shelf.any()
    assert out["landIceMask"].dtype == np.int32
    assert np.array_equal(out["landIceMask"], shelf.astype(np.int32))
    assert np.array_equal(out["landIceFraction"], shelf.astype(out["landIceFraction"].dtype))
    assert not out["landIcePressure"][~shelf].any() and not out["landIceDraft"][~shelf].any()
    # floating: rhoi H = rhoo (sea level - draft)
    p = out["landIcePressure"][shelf].astype(np.float64)
    flotation = (config["config_ocean_density"] * seam.GRAVITY
                 * (config["config_sea_level"] - out["landIceDraft"][shelf].astype(np.float64)))
    assert np.all(np.abs(p - flotation) <= 4 * TOLERANCE[precision] * p), np.max(np.abs(p - flotation) / p)


@requires_gpu
@pytest.mark.parametrize("precision", ["r4", "r8"])
def test_the_floating_basal_mass_balance_import_is_matched(precision):
    import cupy as cp
    from woof.hex.coupled.exchange import FluxChannel, check_matched
    from woof.hex.mpas_ice import forward_v842 as fw, ocean_seam_v842 as seam
    case = _case("hydro_marine_uniform", precision)
    config = dict(case["config"], config_basal_mass_bal_float="file")
    pools = fo.device_pools(case)
    clock = fw.cold_start(pools=pools, config=config)
    n = pools["geometry"]["thickness"].shape[0]
    real = pools["geometry"]["thickness"].dtype
    sent = np.where(np.arange(n) % 3 == 0, -2.0e-5, 1.0e-6).astype(real)
    seam.import_fields(pools=pools, config=config, fields={"ocn2li_floating_basal_mass_bal": cp.asarray(sent)})
    channel = FluxChannel("ocn2li_floating_basal_mass_bal", "ocean", "landice", "floating basal mass balance",
                          "kg/m^2/s")
    check_matched(channel, sent, _host(pools, "geometry", "floatingBasalMassBal"))
    # the 'file' path keeps it through a step
    fw.advance(pools=pools, config=config, clock=clock, n_steps=1)
    check_matched(channel, sent, _host(pools, "geometry", "floatingBasalMassBal"))


@requires_gpu
def test_the_seam_refuses_by_name():
    import cupy as cp
    from woof.hex.errors import ConfigurationRefusal
    from woof.hex.mpas_ice import forward_v842 as fw, ocean_seam_v842 as seam
    for name in ("li2ocn_momentum_x", "li2ocn_momentum_y"):
        with pytest.raises(ConfigurationRefusal, match="calving momentum"):
            seam.refuse_channel(name)
    with pytest.raises(ConfigurationRefusal, match="not a field"):
        seam.refuse_channel("li2ocn_mass")
    case = _case("hydro_marine_uniform", "r4")
    pools = fo.device_pools(case)
    fw.cold_start(pools=pools, config=case["config"])
    n = pools["geometry"]["thickness"].shape[0]
    melt = cp.zeros(n, cp.float32)
    with pytest.raises(ConfigurationRefusal, match="config_basal_mass_bal_float"):
        seam.import_fields(pools=pools, config=dict(case["config"], config_basal_mass_bal_float="constant"),
                           fields={"ocn2li_floating_basal_mass_bal": melt})
    with pytest.raises(ConfigurationRefusal, match="landIcePressure"):
        seam.import_fields(pools=pools, config=case["config"], fields={"landIcePressure": melt})
    with pytest.raises(ValueError, match="shape"):
        seam.import_fields(pools=pools, config=dict(case["config"], config_basal_mass_bal_float="file"),
                           fields={"ocn2li_floating_basal_mass_bal": melt[:-1]})
    # exports need a step's deltat
    pools["mesh"]["deltat"][...] = 0
    with pytest.raises(ValueError, match="deltat"):
        seam.export_fields(pools=pools, config=case["config"])


@requires_gpu
def test_runoff_into_a_halo_cell_is_refused_not_dropped():
    """On a partitioned mesh a receiving ocean cell past nCellsSolve is
    another rank's: the single-rank seam refuses it by name."""
    import cupy as cp
    from woof.hex.errors import ConfigurationRefusal
    from woof.hex.mpas_ice import ocean_seam_v842 as seam
    case = _case("hydro_marine_uniform", "r8")
    for pools, config in _steps(case):
        rate, receiver = seam.grounding_line_discharge(pools, config)
        if (receiver >= 0).any():
            break
    seam.export_fields(pools=pools, config=config)   # unpartitioned: every receiver is owned
    ncs = int(receiver[receiver >= 0].max())
    pools["mesh"]["nCellsSolve"] = cp.asarray([ncs], cp.int32)
    with pytest.raises(ConfigurationRefusal, match="halo cells"):
        seam.export_fields(pools=pools, config=config)
