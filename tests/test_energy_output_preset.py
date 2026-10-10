"""The ``energy`` history preset and the surface solar trio it samples.

CPU-only.  What is pinned here:

* the ``energy`` preset's membership, and that every member is a name the
  writer can carry;
* that a member a run does not produce (SWDDNI with no direct/diffuse
  shortwave, QICE under a warm-rain scheme) is simply absent from the tape,
  the rule every preset already follows -- never zero-filled, never refused;
* the WRF schema rows of SWDDNI, SWDDIF and COSZEN (units, stagger, type,
  Registry citation, history flag);
* the conditional-publication rule in ``woof.io.history_layout``: each of
  the three appears exactly when its producer exists;
* the disk planner and the preflight allocation rows agree with that rule
  per radiation selector;
* the restart and tiling classifications of the three driver buffers.

The device half (the radiation adapters actually returning the beam, and
the driver turning it into SWDDNI) is ``tests/test_energy_output_preset_gpu.py``.
"""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from woof.config import RunConfig
from woof.io import history_selection as hs
from woof.io.history_layout import (physics_history_fields,
                                    produced_history_shapes,
                                    surface_solar_history_fields)
from woof.io.wrf_output_schema import (HISTORY_FIELDS_BY_NETCDF_NAME,
                                       OUTPUT_FIELDS_BY_NETCDF_NAME,
                                       SURFACE_SOLAR_OUTPUT_FIELDS)

ENERGY_SPEC = frozenset({
    "U", "V", "W", "T", "PH", "PHB", "HGT", "P", "PB", "QVAPOR", "QCLOUD",
    "QRAIN", "QICE", "QSNOW", "QGRAUP", "T2", "Q2", "U10", "V10", "PSFC",
    "SWDOWN",
    "SWDDNI", "SWDDIF", "COSZEN", "RAINNC", "RAINC", "SINALPHA", "COSALPHA",
    "XLAT", "XLONG",
})


def _cfg(ra_lw: int, ra_sw: int, **overrides) -> RunConfig:
    return replace(RunConfig(nx=16, ny=12, nz=8, dx=3000.0, dy=3000.0,
                             ztop=12000.0, dt=6.0, run_seconds=120.0,
                             moist=True, mp_physics=8, sf_surface_physics=2,
                             sf_sfclay_physics=1, bl_pbl_physics=1),
                   ra_lw_physics=ra_lw, ra_sw_physics=ra_sw, **overrides)


# ------------------------------------------------------------------ preset

def test_energy_preset_is_exactly_the_requested_fields():
    assert hs.HISTORY_PRESETS["energy"] == ENERGY_SPEC


def test_every_energy_member_is_a_writable_history_name():
    assert ENERGY_SPEC <= hs.HISTORY_VOCABULARY


def test_energy_keeps_structural_fields_and_sheds_render_only_volumes():
    selection = hs.HistorySelection.from_mapping({"preset": "energy"})
    produced = tuple(sorted(hs.HISTORY_VOCABULARY))
    kept = set(selection.select(produced))
    assert hs.STRUCTURAL_FIELDS <= kept
    assert ENERGY_SPEC <= kept
    assert kept == ENERGY_SPEC | hs.STRUCTURAL_FIELDS
    assert not ({"REFL_10CM", "OLR", "EXCH_H", "TSK"} & kept)


def test_energy_tolerates_fields_the_run_does_not_produce():
    """A warm-rain, radiation-off run: no SWDDNI/SWDDIF/COSZEN, no ice.

    The preset is a filter over what the run produced, exactly as
    ``severe`` on a warm-rain scheme has no QICE: the absent names are
    absent from the tape, nothing raises and nothing is zero-filled.
    """
    selection = hs.HistorySelection.from_mapping({"preset": "energy"})
    produced = ("Times", "XTIME", "ITIMESTEP", "T", "U", "V", "W", "PH",
                "PHB", "MU", "MUB", "HGT", "P", "PB", "PSFC", "P_TOP", "ZNU",
                "ZNW", "QVAPOR", "QCLOUD", "QRAIN", "XLAT", "XLONG", "T2",
                "Q2", "U10", "V10", "RAINC", "RAINNC", "SINALPHA",
                "COSALPHA", "TSK")
    kept = selection.select(produced)
    assert not ({"SWDDNI", "SWDDIF", "COSZEN", "QICE", "QSNOW",
                 "SWDOWN"} & set(kept))
    assert "MU" not in kept and "TSK" not in kept
    # Order is the produced order: the filter only removes.
    assert list(kept) == [name for name in produced if name in kept]
    attrs = selection.wrfout_attrs(produced)
    assert attrs["GPUWM_HISTORY_PRESET"] == "energy"
    assert set(attrs["GPUWM_HISTORY_DROPPED"].split(",")) == {"MU", "TSK"}


def test_energy_may_be_trimmed_but_not_combined_with_history_vars():
    trimmed = hs.HistorySelection.from_mapping(
        {"preset": "energy", "history_drop": ["QSNOW"]})
    assert not trimmed.keeps("QSNOW") and trimmed.keeps("SWDDNI")
    with pytest.raises(ValueError, match="together with history_vars"):
        hs.HistorySelection.from_mapping(
            {"preset": "energy", "history_vars": ["T2"]})


def test_an_unknown_preset_refusal_names_energy():
    with pytest.raises(ValueError) as excinfo:
        hs.HistorySelection.from_mapping({"preset": "enrgy"})
    assert "'energy'" in str(excinfo.value)


def test_energy_product_cost_is_measured_not_borrowed_from_full():
    """It used to fall through to the ladder's full figure and claim all
    162 products were kept, while it sheds REFL_10CM, OLR and MU."""
    selection = hs.HistorySelection.from_mapping({"preset": "energy"})
    kept, total, exact = selection.product_cost()
    assert exact
    assert (kept, total) == hs.SEPARATELY_MEASURED_PRESET_PRODUCT_COUNTS[
        "energy"][:2]
    assert kept < total
    assert "energy" not in hs.PRESET_PRODUCT_COUNTS
    trimmed = hs.HistorySelection.from_mapping(
        {"preset": "energy", "history_drop": ["QSNOW"]})
    assert trimmed.product_cost() == (kept, total, False)


def test_energy_warning_states_its_own_basis_not_the_ladder(capsys):
    selection = hs.HistorySelection.from_mapping({"preset": "energy"})
    selection.warn_lost_products(sorted(hs.HISTORY_VOCABULARY), where="d01")
    text = capsys.readouterr().err
    kept, total, _ = hs.SEPARATELY_MEASURED_PRESET_PRODUCT_COUNTS["energy"][
        :3]
    assert f"keeps {kept} of {total} render products" in text
    assert "keeps all" not in text
    # The ladder attribution names fields energy keeps; it must not appear.
    assert "to U,V,W" not in text


# ------------------------------------------------------------------ schema

@pytest.mark.parametrize("name, description, units, registry, history", [
    ("SWDDNI", "Shortwave surface downward direct normal irradiance",
     "W m-2", "Registry.EM_COMMON:1719", False),
    ("SWDDIF", "Shortwave surface downward diffuse irradiance",
     "W m-2", "Registry.EM_COMMON:1723", False),
    ("COSZEN", "COS of SOLAR ZENITH ANGLE", "dimensionless",
     "Registry.EM_COMMON:997", True),
])
def test_surface_solar_schema_rows(name, description, units, registry,
                                   history):
    row = HISTORY_FIELDS_BY_NETCDF_NAME[name]
    assert row is SURFACE_SOLAR_OUTPUT_FIELDS[name]
    assert (row.netcdf_name, row.dtype, row.stagger) == (name, "f4", "")
    assert row.field_type == 104
    assert (row.description, row.units) == (description, units)
    assert row.registry == registry
    assert row.wrf_history is history
    # Model-state rows: the selector-driven inventory is untouched.
    assert name not in OUTPUT_FIELDS_BY_NETCDF_NAME
    assert name in hs.HISTORY_VOCABULARY


# ----------------------------------------------------- conditional writing

def _physics(**attrs):
    base = dict(fields={}, surface_dni=None, surface_dif=None,
                radiation_coszen=None, swint=None)
    base.update(attrs)
    return SimpleNamespace(**base)


def test_solar_trio_is_published_exactly_when_produced():
    dni, dif, cos = (np.full((2, 3), v, np.float32) for v in (1, 2, 3))
    out = surface_solar_history_fields(
        _physics(surface_dni=dni, surface_dif=dif, radiation_coszen=cos))
    assert out["SWDDNI"] is dni and out["SWDDIF"] is dif
    assert out["COSZEN"] is cos
    # Dudhia: COSZEN only, no fabricated split.
    out = surface_solar_history_fields(_physics(radiation_coszen=cos))
    assert set(out) == {"COSZEN"}
    # No producer at all.
    assert surface_solar_history_fields(_physics()) == {}


def test_half_a_direct_diffuse_pair_publishes_neither():
    dni = np.zeros((2, 3), np.float32)
    assert surface_solar_history_fields(_physics(surface_dni=dni)) == {}


def test_swint_publishes_the_per_step_interpolated_pair():
    held = np.zeros((2, 3), np.float32)
    per_step = {"swddni": np.ones((2, 3), np.float32),
                "swddif": np.full((2, 3), 2.0, np.float32)}
    out = surface_solar_history_fields(_physics(
        surface_dni=held, surface_dif=held, swint=object(), fields=per_step))
    assert out["SWDDNI"] is per_step["swddni"]
    assert out["SWDDIF"] is per_step["swddif"]
    # Without swint the radiation-call pair is published even if fields
    # happen to hold the BEP+BEM/slope_rad arrays.
    out = surface_solar_history_fields(_physics(
        surface_dni=held, surface_dif=held, fields=per_step))
    assert out["SWDDNI"] is held


def _driver_namespace(radiation_active: bool, **attrs):
    surface = np.zeros((2, 3), np.float32)
    fields = {name: surface for name in (
        "tsk", "t2", "th2", "q2", "u10", "v10", "ust", "hfx", "qfx", "lh",
        "pblh", "grdflx", "psim", "psih", "swdown", "glw")}
    namespace = _physics(
        fields=fields, radiation_active=radiation_active,
        rthratenlw=None, rthratensw=None, sase_active=False,
        hmix_k_diag=None, topo_shortwave=None, olr=None, rainc=None,
        mp_physics=0, microphysics=None,
        _zero_accumulator=lambda: np.zeros((2, 3), np.float32))
    for key, value in attrs.items():
        setattr(namespace, key, value)
    return namespace


def test_physics_history_carries_the_trio_only_while_radiation_runs():
    plane = np.ones((2, 3), np.float32)
    on = physics_history_fields(_driver_namespace(
        True, surface_dni=plane, surface_dif=plane, radiation_coszen=plane))
    assert {"SWDDNI", "SWDDIF", "COSZEN", "SWDOWN"} <= set(on)
    off = physics_history_fields(_driver_namespace(
        False, surface_dni=plane, surface_dif=plane, radiation_coszen=plane))
    assert not ({"SWDDNI", "SWDDIF", "COSZEN", "SWDOWN"} & set(off))


# ------------------------------------------- planner and allocation rows

@pytest.mark.parametrize("ra_lw, ra_sw, expected", [
    (4, 4, {"SWDDNI", "SWDDIF", "COSZEN"}),
    (1, 1, {"COSZEN"}),
    (0, 1, {"COSZEN"}),
    (4, 1, {"COSZEN"}),
    (1, 4, {"SWDDNI", "SWDDIF", "COSZEN"}),
    (0, 0, set()),
])
def test_planner_prices_the_trio_per_radiation_selector(ra_lw, ra_sw,
                                                         expected):
    from woof.core.preflight import physics_array_shapes

    cfg = _cfg(ra_lw, ra_sw)
    shapes = produced_history_shapes(cfg)
    assert {"SWDDNI", "SWDDIF", "COSZEN"} & set(shapes) == expected
    for name in expected:
        assert shapes[name] == (cfg.ny, cfg.nx)
    allocated = physics_array_shapes(cfg)
    rows = {"surface_dni": "SWDDNI", "surface_dif": "SWDDIF",
            "radiation_coszen": "COSZEN"}
    assert {rows[key] for key in rows if key in allocated} == expected


def test_energy_frame_is_smaller_than_full_and_keeps_the_trio():
    from woof.io.history_layout import history_frame_bytes

    cfg = _cfg(4, 4)
    shapes = produced_history_shapes(cfg)
    energy = hs.HistorySelection.from_mapping({"preset": "energy"})
    kept = set(energy.select(shapes))
    assert {"SWDDNI", "SWDDIF", "COSZEN", "SWDOWN", "U", "QVAPOR"} <= kept
    assert history_frame_bytes(cfg, energy) < history_frame_bytes(cfg)


# ------------------------------------------------ restart and tiling class

def test_driver_buffers_are_checkpoint_carried_and_tile_scattered():
    from woof.io import restart
    from tilestream.physics_inventory import OUTPUT_ONLY_DRIVER_ATTRS

    for name in ("surface_dni", "surface_dif", "radiation_coszen"):
        assert name in restart.DRIVER_CHECKPOINT_ONLY_ATTRS
        assert name not in restart.DRIVER_REBUILT_ATTRS
        assert name not in restart.DRIVER_SERIALIZED_ATTRS
        assert name in OUTPUT_ONLY_DRIVER_ATTRS
