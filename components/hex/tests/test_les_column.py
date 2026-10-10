"""The LES-ready column: free level count, the LES vertical preset, PBL off.

CPU-only.  Each new control has a focused failure test beside its success
test (CONTRIBUTING.md): the level count is read from the files and a
disagreeing init refuses; the device footprint is priced per level; the LES
preset is thin near the ground, stretches gently and keeps the dycore's
damping layer; ``--pbl off`` is admitted only where a closure or the
resolution carries the boundary layer, and a run without the flag is
unchanged.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from woof.hex import device_admission, pbl_admission
from woof.hex import forecast_door as door
from woof.hex.errors import ConfigurationRefusal
from woof.hex.vertical_spec import (
    DYCORE_DAMPING_START_M,
    LES_MAX_STRETCH,
    VerticalSpec,
    VerticalSpecError,
    les_interfaces,
    les_layer_report,
    load_vertical_spec,
    resolve_preset,
)


# ---------------------------------------------------------------------------
# the LES vertical preset
# ---------------------------------------------------------------------------
def test_les_preset_is_thin_near_the_ground_and_stretches_gently():
    spec = VerticalSpec.les()
    spec.validate()
    assert spec.scheme == "specified"
    assert spec.n_vert_levels == 80
    zw = np.asarray(spec.specified_interfaces_m)
    dz = np.diff(zw)
    assert zw[0] == 0.0 and zw[-1] == spec.ztop_m == 30_000.0
    # About 20-25 m through the lowest hundred-odd metres.
    assert np.all(dz[:7] >= 20.0 - 1e-9) and np.all(dz[:7] <= 25.0)
    assert dz[0] == pytest.approx(20.0)
    ratios = dz[1:] / dz[:-1]
    assert ratios.max() <= LES_MAX_STRETCH
    assert ratios.min() >= 1.0 - 1e-12
    report = les_layer_report(spec)
    assert report["layers_above_damping_start"] >= 3
    assert report["layers_below_100m"] >= 4


def test_les_preset_keeps_the_dycore_damping_layer():
    with pytest.raises(VerticalSpecError, match="damping"):
        VerticalSpec.les(top_m=20_000.0)
    with pytest.raises(VerticalSpecError, match="damping"):
        les_interfaces(top_m=DYCORE_DAMPING_START_M)


def test_les_preset_refuses_a_column_it_cannot_build_gently():
    # Too few levels: reaching 30 km from 20 m needs a steep stretch.
    with pytest.raises(VerticalSpecError, match="levels reach it"):
        VerticalSpec.les(levels=30)
    # Too thick: the surface thickness alone overshoots the top.
    with pytest.raises(VerticalSpecError, match="already reach"):
        les_interfaces(levels=80, dz_surface_m=400.0)


def test_les_preset_refuses_a_column_the_forecast_cannot_run():
    from woof.hex.vertical_spec import HEX_MAX_COLUMN_LEVELS

    assert VerticalSpec.les(levels=HEX_MAX_COLUMN_LEVELS).n_vert_levels == 80
    with pytest.raises(VerticalSpecError, match="WSM6"):
        VerticalSpec.les(levels=HEX_MAX_COLUMN_LEVELS + 1)


def test_the_column_ceiling_is_the_engines_wsm6_tier():
    from woof.core import wsm6_constants
    from woof.hex.vertical_spec import HEX_MAX_COLUMN_LEVELS

    assert HEX_MAX_COLUMN_LEVELS == wsm6_constants.WSM6_DEEP_KMAX
    assert device_admission.WSM6_DEEP_TIER_LEVELS == wsm6_constants.WSM6_DEEP_KMAX
    assert device_admission.WSM6_SHALLOW_TIER_LEVELS == wsm6_constants.WSM6_SHALLOW_KMAX
    from woof.core import preflight

    frame = preflight.WSM6_TIER_FRAME
    assert device_admission.WSM6_FRAME_BYTES_PER_LEVEL == frame.bytes_per_level
    for levels in (55, 64, 70, 80):
        assert device_admission.wsm6_local_frame_bytes(levels) == frame.frame_bytes(
            wsm6_constants.wsm6_level_tier(levels)
        )


def test_les_preset_passes_the_vertical_construction(tmp_path):
    """The preset goes through the sole numerical path on a real mesh shape."""

    from woof.hex.vertical import build_vertical_grid

    spec = VerticalSpec.les(levels=80, dz_surface_m=25.0)
    mesh = _tiny_mesh()
    vertical = build_vertical_grid(
        mesh, np.zeros(mesh.n_cells), n_vert_levels=spec.n_vert_levels,
        ztop=spec.ztop_m, scheme="specified",
        specified_zw=np.asarray(spec.specified_interfaces_m),
    )
    assert vertical.zgrid.shape == (81, mesh.n_cells)
    assert np.allclose(vertical.zgrid[:, 0], spec.specified_interfaces_m)


def test_named_presets_resolve_and_bad_ones_refuse(tmp_path):
    assert resolve_preset("preset:default") == VerticalSpec()
    assert resolve_preset("preset:les") == VerticalSpec.les()
    assert resolve_preset("preset:les:levels=70,dz_surface_m=25").n_vert_levels == 70
    assert load_vertical_spec("preset:les").sha256() == VerticalSpec.les().sha256()
    for bad, match in (
        ("preset:nope", "no vertical preset"),
        ("preset:les:colour=red", "takes"),
        ("preset:les:levels=many", "not a int"),
    ):
        with pytest.raises(VerticalSpecError, match=match):
            resolve_preset(bad)
    # A JSON file still works, and from_file reads a preset too.
    path = tmp_path / "les.json"
    path.write_bytes(VerticalSpec.les().canonical_bytes())
    assert VerticalSpec.from_file(path) == VerticalSpec.les()
    assert VerticalSpec.from_file("preset:les") == VerticalSpec.les()


def test_a_preset_is_materialized_as_a_file_the_receipt_can_name(tmp_path):
    from woof.hex.vertical_spec import _materialize_spec_file

    out = tmp_path / "artifact.nc"
    written = _materialize_spec_file("preset:les", out)
    assert written.is_file()
    assert VerticalSpec.from_file(written) == VerticalSpec.les()


# ---------------------------------------------------------------------------
# the level count comes from the files
# ---------------------------------------------------------------------------
def _row(levels: int, *, name: str = "les-row"):
    from woof.hex.drivers import mpas_mesh_binding as binding

    return binding.MeshBinding(
        name=name, n_cells=10, n_edges=30, n_levels=levels,
        n_interfaces=levels + 1, n_soil_levels=4, nominal_dx_m=100.0,
        dt_seconds=5.0, grid_bytes=1, grid_sha256="0" * 64, static_bytes=1,
        static_sha256="0" * 64,
    )


def _init(path: Path, levels: int, soil: int = 4) -> Path:
    netCDF4 = pytest.importorskip("netCDF4")
    with netCDF4.Dataset(str(path), "w") as dataset:
        dataset.createDimension("nCells", 10)
        dataset.createDimension("nVertLevels", levels)
        dataset.createDimension("nVertLevelsP1", levels + 1)
        dataset.createDimension("nSoilLevels", soil)
        variable = dataset.createVariable("theta", "f4", ("nCells", "nVertLevels"))
        variable[:] = 300.0
    return path


def test_the_column_is_read_from_the_init(tmp_path):
    from woof.hex.drivers.mpas_mesh_binding import _inspect_init_column

    init = _init(tmp_path / "init.nc", 80)
    observed = _inspect_init_column(init, _row(80))
    assert observed["nVertLevels"] == 80
    assert observed["nVertLevelsP1"] == 81
    assert observed["source"].startswith("init ")
    # No init: the row's declaration, and the record says so.
    assert "registry row" in _inspect_init_column(None, _row(80))["source"]


def test_an_init_whose_column_disagrees_with_the_row_refuses(tmp_path):
    from woof.hex.drivers.mpas_mesh_binding import (
        MeshBindingMismatch,
        _inspect_init_column,
    )

    init = _init(tmp_path / "init.nc", 55)
    with pytest.raises(MeshBindingMismatch, match="nVertLevels=55"):
        _inspect_init_column(init, _row(80))
    soil = _init(tmp_path / "soil.nc", 80, soil=9)
    with pytest.raises(MeshBindingMismatch, match="nSoilLevels"):
        _inspect_init_column(soil, _row(80))
    with pytest.raises(MeshBindingMismatch, match="does not exist"):
        _inspect_init_column(tmp_path / "missing.nc", _row(80))
    with pytest.raises(MeshBindingMismatch, match="WSM6"):
        _inspect_init_column(None, _row(100))


def test_a_row_that_describes_no_column_refuses():
    from woof.hex.drivers import mpas_mesh_binding as binding

    bad = binding.MeshBinding(
        name="bad", n_cells=10, n_edges=30, n_levels=80, n_interfaces=80,
        n_soil_levels=4, nominal_dx_m=100.0, dt_seconds=5.0, grid_bytes=1,
        grid_sha256="0" * 64, static_bytes=1, static_sha256="0" * 64,
    )
    with pytest.raises(binding.MeshBindingMismatch, match="levels \\+ 1"):
        binding._inspect_init_column(None, bad)


def test_every_registered_row_still_declares_the_native_column():
    from woof.hex.drivers import mpas_mesh_binding as binding

    for row in binding.MESH_BINDINGS.values():
        assert (row.n_levels, row.n_interfaces) == (55, 56), row.name


def test_the_bind_no_longer_asserts_a_module_column():
    """The hard ``module N_LEVELS == registry`` check is gone; the bind
    rebinds the column on proof AND forecast modules, and checks the init."""

    import inspect

    from woof.hex.drivers import mpas_mesh_binding as binding

    source = inspect.getsource(binding.bind_mesh)
    assert "vertical structure \"\n                \"is not rebound" not in source
    assert "proof.N_LEVELS, proof.N_INTERFACES = nz, nzp1" in source
    assert "forecast.N_LEVELS, forecast.N_INTERFACES = nz, nzp1" in source
    assert "_inspect_init_column" in source


# ---------------------------------------------------------------------------
# device admission prices memory per level
# ---------------------------------------------------------------------------
def test_the_per_cell_slope_scales_with_a_deeper_column():
    base = device_admission.model_for_card(device_admission.REFERENCE_CARD)
    deep = device_admission.model_for_card(
        device_admission.REFERENCE_CARD, levels=80
    )
    assert base.level_scale() == 1.0
    assert deep.level_scale() == pytest.approx(80 / 55)
    assert deep.level_bytes_per_cell() == pytest.approx(80 / 55 * base.bytes_per_cell)
    cells = 100_000
    assert deep.required_bytes(cells) > base.required_bytes(cells)
    growth = deep.predict_bytes(cells) - base.predict_bytes(cells)
    assert growth >= (80 / 55 - 1.0) * base.bytes_per_cell * cells
    assert deep.max_cells(24 * 1024**3) < base.max_cells(24 * 1024**3)
    assert deep.measured is False
    record = deep.as_dict()
    assert record["levels"] == 80
    assert "DERIVED" in record["level_scaling"]


def test_the_deep_wsm6_tier_charges_its_wider_local_frame():
    card = device_admission.REFERENCE_CARD
    at64 = device_admission.model_for_card(card, levels=64)
    at80 = device_admission.model_for_card(card, levels=80)
    assert at64.local_store_extra_bytes() == 0
    assert at80.local_store_extra_bytes() == 1_792 * card.resident_threads
    with pytest.raises(ValueError, match="WSM6"):
        device_admission.model_for_card(card, levels=81).predict_bytes(10)


def test_a_shallower_column_keeps_the_measured_slope():
    base = device_admission.model_for_card(device_admission.REFERENCE_CARD)
    shallow = device_admission.model_for_card(
        device_admission.REFERENCE_CARD, levels=40
    )
    assert shallow.level_bytes_per_cell() == base.bytes_per_cell
    assert "level_scaling" not in shallow.as_dict()


def test_the_native_column_floor_is_unchanged():
    assert device_admission.model_at_levels(55) is device_admission.FOOTPRINT_MODEL
    cells = 163_842
    assert device_admission.required_free_bytes(
        cells, device_admission.model_at_levels(55)
    ) == device_admission.native_device_floor_bytes()
    assert device_admission.required_free_bytes(
        cells, device_admission.model_at_levels(80)
    ) > device_admission.native_device_floor_bytes()
    with pytest.raises(ValueError):
        device_admission.model_at_levels(0)


def test_the_door_prices_the_rows_column(tmp_path):
    registry = {
        "les": door.MeshRow("les", 40_000, 5.0, 937.5, levels=80),
    }
    arguments = _door_namespace(tmp_path, mesh="les")
    request = door.resolve_request(arguments, registry=registry)
    assert request.levels == 80
    model = door.resolve_admission_model(request)
    assert model.levels == 80
    assert model.level_scale() == pytest.approx(80 / 55)
    with pytest.raises(door.ForecastDoorRefusal, match="WSM6"):
        door.resolve_request(
            _door_namespace(tmp_path / "deep", mesh="les"),
            registry={"les": door.MeshRow("les", 40_000, 5.0, 937.5, levels=100)},
        )
    native = door.resolve_request(
        _door_namespace(tmp_path / "n", mesh="les"),
        registry={"les": door.MeshRow("les", 40_000, 5.0, 937.5)},
    )
    assert door.resolve_admission_model(native).levels == 55


# ---------------------------------------------------------------------------
# --pbl {ysu,off}
# ---------------------------------------------------------------------------
def test_pbl_ysu_is_the_default_and_decides_nothing_new():
    decision = pbl_admission.pbl_decision(finest_spacing_m=25_000.0)
    assert decision["scheme"] == "ysu"
    assert decision["config_pbl_scheme"] == "bl_ysu"
    assert decision["bl_pbl_physics"] == 1
    assert decision["warnings"] == []


def test_pbl_off_is_admitted_on_a_fine_mesh_and_warns_without_a_closure():
    decision = pbl_admission.pbl_decision(
        requested="off", finest_spacing_m=100.0, convection_scheme="off"
    )
    assert decision["source"] == "resolution"
    assert decision["bl_pbl_physics"] == 0
    assert decision["anchor_evidence"] == "unanchored-configuration"
    assert any("NO TURBULENCE CLOSURE" in text for text in decision["warnings"])


def test_pbl_off_is_admitted_with_an_les_closure_at_any_spacing():
    decision = pbl_admission.pbl_decision(
        requested="off", finest_spacing_m=3_000.0, les_model="3d_smagorinsky",
        convection_scheme="off",
    )
    assert decision["source"] == "les-closure"
    assert decision["warnings"] == []


def test_pbl_off_in_the_gray_zone_refuses_unless_overridden():
    with pytest.raises(pbl_admission.PblAdmissionError, match="--les-model"):
        pbl_admission.pbl_decision(requested="off", finest_spacing_m=3_000.0)
    with pytest.raises(pbl_admission.PblAdmissionError, match="unknown spacing"):
        pbl_admission.pbl_decision(requested="off", finest_spacing_m=None)
    overridden = pbl_admission.pbl_decision(
        requested="off", finest_spacing_m=3_000.0, allow_gray_zone=True
    )
    assert overridden["source"] == "override"
    assert any("GRAY ZONE" in text for text in overridden["warnings"])


def test_pbl_off_with_grell_freitas_refuses_whatever_the_spacing():
    with pytest.raises(pbl_admission.PblAdmissionError, match="KPBL"):
        pbl_admission.pbl_decision(
            requested="off", finest_spacing_m=100.0,
            les_model="prognostic_tke", convection_scheme="cu_grell_freitas",
        )
    with pytest.raises(pbl_admission.PblAdmissionError, match="not one of"):
        pbl_admission.pbl_decision(requested="mynn", finest_spacing_m=100.0)


def test_the_config_admits_pbl_off_with_convection_off_only():
    from woof.hex.config_v841 import V841MpasColumnPhysicsGwdoConfig

    clocks = {
        "config_dt": 5.0, "config_bldt_seconds": 5.0,
        "config_cudt_seconds": None, "config_convection_scheme": "off",
    }
    V841MpasColumnPhysicsGwdoConfig(**clocks).validate()
    V841MpasColumnPhysicsGwdoConfig(**clocks, config_pbl_scheme="off").validate()
    with pytest.raises(ConfigurationRefusal, match="config_pbl_scheme"):
        V841MpasColumnPhysicsGwdoConfig(
            **clocks, config_pbl_scheme="bl_mynn"
        ).validate()
    with pytest.raises(ConfigurationRefusal, match="KPBL"):
        V841MpasColumnPhysicsGwdoConfig(
            config_dt=5.0, config_bldt_seconds=5.0, config_cudt_seconds=5.0,
            config_convection_scheme="cu_grell_freitas", config_pbl_scheme="off",
        ).validate()


def test_the_forecast_config_carries_the_pbl_slot():
    from woof.hex.drivers import run_cuda_v841_forecast as forecast

    proven = forecast.build_forecast_config(dt_seconds=5.0, convection_scheme="off")
    assert proven.config_pbl_scheme == "bl_ysu"
    off = forecast.build_forecast_config(
        dt_seconds=5.0, convection_scheme="off", pbl_scheme="off"
    )
    off.validate()
    assert off.config_pbl_scheme == "off"


def _constructor_values(**extra):
    ncol = 3
    values = {
        "n_levels": 4, "n_columns": ncol, "dt": 5.0,
        "radiation_seconds": 600.0, "surface_pbl_seconds": 5.0,
        "cumulus_seconds": None, "cumulus_scheme": None,
        "microphysics_scheme": "wsm6", "start_time": datetime(2026, 1, 1),
        "p_top_pa": 1000.0, "dx_m": 100.0, "gf_ishallow": 0,
        "wsm6_hail_opt": 0, "xice_threshold": 0.02,
        "z_interface_nominal_m": np.array([0.0, 20.0, 45.0, 80.0, 120.0]),
        "ivgtyp": np.full(ncol, 10, np.int32),
        "isltyp": np.full(ncol, 6, np.int32),
        "soil_temperature": np.full((4, ncol), 285.0, np.float32),
        "soil_moisture": np.full((4, ncol), 0.3, np.float32),
        "dx_column_m": np.full(ncol, 100.0, np.float32),
    }
    for name in ("latitude_deg", "longitude_deg", "terrain_height_m",
                 "landmask", "vegfra", "tsk", "tmn", "xice", "snow",
                 "snow_depth"):
        values[name] = np.ones(ncol, np.float32)
    values["xland"] = np.ones(ncol, np.float32)
    values.update(extra)
    return values


def test_the_sealed_constructor_keeps_a_ysu_identity_byte_identical():
    from woof.hex.cuda_arwen_physics_v841 import SealedArwenConstructorV841

    absent = SealedArwenConstructorV841.from_mapping(_constructor_values())
    explicit = SealedArwenConstructorV841.from_mapping(
        _constructor_values(pbl_scheme="ysu")
    )
    off = SealedArwenConstructorV841.from_mapping(
        _constructor_values(pbl_scheme="off")
    )
    assert absent.identity_sha256 == explicit.identity_sha256
    assert "pbl_scheme" not in absent.arwen_kwargs()
    assert "pbl_scheme" not in absent.receipt()
    assert off.identity_sha256 != absent.identity_sha256
    assert off.arwen_kwargs()["pbl_scheme"] == "off"
    assert off.receipt()["pbl_scheme"] == "off"


def test_the_sealed_constructor_refuses_a_bad_pbl_slot():
    from woof.hex.cuda_arwen_physics_v841 import SealedArwenConstructorV841

    with pytest.raises(ValueError, match="pbl_scheme must be one of"):
        SealedArwenConstructorV841.from_mapping(_constructor_values(pbl_scheme="mynn"))
    with pytest.raises(ValueError, match="KPBL"):
        SealedArwenConstructorV841.from_mapping(
            _constructor_values(
                pbl_scheme="off", cumulus_scheme="gf", cumulus_seconds=5.0,
            )
        )


def test_the_engine_seam_refuses_a_bad_pbl_slot_before_the_device():
    from woof.core import mpas_column_batch as mcb

    kwargs = dict(
        n_levels=4, n_columns=3, dt=5.0, radiation_seconds=600.0,
        surface_pbl_seconds=5.0, start_time=datetime(2026, 1, 1),
        latitude_deg=np.zeros(3), longitude_deg=np.zeros(3),
        terrain_height_m=np.zeros(3),
        z_interface_nominal_m=np.array([0.0, 20.0, 45.0, 80.0, 120.0]),
        p_top_pa=1000.0, dx_m=100.0,
    )
    with pytest.raises(ValueError, match="pbl_scheme must be one of"):
        mcb.run_mpas_column_batch(pbl_scheme="mynn", **kwargs)
    with pytest.raises(ValueError, match="KPBL"):
        mcb.run_mpas_column_batch(
            pbl_scheme="off", cumulus_scheme="gf", cumulus_seconds=5.0, **kwargs
        )


def test_the_preset_suite_names_the_slot_that_ran():
    from woof.hex import forecast_preset

    row = forecast_preset.resolve_preset()
    assert forecast_preset.effective_suite(row) == dict(row.suite)
    off = forecast_preset.effective_suite(row, "off")
    assert off["boundary_layer"] == "off" and off["bl_pbl_physics"] == 0
    with pytest.raises(ConfigurationRefusal):
        forecast_preset.effective_suite(row, "mynn")


def _door_namespace(tmp_path: Path, **overrides) -> argparse.Namespace:
    tmp_path.mkdir(parents=True, exist_ok=True)
    grid = tmp_path / "g.nc"
    static = tmp_path / "s.nc"
    init = tmp_path / "i.nc"
    checkout = tmp_path / "woof"
    for path in (grid, static, init):
        path.write_bytes(b"not really netcdf")
    checkout.mkdir(exist_ok=True)
    parser = argparse.ArgumentParser()
    door.add_forecast_arguments(parser)
    arguments = parser.parse_args([
        "--mesh", overrides.pop("mesh", "fine"), "--grid", str(grid),
        "--static", str(static), "--init", str(init), "--init-source", "test",
        "--hours", "0.1", "--history-every-minutes", "6",
        "--out", str(tmp_path / "out"), "--gpuwm-checkout", str(checkout),
    ])
    for key, value in overrides.items():
        setattr(arguments, key, value)
    return arguments


@pytest.fixture(autouse=True)
def _door_seams(monkeypatch):
    monkeypatch.setattr(
        door, "read_card_profile", lambda: device_admission.REFERENCE_CARD
    )
    monkeypatch.setattr(door, "seam_source_problem", lambda checkout: None)
    monkeypatch.setattr(door, "backend_pin_problem", lambda row, checkout: None)


def _registry():
    return {
        "fine": door.MeshRow("fine", 40_000, 5.0, 937.5),
        "coarse": door.MeshRow("coarse", 40_962, 20.0, 2_000.0),
    }


def test_without_the_flag_the_driver_argv_is_unchanged(tmp_path):
    request = door.resolve_request(_door_namespace(tmp_path), registry=_registry())
    assert request.pbl == "ysu"
    argv = door.build_driver_argv(request)
    assert "--pbl" not in argv
    assert "--allow-pbl-off-gray-zone" not in argv


def test_pbl_off_on_a_fine_mesh_reaches_the_driver(tmp_path):
    request = door.resolve_request(
        _door_namespace(tmp_path, pbl="off"), registry=_registry()
    )
    assert request.pbl_decision["source"] == "resolution"
    argv = door.build_driver_argv(request)
    assert argv[argv.index("--pbl") + 1] == "off"


def test_pbl_off_on_a_coarse_row_is_deferred_to_the_bind(tmp_path):
    """The door sees the nominal spacing only; the bind measures the finest
    edge.  The spacing refusal is therefore the bind's, not a second answer."""

    request = door.resolve_request(
        _door_namespace(tmp_path, mesh="coarse", pbl="off"),
        registry=_registry(),
    )
    assert request.pbl_decision["source"] == "deferred-to-bind"
    assert "--les-model" in request.pbl_decision["note"]


def test_pbl_off_with_grell_freitas_is_refused_at_the_door(tmp_path):
    with pytest.raises(door.ForecastDoorRefusal, match="KPBL"):
        door.resolve_request(
            _door_namespace(tmp_path, mesh="coarse", pbl="off", convection="gf"),
            registry={"coarse": door.MeshRow("coarse", 40_962, 20.0, 2_000.0)},
        )


def test_the_bind_refuses_pbl_off_in_the_gray_zone():
    import inspect

    from woof.hex.drivers import mpas_mesh_binding as binding

    source = inspect.getsource(binding.bind_mesh)
    assert "pbl_admission.pbl_decision(" in source
    assert 'finest_spacing_m=float(convection_decision["finest_spacing_m"])' in source
    assert "raise MeshBindingError" in source


def test_pbl_off_override_reaches_the_driver(tmp_path):
    request = door.resolve_request(
        _door_namespace(
            tmp_path, mesh="coarse", pbl="off", convection="off",
            allow_pbl_off_gray_zone=True,
        ),
        registry=_registry(),
    )
    assert request.pbl_decision["source"] == "override"
    assert "--allow-pbl-off-gray-zone" in door.build_driver_argv(request)


def test_pbl_off_with_an_les_closure_is_admitted_at_the_door(tmp_path):
    request = door.resolve_request(
        _door_namespace(
            tmp_path, mesh="coarse", pbl="off", convection="off",
            les_model="3d_smagorinsky",
        ),
        registry=_registry(),
    )
    assert request.pbl_decision["source"] == "les-closure"


def test_the_forecast_parser_offers_pbl():
    parser = argparse.ArgumentParser()
    door.add_forecast_arguments(parser)
    assert parser.parse_args([]).pbl == "ysu"
    assert parser.parse_args(["--pbl", "off"]).pbl == "off"
    with pytest.raises(SystemExit):
        parser.parse_args(["--pbl", "mynn"])
    from woof.hex.drivers import run_cuda_v841_forecast as forecast

    parsed = forecast.parse_args(["--init", "i.nc", "--init-source", "x",
                                  "--hours", "1", "--history-every-minutes", "30",
                                  "--preflight-only", "--pbl", "off"])
    assert parsed.pbl == "off"


# ---------------------------------------------------------------------------
# GWDO follows the column
# ---------------------------------------------------------------------------
def test_the_gwdo_kernel_source_is_unchanged_at_55_and_sized_otherwise():
    from woof.hex import cuda_gwdo_v841 as gwdo

    assert gwdo.gwdo_kernel_source(55) is gwdo._CUDA_SOURCE
    deep = gwdo.gwdo_kernel_source(100)
    assert "#define GWDO_NLEV 100\n" in deep
    assert deep.replace("#define GWDO_NLEV 100\n", "#define GWDO_NLEV 55\n") == gwdo._CUDA_SOURCE
    variant = gwdo.gwdo_kernel_variant(100)
    assert variant["anchored_kernel_sha256"] == gwdo.CUDA_GWDO_V841_KERNEL_SHA256
    assert variant["kernel_sha256"] != gwdo.CUDA_GWDO_V841_KERNEL_SHA256
    with pytest.raises(ValueError, match="outside"):
        gwdo.gwdo_kernel_source(gwdo.GWDO_MAX_LEVELS + 1)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
class _TinyMesh:
    """A 7-cell hexagonal patch: enough topology for the vertical builder."""

    def __init__(self):
        n_cells = 7
        self.n_cells = n_cells
        edges = []
        cells_on_cell = np.zeros((n_cells, 6), dtype=np.int64)
        edges_on_cell = np.zeros((n_cells, 6), dtype=np.int64)
        n_edges_on_cell = np.full(n_cells, 6, dtype=np.int64)
        lookup = {}
        neighbours = {0: [1, 2, 3, 4, 5, 6]}
        for ring in range(1, 7):
            neighbours[ring] = [0, 1 + (ring % 6), 1 + ((ring - 2) % 6), 0, 0, 0]
        for cell in range(n_cells):
            for slot, other in enumerate(neighbours[cell]):
                key = tuple(sorted((cell, other)))
                if key not in lookup:
                    lookup[key] = len(edges)
                    edges.append(key)
                cells_on_cell[cell, slot] = other
                edges_on_cell[cell, slot] = lookup[key]
        n_edges = len(edges)
        self.arrays = {
            "areaCell": np.full(n_cells, 1.0e4),
            "nEdgesOnCell": n_edges_on_cell,
            "cellsOnCell": cells_on_cell,
            "edgesOnCell": edges_on_cell,
            "dvEdge": np.full(n_edges, 60.0),
            "dcEdge": np.full(n_edges, 100.0),
            "cellsOnEdge": np.asarray(edges, dtype=np.int64),
        }
        self.dimensions = {"nCells": n_cells, "nEdges": n_edges}

    def __getattr__(self, name):
        arrays = self.__dict__.get("arrays", {})
        if name in arrays:
            return arrays[name]
        raise AttributeError(name)


def _tiny_mesh():
    return _TinyMesh()


def test_the_point_planner_prices_the_declared_column():
    from woof.hex.mesh_point import device_verdict, resolve_card

    card = resolve_card("32gb")
    native = device_verdict(40_000, card)
    deep = device_verdict(40_000, card, levels=80)
    assert native["levels"] == 55 and deep["levels"] == 80
    assert deep["required_free_mib"] > native["required_free_mib"]
    assert deep["cells_that_fit"] < native["cells_that_fit"]
