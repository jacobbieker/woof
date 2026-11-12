"""CLM lake on the hex generic seam IS WRF's lake kernel on the same inputs.

The hex forecast seals the lake's per-cell inputs (woof.hex.surface_inputs)
into the suite constructor, which hands them to the engine's generic column
seam as initialize_physics keywords.  This test builds the seam through that
sealed constructor, steps it, and replays every lake call through WRF's own
lake column kernel (woof.core.lake: lakeini, then the lake step) on a WRF-shaped
(2, 4) grid holding the same cells: the initial lake state and every step's
lake state and surface outputs agree bit for bit.
"""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

import mpas_column_physics_cases as cases  # noqa: E402
from woof.core import mpas_column_suite as suite  # noqa: E402

NZ, NCOL, DT = cases.NZ, cases.NCOL, cases.DT
GRID = (2, 4)
LAKES = np.array([1, 0, 0, 1, 0, 1, 0, 0], dtype=bool)
DEPTH = np.array([12.5, 10.0, 10.0, 3.25, 10.0, 41.0, 10.0, 10.0], np.float32)


def _physics():
    return dict(suite.MPAS_DEFAULT_PHYSICS, sf_lake_physics=1)


def _mapping(physics, surface_inputs):
    plan = suite.column_physics_plan(
        physics, n_levels=NZ, dt=DT, radiation_seconds=120.0,
        surface_pbl_seconds=DT)
    surface = cases.surface(plan)
    surface["landmask"][LAKES] = 0.0
    surface["xland"][LAKES] = 2.0
    surface["ivgtyp"][LAKES] = 17
    surface["isltyp"][LAKES] = 14
    surface["tsk"][LAKES] = 284.0
    rng = np.random.default_rng(2)
    values = {
        "n_levels": NZ, "n_columns": NCOL, "dt": DT,
        "radiation_seconds": 120.0, "surface_pbl_seconds": DT,
        "cumulus_seconds": None, "cumulus_scheme": None,
        "microphysics_scheme": "wsm6", "start_time": cases.START,
        "p_top_pa": cases.P_TOP, "dx_m": 12000.0, "gf_ishallow": 0,
        "wsm6_hail_opt": 0, "fractional_seaice": None,
        "orographic_statistics": None, "xice_threshold": 0.5,
        "radiation_scheme": "rrtmg_legacy", "surface_layer_scheme": "mm5_revised",
        "land_surface_scheme": "noahmp", "pbl_scheme": "ysu", "drag_scheme": "off",
        "latitude_deg": np.full(NCOL, 44.0, np.float32),
        "longitude_deg": np.full(NCOL, -84.0, np.float32),
        "terrain_height_m": rng.uniform(170.0, 200.0, NCOL).astype(np.float32),
        "z_interface_nominal_m": np.linspace(0.0, cases.ZTOP, NZ + 1),
        "dx_column_m": np.full(NCOL, 12000.0, np.float32),
        "physics": physics,
        **{name: np.asarray(value) for name, value in surface.items()},
    }
    if surface_inputs is not None:
        values["surface_inputs"] = surface_inputs
    return values


def _inputs():
    return {"lakemask": LAKES.astype(np.float32), "lake_mask_flag": 1,
            "lake_depth": DEPTH.copy(), "lake_depth_flag": 1}


def _seal(values):
    from woof.hex.cuda_suite_physics_v841 import SealedSuiteConstructorV841

    return SealedSuiteConstructorV841.from_mapping(values)


def test_the_sealed_constructor_refuses_a_lake_run_without_its_inputs():
    with pytest.raises(ValueError, match="missing .*lakemask"):
        _seal(_mapping(_physics(), None))
    without = _seal(_mapping(dict(suite.MPAS_DEFAULT_PHYSICS), None))
    assert "surface_inputs" not in without.receipt()
    with pytest.raises(ValueError, match="extra .*lake_depth"):
        _seal(_mapping(dict(suite.MPAS_DEFAULT_PHYSICS), _inputs()))
    sealed = _seal(_mapping(_physics(), _inputs()))
    assert sealed.receipt()["surface_inputs"] == [
        "lake_depth", "lake_depth_flag", "lake_mask_flag", "lakemask"]
    other = _inputs()
    other["lake_depth"][0] = 13.0
    assert _seal(_mapping(_physics(), other)).identity_sha256 != sealed.identity_sha256


def _host(value):
    return cp.asnumpy(cp.asarray(value)).copy()


def test_hex_lake_columns_are_wrf_lake_kernel_columns():
    from woof.core.lake import initialize_lake
    from woof.core.lake_schema import LAKE_OUTPUT_LAYOUT

    sealed = _seal(_mapping(_physics(), _inputs()))
    kwargs = sealed.arwen_kwargs()
    np.testing.assert_array_equal(kwargs["surface"]["lake_depth"], DEPTH)
    seam = suite.run_mpas_column_physics(**kwargs)
    driver = seam._driver
    lake = driver.lake
    assert lake is not None and lake.column_count == int(LAKES.sum())
    fields = driver.fields
    initial = {name: _host(fields[name]) for name in ("lake_columns", "lake_static")}

    calls = []
    step = lake.step

    def capture(fields, atmosphere, *, dt, precipitation):
        before = {name: _host(fields[name]) for name, _ in LAKE_OUTPUT_LAYOUT}
        forcing = {name: _host(fields[name]) for name in ("glw", "emiss", "swdown")}
        forcing["albedo_in"] = _host(fields["albedo"])
        atm = {name: _host(atmosphere[name][:2]) for name in
               ("temperature", "p_interface", "dz", "qv", "u", "v")}
        rain = _host(precipitation)
        step(fields, atmosphere, dt=dt, precipitation=precipitation)
        after = {name: _host(fields[name]) for name, _ in LAKE_OUTPUT_LAYOUT}
        after["lake_columns"] = _host(fields["lake_columns"])
        calls.append((dt, before, forcing, atm, rain, after))

    lake.step = capture
    frames = cases.run(cp, seam, 4)
    assert not cases._finite(frames)
    assert calls, "the lake was never called"

    # WRF: lakeini on a (2, 4) grid of the same cells, from the same inputs.
    def grid(values):
        values = np.asarray(values)
        return cp.asarray(values.reshape(values.shape[:-2] + GRID)
                          if values.ndim >= 2 and values.shape[-2] == 1
                          else values.reshape(values.shape[:-1] + GRID))

    host = kwargs["surface"]
    wrf = {"tsk": grid(host["tsk"]), "isltyp": grid(host["isltyp"]),
           "snow": grid(host["snow"]), "xice": grid(host["xice"]),
           "lakemask": grid(host["lakemask"])}
    replay = initialize_lake(
        wrf, latitude=grid(kwargs["latitude_deg"]),
        lake_depth=grid(host["lake_depth"]), lake_depth_flag=1,
        use_lakedepth=1, lakedepth_default=50.0, iswater=17, xice_threshold=0.5)
    for name, value in initial.items():
        np.testing.assert_array_equal(
            _host(wrf[name]).view(np.uint32),
            value.reshape(value.shape[:-2] + GRID).view(np.uint32), err_msg=name)

    for index, (dt, before, forcing, atm, rain, after) in enumerate(calls):
        for name, value in before.items():
            wrf[name] = grid(value)
        wrf["glw"], wrf["emiss"], wrf["swdown"] = (
            grid(forcing["glw"]), grid(forcing["emiss"]), grid(forcing["swdown"]))
        wrf["albedo"] = grid(forcing["albedo_in"])
        replay.step(wrf, {name: grid(value) for name, value in atm.items()},
                    dt=dt, precipitation=grid(rain))
        for name, value in after.items():
            np.testing.assert_array_equal(
                _host(wrf[name]).view(np.uint32),
                value.reshape(value.shape[:-2] + GRID).view(np.uint32),
                err_msg=f"lake call {index}: {name}")


def _lake_seam():
    return suite.run_mpas_column_physics(
        **_seal(_mapping(_physics(), _inputs())).arwen_kwargs())


def test_the_lake_seam_restarts_bit_identically():
    from woof.core.lake_schema import LAKE_OUTPUT_LAYOUT

    names = ("lake_columns", *(name for name, _ in LAKE_OUTPUT_LAYOUT))
    reference_seam = _lake_seam()
    reference = []
    fields, species = cases.profile(cp, reference_seam.plan)
    theta = fields["theta"].copy()
    for _ in range(4):
        reference_seam.run_phase1(
            dt=DT, u=fields["u"], v=fields["v"], theta=theta,
            pressure=fields["pressure"],
            pressure_interface=fields["pressure_interface"],
            z_interface=fields["z_interface"], w=fields["w"],
            rho_dry=fields["rho_dry"], species=species)
        reference_seam.run_phase2(
            theta=theta, pressure=fields["pressure"],
            z_interface=fields["z_interface"], w=fields["w"],
            rho_dry=fields["rho_dry"], species=species)
        reference.append({
            "theta": _host(theta),
            "species": {name: _host(value) for name, value in species.items()},
            **{name: _host(reference_seam._driver.fields[name]) for name in names}})
    first = _lake_seam()
    cases.run(cp, first, 2)
    payload = first.export_state()
    assert any("lake_columns" in key for key in payload["arrays"])
    second = _lake_seam()
    second.restore_state(payload)
    fields, _ = cases.profile(cp, second.plan)
    theta = cp.asarray(reference[1]["theta"])
    species = {name: cp.asarray(reference[1]["species"][name])
               for name in second.plan.species}
    for step in (2, 3):
        second.run_phase1(
            dt=DT, u=fields["u"], v=fields["v"], theta=theta,
            pressure=fields["pressure"],
            pressure_interface=fields["pressure_interface"],
            z_interface=fields["z_interface"], w=fields["w"],
            rho_dry=fields["rho_dry"], species=species)
        second.run_phase2(
            theta=theta, pressure=fields["pressure"],
            z_interface=fields["z_interface"], w=fields["w"],
            rho_dry=fields["rho_dry"], species=species)
        np.testing.assert_array_equal(_host(theta).view(np.uint32),
                                      reference[step]["theta"].view(np.uint32))
        for name in names:
            np.testing.assert_array_equal(
                _host(second._driver.fields[name]).view(np.uint32),
                reference[step][name].view(np.uint32), err_msg=f"step {step} {name}")


URBAN = np.array([1, 1, 0, 1, 0, 0, 1, 0], dtype=bool)
FRC = np.array([0.81, 0.405, 0.0, 0.9, 0.0, 0.0, 0.27, 0.0], np.float32)


def _urban_mapping():
    physics = dict(suite.MPAS_DEFAULT_PHYSICS, sf_urban_physics=1)
    values = _mapping(physics, {"frc_urb2d": FRC.copy()})
    for name in ("landmask", "xland", "isltyp", "tsk"):
        values[name] = np.asarray(cases.surface(suite.column_physics_plan(
            physics))[name]).copy()
    values["ivgtyp"] = np.where(URBAN, 13, 10).astype(np.int32)
    return values


def test_the_urban_seam_refuses_without_its_fraction_and_carries_it():
    values = _urban_mapping()
    sealed = _seal(values)
    assert sealed.receipt()["surface_inputs"] == ["frc_urb2d"]
    values.pop("surface_inputs")
    with pytest.raises(ValueError, match="missing .*frc_urb2d"):
        _seal(values)


def test_hex_urban_columns_are_wrf_urban_twin_columns():
    """Every SLUCM call on the hex seam equals WRF's urban model replayed on a
    twin built by the same urban_param_init/urban_var_init from the same
    staged FRC_URB2D (tools/hex_urban_twin.py, run in-process here)."""
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "tools" / "hex_urban_twin.py"
    spec = importlib.util.spec_from_file_location("hex_urban_twin", path)
    twin = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(twin)
    summary: dict = {}
    restore = twin._install(summary)
    try:
        seam = suite.run_mpas_column_physics(**_seal(_urban_mapping()).arwen_kwargs())
        frc = _host(seam._driver.urban.fields["frc_urb2d"]).reshape(-1)
        # urban_var_init keeps the staged fraction on the urban cells.
        np.testing.assert_array_equal(frc[URBAN], FRC[URBAN])
        frames = cases.run(cp, seam, 4)
    finally:
        restore()
    assert not cases._finite(frames)
    assert summary["calls_after_lsm"] >= 1
    assert not any(summary["initial_words_differing"].values()), summary
    assert not any(summary["step_words_differing"].values()), {
        key: value for key, value in summary["step_words_differing"].items() if value}


def test_the_urban_seam_restarts_bit_identically():
    def make():
        return suite.run_mpas_column_physics(**_seal(_urban_mapping()).arwen_kwargs())

    def step(seam, theta, species, fields):
        seam.run_phase1(
            dt=DT, u=fields["u"], v=fields["v"], theta=theta,
            pressure=fields["pressure"],
            pressure_interface=fields["pressure_interface"],
            z_interface=fields["z_interface"], w=fields["w"],
            rho_dry=fields["rho_dry"], species=species)
        seam.run_phase2(
            theta=theta, pressure=fields["pressure"],
            z_interface=fields["z_interface"], w=fields["w"],
            rho_dry=fields["rho_dry"], species=species)

    reference_seam = make()
    names = tuple(reference_seam._driver.urban.names)
    fields, species = cases.profile(cp, reference_seam.plan)
    theta = fields["theta"].copy()
    reference = []
    for _ in range(4):
        step(reference_seam, theta, species, fields)
        reference.append({
            "theta": _host(theta),
            "species": {name: _host(value) for name, value in species.items()},
            **{name: _host(reference_seam._driver.fields[name]) for name in names}})
    first = make()
    cases.run(cp, first, 2)
    payload = first.export_state()
    second = make()
    second.restore_state(payload)
    fields, _ = cases.profile(cp, second.plan)
    theta = cp.asarray(reference[1]["theta"])
    species = {name: cp.asarray(reference[1]["species"][name])
               for name in second.plan.species}
    for index in (2, 3):
        step(second, theta, species, fields)
        np.testing.assert_array_equal(_host(theta).view(np.uint32),
                                      reference[index]["theta"].view(np.uint32))
        for name in names:
            a, b = _host(second._driver.fields[name]), reference[index][name]
            np.testing.assert_array_equal(a, b, err_msg=f"step {index} {name}")


def _mosaic_mapping(landusef=True):
    physics = dict(suite.MPAS_DEFAULT_PHYSICS, sf_surface_physics=2,
                   sf_surface_mosaic=1)
    stack = np.zeros((21, NCOL), np.float32)
    # Mixed cells: grass with cropland and shrubs, two of them part urban.
    stack[9] = [0.6, 0.5, 1.0, 0.4, 0.7, 0.3, 0.8, 0.5]
    stack[11] = [0.3, 0.2, 0.0, 0.4, 0.3, 0.5, 0.0, 0.25]
    stack[6] = 1.0 - stack[9] - stack[11]
    stack[12, 1], stack[6, 1] = 0.2, stack[6, 1] - 0.2
    values = _mapping(physics, {"landusef": stack} if landusef else None)
    values["land_surface_scheme"] = "noah"
    return values


def test_the_mosaic_seam_refuses_without_landusef():
    with pytest.raises(ValueError, match="missing .*landusef"):
        _seal(_mosaic_mapping(landusef=False))


def _twin_module(name):
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_hex_mosaic_tiles_are_wrf_mosaic_twin_tiles():
    summary: dict = {}
    restore = _twin_module("hex_mosaic_twin")._install(summary)
    try:
        seam = suite.run_mpas_column_physics(**_seal(_mosaic_mapping()).arwen_kwargs())
        assert seam._driver.noah_mosaic is not None
        frames = cases.run(cp, seam, 4)
    finally:
        restore()
    assert not cases._finite(frames)
    assert summary["calls"] >= 1 and summary["tile_arrays"] > 0
    assert not any(summary["initial_words_differing"].values()), summary
    assert not any(summary["step_words_differing"].values()), {
        key: value for key, value in summary["step_words_differing"].items() if value}


def test_the_mosaic_seam_restarts_bit_identically():
    def make():
        return suite.run_mpas_column_physics(**_seal(_mosaic_mapping()).arwen_kwargs())

    def step(seam, theta, species, fields):
        seam.run_phase1(
            dt=DT, u=fields["u"], v=fields["v"], theta=theta,
            pressure=fields["pressure"],
            pressure_interface=fields["pressure_interface"],
            z_interface=fields["z_interface"], w=fields["w"],
            rho_dry=fields["rho_dry"], species=species)
        seam.run_phase2(
            theta=theta, pressure=fields["pressure"],
            z_interface=fields["z_interface"], w=fields["w"],
            rho_dry=fields["rho_dry"], species=species)

    reference_seam = make()
    names = tuple(key for key in reference_seam._driver.fields if key.endswith("_mosaic"))
    assert names
    fields, species = cases.profile(cp, reference_seam.plan)
    theta = fields["theta"].copy()
    reference = []
    for _ in range(4):
        step(reference_seam, theta, species, fields)
        reference.append({
            "theta": _host(theta),
            "species": {name: _host(value) for name, value in species.items()},
            **{name: _host(reference_seam._driver.fields[name]) for name in names}})
    first = make()
    cases.run(cp, first, 2)
    payload = first.export_state()
    second = make()
    second.restore_state(payload)
    fields, _ = cases.profile(cp, second.plan)
    theta = cp.asarray(reference[1]["theta"])
    species = {name: cp.asarray(reference[1]["species"][name])
               for name in second.plan.species}
    for index in (2, 3):
        step(second, theta, species, fields)
        np.testing.assert_array_equal(_host(theta).view(np.uint32),
                                      reference[index]["theta"].view(np.uint32))
        for name in names:
            np.testing.assert_array_equal(
                _host(second._driver.fields[name]), reference[index][name],
                err_msg=f"step {index} {name}")


def test_the_mosaic_seam_reads_the_drivers_resolved_landmask():
    """An omitted landmask is initialize_physics' default (land); the tile
    door reads the driver's resolved mask, so the seam builds either way."""
    kwargs = _seal(_mosaic_mapping()).arwen_kwargs()
    kwargs["surface"] = {key: value for key, value in kwargs["surface"].items()
                         if key != "landmask"}
    seam = suite.run_mpas_column_physics(**kwargs)
    assert seam._driver.noah_mosaic is not None
