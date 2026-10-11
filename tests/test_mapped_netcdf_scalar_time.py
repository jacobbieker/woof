"""Scalar-time NetCDF, speed/direction winds and declared screen height.

The Met Office global deterministic NetCDF (Iris-written) is the source
these features exist for: one variable per file, ``time``,
``forecast_reference_time`` and ``forecast_period`` as 0-d coordinates,
winds published as speed and meteorological direction, and the screen
fields at 1.5 m.  The fixtures below reproduce that layout on a small
grid, and every decode here runs through BOTH mapped engines and is held
to byte equality, because a feature only one engine reads is a parity
break waiting for the default to flip.
"""

from __future__ import annotations

from datetime import datetime, timedelta
import json
from pathlib import Path

import netCDF4
import numpy as np
import pytest

from woof import mapped_engine_bridge as engine_bridge
from woof.core import portable_math as pm
from woof.mapped_source import _decode_mapped_source_python, load_mapping

from test_mapped_source import POLICY, SURFACE, THREE_D


EPOCH_UNITS = "seconds since 1970-01-01 00:00:00"
CYCLE = datetime(2026, 10, 11, 0, 0)
LEVELS_PA = [100000.0, 85000.0, 70000.0]
#: The humidity file carries fewer levels than the rest, as the Met Office
#: relative humidity stops at 10 hPa while temperature reaches 0.4 hPa.
RH_LEVELS_PA = [100000.0, 85000.0, 70000.0, 50000.0]
ALL_LEVELS_PA = [100000.0, 85000.0, 70000.0, 50000.0, 25000.0]
NY, NX = 5, 6


def _seconds(instant: datetime) -> int:
    return int((instant - datetime(1970, 1, 1)).total_seconds())


def _scalar(dataset, name, dtype, value, **attributes):
    variable = dataset.createVariable(name, dtype)
    for key, item in attributes.items():
        variable.setncattr(key, item)
    variable[...] = value
    return variable


def _write_mo_file(
    path: Path,
    variables: dict[str, tuple[str, str, float | np.ndarray]],
    *,
    lead_hours: int = 0,
    levels: list[float] | None = None,
    height_m: float | None = None,
    period_seconds: int | None = None,
    reference: datetime = CYCLE,
    time_override: datetime | None = None,
) -> Path:
    """One Iris-shaped file: scalar time triplet, lat/lon axes, CF names.

    ``variables`` maps a variable name to (standard_name, units, value);
    a value is broadcast over (pressure,) latitude, longitude.
    """

    valid = reference + timedelta(hours=lead_hours)
    with netCDF4.Dataset(path, "w") as dataset:
        dataset.Conventions = "CF-1.7, UKMO-1.0"
        if levels is not None:
            dataset.createDimension("pressure", len(levels))
        dataset.createDimension("latitude", NY)
        dataset.createDimension("longitude", NX)
        crs = dataset.createVariable("latitude_longitude", "i4")
        crs.grid_mapping_name = "latitude_longitude"
        crs.longitude_of_prime_meridian = 0.0
        crs.earth_radius = 6371229.0
        if levels is not None:
            pressure = dataset.createVariable("pressure", "f4", ("pressure",))
            pressure.axis = "Z"
            pressure.units = "Pa"
            pressure.long_name = "pressure"
            pressure.positive = "down"
            pressure[:] = levels
        latitude = dataset.createVariable("latitude", "f4", ("latitude",))
        latitude.axis = "Y"
        latitude.units = "degrees_north"
        latitude.standard_name = "latitude"
        latitude[:] = np.linspace(50.0, 54.0, NY)
        longitude = dataset.createVariable("longitude", "f4", ("longitude",))
        longitude.axis = "X"
        longitude.units = "degrees_east"
        longitude.standard_name = "longitude"
        longitude[:] = np.linspace(-30.0, -25.0, NX)
        period = (
            int((valid - reference).total_seconds())
            if period_seconds is None else period_seconds
        )
        _scalar(dataset, "forecast_period", "i4", period,
                units="seconds", standard_name="forecast_period")
        _scalar(dataset, "forecast_reference_time", "i8", _seconds(reference),
                units=EPOCH_UNITS, standard_name="forecast_reference_time",
                calendar="standard")
        _scalar(dataset, "time", "i8", _seconds(time_override or valid),
                units=EPOCH_UNITS, standard_name="time", calendar="standard")
        coordinates = "forecast_period forecast_reference_time time"
        if height_m is not None:
            _scalar(dataset, "height", "f4", height_m,
                    units="m", standard_name="height", positive="up")
            coordinates = "forecast_period forecast_reference_time height time"
        dimensions = (("pressure",) if levels is not None else ()) + (
            "latitude", "longitude")
        for name, (standard_name, units, value) in variables.items():
            variable = dataset.createVariable(
                name, "f4", dimensions, zlib=True, complevel=1)
            variable.standard_name = standard_name
            variable.units = units
            variable.grid_mapping = "latitude_longitude"
            variable.coordinates = coordinates
            shape = tuple(len(dataset.dimensions[item]) for item in dimensions)
            data = np.broadcast_to(np.asarray(value, dtype=np.float32), shape)
            variable[:] = data
    return path


def _direction_grid() -> np.ndarray:
    """Directions that exercise every quadrant, 0 and 360 included."""

    values = np.linspace(0.0, 360.0, NY * NX).reshape(NY, NX)
    return values.astype(np.float32)


def _speed_grid() -> np.ndarray:
    return (np.arange(NY * NX, dtype=np.float32).reshape(NY, NX) * 0.75 + 0.5)


def _write_cycle(directory: Path, lead_hours: int, **overrides) -> list[Path]:
    """The files of one lead: pressure levels, humidity, surface, screen."""

    directory.mkdir(parents=True, exist_ok=True)
    stamp = f"PT{lead_hours:04d}H00M"
    level_count = len(ALL_LEVELS_PA)
    speed3 = np.broadcast_to(_speed_grid(), (level_count, NY, NX))
    direction3 = np.broadcast_to(_direction_grid(), (level_count, NY, NX))
    files = [
        _write_mo_file(directory / f"{stamp}-pl.nc", {
            "air_temperature": ("air_temperature", "K", 270.0 + lead_hours),
            "geopotential_height": ("geopotential_height", "m",
                                    np.array([110.0, 1500.0, 3000.0, 5600.0,
                                              10400.0])[:, None, None]),
            "wind_speed": ("wind_speed", "m s-1", speed3),
            "wind_from_direction": ("wind_from_direction", "degrees",
                                    direction3),
        }, lead_hours=lead_hours, levels=ALL_LEVELS_PA, **overrides),
        _write_mo_file(directory / f"{stamp}-rh.nc", {
            "relative_humidity": ("relative_humidity", "%", 70.0),
        }, lead_hours=lead_hours, levels=RH_LEVELS_PA, **overrides),
        _write_mo_file(directory / f"{stamp}-sfc.nc", {
            "surface_air_pressure": ("surface_air_pressure", "Pa", 101000.0),
            "surface_altitude": ("surface_altitude", "m", 25.0),
            "surface_temperature": ("surface_temperature", "K", 285.0),
            "land_binary_mask": ("land_binary_mask", "1", 0.75),
            "wind_speed_10m": ("wind_speed", "m s-1", _speed_grid()),
            "wind_from_direction_10m": ("wind_from_direction", "degrees",
                                        _direction_grid()),
        }, lead_hours=lead_hours, **overrides),
        _write_mo_file(directory / f"{stamp}-screen.nc", {
            "screen_temperature": ("air_temperature", "K", 284.0),
            "screen_specific_humidity": ("specific_humidity", "kg kg-1", 0.006),
        }, lead_hours=lead_hours, height_m=1.5, **overrides),
    ]
    soil_path = directory / f"{stamp}-soil.nc"
    files.append(soil_path)
    _write_soil_file(soil_path, lead_hours, **overrides)
    return files


def _write_soil_file(path: Path, lead_hours: int, **overrides) -> None:
    """The soil column the Met Office does not publish (GFS supplies it)."""

    _write_mo_file(path, {}, lead_hours=lead_hours, **overrides)
    with netCDF4.Dataset(path, "a") as dataset:
        dataset.createDimension("soil", 4)
        for name, value in (("soil_temperature", 284.0),
                            ("volumetric_soil_moisture", 0.25)):
            variable = dataset.createVariable(
                name, "f4", ("soil", "latitude", "longitude"))
            variable.units = "K" if name == "soil_temperature" else "m3 m-3"
            variable[:] = np.full((4, NY, NX), value, dtype=np.float32)


def _selector(name: str, **attributes) -> dict[str, object]:
    selector: dict[str, object] = {"format": "netcdf", "name": name}
    if attributes:
        selector["attributes"] = attributes
    return selector


def _direct(selector, units, axes, location, **extra) -> dict[str, object]:
    return {
        "selectors": [selector],
        "units": {"source": units, "target": units, "scale": 1.0, "offset": 0.0},
        "source_axes": list(axes),
        "target_axes": list(axes),
        "location": location,
        "staggering": "none",
        "missing": {"kind": "reject"},
        **extra,
    }


def _derived(derivation, units, axes, location, **extra) -> dict[str, object]:
    return {
        "selectors": [],
        "derivation": derivation,
        "units": {"source": units, "target": units, "scale": 1.0, "offset": 0.0},
        "source_axes": list(axes),
        "target_axes": list(axes),
        "location": location,
        "staggering": "none",
        "missing": {"kind": "reject"},
        **extra,
    }


def _mapping() -> dict[str, object]:
    column = ("vertical", "y", "x")
    plane = ("y", "x")
    direction = {"standard_name": "wind_from_direction"}
    fields = {
        "air_temperature": _direct(_selector("air_temperature"), "K", column, "mass"),
        "geopotential_height": _direct(
            _selector("geopotential_height"), "m", column, "mass"),
        "relative_humidity": _direct(
            _selector("relative_humidity"), "%", column, "mass",
            dependency_only=True),
        "wind_speed": _direct(
            _selector("wind_speed"), "m s-1", column, "mass",
            dependency_only=True),
        "wind_from_direction": _direct(
            _selector("wind_from_direction", **direction), "degrees", column,
            "mass", dependency_only=True),
        "air_pressure": _derived("pressure", "Pa", column, "mass"),
        "specific_humidity": _derived("q-from-rh", "kg kg-1", column, "mass"),
        "eastward_wind": _derived("u-from-speed", "m s-1", column, "mass"),
        "northward_wind": _derived("v-from-speed", "m s-1", column, "mass"),
        "surface_pressure": _direct(
            _selector("surface_air_pressure"), "Pa", plane, "surface"),
        "terrain_height": _direct(
            _selector("surface_altitude"), "m", plane, "surface"),
        "skin_temperature": _direct(
            _selector("surface_temperature"), "K", plane, "surface"),
        "air_temperature_2m": _direct(
            _selector("screen_temperature"), "K", plane, "surface",
            screen_height_m=1.5),
        "specific_humidity_2m": _direct(
            _selector("screen_specific_humidity"), "kg kg-1", plane, "surface",
            screen_height_m=1.5),
        "wind_speed_10m": _direct(
            _selector("wind_speed_10m"), "m s-1", plane, "surface",
            dependency_only=True),
        "wind_from_direction_10m": _direct(
            _selector("wind_from_direction_10m", **direction), "degrees",
            plane, "surface", dependency_only=True),
        "eastward_wind_10m": _derived("u10-from-speed", "m s-1", plane, "surface"),
        "northward_wind_10m": _derived("v10-from-speed", "m s-1", plane, "surface"),
        "land_fraction": _direct(
            _selector("land_binary_mask"), "1", plane, "surface"),
        "soil_temperature": _direct(
            _selector("soil_temperature"), "K", ("soil", "y", "x"), "soil"),
        "volumetric_soil_moisture": _direct(
            _selector("volumetric_soil_moisture"), "m3 m-3",
            ("soil", "y", "x"), "soil"),
    }
    required = [
        {"name": name, "axes": list(column), "location": "mass",
         "target_units": units}
        for name, units in THREE_D.items()
    ] + [
        {"name": name,
         "axes": ["soil", "y", "x"] if location == "soil" else ["y", "x"],
         "location": location, "target_units": units}
        for name, (units, location) in SURFACE.items()
    ]
    return {
        "schema": "rw-wps.mapping.v1",
        "name": "synthetic-met-office-scalar-time",
        "format": "netcdf",
        "coordinates": {
            "horizontal": {
                "kind": "variables",
                "latitude": _selector("latitude"),
                "longitude": _selector("longitude"),
            },
            "vertical": {
                "kind": "pressure", "selector": _selector("pressure"),
                "units": "Pa", "positive": "down", "levels": LEVELS_PA,
            },
            "time": {
                "kind": "dimension",
                "time_coordinate": "scalar",
                "selector": {"name": ["time"], "standard_name": "time"},
                "units": EPOCH_UNITS,
                "calendar": "standard",
                "reference_time_selector": {
                    "name": ["forecast_reference_time"],
                    "standard_name": "forecast_reference_time",
                },
            },
        },
        "fields": fields,
        "derivations": [
            {"name": "pressure", "operation": "pressure_from_vertical_coordinate"},
            {"name": "q-from-rh", "operation": "specific_humidity_from_rh",
             "relative_humidity": "relative_humidity",
             "temperature": "air_temperature", "pressure": "air_pressure"},
            {"name": "u-from-speed",
             "operation": "eastward_wind_from_speed_direction",
             "speed": "wind_speed", "direction": "wind_from_direction"},
            {"name": "v-from-speed",
             "operation": "northward_wind_from_speed_direction",
             "speed": "wind_speed", "direction": "wind_from_direction"},
            {"name": "u10-from-speed",
             "operation": "eastward_wind_from_speed_direction",
             "speed": "wind_speed_10m", "direction": "wind_from_direction_10m"},
            {"name": "v10-from-speed",
             "operation": "northward_wind_from_speed_direction",
             "speed": "wind_speed_10m", "direction": "wind_from_direction_10m"},
        ],
        "target": {
            "name": "woof/wrf-real initialization",
            "physics_suite": "WSM6+YSU+Noah",
            "max_dom": 1,
            "require_lateral_boundaries": True,
            "target_vertical_levels": 49,
            "soil_layer_count": 4,
            "boundary_interval_seconds": 10800,
            "required_fields": required,
            "pressure_requirement": "air_pressure",
            "policy_controlled_fields": POLICY,
            "initialization_policies": {
                name: "explicit_zero_with_adapter_validation" for name in POLICY
            },
        },
    }


def _write_mapping(path: Path, mapping: dict[str, object]) -> Path:
    path.write_text(json.dumps(mapping, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    return path


def _require_engine():
    try:
        return engine_bridge.require_engine()
    except engine_bridge.EngineUnavailable as error:
        pytest.skip(f"gpuwm_mapped_engine is not built: {error}")


def _require_portable_math():
    """Skip a byte comparison the Python engine cannot make here.

    The speed/direction winds and the humidity derivations take
    ``portable_math`` in the Python engine and the same ``libm`` crate in
    the Rust engine.  Without the CPU preprocessing library the Python
    side falls back to the host C library, which is a build fact, not an
    engine verdict, so it is said by name.
    """

    if pm.implementation() != pm.IMPLEMENTATION:
        pytest.skip("the CPU preprocessing library with the portable math "
                    "entries is not built here, so the Python engine runs "
                    "the host C library fallback, not the portable contract")


def _rust_decode(mapping: Path, files, work: Path):
    """The Rust engine's frames, for a byte comparison with the Python's."""

    _require_portable_math()
    _require_engine()
    directory = work / "frameset"
    engine_bridge.run_engine("decode", mapping=mapping, files=files,
                             output=directory)
    return engine_bridge.read_frameset(directory)


def _assert_frames_byte_equal(python_frames, rust_frames) -> None:
    assert len(python_frames) == len(rust_frames)
    for left, right in zip(python_frames, rust_frames):
        assert left.valid_time == right.valid_time
        assert left.source_cycle == right.source_cycle
        assert left.grid_fingerprint == right.grid_fingerprint
        assert sorted(left.fields) == sorted(right.fields)
        for name in left.fields:
            ours, theirs = left.fields[name], right.fields[name]
            assert ours.values.dtype == theirs.values.dtype == np.float64, name
            assert ours.values.shape == theirs.values.shape, name
            assert np.ascontiguousarray(ours.values).tobytes() == \
                np.ascontiguousarray(theirs.values).tobytes(), name
            assert tuple(ours.source_references) == \
                tuple(theirs.source_references), name
        assert left.header.to_dict() == right.header.to_dict()


def _refusals_agree(mapping: Path, files, work: Path) -> str:
    """Both engines refuse, with one exception type and one sentence."""

    with pytest.raises(ValueError) as python_error:
        _decode_mapped_source_python(mapping, files)
    _require_engine()
    with pytest.raises(ValueError) as rust_error:
        engine_bridge.run_engine("decode", mapping=mapping, files=files,
                                 output=work / "refused")
    python_text = str(python_error.value)
    assert str(rust_error.value).startswith(python_text), (
        python_text, str(rust_error.value))
    return python_text


@pytest.fixture()
def cycle_files(tmp_path):
    files = [*_write_cycle(tmp_path / "src", 0), *_write_cycle(tmp_path / "src", 3)]
    return [path.resolve() for path in files]


def test_scalar_time_decodes_identically_through_both_engines(tmp_path, cycle_files):
    mapping = _write_mapping(tmp_path / "mapping.json", _mapping())
    python_frames = _decode_mapped_source_python(mapping, cycle_files)
    rust_frames = _rust_decode(mapping, cycle_files, tmp_path)
    _assert_frames_byte_equal(python_frames, rust_frames)

    # Two leads of one run share ONE cycle: the reference time, not the
    # valid time, so a same-cycle borrow does not see mixed cycles.
    assert [frame.valid_time for frame in python_frames] == [
        CYCLE, CYCLE + timedelta(hours=3)]
    assert {frame.source_cycle for frame in python_frames} == {CYCLE}
    # 37/33-style level sets: the humidity file's extra levels and the
    # pressure file's extra levels are both cut to the declared ladder.
    assert python_frames[0].vertical_values.tolist() == LEVELS_PA


def test_speed_and_direction_become_earth_relative_components(tmp_path, cycle_files):
    mapping = _write_mapping(tmp_path / "mapping.json", _mapping())
    frame = _decode_mapped_source_python(mapping, cycle_files)[0]
    speed = _speed_grid().astype(np.float64)
    radians = _direction_grid().astype(np.float64) * (np.pi / 180.0)
    u10 = frame.fields["eastward_wind_10m"].values
    v10 = frame.fields["northward_wind_10m"].values
    # The portable sin and cos, which the Rust engine's libm crate repeats.
    assert np.array_equal(u10, -speed * pm.sin(radians))
    assert np.array_equal(v10, -speed * pm.cos(radians))
    # Speed is preserved, and the meteorological convention holds: FROM
    # the north (0 and 360) blows southward.
    assert np.allclose(np.hypot(u10, v10), speed, rtol=0, atol=1e-12)
    directions = _direction_grid().astype(np.float64)
    for degrees, sign_u, sign_v in ((0.0, 0, -1), (360.0, 0, -1)):
        cell = np.argwhere(directions == degrees)[0]
        assert np.sign(round(v10[tuple(cell)], 9)) == sign_v
        assert abs(u10[tuple(cell)]) < 1e-12 * speed[tuple(cell)] + 1e-12
    # A wind FROM the west half (180..360 exclusive) blows eastward: u > 0.
    from_west = (directions > 180.0) & (directions < 360.0)
    assert (u10[from_west] > 0).all() and (u10[(directions > 0) & (directions < 180.0)] < 0).all()
    u = frame.fields["eastward_wind"].values
    assert np.array_equal(u[0], u10)


def test_declared_screen_height_is_recorded_and_checked(tmp_path, cycle_files):
    mapping = _write_mapping(tmp_path / "mapping.json", _mapping())
    frame = _decode_mapped_source_python(mapping, cycle_files)[0]
    assert "@screen_height_m=1.5" in frame.fields["air_temperature_2m"].source_references
    assert "@screen_height_m=1.5" in frame.fields["specific_humidity_2m"].source_references
    assert not any(
        reference.startswith("@screen_height_m")
        for reference in frame.fields["skin_temperature"].source_references)

    wrong = _mapping()
    wrong["fields"]["air_temperature_2m"]["screen_height_m"] = 2.0
    message = _refusals_agree(
        _write_mapping(tmp_path / "wrong.json", wrong), cycle_files, tmp_path)
    assert "screen_height_m=2.0" in message and "height 1.5 m" in message


def test_two_runs_for_one_valid_time_refuse_in_both(tmp_path):
    """Each file passes its own reference + period check; the mix does not."""

    files = [path.resolve() for path in _write_cycle(tmp_path / "src", 3)]
    later_run = _write_mo_file(tmp_path / "later-pl.nc", {
        "air_temperature_spare": ("air_temperature", "K", 271.0),
    }, lead_hours=0, levels=ALL_LEVELS_PA, reference=CYCLE + timedelta(hours=3))
    with netCDF4.Dataset(files[0], "a") as dataset:
        dataset.renameVariable("air_temperature", "air_temperature_old")
    with netCDF4.Dataset(later_run, "a") as dataset:
        dataset.renameVariable("air_temperature_spare", "air_temperature")
    files = [*files, later_run.resolve()]
    mapping = _mapping()
    mapping["target"]["require_lateral_boundaries"] = False
    path = _write_mapping(tmp_path / "mapping.json", mapping)
    message = _refusals_agree(path, files, tmp_path)
    assert "NetCDF files mix source cycles at 2026-10-11T03:00:00: " \
        "2026-10-11T00:00:00 and 2026-10-11T03:00:00" in message


def test_a_non_gregorian_reference_calendar_refuses_in_both(tmp_path):
    files = [path.resolve() for path in (
        *_write_cycle(tmp_path / "src", 0), *_write_cycle(tmp_path / "src", 3))]
    with netCDF4.Dataset(files[0], "a") as dataset:
        dataset.variables["forecast_reference_time"].calendar = "360_day"
    path = _write_mapping(tmp_path / "mapping.json", _mapping())
    message = _refusals_agree(path, files, tmp_path)
    assert "calendar '360_day' is not supported" in message


def test_a_period_that_does_not_reach_the_valid_time_refuses_in_both(tmp_path):
    files = [path.resolve() for path in (
        *_write_cycle(tmp_path / "src", 0),
        *_write_cycle(tmp_path / "src", 3, period_seconds=7200))]
    mapping = _write_mapping(tmp_path / "mapping.json", _mapping())
    message = _refusals_agree(mapping, files, tmp_path)
    assert "is not reference time 2026-10-11T00:00:00 plus forecast_period " \
        "7200 s" in message


def test_without_a_reference_selector_the_valid_time_is_the_cycle(tmp_path, cycle_files):
    mapping = _mapping()
    del mapping["coordinates"]["time"]["reference_time_selector"]
    path = _write_mapping(tmp_path / "mapping.json", mapping)
    python_frames = _decode_mapped_source_python(path, cycle_files)
    assert [frame.source_cycle for frame in python_frames] == [
        frame.valid_time for frame in python_frames]
    _assert_frames_byte_equal(
        python_frames, _rust_decode(path, cycle_files, tmp_path))


def test_kind_scalar_is_the_same_opt_in(tmp_path, cycle_files):
    mapping = _mapping()
    mapping["coordinates"]["time"]["kind"] = "scalar"
    del mapping["coordinates"]["time"]["time_coordinate"]
    path = _write_mapping(tmp_path / "mapping.json", mapping)
    python_frames = _decode_mapped_source_python(path, cycle_files)
    assert {frame.source_cycle for frame in python_frames} == {CYCLE}
    _assert_frames_byte_equal(
        python_frames, _rust_decode(path, cycle_files, tmp_path))


def test_a_dimension_mapping_still_refuses_a_scalar_file(tmp_path, cycle_files):
    mapping = _mapping()
    time = mapping["coordinates"]["time"]
    del time["time_coordinate"], time["reference_time_selector"]
    path = _write_mapping(tmp_path / "mapping.json", mapping)
    with pytest.raises(ValueError, match="coordinate dimensions"):
        _decode_mapped_source_python(path, cycle_files)
    _require_engine()
    with pytest.raises(ValueError, match="exactly one dimension"):
        engine_bridge.run_engine("decode", mapping=path, files=cycle_files,
                                 output=tmp_path / "refused")


def test_out_of_range_direction_refuses_in_both(tmp_path):
    directory = tmp_path / "src"
    files = [path.resolve() for path in
             (*_write_cycle(directory, 0), *_write_cycle(directory, 3))]
    with netCDF4.Dataset(files[2], "a") as dataset:
        dataset.variables["wind_from_direction_10m"][0, 0] = 361.0
    mapping = _write_mapping(tmp_path / "mapping.json", _mapping())
    message = _refusals_agree(mapping, files, tmp_path)
    assert "1 wind direction value(s) outside [0, 360] degrees" in message


@pytest.mark.parametrize("mutation, expected", [
    (lambda m: m["fields"]["air_temperature"]["source_axes"].insert(0, "time"),
     "names a 'time' axis, but the mapping's time coordinate is scalar"),
    (lambda m: m["fields"]["wind_from_direction"]["selectors"][0].pop("attributes"),
     "must pin attributes.standard_name='wind_from_direction'"),
    (lambda m: m["fields"]["wind_from_direction"]["units"].update(
        source="radians", target="radians"),
     "must be degrees in and out"),
    (lambda m: m["fields"]["wind_speed"]["units"].update(
        source="knots", target="knots"),
     "are not 'm s-1'"),
    (lambda m: m["coordinates"]["time"].update(time_coordinate="dimension"),
     "time_coordinate may only be 'scalar'"),
    (lambda m: m["coordinates"]["time"].pop("time_coordinate"),
     "reference_time_selector is read beside a scalar time coordinate only"),
    (lambda m: m["fields"]["skin_temperature"].update(screen_height_m=50.0),
     "screen_height_m must be a number of metres"),
    (lambda m: m["fields"]["eastward_wind_10m"].update(screen_height_m=1.5),
     "must declare selectors"),
])
def test_the_new_declarations_refuse_by_name(tmp_path, mutation, expected):
    mapping = _mapping()
    mutation(mapping)
    path = _write_mapping(tmp_path / "mapping.json", mapping)
    with pytest.raises(ValueError, match=expected.replace("(", r"\(")
                       .replace(")", r"\)").replace("[", r"\[")):
        load_mapping(path)


def test_a_scalar_time_supplement_decodes_through_both_engines(tmp_path):
    """The orography supplement: step 0 only, its scalar time the cycle."""

    orography = _write_mo_file(tmp_path / "PT0000H00M-height_of_orography.nc", {
        "surface_altitude": ("surface_altitude", "m",
                             np.arange(NY * NX, dtype=np.float32).reshape(NY, NX)),
    }).resolve()
    full = _mapping()

    # The composition's partitioned donor decode is `_decode_netcdf` on a
    # copy of the mapping cut to the borrowed fields.
    from woof.mapped_composition import _decode_partition, _partition_mapping

    partition = _partition_mapping(
        load_mapping(_write_mapping(tmp_path / "full.json", full)),
        terrain_only=True)
    collection = _decode_partition(partition, (orography,), {})
    ((valid, member, name),) = collection.direct
    assert (valid, member, name) == (CYCLE, None, "terrain_height")
    assert collection.source_cycles == {(CYCLE, None): CYCLE}

    # The Rust engine reads the same partition: its `inspect` decodes the
    # direct fields a mapping owes (the engine trusts the front door's
    # validation and does not re-run it), so the partition document is
    # handed over as the composition builds it.
    path = _write_mapping(tmp_path / "terrain.json", partition)
    _require_engine()
    result = engine_bridge.run_engine(
        "inspect", mapping=path, files=[orography],
        output=tmp_path / "inspection")
    from woof.mapped_source import _array_sha256, _inspection_from_stdout

    rows = _inspection_from_stdout(str(result["stdout"]))["frames"]
    assert len(rows) == 1
    assert rows[0]["valid_time"] == CYCLE.isoformat()
    assert rows[0]["source_cycle"] == CYCLE.isoformat()
    assert rows[0]["decoded_direct_fields"] == ["terrain_height"]
    assert rows[0]["fields"]["terrain_height"]["sha256"] == _array_sha256(
        collection.direct[(CYCLE, None, "terrain_height")].values)
