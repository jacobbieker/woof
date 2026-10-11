"""Each transcendental in the mapped engines runs one implementation, in both.

Breakage prevented: the Rust ``gpuwm_mapped_engine`` called the platform C
library's ``exp`` and ``ln`` where its Python twin calls
:mod:`woof.core.portable_math` (the vendored ``libm`` crate, a Rust port of
musl's libm).  ``specific_humidity_from_rh`` and
``specific_humidity_from_dewpoint`` therefore differed in the last bit on
real data -- about one ``exp`` argument in ten rounds differently in glibc
and musl -- and so did the hydrostatic height and the height at the
surface pressure, which take ``log`` the same way.  No exemption covered
it: on any platform other than the one that measured a committed golden,
the parity battery runs the live two-engine comparison, which these
fields failed for every humidity source.

The engine now calls the ``libm`` crate wherever its twin calls
``portable_math``, and the C library wherever its twin calls ``math.*`` on
purpose (the sea-level reduction, the hypsometric completion).  The list
is the "Transcendentals" section of
``tools/rw_wps/crates/mapped-engine/src/derive.rs``.  Two halves hold it:

1. Both engines decode synthetic sources of 164,737 pseudo-random cells
   per plane plus edge cases, and every array must be byte-identical.  The
   sources reach every portable decode derivation (the three humidity
   relations, both speed/direction wind pairs, the hybrid hydrostatic
   height) and the sea-level reduction from the system list.  The
   hypsometric completion and the Lambert rotation, the rest of the system
   list, are compared by their own two-engine tests.
2. Four portable kernels are pinned: the humidity relations, the winds and
   the height at the surface pressure, which only composition reaches.  The
   same deterministic arguments (splitmix64, generated identically by
   ``derive.rs``'s unit tests) go through the Python kernels, and the sha256
   of the answers must equal the one the Rust unit tests assert, so the two
   engines agree on every host and C library, not only on this one.  The
   hydrostatic height has no pin; half 1 is its comparison.

Every ``libm`` the engines link is checked to be the version
``portable_math`` names, so a lock refresh in one workspace cannot part
them quietly.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import tomllib

import netCDF4
import numpy as np
import pytest

import test_mapped_netcdf_scalar_time as scalar_time
from test_mapped_netcdf_scalar_time import (
    CYCLE, EPOCH_UNITS, _assert_frames_byte_equal, _derived, _direct,
    _mapping, _require_portable_math, _rust_decode, _scalar, _seconds,
    _selector, _write_mapping, _write_mo_file, _write_soil_file)
from woof.core import portable_math as pm
from woof.mapped_source import (CanonicalField, _decode_mapped_source_python,
                                _height_at_surface_pressure,
                                _wind_component_from_speed_direction)
from woof.ingest.real import (_saturation_mixing_ratio,
                              _surface_relative_humidity)


# ---------------------------------------------------------------------------
# 1. Both engines, one synthetic source, every transcendental derivation.
# ---------------------------------------------------------------------------

#: 164,737 cells per plane.  The fixture writer spans 50..54 N and
#: 30..25 W, so these counts give dyadic spacings (1/64 and 1/128 degree)
#: that float32 axes hold exactly, as a regular grid must.
NY, NX = 257, 641
LEVELS_PA = [100000.0, 92500.0, 85000.0, 70000.0, 50000.0]
#: Mean level heights of LEVELS_PA; each column is jittered around them.
LEVEL_HEIGHTS_M = [100.0, 760.0, 1450.0, 3000.0, 5600.0]


def _plane(rng, low, high, edges=()):
    """A float32 plane of uniform values, its first cells the edges."""

    values = rng.uniform(low, high, NY * NX)
    values[:len(edges)] = edges
    return values.astype(np.float32).reshape(NY, NX)


def _column(rng, low, high, edges=()):
    return np.stack([_plane(rng, low, high, edges) for _ in LEVELS_PA])


#: Wind directions where sin or cos is exactly 0 or +-1 in degrees, and
#: the ends of the admitted range.
DIRECTION_EDGES = (0.0, 360.0, 90.0, 180.0, 270.0, 45.0, 1.0e-6, 359.999,
                   135.0, 225.0, 315.0, 0.5)
SPEED_EDGES = (0.0, 0.0, 1.0e-3, 100.0, 7.5, 3.0e-7, 50.0, 0.25)
#: Relative humidity below 0 and above 100 (both clamped), both clamps
#: themselves, and tiny values.
RH_EDGES = (0.0, 100.0, -5.0, 120.0, 1.0e-4, 99.9999, 50.0, 0.5)
#: Temperatures at the freezing point, very cold, and hot enough that the
#: saturation pressure passes p / 100 on the upper levels (the 1e-6 floor).
TEMPERATURE_EDGES = (273.15, 180.0, 330.0, 370.0, 233.15, 300.0, 255.5, 365.0)


def _write_sources(directory, rng):
    directory.mkdir(parents=True, exist_ok=True)
    heights = np.stack([
        _plane(rng, mean - 40.0, mean + 40.0) for mean in LEVEL_HEIGHTS_M])
    temperature = _column(rng, 180.0, 330.0, TEMPERATURE_EDGES)
    dewpoint = _plane(rng, 200.0, 310.0, (273.15, 200.0, 310.0, 250.0))
    # Mostly drier than saturated, some supersaturated: RH above 100,
    # unclipped in the 2 m relative humidity and clamped in the humidity.
    screen = (dewpoint + _plane(rng, -3.0, 35.0, (0.0, 0.0, 0.0, -3.0))
              ).astype(np.float32)
    files = [
        _write_mo_file(directory / "pl.nc", {
            "air_temperature": ("air_temperature", "K", temperature),
            "geopotential_height": ("geopotential_height", "m", heights),
            "wind_speed": ("wind_speed", "m s-1",
                           _column(rng, 0.0, 80.0, SPEED_EDGES)),
            "wind_from_direction": ("wind_from_direction", "degrees",
                                    _column(rng, 0.0, 360.0, DIRECTION_EDGES)),
        }, levels=LEVELS_PA),
        _write_mo_file(directory / "rh.nc", {
            "relative_humidity": ("relative_humidity", "%",
                                  _column(rng, -5.0, 110.0, RH_EDGES)),
        }, levels=LEVELS_PA),
        _write_mo_file(directory / "sfc.nc", {
            # Every branch of the sea-level reduction: under 50 m, between
            # two level heights, below the lowest level with the sea-level
            # pressure above and below 1000 hPa.
            "air_pressure_at_sea_level": (
                "air_pressure_at_sea_level", "Pa",
                _plane(rng, 97000.0, 104000.0, (101325.0, 97000.0, 104000.0))),
            "surface_altitude": ("surface_altitude", "m",
                                 _plane(rng, -20.0, 2500.0,
                                        (0.0, 49.0, 50.0, 60.0, 2400.0))),
            "surface_temperature": ("surface_temperature", "K",
                                    _plane(rng, 220.0, 320.0)),
            "land_binary_mask": ("land_binary_mask", "1", 1.0),
            "wind_speed_10m": ("wind_speed", "m s-1",
                               _plane(rng, 0.0, 40.0, SPEED_EDGES)),
            "wind_from_direction_10m": (
                "wind_from_direction", "degrees",
                _plane(rng, 0.0, 360.0, DIRECTION_EDGES)),
        }),
        _write_mo_file(directory / "screen.nc", {
            "screen_temperature": ("air_temperature", "K", screen),
            "screen_dewpoint": ("dew_point_temperature", "K", dewpoint),
        }, height_m=1.5),
    ]
    soil = directory / "soil.nc"
    _write_soil_file(soil, 0)
    files.append(soil)
    return [path.resolve() for path in files]


def _transcendental_mapping():
    mapping = _mapping()
    plane = ("y", "x")
    mapping["name"] = "synthetic-transcendental-parity"
    mapping["coordinates"]["vertical"]["levels"] = LEVELS_PA
    # One valid time: the comparison is about values, not the series.
    mapping["target"]["require_lateral_boundaries"] = False
    fields = mapping["fields"]
    del fields["specific_humidity_2m"]
    fields.update({
        "air_pressure_at_mean_sea_level": _direct(
            _selector("air_pressure_at_sea_level"), "Pa", plane, "surface",
            dependency_only=True),
        "surface_pressure": _derived("ps-from-slp", "Pa", plane, "surface"),
        "dewpoint_temperature_2m": _direct(
            _selector("screen_dewpoint"), "K", plane, "surface",
            screen_height_m=1.5, dependency_only=True),
        "specific_humidity_2m": _derived(
            "q2-from-dewpoint", "kg kg-1", plane, "surface"),
        "relative_humidity_2m": _derived(
            "rh2-from-dewpoint", "%", plane, "surface"),
    })
    mapping["derivations"] += [
        {"name": "ps-from-slp", "operation": "surface_pressure_from_sea_level",
         "sea_level_pressure": "air_pressure_at_mean_sea_level",
         "level_height": "geopotential_height", "pressure": "air_pressure",
         "surface_height": "terrain_height"},
        {"name": "q2-from-dewpoint",
         "operation": "specific_humidity_from_dewpoint",
         "dewpoint": "dewpoint_temperature_2m",
         "temperature": "air_temperature_2m", "pressure": "surface_pressure"},
        {"name": "rh2-from-dewpoint",
         "operation": "relative_humidity_from_dewpoint",
         "dewpoint": "dewpoint_temperature_2m",
         "temperature": "air_temperature_2m"},
    ]
    return mapping


#: The fields a transcendental produces in the synthetic source: the three
#: humidity derivations, both speed/direction wind pairs and the sea-level
#: surface pressure.
TRANSCENDENTAL_FIELDS = (
    "specific_humidity", "specific_humidity_2m", "relative_humidity_2m",
    "eastward_wind", "northward_wind", "eastward_wind_10m",
    "northward_wind_10m", "surface_pressure",
)


def _differing(python_frames, rust_frames, names):
    """``{field: differing element count}`` over every frame."""

    counts = {}
    for left, right in zip(python_frames, rust_frames):
        for name in names:
            ours = np.ascontiguousarray(left.fields[name].values).view(np.uint64)
            theirs = np.ascontiguousarray(right.fields[name].values).view(np.uint64)
            counts[name] = counts.get(name, 0) + int(np.count_nonzero(ours != theirs))
    return counts


def test_both_engines_agree_on_every_transcendental_derivation(
        tmp_path, monkeypatch):
    _require_portable_math()
    monkeypatch.setattr(scalar_time, "NY", NY)
    monkeypatch.setattr(scalar_time, "NX", NX)
    files = _write_sources(tmp_path / "src", np.random.default_rng(20261011))
    mapping = _write_mapping(tmp_path / "mapping.json",
                             _transcendental_mapping())
    python_frames = _decode_mapped_source_python(mapping, files)
    rust_frames = _rust_decode(mapping, files, tmp_path)
    assert len(python_frames) == len(rust_frames) == 1
    frame = python_frames[0]
    for name in TRANSCENDENTAL_FIELDS:
        values = frame.fields[name].values
        assert values.size >= NY * NX and np.isfinite(values).all(), name
    # The fields carry the derivations' interesting answers, so the
    # comparison below is not over a trivial array.
    assert np.any(frame.fields["specific_humidity"].values == 1.0e-6
                  / (1.0 + 1.0e-6))
    assert np.any(frame.fields["relative_humidity_2m"].values > 100.0)
    assert np.any(frame.fields["surface_pressure"].values > 100000.0)
    differing = _differing(python_frames, rust_frames, TRANSCENDENTAL_FIELDS)
    assert not any(differing.values()), (
        "the engines disagree on transcendental derivations "
        f"(differing elements per field): {differing}")
    _assert_frames_byte_equal(python_frames, rust_frames)


#: Half-level interfaces top down, the top one 0 Pa so the top full level
#: takes the provider's 0.1 Pa clamp; strictly increasing for any surface
#: pressure from 500 to 1100 hPa.
HYBRID_A = [0.0, 2000.0, 5000.0, 8000.0, 7000.0, 3000.0, 0.0]
HYBRID_B = [0.0, 0.0, 0.05, 0.2, 0.5, 0.8, 1.0]
MODEL_LEVELS = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


def _write_hybrid_source(path, rng):
    """One model-level file: the column, the surface and the soil."""

    count = len(MODEL_LEVELS)

    def column(low, high):
        return np.stack([_plane(rng, low, high) for _ in range(count)])

    variables = {
        ("model_level", "latitude", "longitude"): {
            "air_temperature": ("K", column(180.0, 320.0)),
            "specific_humidity": ("kg kg-1", column(0.0, 0.025)),
            "x_wind": ("m s-1", column(-60.0, 60.0)),
            "y_wind": ("m s-1", column(-60.0, 60.0)),
        },
        ("latitude", "longitude"): {
            "surface_air_pressure": (
                "Pa", _plane(rng, 50000.0, 110000.0, (50000.0, 110000.0))),
            "surface_altitude": ("m", _plane(rng, -400.0, 5000.0)),
            "surface_temperature": ("K", _plane(rng, 220.0, 320.0)),
            "screen_temperature": ("K", _plane(rng, 220.0, 320.0)),
            "screen_specific_humidity": ("kg kg-1", _plane(rng, 0.0, 0.02)),
            "x_wind_10m": ("m s-1", _plane(rng, -30.0, 30.0)),
            "y_wind_10m": ("m s-1", _plane(rng, -30.0, 30.0)),
            "land_binary_mask": ("1", np.ones((NY, NX), dtype=np.float32)),
        },
        ("soil", "latitude", "longitude"): {
            "soil_temperature": ("K", np.full((4, NY, NX), 284.0, np.float32)),
            "volumetric_soil_moisture": (
                "m3 m-3", np.full((4, NY, NX), 0.25, np.float32)),
        },
    }
    with netCDF4.Dataset(path, "w") as dataset:
        dataset.createDimension("model_level", count)
        dataset.createDimension("latitude", NY)
        dataset.createDimension("longitude", NX)
        dataset.createDimension("soil", 4)
        level = dataset.createVariable("model_level", "f4", ("model_level",))
        level.units = "1"
        level.positive = "down"
        level[:] = MODEL_LEVELS
        latitude = dataset.createVariable("latitude", "f4", ("latitude",))
        latitude.units = "degrees_north"
        latitude[:] = np.linspace(50.0, 54.0, NY)
        longitude = dataset.createVariable("longitude", "f4", ("longitude",))
        longitude.units = "degrees_east"
        longitude[:] = np.linspace(-30.0, -25.0, NX)
        _scalar(dataset, "forecast_reference_time", "i8", _seconds(CYCLE),
                units=EPOCH_UNITS, standard_name="forecast_reference_time",
                calendar="standard")
        _scalar(dataset, "time", "i8", _seconds(CYCLE), units=EPOCH_UNITS,
                standard_name="time", calendar="standard")
        for dimensions, group in variables.items():
            for name, (units, values) in group.items():
                variable = dataset.createVariable(name, "f4", dimensions)
                variable.units = units
                variable[:] = values
    return path.resolve()


def _hybrid_mapping():
    mapping = _mapping()
    column = ("vertical", "y", "x")
    plane = ("y", "x")
    soil = ("soil", "y", "x")
    mapping["name"] = "synthetic-hydrostatic-parity"
    mapping["target"]["require_lateral_boundaries"] = False
    mapping["coordinates"]["vertical"] = {
        "kind": "hybrid_sigma_pressure", "selector": _selector("model_level"),
        "units": "1", "positive": "down", "levels": MODEL_LEVELS,
        "hybrid_a": HYBRID_A, "hybrid_b": HYBRID_B,
        "surface_pressure_field": "surface_pressure",
    }
    mapping["fields"] = {
        "air_temperature": _direct(
            _selector("air_temperature"), "K", column, "mass"),
        "specific_humidity": _direct(
            _selector("specific_humidity"), "kg kg-1", column, "mass"),
        "eastward_wind": _direct(_selector("x_wind"), "m s-1", column, "mass"),
        "northward_wind": _direct(_selector("y_wind"), "m s-1", column, "mass"),
        "air_pressure": _derived("pressure", "Pa", column, "mass"),
        "geopotential_height": _derived("z-hydrostatic", "m", column, "mass"),
        "surface_pressure": _direct(
            _selector("surface_air_pressure"), "Pa", plane, "surface"),
        "terrain_height": _direct(
            _selector("surface_altitude"), "m", plane, "surface"),
        "skin_temperature": _direct(
            _selector("surface_temperature"), "K", plane, "surface"),
        "air_temperature_2m": _direct(
            _selector("screen_temperature"), "K", plane, "surface"),
        "specific_humidity_2m": _direct(
            _selector("screen_specific_humidity"), "kg kg-1", plane, "surface"),
        "eastward_wind_10m": _direct(
            _selector("x_wind_10m"), "m s-1", plane, "surface"),
        "northward_wind_10m": _direct(
            _selector("y_wind_10m"), "m s-1", plane, "surface"),
        "land_fraction": _direct(
            _selector("land_binary_mask"), "1", plane, "surface"),
        "soil_temperature": _direct(
            _selector("soil_temperature"), "K", soil, "soil"),
        "volumetric_soil_moisture": _direct(
            _selector("volumetric_soil_moisture"), "m3 m-3", soil, "soil"),
    }
    mapping["derivations"] = [
        {"name": "pressure", "operation": "pressure_from_vertical_coordinate"},
        {"name": "z-hydrostatic",
         "operation": "geopotential_height_hydrostatic",
         "temperature": "air_temperature",
         "specific_humidity": "specific_humidity",
         "surface_geopotential_height": "terrain_height"},
    ]
    return mapping


def test_both_engines_agree_on_the_hydrostatic_height(tmp_path):
    """Six model levels over 164,737 columns: about a million ``log``
    ratios, the 0 Pa top interface's clamp among them."""

    _require_portable_math()
    files = [_write_hybrid_source(tmp_path / "ml.nc",
                                  np.random.default_rng(1011))]
    mapping = _write_mapping(tmp_path / "mapping.json", _hybrid_mapping())
    python_frames = _decode_mapped_source_python(mapping, files)
    rust_frames = _rust_decode(mapping, files, tmp_path)
    height = python_frames[0].fields["geopotential_height"].values
    assert height.shape == (len(MODEL_LEVELS), NY, NX)
    assert np.isfinite(height).all()
    differing = _differing(python_frames, rust_frames,
                           ("geopotential_height", "air_pressure"))
    assert not any(differing.values()), (
        "the engines disagree on the hydrostatic height "
        f"(differing elements per field): {differing}")
    _assert_frames_byte_equal(python_frames, rust_frames)


# ---------------------------------------------------------------------------
# 2. The portable kernels, pinned on both sides.
# ---------------------------------------------------------------------------

#: sha256 of each kernel's little-endian answers.  The SAME strings are
#: asserted by ``mod portable_pins`` in
#: ``tools/rw_wps/crates/mapped-engine/src/derive.rs`` over the same
#: arguments, so equal pins are byte-equal engines.  Measured on a
#: development machine (glibc, no AVX-512) on both sides; nothing in
#: these kernels reads the host's C library.
PORTABLE_KERNEL_PINS = {
    "surface_relative_humidity":
        "93040713b9a02fcbac20ef77cc68aed81b84417beef7367ccfe3c9586833150a",
    "saturation_mixing_ratio":
        "f8d57c329cc5d221ce07539924ed08981619f044e172108ba7e0aaca6730029a",
    "wind_from_speed_direction":
        "443a18dbfd70a9e30d01f25b77b4abed8cce7c124f63817da3f6f7b7195277b6",
    "height_at_surface_pressure":
        "b8f60d610cfa263c97c4e1ba7d6025824946400fc7eaf1b8f4f51baddc503f2e",
}

COUNT = 100_000


def _splitmix64(seed: int, count: int) -> np.ndarray:
    """splitmix64's outputs 1..count from ``seed``, as ``derive.rs`` has it."""

    index = np.arange(1, count + 1, dtype=np.uint64)
    with np.errstate(over="ignore"):
        z = np.uint64(seed) + index * np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def _arguments(edges, seed: int, low: float, high: float) -> np.ndarray:
    """``edges``, then COUNT uniform values in ``[low, high)``."""

    unit = (_splitmix64(seed, COUNT) >> np.uint64(11)).astype(np.float64) \
        * 2.0 ** -53
    return np.concatenate([np.asarray(edges, dtype=np.float64),
                           low + (high - low) * unit])


def _digest(*outputs) -> str:
    digest = hashlib.sha256()
    for output in outputs:
        digest.update(np.ascontiguousarray(output, dtype="<f8").tobytes())
    return digest.hexdigest()


def _field(name, units, values):
    values = np.asarray(values, dtype=np.float64).reshape(1, -1)
    return CanonicalField(
        name=name, units=units, axes=("y", "x"), location="surface",
        staggering="none", values=values, missing_count=0,
        source_references=(f"@test.{name}",))


def test_splitmix64_matches_the_reference_stream():
    # The published first output from seed 0, as derive.rs checks too.
    assert int(_splitmix64(0, 1)[0]) == 0xE220A8397B1DCDAF


def test_surface_relative_humidity_is_pinned():
    _require_portable_math()
    dewpoint = _arguments((273.15, 180.0, 320.0, 300.0, 250.5), 1, 180.0, 320.0)
    temperature = _arguments((273.15, 330.0, 180.0, 300.0, 260.25), 2,
                             180.0, 330.0)
    values = _surface_relative_humidity(dewpoint, temperature)
    assert _digest(values) == PORTABLE_KERNEL_PINS["surface_relative_humidity"]


def test_saturation_mixing_ratio_is_pinned():
    _require_portable_math()
    svp3 = 29.65
    temperature = _arguments(
        (273.15, 0.0, svp3, 300.0, 300.0, 300.0, 300.0, 380.0), 3, 150.0, 380.0)
    pressure = _arguments(
        (101325.0,) * 6 + (1.0, 50000.0), 4, 100.0, 110000.0)
    relative_humidity = _arguments(
        (100.0, 50.0, 50.0, 0.0, -1.0, 150.0, 50.0, 100.0), 5, -10.0, 120.0)
    mixing_ratio = _saturation_mixing_ratio(
        temperature, pressure, relative_humidity)
    specific = mixing_ratio / (1.0 + mixing_ratio)
    assert _digest(mixing_ratio, specific) == \
        PORTABLE_KERNEL_PINS["saturation_mixing_ratio"]


def test_wind_components_from_speed_direction_are_pinned():
    _require_portable_math()
    directions = (0.0, 360.0, 90.0, 180.0, 270.0, 45.0, 1.0e-9, 359.999999)
    speed = _field("speed", "m s-1", _arguments((10.0,) * 8, 6, 0.0, 100.0))
    direction = _field("direction", "degrees",
                       _arguments(directions, 7, 0.0, 360.0))
    u = _wind_component_from_speed_direction(
        "eastward_wind_from_speed_direction", speed, direction, "u")
    v = _wind_component_from_speed_direction(
        "northward_wind_from_speed_direction", speed, direction, "v")
    assert _digest(u, v) == PORTABLE_KERNEL_PINS["wind_from_speed_direction"]


#: Declared out of order on purpose; the kernel sorts the ladder.
PIN_LEVELS_PA = (50000.0, 100000.0, 85000.0, 10000.0, 92500.0, 30000.0,
                 70000.0)
PIN_LEVEL_HEIGHTS_M = (5600.0, 110.0, 1450.0, 16200.0, 760.0, 9200.0, 3000.0)


def test_height_at_surface_pressure_is_pinned():
    _require_portable_math()
    # Surface pressures on a level, between two, and below the deepest
    # (the dewpoint branch).
    surface = _arguments((92500.0, 100000.0, 101325.0, 10000.5, 108000.0), 8,
                         40000.0, 108000.0)
    cells = surface.size
    heights, temperature, humidity = [], [], []
    for level, mean in enumerate(PIN_LEVEL_HEIGHTS_M):
        seed = 100 + 10 * level
        heights.append(_arguments((mean,) * 5, seed, mean - 50.0, mean + 50.0))
        temperature.append(_arguments((250.0,) * 5, seed + 1, 200.0, 310.0))
        humidity.append(_arguments((0.005,) * 5, seed + 2, 0.0, 0.02))

    def column(rows):
        return np.stack(rows).reshape(len(PIN_LEVELS_PA), 1, cells)

    height, counts = _height_at_surface_pressure(
        PIN_LEVELS_PA, column(heights), column(temperature), column(humidity),
        surface.reshape(1, cells),
        _arguments((288.0,) * 4 + (300.0,), 9, 220.0, 320.0).reshape(1, cells),
        _arguments((280.0, 280.0, 280.0, 200.0, 299.0), 10,
                   200.0, 305.0).reshape(1, cells))
    assert 0 < counts["cells_below_lowest_level"] < cells
    assert _digest(height) == PORTABLE_KERNEL_PINS["height_at_surface_pressure"]


#: Every Cargo workspace that links the ``libm`` crate one engine computes
#: with: the CPU preprocessing library (portable_math's), and the two that
#: build the mapped engine.
LIBM_WORKSPACES = ("tools/grib1_bridge", "tools/rw_wps", "tools/zarr_bridge")


def test_every_libm_the_engines_link_is_the_portable_version():
    # "rust-libm-0.2.16-v1": the crate version is the third field.
    version = pm.IMPLEMENTATION.split("-")[2]
    root = Path(__file__).resolve().parents[1]
    for workspace in LIBM_WORKSPACES:
        lock = tomllib.loads((root / workspace / "Cargo.lock").read_text(
            encoding="utf-8"))
        locked = {package["version"] for package in lock["package"]
                  if package["name"] == "libm"}
        assert locked == {version}, (
            f"{workspace}/Cargo.lock locks libm {sorted(locked)}, but "
            f"portable_math runs libm {version}; the two engines then take "
            "different exp, log, sin and cos")
