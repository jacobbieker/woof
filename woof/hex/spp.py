"""WRF's SPP (``spp_conv``, ``spp_pbl``, ``spp_lsm``) on hex columns.

On the WRF path a member's ``StochasticTimestepHook`` draws one pattern per
selected scheme on the (i, j) grid (WRF's doubly periodic FFT generator) and
hands it to ``PhysicsDriver.bind_spp_patterns`` before each step; the
consumers (GF, MYNN and its surface layer, RUC) read it as WRF's
``pattern_spp_*``.  Hex runs those same consumers behind the generic column
seam (``woof.core.mpas_column_suite.MpasColumnPhysics``), whose driver is
that ``PhysicsDriver``, so all hex has to supply is the pattern.

The pattern is hex's spherical-harmonic AR(1) process
(:class:`woof.hex.stochastic.SphericalPattern`) synthesised at the column
centres, so one law covers global, variable-resolution and limited-area
meshes (and every rank of a partitioned run draws the same global field).
WRF's SPP parameter rows (``StochasticConfig.wrf_reference("spp_*")``,
Registry/registry.stoch) map onto it as:

* ``lengthscale_m`` -> the Gaussian correlation length (correlation
  exp(-r**2/(2 L**2)); WRF's spectrum on its grid is not reproduced);
* ``timescale_s`` -> the AR(1) lag correlation exp(-dt/tau);
* ``stddev`` scales the unit-variance draw and ``stddev * cutoff_sigma``
  clips it, as WRF's ``gridpoint`` does;
* ``vertical_structure = 0``: the same value at every level (WRF's
  ``vertstruc_spp_* = 0``), broadcast to the driver's own pattern shapes;
* seeds: one independent stream per scheme from WRF's ``iseed_spp_*``
  labels (``woof.ensemble.stochastic_seeds``) and the run's member identity.

A limited-area mesh is handled as hex's SPPT and SKEB handle it
(``MeshHarmonics.cell_mask``): the lateral-boundary zone (``bdyMaskCell`` >
0) and the padded garbage column carry a zero pattern, so their parameters
are WRF's unperturbed values.

The pattern is drawn at each step's start time; a repeated call at the same
time (a retried step) returns the same words.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

SCHEMES = ("conv", "pbl", "lsm")
SCHEMA = "gpuwm-hex.spp/v1"


def selected(cfg) -> tuple[str, ...]:
    """The SPP schemes a RunConfig switches on (none for the default run)."""
    return tuple(name for name in SCHEMES if int(getattr(cfg, f"spp_{name}", 0) or 0))


def wrf_configs(names) -> dict:
    """WRF's reference SPP parameter rows, one per selected scheme."""
    from woof.ensemble.stochastic import StochasticConfig as WrfConfig
    return {name: WrfConfig.wrf_reference("spp_" + name) for name in names}


def hex_config(wrf, *, member: int = 0, seed: int = 53):
    """The hex pattern law for one WRF SPP row (see the module docstring)."""
    from woof.ensemble.stochastic_seeds import WRF_SEED_DEFAULTS, process_seed
    from woof.hex.stochastic import StochasticConfig
    return StochasticConfig(member=int(member),
                            seed=process_seed(int(seed), wrf.kind, WRF_SEED_DEFAULTS),
                            length_m=float(wrf.lengthscale_m), time_s=float(wrf.timescale_s))


class HexSppPatterns:
    """One run's SPP patterns at its physics columns.

    ``latitude_rad`` / ``longitude_rad`` are the columns' centres (a non-finite
    entry is a padded column and gets a zero pattern); ``active`` (optional,
    one bool per column) is False in a limited-area mesh's boundary zone.
    ``shapes`` maps each selected scheme to the driver's pattern shape
    ``(levels, 1, columns)`` (``PhysicsDriver._spp_shapes``).
    """

    def __init__(self, names, *, latitude_rad, longitude_rad, shapes, active=None,
                 member: int = 0, seed: int = 53, radius: float | None = None):
        import cupy as cp
        from woof.hex.stochastic import (EARTH_RADIUS_M, PointHarmonics,
                                         SphericalPattern, pattern_degree)
        names = tuple(names)
        if not names or set(names) - set(SCHEMES):
            raise ValueError(f"SPP needs one or more of {SCHEMES}, got {names}")
        lat = np.asarray(latitude_rad, dtype=np.float64).reshape(-1)
        lon = np.asarray(longitude_rad, dtype=np.float64).reshape(-1)
        if lat.shape != lon.shape or not lat.size:
            raise ValueError("SPP column latitude and longitude must be one value per column")
        columns = int(lat.size)
        valid = np.isfinite(lat) & np.isfinite(lon)
        if active is not None:
            active = np.asarray(active, dtype=bool).reshape(-1)
            if active.shape != (columns,):
                raise ValueError(f"SPP active mask must have one value per column ({columns},)")
            valid &= active
        for name in names:
            shape = tuple(shapes[name])
            if len(shape) != 3 or shape[1:] != (1, columns):
                raise ValueError(f"SPP {name} pattern shape {shape} is not (levels, 1, {columns})")
        self.names, self.columns = names, columns
        self.shapes = {name: tuple(shapes[name]) for name in names}
        self.wrf = wrf_configs(names)
        radius = EARTH_RADIUS_M if radius is None else float(radius)
        self.configs = {name: hex_config(self.wrf[name], member=member, seed=seed)
                        for name in names}
        degrees = {name: pattern_degree(c, c.length_m, radius) for name, c in self.configs.items()}
        self.resource = PointHarmonics(np.where(valid, lat, 0.0), np.where(valid, lon, 0.0),
                                       max(degrees.values()), radius)
        self.processes = {name: SphericalPattern(self.resource, self.configs[name],
                                                 degree=degrees[name], channels=1)
                          for name in names}
        self.active = cp.asarray(valid)
        self.values = {name: cp.zeros(columns, dtype=cp.float32) for name in names}
        self.time = None
        self.receipt = {
            "schema": SCHEMA, "schemes": list(names), "columns": columns,
            "masked_columns": int(columns - int(valid.sum())),
            "degree": dict(degrees), "member": int(member), "seed": int(seed),
            "parameters": {name: {"stddev": w.stddev, "cutoff_sigma": w.cutoff_sigma,
                                  "lengthscale_m": w.lengthscale_m,
                                  "timescale_s": w.timescale_s}
                           for name, w in self.wrf.items()},
            "law": "hex SphericalPattern at the column centres (Gaussian correlation, AR(1) in time), "
                   "scaled by WRF's stddev and clipped at stddev*cutoff_sigma, constant in height; "
                   "zero in the boundary zone and padded columns",
        }

    def advance(self, time_seconds: float) -> dict:
        """The (columns,) float32 pattern of each scheme at ``time_seconds``."""
        import cupy as cp
        time_seconds = float(time_seconds)
        if not math.isfinite(time_seconds) or time_seconds < 0:
            raise ValueError("SPP pattern clock must be finite and non-negative")
        if self.time == time_seconds:
            return self.values
        for name, process in self.processes.items():
            wrf = self.wrf[name]
            unit = process.advance(time_seconds)[:, 0]
            std = np.float32(wrf.stddev)
            # WRF's gridpoint: cutoff = float32(stddev * cutoff_sigma).
            limit = np.float32(wrf.stddev * wrf.cutoff_sigma)
            value = cp.clip(unit * std, -limit, limit)
            self.values[name][...] = cp.where(self.active, value, cp.float32(0.0))
        self.time = time_seconds
        return self.values

    def patterns(self, time_seconds: float) -> dict:
        """The driver's ``spp_patterns`` mapping: each value at every level."""
        import cupy as cp
        values = self.advance(time_seconds)
        return {name: cp.broadcast_to(values[name].reshape(1, 1, -1), self.shapes[name])
                for name in self.names}

    def bind(self, driver, time_seconds: float) -> None:
        bind = getattr(driver, "bind_spp_patterns", None)
        if bind is None:
            raise ValueError("SPP needs WRF's physics driver (the generic column seam) "
                             "to consume its patterns")
        bind(self.patterns(time_seconds))

    def snapshot(self) -> dict:
        return {"schema": SCHEMA, "schemes": list(self.names), "time": self.time,
                "processes": {name: p.snapshot() for name, p in self.processes.items()}}

    def restore(self, value: Mapping) -> None:
        if (not isinstance(value, Mapping) or value.get("schema") != SCHEMA
                or list(value.get("schemes", ())) != list(self.names)
                or set(value.get("processes", {})) != set(self.names)):
            raise ValueError("SPP restart state names other schemes than this run")
        for name, process in self.processes.items():
            process.restore(value["processes"][name])
        self.time = None
        clock = value.get("time")
        if clock is not None:
            # Rebuild the carried values from the restored coefficients.
            self.advance(float(clock))


def attach(backend, physics_mesh, *, stochastic=None):
    """Bind an SPP pattern source to a generic-seam backend whose selection
    switches SPP on; return its receipt, or None (and attach nothing) when
    no scheme is selected."""
    seam = getattr(backend, "_seam", None)
    cfg = getattr(seam, "_cfg", None)
    names = () if cfg is None else selected(cfg)
    if not names:
        return None
    driver = getattr(seam, "_driver", None)
    if driver is None or not hasattr(driver, "bind_spp_patterns"):
        raise ValueError("SPP (spp_conv/spp_pbl/spp_lsm) runs on the generic column seam, "
                         "whose WRF physics driver consumes the patterns")
    from woof.config import validate_spp_config
    validate_spp_config(cfg)

    def value(name):
        if hasattr(physics_mesh, name):
            return getattr(physics_mesh, name)
        arrays = getattr(physics_mesh, "arrays", None)
        if isinstance(arrays, Mapping) and name in arrays:
            return arrays[name]
        return None

    columns = int(seam._ncol)
    lat = np.asarray(value("latCell"), dtype=np.float64).reshape(-1)[:columns]
    lon = np.asarray(value("lonCell"), dtype=np.float64).reshape(-1)[:columns]
    if lat.shape != (columns,) or lon.shape != (columns,):
        raise ValueError(f"SPP needs a centre for each of the {columns} physics columns")
    active = np.ones(columns, dtype=bool)
    mask = value("bdyMaskCell")
    if mask is not None:
        mask = np.asarray(mask).reshape(-1)[:columns]
        active[:mask.size] &= mask == 0
    # The padded garbage column of a limited-area mesh
    # (cuda_regional_forecast_v841.PaddedRegionalHostMesh): not a cell.
    garbage = getattr(physics_mesh, "garbage_cell", None)
    if garbage is not None and 0 <= int(garbage) < columns:
        active[int(garbage)] = False
    from woof.hex.stochastic import StochasticConfig
    config = StochasticConfig.parse(stochastic)
    radius = None
    attrs = getattr(physics_mesh, "attrs", None)
    if isinstance(attrs, Mapping) and attrs.get("sphere_radius") is not None:
        radius = float(attrs["sphere_radius"])
    source = HexSppPatterns(names, latitude_rad=lat, longitude_rad=lon,
                            shapes={name: driver._spp_shapes[name] for name in names},
                            active=active, member=config.member, seed=config.seed,
                            radius=radius)
    backend.spp_patterns = source
    return source.receipt
