"""``woof hex ice run``: MALI v8.4.2's forward mode on an MPAS land-ice mesh.

Reads a MALI namelist (``namelist.landice``) and an MPAS land-ice input file
(a mesh with its fields, as MALI's ``input`` stream reads it), runs
:func:`woof.hex.mpas_ice.forward_v842.run` (``li_simulation_clock_init``,
``li_core_init``, the initial solve and the time-step loop, every piece word
for word against the v8.4.2 oracle) and writes the final state.

- The namelist: every option starts at its v8.4.2 Registry default
  (``data/registry-options*.json``: ``Registry.xml`` and the included
  ``Registry_subglacial_hydro.xml``); the namelist overrides them. A name
  that is not a v8.4.2 land-ice option is refused by name, as MPAS's own
  namelist read stops on it. Options the port does not run are refused by
  :func:`woof.hex.mpas_ice.forward_v842.check_config`.
- The input file: every field the forward mode carries is read by its
  Registry name from the file's ``Time`` record at ``config_start_time``
  (the only record when the file has no ``xtime``); a field the file does
  not carry takes its Registry ``default_value``, as MPAS allocates it.
  ``on_a_sphere`` and ``sphere_radius`` are the file's global attributes.
  ``edgeNormalVectors`` and ``coeffs_reconstruct`` must be in the file:
  MPAS's vector reconstruction initialisation is not ported word for word
  (the oracle takes both from its input), so a file without them is refused
  by name. MPAS writes both to its output and restart streams.
- The output: ``output.nc`` (the final state of every field, one ``Time``
  record with ``xtime``) and ``restart.npz``
  (:func:`woof.hex.mpas_ice.forward_v842.write_restart`, bitwise); a run
  started from ``--restart`` continues as if never stopped.

At RKIND 8 the strain-rate angle ``eTheta`` (glibc's binary64 ``atan``) is
left out of the run and the output unless ``WOOF_GLIBC239_LIBM64=1``.

One block owns every cell (``nCellsSolve = nCells``); MPAS's multi-block
decomposition, PIO streams and restart stream are refused by name.
"""
from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path

import numpy as np

from woof.hex.errors import ConfigurationRefusal

from . import forward_v842, rkind

OPTION_FILES = ("registry-options.json", "registry-options-subglacial-hydro.json")
#: the fields the forward mode carries with a non-zero Registry default_value
FIELD_DEFAULTS = {"temperature": 273.15, "surfaceAirTemperature": 273.15, "surfaceTemperature": 273.15,
                  "basalTemperature": 273.15, "basalHeatFlux": 0.06}
TEXT_DEFAULTS = {"xtime": "", "simulationStartTime": "no_date_available"}
#: MPAS's vector reconstruction initialisation: read, or computed on the host
RECONSTRUCTION = ("edgeNormalVectors", "coeffs_reconstruct")
POOLS = ("mesh", "geometry", "thermal", "velocity", "hydro", "scratch")
STRLEN = 64


def _options() -> dict[str, dict]:
    table = {}
    for name in OPTION_FILES:
        data = json.loads(files("woof.hex.mpas_ice").joinpath("data", name).read_text(encoding="utf-8"))
        table.update({o["name"]: o for o in data["options"]})
    return table


def _literal(text: str, kind: str):
    """A Registry ``default_value`` (Fortran literal) as Python."""
    text = text.strip()
    if kind == "logical":
        return text.strip(".").lower() in ("true", "t")
    if kind == "integer":
        return int(text)
    if kind == "real":
        return float(text.lower().replace("d", "e"))
    return text.strip("'\"")


def _coerce(name: str, value, kind: str):
    if kind == "logical" and isinstance(value, bool):
        return value
    if kind == "integer" and isinstance(value, int) and not isinstance(value, bool):
        return value
    if kind == "real" and isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if kind == "character" and isinstance(value, str):
        return value
    raise ConfigurationRefusal(knob=name, value=value, reason=f"not a {kind} value",
                               declaration=f"{name} is a v8.4.2 {kind} option")


def default_config() -> dict:
    """Every v8.4.2 land-ice option at its Registry default."""
    return {name: _literal(o["default"], o["type"]) for name, o in _options().items()}


def read_namelist(path) -> dict:
    """The configuration: the Registry defaults with ``path``'s settings."""
    from woof.fortran_namelist import parse_namelist
    options = _options()
    spelled = {name.lower(): name for name in options}   # Fortran names are case blind
    config = default_config()
    for group, table in parse_namelist(path).items():
        for name, values in table.items():
            name = spelled.get(name, name)
            if name not in options:
                raise ConfigurationRefusal(
                    knob=f"&{group}/{name}", value=values, reason="not a MALI v8.4.2 namelist option",
                    declaration="woof hex ice options lists the land-ice core's options")
            if len(values) != 1:
                raise ConfigurationRefusal(knob=name, value=values, reason="a land-ice option takes one value",
                                           declaration=f"{name} = <one value>")
            config[name] = _coerce(name, values[0], options[name]["type"])
    return config


def _record(var, index: int | None) -> np.ndarray:
    if var.dimensions and var.dimensions[0] == "Time":
        return np.asarray(var[index if index is not None else 0])
    return np.asarray(var[...])


def _text(value) -> str:
    raw = np.asarray(value)
    if raw.dtype.kind in "SU" and raw.ndim:
        raw = b"".join(x if isinstance(x, bytes) else str(x).encode() for x in raw.reshape(-1))
    elif raw.dtype.kind == "S":
        raw = raw.tobytes()
    else:
        raw = str(raw).encode()
    return raw.decode("ascii", "replace").rstrip("\x00 ")


def _time_index(ds, config: dict) -> int | None:
    n = len(ds.dimensions["Time"]) if "Time" in ds.dimensions else 0
    if n <= 1:
        return 0 if n else None
    if "xtime" not in ds.variables:
        raise ConfigurationRefusal(knob="input.Time", value=n, reason="several Time records and no xtime",
                                   declaration="an input file with one Time record, or with xtime")
    stamps = [_text(ds.variables["xtime"][i]) for i in range(n)]
    start = config["config_start_time"]
    if start not in stamps:
        raise ConfigurationRefusal(knob="config_start_time", value=start,
                                   reason="no input Time record has this xtime",
                                   declaration="config_start_time = one of " + ", ".join(stamps))
    return stamps.index(start)


def _dims(ds) -> dict[str, int]:
    need = ("nCells", "nEdges", "nVertices", "nVertLevels", "maxEdges", "maxEdges2", "vertexDegree")
    missing = [d for d in need if d not in ds.dimensions]
    if missing:
        raise ConfigurationRefusal(knob="input.dimensions", value=missing, reason="the input is not an MPAS mesh",
                                   declaration="an MPAS land-ice input with " + ", ".join(need))
    dims = {d: len(ds.dimensions[d]) for d in need}
    if dims["vertexDegree"] != 3:
        raise ConfigurationRefusal(knob="vertexDegree", value=dims["vertexDegree"],
                                   reason="MALI runs on Voronoi meshes (triangles at vertices)",
                                   declaration="vertexDegree = 3")
    dims["nVertInterfaces"] = dims["nVertLevels"] + 1
    return dims


def _attribute(ds, name: str, default=None):
    return ds.getncattr(name) if name in ds.ncattrs() else default


def _truth(value) -> bool:
    return str(value).strip().upper() in ("YES", "TRUE", "T", "1")


def read_input(path, *, config: dict, precision: str) -> tuple[dict, dict]:
    """The host pools (numpy arrays, text for the time stamps) and the
    mesh's configuration (``on_a_sphere``, ``sphere_radius``) from ``path``."""
    import netCDF4

    from woof.verify import mali_forward_oracle as fo
    rkind.require(precision)
    real = rkind.PRECISIONS[precision]
    with netCDF4.Dataset(Path(path)) as ds:
        ds.set_auto_mask(False)
        dims = _dims(ds)
        index = _time_index(ds, config)
        mesh_config = {"on_a_sphere": _truth(_attribute(ds, "on_a_sphere", "NO")),
                       "sphere_radius": float(_attribute(ds, "sphere_radius", 0.0))}
        pools: dict[str, dict] = {p: {} for p in POOLS}
        for pool, name, kind, shape in fo.fields_for(config):
            if kind == "d":
                continue
            size = tuple(dims[s] if isinstance(s, str) else s for s in shape)
            source = name.split("@")[0]
            if kind == "c":
                pools[pool][name] = (_text(_record(ds.variables[name], index)) if name in ds.variables
                                     else TEXT_DEFAULTS[name])
                continue
            dtype = np.int32 if kind == "i" else real
            if "@" in name or pool == "scratch" or source not in ds.variables:
                value = np.full(size, FIELD_DEFAULTS.get(source, 0), dtype)
            else:
                value = _record(ds.variables[source], index)
                if value.shape != size:
                    raise ConfigurationRefusal(knob=f"input.{source}", value=value.shape,
                                               reason=f"MALI's {source} is {size}",
                                               declaration=f"{source} with dimensions {shape}")
                value = np.ascontiguousarray(value, dtype)
            pools[pool][name] = value
        missing = [n for n in RECONSTRUCTION if n not in ds.variables]
        simple_missing = [n for n in ("xEdge", "yEdge") if n not in ds.variables]
    if precision == "r8" and not rkind.trig64():
        # the strain-rate angle calls glibc's binary64 atan (LGPL): left out at
        # RKIND 8 unless WOOF_GLIBC239_LIBM64=1
        del pools["velocity"]["eTheta"]
    if missing:
        raise ConfigurationRefusal(
            knob="input." + missing[0], value="absent",
            reason=("MPAS's vector reconstruction initialisation is not ported word for word; the oracle "
                    "takes edgeNormalVectors and coeffs_reconstruct from its input"),
            declaration="an input file carrying edgeNormalVectors and coeffs_reconstruct (MPAS writes both "
                        "to its output and restart streams)")
    if config.get("config_velocity_solver") == "simple" and simple_missing:
        raise ConfigurationRefusal(
            knob="input." + simple_missing[0], value="absent",
            reason="config_velocity_solver = 'simple' forms its prescribed velocity from the edge points",
            declaration="an MPAS mesh with xEdge and yEdge")
    return pools, mesh_config


def device_pools(pools: dict) -> dict:
    """Host pools as the port takes them (device arrays; scalars as one word)."""
    import cupy as cp
    return {pool: {name: value if isinstance(value, str)
                   else cp.asarray(np.ascontiguousarray(value).reshape(np.shape(value) or (1,)))
                   for name, value in fields.items()}
            for pool, fields in pools.items()}


def write_output(path, *, pools: dict, clock: forward_v842.Clock, config: dict | None = None) -> None:
    """The state of every non-scratch field, one ``Time`` record, by Registry name."""
    import cupy as cp
    import netCDF4

    from woof.verify import mali_forward_oracle as fo
    g = pools["geometry"]
    dims = {"nCells": g["thickness"].shape[0], "nEdges": pools["mesh"]["dvEdge"].shape[0],
            "nVertices": g["vertexMask"].shape[0], "nVertLevels": g["layerThickness"].shape[1],
            "maxEdges": pools["mesh"]["edgesOnCell"].shape[1],
            "maxEdges2": pools["mesh"]["edgesOnEdge"].shape[1]}
    dims["nVertInterfaces"] = dims["nVertLevels"] + 1
    with netCDF4.Dataset(Path(path), "w") as ds:
        ds.createDimension("Time", None)
        ds.createDimension("StrLen", STRLEN)
        for name, size in dims.items():
            ds.createDimension(name, size)
        ds.createDimension("vertexDegree", 3)
        for extent in (2, 3):
            ds.createDimension(f"R{extent}", extent)
        ds.setncattr("model", "woof hex ice (MALI v8.4.2 forward mode)")
        ds.setncattr("clock_now", clock.time_stamp)
        for pool, name, kind, shape in fo.fields_for(config or {}):
            if kind == "d" or pool == "scratch" or "@" in name or name not in pools.get(pool, {}):
                continue
            value = pools[pool][name]
            if kind == "c":
                var = ds.createVariable(name, "S1", ("Time", "StrLen"))
                var[0] = np.frombuffer(value.ljust(STRLEN)[:STRLEN].encode("ascii"), "S1")
                continue
            host = cp.asnumpy(value)
            axes = tuple(s if isinstance(s, str) else f"R{s}" for s in shape)
            host = host.reshape(tuple(dims[a] if a in dims else int(a[1:]) for a in axes)) if axes else host[:1]
            var = ds.createVariable(name, host.dtype, ("Time",) + axes)
            var[0] = host if axes else host[0]


def run(*, namelist, input=None, out, precision: str | None = None, restart=None, n_steps: int | None = None):
    """A run from ``input`` (a cold start, at ``precision``, 'r4' by
    default) or from ``restart`` (at its stored precision, which
    ``precision`` must name if given); writes ``out/output.nc`` and
    ``out/restart.npz``. Returns the clock."""
    out = Path(out)
    config = read_namelist(namelist)
    if restart is not None:
        config.update(_restart_mesh_config(restart))
        pools, clock = forward_v842.read_restart(restart, config=config, precision=precision)
        forward_v842.advance(pools=pools, config=config, clock=clock, n_steps=n_steps)
    else:
        if input is None:
            raise ValueError("a cold start needs --input")
        host, mesh_config = read_input(input, config=config, precision=precision or "r4")
        config.update(mesh_config)
        pools = device_pools(host)
        clock = forward_v842.cold_start(pools=pools, config=config)
        forward_v842.advance(pools=pools, config=config, clock=clock, n_steps=n_steps)
    out.mkdir(parents=True, exist_ok=True)
    write_output(out / "output.nc", pools=pools, clock=clock, config=config)
    forward_v842.write_restart(out / "restart.npz", pools=pools, clock=clock)
    _save_mesh_config(out / "restart.npz", config)
    return clock


def _mesh_config_path(restart) -> Path:
    return Path(str(restart) + ".mesh.json")


def _save_mesh_config(restart: Path, config: dict) -> None:
    _mesh_config_path(restart).write_text(json.dumps(
        {"on_a_sphere": bool(config.get("on_a_sphere", False)),
         "sphere_radius": float(config.get("sphere_radius", 0.0))}) + "\n", encoding="utf-8")


def _restart_mesh_config(restart) -> dict:
    path = _mesh_config_path(restart)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def main(args) -> int:
    clock = run(namelist=args.namelist, input=args.input, out=args.out, precision=args.precision,
                restart=args.restart, n_steps=args.steps)
    print(json.dumps({"model": "ice", "clock": clock.time_stamp, "out": str(args.out)}))
    return 0
