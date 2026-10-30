"""RRTMG per-band boundary fluxes on the MPAS suite route (rrtmg_spectral_output).

A WOOF-only output; MPAS-A v8.4.2 has none.  The &radiation namelist option
``config_rrtmg_spectral_output`` = 'none' (default) | 'bands' | 'gpoints'
(:func:`resolve`) adds, beside each RRTMG boundary flux, its spectral split:
``<name>_band`` on nSWBands = 14 (shortwave) or nLWBands = 16 (longwave), or
``sw<name>_gpt`` on nSWGpoints = 112 under 'gpoints' (longwave stays per band:
a g-point is a quadrature point, not a wavelength interval).  The names and
quantities are the WRF option's (woof.core.physics_inventory SW_/LW_SPECTRAL_FLUXES);
the slabs are the same engine kernels (kernels/rrtmg_sw_spectral.cuh,
rrtmg_lw_zbatched.cu rlw_rtrn_band_rows), whose words against compiled WRF are
held by tests/test_rrtmg_spectral_output.py.

The fields live in diag_physics as [band, nCells] (Registry ``nSWBands nCells
Time``), gated by packages that only an active option turns on, so a run without
it allocates, reads and writes nothing new and the pinned Registry is untouched:
:func:`extend_registry` adds the dimensions, fields and diagnostics-stream
members to the run's parsed Registry tree (pool_v842.RegistryLayout), and
:data:`FIELDS` gives the stream writer their definitions (streams_v842).
Broadband words do not depend on the option (the spectral launches only read
the solver's buffers).
"""
from __future__ import annotations

from typing import Any, Mapping

from woof.core.physics_inventory import LW_SPECTRAL_FLUXES, SW_SPECTRAL_FLUXES

#: The WOOF-only namelist group and its one option.
GROUP = "radiation"
OPTION = "config_rrtmg_spectral_output"
MODES = ("none", "bands", "gpoints")

#: Spectral axes; RRTMG's shortwave bands 16..29 and g-points, longwave bands 1..16.
DIMENSIONS = {"nSWBands": 14, "nSWGpoints": 112, "nLWBands": 16}

#: Registry packages, one per allocated family.
PACKAGES = {("sw", "bands"): "rrtmg_spectral_sw_bands",
            ("sw", "gpoints"): "rrtmg_spectral_sw_gpoints",
            ("lw", "bands"): "rrtmg_spectral_lw_bands"}

#: The MPAS boundary-flux descriptions (Registry.xml:3125-3290); swddirc has no
#: MPAS broadband field and takes WRF's SWDDIRC meaning.
_DESCRIPTIONS = {
    "swupt": "all-sky upward top-of-atmosphere shortwave radiation flux",
    "swuptc": "clear-sky upward top-of-atmosphere shortwave radiation flux",
    "swdnt": "all-sky downward top-of-atmosphere shortwave radiation flux",
    "swdntc": "clear-sky downward top-of-atmosphere shortwave radiation flux",
    "swupb": "all-sky upward surface shortwave radiation flux",
    "swupbc": "clear-sky upward surface shortwave radiation flux",
    "swdnb": "all-sky downward surface shortwave radiation flux",
    "swdnbc": "clear-sky downward surface shortwave radiation flux",
    "swddir": "shortwave surface downward direct irradiance",
    "swddirc": "clear-sky shortwave surface downward direct irradiance",
    "lwupt": "all-sky upward top-of-the-atmosphere longwave radiation flux",
    "lwuptc": "clear-sky upward top-of-the-atmosphere longwave radiation flux",
    "lwupb": "all-sky upward surface longwave radiation flux",
    "lwupbc": "clear-sky upward surface longwave radiation flux",
    "lwdnb": "all-sky downward surface longwave radiation flux",
    "lwdnbc": "clear-sky downward surface longwave radiation flux",
}


def _families():
    """(side, mode, suffix, dimension, table) for each allocatable family."""
    return (("sw", "bands", "_band", "nSWBands", SW_SPECTRAL_FLUXES),
            ("sw", "gpoints", "_gpt", "nSWGpoints", SW_SPECTRAL_FLUXES),
            ("lw", "bands", "_band", "nLWBands", LW_SPECTRAL_FLUXES))


def _spectral_attributes(dimension: str) -> dict[str, Any]:
    """Band edges, cm-1, as the WRF history writes them (woof.io.wrfout)."""
    if dimension == "nLWBands":
        from woof.io.wrf_output_schema import RRTMG_LW_WAVENUMBERS
        low, high = zip(*RRTMG_LW_WAVENUMBERS)
        extra = {}
    else:
        from woof.core.rrtmg_sw import band_wavenumbers, gpoint_bands
        low, high = band_wavenumbers()
        extra = ({"band_of_gpoint": {"dtype": "i4", "values": list(gpoint_bands())}}
                 if dimension == "nSWGpoints" else {})
    return {"wavenumber_low": {"dtype": "f4", "values": [float(v) for v in low]},
            "wavenumber_high": {"dtype": "f4", "values": [float(v) for v in high]},
            "wavenumber_units": "cm-1", **extra}


def _definitions() -> dict[str, dict[str, Any]]:
    out = {}
    for side, mode, suffix, dimension, table in _families():
        what = "RRTMG band" if suffix == "_band" else "RRTMG g-point"
        for base in table:
            out[base + suffix] = {
                "type": "real", "dimensions": [dimension, "nCells", "Time"],
                "attrs": {"units": "W m^{-2}", "long_name": f"{_DESCRIPTIONS[base]}, per {what}"},
                "gates": [PACKAGES[side, mode]], "flux": base, "dimension": dimension}
    return out


#: field -> its Registry row (streams_v842's field shape plus the broadband
#: ``flux`` it splits and its spectral ``dimension``); the band edges are
#: added by :func:`field_definition`.
FIELDS: dict[str, dict[str, Any]] = _definitions()


def field_definition(name: str) -> dict[str, Any]:
    """streams_v842.field_definition's row for a spectral field: Registry-style
    units/long_name, then the band edges the WRF history carries."""
    row = FIELDS[name]
    return {"type": row["type"], "dimensions": list(row["dimensions"]),
            "attrs": {**row["attrs"], **_spectral_attributes(row["dimension"])},
            "gates": list(row["gates"])}


def resolve(groups: Mapping[str, Mapping[str, Any]]) -> str:
    """The option from the namelist's &radiation group ('none' when absent)."""
    supplied = {str(k).casefold(): v for k, v in groups.get(GROUP, {}).items()}
    unknown = sorted(set(supplied) - {OPTION})
    if unknown:
        raise ValueError(f"&{GROUP} has no option {unknown[0]!r}; its one (WOOF) option is {OPTION}")
    value = supplied.get(OPTION, "none")
    if not isinstance(value, str) or value.strip() not in MODES:
        raise ValueError(f"{OPTION} must be 'none', 'bands' or 'gpoints', not {value!r}")
    return value.strip()


def sides(config: Any) -> tuple[str, ...]:
    """The radiation sides that produce spectral fields in this run: those running RRTMG."""
    mode = str(getattr(config, "spectral_output", "none") or "none")
    if mode == "none":
        return ()
    selectors = config.selectors
    active = getattr(config, "active", None)
    out = []
    for side, slot, scheme in (("sw", "radiation_sw", "rrtmg_sw"), ("lw", "radiation_lw", "rrtmg_lw")):
        if selectors.get(slot) == scheme and (active is None or active(slot)):
            out.append(side)
    return tuple(out)


def packages(config: Any) -> set[str]:
    """The Registry packages an active option turns on (pool_v842.active_packages)."""
    mode = str(getattr(config, "spectral_output", "none") or "none")
    return {PACKAGES[side, "bands" if side == "lw" else mode] for side in sides(config)}


def slot_fields(side: str, mode: str) -> dict[str, tuple[int, str]]:
    """field -> (quantity index, level 't'|'b') written by one side's slot under ``mode``."""
    if mode == "none":
        return {}
    suffix = "_gpt" if side == "sw" and mode == "gpoints" else "_band"
    table = SW_SPECTRAL_FLUXES if side == "sw" else LW_SPECTRAL_FLUXES
    return {base + suffix: value for base, value in table.items()}


def boundary_slabs(slab, fields: Mapping[str, tuple[int, str]]) -> dict[str, Any]:
    """(column, count) slabs of the engine's (quantity, column, level, count) slab at the
    surface (level 0) and top (last level), the WRF adapter's selection (rrtmg_legacy)."""
    return {name: slab[q, :, -1 if level == "t" else 0, :] for name, (q, level) in fields.items()}


def extend_registry(root, active: set[str]) -> None:
    """Add the spectral dimensions, diag_physics fields and diagnostics-stream members
    to a parsed Registry tree, for the packages in ``active`` only.

    MPAS Registry.xml:8-55 (dims), 2502 (diag_physics), 1005-1128 (diagnostics).
    An empty ``active`` leaves the tree exactly as parsed.
    """
    import xml.etree.ElementTree as ET
    if not active:
        return
    wanted = {name: row for name, row in FIELDS.items() if row["gates"][0] in active}
    dims = root.find("./dims")
    for name in dict.fromkeys(row["dimension"] for row in wanted.values()):
        if dims.find(f"./dim[@name='{name}']") is None:
            ET.SubElement(dims, "dim", {"name": name, "definition": str(DIMENSIONS[name])})
    struct = root.find("./var_struct[@name='diag_physics']")
    stream = root.find("./streams/stream[@name='diagnostics']")
    for name, row in wanted.items():
        if struct.find(f"./var[@name='{name}']") is None:
            ET.SubElement(struct, "var", {
                "name": name, "type": "real", "dimensions": " ".join(row["dimensions"]),
                "units": row["attrs"]["units"], "description": row["attrs"]["long_name"],
                "default_value": "0.0", "packages": row["gates"][0]})
        if stream is not None and stream.find(f"./var[@name='{name}']") is None:
            ET.SubElement(stream, "var", {"name": name, "packages": row["gates"][0]})
