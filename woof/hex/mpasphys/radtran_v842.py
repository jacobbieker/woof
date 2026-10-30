"""woof.radtran spectral diagnostic on the MPAS suite route (WOOF-only &radtran group).

MPAS-A v8.4.2 has no such output.  The namelist group ``&radtran`` (WOOF's own,
never a native config) carries

    config_radtran_wavelengths      = '500 760.5'  wavelengths [nm], quoted list
    config_radtran_umu              = '0.6 -0.8'   radiance directions (cosines)
    config_radtran_phi              = '0 90'       relative azimuths [deg]
    config_radtran_mol_abs_param    = 'reptran coarse'
    config_radtran_solver           = 'disort' | 'twostr'
    config_radtran_streams          = 0            (6, or 16 with radiances)
    config_radtran_pseudospherical  = .false.

(:func:`resolve`).  With wavelengths set, the Registry gains (only in this run's
parsed tree, :func:`extend_registry`) the dimensions nRadtranWavelengths and
nRadtranDirections, the diag_physics fields rt_edir, rt_edn, rt_eup (surface),
rt_eup_toa [nRadtranWavelengths, nCells] in W m-2 nm-1, rt_rad
[nRadtranWavelengths, nRadtranDirections, nCells] in W m-2 nm-1 sr-1 and the
axis fields rt_wavelength, rt_band_min, rt_band_max [nRadtranWavelengths] and
rt_umu, rt_phi [nRadtranDirections], all members of the diagnostics stream and
gated by the package ``radtran_diag``.

They are computed when a stream carrying them is written
(:class:`RadtranOutput`, called by streams_output_v842.PhysicsStreams.write):
the cadence is that stream's output interval.  Columns are the model's own:
zgrid interfaces, diag pressure/theta/exner/rho, the scalars qv/qc/qi,
re_cloud/re_ice where the microphysics diagnoses them, sfc_albedo; the solar
zenith angle at the output time (woof.core.solar).  Interface pressures are
interpolated log-linearly in height between mass levels and extrapolated
hydrostatically to the surface and model top (documented choice).  No other
field is read for writing or changed.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

GROUP = "radtran"
PREFIX = "config_radtran_"
OPTIONS = ("config_radtran_wavelengths", "config_radtran_umu", "config_radtran_phi",
           "config_radtran_mol_abs_param", "config_radtran_solver", "config_radtran_streams",
           "config_radtran_pseudospherical")
PACKAGE = "radtran_diag"
FLUX_FIELDS = ("rt_edir", "rt_edn", "rt_eup", "rt_eup_toa")
WAVELENGTH_FIELDS = ("rt_wavelength", "rt_band_min", "rt_band_max")
DIRECTION_FIELDS = ("rt_umu", "rt_phi")
W_DIM, D_DIM = "nRadtranWavelengths", "nRadtranDirections"

_LONG = {
    "rt_edir": ("W m^{-2} nm^{-1}", "direct horizontal spectral irradiance at the surface, REPTRAN band mean"),
    "rt_edn": ("W m^{-2} nm^{-1}", "diffuse downward spectral irradiance at the surface, REPTRAN band mean"),
    "rt_eup": ("W m^{-2} nm^{-1}", "upward spectral irradiance at the surface, REPTRAN band mean"),
    "rt_eup_toa": ("W m^{-2} nm^{-1}", "upward spectral irradiance at the top of the atmosphere, band mean"),
    "rt_rad": ("W m^{-2} nm^{-1} sr^{-1}", "spectral radiance: upwelling at the top of the atmosphere "
               "(umu > 0), downwelling at the surface (umu < 0), REPTRAN band mean"),
    "rt_wavelength": ("nm", "requested wavelength of each radtran entry"),
    "rt_band_min": ("nm", "lower edge of the REPTRAN band containing the wavelength"),
    "rt_band_max": ("nm", "upper edge of the REPTRAN band containing the wavelength"),
    "rt_umu": ("1", "cosine of the polar angle of each radiance direction (> 0 upwelling)"),
    "rt_phi": ("degree", "azimuth of each radiance direction relative to the sun"),
}


def _dims(name):
    if name in FLUX_FIELDS:
        return [W_DIM, "nCells", "Time"]
    if name == "rt_rad":
        return [W_DIM, D_DIM, "nCells", "Time"]
    if name in WAVELENGTH_FIELDS:
        return [W_DIM, "Time"]
    return [D_DIM, "Time"]


#: field -> streams_v842 row (Registry-style units/long_name, dimensions, gate)
FIELDS: dict[str, dict[str, Any]] = {
    name: {"type": "real", "dimensions": _dims(name),
           "attrs": {"units": units, "long_name": long_name}, "gates": [PACKAGE]}
    for name, (units, long_name) in _LONG.items()}


def field_definition(name: str) -> dict[str, Any]:
    row = FIELDS[name]
    return {"type": row["type"], "dimensions": list(row["dimensions"]), "attrs": dict(row["attrs"]),
            "gates": list(row["gates"])}


def _floats(key, value):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return (float(value),)
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a quoted list of numbers, not {value!r}")
    try:
        return tuple(float(v) for v in value.replace(",", " ").split())
    except ValueError:
        raise ValueError(f"{key} must be a quoted list of numbers, not {value!r}") from None


def resolve(groups: Mapping[str, Mapping[str, Any]]):
    """The &radtran group as a woof.radtran.model.Diagnostic, or None when off."""
    from woof.radtran import model
    supplied = {str(k).casefold(): v for k, v in groups.get(GROUP, {}).items()}
    unknown = sorted(set(supplied) - set(OPTIONS))
    if unknown:
        raise ValueError(f"&{GROUP} has no option {unknown[0]!r}; its (WOOF) options are {', '.join(OPTIONS)}")
    waves = _floats("config_radtran_wavelengths", supplied.get("config_radtran_wavelengths", ""))
    if not waves:
        if len(supplied) > 0:
            raise ValueError("&radtran options need config_radtran_wavelengths")
        return None
    streams = supplied.get("config_radtran_streams", 0)
    pseudo = supplied.get("config_radtran_pseudospherical", False)
    if not isinstance(streams, int) or isinstance(streams, bool):
        raise ValueError(f"config_radtran_streams must be an integer, not {streams!r}")
    if not isinstance(pseudo, bool):
        raise ValueError(f"config_radtran_pseudospherical must be logical, not {pseudo!r}")
    diag = model.Diagnostic(
        wavelengths=waves, interval_s=1.0,
        umu=_floats("config_radtran_umu", supplied.get("config_radtran_umu", "")),
        phi=_floats("config_radtran_phi", supplied.get("config_radtran_phi", "")),
        mol_abs_param=str(supplied.get("config_radtran_mol_abs_param", "reptran coarse")).strip(),
        solver=str(supplied.get("config_radtran_solver", "disort")).strip(),
        streams=int(streams), pseudospherical=pseudo)
    model.validate(diag)
    return diag


def dimensions(diag) -> dict[str, int]:
    out = {W_DIM: len(diag.wavelengths)}
    if diag.umu:
        out[D_DIM] = len(diag.directions())
    return out


def field_names(diag) -> tuple[str, ...]:
    names = (*FLUX_FIELDS, *WAVELENGTH_FIELDS)
    if diag.umu:
        names += ("rt_rad", *DIRECTION_FIELDS)
    return names


def packages(config: Any) -> set[str]:
    return {PACKAGE} if getattr(config, "radtran", None) is not None else set()


def extend_registry(root, config: Any) -> None:
    """Add the radtran dimensions, diag_physics fields and diagnostics-stream
    members to a parsed Registry tree, only when the run carries &radtran."""
    import xml.etree.ElementTree as ET
    diag = getattr(config, "radtran", None)
    if diag is None:
        return
    dims = root.find("./dims")
    for name, size in dimensions(diag).items():
        if dims.find(f"./dim[@name='{name}']") is None:
            ET.SubElement(dims, "dim", {"name": name, "definition": str(int(size))})
    struct = root.find("./var_struct[@name='diag_physics']")
    stream = root.find("./streams/stream[@name='diagnostics']")
    for name in field_names(diag):
        row = FIELDS[name]
        if struct.find(f"./var[@name='{name}']") is None:
            ET.SubElement(struct, "var", {
                "name": name, "type": "real", "dimensions": " ".join(row["dimensions"]),
                "units": row["attrs"]["units"], "description": row["attrs"]["long_name"],
                "default_value": "0.0", "packages": PACKAGE})
        if stream is not None and stream.find(f"./var[@name='{name}']") is None:
            ET.SubElement(stream, "var", {"name": name, "packages": PACKAGE})


# ---------------------------------------------------------------------------
# The computation at output time.
# ---------------------------------------------------------------------------

G = 9.80665
RD = 287.04


def interface_pressure(zgrid, p_mass, t_mass, qv_mass):
    """(nz+1, n) interface pressures from (nz, n) mass-level ones [Pa]:
    log-linear in height inside, hydrostatic (virtual temperature of the
    adjacent layer) to the surface and model top."""
    zm = 0.5 * (zgrid[:-1] + zgrid[1:])
    lp = np.log(p_mass)
    out = np.empty_like(zgrid)
    w = (zgrid[1:-1] - zm[:-1]) / (zm[1:] - zm[:-1])
    out[1:-1] = np.exp(lp[:-1] + w * (lp[1:] - lp[:-1]))
    tv = t_mass * (1.0 + 0.608 * qv_mass)
    out[0] = p_mass[0] * np.exp(G * (zm[0] - zgrid[0]) / (RD * tv[0]))
    out[-1] = p_mass[-1] * np.exp(-G * (zgrid[-1] - zm[-1]) / (RD * tv[-1]))
    return out


class RadtranOutput:
    """Fills the rt_* fields of the session's diag_physics pool before a write."""

    def __init__(self, session):
        self.session = session
        self.diag = session.config.radtran
        self.calls = 0
        self.last_time = None

    def wanted(self, streams, packages) -> bool:
        names = set(field_names(self.diag))
        return any(names.intersection(s.fields(packages, streams=streams)) for s in streams)

    def _host(self, pool, name):
        value = self.session.field(pool, name)
        return np.asarray(value.get() if hasattr(value, "get") else value, dtype=np.float64)

    def _put(self, name, value):
        dtype = self.session.native._fields["diag_physics", name][0]
        self.session.put("diag_physics", name, np.ascontiguousarray(value, dtype=dtype))

    def compute(self, when) -> None:
        from woof.radtran import model
        layout = self.session.layout
        zgrid = self._host("mesh", "zgrid")
        p = self._host("diag", "pressure")
        theta = self._host("diag", "theta")
        exner = self._host("diag", "exner")
        rho = self._host("diag", "rho")
        scal = self.session.field("state", "scalars")
        scal = np.asarray(scal.get() if hasattr(scal, "get") else scal, dtype=np.float64)
        q = lambda key: (scal[layout.indices["state", key] - 1]
                         if ("state", key) in layout.indices and layout.indices["state", key] > 0 else None)
        qv, qc, qi = q("index_qv"), q("index_qc"), q("index_qi")
        n = p.shape[-1]
        zgrid = zgrid[:, :n]
        t = theta * exner
        pw = interface_pressure(zgrid, p, t, np.maximum(qv, 0.0))
        lat = np.degrees(self._host("mesh", "latCell")[:n])
        lon = np.degrees(self._host("mesh", "lonCell")[:n])
        albedo = np.clip(self._host("diag_physics", "sfc_albedo")[:n], 0.0, 1.0)

        def eff(name):
            try:
                v = self._host("diag_physics", name)
            except (KeyError, ValueError):
                return None
            # MPAS carries effective radii in metres (Registry re_cloud/re_ice)
            return v[:, :n].T * 1e6 if np.any(v > 0) else None

        mu0 = solar_cosine(when, lat, lon)
        out = model.compute(self.diag, z_w_m=zgrid.T, p_w_pa=pw.T, t_mass=t.T, qv_mass=qv.T,
                            umu0=mu0, albedo=albedo,
                            qc_mass=None if qc is None else qc.T, qi_mass=None if qi is None else qi.T,
                            rho_dry=rho.T, re_cloud_um=eff("re_cloud"), re_ice_um=eff("re_ice"),
                            day_of_year=int(when.timetuple().tm_yday))
        for name in FLUX_FIELDS:
            self._put(name, out[name[3:]])
        self._put("rt_wavelength", out["wavelength"])
        self._put("rt_band_min", out["band_min"])
        self._put("rt_band_max", out["band_max"])
        if "rad" in out:
            self._put("rt_rad", out["rad"])
            dirs = self.diag.directions()
            self._put("rt_umu", np.array([d[0] for d in dirs]))
            self._put("rt_phi", np.array([d[1] for d in dirs]))
        self.calls += 1
        self.last_time = when

    def receipt(self):
        return dict(calls=self.calls, last_time=None if self.last_time is None else str(self.last_time),
                    wavelengths=list(self.diag.wavelengths), directions=self.diag.directions())


def solar_cosine(when, lat_deg, lon_deg) -> np.ndarray:
    """cos(solar zenith) at ``when`` (UTC) for each cell (woof.core.solar, WRF's formulae)."""
    import cupy as cp
    from woof.core.solar import solar_geometry_device
    from woof.core.swint import calendar_scalars
    julian, xtime, gmt = calendar_scalars(when.replace(tzinfo=None), 0.0)
    cz = solar_geometry_device(julian, xtime, gmt, cp.asarray(np.float32(lat_deg)),
                               cp.asarray(np.float32(lon_deg)))[1]
    return np.asarray(cp.asnumpy(cz), dtype=np.float64)
