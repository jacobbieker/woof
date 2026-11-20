"""Gridded urban morphology (WPS ``URB_PARAM``) for the urban canopy schemes.

``RunConfig.urban_param_file`` names a NetCDF on the domain's mass grid that
carries WPS's 132-slot ``URB_PARAM`` block: a ``geo_em``, a ``met_em`` or a
``woof-buildings.v1`` product (:mod:`woof.static.buildings`).  This module
reads it and does what WRF v4.7.1's real.exe does with it
(``dyn_em/module_initialize_real.F:3152-3226``) for the parts woof can
match, refusing the rest:

* the split.  ``LP_URB2D = URB_PARAM(91)``, ``LB_URB2D = (95)``,
  ``HGT_URB2D = (94)`` for every urban option; ``MH_URB2D = (92)``,
  ``STDH_URB2D = (93)`` and ``LF_URB2D(k) = (95 + k)``, k = 1..4, for
  ``sf_urban_physics = 1`` only; ``HI_URB2D(k) = (117 + k)``, k = 1..15, for
  options 2 and 3 only.  The arrays an option does not read stay at the
  Registry's zero, as in WRF.  (``skip_middle_points_t`` never skips at
  real.exe's first time level, ``hold_ups`` true, so every column is split.)
* ``FRC_URB2D``.  When the file carries one (``geo_em``/``met_em`` do; the
  woof product does not) it becomes the domain's urban fraction in place of
  the route's, after real.exe's fill: ``FRC_URB2D == 0`` on an urban
  category (13/24/25/26 for NLCD40 and MODIFIED_IGBP_MODIS_NOAH, 1 for USGS)
  becomes 0.9.  real.exe's OTHER rule -- ``FRC_URB2D >= 0.5`` on a non-urban
  category below 30 reclassifies the column's IVGTYP to urban -- cannot be
  matched here: by the time woof initialises the urban model the land
  surface has already been built from the route's IVGTYP.  A file that would
  reclassify any column is refused (a ``wrfinput`` route, already through
  real.exe, never triggers it).
* ``urban_var_init``'s keep/zero rule (``module_sf_urban.F:2788-2833``) is
  applied by :func:`woof.core.urban_state.urban_var_init_host`, which this
  module feeds.

Refusals, before any physics runs: a grid that is not the domain's
(shape, and latitude/longitude and DX/DY where both sides carry them); a
``Time`` axis longer than one; masked or non-finite values on any urban
column whose ``HGT_URB2D > 0`` (elsewhere urban_var_init zeroes the
column, as WRF does); and, for option 1, the morphology on which WRF's NUDAPT arm
(``module_sf_urban.F:649-788``) is not finite: a column it would run on
(urban, ``HGT_URB2D > 0``, ``MH_URB2D > 0``) with ``LP_URB2D`` outside
(0, 1), a ``LF_URB2D`` sector <= 0, or a negative ``LB_URB2D`` or
``STDH_URB2D``.  The NUDAPT oracle (``tools/urban_wrf471_oracle/
ucm_nudapt_oracle.F90``, ``probe`` variant) shows WRF itself going
non-finite on the zero-sector and zero-LP cases.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "URB_PARAM_INDEX",
    "URB_PARAM_SIZE",
    "UrbanParamError",
    "UrbanMorphology",
    "read_urban_param",
    "split_urb_param",
    "real_frc_fill",
    "real_ivgtyp_promotion",
    "check_nudapt_columns",
    "load_urban_morphology",
]

#: 1-based URB_PARAM slots real.exe reads (module_initialize_real.F:3189-3215).
URB_PARAM_SIZE = 132
URB_PARAM_INDEX = {
    "LP_URB2D": 91, "MH_URB2D": 92, "STDH_URB2D": 93, "HGT_URB2D": 94,
    "LB_URB2D": 95, "LF_URB2D": 96, "HI_URB2D": 118,
}
NUM_LF = 4
NUM_HI = 15

#: real.exe's urban categories per MMINLU (module_initialize_real.F:3168-3184).
_URBAN_CATEGORIES = {
    "NLCD40": (13, 24, 25, 26),
    "MODIFIED_IGBP_MODIS_NOAH": (13, 24, 25, 26),
    "USGS": (1,),
}
_PROMOTED_TO = {"NLCD40": 13, "MODIFIED_IGBP_MODIS_NOAH": 13, "USGS": 1}


class UrbanParamError(ValueError):
    """The gridded urban morphology cannot be used as WRF would use it."""


@dataclass(frozen=True)
class UrbanMorphology:
    """One domain's real.exe-split morphology (host float32 arrays).

    ``lf`` is (4, ny, nx), ``hi`` (15, ny, nx); ``frc`` is the file's own
    FRC_URB2D after real.exe's 0.9 fill, or None when the file has none.
    """

    lp: np.ndarray
    lb: np.ndarray
    hgt: np.ndarray
    mh: np.ndarray
    stdh: np.ndarray
    lf: np.ndarray
    hi: np.ndarray
    frc: np.ndarray | None
    source: str
    sha256: str = ""


def _plain(var) -> np.ndarray:
    """A netCDF variable as float32, masked cells as NaN (refused later)."""
    data = var[:]
    if np.ma.isMaskedArray(data):
        data = np.ma.filled(data.astype(np.float64), np.nan)
    return np.asarray(data, dtype=np.float32)


def read_urban_param(path, *, shape) -> tuple[np.ndarray, np.ndarray | None, dict]:
    """Read ``URB_PARAM`` as (132, ny, nx) float32 and FRC_URB2D (or None).

    Accepts WPS's (Time, 132, south_north, west_east) layout (geo_em,
    met_em) and the woof product's (urb_param, y, x).  The third value holds
    what the file says about its grid (``lat``/``lon`` arrays, ``dx``/``dy``)
    for :func:`load_urban_morphology` to check.
    """
    import netCDF4

    path = Path(path)
    if not path.is_file():
        raise UrbanParamError(f"urban_param_file {str(path)!r} does not exist")
    ny, nx = (int(v) for v in shape)
    with netCDF4.Dataset(path) as ds:
        if "URB_PARAM" not in ds.variables:
            raise UrbanParamError(
                f"{path}: no URB_PARAM variable (WPS writes it into geo_em/"
                "met_em when GEOGRID.TBL names a NUDAPT/WUDAPT URB_PARAM "
                "source; woof's buildings product writes it too)")
        var = ds.variables["URB_PARAM"]
        block = _plain(var)
        dims = var.dimensions
        if block.ndim == 4:
            if dims[0].lower() != "time":
                raise UrbanParamError(f"{path}: URB_PARAM dimensions {dims} "
                                      "are not (Time, 132, y, x)")
            if block.shape[0] != 1:
                raise UrbanParamError(
                    f"{path}: URB_PARAM has {block.shape[0]} times; the "
                    "morphology is static, so give a file with one")
            block = block[0]
        if block.ndim != 3 or block.shape[0] != URB_PARAM_SIZE:
            raise UrbanParamError(
                f"{path}: URB_PARAM shape {tuple(var.shape)} is not WPS's "
                f"({URB_PARAM_SIZE}, south_north, west_east)")
        if block.shape[1:] != (ny, nx):
            raise UrbanParamError(
                f"{path}: URB_PARAM is on a {block.shape[1]} x {block.shape[2]} "
                f"grid but this domain's mass grid is {ny} x {nx}; the "
                "morphology must be on the domain's own grid (staggered or "
                "parent-grid files are not regridded)")
        frc = None
        if "FRC_URB2D" in ds.variables:
            frc = _plain(ds.variables["FRC_URB2D"])
            if frc.ndim == 3:
                frc = frc[0]
            if frc.shape != (ny, nx):
                raise UrbanParamError(f"{path}: FRC_URB2D shape {frc.shape} "
                                      f"is not the domain's ({ny}, {nx})")
        grid: dict = {}
        for lat_name, lon_name in (("XLAT_M", "XLONG_M"), ("XLAT", "XLONG"),
                                   ("lat", "lon"), ("latitude", "longitude")):
            if lat_name in ds.variables and lon_name in ds.variables:
                lat = _plain(ds.variables[lat_name])
                lon = _plain(ds.variables[lon_name])
                if lat.ndim == 3:
                    lat, lon = lat[0], lon[0]
                if lat.shape == (ny, nx) and lon.shape == (ny, nx):
                    grid["lat"], grid["lon"] = lat, lon
                break
        for key in ("DX", "DY"):
            if key in ds.ncattrs():
                grid[key.lower()] = float(np.asarray(ds.getncattr(key)).ravel()[0])
        grid["schema"] = getattr(ds, "schema", None)
    return block, frc, grid


def split_urb_param(block: np.ndarray, option: int) -> dict[str, np.ndarray]:
    """real.exe's split of URB_PARAM into the urban morphology arrays.

    ``module_initialize_real.F:3189-3215``: LP/LB/HGT for options 1-3,
    HI for 2/3, MH/STDH/LF for 1.  Arrays the option does not read are the
    Registry's zero.
    """
    option = int(option)
    if option not in (1, 2, 3):
        raise UrbanParamError(f"sf_urban_physics={option} reads no URB_PARAM")
    block = np.asarray(block, dtype=np.float32)
    _, ny, nx = block.shape
    at = lambda index: block[index - 1]
    zero = np.zeros((ny, nx), np.float32)
    out = {
        "lp": at(URB_PARAM_INDEX["LP_URB2D"]).copy(),
        "lb": at(URB_PARAM_INDEX["LB_URB2D"]).copy(),
        "hgt": at(URB_PARAM_INDEX["HGT_URB2D"]).copy(),
        "mh": zero.copy(), "stdh": zero.copy(),
        "lf": np.zeros((NUM_LF, ny, nx), np.float32),
        "hi": np.zeros((NUM_HI, ny, nx), np.float32),
    }
    if option == 1:
        out["mh"] = at(URB_PARAM_INDEX["MH_URB2D"]).copy()
        out["stdh"] = at(URB_PARAM_INDEX["STDH_URB2D"]).copy()
        base = URB_PARAM_INDEX["LF_URB2D"] - 1
        out["lf"] = block[base:base + NUM_LF].copy()
    else:
        base = URB_PARAM_INDEX["HI_URB2D"] - 1
        out["hi"] = block[base:base + NUM_HI].copy()
    return out


def real_ivgtyp_promotion(frc, ivgtyp, mminlu: str) -> np.ndarray:
    """The columns real.exe would reclassify to urban (bool, ny x nx).

    ``module_initialize_real.F:3168-3175``: FRC_URB2D >= 0.5 on a category
    that is not urban and below 30 becomes 13 (NLCD40,
    MODIFIED_IGBP_MODIS_NOAH) or 1 (USGS).  Other legends: none.
    """
    frc = np.asarray(frc, dtype=np.float32)
    ivgtyp = np.asarray(ivgtyp).astype(np.int64)
    cats = _URBAN_CATEGORIES.get(str(mminlu).strip().upper())
    if cats is None:
        return np.zeros(ivgtyp.shape, bool)
    return (frc >= np.float32(0.5)) & ~np.isin(ivgtyp, cats) & (ivgtyp < 30)


def real_frc_fill(frc, ivgtyp, mminlu: str) -> np.ndarray:
    """real.exe's FRC_URB2D = 0.9 where it is 0 on an urban category
    (``module_initialize_real.F:3177-3181``); a new float32 array."""
    frc = np.array(frc, dtype=np.float32)
    ivgtyp = np.asarray(ivgtyp).astype(np.int64)
    cats = _URBAN_CATEGORIES.get(str(mminlu).strip().upper())
    if cats is None:
        return frc
    frc[(frc == np.float32(0.0)) & np.isin(ivgtyp, cats)] = np.float32(0.9)
    return frc


def check_nudapt_columns(morph: dict, urban, *, where: str = "") -> None:
    """Refuse option-1 morphology on which WRF's NUDAPT arm is not finite.

    ``morph`` is the split (after urban_var_init's keep/zero); ``urban`` the
    UTYPE_URB2D > 0 mask.  The arm runs where ``MH_URB2D > 0``
    (module_sf_urban.F:649).
    """
    urban = np.asarray(urban, bool)
    active = urban & (morph["mh"] > 0)
    if not active.any():
        return
    lp, lb, stdh, lf = morph["lp"], morph["lb"], morph["stdh"], morph["lf"]
    problems = []
    bad = active & ~((lp > 0) & (lp < 1))
    if bad.any():
        problems.append(
            f"LP_URB2D outside (0, 1) on {int(bad.sum())} columns (LP = 0 "
            "with LB > LP makes the street width 0*Inf, :687-688; LP >= 1 "
            "leaves no road, RW <= 0, and the sky-view factor divides by it, "
            ":775)")
    bad = active & (lf <= 0).any(axis=0)
    if bad.any():
        problems.append(
            f"a LF_URB2D sector <= 0 on {int(bad.sum())} columns (lambda_f = "
            "0 gives Z0C = 0 when the wind is in that sector, :743-744, and "
            "the canopy wind then takes LOG(0): WRF's own result is NaN)")
    bad = active & ((lb < 0) | (stdh < 0))
    if bad.any():
        problems.append(
            f"negative LB_URB2D or STDH_URB2D on {int(bad.sum())} columns "
            "(not a morphology; WPS's missing value?)")
    if problems:
        raise UrbanParamError(
            f"urban_param_file{where}: the NUDAPT arm of WRF's single-layer "
            "UCM would run on undefined morphology: " + "; ".join(problems))


def _check_grid(grid: dict, path, *, latitude=None, longitude=None,
                dx=None, dy=None) -> None:
    if latitude is not None and "lat" in grid:
        lat = np.asarray(latitude, np.float64)
        lon = np.asarray(longitude, np.float64)
        if lat.shape == grid["lat"].shape:
            dlat = np.nanmax(np.abs(lat - grid["lat"]))
            dlon = np.abs(lon - grid["lon"])
            dlon = np.nanmax(np.minimum(dlon, 360.0 - dlon))
            if not (dlat <= 1e-3 and dlon <= 1e-3):
                raise UrbanParamError(
                    f"{path}: its mass-point latitude/longitude differ from "
                    f"this domain's by up to {dlat:.4g} / {dlon:.4g} degrees; "
                    "it is another grid")
    for key, value in (("dx", dx), ("dy", dy)):
        if value is not None and key in grid:
            if abs(float(value) - grid[key]) > 1e-3 * max(1.0, abs(float(value))):
                raise UrbanParamError(
                    f"{path}: {key.upper()} = {grid[key]:g} m but this domain's "
                    f"is {float(value):g} m")


def load_urban_morphology(path, *, option: int, ivgtyp, urban, mminlu: str,
                          latitude=None, longitude=None, dx=None,
                          dy=None) -> UrbanMorphology:
    """Read, check and split one domain's ``urban_param_file``.

    ``ivgtyp`` is the route's land-use category grid and ``urban`` the
    UTYPE_URB2D > 0 mask urban_var_init will use (from the same IVGTYP).
    """
    from woof.file_digest import sha256_file

    ivgtyp = np.asarray(ivgtyp)
    shape = ivgtyp.shape
    try:
        digest = sha256_file(Path(path), block_bytes=1024 * 1024)
    except OSError as exc:
        raise UrbanParamError(
            f"urban_param_file {str(path)!r} does not exist or is "
            "unreadable") from exc
    block, frc, grid = read_urban_param(path, shape=shape)
    _check_grid(grid, path, latitude=latitude, longitude=longitude,
                dx=dx, dy=dy)
    morph = split_urb_param(block, option)
    urban = np.asarray(urban, bool)
    # urban_var_init keeps the morphology only where HGT_URB2D > 0 (a NaN
    # compares false, so WRF zeroes such a column and runs it on the table
    # -- partial-coverage files masked outside NUDAPT run).  A non-finite
    # value on a column WRF keeps would reach the canopy model: refused.
    kept = urban & (morph["hgt"] > 0)
    names = ("lp", "lb", "mh", "stdh", "lf", "hi")
    for name in names:
        value = morph[name]
        cols = kept if value.ndim == 2 else kept[None]
        bad = cols & ~np.isfinite(value)
        if bad.any():
            raise UrbanParamError(
                f"{path}: {name.upper()}_URB2D is missing or non-finite on "
                f"{int(bad.any(axis=0).sum() if value.ndim == 3 else bad.sum())}"
                " urban columns whose HGT_URB2D > 0; WRF would carry the "
                "value into the canopy model, so the file is refused rather "
                "than guessed at")
    # Masked/NaN slots elsewhere become urban_var_init's zero, as in WRF.
    for name in names + ("hgt",):
        morph[name] = np.where(np.isfinite(morph[name]), morph[name],
                               np.float32(0.0)).astype(np.float32)
    if frc is not None:
        bad = ~np.isfinite(frc)
        if bad.any():
            raise UrbanParamError(f"{path}: FRC_URB2D is missing or non-finite "
                                  f"on {int(bad.sum())} columns")
        promoted = real_ivgtyp_promotion(frc, ivgtyp, mminlu)
        if promoted.any():
            raise UrbanParamError(
                f"{path}: real.exe would reclassify {int(promoted.sum())} "
                f"columns to urban (FRC_URB2D >= 0.5 on a non-urban {mminlu} "
                "category, module_initialize_real.F:3168-3175), but this "
                "route's land surface is already initialised from its own "
                "land-use grid, so that cannot be reproduced; run from a "
                "wrfinput real.exe made from this geo_em, or give a file "
                "without FRC_URB2D (the route's urban fraction is then kept)")
        frc = real_frc_fill(frc, ivgtyp, mminlu)
    return UrbanMorphology(frc=frc, source=str(path), sha256=digest, **morph)
