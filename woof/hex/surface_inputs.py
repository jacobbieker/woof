"""Per-cell surface and static inputs for the generic column seam (plan 1.2).

THE GAP THIS CLOSES.  The R1 generic seam (``woof.core.mpas_column_suite.
run_mpas_column_physics``) takes any WRF column selection that WOOF's own
validators accept, and hands the seam's ``surface`` mapping straight to
``woof.core.physics.initialize_physics``.  The hex forecast filled that mapping
with the dozen fields every land surface reads (landmask, xland, vegfra, tsk,
tmn, xice, snow, ivgtyp, isltyp, soil) and nothing else, so a scheme that reads
more -- CLM lake's mask and bathymetry, the slab's thermal inertia, PX's
category fractions, an urban fraction -- either fell to a WRF fallback branch
the run never chose or stopped inside ``initialize_physics``.

THE CONTRACT.
* :data:`CONTRACT` declares, per WRF column scheme, the ``initialize_physics``
  keywords the scheme reads beyond that dozen, and where hex stages each one.
* :func:`required_inputs` evaluates the contract on a selection's resolved
  ``RunConfig`` (the engine plan's ``config``), so a scheme's own defaults
  (``use_lakedepth = 1``) decide what is read, exactly as on WRF.
* The hex static builder (:mod:`woof.hex.surface_statics`, ``woof hex
  surface-statics``) produces the staged inputs on the mesh into a sidecar file,
  ``<mesh>.surface_statics.nc``; :func:`bind` reads that file for a run and
  returns the keyword mapping the sealed constructor carries (its arrays enter
  the constructor digest) and the receipt.
* An input the selection reads and the run does not supply is REFUSED by name;
  nothing here defaults one.  An input supplied for a selection that reads none
  is refused too (accepted and never read).
* A selection that reads no extra input binds an EMPTY mapping, and the sealed
  constructor, its identity and every number the seam computes are then exactly
  what they were before this module existed.

:func:`match_statics_source` is the orographic statics matcher: a mesh file
that carries a SUPERSET of the fields a selection reads serves it (topo_wind
beside a drag), reading only the fields named.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import numpy as np

from .errors import ConfigurationRefusal

#: The schema the sidecar file carries (``woof hex surface-statics``).
SCHEMA = "woof-hex.surface-statics.v1"
#: The sidecar's file name beside the mesh, as ``<mesh>.ugwp_oro_data.nc`` is.
FILE_SUFFIX = ".surface_statics.nc"


@dataclass(frozen=True)
class SurfaceInput:
    """One ``initialize_physics`` keyword a scheme reads, and its hex source.

    ``kind`` is ``"cell"`` (one value per cell, float32 or int32 as
    ``dtype`` says) or ``"flag"`` (one integer the sidecar records as a global
    attribute, WRF's geogrid/real flags).  ``variable`` is the sidecar's name
    for it (WRF's own upper-case name).  ``staged`` says whether the hex static
    builder produces it today; a required input that is not staged is refused
    with ``needs`` as the remedy.
    """

    keyword: str
    variable: str
    kind: str
    dtype: str
    staged: bool
    source: str
    needs: str = ""
    #: The only values the input may hold: a lake mask is 0 or 1, and a
    #: geogrid flag the builder stages is 1 (0 would select a WRF fallback
    #: branch the sidecar did not build for).  ``None``: any finite value.
    allowed: tuple[int, ...] | None = None
    #: The closed range a ``cell`` input's values must lie in (a fraction is
    #: in [0, 1]); ``None``: any finite value.
    bounds: tuple[float, float] | None = None


@dataclass(frozen=True)
class SchemeInputs:
    """A WRF column scheme and the per-cell inputs it reads when active."""

    scheme: str
    selector: str
    active: Callable[[Any], bool]
    inputs: Callable[[Any], tuple[SurfaceInput, ...]]


def _value(cfg, key, default=0):
    value = getattr(cfg, key, default)
    return default if value is None else value


# ---------------------------------------------------------------------------
# CLM lake (sf_lake_physics = 1), the contract's first staged user
# ---------------------------------------------------------------------------
_LAKEMASK = SurfaceInput(
    "lakemask", "LAKEMASK", "cell", "float32", True,
    "WRF real.exe's rule on the mesh: a cell whose dominant category in the "
    "with-lakes MODIS land use is ISLAKE (21), a water cell of the run's own "
    "static (woof hex surface-statics)", allowed=(0, 1))
_LAKEFLAG = SurfaceInput(
    "lake_mask_flag", "LAKEFLAG", "flag", "int32", True,
    "1: the mask comes from geography (WRF's LAKEFLAG); 0 would select "
    "lakeini's ISWATER-and-elevation fallback, which hex does not stage",
    allowed=(1,))
_LAKE_DEPTH = SurfaceInput(
    "lake_depth", "LAKE_DEPTH", "cell", "float32", True,
    "WPS lake_depth bathymetry (GEOGRID.TBL LAKE_DEPTH: average_gcell(1.0)+"
    "search(5), masked over land, fill 10 m) aggregated onto each Voronoi cell "
    "(woof hex surface-statics --geog-root)")
_LAKE_DEPTH_FLAG = SurfaceInput(
    "lake_depth_flag", "LAKE_DEPTH_FLAG", "flag", "int32", True,
    "1: LAKE_DEPTH was built from the bathymetry dataset (geogrid's "
    "FLAG_LAKE_DEPTH); 0 would discard the staged bathymetry",
    allowed=(1,))


def _lake_inputs(cfg) -> tuple[SurfaceInput, ...]:
    inputs = (_LAKEMASK, _LAKEFLAG)
    if int(_value(cfg, "use_lakedepth", 1)) == 1:
        # use_lakedepth = 0 is WRF's declared lakedepth_default everywhere: a
        # namelist choice, so the bathymetry is then not read at all.
        inputs += (_LAKE_DEPTH, _LAKE_DEPTH_FLAG)
    return inputs


# ---------------------------------------------------------------------------
# Schemes the contract now carries and the builder does not stage yet.  Each
# is refused by name, with what it still needs; staging one is a builder row
# plus ``staged=True`` here.
# ---------------------------------------------------------------------------
def _unstaged(keyword, variable, dtype, needs, kind="cell"):
    return SurfaceInput(keyword, variable, kind, dtype, False, "not staged", needs)


_THERMAL_INERTIA = _unstaged(
    "thermal_inertia", "THC", "float32",
    "LANDUSE.TBL THERIN/100 per cell from the static's land use (plan 1.6)")
_LANDUSEF = _unstaged(
    "landusef", "LANDUSEF", "float32",
    "per-cell land-use category FRACTIONS from the static builder (the MPAS "
    "static carries the dominant category only) (plan 1.6)")
_SOILCTOP = _unstaged(
    "soilctop", "SOILCTOP", "float32",
    "per-cell top-soil category fractions from the static builder (plan 1.6)")
_SOILCBOT = _unstaged(
    "soilcbot", "SOILCBOT", "float32",
    "per-cell bottom-soil category fractions from the static builder (plan 1.6)")
_FRC_URB2D = SurfaceInput(
    "frc_urb2d", "FRC_URB2D", "cell", "float32", True,
    "SLUCM's urban fraction as WOOF's WRF preparation estimates it: the cell's "
    "MODIS land-use category fractions (pixel shares per Voronoi cell) times "
    "URBPARM.TBL FRC_URB, on dominant-urban land cells (woof hex "
    "surface-statics --fields urban)", bounds=(0.0, 1.0))
_LCZ = _unstaged(
    "ivgtyp", "LU_INDEX (LCZ 51-61)", "int32",
    "the Local Climate Zone classes 51-61 kept distinct per cell (CGLC-MODIS-"
    "LCZ) instead of the static's collapsed ISURBAN, for URBPARM_LCZ.TBL")


def _urban_inputs(cfg) -> tuple[SurfaceInput, ...]:
    if int(_value(cfg, "use_wudapt_lcz")) == 1:
        return (_LCZ,)
    if int(_value(cfg, "sf_urban_physics")) == 1:
        return (_FRC_URB2D,)
    # BEP and BEP+BEM run on URBPARM.TBL's own fractions, exactly as WOOF's
    # WRF preparation hands them (woof.static.highres: only SLUCM takes the
    # land-cover estimate), so they read no per-cell input beyond the init's.
    return ()
_IRRIGATION = _unstaged(
    "irrigation", "IRRIGATION", "float32",
    "the per-cell irrigated fraction from the irrigation dataset (plan 6.8)")
_IRR_PH = _unstaged(
    "irr_ph_field", "IRR_RAND_FIELD", "float32",
    "the per-cell irrigation phase field (plan 6.8)")
_LIGHTNING = _unstaged(
    "lightning_inputs", "lightning_inputs", "object",
    "the run's own reflectivity/updraft columns and cell geometry for the "
    "flash-rate parameterisation (plan 2.3)", kind="object")
_CROP = _unstaged(
    "noahmp_crop_inputs", "CROPCAT/CROPTYPE", "float32",
    "Noah-MP crop statics (CROPTYPE, PLANTING, HARVEST, SEASON_GDD)")
_GROUNDWATER = _unstaged(
    "noahmp_groundwater_inputs", "EQZWT/RIVERBED/RECHCLIM", "float32",
    "Noah-MP MMF groundwater statics on the mesh (opt_run = 5)")


#: Every WRF column scheme that reads per-cell inputs beyond the dozen the hex
#: forecast always passes.  Order is the receipt's.
CONTRACT: tuple[SchemeInputs, ...] = (
    SchemeInputs("clm_lake", "sf_lake_physics=1",
                 lambda cfg: int(_value(cfg, "sf_lake_physics")) == 1,
                 _lake_inputs),
    SchemeInputs("slab", "sf_surface_physics=1",
                 lambda cfg: int(_value(cfg, "sf_surface_physics")) == 1,
                 lambda cfg: (_THERMAL_INERTIA,)),
    SchemeInputs("pleim_xiu_lsm", "sf_surface_physics=7",
                 lambda cfg: int(_value(cfg, "sf_surface_physics")) == 7,
                 lambda cfg: (_LANDUSEF, _SOILCBOT)),
    SchemeInputs("ruc_mosaic", "sf_surface_physics=3 with mosaic_lu/mosaic_soil",
                 lambda cfg: int(_value(cfg, "sf_surface_physics")) == 3
                 and (int(_value(cfg, "mosaic_lu")) or int(_value(cfg, "mosaic_soil"))),
                 lambda cfg: tuple(
                     row for switch, row in (("mosaic_lu", _LANDUSEF),
                                             ("mosaic_soil", _SOILCTOP))
                     if int(_value(cfg, switch)))),
    SchemeInputs("noah_mosaic", "sf_surface_physics=2 with sf_surface_mosaic=1",
                 lambda cfg: int(_value(cfg, "sf_surface_physics")) == 2
                 and int(_value(cfg, "sf_surface_mosaic")) == 1,
                 lambda cfg: (_LANDUSEF,)),
    SchemeInputs("urban", "sf_urban_physics=1/2/3",
                 lambda cfg: int(_value(cfg, "sf_urban_physics")) > 0,
                 _urban_inputs),
    SchemeInputs("irrigation", "sf_surf_irr_scheme>0",
                 lambda cfg: int(_value(cfg, "sf_surf_irr_scheme")) > 0,
                 lambda cfg: (_IRRIGATION, _IRR_PH)),
    SchemeInputs("lightning", "lightning_option>0",
                 lambda cfg: int(_value(cfg, "lightning_option")) > 0,
                 lambda cfg: (_LIGHTNING,)),
    SchemeInputs("noahmp_crop", "sf_surface_physics=4 with opt_crop>0",
                 lambda cfg: int(_value(cfg, "sf_surface_physics")) == 4
                 and int(_value(cfg, "opt_crop")) > 0,
                 lambda cfg: (_CROP,)),
    SchemeInputs("noahmp_groundwater", "sf_surface_physics=4 with opt_run=5",
                 lambda cfg: int(_value(cfg, "sf_surface_physics")) == 4
                 and int(_value(cfg, "opt_run", 1)) == 5,
                 lambda cfg: (_GROUNDWATER,)),
)


@dataclass(frozen=True)
class RequiredInput:
    scheme: str
    selector: str
    input: SurfaceInput


def required_inputs(cfg) -> tuple[RequiredInput, ...]:
    """The per-cell inputs the resolved selection ``cfg`` reads, in order."""

    out: list[RequiredInput] = []
    seen: set[str] = set()
    for row in CONTRACT:
        if not row.active(cfg):
            continue
        for item in row.inputs(cfg):
            if item.keyword in seen:
                continue
            seen.add(item.keyword)
            out.append(RequiredInput(row.scheme, row.selector, item))
    return tuple(out)


def unstaged_problems(cfg) -> list[str]:
    """One sentence per required input hex does not stage yet."""

    return [
        f"{need.selector} ({need.scheme}) reads {need.input.keyword} "
        f"({need.input.variable}) per cell, which the hex static builder does "
        f"not stage yet: it needs {need.input.needs}"
        for need in required_inputs(cfg) if not need.input.staged
    ]


def plan_config(physics: Mapping[str, Any]):
    """The resolved ``RunConfig`` of a generic selection (pure host)."""

    from woof.core.mpas_column_suite import column_physics_plan

    return column_physics_plan(dict(physics)).config


def reads_surface_statics(physics: Mapping[str, Any] | None) -> bool:
    """Whether a generic selection reads anything from the sidecar file."""

    if not physics:
        return False
    return bool(required_inputs(plan_config(physics)))


def default_path(grid: str | Path, mesh: str) -> Path:
    return Path(grid).expanduser().absolute().parent / f"{mesh}{FILE_SUFFIX}"


def builder_default_path(static: str | Path) -> Path:
    """Where ``woof hex surface-statics`` writes without ``-o``: beside the
    static, named from it with a trailing ``.static`` dropped."""

    static = Path(static).expanduser().absolute()
    stem = static.name[:-3] if static.name.endswith(".nc") else static.name
    stem = stem[:-7] if stem.endswith(".static") else stem
    return static.with_name(f"{stem}{FILE_SUFFIX}")


def default_candidates(grid: str | Path, static: str | Path | None,
                       mesh: str) -> tuple[Path, ...]:
    """The places a forecast looks for the sidecar when none is named, in
    order: ``<mesh>.surface_statics.nc`` beside --grid, the same name beside
    --static, and the builder's own default beside --static."""

    places = [default_path(grid, mesh)]
    if static is not None:
        places += [default_path(static, mesh), builder_default_path(static)]
    return tuple(dict.fromkeys(places))


#: Largest centre difference (radians, about 6 mm on the Earth) at which a
#: sidecar's cells are this run's cells.  Longitudes are compared modulo 2 pi:
#: a static stores them in [0, 2 pi) and the loaded mesh in (-pi, pi].
CELL_CENTRE_TOLERANCE_RAD = 1.0e-9


def same_cells(lat_a, lon_a, lat_b, lon_b) -> bool:
    """Whether two centre lists name the same cells in the same order."""

    lat_a, lat_b = (np.asarray(v, dtype=np.float64).reshape(-1) for v in (lat_a, lat_b))
    lon_a, lon_b = (np.asarray(v, dtype=np.float64).reshape(-1) for v in (lon_a, lon_b))
    if lat_a.shape != lat_b.shape or lon_a.shape != lon_b.shape:
        return False
    dlon = np.abs(np.mod(lon_a - lon_b + np.pi, 2.0 * np.pi) - np.pi)
    return bool(np.all(np.abs(lat_a - lat_b) <= CELL_CENTRE_TOLERANCE_RAD)
                and np.all(dlon <= CELL_CENTRE_TOLERANCE_RAD))


def _array_identity(values: np.ndarray) -> dict[str, Any]:
    contiguous = np.ascontiguousarray(values)
    return {"dtype": contiguous.dtype.str, "shape": list(contiguous.shape),
            "sha256": sha256(contiguous.tobytes()).hexdigest()}


def urban_table_sha256(cfg) -> str:
    """SHA-256 of the URBPARM.TBL a run's urban model reads, chosen exactly as
    woof.core.physics._attach_urban chooses it (urban_table_dir, else the
    urban_ucm_generation's vendored table)."""

    from woof.core.urban_tables import URBAN_TABLE_DIR, urban_table_path
    from woof.tables import table_path

    generation = str(_value(cfg, "urban_ucm_generation", "wrf_471"))
    custom = str(_value(cfg, "urban_table_dir", "") or "")
    if custom:
        path = urban_table_path(0, Path(custom))
    else:
        directory = URBAN_TABLE_DIR / "wrf480" if generation == "wrf_480" else None
        path = table_path("urban-480" if generation == "wrf_480" else "urban",
                          urban_table_path(0, directory))
    return sha256(Path(path).read_bytes()).hexdigest()


def _check_allowed(item: SurfaceInput, values, where: str) -> None:
    if item.bounds is not None:
        low, high = item.bounds
        array = np.asarray(values, dtype=np.float64)
        if np.any(array < low) or np.any(array > high):
            raise ValueError(
                f"{where}: {item.variable} holds values outside [{low:g}, {high:g}] "
                f"(min {float(array.min()):g}, max {float(array.max()):g}), "
                f"which {item.keyword} cannot be")
    if item.allowed is None:
        return
    bad = np.setdiff1d(np.unique(np.asarray(values)), np.asarray(item.allowed))
    if bad.size:
        raise ValueError(
            f"{where}: {item.variable} holds {bad.tolist()[:5]}, and "
            f"{item.keyword} admits only {list(item.allowed)}")


def bind(physics: Mapping[str, Any] | None, path: str | Path | None, *,
         latitude_rad, longitude_rad,
         static_path: str | Path | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """The surface keywords a generic selection reads, from the sidecar.

    Returns ``(inputs, receipt)``.  ``inputs`` is empty, and ``receipt`` says
    so, for a selection that reads none; ``path`` is then refused if given.
    Every refusal names the input and the remedy.
    """

    if not physics:
        if path is not None:
            raise ConfigurationRefusal(
                "surface_statics", str(path),
                "this run's physics row is not the generic column seam, which "
                "is the only seam that reads the surface statics file; it would "
                "be accepted and never read", "no --surface-statics")
        return {}, {"required": []}
    cfg = plan_config(physics)
    required = required_inputs(cfg)
    if not required:
        if path is not None:
            raise ConfigurationRefusal(
                "surface_statics", str(path),
                "this selection reads no per-cell surface input beyond the "
                "init's own fields; the file would be accepted and never read",
                "no --surface-statics")
        return {}, {"required": []}
    problems = unstaged_problems(cfg)
    if problems:
        raise ConfigurationRefusal(
            "physics", ", ".join(sorted({need.selector for need in required
                                         if not need.input.staged})),
            "missing surface input: " + "; ".join(problems),
            "a selection whose per-cell inputs hex stages (woof.hex."
            "surface_inputs.CONTRACT), or stage them first")
    names = [need.input.keyword for need in required]
    if path is None:
        raise ConfigurationRefusal(
            "surface_statics", None,
            f"this selection reads {names} per cell "
            f"({', '.join(sorted({need.selector for need in required}))}) and no "
            "surface statics file was given; a missing input is never defaulted",
            "--surface-statics <mesh>.surface_statics.nc, built by `woof hex "
            "surface-statics`")
    source = Path(path).expanduser().resolve(strict=True)
    import netCDF4

    lat = np.asarray(latitude_rad, dtype=np.float64).reshape(-1)
    lon = np.asarray(longitude_rad, dtype=np.float64).reshape(-1)
    n_cells = lat.size
    inputs: dict[str, Any] = {}
    receipt_inputs: dict[str, Any] = {}
    with netCDF4.Dataset(source, "r") as dataset:
        schema = getattr(dataset, "schema", None)
        if schema != SCHEMA:
            raise ConfigurationRefusal(
                "surface_statics", str(source),
                f"the file's schema is {schema!r}, not {SCHEMA!r}",
                "a file written by `woof hex surface-statics`")
        if not ("latCell" in dataset.variables and "lonCell" in dataset.variables
                and same_cells(dataset.variables["latCell"][:],
                               dataset.variables["lonCell"][:], lat, lon)):
            raise ConfigurationRefusal(
                "surface_statics", str(source),
                "the file was built on other cells than this run's mesh (its "
                "latCell/lonCell are not the mesh's, in the mesh's order), so "
                "each value would land on another cell",
                "the surface statics file of this run's mesh")
        provenance = json.loads(getattr(dataset, "provenance_json", "{}"))
        if static_path is not None:
            # Two statics can share every cell centre and still differ in the
            # land use the mask and fractions were built from.
            run_static = sha256(Path(static_path).expanduser().resolve(
                strict=True).read_bytes()).hexdigest()
            if provenance.get("parent_static_sha256") != run_static:
                raise ConfigurationRefusal(
                    "surface_statics", str(source),
                    "the file was built from another static file than this "
                    f"run's ({provenance.get('parent_static', 'unrecorded')}, "
                    f"sha256 {provenance.get('parent_static_sha256')}, against "
                    f"{run_static}), so its per-cell values describe another "
                    "land use", "the surface statics file built from this run's --static")
        if any(need.input.keyword == "frc_urb2d" for need in required):
            # The fraction weighs each urban category by URBPARM.TBL's FRC_URB;
            # the run's urban model must read the same table bytes.
            built = (provenance.get("urban") or {}).get("urbparm_sha256")
            run_table = urban_table_sha256(cfg)
            if built != run_table:
                raise ConfigurationRefusal(
                    "surface_statics", str(source),
                    f"FRC_URB2D was estimated with the URBPARM.TBL of sha256 "
                    f"{built}, and this run's urban model reads {run_table} "
                    "(urban_table_dir / urban_ucm_generation), so the staged "
                    "fractions would weigh the categories by another table",
                    "the default urban table, or a surface statics file built "
                    "with the run's table")
        for need in required:
            item = need.input
            if item.kind == "flag":
                if item.variable not in dataset.ncattrs():
                    raise ConfigurationRefusal(
                        "surface_statics", str(source),
                        f"{need.selector} reads {item.keyword} and the file "
                        f"records no {item.variable}",
                        f"a surface statics file built with {need.scheme}'s inputs")
                value = int(dataset.getncattr(item.variable))
                if item.allowed is not None and value not in item.allowed:
                    raise ConfigurationRefusal(
                        "surface_statics", str(source),
                        f"{item.variable} is {value}, not {list(item.allowed)}: "
                        f"{item.source}", "a file whose inputs the builder staged")
                inputs[item.keyword] = value
                receipt_inputs[item.keyword] = {"variable": item.variable,
                                                "value": value}
                continue
            if item.variable not in dataset.variables:
                raise ConfigurationRefusal(
                    "surface_statics", str(source),
                    f"{need.selector} reads {item.keyword} per cell and the "
                    f"file carries no {item.variable}",
                    f"`woof hex surface-statics` with the {need.scheme} fields")
            variable = dataset.variables[item.variable]
            if tuple(variable.dimensions) != ("nCells",):
                raise ValueError(
                    f"{item.variable} dimensions {variable.dimensions} != ('nCells',)")
            values = np.ascontiguousarray(
                np.asarray(variable[:]), dtype=np.dtype(item.dtype))
            if values.shape != (n_cells,):
                raise ValueError(
                    f"{item.variable} has {values.shape} cells, the mesh {n_cells}")
            if not np.all(np.isfinite(values)):
                raise FloatingPointError(f"{item.variable} carries non-finite values")
            try:
                _check_allowed(item, values, str(source))
            except ValueError as error:
                raise ConfigurationRefusal(
                    "surface_statics", str(source), str(error),
                    "a file whose inputs the builder staged") from None
            values.setflags(write=False)
            inputs[item.keyword] = values
            receipt_inputs[item.keyword] = {
                "variable": item.variable, **_array_identity(values),
                **({"cells_set": int(np.count_nonzero(values))}
                   if item.keyword == "lakemask" else {}),
            }
    receipt = {
        "path": str(source),
        "file_sha256": sha256(source.read_bytes()).hexdigest(),
        "schema": SCHEMA,
        "required": [
            {"scheme": need.scheme, "selector": need.selector,
             "keyword": need.input.keyword, "variable": need.input.variable,
             "source": need.input.source}
            for need in required],
        "inputs": receipt_inputs,
        "provenance": provenance,
    }
    return inputs, receipt


def seal(cfg, given: Mapping[str, Any] | None, *, n_columns: int
         ) -> dict[str, Any]:
    """Hold a constructor's surface inputs to exactly the selection's set.

    The sealed constructor's half of the contract: what ``cfg`` reads must be
    present (never defaulted) and nothing else may ride along.  Arrays come
    back as read-only host copies of the contract's dtype and ``(n_columns,)``
    shape; flags as ints.
    """

    given = {} if given is None else dict(given)
    required = required_inputs(cfg)
    names = [need.input.keyword for need in required]
    missing = [name for name in names if name not in given]
    extra = sorted(name for name in given if name not in names)
    if missing or extra:
        raise ValueError(
            f"surface inputs must be exactly {names} for this selection "
            f"(missing {missing}, extra {extra}); a scheme's per-cell input is "
            "never defaulted, and one no scheme reads is never accepted")
    sealed: dict[str, Any] = {}
    for need in required:
        item = need.input
        value = given[item.keyword]
        if item.kind == "flag":
            if isinstance(value, (bool, np.bool_)) or int(value) != value:
                raise ValueError(f"{item.keyword} must be an integer flag")
            if item.allowed is not None and int(value) not in item.allowed:
                raise ValueError(
                    f"{item.keyword}={int(value)} is not one of {list(item.allowed)}: "
                    f"{item.source}")
            sealed[item.keyword] = int(value)
            continue
        array = np.asarray(value)
        if array.dtype != np.dtype(item.dtype) or array.shape != (int(n_columns),):
            raise ValueError(
                f"{item.keyword} must be {item.dtype}[{int(n_columns)}], got "
                f"{array.dtype}{list(array.shape)}")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{item.keyword} must be finite")
        _check_allowed(item, array, "the sealed surface inputs")
        array = np.array(array, copy=True, order="C")
        array.setflags(write=False)
        sealed[item.keyword] = array
    return sealed


def contract_table() -> list[dict[str, Any]]:
    """The contract as rows (docs, ``woof hex physics`` listings and tests)."""

    rows = []
    for row in CONTRACT:
        for cfg in _probe_configs(row):
            for item in row.inputs(cfg):
                rows.append({"scheme": row.scheme, "selector": row.selector,
                             "keyword": item.keyword, "variable": item.variable,
                             "staged": item.staged,
                             "source": item.source if item.staged else item.needs})
    unique = []
    for entry in rows:
        if entry not in unique:
            unique.append(entry)
    return unique


def _probe_configs(row: SchemeInputs):
    # The inputs of a row at its widest (every switch of the scheme on).
    from types import SimpleNamespace

    yield SimpleNamespace(use_lakedepth=1, mosaic_lu=1, mosaic_soil=1,
                          sf_urban_physics=1, use_wudapt_lcz=0)
    if row.scheme == "urban":
        yield SimpleNamespace(sf_urban_physics=1, use_wudapt_lcz=1)


# ---------------------------------------------------------------------------
# orographic statics: superset matching
# ---------------------------------------------------------------------------
def match_statics_source(fields: Sequence[str],
                         sources: Sequence[Mapping[str, Any]]
                         ) -> Mapping[str, Any] | None:
    """The mesh file source that carries every one of ``fields``.

    An exact match wins; otherwise the first source whose variables are a
    SUPERSET serves, with its ``variables`` cut to exactly ``fields`` so the
    reader reads (and the seam receives) only the statics the selection names.
    ``None`` when no single source carries them all.
    """

    wanted = set(fields)
    for source in sources:
        if wanted == set(source["variables"]):
            return source
    for source in sources:
        variables = dict(source["variables"])
        if wanted <= set(variables):
            cut = dict(source)
            cut["variables"] = {name: variables[name] for name in variables
                                if name in wanted}
            return cut
    return None
