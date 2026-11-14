"""WRF grid (grid_fdda = 1) and spectral (grid_fdda = 2) analysis nudging on hex.

The controls are WRF's &fdda keys, read by woof's own WRF FDDA configuration
(:class:`woof.spectral_nudging.config.NudgingConfig`, from a TOML ``[nudging]``
table or a WRF ``namelist.input`` through :func:`woof.spectral_nudging.namelist.translate`).
The per-point arithmetic is woof's WRF FDDA code, compiled from its own files:

* ``woof/spectral_nudging/forecast.cu`` ``wrf_weights``: the WRF v4.7.1
  module_fdda_psufddagd / module_fdda_spnudging vertical weights
  (if_no_pbl_nudging_*, if_zfac_*, k_zfac_*, dk_zfac_*);
* ``woof/spectral_nudging/kernels.cu`` ``difference`` and ``relax``: the
  analysis interpolation in time, the analysis-minus-model difference, the
  coefficient (guv, gt, gq), the fgdt/fgdtzero cadence with its held tendency
  and the gfdda_end_h / if_ramping / dtramp_min end ramp.

What hex adds is the mesh: the fields live at the cell centres (u and v are the
zonal and meridional winds of the dycore's own reconstruction,
``reconstruct_winds_v841_f32``), and the horizontal filter of spectral nudging
is the hex spherical-harmonic basis of :mod:`woof.hex.stochastic` (a
least-squares projection over the interior cells; on a limited area the
lateral-boundary zone is excluded exactly as the stochastic patterns exclude
it, and the degrees the region cannot resolve are dropped).

The tendencies enter at the IAU seat (the held physics tendencies, after the
coupling and before the dynamics) through the MPAS IAU kernels of
:mod:`woof.hex.iau_v842` with unit weight and no density increment: the
temperature and vapour tendencies by ``iau_cells_correct``, the cell wind
tendency projected to edges by the physics projection
(:func:`woof.hex.cuda_physics_v841.project_mpas_momentum_v842`) and added by
``iau_edges``.

Off is the default and touches nothing: no module is imported, no buffer is
allocated, and the step's held-forcing slot is unchanged.

Refused by name (:func:`refusals`): geopotential nudging (gph: MPAS's height
coordinate has no prognostic geopotential), surface analysis nudging
(grid_sfdda), the woof "regional" downscaling convention, a domain other than
the root, a spectral truncation finer than the basis this mesh supports, and
analyses that do not cover the nudging window.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

#: The fields hex nudges, in the WRF FDDA field order of woof.spectral_nudging.
FIELDS = ("u", "v", "temperature", "qv")
#: forecast.cu wrf_weights field index (u and v are A-grid cell fields here,
#: so they take the mass-point index with the wind controls).
_WEIGHT_FIELD = {"uv": 2, "t": 2, "q": 4}
#: Largest spherical-harmonic degree the spectral filter builds, and the most
#: basis words (cells x modes) it holds on the host while building.
MAX_SPECTRAL_DEGREE = 127
MAX_BASIS_WORDS = 3.0e8
#: MPAS limited-area rings: 1..5 relaxation zone, 6..7 specified zone.
SPECIFIED_ZONE_FIRST_RING = 6
GRAVITY_KERNEL = np.float32(9.81)
RVORD = np.float32(461.6) / np.float32(287.0)


class HexNudgingRefusal(ValueError):
    """A nudging request hex cannot honour, with its name."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code


# ----------------------------------------------------------------------------
# configuration
# ----------------------------------------------------------------------------

def load_config(path: str | Path):
    """The root domain's NudgingConfig from a TOML [nudging] table or a WRF &fdda namelist."""
    from woof.spectral_nudging.config import from_mapping

    path = Path(path)
    text = path.read_text()
    if path.suffix.lower() == ".toml":
        import tomllib

        document = tomllib.loads(text)
        row = dict(document.get("nudging", document))
        # Hex runs WRF FDDA only, so an unstated convention is WRF's (the WRF path's TOML
        # default for spectral mode is woof's regional downscaling, refused here).
        row.setdefault("convention", "wrf")
        options = from_mapping(row, domain_ids=_toml_domains(row))
    else:
        from woof.fortran_namelist import parse_namelist_text
        from woof.namelist_import import _Section
        from woof.spectral_nudging.namelist import translate

        sections = parse_namelist_text(text, source=str(path))
        if "fdda" not in sections:
            raise HexNudgingRefusal("HEX_NUDGING_NO_FDDA", f"{path} has no &fdda group")
        count = max([len(v) for v in sections["fdda"].values()] + [1])
        dropped: list[str] = []
        section = _Section("fdda", sections["fdda"], str(path))
        table = translate(section, list(range(1, count + 1)),
                          lambda group, key, entries, why: dropped.append(key))
        options = from_mapping({"domain": table["domain"]}, domain_ids=tuple(range(1, count + 1)))
    if options is None:
        return None
    domains = dict(options.domains)
    if set(domains) - {1}:
        raise HexNudgingRefusal(
            "HEX_NUDGING_DOMAIN",
            f"nudging is requested for domain(s) {sorted(set(domains) - {1})}; a hex run is one "
            "variable-resolution mesh (domain 1), and a WRF-path nest under hex nudges on the WRF path")
    return domains.get(1)


def _toml_domains(row: Mapping[str, Any]) -> tuple[int, ...]:
    extra = tuple(int(key) for key in (row.get("domain") or {}))
    return tuple(sorted({1, *extra}))


def refusals(config, *, limited_area: bool) -> list[HexNudgingRefusal]:
    """Every named reason this configuration cannot run on a hex mesh."""
    found: list[HexNudgingRefusal] = []
    if config.convention != "wrf":
        found.append(HexNudgingRefusal(
            "HEX_NUDGING_CONVENTION",
            "hex runs WRF FDDA (grid_fdda 1/2, convention='wrf'); woof's 'regional' downscaling "
            "convention filters physical temperature on WRF eta levels with a height ramp, and has no mesh form"))
    if config.grid_sfdda:
        found.append(HexNudgingRefusal(
            "HEX_NUDGING_SFDDA",
            "surface analysis nudging (grid_sfdda) reads WRF auxinput9 surface analyses and the "
            "SFDDAGD surface layer; hex stages neither"))
    if config.mode == "spectral" and (config.gph or 0) > 0:
        found.append(HexNudgingRefusal(
            "HEX_NUDGING_GPH",
            "geopotential nudging (gph) needs a prognostic geopotential; MPAS's height coordinate "
            "has none (zgrid is fixed), so set gph = 0"))
    if all(config.coefficient(name) == 0 for name in FIELDS):
        found.append(HexNudgingRefusal(
            "HEX_NUDGING_NO_FIELD", "every nudging coefficient (guv, gt, gq) is zero"))
    if config.gfdda_interval_m is None or not config.gfdda_interval_m > 0:
        found.append(HexNudgingRefusal(
            "HEX_NUDGING_INTERVAL", "gfdda_interval_m must be positive: it spaces the analyses"))
    if config.mode == "spectral" and limited_area and config.cutoff_x == 0 and config.cutoff_y == 0:
        pass  # the domain mean alone: admitted (WRF keeps the k = 0 mode)
    return found


def spectral_degree(config, mesh) -> int:
    """The total spherical-harmonic degree the hex filter keeps for xwavenum / ywavenum.

    WRF keeps the Fourier modes kx <= xwavenum, ky <= ywavenum of its domain
    rectangle, i.e. wavelengths of at least Lx/xwavenum and Ly/ywavenum.  The
    mesh has no rectangle, so hex keeps every degree l <= L with
    L = max(round(2 pi R xwavenum / Lx), round(2 pi R ywavenum / Ly)), the
    degrees whose wavelength 2 pi R / l is at least the shorter of the two:
    Lx, Ly are the zonal and meridional extents of the interior cells (a
    global mesh: Lx = 2 pi R and Ly = pi R).
    """
    a = mesh.arrays
    n = int(mesh.n_cells)
    radius = float(mesh.attrs.get("sphere_radius", 6371229.0))
    if not _limited(mesh):
        return int(max(config.cutoff_x, 2 * config.cutoff_y))
    interior = _interior_mask(mesh)
    lat = np.asarray(a["latCell"], dtype=np.float64)[:n][interior]
    lon = np.asarray(a["lonCell"], dtype=np.float64)[:n][interior]
    centre = math.atan2(np.mean(np.sin(lon)), np.mean(np.cos(lon)))
    dlon = np.angle(np.exp(1j * (lon - centre)))
    lx = radius * math.cos(float(np.mean(lat))) * float(dlon.max() - dlon.min())
    ly = radius * float(lat.max() - lat.min())
    degrees = [0]
    for count, extent in ((config.cutoff_x, lx), (config.cutoff_y, ly)):
        if count > 0:
            if not extent > 0:
                raise HexNudgingRefusal("HEX_SPECTRAL_NUDGING_DEGREE", "the interior has no extent")
            degrees.append(int(round(2 * math.pi * radius * count / extent)))
    return max(degrees)


def _limited(mesh) -> bool:
    a = mesh.arrays
    return "bdyMaskCell" in a and bool(np.any(np.asarray(a["bdyMaskCell"])[:int(mesh.n_cells)] > 0))


def _interior_mask(mesh) -> np.ndarray:
    a = mesh.arrays
    n = int(mesh.n_cells)
    if "bdyMaskCell" not in a:
        return np.ones(n, dtype=bool)
    return np.asarray(a["bdyMaskCell"])[:n] == 0


def nudged_cell_mask(mesh, padded_cells: int) -> np.ndarray:
    """1 where a tendency may be added: every real cell outside the specified zone."""
    n = int(mesh.n_cells)
    mask = np.zeros(int(padded_cells), dtype=np.float32)
    keep = np.ones(n, dtype=bool)
    if "bdyMaskCell" in mesh.arrays:
        keep = np.asarray(mesh.arrays["bdyMaskCell"])[:n] < SPECIFIED_ZONE_FIRST_RING
    mask[:n] = keep
    return mask


# ----------------------------------------------------------------------------
# analyses (hex init-format frames on this mesh, e.g. from woof.hex.analysis_remap)
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class AnalysisFrame:
    path: Path
    valid_time: str
    offset_seconds: float
    sha256: str


def _xtime(dataset) -> str:
    raw = np.asarray(dataset["xtime"][0]).tobytes().decode("ascii").strip("\x00 ")
    datetime.strptime(raw, "%Y-%m-%d_%H:%M:%S")
    return raw


def analysis_frames(paths: Sequence[str | Path], *, start_time: str) -> list[AnalysisFrame]:
    """Each analysis's valid time, as seconds after the run's initial time, in order."""
    from netCDF4 import Dataset

    start = datetime.strptime(start_time.replace("T", "_"), "%Y-%m-%d_%H:%M:%S")
    frames = []
    for raw in paths:
        path = Path(raw).expanduser().resolve(strict=True)
        with Dataset(path) as d:
            missing = [name for name in ("xtime", "theta", "qv", "u") if name not in d.variables]
            if missing:
                raise HexNudgingRefusal("HEX_NUDGING_ANALYSIS_FIELDS",
                                        f"{path} lacks {missing}; analyses are hex init-format frames "
                                        "(woof.hex.analysis_remap writes them)")
            valid = _xtime(d)
        offset = (datetime.strptime(valid, "%Y-%m-%d_%H:%M:%S") - start).total_seconds()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        frames.append(AnalysisFrame(path, valid, float(offset), digest))
    frames.sort(key=lambda frame: frame.offset_seconds)
    return frames


def window_seconds(config, run_seconds: float) -> float:
    """The last model time the nudging reads an analysis at (WRF gfdda_end_h, or the run's end)."""
    end = run_seconds if config.gfdda_end_h is None else min(run_seconds, config.gfdda_end_h * 3600.0)
    return float(end)


def check_frames(config, frames: Sequence[AnalysisFrame], *, run_seconds: float, dt: float) -> None:
    """WRF's analysis clock: frames at 0, interval, 2 interval, ... covering the window."""
    interval = float(config.gfdda_interval_m) * 60.0
    if not math.isclose(math.fmod(interval, dt), 0.0, abs_tol=1e-9) and not math.isclose(
            math.fmod(interval, dt), dt, abs_tol=1e-9):
        raise HexNudgingRefusal("HEX_NUDGING_INTERVAL_DT",
                                f"gfdda_interval_m = {config.gfdda_interval_m} is not a whole number of "
                                f"{dt:g} s steps, so a step would straddle two analysis intervals")
    if len(frames) < 2:
        raise HexNudgingRefusal("HEX_NUDGING_ANALYSES_SHORT",
                                "nudging needs at least two analyses (the start and end of an interval)")
    for index, frame in enumerate(frames):
        if not math.isclose(frame.offset_seconds, index * interval, abs_tol=1e-6):
            raise HexNudgingRefusal(
                "HEX_NUDGING_ANALYSIS_CLOCK",
                f"analysis {index} ({frame.path.name}) is valid at {frame.valid_time}, "
                f"{frame.offset_seconds:g} s after the start; WRF reads analyses at the start and "
                f"every gfdda_interval_m = {config.gfdda_interval_m} min")
    need = window_seconds(config, run_seconds)
    if frames[-1].offset_seconds < need:
        raise HexNudgingRefusal(
            "HEX_NUDGING_ANALYSES_SHORT",
            f"the analyses end {frames[-1].offset_seconds:g} s after the start and the nudging "
            f"runs to {need:g} s; add analyses or set gfdda_end_h")


def read_analysis(frame: AnalysisFrame, *, n_cells: int, n_edges: int, n_levels: int) -> dict[str, np.ndarray]:
    """theta, qv [level, cell] and u [level, edge] of one frame, float32, mesh order."""
    from netCDF4 import Dataset

    out = {}
    with Dataset(frame.path) as d:
        for name, entity, count in (("theta", "nCells", n_cells), ("qv", "nCells", n_cells),
                                    ("u", "nEdges", n_edges)):
            variable = d[name]
            dims = [dim for dim in variable.dimensions if dim != "Time"]
            index = tuple(0 if dim == "Time" else slice(None) for dim in variable.dimensions)
            value = np.asarray(variable[index], dtype=np.float32)
            if dims == [entity, "nVertLevels"]:
                value = value.T
            elif dims != ["nVertLevels", entity]:
                raise HexNudgingRefusal("HEX_NUDGING_ANALYSIS_FIELDS",
                                        f"{frame.path.name}:{name} has dimensions {variable.dimensions}")
            if value.shape != (n_levels, count):
                raise HexNudgingRefusal(
                    "HEX_NUDGING_ANALYSIS_MESH",
                    f"{frame.path.name}:{name} is {value.shape}; this mesh is {(n_levels, count)}: "
                    "remap the analysis onto the run's mesh (woof.hex.analysis_remap)")
            if not np.all(np.isfinite(value)):
                raise HexNudgingRefusal("HEX_NUDGING_ANALYSIS_FIELDS", f"{frame.path.name}:{name} is not finite")
            out[name] = np.ascontiguousarray(value)
    return out


# ----------------------------------------------------------------------------
# kernels: woof's WRF FDDA sources
# ----------------------------------------------------------------------------

_FDDA_SOURCE = Path(__file__).resolve().parents[1] / "spectral_nudging" / "kernels.cu"
_WEIGHT_SOURCE = Path(__file__).resolve().parents[1] / "spectral_nudging" / "forecast.cu"
#: cuComplex.h is NVRTC's; a facade backend (Metal, host) compiles the same
#: entry points with float2 standing in for cuFloatComplex.  The device timing
#: stamp (PTX %globaltimer) is not one of the entry points hex launches.
_COMPLEX_SHIM = ("typedef float2 cuFloatComplex;\n"
                 "__device__ inline cuFloatComplex make_cuFloatComplex(float r, float i) "
                 "{ return make_float2(r, i); }\n"
                 "__device__ inline float cuCrealf(cuFloatComplex z) { return z.x; }\n")
_TIMING_MARK = "// Device-side timing samples"
_MODULES: dict[Any, Any] = {}


def fdda_sources() -> tuple[str, str]:
    """The two WRF FDDA sources as this backend compiles them."""
    from woof import backend

    fdda, weights = _FDDA_SOURCE.read_text(), _WEIGHT_SOURCE.read_text()
    if backend.active() in backend.FACADE_BACKENDS:
        head, mark, _ = fdda.partition(_TIMING_MARK)
        if not mark or "#include <cuComplex.h>" not in head:
            raise RuntimeError("woof/spectral_nudging/kernels.cu changed layout; update the hex facade shim")
        fdda = head.replace("#include <cuComplex.h>", _COMPLEX_SHIM)
    return fdda, weights


def fdda_kernels():
    import cupy as cp
    from woof.certify.kernel_manifest import record_module

    key = cp.cuda.Device().id
    if key not in _MODULES:
        fdda, weights = fdda_sources()
        options = ("--std=c++11",)
        first = cp.RawModule(code=fdda, options=options)
        record_module("woof.hex.nudging:wrf_fdda", source=fdda, options=options, module=first)
        second = cp.RawModule(code=weights, options=options)
        record_module("woof.hex.nudging:wrf_weights", source=weights, options=options, module=second)
        _MODULES[key] = {"difference": first.get_function("difference"),
                         "relax": first.get_function("relax"),
                         "wrf_weights": second.get_function("wrf_weights")}
    return _MODULES[key]


def _theta_kernel():
    import cupy as cp

    key = ("theta", cp.cuda.Device().id)
    if key not in _MODULES:
        # forecast.cu pack_fields' theta_m conversion, word for word.
        _MODULES[key] = cp.ElementwiseKernel(
            "float32 theta_m, float32 qv", "float32 theta",
            "theta = theta_m/(1.f+(461.6f/287.f)*qv);", "hex_nudging_dry_theta")
    return _MODULES[key]


def _coupling_kernels(cache):
    """woof.hex.iau_v842's IAU kernels: its NVRTC route on CUDA, the same source and options on a facade."""
    from woof import backend

    if backend.active() not in backend.FACADE_BACKENDS:
        from .iau_v842 import iau_kernels

        return iau_kernels(cache)
    import cupy as cp
    from woof.certify.kernel_manifest import record_module
    from . import numerics as hex_numerics
    from .iau_v842 import _CUDA_SOURCE

    key = ("iau", cp.cuda.Device().id)
    if key not in _MODULES:
        options = hex_numerics.options(("--std=c++17", "--fmad=false", "--ftz=false",
                                         "--prec-div=true", "--prec-sqrt=true"))
        module = cp.RawModule(code=_CUDA_SOURCE, options=options)
        record_module("hexcore.iau_v842", source=_CUDA_SOURCE, options=options, module=module)
        _MODULES[key] = {name: module.get_function(name) for name in ("iau_edges", "iau_cells_correct")}
    return _MODULES[key]


# ----------------------------------------------------------------------------
# the spectral filter: the hex spherical-harmonic basis
# ----------------------------------------------------------------------------

class HarmonicFilter:
    """Least-squares projection onto the spherical harmonics of degree <= L.

    The basis is the stochastic patterns' harmonic synthesis
    (:class:`woof.hex.stochastic.PointHarmonics`, orthonormal real harmonics
    at the cell centres) plus the constant.  The fit is weighted by cell area
    over the interior cells; a limited area excludes its lateral-boundary zone
    (bdyMaskCell > 0), as the stochastic patterns do, and keeps only the
    combinations of modes the region resolves: the singular vectors of the
    area-weighted basis whose squared singular value is above ``rcond`` of the
    largest (a global mesh keeps every mode; a region keeps its well
    concentrated, Slepian-like combinations).  The projection is stored as
    two thin factors, so ``apply`` (float32 [level, cell] to the filtered
    field on every real cell) costs O(cells x kept) per level.
    """

    def __init__(self, mesh, degree: int, *, rcond: float = 1.0e-6):
        n = int(mesh.n_cells)
        modes = (int(degree) + 1) ** 2
        if not 0 <= degree <= MAX_SPECTRAL_DEGREE or n * modes > MAX_BASIS_WORDS:
            raise HexNudgingRefusal(
                "HEX_SPECTRAL_NUDGING_DEGREE",
                f"xwavenum/ywavenum ask for spherical-harmonic degree {degree} ({modes} modes) on {n} cells; the "
                f"hex filter builds degrees up to {MAX_SPECTRAL_DEGREE} and at most {MAX_BASIS_WORDS:.0e} basis "
                "words (a limited area must span at least the kept wavelength 2 pi R / L); lower the "
                "wavenumbers or use grid nudging (grid_fdda = 1)")
        import cupy as cp

        columns = [np.full((n, 1), 1.0 / math.sqrt(4.0 * math.pi))]
        if degree >= 1:
            from .stochastic import PointHarmonics

            resource = PointHarmonics(np.asarray(mesh.arrays["latCell"], dtype=np.float64)[:n],
                                      np.asarray(mesh.arrays["lonCell"], dtype=np.float64)[:n], degree,
                                      float(mesh.attrs.get("sphere_radius", 6371229.0)))
            columns.append(cp.asnumpy(resource.basis).astype(np.float64))
            del resource
        basis = np.ascontiguousarray(np.concatenate(columns, axis=1))
        del columns
        interior = _interior_mask(mesh)
        area = np.asarray(mesh.arrays["areaCell"], dtype=np.float64)[:n] * interior
        root = np.sqrt(area / area.sum())[interior]
        weighted = basis[interior] * root[:, None]
        if weighted.shape[1] <= weighted.shape[0]:
            values, right = np.linalg.eigh(weighted.T @ weighted)
            keep = values > rcond * values.max()
            sigma = np.sqrt(values[keep])
            right = right[:, keep]
            left = (weighted @ right) / sigma
        else:
            values, left = np.linalg.eigh(weighted @ weighted.T)
            keep = values > rcond * values.max()
            sigma = np.sqrt(values[keep])
            left = left[:, keep]
            right = (weighted.T @ left) / sigma
        synthesis = (basis @ right) / sigma                       # [cell, kept]
        analysis = np.zeros((left.shape[1], n))                   # [kept, cell]
        analysis[:, interior] = (left * root[:, None]).T
        self.degree = int(degree)
        self.modes = int(basis.shape[1])
        self.kept = int(keep.sum())
        self.cells = n
        self.limited = bool(not interior.all())
        self.cp = cp
        self._synthesis_t = cp.asarray(np.ascontiguousarray(synthesis.T, dtype=np.float32))
        self._analysis_t = cp.asarray(np.ascontiguousarray(analysis.T, dtype=np.float32))

    def apply(self, field):
        """field float32 [level, >= cells] -> the filtered [level, cells]."""
        coefficients = self.cp.ascontiguousarray(field[:, :self.cells]) @ self._analysis_t
        return coefficients @ self._synthesis_t

    def receipt(self) -> dict[str, Any]:
        return {"basis": "woof.hex.stochastic.PointHarmonics real spherical harmonics plus the constant",
                "degree": self.degree, "modes": self.modes, "modes_kept": self.kept,
                "fit": ("area-weighted least squares over the interior cells (bdyMaskCell = 0)"
                        if self.limited else "area-weighted least squares over every cell")}


# ----------------------------------------------------------------------------
# the forcing
# ----------------------------------------------------------------------------

def _host_time_weight(elapsed: float, end: float, ramp: float) -> float:
    """kernels.cu time_weight, on the host, to skip a step the kernels would not force."""
    if end < 0.0:
        return 1.0
    stop = end + max(0.0, ramp)
    if ramp != 0.0:
        return min(1.0, max(0.0, (stop - elapsed) / abs(ramp)))
    return 1.0 if elapsed <= end else 0.0


class HexNudging:
    """Grid or spectral WRF FDDA on hex cell columns, added at the IAU seat.

    ``add`` has the held-forcing signature of woof.hex.iau_native /
    woof.hex.iau_forcing; ``tendencies`` returns this step's per-cell
    nudging tendencies (du, dv in m s-2, dtheta in K s-1, dqv in kg kg-1 s-1)
    without adding them, for the per-column comparison with WRF.
    """

    def __init__(self, config, *, frames: Sequence[AnalysisFrame], mesh, prep_geometry,
                 physics_geometry, zgrid, scalar_names: Sequence[str], dt: float,
                 kernel_cache, run_seconds: float, pbl_height=None):
        import cupy as cp

        problems = refusals(config, limited_area=_limited(mesh))
        if problems:
            raise problems[0]
        check_frames(config, frames, run_seconds=run_seconds, dt=dt)
        self.config, self.frames, self.mesh = config, list(frames), mesh
        self.cp = cp
        self.dt = float(dt)
        self.cache = kernel_cache
        self.prep_geometry, self.physics_geometry = prep_geometry, physics_geometry
        self.names = tuple(scalar_names)
        if "qv" not in self.names:
            raise HexNudgingRefusal("HEX_NUDGING_SCALARS", "the run carries no qv")
        self.index_qv = self.names.index("qv")
        self.nc = int(prep_geometry.n_cells)
        self.ne = int(prep_geometry.n_edges)
        z = np.asarray(zgrid, dtype=np.float32)
        if z.ndim != 2 or z.shape[1] not in (int(mesh.n_cells), self.nc):
            raise ValueError("zgrid must be float32 [level + 1, cell]")
        self.nz = int(z.shape[0]) - 1
        padded = np.zeros((self.nz + 1, self.nc), dtype=np.float32)
        padded[:, :z.shape[1]] = z
        if z.shape[1] < self.nc:
            padded[:, z.shape[1]:] = z[:, -1:]
        # forecast.cu reads heights as geopotential / 9.81f over the terrain.
        self.phb = cp.asarray(np.ascontiguousarray(padded * GRAVITY_KERNEL))
        self.php = cp.zeros_like(self.phb)
        self.ht = cp.asarray(np.ascontiguousarray(padded[0]))
        self.mask = cp.asarray(nudged_cell_mask(mesh, self.nc)[None, :])
        self.active = tuple(name for name in FIELDS if config.coefficient(name) > 0)
        self.uses_pbl = any(getattr(config, "if_no_pbl_nudging_" + suffix)
                            for suffix in ("uv", "t", "q"))
        self.pbl_height = pbl_height
        if self.uses_pbl and pbl_height is None:
            raise HexNudgingRefusal(
                "HEX_NUDGING_PBL_HEIGHT",
                "if_no_pbl_nudging_* needs the PBL height, and this route publishes none")
        self.interval = float(config.gfdda_interval_m) * 60.0
        self.cadence = int(config.cadence_steps(self.dt))
        self.end = -1.0 if config.gfdda_end_h is None else float(config.gfdda_end_h) * 3600.0
        self.ramp = float(config.dtramp_min) * 60.0 if config.if_ramping else 0.0
        self.filter = None
        if config.mode == "spectral":
            self.filter = HarmonicFilter(mesh, spectral_degree(config, mesh))
        self.kernels = fdda_kernels()
        shape = (self.nz, self.nc)
        self.clock = cp.zeros(4, dtype=cp.float64)
        self.error = cp.zeros(1, dtype=cp.uint32)
        self.work = {name: cp.zeros(shape + (2,), dtype=cp.float32) for name in self._work_names()}
        self.held = {name: cp.zeros(shape, dtype=cp.float32) for name in self.active}
        self.scratch = {name: cp.zeros(shape, dtype=cp.float32) for name in self.active}
        self.weights: dict[str, Any] = {}
        self._wind = {name: cp.zeros(shape, dtype=cp.float32) for name in ("x", "y", "z", "u", "v")}
        self._invalid = cp.zeros(1, dtype=cp.int32)
        self._theta_amb = cp.zeros(shape, dtype=cp.float32)
        self._scalars_amb = cp.zeros((self.index_qv + 1,) + shape, dtype=cp.float32)
        self._rho_amb = cp.zeros(shape, dtype=cp.float32)
        self._unit_zz = None
        self._host_frames: dict[int, dict[str, np.ndarray]] = {}
        self._pair: tuple[int, dict[str, Any], dict[str, Any]] | None = None
        self.calls = self.forced_steps = 0
        self.first_time = self.last_time = None

    def _work_names(self) -> tuple[str, ...]:
        names = []
        if "u" in self.active or "v" in self.active:
            names += ["x", "y", "z"] if self.filter is not None else ["u", "v"]
        names += [name for name in ("temperature", "qv") if name in self.active]
        return tuple(names)

    # -- analyses ------------------------------------------------------------
    def _frame_device(self, index: int) -> dict[str, Any]:
        cp = self.cp
        host = self._host_frames.get(index)
        if host is None:
            raw = read_analysis(self.frames[index], n_cells=int(self.mesh.n_cells),
                                n_edges=int(self.mesh.n_edges), n_levels=self.nz)
            host = {name: self._pad(value, self.nc if name != "u" else self.ne) for name, value in raw.items()}
            self._host_frames[index] = host
        device = {"temperature": cp.asarray(host["theta"]), "qv": cp.asarray(host["qv"])}
        winds = self._reconstruct(cp.asarray(host["u"]))
        device.update({name: winds[name].copy() for name in ("x", "y", "z", "u", "v")})
        return device

    @staticmethod
    def _pad(value: np.ndarray, count: int) -> np.ndarray:
        if value.shape[1] == count:
            return value
        out = np.zeros((value.shape[0], count), dtype=np.float32)
        out[:, :value.shape[1]] = value
        return out

    def _bracket(self, elapsed: float) -> int:
        frames = len(self.frames)
        if self.end >= 0.0 and elapsed >= self.end:
            index = int(math.floor(self.end / self.interval + 1e-9))
        else:
            index = int(math.floor(elapsed / self.interval + 1e-9))
        return max(0, min(index, frames - 2))

    def _endpoints(self, index: int):
        if self._pair is None or self._pair[0] != index:
            reuse = self._pair[2] if self._pair is not None and self._pair[0] + 1 == index else None
            start = reuse if reuse is not None else self._frame_device(index)
            self._pair = (index, start, self._frame_device(index + 1))
        return self._pair[1], self._pair[2]

    # -- model fields --------------------------------------------------------
    def _reconstruct(self, normal_velocity):
        from .cuda_physics_prep_v841 import _launch

        geometry = self.prep_geometry
        wind = self._wind
        self._invalid.fill(0)
        _launch(self.cache, "reconstruct_winds_v841_f32", self.nc, (
            np.int32(self.nz), np.int32(self.nc), np.int32(self.ne), np.int32(geometry.max_edges),
            normal_velocity, geometry._edges_on_cell, geometry._n_edges_on_cell,
            geometry._coeffs_reconstruct, geometry._lat_cell, geometry._lon_cell,
            geometry._sealed_edges_on_cell, geometry._sealed_n_edges_on_cell,
            geometry._sealed_coeffs_reconstruct, geometry._sealed_lat_cell,
            geometry._sealed_lon_cell, wind["x"], wind["y"], wind["z"], wind["u"], wind["v"],
            self._invalid))
        return wind

    def _vertical_weights(self, pblh):
        cp = self.cp
        cfg = self.config
        if self.weights and not self.uses_pbl:
            return self.weights
        if pblh is None:
            pblh = cp.zeros((self.nc,), dtype=cp.float32)
        else:
            pblh = cp.ascontiguousarray(cp.asarray(pblh, dtype=cp.float32).reshape(-1))
            if pblh.size < self.nc:
                pblh = cp.concatenate([pblh, cp.zeros((self.nc - pblh.size,), dtype=cp.float32)])
            elif pblh.size > self.nc:
                raise ValueError("PBL height has more columns than the mesh")
        block = (256,)
        for suffix in ("uv", "t", "q"):
            needed = {"uv": ("u", "v"), "t": ("temperature",), "q": ("qv",)}[suffix]
            if not any(name in self.active for name in needed):
                continue
            weight = self.weights.get(suffix)
            if weight is None:
                weight = self.weights[suffix] = cp.empty((self.nz, self.nc), dtype=cp.float32)
            controls = tuple(np.int32(v) for v in (
                _WEIGHT_FIELD[suffix], cfg.mode == "grid", getattr(cfg, "if_no_pbl_nudging_" + suffix),
                getattr(cfg, "if_zfac_" + suffix), getattr(cfg, "k_zfac_" + suffix),
                getattr(cfg, "dk_zfac_" + suffix)))
            self.kernels["wrf_weights"](((self.nc + 255) // 256,), block, (
                self.php, self.phb, self.ht, pblh, weight,
                np.int32(self.nz), np.int32(1), np.int32(self.nc), np.int32(1), *controls))
            weight *= self.mask
        return self.weights

    # -- the step --------------------------------------------------------------
    def tendencies(self, *, time_seconds: float, theta_m, scalars, normal_velocity, pblh=None):
        """This step's nudging tendencies on the cell columns (None when the window is over)."""
        cp = self.cp
        cfg = self.config
        elapsed = float(time_seconds)
        if _host_time_weight(elapsed, self.end, self.ramp) <= 0.0:
            return None
        index = self._bracket(elapsed)
        a, b = self._endpoints(index)
        self.clock.set(np.asarray([elapsed, self.dt, index * self.interval, (index + 1) * self.interval],
                                  dtype=np.float64))
        weights = self._vertical_weights(pblh)
        model: dict[str, Any] = {}
        if "temperature" in self.active:
            _theta_kernel()(theta_m, scalars[self.index_qv], self.scratch["temperature"])
            model["temperature"] = self.scratch["temperature"]
        if "qv" in self.active:
            self.scratch["qv"][...] = scalars[self.index_qv]
            model["qv"] = self.scratch["qv"]
        if "u" in self.active or "v" in self.active:
            wind = self._reconstruct(normal_velocity)
            for name in (("x", "y", "z") if self.filter is not None else ("u", "v")):
                model[name] = wind[name]
        n = self.nz * self.nc
        grid, block = ((n + 255) // 256,), (256,)
        wrf = np.int32(1)
        cadence = np.int32(self.cadence)
        end, ramp = np.float64(self.end), np.float64(self.ramp)
        grid_mode = np.int32(cfg.mode == "grid")
        for name in self.work:
            self.kernels["difference"](grid, block, (
                model[name], a[name], b[name], self.work[name], self.clock, self.error,
                np.int32(n), wrf, cadence, end, ramp, grid_mode))
        filtered: dict[str, Any] = {}
        if self.filter is None:
            for name in self.work:
                filtered[name] = self.work[name]
        else:
            for name in self.work:
                smooth = self.filter.apply(self.work[name][..., 0])
                out = cp.zeros_like(self.work[name])
                out[:, :self.filter.cells, 0] = smooth
                filtered[name] = out
            if "x" in filtered:
                east, north = self._east_north(filtered["x"][..., 0], filtered["y"][..., 0],
                                               filtered["z"][..., 0])
                for name, value in (("u", east), ("v", north)):
                    out = cp.zeros((self.nz, self.nc, 2), dtype=cp.float32)
                    out[..., 0] = value
                    filtered[name] = out
        dummy = self.ht
        for name in self.active:
            suffix = {"u": "uv", "v": "uv", "temperature": "t", "qv": "q"}[name]
            # relax also adds dt * held to its first argument; a scratch copy takes it.
            value = self.scratch[name]
            self.kernels["relax"](grid, block, (
                value, filtered[name], dummy, dummy, self.clock,
                np.int32(n), np.int32(self.nc), np.int32(1), np.int32(0),
                np.float32(cfg.minimum_height_m), np.float32(cfg.ramp_depth_m),
                np.float32(1.0 / cfg.coefficient(name)), weights[suffix], self.held[name],
                wrf, cadence, np.int32(cfg.fgdtzero), end, ramp))
        self.calls += 1
        self.first_time = elapsed if self.first_time is None else self.first_time
        self.last_time = elapsed
        return dict(self.held)

    def _east_north(self, x, y, z):
        cp = self.cp
        geometry = self.prep_geometry
        lat, lon = geometry._lat_cell, geometry._lon_cell
        clat, slat, clon, slon = cp.cos(lat), cp.sin(lat), cp.cos(lon), cp.sin(lon)
        east = -x * slon + y * clon
        north = -(x * clon + y * slon) * slat + z * clat
        return east.astype(cp.float32), north.astype(cp.float32)

    def add(self, index: int, held: Any, start: float, dt: float, *, driver: Any = None,
            rho_edge: Any, normal_velocity: Any, state: Any, theta_m: Any = None,
            pblh: Any = None, **_unused: Any) -> None:
        """This step's nudging tendencies onto the held physics tendencies (in place)."""
        cp = self.cp
        if float(dt) != self.dt:
            raise ValueError("nudging was built for a fixed time step")
        if theta_m is None:
            theta_m = cp.ascontiguousarray(state.rho_theta / state.rho)
        if pblh is None and self.pbl_height is not None:
            pblh = self.pbl_height()
        tend = self.tendencies(time_seconds=start, theta_m=theta_m, scalars=state.scalars,
                               normal_velocity=normal_velocity, pblh=pblh)
        if tend is None:
            return
        cache = self.cache if driver is None else driver.cache
        self.couple(tend, held=held, rho_zz=state.rho, theta_m=theta_m, scalars=state.scalars,
                    rho_edge=rho_edge, cache=cache)
        self.forced_steps += 1

    def couple(self, tend: Mapping[str, Any], *, held, rho_zz, theta_m, scalars, rho_edge, cache):
        """The MPAS IAU kernels with unit weight: d(rho theta_m), d(rho qv) and d(rho_edge u)."""
        cp = self.cp
        kernels = _coupling_kernels(cache)
        nl, nc = self.nz, self.nc
        if "temperature" in tend or "qv" in tend:
            # zz divides only the (zero) density increment, so unit zz changes no word.
            if self._unit_zz is None:
                self._unit_zz = cp.ones((nl, nc), dtype=cp.float32)
            self._theta_amb[...] = tend["temperature"] if "temperature" in tend else 0
            vapour = "qv" in tend
            if vapour:
                self._scalars_amb[self.index_qv] = tend["qv"]
            qv1 = self.index_qv + 1
            kernels["iau_cells_correct"](((nl * nc + 127) // 128,), (128,), (
                np.int32(nl), np.int32(nc), np.int32(nc),
                np.int32(qv1 - 1 if vapour else 0), np.int32(qv1 if vapour else 0),
                np.int32(qv1 - 1), np.float32(1.0), RVORD,
                self._unit_zz, rho_zz, theta_m, scalars,
                self._rho_amb, self._theta_amb, self._scalars_amb,
                held.rho, held.rho_theta, held.scalars))
        if "u" in tend or "v" in tend:
            from .cuda_physics_v841 import project_mpas_momentum_v842

            zero = cp.zeros((nl, nc), dtype=cp.float32)
            edges = project_mpas_momentum_v842(
                geometry=self.physics_geometry, zonal=tend.get("u", zero),
                meridional=tend.get("v", zero), invalid_flag=cp.zeros(1, dtype=cp.int32),
                kernel_cache=cache)
            ne = int(held.rho_u.shape[1])
            kernels["iau_edges"](((nl * ne + 127) // 128,), (128,), (
                np.int32(nl), np.int32(ne), np.int32(ne), np.float32(1.0), rho_edge, edges, held.rho_u))

    def receipt(self) -> dict[str, Any]:
        cfg = self.config
        return {
            "schema": "gpuwm-hex.nudging/v1",
            "mode": "grid (grid_fdda = 1)" if cfg.mode == "grid" else "spectral (grid_fdda = 2)",
            "fields": list(self.active),
            "coefficients": {name: float(cfg.coefficient(name)) for name in self.active},
            "controls": {key: getattr(cfg, key) for key in (
                "if_no_pbl_nudging_uv", "if_no_pbl_nudging_t", "if_no_pbl_nudging_q",
                "if_zfac_uv", "if_zfac_t", "if_zfac_q", "k_zfac_uv", "k_zfac_t", "k_zfac_q",
                "dk_zfac_uv", "dk_zfac_t", "dk_zfac_q", "fgdt", "fgdtzero", "gfdda_interval_m",
                "gfdda_end_h", "if_ramping", "dtramp_min", "cutoff_x", "cutoff_y")},
            "cadence_steps": self.cadence,
            "analyses": [{"path": str(f.path), "valid_time": f.valid_time,
                          "offset_seconds": f.offset_seconds, "sha256": f.sha256} for f in self.frames],
            "filter": None if self.filter is None else self.filter.receipt(),
            "kernels": "woof/spectral_nudging kernels.cu (difference, relax), forecast.cu (wrf_weights); "
                       "coupling woof.hex.iau_v842 iau_cells_correct / iau_edges, weight 1",
            "seat": "held physics tendencies after the coupling, before the dynamics (the IAU seat)",
            "calls": self.calls, "forced_steps": self.forced_steps,
            "first_time_seconds": self.first_time, "last_time_seconds": self.last_time,
            "analysis_interval_error": int(self.error.get()[0]),
        }


def _seam_pbl_height(backend):
    """The WOOF column route's PBL height (the seam's fields/pblh), read at each forced step."""
    import cupy as cp

    seam = getattr(backend, "_seam", None)
    manifest = getattr(seam, "_restart_manifest", None)
    if manifest is None or "fields/pblh" not in manifest():
        return None

    def read():
        value = manifest()["fields/pblh"]
        return cp.asarray(value, dtype=cp.float32).reshape(-1)
    return read


def attach_v841(request: Mapping[str, Any], *, stack: Mapping[str, Any], host: Mapping[str, Any],
                cache: Any, dt: float, run_seconds: float, restart_seconds: float = 0.0) -> HexNudging:
    """The fixed-step WOOF column route (R1): build the forcing for this stack."""
    config = load_config(request["config"])
    if config is None:
        raise HexNudgingRefusal("HEX_NUDGING_OFF", f"{request['config']} enables no nudging; drop --nudging")
    if restart_seconds > 0 and config.cadence_steps(dt) > 1:
        raise HexNudgingRefusal(
            "HEX_NUDGING_RESTART_CADENCE",
            "a restart cannot rebuild the tendency fgdt holds between calculations; run with fgdt = 0 "
            "(every step) or start cold")
    driver = stack["driver"]
    prepared = host["prepared"]
    if bool(getattr(driver.config, "config_local_timestep", False)):
        raise HexNudgingRefusal("HEX_NUDGING_LTS", "nudging runs on fixed-step hex runs; drop --local-timestep")
    frames = analysis_frames(request["analyses"], start_time=str(request["start_time"]))
    return HexNudging(
        config, frames=frames, mesh=prepared.mesh, prep_geometry=stack["prep_geometry"],
        physics_geometry=stack["physics_geometry"], zgrid=prepared.vertical.zgrid,
        scalar_names=host["scalar_names"], dt=dt, kernel_cache=cache, run_seconds=run_seconds,
        pbl_height=_seam_pbl_height(stack["backend"]))


def attach_v842(arguments: Any, *, mesh, geometry_mesh, prep_geometry, zgrid, names: Sequence[str], dt: float,
                cache: Any, run_seconds: float, start_time: str, restarted: bool, partitioned: bool,
                pbl_height) -> HexNudging:
    """The MPAS v8.4.2 suite route (R2): the forcing at its IAU seat (runtime_v842 stage sources)."""
    from .cuda_physics_v841 import CudaPhysicsGeometryV841

    config = load_config(arguments.nudging)
    if config is None:
        raise HexNudgingRefusal("HEX_NUDGING_OFF", f"{arguments.nudging} enables no nudging; drop --nudging")
    if partitioned:
        raise HexNudgingRefusal("HEX_NUDGING_PARTITIONED",
                                "nudging runs on one card on a fixed mesh; the partitioned, rate-class and moving executors have no "
                                "nudging seat (pin one device, no local time step)")
    if restarted and config.cadence_steps(dt) > 1:
        raise HexNudgingRefusal(
            "HEX_NUDGING_RESTART_CADENCE",
            "a restart cannot rebuild the tendency fgdt holds between calculations; run with fgdt = 0 "
            "(every step) or start cold")
    frames = analysis_frames(getattr(arguments, "nudging_analysis", None) or (), start_time=str(start_time))
    return HexNudging(
        config, frames=frames, mesh=mesh, prep_geometry=prep_geometry,
        physics_geometry=CudaPhysicsGeometryV841.from_host(geometry_mesh), zgrid=zgrid, scalar_names=names,
        dt=dt, kernel_cache=cache, run_seconds=run_seconds, pbl_height=pbl_height)
