"""WRF observation nudging (obs_nudge_opt = 1) on hex: one mesh cell is one WRF column.

The controls are WRF's &fdda observation keys (obs_nudge_wind/temp/mois/pstr,
obs_coef_*, obs_rinxy, obs_rinsig, obs_twindo, obs_npfi, obs_ionf,
obs_idynin/obs_dtramp, obs_prt_*, obs_sfcfact/sfcfacr, obs_no_pbl_nudge_*,
obs_sfc_scheme_*, obs_nudgez*, obs_dpsmx, obs_max_sndng_gap, max_obs,
fdda_start/fdda_end), read from a WRF ``namelist.input`` &fdda group (woof's
WRF route reader, :func:`woof.fddaobs.namelist.consume`) or a TOML table
(``[fdda]``, or the WRF experiment's ``[run]``), and checked by the WRF route's
own :func:`woof.fddaobs.config.validate`.

Observations are WRF ``OBS_DOMAIN101`` files (and their 02, 03, ... volumes),
decoded and admitted by the WRF route's Rust reader (``rw_fddaobs``: native
formats, QC, surface-pressure reconstruction, RH to mixing ratio, duplicate
replacement, the max_obs window check), in its geographic mode (map_proj 0:
the carrier's i, j are longitude and latitude, with no projection, domain trim
or wind rotation).

The arithmetic is WRF v4.7.1's module_fddaobs_rtfdda.F as woof's WRF route
compiles it: ``woof/fddaobs/forecast.cu`` ``fddaobs_clock`` (the obs_ionf
cadence, the twindo report window, fdda_start/fdda_end and the obs_idynin
ramp) and ``woof/fddaobs/kernels.cu`` (``timewt``, ``fddaobs_vif`` and the
Controls/Frame records) unchanged, and ``woof/hex/obs_nudging.cu``, which is
ERROB and NUDOB's cell schedule expression for expression with the mesh in
place of the grid.  What the unstructured mesh changes (each is documented in
the kernel file too):

* **Distances.** WRF's horizontal weight (rinxy, the sfcfacr / rinfmn..rinfmx
  radius, the dpsmx pressure-difference term) uses the planar grid distance in
  grid lengths, (i - ri)^2 + (j - rj)^2, i.e. map-projected distance / dx.  Hex
  uses the great-circle distance on the mesh sphere in units of
  :data:`DISTANCE_UNIT_M` (so rinxy is still kilometres), from a cell KD-tree:
  no map factor, no grid staggering, no i/j bounding boxes.  Where the two
  distances agree the weights are the same words (tests/test_hex_obs_nudging_gpu.py).
* **Interpolation to the report.** ERROB's bilinear interpolation on the
  (u, v or mass) grid is the barycentric interpolation on the Delaunay triangle
  of the three cell centres around the report (gnomonic barycentrics); u, v and
  mass all live at cell centres.  The report's PBL level is the containing
  cell's (the nearest cell, WRF's nearest grid point).  At a cell centre the two
  agree word for word.
* **Fields.** u, v are the dycore's zonal and meridional cell winds
  (``reconstruct_winds_v841_f32``); theta is dry theta from theta_m; WRF's
  base / perturbation pressure split is MPAS's pressure_base / pressure_p;
  heights are the fixed mid-level heights of zgrid.  The surface ratios
  (uratx, vratx, tratx) and the PBL inputs (pblh, kpbl, the surface-layer
  regime, u10, v10, th2) are the column physics' own fields.
* **Domain.** WRF admits a report inside its domain less one row; hex admits a
  report whose containing cell is not in the specified zone of a limited-area
  mesh (every report on a global mesh).  WRF does not nudge its outermost rows;
  hex adds no tendency in the specified zone (bdyMaskCell >= 6), like grid
  nudging.
* **Tendencies.** WRF's coupled tendency (mu/msf, c1h mu + c2h) is the rate
  coefficient * weight * tfaci, added at the IAU seat through the same coupling
  as grid nudging (:meth:`woof.hex.nudging.HexNudging.couple`).
* **Clock.** The hex step is WRF's step: xtime is the step's start, ktau the
  step index (obs_ionf counts hex steps).  Hex steps are fixed, so an adaptive
  or local time step is refused by name.
* **Restart.** WRF rebuilds its report book from the files and forces a weight
  refresh on its first call (IFREST).  Hex checkpoints the whole mutable carrier
  (reports, rko, QC, weights, ratios, step clock and frame), so a restart
  continues the straight run word for word.

Refused by name (:func:`refusals`): obs_scl_neg_qv_innov (WRF scales the
surface innovation in place as it walks each grid row, an order the mesh has
no equivalent of), a configuration that nudges nothing, a nested domain, and
an rw_fddaobs that predates the geographic mode.  obs_nudge_pstr and
obs_coef_pstr are WRF no-ops, as on the WRF route; obs_rinsig, obs_npfi and
the obs_prt_* / obs_ipf_* prints are carried, not used (WRF v4.7.1 reads them
only to print).
"""
from __future__ import annotations

from dataclasses import fields as dataclass_fields
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np

#: Hex distances are in kilometres, so WRF's rindx = obs_rinxy * 1000 / dx is obs_rinxy.
DISTANCE_UNIT_M = 1000.0
#: The fields WRF's NUDOB nudges, as (ivar, name, the switch, the coefficient).
FIELDS = ((1, "u", "obs_nudge_wind", "obs_coef_wind"), (2, "v", "obs_nudge_wind", "obs_coef_wind"),
          (3, "temperature", "obs_nudge_temp", "obs_coef_temp"), (4, "qv", "obs_nudge_mois", "obs_coef_mois"))
#: MPAS limited-area rings 6..7 are the specified zone.
SPECIFIED_ZONE_FIRST_RING = 6
NVAR, NERR = 6, 10
#: WRF NUDOB's fixed radius-of-influence ramp (rinfmn, rinfmx) and free-atmosphere pressure (kPa).
RINFMN, RINFMX, PFREE = 1.0, 2.0, 50.0
SCHEMA = "gpuwm-hex.obs-nudging/v1"
_KERNEL_SOURCE = Path(__file__).with_name("obs_nudging.cu")
_FDDAOBS = Path(__file__).resolve().parents[1] / "fddaobs"
_MODULES: dict[Any, Any] = {}


class HexObsNudgingRefusal(ValueError):
    """An observation-nudging request hex cannot honour, with its name."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code


# ----------------------------------------------------------------------------
# configuration
# ----------------------------------------------------------------------------

def _run_defaults() -> dict[str, Any]:
    from woof.config import RunConfig

    names = ("max_obs", "fdda_start", "fdda_end")
    return {f.name: f.default for f in dataclass_fields(RunConfig)
            if f.name.startswith("obs_") or f.name in names}


def load_config(path: str | Path, *, input_file: str | Path | None = None) -> SimpleNamespace:
    """The root domain's observation-nudging controls from a TOML table or a WRF &fdda group.

    The observation file is ``input_file``, else the configuration's
    obs_inputfile, else WRF's default ``OBS_DOMAIN101``; a relative name is
    beside the configuration file.
    """
    path = Path(path).expanduser().resolve(strict=True)
    row = _run_defaults()
    text = path.read_text()
    if path.suffix.lower() == ".toml":
        import tomllib

        document = tomllib.loads(text)
        table = document.get("fdda", document.get("run", document))
        given = {key: value for key, value in table.items()
                 if key.startswith("obs_") or key in ("max_obs", "fdda_start", "fdda_end")}
        unknown = sorted(set(given) - set(row))
        if unknown:
            raise HexObsNudgingRefusal("HEX_OBS_NUDGING_KEYS",
                                       f"{path}: unknown observation-nudging key(s) {unknown}")
        for key, value in given.items():
            if isinstance(value, list):
                if len(value) > 1 and key == "obs_nudge_opt" and any(value[1:]):
                    raise HexObsNudgingRefusal("HEX_OBS_NUDGING_DOMAIN", _DOMAIN)
                value = value[0]
            row[key] = value
    else:
        from woof.fortran_namelist import parse_namelist_text
        from woof.namelist_import import _Section
        from woof.fddaobs.namelist import consume

        sections = parse_namelist_text(text, source=str(path))
        if "fdda" not in sections:
            raise HexObsNudgingRefusal("HEX_OBS_NUDGING_NO_FDDA", f"{path} has no &fdda group")
        count = max([len(v) if isinstance(v, list) else 1 for v in sections["fdda"].values()] + [1])
        section = _Section("fdda", sections["fdda"], str(path))
        try:
            columns = consume(section, count)
        except ValueError as error:
            raise HexObsNudgingRefusal("HEX_OBS_NUDGING_CONFIG", str(error)) from None
        if any(columns["obs_nudge_opt"][1:]):
            raise HexObsNudgingRefusal("HEX_OBS_NUDGING_DOMAIN", _DOMAIN)
        row.update({key: values[0] for key, values in columns.items()})
        name = sections["fdda"].get("obs_inputfile")
        if name is not None:
            row["obs_inputfile"] = name[0] if isinstance(name, list) else name
    chosen = input_file or row.get("obs_inputfile") or "OBS_DOMAIN101"
    file = Path(chosen).expanduser()
    row["obs_inputfile"] = str(file if file.is_absolute() else (path.parent / file).resolve())
    row["config_path"] = str(path)
    return SimpleNamespace(**row)


_DOMAIN = ("observation nudging is requested for a nested domain; a hex run is one variable-resolution "
           "mesh (domain 1), and a WRF-path nest under hex nudges on the WRF path")


def refusals(cfg) -> list[HexObsNudgingRefusal]:
    """Every named reason these controls cannot run on a hex mesh."""
    found: list[HexObsNudgingRefusal] = []
    if cfg.obs_nudge_opt != 1:
        found.append(HexObsNudgingRefusal("HEX_OBS_NUDGING_OFF", "obs_nudge_opt is not 1; drop --obs-nudging"))
        return found
    if cfg.obs_scl_neg_qv_innov:
        found.append(HexObsNudgingRefusal(
            "HEX_OBS_NUDGING_NEG_QV",
            "obs_scl_neg_qv_innov scales WRF's surface moisture innovation in place as NUDOB walks each grid "
            "row (the scaled value carries to the next i), an order a mesh has no equivalent of; set it to 0"))
    if not (cfg.obs_nudge_wind or cfg.obs_nudge_temp or cfg.obs_nudge_mois):
        found.append(HexObsNudgingRefusal(
            "HEX_OBS_NUDGING_NO_FIELD", "obs_nudge_wind, obs_nudge_temp and obs_nudge_mois are all 0"))
    if cfg.obs_levelidn != 0 or cfg.obs_parent_grid_ratio != 1 or cfg.obs_domain_id != 1:
        found.append(HexObsNudgingRefusal("HEX_OBS_NUDGING_DOMAIN", _DOMAIN))
    try:
        from woof.fddaobs.config import validate

        # map_proj is the WRF grid's; the mesh has none (geographic mode), so the
        # projection check is answered here and every other WRF check runs as is.
        validate(SimpleNamespace(**{**vars(cfg), "map_proj": 1}))
    except ValueError as error:
        found.append(HexObsNudgingRefusal("HEX_OBS_NUDGING_CONFIG", str(error)))
    return found


# ----------------------------------------------------------------------------
# observations: the WRF route's Rust reader in geographic mode
# ----------------------------------------------------------------------------

def read_observations(cfg, *, original_start: datetime, dt: float) -> dict[str, Any]:
    """rw_fddaobs's admitted report rows, with longitude/latitude in place of i/j."""
    from woof import bridges

    binary = bridges.find_artifact("GPUWM_RW_FDDAOBS", bridges.executable_name("rw_fddaobs"),
                                   crate_relative=bridges.RUSTWX_CRATE_RELATIVE)
    if binary is None:
        raise HexObsNudgingRefusal(
            "HEX_OBS_NUDGING_READER",
            "rw_fddaobs is missing; install its bridge bundle or build tools/rustwx with "
            "cargo build --release --locked -p rw-obs --bin rw_fddaobs")
    inputfile = Path(cfg.obs_inputfile)
    if not inputfile.is_file():
        raise HexObsNudgingRefusal("HEX_OBS_NUDGING_INPUT", f"observation nudging requires {inputfile}")
    start = original_start.replace(tzinfo=timezone.utc) if original_start.tzinfo is None else original_start
    projection = dict(map_proj=0, latitude_origin=0.0, longitude_origin=0.0, truelat1=0.0, truelat2=0.0,
                      stand_lon=0.0, dx=0.0, dy=0.0, e_we=1, e_sn=1)
    with tempfile.TemporaryDirectory(prefix="woof-hex-obs-") as directory:
        directory = Path(directory)
        request = dict(input=str(inputfile), output=str(directory / "observations.pack"), projection=projection,
                       original_start_unix_seconds=int(start.timestamp()),
                       rovg=float(np.float32(287.) / np.float32(9.81)),
                       window=dict(capacity=cfg.max_obs, begin=cfg.fdda_start, end=cfg.fdda_end,
                                   hours=cfg.obs_twindo, dt_minutes=dt / 60.0, interval=cfg.obs_ionf,
                                   adaptive=False))
        (directory / "request.json").write_text(json.dumps(request))
        result = subprocess.run([str(binary), str(directory / "request.json")], capture_output=True, text=True)
        if result.returncode != 0:
            message = (result.stderr or result.stdout).strip()
            if "not an ARW projection" in message:
                raise HexObsNudgingRefusal(
                    "HEX_OBS_NUDGING_READER",
                    f"{binary} predates the geographic mode hex reads (map_proj 0); rebuild rw_fddaobs")
            if "STOP 122" in message:
                raise HexObsNudgingRefusal("HEX_OBS_NUDGING_MAX_OBS", message)
            raise HexObsNudgingRefusal("HEX_OBS_NUDGING_INPUT", f"rw_fddaobs: {message}")
        packed = (directory / "observations.pack").read_bytes()
    if packed[:8] != b"GPWMOBS1":
        raise ValueError("observation pack ABI differs from gpuwm-fddaobs-pack-v1")
    length = int.from_bytes(packed[8:12], "little")
    metadata = json.loads(packed[12:12 + length])
    count = int(metadata["rows"])
    host = np.frombuffer(packed, dtype="<f4", offset=12 + length)
    if host.size != count * (8 + NVAR):
        raise ValueError("observation pack carrier length is inconsistent")
    names = ("longitude", "latitude", "rko", "time", "nlev", "lev", "platform", "elevation")
    rows = {name: np.array(host[n * count:(n + 1) * count], dtype=np.float32) for n, name in enumerate(names)}
    rows["varobs"] = np.array(host[8 * count:], dtype=np.float32).reshape(count, NVAR)
    rows["metadata"] = {k: v for k, v in metadata.items() if k != "workers"}
    return rows


# ----------------------------------------------------------------------------
# geometry: the cell KD-tree
# ----------------------------------------------------------------------------

def _unit(lat, lon):
    lat, lon = np.asarray(lat, dtype=np.float64), np.asarray(lon, dtype=np.float64)
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1)


class MeshLocator:
    """Report placement on the mesh: containing cell, Delaunay triangle and reach lists."""

    def __init__(self, mesh):
        from scipy.spatial import cKDTree

        a = mesh.arrays
        self.n = int(mesh.n_cells)
        self.radius = float(mesh.attrs.get("sphere_radius", 6371229.0))
        self.xyz = _unit(np.asarray(a["latCell"])[:self.n], np.asarray(a["lonCell"])[:self.n])
        self.tree = cKDTree(self.xyz)
        self.bdy = (np.asarray(a["bdyMaskCell"])[:self.n] if "bdyMaskCell" in a
                    else np.zeros(self.n, dtype=np.int64))
        self.vertices_on_cell = np.asarray(a["verticesOnCell"])[:self.n]
        self.n_edges_on_cell = np.asarray(a["nEdgesOnCell"])[:self.n]
        self.cells_on_vertex = np.asarray(a["cellsOnVertex"])

    def locate(self, lat_deg, lon_deg):
        """(nearest cell, triangle corners [m, 3], barycentric weights [m, 3] float32, admitted)."""
        p = _unit(np.radians(lat_deg), np.radians(lon_deg))
        _, near = self.tree.query(p)
        near = np.asarray(near, dtype=np.int64).reshape(-1)
        corners = np.repeat(near[:, None], 3, axis=1)
        weights = np.zeros((near.size, 3), dtype=np.float64)
        weights[:, 0] = 1.0
        for row, (point, cell) in enumerate(zip(p.reshape(-1, 3), near)):
            best = None
            for vertex in self.vertices_on_cell[cell, :int(self.n_edges_on_cell[cell])]:
                if vertex < 0:
                    continue
                triangle = self.cells_on_vertex[vertex]
                if np.any(triangle < 0) or np.any(triangle >= self.n):
                    continue
                matrix = self.xyz[triangle].T
                try:
                    solved = np.linalg.solve(matrix, point)
                except np.linalg.LinAlgError:
                    continue
                if not solved.sum() > 0:
                    continue
                solved = solved / solved.sum()
                if best is None or solved.min() > best[1].min():
                    best = (triangle, solved)
            if best is not None and best[1].min() >= -1e-9:
                corners[row] = best[0]
                weights[row] = np.clip(best[1], 0.0, None) / np.clip(best[1], 0.0, None).sum()
        admitted = self.bdy[near] < SPECIFIED_ZONE_FIRST_RING
        return near, corners, weights.astype(np.float32), admitted

    def reach(self, lat_deg, lon_deg, radius_m: float):
        """CSR lists per cell: the report indices within radius_m and their squared distances (km^2)."""
        p = _unit(np.radians(lat_deg), np.radians(lon_deg)).reshape(-1, 3)
        angle = min(math.pi, radius_m / self.radius)
        chord = 2.0 * math.sin(angle / 2.0) * (1.0 + 1e-6)
        lists = self.tree.query_ball_point(p, chord) if len(p) else []
        report, cell = [], []
        for index, cells in enumerate(lists):
            report.extend([index] * len(cells))
            cell.extend(cells)
        report = np.asarray(report, dtype=np.int64)
        cell = np.asarray(cell, dtype=np.int64)
        if report.size:
            dot = np.clip(np.einsum("ij,ij->i", p[report], self.xyz[cell]), -1.0, 1.0)
            cross = np.linalg.norm(np.cross(p[report], self.xyz[cell]), axis=1)
            distance = self.radius * np.arctan2(cross, dot)
        else:
            distance = np.zeros(0)
        order = np.lexsort((report, cell))
        report, cell, distance = report[order], cell[order], distance[order]
        start = np.zeros(self.n + 1, dtype=np.int64)
        np.add.at(start, cell + 1, 1)
        start = np.cumsum(start)
        rsq = (distance / DISTANCE_UNIT_M) ** 2
        return start, report, rsq.astype(np.float32)


def great_circle_m(radius, lat1, lon1, lat2, lon2):
    a, b = _unit(lat1, lon1), _unit(lat2, lon2)
    return radius * np.arctan2(np.linalg.norm(np.cross(a, b), axis=-1), np.einsum("...i,...i", a, b))


# ----------------------------------------------------------------------------
# kernels
# ----------------------------------------------------------------------------

def _between(text: str, first: str, stop: str | None, label: str) -> str:
    begin = text.find(first)
    end = len(text) if stop is None else text.find(stop, begin)
    if begin < 0 or end < 0:
        raise RuntimeError(f"woof/fddaobs/{label} changed layout; update woof.hex.obs_nudging's cut")
    return text[begin:end]


def kernel_source() -> str:
    """The shared WRF pieces of woof/fddaobs, cut unchanged, then the hex kernels.

    Cut, not included whole: fddaobs_nudob_rows assigns through a conditional
    thread/device lvalue that MSL does not accept, and fddaobs_stamp reads the
    PTX %globaltimer; hex launches neither, so every backend compiles one text.
    """
    kernels = (_FDDAOBS / "kernels.cu").read_text()
    forecast = (_FDDAOBS / "forecast.cu").read_text()
    head = _between(kernels, "struct Frame", 'extern "C" __global__ void fddaobs_ratios', "kernels.cu")
    weight = _between(kernels, "__device__ float timewt", "/* Parallel columns", "kernels.cu")
    vif = _between(kernels, 'extern "C" __global__ void fddaobs_vif', None, "kernels.cu")
    clock = _between(forecast, 'extern "C" __global__ void fddaobs_publish',
                     'extern "C" __global__ void fddaobs_pack', "forecast.cu")
    return "\n".join(("// cut from woof/fddaobs/kernels.cu (WRF v4.7.1 module_fddaobs_rtfdda.F)",
                      head, weight, vif, "// cut from woof/fddaobs/forecast.cu", clock,
                      _KERNEL_SOURCE.read_text()))


OPTIONS = ("--std=c++17",)


def kernels():
    import cupy as cp
    from woof.certify.kernel_manifest import record_module

    key = cp.cuda.Device().id
    if key not in _MODULES:
        source = kernel_source()
        module = cp.RawModule(code=source, options=OPTIONS)
        record_module("woof.hex.obs_nudging", source=source, options=OPTIONS, module=module)
        _MODULES[key] = {name: module.get_function(name) for name in (
            "fddaobs_publish", "fddaobs_clock", "fddaobs_vif", "hex_fddaobs_ratios",
            "hex_fddaobs_errob", "hex_fddaobs_nudob")}
    return _MODULES[key]


def _record(values, types):
    return np.array(tuple(values), dtype=[("v" + str(i), t) for i, t in enumerate(types)])[()]


def grid_record(nc: int, nz: int):
    """WRF's Grid for a one-row grid of nc mass columns: at(i,k,1) is hex [k-1][i-1]."""
    return _record([1, nc + 1, 1, 2, 1, nz + 1, 1, nc, 1, 1, 1, nz, 1, nc, 1, 1, 1, nz], ["i4"] * 18)


#: Float32 (level, cell) arrays HexObsNudging holds besides the scalar ambient copy: z, up to four tendencies,
#: savwt (NVAR words), tb/theta work, five wind work arrays, the ambient theta and density, and the unit zz.
LEVEL_ARRAYS = 1 + 4 + NVAR + 2 + 5 + 2 + 1
#: Per-cell planes: ht, dz1, mask, three surface ratios, three VIF height pairs and tops, and seam fields.
PLANE_ARRAYS = 32
#: Per-report words: varobs, errf, time/nlev/lev/rko, three corners and weights, the nearest cell.
REPORT_WORDS = NVAR + NERR + 4 + 6 + 1


def device_bytes(cells: int, levels: int, *, max_obs: int, scalars: int = 8) -> int:
    """Device storage observation nudging adds before its reach lists (priced after placement).

    ``scalars`` bounds the ambient scalar copy (through qv's index).  The reach lists depend on where the
    reports fall; :class:`HexObsNudging` checks them against the device's free memory before allocating.
    """
    level = 4 * int(cells) * int(levels) * (LEVEL_ARRAYS + int(scalars))
    return level + 4 * int(cells) * PLANE_ARRAYS + 4 * int(max_obs) * REPORT_WORDS


def controls_identity(cfg) -> dict[str, Any]:
    """Every numerical control a checkpoint carrier depends on (paths and print switches excluded)."""
    return {key: value for key, value in sorted(vars(cfg).items())
            if key not in ("obs_inputfile", "config_path") and not key.startswith(("obs_prt_", "obs_ipf_"))}


def controls_record(cfg, ivar: int, dt: float):
    """WRF's Controls record, as woof/fddaobs/forecast.py fills it, with the hex distance unit for dx."""
    suffix = "uv" if ivar < 3 else "t" if ivar == 3 else "q"
    no_pbl = getattr(cfg, "obs_no_pbl_nudge_" + suffix)
    coef = cfg.obs_coef_wind if ivar < 3 else cfg.obs_coef_temp if ivar == 3 else cfg.obs_coef_mois
    return _record([1, 0, 0, 0, cfg.obs_ionf, 1 - no_pbl, cfg.obs_sfc_scheme_horiz, cfg.obs_sfc_scheme_vert,
                    cfg.obs_scl_neg_qv_innov, 0., dt / 60., cfg.obs_rinxy, DISTANCE_UNIT_M, cfg.obs_twindo,
                    cfg.obs_sfcfact, cfg.obs_sfcfacr, RINFMN, RINFMX, PFREE, 1. / cfg.obs_dpsmx, 1., coef,
                    cfg.obs_max_sndng_gap], ["i4"] * 9 + ["f4"] * 14)


_VIF_CONTROLS = {suffix: tuple(f"obs_nudgez{kind}r{regime}_{suffix}" for regime in (1, 2, 4)
                               for kind in ("full", "ramp")) for suffix in ("uv", "t", "q")}


# ----------------------------------------------------------------------------
# the forcing
# ----------------------------------------------------------------------------

def _admit_reach(cp, required: int) -> None:
    """Refuse by name, before allocating, a reach list the device cannot hold (dense reports, wide radii)."""
    try:
        free = int(cp.cuda.runtime.memGetInfo()[0])
    except Exception:  # The Metal facade has no free-memory query; unified memory is admitted at the door.
        return
    if required > free:
        raise HexObsNudgingRefusal(
            "HEX_OBS_NUDGING_MEMORY",
            f"the report reach lists need {required} bytes and the device has {free} free; "
            "lower obs_rinxy or max_obs, or thin the reports")


class HexObsNudging:
    """WRF observation nudging on hex cell columns, added at the IAU seat.

    ``add`` has the held-forcing signature of woof.hex.iau_native / woof.hex.nudging;
    ``step_tendencies`` runs one WRF fddaobs call and returns the per-cell
    tendencies (du, dv in m s-2, dtheta in K s-1, dqv in kg kg-1 s-1).
    """

    def __init__(self, cfg, *, observations: Mapping[str, Any], mesh, prep_geometry, physics_geometry,
                 zgrid, scalar_names, dt: float, kernel_cache, surface, restart_seconds: float = 0.0):
        import cupy as cp

        problems = refusals(cfg)
        if problems:
            raise problems[0]
        self.cfg, self.cp, self.dt = cfg, cp, float(dt)
        self.cache = kernel_cache
        self.prep_geometry, self.physics_geometry = prep_geometry, physics_geometry
        self.names = tuple(scalar_names)
        if "qv" not in self.names:
            raise HexObsNudgingRefusal("HEX_OBS_NUDGING_SCALARS", "the run carries no qv")
        self.index_qv = self.names.index("qv")
        self.nc = int(prep_geometry.n_cells)
        self.ne = int(getattr(prep_geometry, "n_edges", 0))
        self.surface = surface
        z = np.asarray(zgrid, dtype=np.float32)
        self.nz = int(z.shape[0]) - 1
        padded = np.zeros((self.nz + 1, self.nc), dtype=np.float32)
        padded[:, :z.shape[1]] = z
        if z.shape[1] < self.nc:
            padded[:, z.shape[1]:] = z[:, -1:]
        middle = (np.float32(.5) * (padded[:-1] + padded[1:])).astype(np.float32)
        self.z = cp.asarray(np.ascontiguousarray(middle))
        self.ht = cp.asarray(np.ascontiguousarray(padded[0]))
        self.dz1 = cp.asarray(np.ascontiguousarray(padded[1] - padded[0]))
        mask = np.zeros(self.nc, dtype=np.float32)
        n = int(mesh.n_cells)
        keep = np.ones(n, dtype=bool)
        if "bdyMaskCell" in mesh.arrays:
            keep = np.asarray(mesh.arrays["bdyMaskCell"])[:n] < SPECIFIED_ZONE_FIRST_RING
        mask[:n] = keep
        self.mask = cp.asarray(mask)
        self._place(observations, mesh)
        self.active = tuple((ivar, name) for ivar, name, switch, _ in FIELDS if getattr(cfg, switch))
        self.k = kernels()
        self.grid = grid_record(self.nc, self.nz)
        self.controls = {ivar: controls_record(cfg, ivar, self.dt) for ivar, _ in self.active}
        shape = (self.nz, self.nc)
        self.tend = {name: cp.zeros(shape, dtype=cp.float32) for _, name in self.active}
        plane = (self.nc,)
        self.uratx, self.vratx, self.tratx = (cp.zeros(plane, dtype=cp.float32) for _ in range(3))
        self.savwt = cp.zeros(shape[0] * shape[1] * NVAR, dtype=cp.float32)
        self.errf = cp.zeros(max(1, self.count) * NERR, dtype=cp.float32)
        self.clock = cp.array([round(float(restart_seconds) / self.dt)], dtype=cp.int64)
        self.frame = cp.zeros(32, dtype=cp.uint8)
        self.frame.view(cp.int32)[3] = -1  # the first call refreshes the report window (WRF IFREST)
        self.published = cp.zeros(2, dtype=cp.float32)
        self.error = cp.zeros(1, dtype=cp.uint32)
        self.vif = {s: cp.asarray([getattr(cfg, k) for k in keys], dtype=cp.float32)
                    for s, keys in _VIF_CONTROLS.items()}
        self.heights = {s: (cp.zeros(plane, dtype=cp.float32), cp.zeros(plane, dtype=cp.float32)) for s in self.vif}
        self.surface_top = {s: cp.zeros(plane, dtype=cp.int32) for s in self.vif}
        # forecast.py raises a positive obs_nudgezfullmin to 1.1 x the height of WRF's first half level
        # (from its analytic base state); the mesh's first mid level stands in for it (the highest over
        # the real cells, so the lowest level is inside every column's full-weight depth).
        n_real = int(mesh.n_cells)
        first_half = np.float32(np.max(middle[0, :n_real] - padded[0, :n_real]))
        self.fullmin = (max(np.float32(cfg.obs_nudgezfullmin), np.float32(1.1) * first_half)
                        if cfg.obs_nudgezfullmin > 0 else np.float32(0.))
        self.work = {name: cp.zeros(shape, dtype=cp.float32) for name in ("tb", "theta")}
        self._wind = {name: cp.zeros(shape, dtype=cp.float32) for name in ("x", "y", "z", "u", "v")}
        self._invalid = cp.zeros(1, dtype=cp.int32)
        self._theta_amb = cp.zeros(shape, dtype=cp.float32)
        self._scalars_amb = cp.zeros((self.index_qv + 1,) + shape, dtype=cp.float32)
        self._rho_amb = cp.zeros(shape, dtype=cp.float32)
        self._unit_zz = None
        self.calls = self.forced_steps = 0
        self.first_time = self.last_time = None

    # -- report book ---------------------------------------------------------
    def _place(self, observations, mesh):
        cp = self.cp
        rows = observations
        count = int(rows["rko"].size)
        locator = MeshLocator(mesh)
        lat, lon = rows["latitude"], rows["longitude"]
        near, corners, weights, admitted = locator.locate(lat, lon) if count else (
            np.zeros(0, np.int64), np.zeros((0, 3), np.int64), np.zeros((0, 3), np.float32), np.zeros(0, bool))
        # A report is admitted whole (every level shares the first level's place).
        first = np.zeros(count, dtype=np.int64)
        index = 0
        while index < count:
            levels = int(rows["nlev"][index] - rows["lev"][index]) + 1
            first[index:index + levels] = index
            index += levels
        keep = admitted[first] if count else admitted
        self.rejected_reports = int(np.sum((rows["lev"] == 1) & ~keep)) if count else 0
        select = np.flatnonzero(keep)
        self.count = int(select.size)
        host = {name: np.ascontiguousarray(rows[name][select]) for name in
                ("time", "nlev", "lev", "rko")}
        self.timeob, self.nlev, self.lev = (cp.asarray(host[n]) for n in ("time", "nlev", "lev"))
        self.rko = cp.asarray(host["rko"])
        self.varobs = cp.asarray(np.ascontiguousarray(rows["varobs"][select]).reshape(-1))
        self.corner = cp.asarray(np.ascontiguousarray(corners[select] + 1, dtype=np.int32).reshape(-1))
        self.bary = cp.asarray(np.ascontiguousarray(weights[select], dtype=np.float32).reshape(-1))
        self.near = cp.asarray(np.ascontiguousarray(near[select] + 1, dtype=np.int32))
        starts = np.flatnonzero(rows["lev"][select] == 1) if self.count else np.zeros(0, np.int64)
        reach_m = self.cfg.obs_rinxy * 1000.0 * max(self.cfg.obs_sfcfacr, RINFMX, RINFMN)
        start, report, rsq = locator.reach(lat[select][starts], lon[select][starts], reach_m)
        _admit_reach(cp, 8 * int(report.size) + 4 * (self.nc + 1))
        padded_start = np.full(self.nc + 1, start[-1], dtype=np.int32)
        padded_start[:start.size] = start
        self.reach_start = cp.asarray(padded_start)
        self.reach = cp.asarray(np.ascontiguousarray(starts[report], dtype=np.int32) if report.size
                                else np.zeros(1, np.int32))
        self.reach_rsq = cp.asarray(rsq if rsq.size else np.zeros(1, np.float32))
        self.pairs = int(report.size)
        identity = dict(rows["metadata"])
        identity.update(admitted_rows=self.count, rejected_reports=self.rejected_reports,
                        cells=int(mesh.n_cells), sphere_radius=locator.radius,
                        mesh_cells_sha256=hashlib.sha256(np.ascontiguousarray(locator.xyz).tobytes()).hexdigest(),
                        source_sha256=hashlib.sha256(kernel_source().encode()).hexdigest(),
                        # Every numerical control: a restart under other coefficients, radii or windows is refused.
                        controls=controls_identity(self.cfg))
        self.metadata = rows["metadata"]
        self.identity = hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest()

    # -- restart ---------------------------------------------------------------
    _CARRIER = ("rko", "varobs", "errf", "savwt", "clock", "frame", "uratx", "vratx", "tratx")

    def snapshot(self) -> dict[str, Any]:
        """The mutable carrier at a step boundary (a checkpoint entry)."""
        cp = self.cp
        return {"schema": SCHEMA, "identity": self.identity,
                **{name: cp.asnumpy(getattr(self, name)).copy() for name in self._CARRIER}}

    def restore(self, carried: Mapping[str, Any]) -> None:
        if carried.get("schema") != SCHEMA or carried.get("identity") != self.identity:
            raise HexObsNudgingRefusal(
                "HEX_OBS_NUDGING_RESTART",
                "the checkpoint's observation-nudging carrier was written for other reports, controls or mesh")
        for name in self._CARRIER:
            target = getattr(self, name)
            value = np.asarray(carried[name])
            if value.shape != target.shape or value.dtype != target.dtype:
                raise HexObsNudgingRefusal("HEX_OBS_NUDGING_RESTART", f"checkpoint carrier {name} has the wrong shape")
            target.set(value)

    # -- model fields --------------------------------------------------------
    def _reconstruct(self, normal_velocity):
        from .nudging import HexNudging

        return HexNudging._reconstruct(self, normal_velocity)

    def _theta(self, theta_m, qv):
        cp = self.cp
        key = ("obs-theta", cp.cuda.Device().id)
        if key not in _MODULES:
            # forecast.cu pack_fields' theta_m conversion (woof.hex.nudging), and fddaobs_pack's theta - 300.
            _MODULES[key] = cp.ElementwiseKernel(
                "float32 theta_m, float32 qv", "float32 theta, float32 tb",
                "theta = theta_m/(1.f+(461.6f/287.f)*qv); tb = theta-300.f;", "hex_obs_nudging_theta")
        _MODULES[key](theta_m, qv, self.work["theta"], self.work["tb"])
        return self.work["theta"], self.work["tb"]

    # -- the step --------------------------------------------------------------
    def step_tendencies(self, *, time_seconds: float, theta_m, qv, normal_velocity, pressure_base,
                        pressure_perturbation, surface: Mapping[str, Any]):
        """One WRF fddaobs call at this step's start; the per-cell tendencies (always written)."""
        cp = self.cp
        cfg = self.cfg
        k = self.k
        nc, nz = self.nc, self.nz
        g = self.grid
        theta, tb = self._theta(theta_m, qv)
        wind = self._reconstruct(normal_velocity)
        pb = cp.ascontiguousarray(pressure_base[:, :nc], dtype=cp.float32)
        pp = cp.ascontiguousarray(pressure_perturbation[:, :nc], dtype=cp.float32)
        pressure = pb[0] + pp[0]
        k["fddaobs_publish"]((1,), (1,), (self.published, np.float32(time_seconds), np.float32(self.dt)))
        k["fddaobs_clock"]((1,), (1,), (self.clock, self.frame, self.timeob, self.nlev, self.lev,
                                        np.int32(self.count), np.int32(cfg.max_obs), np.float32(self.dt),
                                        np.float32(cfg.fdda_start), np.float32(cfg.fdda_end),
                                        np.float32(cfg.obs_twindo), np.int32(cfg.obs_idynin),
                                        np.float32(cfg.obs_dtramp), np.int32(cfg.obs_ionf), self.published))
        block = (128,)
        k["hex_fddaobs_ratios"](((nc + 127) // 128,), block, (
            wind["u"], wind["v"], theta, pressure, surface["u10"], surface["v10"], surface["th2"],
            self.uratx, self.vratx, self.tratx, np.int32(nc), np.int32(surface["sfclay"]),
            np.int32(surface["stepbl"]), self.frame))
        if self.count:
            k["hex_fddaobs_errob"](((self.count + 127) // 128,), block, (
                wind["u"], wind["v"], tb, qv, pb, pp, self.z, self.uratx, self.vratx, self.tratx, surface["kpbl"],
                self.corner, self.bary, self.near, self.rko, self.varobs, self.errf,
                np.int32(self.count), np.int32(NVAR), np.int32(NERR), np.int32(cfg.obs_nudge_wind),
                np.int32(cfg.obs_nudge_temp), np.int32(cfg.obs_nudge_mois), np.float32(300.),
                np.float32(287. / 1004.5), g, self.frame))
        for suffix in self.vif:
            k["fddaobs_vif"](((nc + 127) // 128, 1), (128, 1), (
                self.vif[suffix], surface["regime"], surface["pblh"], *self.heights[suffix],
                np.float32(cfg.obs_nudgezmax), self.fullmin, np.float32(cfg.obs_nudgezrampmin), g, self.error,
                self.z, self.ht, self.surface_top[suffix]))
        launch = (((nc + 31) // 32, (nz + 3) // 4), (32, 4))
        for ivar, name in self.active:
            suffix = "uv" if ivar < 3 else "t" if ivar == 3 else "q"
            k["hex_fddaobs_nudob"](launch[0], launch[1], (
                self.tend[name], self.savwt, self.mask, self.reach_start, self.reach, self.reach_rsq,
                self.rko, self.timeob, self.varobs, self.errf, self.nlev, self.lev, pb, pp, surface["kpbl"],
                *self.heights[suffix], self.ht, self.z, np.int32(NVAR), np.int32(NERR), np.int32(ivar),
                self.controls[ivar], g, self.frame, self.surface_top[suffix]))
        self.calls += 1
        self.first_time = float(time_seconds) if self.first_time is None else self.first_time
        self.last_time = float(time_seconds)
        return dict(self.tend)

    def add(self, index: int, held: Any, start: float, dt: float, *, driver: Any = None,
            rho_edge: Any, normal_velocity: Any, state: Any, **_unused: Any) -> None:
        """This step's observation-nudging tendencies onto the held physics tendencies (in place)."""
        cp = self.cp
        if float(dt) != self.dt:
            raise HexObsNudgingRefusal("HEX_OBS_NUDGING_ADAPTIVE", "observation nudging was built for a fixed time step")
        theta_m = cp.ascontiguousarray(state.rho_theta / state.rho)
        atmosphere = driver.atmosphere
        tend = self.step_tendencies(
            time_seconds=start, theta_m=theta_m, qv=state.scalars[self.index_qv], normal_velocity=normal_velocity,
            pressure_base=atmosphere.reference.pressure_base,
            pressure_perturbation=atmosphere.saved.pressure_perturbation, surface=self.surface())
        from .nudging import HexNudging

        cache = self.cache if driver is None else driver.cache
        HexNudging.couple(self, tend, held=held, rho_zz=state.rho, theta_m=theta_m, scalars=state.scalars,
                          rho_edge=rho_edge, cache=cache)
        self.forced_steps += 1

    def receipt(self) -> dict[str, Any]:
        cfg = self.cfg
        error = int(self.error.get()[0])
        if error:
            raise RuntimeError("WRF fddaobs_driver STOP 6: surface-layer regime must be 1, 2, 3 or 4")
        return {
            "schema": SCHEMA,
            "fields": [name for _, name in self.active],
            "observations": {"file": cfg.obs_inputfile, "rows": self.count, "rejected_reports": self.rejected_reports,
                             "cell_pairs": self.pairs, "pack": self.metadata},
            "controls": {key: getattr(cfg, key) for key in sorted(vars(cfg))
                         if key.startswith("obs_") or key in ("max_obs", "fdda_start", "fdda_end")},
            "distance": f"great-circle on the mesh sphere, in {DISTANCE_UNIT_M:g} m units (rindx = obs_rinxy)",
            "kernels": "woof/fddaobs kernels.cu (timewt, fddaobs_vif) and forecast.cu (fddaobs_clock), "
                       "woof/hex/obs_nudging.cu (ERROB, NUDOB on cells); coupling woof.hex.nudging.HexNudging.couple",
            "seat": "held physics tendencies after the coupling, before the dynamics (the IAU seat)",
            "identity": self.identity, "calls": self.calls, "forced_steps": self.forced_steps,
            "first_time_seconds": self.first_time, "last_time_seconds": self.last_time,
        }


# ----------------------------------------------------------------------------
# the WOOF column route (R1)
# ----------------------------------------------------------------------------

def _seam_surface(backend, nc: int, *, z_mid, ht, dz1):
    """The column physics' PBL and surface-layer fields, padded to the mesh's columns, read per step."""
    import cupy as cp

    seam = getattr(backend, "_seam", None)
    driver = getattr(seam, "_driver", None)
    fields = getattr(driver, "fields", None)
    if fields is None or "pblh" not in fields:
        raise HexObsNudgingRefusal("HEX_OBS_NUDGING_PBL",
                                   "observation nudging needs the column physics' PBL height; this route publishes none")
    cfg = getattr(driver, "cfg", None)
    sfclay = int(getattr(cfg, "sf_sfclay_physics", 1) or 0)
    native_regime = sfclay in (1, 5, 7, 91) and "regime" in fields
    missing = [name for name in ("u10", "v10", "th2") if sfclay and name not in fields]
    if missing:
        raise HexObsNudgingRefusal("HEX_OBS_NUDGING_SURFACE",
                                   f"the surface layer publishes no {missing}; WRF's surface ratios need them")
    stepbl = int(getattr(driver, "stepbl", 1) or 1)
    zero = cp.zeros(nc, dtype=cp.float32)

    def column(value, dtype):
        flat = cp.asarray(value).reshape(-1).astype(dtype, copy=False)
        if flat.size > nc:
            raise ValueError("a surface field has more columns than the mesh")
        if flat.size < nc:
            flat = cp.concatenate([flat, cp.zeros(nc - flat.size, dtype=dtype)])
        return cp.ascontiguousarray(flat)

    lowest_layer = cp.ascontiguousarray(dz1)

    def read():
        pblh = column(fields["pblh"], cp.float32)
        out = {"pblh": pblh, "sfclay": int(bool(sfclay)), "stepbl": stepbl}
        if "kpbl" in fields:
            out["kpbl"] = column(fields["kpbl"], cp.int32)
        else:
            out["kpbl"] = cp.ascontiguousarray(1 + cp.sum(z_mid - ht[None, :] < pblh[None, :], axis=0).astype(cp.int32))
        if native_regime:
            out["regime"] = column(fields["regime"], cp.float32)
        else:
            # forecast.cu fddaobs_pack's fallback: unstable (4) when the PBL tops the lowest layer, else 1.
            out["regime"] = cp.where(pblh > lowest_layer, cp.float32(4.), cp.float32(1.)).astype(cp.float32)
        for name in ("u10", "v10", "th2"):
            out[name] = column(fields[name], cp.float32) if name in fields else zero
        return out
    return read


def attach_v841(request: Mapping[str, Any], *, stack: Mapping[str, Any], host: Mapping[str, Any],
                cache: Any, dt: float, restart_seconds: float = 0.0,
                carried: Mapping[str, Any] | None = None) -> HexObsNudging:
    """The fixed-step WOOF column route (R1): build the forcing for this stack."""
    cfg = load_config(request["config"], input_file=request.get("input"))
    problems = refusals(cfg)
    if problems:
        raise problems[0]
    driver = stack["driver"]
    if bool(getattr(driver.config, "config_local_timestep", False)):
        raise HexObsNudgingRefusal("HEX_OBS_NUDGING_LTS", "observation nudging runs on fixed-step hex runs; "
                                   "drop the local time step")
    prepared = host["prepared"]
    start = datetime.strptime(str(request["start_time"]).replace("T", "_"), "%Y-%m-%d_%H:%M:%S")
    rows = read_observations(cfg, original_start=start, dt=dt)
    geometry = stack["prep_geometry"]
    forcing = HexObsNudging(
        cfg, observations=rows, mesh=prepared.mesh, prep_geometry=geometry,
        physics_geometry=stack["physics_geometry"], zgrid=prepared.vertical.zgrid,
        scalar_names=host["scalar_names"], dt=dt, kernel_cache=cache,
        surface=lambda: None, restart_seconds=restart_seconds)
    forcing.surface = _seam_surface(stack["backend"], forcing.nc, z_mid=forcing.z, ht=forcing.ht, dz1=forcing.dz1)
    if carried is not None:
        forcing.restore(carried)
    return forcing
