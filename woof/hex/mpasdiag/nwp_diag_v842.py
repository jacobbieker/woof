"""WRF's nwp_diagnostics history maxima on the MPAS-A v8.4.2 suite route (config_nwp_diagnostics).

A WOOF-only output; MPAS-A v8.4.2 has none.  The WOOF namelist group ``&diagnostics`` option
``config_nwp_diagnostics`` (logical, default false; :func:`resolve`) adds the maxima the WRF route writes under
``nwp_diagnostics = 1`` and ``output_diagnostics = 1`` as ``diag`` fields [nCells] (package ``nwp_diagnostics``,
members of the ``diagnostics`` stream); with it off nothing here is allocated, constructed or compiled.

Fields (MPAS style lower case of WRF's name)::

    wspd10max  WSPD10MAX   m s-1   max 10 m wind speed
    u10max     U10MAX      m s-1   10 m u at the instant WSPD10MAX last grew
    v10max     V10MAX      m s-1   10 m v at that instant
    gust       GUST        m s-1   max UPP-method surface gust
    w_up_max   W_UP_MAX    m s-1   max w below 400 hPa
    w_dn_max   W_DN_MAX    m s-1   min w below 400 hPa
    grpl_max   GRPL_MAX    kg m-2  max column-integrated graupel
    refd_max   REFD_MAX    dBZ     max lowest-level reflectivity (refl10cm)
    hail_maxk1 HAIL_MAXK1  m       max PSD hail diameter at the lowest level
    hail_max2d HAIL_MAX2D  m       max PSD hail diameter in the column
    t2max      T2MAX       K       max 2 m temperature
    t2min      T2MIN       K       min 2 m temperature

Nothing is re-derived: the WRF 4.7.1 module_diag_nwp.F windows run through the WRF route's own kernels
(woof/core/kernels/nwp_windows.cu via ``nwp_prepare_resident``/``nwp_window_resident``, and
woof/core/kernels/nwp_hail.cu via woof.core.nwp_hail), U10MAX/V10MAX/GUST through the WRF route's
woof/core/kernels/wind_gust_diag.cu (``mass_winds``: the gust reads mass-point winds), each with one mesh
cell as one WRF column (``ny = 1``, ``nx = nCells``; every kernel is column local).  The kernels see the mesh
column the way WRF's see a column:

* ``PH = 0`` and ``PHB = 9.81 * zgrid`` (as solar_diagnostics_v842), so heights are ``zgrid`` as WRF's;
* ``p`` is ``pressure_base + pressure_p``; the interface pressures of the 400 hPa test interpolate with MPAS's
  own ``fzm``/``fzp`` weights (the vertical interpolation weights to w points, WRF's ``fnm``/``fnp``);
* the density is the moist density ``rho_zz * zz * (1 + qv)`` (WRF's ``(1/alt)*(1+qv)`` with ``alt = 1/(rho_zz*zz)``);
* the winds are MPAS's cell-centre reconstructed ``uReconstructZonal/Meridional``; 10 m wind, 2 m temperature
  and the boundary-layer height are ``u10``, ``v10``, ``t2m`` and ``hpbl``.

Reset semantics.  WRF zeroes each maximum on the first step after a history write (``MOD(itimestep-1, idump) == 0``)
and then folds that step.  Here a maximum is folded after every step and zeroed right after the frame that carries
it is written (and, like MPAS's own maxima, only when a stream due at that frame carries it), so each frame's
window is (previous frame, this frame] exactly as in WRF.  The kernels' own step counter is held off their reset
(``idump`` far beyond any step), so one reset rule is in force.  The t = 0 frame carries the initial state's
fold, as MPAS's convective maxima do.  T2MAX/T2MIN are WRF's CLWRF words (woof/core/kernels/clwrf.cu
``clwrf_six``): the running extremes of ``t2 - 273`` seeded by the interval's first sample, ``+ 273`` when
published.

Definitional differences from the WRF route:

* UP_HELI_MAX is not repeated: the route's ``updraft_helicity_max`` is MPAS's 2-5 km updraft helicity
  (vertex vorticity kite-averaged to the cell, mpas_convective_diagnostics.F) and WRF's finite-difference
  ``cal_helicity`` needs the structured grid's staggered metrics; the optional 0-1/0-2/0-3 km layers are
  woof.hex.uh_layers.
* W_MEAN is not published: WRF averages w over eta 0.5-0.8 and MPAS has no eta coordinate.
* REFD_MAX reads ``refl10cm`` (``diag_physics``; Thompson/WSM6 packages); without it the field stays 0, as WRF's does without
  reflectivity.  HAIL_* run for Thompson (mp 8), Thompson-aerosol (28) and WSM6 (6) and stay 0 for the other
  schemes, WRF's no-PSD branch.
* The 400 hPa mask uses the end-of-step state (WRF: the pressure built from the start-of-step state), as
  woof/core/wind_gust_diag.py's ratified divergence 2.
* GRPL_MAX needs ``qg`` and stays 0 without it; WSPD10MAX/U10MAX/V10MAX/GUST need the surface layer package
  (``u10``) and GUST also ``hpbl``.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .utils import Direction

GROUP = "diagnostics"
OPTION = "config_nwp_diagnostics"
PACKAGE = "nwp_diagnostics"

#: MPAS field name -> (WRF name, units, description), in the order the accumulators are listed.
_ROWS = (
    ("wspd10max", "WSPD10MAX", "m s-1", "Maximum 10 m wind speed in history interval"),
    ("u10max", "U10MAX", "m s-1", "10 m u wind at the time of maximum 10 m wind speed in history interval"),
    ("v10max", "V10MAX", "m s-1", "10 m v wind at the time of maximum 10 m wind speed in history interval"),
    ("gust", "GUST", "m s-1", "Maximum UPP-method surface wind gust in history interval"),
    ("w_up_max", "W_UP_MAX", "m s-1", "Maximum updraft velocity below 400 hPa"),
    ("w_dn_max", "W_DN_MAX", "m s-1", "Minimum downdraft velocity below 400 hPa"),
    ("grpl_max", "GRPL_MAX", "kg m-2", "Maximum column integrated graupel"),
    ("refd_max", "REFD_MAX", "dBZ", "Maximum lowest-level derived radar reflectivity"),
    ("hail_maxk1", "HAIL_MAXK1", "m", "Maximum PSD hail diameter at lowest level"),
    ("hail_max2d", "HAIL_MAX2D", "m", "Maximum PSD hail diameter in column"),
    ("t2max", "T2MAX", "K", "Maximum 2 m temperature in history interval"),
    ("t2min", "T2MIN", "K", "Minimum 2 m temperature in history interval"),
)
FIELDS: dict[str, dict[str, Any]] = {
    name: {"type": "real", "dimensions": ["nCells", "Time"],
           "attrs": {"units": units, "long_name": description, "wrf_name": wrf},
           "gates": [PACKAGE]}
    for name, wrf, units, description in _ROWS}
NAMES = tuple(FIELDS)

#: MPAS config_microp_scheme -> the mp_physics number WRF's PSD hail leaf (nwp_hail.py) takes.
HAIL_MP = {"mp_thompson": 8, "mp_thompson_aerosols": 28, "mp_wsm6": 6}

#: Prefix of this module's keys in the output diagnostics' checkpoint accumulator dict.
STATE_PREFIX = "nwp:"
#: Held beyond any step count so the WRF kernels' own history reset never fires (see the module docstring).
_NO_RESET_MINUTES = 1_000_000
_G = np.float32(9.81)


def field_definition(name: str) -> dict[str, Any]:
    row = FIELDS[name]
    return {"type": row["type"], "dimensions": list(row["dimensions"]),
            "attrs": dict(row["attrs"]), "gates": list(row["gates"])}


def parse_flag(groups: Mapping[str, Mapping[str, Any]]) -> bool:
    """The option from the namelist's &diagnostics group (false when absent)."""
    supplied = {str(k).casefold(): v for k, v in groups.get(GROUP, {}).items()}
    value = supplied.get(OPTION, False)
    if isinstance(value, str) and value.strip().strip(".").casefold() in ("true", "t", "false", "f"):
        value = value.strip().strip(".").casefold() in ("true", "t")
    if not isinstance(value, bool):
        raise ValueError(f"{OPTION} is logical (.true./.false.), not {value!r}")
    return value


def packages(config: Any) -> set[str]:
    return {PACKAGE} if getattr(config, "nwp_diagnostics", False) else set()


def extend_registry(root, active: set[str]) -> None:
    """Add the fields to the ``diag`` struct and the ``diagnostics`` stream of a parsed Registry tree when the
    package is active; otherwise leave it as parsed."""
    import xml.etree.ElementTree as ET
    if PACKAGE not in active:
        return
    struct = root.find("./var_struct[@name='diag']")
    stream = root.find("./streams/stream[@name='diagnostics']")
    for name, row in FIELDS.items():
        if struct.find(f"./var[@name='{name}']") is None:
            ET.SubElement(struct, "var", {
                "name": name, "type": "real", "dimensions": " ".join(row["dimensions"]),
                "units": row["attrs"]["units"], "description": row["attrs"]["long_name"],
                "default_value": "0.0", "packages": PACKAGE})
        if stream is not None and stream.find(f"./var[@name='{name}']") is None:
            ET.SubElement(stream, "var", {"name": name, "packages": PACKAGE})


class NwpDiagnosticsModule:
    """Fold the maxima after every step, publish them before a write, reset them after it.

    ``owner`` is the route's ResidentOutputDiagnostics (its device pools, buffers and stream views).
    """

    def __init__(self, owner):
        self.owner = owner

    # -- setup -----------------------------------------------------------------------------------------
    def setup(self, utils):
        cp = self.cp = self.owner.cp
        session = self.owner.session
        dims = session.layout.dimensions
        self.nc, self.nl = int(dims["nCellsSolve"]), int(dims["nVertLevels"])
        self.utils = utils
        included = [n for n in NAMES if utils.stream_inclusion_count(n, Direction.OUTPUT) > 0]
        self.accumulates = bool(included)
        options = getattr(session.config, "execution_options", None) or {}
        self.mp = HAIL_MP.get(options.get("config_microp_scheme"), 0)
        self.thompson = self.mp in (8, 28)
        nc = self.nc
        f32 = cp.float32
        self.acc = {name: cp.zeros(nc, dtype=f32) for name in NAMES}
        # Celsius extremes (CLWRF), seeded by the first sample.
        self.acc["t2max"] = cp.full(nc, -np.inf, dtype=f32)
        self.acc["t2min"] = cp.full(nc, np.inf, dtype=f32)
        # Private carriers of the kernels' outputs this module does not publish.
        self.scratch = {name: cp.zeros(nc, dtype=f32)
                        for name in ("up_heli", "w_mean", "g_wspd", "g_wup", "g_wdn")}
        self.zero2d = cp.zeros(nc, dtype=f32)
        self.below = cp.zeros(((self.nl + 1 + 31) // 32, nc), dtype=cp.uint32)
        self.rho = cp.zeros((self.nl, nc), dtype=f32)
        self.fnm = cp.ascontiguousarray(self._get("mesh", "fzm")[:self.nl], dtype=f32)
        self.fnp = cp.ascontiguousarray(self._get("mesh", "fzp")[:self.nl], dtype=f32)
        self.znw = cp.zeros(self.nl + 1, dtype=f32)     # no eta: W_MEAN is not published
        self.ph = cp.zeros((self.nl + 1, nc), dtype=f32)

    # -- inputs ----------------------------------------------------------------------------------------
    def _has(self, pool, name):
        return any(row["pool"] == pool and row.get("pool_name", row.get("name")) == name
                   for row in self.owner.session.layout.fields)

    def _get(self, pool, name):
        return self.owner.authority(pool, name)

    def _levels(self, pool, name, levels):
        """(levels, nCells) contiguous float32 words of a native (level, cell) field."""
        cp = self.cp
        return cp.ascontiguousarray(self._get(pool, name)[:levels, :self.nc], dtype=cp.float32)

    def _plane(self, pool, name):
        cp = self.cp
        if not self._has(pool, name):
            return self.zero2d
        return cp.ascontiguousarray(self._get(pool, name)[:self.nc], dtype=cp.float32)

    def _scalar(self, name):
        key = ("state", "index_" + name)
        indices = self.owner.session.layout.indices
        # RegistryLayout keeps index_<name> = -1 for a scalar the scheme does not carry.
        if key not in indices or int(indices[key]) < 1:
            return None
        row = int(indices[key]) - 1
        return self.cp.ascontiguousarray(self._get("state", "scalars")[row, :self.nl, :self.nc],
                                         dtype=self.cp.float32)

    # -- the per-step fold -----------------------------------------------------------------------------
    def update(self):
        """The WRF windows (and PSD hail, gust, t2 extremes) on this step's end state."""
        if not self.accumulates or not self.nc:
            return
        from woof.core.kernels import get_kernel
        from woof.core.nwp_hail import update as update_psd
        from woof.core.wind_gust_diag import device_wind_maxima_step
        cp, nc, nl = self.cp, self.nc, self.nl
        f32, i32 = np.float32, np.int32
        w = self._levels("state", "w", nl + 1)
        zgrid = self._levels("mesh", "zgrid", nl + 1)
        phb = cp.multiply(zgrid, _G)
        p = cp.add(self._levels("diag", "pressure_base", nl), self._levels("diag", "pressure_p", nl))
        rho_dry = cp.multiply(self._levels("state", "rho_zz", nl), self._levels("mesh", "zz", nl))
        alt = cp.divide(f32(1), rho_dry)
        qv, qg = self._scalar("qv"), self._scalar("qg")
        u10, v10 = self._plane("diag_physics", "u10"), self._plane("diag_physics", "v10")
        has_surface = self._has("diag_physics", "u10")
        n = nc
        grid = ((n + 127) // 128,)
        get_kernel("nwp_windows", "nwp_prepare_resident")(grid, (128,),
            (p, alt, alt if qv is None else qv, i32(qv is not None), self.ph, phb, i32(1),
             self.fnm, self.fnp, self.rho, self.below, self.rho, i32(0), i32(nl), i32(n)))
        refl = self.zero2d
        if self._has("diag_physics", "refl10cm"):
            refl = self._levels("diag_physics", "refl10cm", 1)[0]
        acc, scratch = self.acc, self.scratch
        # U10MAX/V10MAX/GUST first: they follow the 10 m peak of their own running WSPD copy, so the
        # published WSPD10MAX stays the nwp window's word.
        if has_surface:
            u = self._levels("diag", "uReconstructZonal", nl).reshape(nl, 1, nc)
            v = self._levels("diag", "uReconstructMeridional", nl).reshape(nl, 1, nc)
            hpbl = self._plane("diag_physics", "hpbl").reshape(1, nc) if self._has("diag_physics", "hpbl") else None
            device_wind_maxima_step(
                u, v, w.reshape(nl + 1, 1, nc), p.reshape(nl, 1, nc), self.ph.reshape(nl + 1, 1, nc),
                phb.reshape(nl + 1, 1, nc), self.fnm, self.fnp,
                (u10.reshape(1, nc), v10.reshape(1, nc), hpbl),
                dict(wspd10max=scratch["g_wspd"], u10max=acc["u10max"], v10max=acc["v10max"],
                     w_up_max=scratch["g_wup"], w_dn_max=scratch["g_wdn"], gust_max=acc["gust"]),
                mass_winds=True)
        get_kernel("nwp_windows", "nwp_window_resident")(grid, (128,),
            (w, self.below, self.znw, self.rho if qg is None else qg, i32(qg is not None), self.rho,
             self.ph, phb, i32(1), u10, v10, refl,
             acc["wspd10max"], scratch["up_heli"], acc["w_up_max"], acc["w_dn_max"], scratch["w_mean"],
             acc["grpl_max"], acc["refd_max"], acc["hail_maxk1"], acc["hail_max2d"],
             i32(nl), i32(1), i32(nc), f32(1), f32(0), i32(_NO_RESET_MINUTES), i32(2), i32(0), i32(0)))
        kw = {}
        if self.thompson:
            kw.update(qr=self._scalar("qr"), nr=self._scalar("nr"))
        if self.mp and (qg is not None or self.mp in (8, 28) and kw["qr"] is not None):
            rho3 = self.rho.reshape(nl, 1, nc)
            update_psd(rho3 if qg is None else qg.reshape(nl, 1, nc), rho3,
                       acc["hail_maxk1"].reshape(1, nc), acc["hail_max2d"].reshape(1, nc),
                       mp_physics=self.mp,
                       **{k: v.reshape(nl, 1, nc) for k, v in kw.items() if v is not None})
        if self._has("diag_physics", "t2m"):
            # clwrf_six on t2 - 273 (Celsius, as WRF accumulates), extremes seeded by the first sample.
            c = cp.subtract(self._plane("diag_physics", "t2m"), f32(273))
            cp.maximum(acc["t2max"], c, out=acc["t2max"])
            cp.minimum(acc["t2min"], c, out=acc["t2min"])

    # -- publication and reset -------------------------------------------------------------------------
    def compute(self):
        """Publish the accumulators into the diag buffers a due stream will write."""
        cp, nc, f32 = self.cp, self.nc, np.float32
        for name in NAMES:
            if not self.utils.field_will_be_written(name):
                continue
            value = self.acc[name]
            if name in ("t2max", "t2min"):
                value = cp.where(cp.isfinite(value), cp.add(value, f32(273)), f32(0))
            self.owner.buffer("diag", name)[:nc] = value

    def reset(self):
        """MPAS's reset-after-output: only a maximum a due stream carries."""
        f32 = np.float32
        names = [n for n in NAMES if self.utils.field_will_be_written(n)]
        for name in names:
            self.acc[name][...] = (-np.inf if name == "t2max" else np.inf if name == "t2min" else 0.0)
        if {"u10max", "v10max", "wspd10max"} & set(names):
            self.scratch["g_wspd"][...] = f32(0)

    # -- checkpoint continuation -----------------------------------------------------------------------
    def state(self):
        cp = self.cp
        out = {STATE_PREFIX + name: cp.asnumpy(value) for name, value in self.acc.items()}
        out[STATE_PREFIX + "g_wspd"] = cp.asnumpy(self.scratch["g_wspd"])
        return out

    def resume(self, state):
        for key, value in state.items():
            if not key.startswith(STATE_PREFIX):
                continue
            name = key[len(STATE_PREFIX):]
            target = self.acc.get(name, self.scratch.get(name))
            if target is not None:
                target[...] = self.cp.asarray(value).reshape(target.shape)
