"""WRF's AFWA diagnostics (afwa_diag_opt and its sub-options) on the MPAS-A v8.4.2 suite route.

A WOOF-only output; MPAS-A v8.4.2 has none.  The WOOF namelist group ``&diagnostics`` takes WRF's selectors under
MPAS style names, all logical (default false)::

    config_afwa_diag_opt     base:     afwa_mslp afwa_pwat afwa_totprecip
    config_afwa_therm_opt    thermal:  afwa_heatidx afwa_wchill afwa_fits
    config_afwa_ptype_opt    ptype:    afwa_rain afwa_snow afwa_ice afwa_fzra afwa_snowfall
    config_afwa_vil_opt      VIL:      afwa_vil afwa_radarvil
    config_afwa_radar_opt    radar:    afwa_refd afwa_refd_com afwa_echotop
    config_afwa_vis_opt      vis:      afwa_vis afwa_vis_dust afwa_vis_alpha
    config_afwa_buoy_opt     buoyancy: afwa_cape afwa_cin afwa_zlfc afwa_plfc afwa_lidx afwa_cape_mu afwa_cin_mu
    config_afwa_ptype_ccn_tmp  real  WRF afwa_ptype_ccn_tmp (default 264.15 K)
    config_afwa_ptype_tot_melt real  WRF afwa_ptype_tot_melt (default 50.)

As native WRF does (module_check_a_mundo), any sub-option switches ``afwa_diag_opt`` on.  The fields are ``diag``
fields [nCells] (packages named by WRF's selector, members of the ``diagnostics`` stream), with WRF's names and
text (the tables of woof.core.afwa_diagnostics).  With every option off nothing here is allocated, constructed or
compiled, and the Registry, the default streams and the driver receipt are the ones a run without this module has.

Nothing is re-derived.  The columns run through the WRF route's own kernels
(woof/core/kernels/afwa_diagnostics.cu, afwa_buoyancy.cu: WRF 4.7.1 module_diag_afwa.F and module_diag_functions.F), one
mesh cell as one WRF column (``nx = nCells``, ``ny = 1``; every kernel is column local), exactly as
woof.core.afwa_diagnostics.update does on a structured grid.  The mesh column is handed over the way WRF hands one
over, the way nwp_diag_v842 does:

* ``PH = 0``, ``PHB = 9.81 * zgrid`` and ``alt = 1 / (rho_zz * zz)``, so the WRF route's own
  ``afwa_prepare_microphysics`` makes the density (dry, WRF's ``1/alt``), the layer thickness and the mid-level
  heights (WRF's ``z``; the lowest is ``z1``);
* ``t_phy`` is ``theta * exner`` (the temperature), ``p`` is ``pressure_base + pressure_p``, ``ht`` is ``zgrid(1)``;
* the winds are the cell-centre ``uReconstructZonal/Meridional``; ``psfc`` is ``surface_pressure`` and 2 m
  temperature, mixing ratio and 10 m wind are ``t2m``, ``q2``, ``u10``, ``v10``; ``swdown`` is ``swdnb``;
* the step amounts are ``raincv``/``rainncv`` and the running totals ``rainc``/``rainnc`` (``diag_physics``).

Timing.  WRF's driver does the instantaneous packages on the step that ends a history interval and the
precipitation-type accumulation on every step.  Here the instantaneous packages (base, thermal, radar, VIL, vis,
buoyancy) are computed on the state of a due frame (``compute``), and ptype is folded after every step from that
step's own end state (WRF folds the post-microphysics state: the one divergence).  The t = 0 frame has no step:
the instantaneous packages are computed on the initial state, the ptype accumulators are zero.  ptype accumulates
for the whole run, as WRF's does (no history reset); the accumulators ride the checkpoint's accumulator dict.

One known difference from WRF's Fortran, inherited from the WRF route's kernel (pinned by
components/hex/tests/test_mpas_afwa_diagnostics.py against the gfortran oracle): precip_type_diagnostics writes
``rh .gt. 80 .and. saturation .eqv. .false.``, which Fortran reads as ``(rh .gt. 80 .and. saturation) .eqv. .false.``;
woof/core/kernels/afwa_diagnostics.cu afwa_ptype implements the evident intent (``rh > 80 && !saturation``).  Columns
with a dry gap between two moist layers can get another precipitation type.  Every other field agrees with the
Fortran to libm round-off.

Refused by name (both would need inputs this route does not carry):

* ``config_afwa_icing_opt``: WRF selects its icing scheme from ``mp_physics`` and needs ``QNC``; none of the v8.4.2
  suite's microphysics (Thompson, Thompson aerosol-aware, WSM6, Kessler) is one WRF's icing selector accepts;
* ``config_afwa_severe_opt``: needs WRF's updraft-helicity and updraft-duration machinery (not repeated here);
* radar, VIL and visibility need the ``qr``, ``qs`` and ``qg`` carriers (visibility also ``qc`` and ``qi``), as WRF's;
  a run whose microphysics has none is refused at setup.  Visibility's dust term is WRF's no-chemistry zero.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from woof.core import afwa_diagnostics as _wrf

from .utils import Direction

GROUP = "diagnostics"
SELECTORS = ("afwa_diag_opt", "afwa_therm_opt", "afwa_ptype_opt", "afwa_vil_opt", "afwa_radar_opt", "afwa_vis_opt",
             "afwa_buoy_opt")
REFUSED = {
    "afwa_icing_opt": "WRF selects its icing scheme from mp_physics and needs QNC, which none of the v8.4.2 suite's "
                      "microphysics schemes is accepted by",
    "afwa_severe_opt": "it needs WRF's updraft-helicity and updraft-duration machinery, which is not repeated on the mesh",
}
OPTION = {name: "config_" + name for name in (*SELECTORS, *REFUSED)}
CCN_OPTION, MELT_OPTION = "config_afwa_ptype_ccn_tmp", "config_afwa_ptype_tot_melt"
OPTIONS = (*OPTION.values(), CCN_OPTION, MELT_OPTION)
#: the selector -> its WRF history rows (woof.core.afwa_diagnostics.PACKAGES)
TABLES = {selector: _wrf.PACKAGES[selector] for selector in SELECTORS}

FIELDS: dict[str, dict[str, Any]] = {}
for _selector, _rows in TABLES.items():
    for _name, (_wrf_name, _units, _text) in _rows.items():
        FIELDS[_name] = {"type": "real", "dimensions": ["nCells", "Time"],
                         "attrs": {"units": _units, "long_name": _text, "wrf_name": _wrf_name},
                         "gates": [_selector]}
NAMES = tuple(FIELDS)
PTYPE_NAMES = tuple(TABLES["afwa_ptype_opt"])
BUOY_NAMES = tuple(TABLES["afwa_buoy_opt"])
VIL_NAMES = tuple(TABLES["afwa_vil_opt"])
RADAR_NAMES = tuple(TABLES["afwa_radar_opt"])
VIS_NAMES = tuple(TABLES["afwa_vis_opt"])
THERMAL_NAMES = tuple(TABLES["afwa_therm_opt"])
BASE_NAMES = tuple(TABLES["afwa_diag_opt"])

#: Prefix of this module's keys in the output diagnostics' checkpoint accumulator dict.
STATE_PREFIX = "afwa:"
_G = np.float32(9.81)


@dataclass(frozen=True)
class AfwaConfig:
    #: the WRF selectors that are on (afwa_diag_opt always among them)
    options: tuple[str, ...] = ()
    ptype_ccn_tmp: float = 264.15
    ptype_tot_melt: float = 50.0


def field_definition(name: str) -> dict[str, Any]:
    row = FIELDS[name]
    return {"type": row["type"], "dimensions": list(row["dimensions"]),
            "attrs": dict(row["attrs"]), "gates": list(row["gates"])}


def _logical(key: str, value: Any) -> bool:
    if isinstance(value, str) and value.strip().strip(".").casefold() in ("true", "t", "false", "f"):
        value = value.strip().strip(".").casefold() in ("true", "t")
    if not isinstance(value, bool):
        raise ValueError(f"{key} is logical (.true./.false.), not {value!r}")
    return value


def resolve(groups: Mapping[str, Mapping[str, Any]]) -> AfwaConfig | None:
    """The options from the namelist's &diagnostics group (None when no AFWA option is on)."""
    supplied = {str(k).casefold(): v for k, v in groups.get(GROUP, {}).items()}
    on = {name: _logical(OPTION[name], supplied.get(OPTION[name], False)) for name in OPTION}
    for name, why in REFUSED.items():
        if on[name]:
            raise ValueError(f"&{GROUP} {OPTION[name]} is not available on the MPAS-A v8.4.2 suite route: {why}")
    ptype_keys = [k for k in (CCN_OPTION, MELT_OPTION) if k in supplied]
    values = {}
    for key, default in ((CCN_OPTION, 264.15), (MELT_OPTION, 50.0)):
        value = supplied.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError(f"{key} is real, not {value!r}")
        values[key] = float(value)
    if not any(on[s] for s in SELECTORS):
        if ptype_keys:
            raise ValueError(f"&{GROUP} {ptype_keys[0]} is set while no AFWA option is on; it would be read by nothing")
        return None
    if ptype_keys and not on["afwa_ptype_opt"]:
        raise ValueError(f"&{GROUP} {ptype_keys[0]} is set while {OPTION['afwa_ptype_opt']} is off; it would be read by nothing")
    # module_check_a_mundo: any sub-option enables the base package
    chosen = tuple(s for s in SELECTORS if on[s] or s == "afwa_diag_opt")
    return AfwaConfig(chosen, values[CCN_OPTION], values[MELT_OPTION])


def packages(config: Any) -> set[str]:
    afwa = getattr(config, "afwa_diagnostics", None)
    return set(afwa.options) if afwa is not None else set()


def extend_registry(root, active: set[str]) -> None:
    """Add the fields to the ``diag`` struct and the ``diagnostics`` stream of a parsed Registry tree for each
    active package; otherwise leave it as parsed."""
    import xml.etree.ElementTree as ET
    struct = root.find("./var_struct[@name='diag']")
    stream = root.find("./streams/stream[@name='diagnostics']")
    for name, row in FIELDS.items():
        package = row["gates"][0]
        if package not in active:
            continue
        if struct.find(f"./var[@name='{name}']") is None:
            ET.SubElement(struct, "var", {
                "name": name, "type": "real", "dimensions": " ".join(row["dimensions"]),
                "units": row["attrs"]["units"], "description": row["attrs"]["long_name"],
                "default_value": "0.0", "packages": package})
        if stream is not None and stream.find(f"./var[@name='{name}']") is None:
            ET.SubElement(stream, "var", {"name": name, "packages": package})


class AfwaModule:
    """Fold ptype after every step, compute the instantaneous packages for a due frame.  ``owner`` is the route's
    ResidentOutputDiagnostics (its device pools, buffers and stream views)."""

    def __init__(self, owner, config: AfwaConfig):
        self.owner, self.config = owner, config
        self._step = None

    # -- setup -----------------------------------------------------------------------------------------
    def setup(self, utils):
        cp = self.cp = self.owner.cp
        dims = self.owner.session.layout.dimensions
        self.nc, self.nl = int(dims["nCellsSolve"]), int(dims["nVertLevels"])
        self.utils = utils
        on = self.on = set(self.config.options)
        for need, names in (("afwa_radar_opt", ("qr", "qs", "qg")), ("afwa_vil_opt", ("qr", "qs", "qg")),
                            ("afwa_vis_opt", ("qr", "qs", "qg", "qc", "qi"))):
            missing = [n for n in names if need in on and not self._has_scalar(n)]
            if missing:
                raise ValueError(f"&{GROUP} {OPTION[need]} needs the {', '.join(missing)} carrier(s), as WRF's native "
                                 "AFWA routine reads them without a guard; the run's microphysics has none")
        self.carried = {name for selector in on for name in TABLES[selector]
                        if utils.stream_inclusion_count(name, Direction.OUTPUT) > 0}
        self.ptype = [n for n in PTYPE_NAMES if n in self.carried]
        self.accumulates = bool(self.ptype)
        # one carrier per field a kernel writes (the ptype ones persist: they accumulate), and the step's precipitation
        self.w = {name: cp.zeros(self.nc, dtype=cp.float32) for name in (*NAMES, "afwa_precip")}
        self.zero = cp.zeros(self.nc, dtype=cp.float32)

    # -- inputs ----------------------------------------------------------------------------------------
    def _has(self, pool, name):
        return any(row["pool"] == pool and row.get("pool_name", row.get("name")) == name
                   for row in self.owner.session.layout.fields)

    def _get(self, pool, name):
        return self.owner.authority(pool, name)

    def _levels(self, pool, name, levels):
        cp = self.cp
        return cp.ascontiguousarray(self._get(pool, name)[:levels, :self.nc], dtype=cp.float32)

    def _plane(self, pool, name):
        if not self._has(pool, name):
            return self.zero
        return self.cp.ascontiguousarray(self._get(pool, name)[:self.nc], dtype=self.cp.float32)

    def _has_scalar(self, name):
        # RegistryLayout keeps index_<name> = -1 for a scalar the scheme does not carry.
        indices = self.owner.session.layout.indices
        return ("state", "index_" + name) in indices and int(indices["state", "index_" + name]) >= 1

    def _scalar(self, name):
        if not self._has_scalar(name):
            return None
        row = int(self.owner.session.layout.indices["state", "index_" + name]) - 1
        return self.cp.ascontiguousarray(self._get("state", "scalars")[row, :self.nl, :self.nc], dtype=self.cp.float32)

    def _columns(self):
        """The mesh columns as WRF's AFWA route holds them: (level, cell) float32 words."""
        from woof.core.kernels import get_kernel
        cp, nc, nl = self.cp, self.nc, self.nl
        f32 = np.float32
        zgrid = self._levels("mesh", "zgrid", nl + 1)
        phb = cp.multiply(zgrid, _G)
        php = cp.zeros_like(phb)
        rho_dry = cp.multiply(self._levels("state", "rho_zz", nl), self._levels("mesh", "zz", nl))
        alt = cp.divide(f32(1), rho_dry)
        n = nc
        rho, dz = cp.zeros((nl, nc), dtype=cp.float32), cp.zeros((nl, nc), dtype=cp.float32)
        z = cp.zeros((nl, nc), dtype=cp.float32)
        z1 = cp.zeros(nc, dtype=cp.float32)
        get_kernel("afwa_diagnostics", "afwa_prepare_microphysics")(
            ((n + 127) // 128,), (128,),
            (alt, php, phb, rho, dz, z1, z, np.int32(1), np.int32(nl), np.int32(n), np.int32(1)))
        temperature = cp.ascontiguousarray(
            cp.multiply(self._levels("diag", "theta", nl), self._levels("diag", "exner", nl)), dtype=cp.float32)
        p = cp.add(self._levels("diag", "pressure_base", nl), self._levels("diag", "pressure_p", nl))
        qv = self._scalar("qv")
        return dict(rho=rho, dz=dz, z=z, z1=z1, t=temperature, t1=cp.ascontiguousarray(temperature[0]), p=p, qv=qv,
                    ht=cp.ascontiguousarray(zgrid[0]))

    # -- the per-step fold ------------------------------------------------------------------------------
    def begin(self, step, when):
        """The step the next :meth:`update` folds (steps <= 0 fold nothing)."""
        self._step = int(step) if int(step) > 0 else None

    def _base(self, col, output):
        from woof.core.kernels import get_kernel
        nc, nl, w = self.nc, self.nl, self.w
        get_kernel("afwa_diagnostics", "afwa_base")(
            ((nc + 127) // 128,), (128,),
            (col["qv"], col["rho"], col["dz"], col["t1"], col["z1"], col["ht"], self._plane("diag", "surface_pressure"),
             self._plane("diag_physics", "raincv"), self._plane("diag_physics", "rainncv"),
             self._plane("diag_physics", "rainc"), self._plane("diag_physics", "rainnc"),
             w["afwa_precip"], w["afwa_totprecip"], w["afwa_pwat"], w["afwa_mslp"], np.int32(nl), np.int32(nc),
             np.int32(self._has_scalar("qc")), np.int32(output)))

    def _need_qv(self, col):
        if col["qv"] is None:
            raise ValueError("AFWA needs the water-vapour carrier qv (WRF's MSLP and PWAT read it unconditionally)")

    def update(self):
        if not self.accumulates or self._step is None or not self.nc:
            return
        from woof.core.kernels import get_kernel
        nc, nl, w = self.nc, self.nl, self.w
        col = self._columns()
        self._need_qv(col)
        self._base(col, 0)
        get_kernel("afwa_diagnostics", "afwa_ptype")(
            ((nc + 127) // 128,), (128,),
            (col["t"], col["p"], col["qv"], col["z"], col["dz"], col["ht"], self._plane("diag_physics", "t2m"),
             self._plane("diag_physics", "swdnb"), w["afwa_precip"], *(w[n] for n in PTYPE_NAMES),
             np.float32(self.config.ptype_ccn_tmp), np.float32(self.config.ptype_tot_melt),
             np.int32(nl), np.int32(nc)))

    # -- the due frame ----------------------------------------------------------------------------------
    def compute(self):
        """Compute and publish what a due stream will write, on the state of this frame."""
        due = {n for n in self.carried if self.utils.field_will_be_written(n)}
        if not due:
            return
        from woof.core.kernels import get_kernel
        cp, i32, nc, nl, w = self.cp, np.int32, self.nc, self.nl, self.w
        col = self._columns()
        self._need_qv(col)
        launch = (((nc + 127) // 128,), (128,))
        plane = self._plane
        self._base(col, 1)
        if due & set(THERMAL_NAMES):
            get_kernel("afwa_diagnostics", "afwa_thermal")(
                *launch, (plane("diag_physics", "t2m"), plane("diag", "surface_pressure"), plane("diag_physics", "q2"),
                          plane("diag_physics", "u10"), plane("diag_physics", "v10"),
                          *(w[n] for n in THERMAL_NAMES), i32(nc)))
        qr, qs, qg = self._scalar("qr"), self._scalar("qs"), self._scalar("qg")
        radar, vil = "afwa_radar_opt" in self.on, "afwa_vil_opt" in self.on
        if (radar and due & set(RADAR_NAMES)) or (vil and due & set(VIL_NAMES)):
            ze = cp.zeros((nl, nc), dtype=cp.float32)
            get_kernel("afwa_diagnostics", "afwa_dbz")(
                ((nc * nl + 127) // 128,), (128,), (col["rho"], col["t"], qr, qs, qg, ze, i32(nc * nl)))
            get_kernel("afwa_diagnostics", "afwa_radar_vil")(
                *launch, (qr, qs, qg, col["rho"], col["dz"], col["z"], ze,
                          *(w[n] for n in VIL_NAMES), *(w[n] for n in RADAR_NAMES),
                          i32(nl), i32(nc), i32(vil), i32(radar)))
        if "afwa_vis_opt" in self.on and due & set(VIS_NAMES):
            get_kernel("afwa_diagnostics", "afwa_visibility")(
                *launch, (col["t"], col["p"], col["qv"], col["z"],
                          self._levels("diag", "uReconstructZonal", nl), self._levels("diag", "uReconstructMeridional", nl),
                          col["ht"], qr, qs, qg, col["rho"], plane("diag", "surface_pressure"), plane("diag_physics", "t2m"),
                          plane("diag_physics", "q2"), w["afwa_pwat"], *(w[n] for n in VIS_NAMES), i32(nl), i32(nc)))
        if "afwa_buoy_opt" in self.on and due & set(BUOY_NAMES):
            work = (cp.zeros((2, nl, nc), dtype=cp.float32), cp.zeros((2, nl, nc), dtype=cp.float32),
                    cp.zeros((2, nl, nc), dtype=cp.float32), cp.zeros((2, 6, nc), dtype=cp.float32))
            _wrf.launch_buoyancy((col["t"], col["p"], col["qv"], col["z"], col["ht"],
                                  *(w[n] for n in BUOY_NAMES), *work, i32(nl), i32(nc)))
        for name in due:
            self.owner.buffer("diag", name)[:nc] = w[name]

    # -- checkpoint continuation -----------------------------------------------------------------------
    def state(self):
        cp = self.cp
        return {STATE_PREFIX + n: cp.asnumpy(self.w[n]) for n in self.ptype}

    def resume(self, state):
        for key, value in state.items():
            if key.startswith(STATE_PREFIX) and key[len(STATE_PREFIX):] in self.ptype:
                target = self.w[key[len(STATE_PREFIX):]]
                target[...] = self.cp.asarray(value).reshape(target.shape)
