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
    config_afwa_severe_opt   severe:   wspd10max afwa_llws w_up_max w_dn_max tcoli_max grpl_flx_max afwa_hail
                                       afwa_tornado afwa_cape afwa_cin afwa_zlfc afwa_plfc afwa_lidx
    config_afwa_ptype_ccn_tmp  real  WRF afwa_ptype_ccn_tmp (default 264.15 K)
    config_afwa_ptype_tot_melt real  WRF afwa_ptype_tot_melt (default 50.)

As native WRF does (module_check_a_mundo), any sub-option switches ``afwa_diag_opt`` on.  The fields are ``diag``
fields [nCells] (packages named by WRF's selector, members of the ``diagnostics`` stream), with WRF's names and
text (the tables of woof.core.afwa_diagnostics).  With every option off nothing here is allocated, constructed or
compiled, and the Registry, the default streams and the driver receipt are the ones a run without this module has.

Nothing is re-derived.  The columns run through the WRF route's own kernels
(woof/core/kernels/afwa_diagnostics.cu, afwa_buoyancy.cu, afwa_severe.cu: WRF 4.7.1 module_diag_afwa.F and
module_diag_functions.F), one mesh cell as one WRF column (``nx = nCells``, ``ny = 1``), exactly as
woof.core.afwa_diagnostics.update does on a structured grid.  The mesh column is handed over the way WRF hands one
over, the way nwp_diag_v842 does:

* ``PH = 0``, ``PHB = 9.81 * zgrid`` and ``alt = 1 / (rho_zz * zz)``, so the WRF route's own
  ``afwa_prepare_microphysics`` makes the density (dry, WRF's ``1/alt``), the layer thickness and the mid-level
  heights (WRF's ``z``; the lowest is ``z1``);
* ``t_phy`` is ``theta * exner`` (the temperature), ``p`` is ``pressure_base + pressure_p``, ``ht`` is ``zgrid(1)``;
* the winds are the cell-centre ``uReconstructZonal/Meridional``; ``psfc`` is ``surface_pressure`` and 2 m
  temperature, mixing ratio and 10 m wind are ``t2m``, ``q2``, ``u10``, ``v10``; ``swdown`` is ``swdnb``;
* the step amounts are ``raincv``/``rainncv`` and the running totals ``rainc``/``rainnc`` (``diag_physics``);
* severe's ``w_2`` is the state ``w`` at the bottom face of each layer, as WRF's.

Timing.  WRF's driver does the instantaneous packages on the step that ends a history interval and the
precipitation-type accumulation and the severe maxima on every step.  Here the instantaneous packages (base,
thermal, radar, VIL, vis, buoyancy) are computed on the state of a due frame (``compute``); ptype and severe are
folded after every step from that step's own end state (WRF folds the post-microphysics state for ptype, and the
start-of-step w and helicity for severe).  The t = 0 frame has no step: the instantaneous packages are computed on
the initial state, the accumulators are zero.  ptype accumulates for the whole run, as WRF's does (no history
reset); the severe maxima restart after each frame that writes one of them (WRF: the first step after every history
dump), and midrh_min_old takes the closed window's midrh_min then, as WRF's.  The accumulators ride the
checkpoint's accumulator dict.

Severe (``config_afwa_severe_opt``).  As in WRF (and woof.core.afwa_severe) the package has two parts:

* every step, for every microphysics: the 10 m gust maximum (``wspd10max``, blended with the 1 km wind under
  heavy precipitation) and the 0-2000 ft shear maximum (``afwa_llws``): ``afwa_severe_driver``;
* WRF calls severe_wx_diagnostics only when the Registry carries ``QV QR QC QI QS QG`` and graupel number
  ``QNG`` (F_QNG), although the routine never reads QNG.  The same guard is kept here (MPAS's ``ng``): with it,
  the column part (``afwa_severe_column``: w maxima below 400 hPa, column ice, graupel flux, the mid-level RH
  minimum and hail; the surface-parcel buoyancy on the first step and on output steps; the tornado wind) runs per
  cell.  None of this route's microphysics (Thompson, Thompson aerosol-aware, WSM6, Kessler) carries ``ng``, so,
  as WRF's own result for mp_physics 8, 28 and 6, those fields stay 0 and only the gust and shear maxima move.

  Definitional differences from WRF in the column part (all reached only with an ``ng`` carrier):

  - the helicity the hail and tornado read (WRF's ``uh``, the step's 2-5 km updraft helicity from
    module_diffusion_em.F cal_helicity) is MPAS's own: the per-step value whose interval maximum the convective
    diagnostics publish as ``updraft_helicity_max`` (mpas_convective_diagnostics.F, the same kernel on a scratch
    row).  MPAS sums ``max(w, 0) * max(zeta, 0)`` over the part of each layer inside 2-5 km AGL, with zeta the
    vertex vorticity kite-averaged to the cell.  WRF sums ``w * zeta`` (signed) over the layers entirely inside
    the band from finite-difference vorticity, sets the column to 0 where any 8-point mean w in the band is not
    positive, and smooths the result over 9 points; WRF samples the first RK stage, MPAS the end of the step;
  - WRF's UP_HELI_MAX (reset by this package) is not repeated: ``updraft_helicity_max`` is the route's field;
  - the tornado duration takes the maximum of the previous mask and duration over the cell and its
    ``cellsOnCell`` neighbours (``afwa_severe_tornado_mesh``) where WRF takes the 3x3 box; WRF's west/south
    edge row exclusion has no mesh counterpart.  A multi-rank run (a halo) is refused for severe, as the
    neighbours' previous words would need an exchange;
  - ``w_mean`` (in WRF's package list, written only by nwp_diagnostics) is not published.

  ``wspd10max``, ``w_up_max`` and ``w_dn_max`` are one WRF array each, written by this package and by
  nwp_diagnostics; ``config_nwp_diagnostics`` together with ``config_afwa_severe_opt`` is refused.

One known difference from WRF's Fortran, inherited from the WRF route's kernel and kept by decision (2026-11-13;
pinned by components/hex/tests/test_mpas_afwa_diagnostics.py against the gfortran oracle): precip_type_diagnostics
writes ``rh .gt. 80 .and. saturation .eqv. .false.``, which Fortran reads as
``(rh .gt. 80 .and. saturation) .eqv. .false.`` (``.eqv.`` binds weaker than ``.and.``), true at every level that is
not both moist and already saturated, so WRF re-takes the cloud top at dry levels too.
woof/core/kernels/afwa_diagnostics.cu afwa_ptype implements the evident intent (``rh > 80 && !saturation``).
Columns with a dry gap between two moist layers can get another precipitation type.  Every other field agrees with
the Fortran to libm round-off.

Refused by name:

* ``config_afwa_icing_opt``: WRF 4.7.1 calls icing_diagnostics only when the Registry carries ``QC``, ``QR`` and
  cloud droplet number ``QNC`` (F_QNC) and its selector names the scheme: of the schemes it names only WDM6
  (mp_physics 16) and Morrison two-moment with aerosols (40) carry QNC.  Thompson (8) and WSM5/WSM6 (4/6) are named
  (droplet-number-free fits) but carry no QNC, so WRF skips them; Thompson aerosol-aware (28) carries QNC but the
  selector does not name it.  WRF has no path that derives QNC.  So none of this route's schemes (mp_thompson,
  mp_thompson_aerosols, mp_wsm6, mp_kessler) gets WRF's icing, and the option is refused rather than written as
  zeros;
* radar, VIL and visibility need the ``qr``, ``qs`` and ``qg`` carriers (visibility also ``qc`` and ``qi``), which
  WRF reads without a guard: on this route mp_wsm6, mp_thompson and mp_thompson_aerosols carry them all;
  any other microphysics (mp_kessler, none) is refused at configuration.  Visibility's dust term is WRF's
  no-chemistry zero.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from woof.core import afwa_diagnostics as _wrf

from .utils import Direction

GROUP = "diagnostics"
SELECTORS = ("afwa_diag_opt", "afwa_therm_opt", "afwa_ptype_opt", "afwa_vil_opt", "afwa_radar_opt", "afwa_vis_opt",
             "afwa_buoy_opt", "afwa_severe_opt")
#: MPAS config_microp_scheme values that carry every hydrometeor WRF's radar, VIL and visibility read
HYDROMETEOR_SCHEMES = ("mp_wsm6", "mp_thompson", "mp_thompson_aerosols")
REFUSED = {
    "afwa_icing_opt": "WRF 4.7.1 calls icing_diagnostics only when the Registry carries QC, QR and the cloud droplet "
                      "number QNC (F_QNC) and its selector names the scheme, which leaves WDM6 (mp_physics 16) and "
                      "Morrison two-moment with aerosols (40); Thompson (8) and WSM6 (6) carry no QNC, Thompson "
                      "aerosol-aware (28) carries it but is not named, and WRF has no path that derives QNC, so none "
                      "of this route's microphysics (mp_thompson, mp_thompson_aerosols, mp_wsm6, mp_kessler) gets "
                      "WRF's icing",
}
OPTION = {name: "config_" + name for name in (*SELECTORS, *REFUSED)}
CCN_OPTION, MELT_OPTION = "config_afwa_ptype_ccn_tmp", "config_afwa_ptype_tot_melt"
OPTIONS = (*OPTION.values(), CCN_OPTION, MELT_OPTION)
#: The severe package's WRF rows this route publishes: WRF's UP_HELI_MAX is the route's updraft_helicity_max and
#: W_MEAN is nwp_diagnostics' (see the module docstring).
SEVERE_ROWS = {name: row for name, row in _wrf.SEVERE_HISTORY.items() if name not in ("up_heli_max", "w_mean")}
#: the selector -> its WRF history rows (woof.core.afwa_diagnostics.PACKAGES)
TABLES = {selector: (SEVERE_ROWS if selector == "afwa_severe_opt" else _wrf.PACKAGES[selector])
          for selector in SELECTORS}
#: One WRF array each, written by the severe package and by nwp_diagnostics (woof.hex.mpasdiag.nwp_diag_v842).
NWP_SHARED = ("wspd10max", "w_up_max", "w_dn_max")

FIELDS: dict[str, dict[str, Any]] = {}
for _selector, _rows in TABLES.items():
    for _name, (_wrf_name, _units, _text) in _rows.items():
        if _name in FIELDS:                     # afwa_cape .. afwa_lidx: the buoyancy and the severe package
            FIELDS[_name]["gates"] = [FIELDS[_name]["gates"][0] + ";" + _selector]
            continue
        FIELDS[_name] = {"type": "real", "dimensions": ["nCells", "Time"],
                         "attrs": {"units": _units, "long_name": _text, "wrf_name": _wrf_name},
                         "gates": [_selector + (";nwp_diagnostics" if _name in NWP_SHARED else "")]}
NAMES = tuple(FIELDS)
PTYPE_NAMES = tuple(TABLES["afwa_ptype_opt"])
BUOY_NAMES = tuple(TABLES["afwa_buoy_opt"])
VIL_NAMES = tuple(TABLES["afwa_vil_opt"])
RADAR_NAMES = tuple(TABLES["afwa_radar_opt"])
VIS_NAMES = tuple(TABLES["afwa_vis_opt"])
THERMAL_NAMES = tuple(TABLES["afwa_therm_opt"])
BASE_NAMES = tuple(TABLES["afwa_diag_opt"])
SEVERE_NAMES = tuple(TABLES["afwa_severe_opt"])
#: the surface-parcel buoyancy the severe package writes (WRF's first five buoyancy rows)
SEVERE_BUOY_NAMES = BUOY_NAMES[:5]
#: severe carriers that are not output but persist between steps (and ride a checkpoint)
SEVERE_STATE = ("afwa_up_heli_max", "afwa_midrh_min", "afwa_midrh_min_old", "afwa_tornado_mask", "afwa_tornado_dur")
#: the scalars WRF's F_QNG guard asks for (MPAS names)
SEVERE_COLUMN_SCALARS = ("qv", "qr", "qc", "qi", "qs", "qg", "ng")

#: Prefix of this module's keys in the output diagnostics' checkpoint accumulator dict.
STATE_PREFIX = "afwa:"
SEVERE_RESET_KEY = STATE_PREFIX + "severe_reset"
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
    if on["afwa_severe_opt"]:
        from .nwp_diag_v842 import OPTION as NWP_OPTION, parse_flag
        if parse_flag(groups):
            raise ValueError(f"&{GROUP} {OPTION['afwa_severe_opt']} with {NWP_OPTION} is not available: WRF's "
                             f"{', '.join(n.upper() for n in NWP_SHARED)} are one array each, written by both "
                             "packages' windows; choose one")
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


def check(config: Any) -> None:
    """Refuse at configuration what the run's microphysics cannot feed: WRF's radar, VIL and visibility read the
    rain, snow and graupel (visibility also cloud water and ice) arrays without a guard."""
    afwa = getattr(config, "afwa_diagnostics", None)
    if afwa is None:
        return
    scheme = config.selectors.get("microphysics") if config.active("microphysics") else "off"
    for selector in ("afwa_radar_opt", "afwa_vil_opt", "afwa_vis_opt"):
        if selector in afwa.options and scheme not in HYDROMETEOR_SCHEMES:
            needs = "qr, qs, qg, qc and qi" if selector == "afwa_vis_opt" else "qr, qs and qg"
            raise ValueError(f"&{GROUP} {OPTION[selector]} needs the {needs} carriers, which WRF's AFWA routine reads "
                             f"without a guard; on this route {', '.join(HYDROMETEOR_SCHEMES)} carry them, and this "
                             f"run's microphysics is {scheme!r}")


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
        gates = [p for p in row["gates"][0].split(";") if p in SELECTORS]
        if not set(gates) & active:
            continue
        package = ";".join(gates)
        if struct.find(f"./var[@name='{name}']") is None:
            ET.SubElement(struct, "var", {
                "name": name, "type": "real", "dimensions": " ".join(row["dimensions"]),
                "units": row["attrs"]["units"], "description": row["attrs"]["long_name"],
                "default_value": "0.0", "packages": package})
        if stream is not None and stream.find(f"./var[@name='{name}']") is None:
            ET.SubElement(stream, "var", {"name": name, "packages": package})


class AfwaModule:
    """Fold ptype and severe after every step, compute the instantaneous packages for a due frame.  ``owner`` is
    the route's ResidentOutputDiagnostics (its device pools, buffers, stream views, convective module and dycore)."""

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
        self.severe = "afwa_severe_opt" in on
        # WRF's F_QNG guard on severe_wx_diagnostics (the module docstring)
        self.severe_column = self.severe and all(self._has_scalar(n) for n in SEVERE_COLUMN_SCALARS)
        self.accumulates = bool(self.ptype) or self.severe
        # one carrier per field a kernel writes (the ptype and severe ones persist: they accumulate), and the step's
        # precipitation
        self.w = {name: cp.zeros(self.nc, dtype=cp.float32) for name in (*NAMES, "afwa_precip")}
        self.zero = cp.zeros(self.nc, dtype=cp.float32)
        self.persistent = list(self.ptype)
        if self.severe:
            if getattr(self.owner, "halo", None) is not None:
                raise ValueError(f"&{GROUP} {OPTION['afwa_severe_opt']} is not available on a multi-rank mesh: the "
                                 "tornado duration reads the neighbours' previous words, which are not exchanged")
            self.dt = np.float32(self.owner.dycore.config.config_dt)
            self.w.update((name, cp.zeros(self.nc, dtype=cp.float32)) for name in SEVERE_STATE)
            self.persistent += [*SEVERE_NAMES, *SEVERE_STATE]
            self._reset = True          # WRF: the first step follows the t = 0 history dump
            self._uh = None
            # the step's shear (WRF's local llws; kept between steps as the WRF route's scratch is)
            self._llws = cp.zeros(self.nc, dtype=cp.float32)

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

    def helicity(self):
        """This step's 2-5 km updraft helicity per cell: MPAS's update_diag (mpas_convective_diagnostics.F) on the
        route's own inputs, folded into a zeroed scratch row, so the value is the step's own (the one whose interval
        maximum is ``updraft_helicity_max``)."""
        from .convective_v842 import ConvectiveDiagnostics
        cp, convective = self.cp, self.owner.convective
        if self._uh is None:
            outputs = cp.zeros((17, convective.n_alloc), dtype=cp.float32)
            driver = ConvectiveDiagnostics(outputs, self.nc, self.nl, ("updraft_helicity_max",))
            driver.vertex_order = convective.driver.vertex_order
            self._uh = (outputs, driver)
        outputs, driver = self._uh
        outputs[14].fill(0)
        driver.update(convective._arrays(("zgrid", "w", "areaCell", "cellsOnVertex", "kiteAreasOnVertex",
                                          "vorticity")))
        return cp.ascontiguousarray(outputs[14, :self.nc])

    def neighbours(self):
        """(cellsOnCell (maxEdges, nCells) one-based, nEdgesOnCell (nCells,)) as int32 device words."""
        cp, nc = self.cp, self.nc
        cells = cp.ascontiguousarray(self._get("mesh", "cellsOnCell")[:, :nc], dtype=cp.int32)
        edges = cp.ascontiguousarray(self._get("mesh", "nEdgesOnCell")[:nc], dtype=cp.int32)
        return cells, edges

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
        if self.severe:
            self._severe(col)
        if self.ptype:
            get_kernel("afwa_diagnostics", "afwa_ptype")(
                ((nc + 127) // 128,), (128,),
                (col["t"], col["p"], col["qv"], col["z"], col["dz"], col["ht"], self._plane("diag_physics", "t2m"),
                 self._plane("diag_physics", "swdnb"), w["afwa_precip"], *(w[n] for n in PTYPE_NAMES),
                 np.float32(self.config.ptype_ccn_tmp), np.float32(self.config.ptype_tot_melt),
                 np.int32(nl), np.int32(nc)))

    def _severe(self, col):
        """woof.core.afwa_severe.update on the mesh columns (WRF 4.7.1 afwa_diagnostics_driver's severe blocks)."""
        from woof import wrf_exact
        from woof.core.afwa_severe import surface_buoyancy
        from woof.core.kernels import get_kernel
        cp, nc, nl, w, i32 = self.cp, self.nc, self.nl, self.w, np.int32
        plane = (((nc + 127) // 128,), (128,))
        reset, self._reset = self._reset, False
        llws = self._llws
        get_kernel("afwa_severe", "afwa_severe_driver")(
            *plane, (self._levels("diag", "uReconstructZonal", nl), self._levels("diag", "uReconstructMeridional", nl),
                     col["z"], col["ht"], self._plane("diag_physics", "u10"), self._plane("diag_physics", "v10"),
                     w["afwa_precip"], w["wspd10max"], w["afwa_llws"], llws, w["w_up_max"], w["w_dn_max"],
                     w["tcoli_max"], w["grpl_flx_max"], w["afwa_up_heli_max"], w["afwa_tornado"],
                     w["afwa_midrh_min_old"], w["afwa_midrh_min"], w["afwa_hail"],
                     i32(reset), self.dt, i32(nl), i32(nc)))
        if not self.severe_column:
            return
        uh = self.helicity()
        rh = cp.zeros((nl, nc), dtype=cp.float32) if wrf_exact.ENABLED else self.zero
        if wrf_exact.ENABLED:
            get_kernel("afwa_severe", "afwa_severe_humidity")(
                ((nc * nl + 255) // 256,), (256,), (col["p"], col["t"], col["qv"], rh, i32(nc * nl)))
        get_kernel("afwa_severe", "afwa_severe_column")(
            *plane, (self._levels("state", "w", nl), col["t"], col["z"], col["p"],
                     self._scalar("qi"), self._scalar("qs"), self._scalar("qg"), col["rho"], col["dz"], rh, col["ht"],
                     self._plane("diag_physics", "t2m"), uh, w["afwa_midrh_min_old"], w["afwa_midrh_min"],
                     w["w_up_max"], w["w_dn_max"], w["afwa_hail"], w["tcoli_max"], w["grpl_flx_max"], i32(nl), i32(nc),
                     col["qv"], i32(not wrf_exact.ENABLED)))
        # WRF: on the first step and on output steps
        if self._step == 1 or any(self.utils.field_will_be_written(n) for n in self.carried):
            work = (cp.zeros((2, nl, nc), dtype=cp.float32), cp.zeros((2, nl, nc), dtype=cp.float32),
                    cp.zeros((2, nl, nc), dtype=cp.float32), cp.zeros((2, 6, nc), dtype=cp.float32))
            mu = (cp.zeros(nc, dtype=cp.float32), cp.zeros(nc, dtype=cp.float32))
            surface_buoyancy((col["t"], col["p"], col["qv"], col["z"], col["ht"],
                              *(w[n] for n in SEVERE_BUOY_NAMES), *mu, *work, i32(nl), i32(nc)))
        mask_prev, dur_prev = w["afwa_tornado_mask"].copy(), w["afwa_tornado_dur"].copy()
        cells, edges = self.neighbours()
        get_kernel("afwa_severe", "afwa_severe_tornado_mesh")(
            *plane, (uh, llws, w["afwa_zlfc"], w["afwa_midrh_min"], w["afwa_midrh_min_old"], mask_prev, dur_prev,
                     w["afwa_tornado_mask"], w["afwa_tornado_dur"], w["afwa_tornado"], cells, edges, self.dt, i32(nc)))

    def reset(self):
        """After a frame is written: the severe maxima restart on the next step if a due stream carried one."""
        if self.severe and any(self.utils.field_will_be_written(n) for n in SEVERE_NAMES if n in self.carried):
            self._reset = True

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
        state = {STATE_PREFIX + n: cp.asnumpy(self.w[n]) for n in self.persistent}
        if self.severe:
            state[SEVERE_RESET_KEY] = np.array([int(self._reset)], dtype=np.int32)
        return state

    def resume(self, state):
        for key, value in state.items():
            if key.startswith(STATE_PREFIX) and key[len(STATE_PREFIX):] in self.persistent:
                target = self.w[key[len(STATE_PREFIX):]]
                target[...] = self.cp.asarray(value).reshape(target.shape)
        if self.severe and SEVERE_RESET_KEY in state:
            self._reset = bool(np.asarray(state[SEVERE_RESET_KEY]).reshape(-1)[0])
