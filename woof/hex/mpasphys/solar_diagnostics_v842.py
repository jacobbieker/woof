"""WRF-Solar diagnostics on the MPAS suite route (config_solar_diagnostics).

A WOOF-only output; MPAS-A v8.4.2 has none.  The WOOF namelist group
``&diagnostics`` option ``config_solar_diagnostics`` (logical, default
false; :func:`resolve`) adds WRF v4.7.1 module_diag_solar.F's 26 diagnostics
(woof.core.solar_diagnostics.SOLAR_OUTPUTS) as ``diag_physics`` fields
[nCells] named as WRF's Registry/registry.solar_fields state variables
(lower case, MPAS style: clrnidx, sza, ghi_accum, cldfrac2d, wvp, lwp, ...),
package-gated so a run without it allocates nothing new, and members of the
``diagnostics`` stream.

The routine is WRF's own (woof/core/kernels/solar_diag.cu, every word held to
compiled WRF, MPAS-shaped inputs included: tests/test_solar_diag_wrf471_oracle.py).
It runs after every completed MPAS step (woof.hex.mpasphys.runtime_v842), the
point WRF's diagnostics driver calls it, on MPAS's own words:

* the end-of-step moisture (``state.scalars`` qv/qc/qi/qs, unclamped);
* the physics preparation's moist density ``rho_p`` (= zz*rho_zz*(1+qv)) and
  ``dz_p`` (zgrid differences) of that state -- WRF's ``rho*dz/(1+qv)``;
* heights as WRF's geopotential: PH = 0 and PHB = 9.81*zgrid (float32), so
  CBASEHT/CTOPHT are WRF's ((PH+PHB)(k)+(PH+PHB)(k+1))/(2g) of the w-level
  heights above sea level (WRF's own "m agl" label notwithstanding);
* the last radiation call's ``cldfrac``, ``coszr`` (MPAS's radiation-interval
  midpoint cosine), ``swdnb``/``swdnt`` and the radiation's qc/qi after the
  cloudiness scheme (``qcrad_p``/``qirad_p``) as QC_TOT/QI_TOT -- MPAS forms
  these on radiation steps only (WRF's driver re-forms them every step under
  solar_diagnostics = 1; MPAS has no per-step cloudiness to call);
* the microphysics' effective radii ``re_cloud/re_ice/re_snow`` in metres --
  WRF's words exactly -- when the MPAS shortwave reads them
  (``config_microp_re`` with Thompson or WSM6).

GHI_ACCUM sums swdnb*dt over the run segment; like WRF (no restart I/O) a
checkpoint does not carry it (checkpoint_v842), so a resumed run starts it at
zero.  A storm-following move carries it with the other diag rows.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from woof.core.solar_diagnostics import NOUT, SOLAR_OUTPUTS

#: The WOOF-only namelist group and its one option.
GROUP = "diagnostics"
OPTION = "config_solar_diagnostics"
PACKAGE = "solar_diagnostics"

#: MPAS field name -> (WRF name, units, description), in the kernel's order.
FIELDS: dict[str, dict[str, Any]] = {
    name.lower(): {"type": "real", "dimensions": ["nCells", "Time"],
                   "attrs": {"units": units.strip() or "1", "long_name": description,
                             "wrf_name": name},
                   "gates": [PACKAGE]}
    for _slot, name, units, description in SOLAR_OUTPUTS}
NAMES = tuple(FIELDS)


def field_definition(name: str) -> dict[str, Any]:
    row = FIELDS[name]
    return {"type": row["type"], "dimensions": list(row["dimensions"]),
            "attrs": dict(row["attrs"]), "gates": list(row["gates"])}


def resolve(groups: Mapping[str, Mapping[str, Any]]) -> bool:
    """The option from the namelist's &diagnostics group (false when absent)."""
    supplied = {str(k).casefold(): v for k, v in groups.get(GROUP, {}).items()}
    unknown = sorted(set(supplied) - {OPTION})
    if unknown:
        raise ValueError(f"&{GROUP} has no option {unknown[0]!r}; its one (WOOF) option is {OPTION}")
    value = supplied.get(OPTION, False)
    if isinstance(value, str) and value.strip().strip(".").casefold() in ("true", "t", "false", "f"):
        value = value.strip().strip(".").casefold() in ("true", "t")
    if not isinstance(value, bool):
        raise ValueError(f"{OPTION} is logical (.true./.false.), not {value!r}")
    return value


def packages(config: Any) -> set[str]:
    return {PACKAGE} if getattr(config, "solar_diagnostics", False) else set()


def check(config: Any) -> None:
    """Refuse the option where its inputs do not exist: the GPU RRTMG shortwave
    (swdnb/swdnt/coszr, and the radiation's cloud copies) and resident physics."""
    name = f"&{GROUP} {OPTION}=.true."
    if config.selectors.get("radiation_sw") != "rrtmg_sw":
        raise ValueError(f"{name} needs the RRTMG shortwave (config_radt_sw_scheme='rrtmg_sw'): the "
                         "clearness index and GHI read its SWDNB/SWDNT; this run has "
                         f"radiation_sw={config.selectors.get('radiation_sw')!r}")
    from .driver_v842 import Backend
    if config.backends.get("radiation_sw") != Backend.HEX_CUDA:
        raise ValueError(f"{name} needs radiation_sw on hex-cuda (it reads the GPU radiation's "
                         "cloud copies, qcrad_p/qirad_p)")


def extend_registry(root, active: set[str]) -> None:
    """Add the 26 diag_physics fields and diagnostics-stream members to a parsed
    Registry tree when the package is active; otherwise leave it as parsed."""
    import xml.etree.ElementTree as ET
    if PACKAGE not in active:
        return
    struct = root.find("./var_struct[@name='diag_physics']")
    stream = root.find("./streams/stream[@name='diagnostics']")
    for name, row in FIELDS.items():
        if struct.find(f"./var[@name='{name}']") is None:
            ET.SubElement(struct, "var", {
                "name": name, "type": "real", "dimensions": " ".join(row["dimensions"]),
                "units": row["attrs"]["units"], "description": row["attrs"]["long_name"],
                "default_value": "0.0", "packages": PACKAGE})
        if stream is not None and stream.find(f"./var[@name='{name}']") is None:
            ET.SubElement(stream, "var", {"name": name, "packages": PACKAGE})


def mpas_solar_inputs(cp, *, zgrid, rho_p, dz_p):
    """WRF's PH, PHB, rho and dz8w from MPAS fields (float32): PH = 0, PHB =
    9.81*zgrid, rho = rho_p (MPAS's moist physics density), dz8w = dz_p."""
    zgrid = cp.ascontiguousarray(zgrid, dtype=cp.float32)
    phb = cp.multiply(zgrid, np.float32(9.81))
    return (cp.zeros_like(phb), cp.ascontiguousarray(phb),
            cp.ascontiguousarray(rho_p, dtype=cp.float32),
            cp.ascontiguousarray(dz_p, dtype=cp.float32))


class MpasSolarDiagnostics:
    """Runs solar_diag.cu after each MPAS step and publishes the 26 fields."""

    def __init__(self, session, coupling, dycore):
        self.session, self.coupling, self.dycore = session, coupling, dycore
        self.cp = cp = dycore.cp
        dims = session.layout.dimensions
        self.nl, self.nc = int(dims["nVertLevels"]), int(dims["nCellsSolve"])
        names = tuple(coupling.names)
        self.rows = {name: names.index(name) for name in ("qv", "qc", "qi", "qs") if name in names}
        from .farms_v842 import radii_declared
        declared = radii_declared(session.config.execution_options)
        self.has_req = (1, 1, 1) if declared else (0, 0, 0)
        self.out = cp.zeros((NOUT, self.nc), dtype=cp.float32)   # GHI_ACCUM sums from zero
        self.work = cp.empty((self.nl, self.nc), dtype=cp.float32)
        self.zero = cp.zeros((self.nl, self.nc), dtype=cp.float32)
        self.steps = 0

    def _prof(self, a, levels=None):
        n = self.nl if levels is None else levels
        return self.cp.ascontiguousarray(self.cp.asarray(a, dtype=self.cp.float32)[:n, :self.nc])

    def _surf(self, a):
        return self.cp.ascontiguousarray(self.cp.asarray(a, dtype=self.cp.float32)[:self.nc])

    def step(self, *, dt):
        from ..cuda_physics_prep_v841 import prepare_mpas_to_phys_cuda_v841
        from woof.core.solar_diagnostics import run_solar_diag, species_flags
        cp, s, c = self.cp, self.session, self.coupling
        atmosphere = self.dycore.atmosphere
        chemistry = getattr(c, "chemistry", None)
        if chemistry is not None:
            atmosphere = chemistry.registry_view(atmosphere)
        columns = prepare_mpas_to_phys_cuda_v841(
            atmosphere, scalar_names=c.names, geometry=c.geometry,
            kernel_cache=c.physics_cache, registry_scalar_names=c.names)
        scalars = atmosphere.state.scalars
        q = {name: self._prof(scalars[row]) for name, row in self.rows.items()}
        ph, phb, rho, dz = mpas_solar_inputs(
            cp, zgrid=self._prof(s.field("mesh", "zgrid"), self.nl + 1),
            rho_p=self._prof(columns.rho_p), dz_p=self._prof(columns.dz_p))
        diag = lambda name: s.field("diag_physics", name)
        pools = s.device_pools
        qc_tot = self._prof(pools.prepared("qcrad_p")) if "qc" in q else self.zero
        qi_tot = self._prof(pools.prepared("qirad_p")) if "qi" in q else self.zero
        radii = [self._prof(diag(n)) if self.has_req[i] else self.zero
                 for i, n in enumerate(("re_cloud", "re_ice", "re_snow"))]
        flags = species_flags(qv="qv" in q, qc="qc" in q, qi="qi" in q, qs="qs" in q,
                              has_reqc=self.has_req[0], has_reqi=self.has_req[1],
                              has_reqs=self.has_req[2])
        run_solar_diag(
            nz=self.nl, ncol=self.nc, dt=float(dt), flags=flags, rho=rho, dz8w=dz,
            ph=ph, phb=phb, cldfra=self._prof(diag("cldfrac")),
            coszen=self._surf(diag("coszr")), swdnb=self._surf(diag("swdnb")),
            swdnt=self._surf(diag("swdnt")),
            qv=q["qv"], qc=q.get("qc", self.zero), qi=q.get("qi", self.zero),
            qs=q.get("qs", self.zero), qc_tot=qc_tot, qi_tot=qi_tot,
            re_cloud=radii[0], re_ice=radii[1], re_snow=radii[2],
            out=self.out, work=self.work)
        for i, name in enumerate(NAMES):
            diag(name)[:self.nc] = self.out[i]
        self.steps += 1
