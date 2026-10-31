"""WRF-Solar EPS on the MPAS suite route (``&wrfsolar_eps``), and MAD-WRF's
refusal.

A WOOF-only option; MPAS-A v8.4.2 has no stochastic perturbation of its
physics inputs.  The WOOF namelist group ``&wrfsolar_eps`` carries WRF's
``&stoch`` multi_perturb keys under MPAS-style names (:data:`OPTIONS`):
``config_multi_perturb`` and, per scheme, a switch and its factors.  The
per-scheme statements are WRF's own drivers' (woof/core/wrfsolar_eps.py,
every word held to compiled WRF v4.7.1), at the MPAS seats of the same
schemes:

=====================  =============================================  ==================================
switch                 MPAS seat                                       perturbed
=====================  =============================================  ==================================
config_pert_farms      the FARMS service (farms_v842)                  its copies of the surface albedo,
                                                                       qv, qc, qs (and the zero aerosol
                                                                       inputs, which stay zero)
config_pert_deng       the Deng service (shcu_deng_v842)               its copies of qv, qc and w (the
                                                                       scheme reads T, not theta)
config_pert_thom       the microphysics slot, mp_thompson and           the driver's qv, qc, qi, qs, ni
                       mp_thompson_aerosols (thompson_v842,            around the substeps, so what the
                       thompson_aero_v842)                             pair leaves reaches the state, as
                                                                       in WRF
config_pert_cld3       the cloudiness slot, cld_fraction_thompson      its copies of t and qv
                       (MPAS's cal_cldfra3)
=====================  =============================================  ==================================

On the FARMS, Deng and cloudiness seats the MPAS services read copies of
the prepared state, so the WRF round trip's residue (the Remove step's
rounding and clamps on WRF's state arrays) has no MPAS counterpart there.

The patterns are not WRF's.  WRF's generator is a doubly periodic FFT on
the (i, j) grid; on the sphere each STOCHPERT.TBL field is a stationary
AR(1) spherical-harmonic process (woof.hex.stochastic.SphericalPattern,
truncated at ``config_multi_perturb_degree``) with the table's decorrelation
time, its unit-variance draw scaled by the table's standard deviation and
clipped at its cutoff, constant in height (the table's random-phase
vertical structure is a property of WRF's FFT and is not reproduced), and
correlation lengths below the truncation's resolution (about 2*pi*R/degree)
are smoothed to it.  Updates follow WRF's cadence: the first step and every
``max(nint(spdt*60/dt), 1)`` steps, after the physics that precede the
dynamics.  Global meshes only (the harmonic basis refuses a regional
mesh's missing neighbours).

MAD-WRF (madwrf_opt, mp_physics = 96, madwrf_cldinit) is refused by name on
this route: woof's MPAS initialization (the Rust init engine) writes no qc,
qi or qs (woof/hex/hrrr_intermediate.py), so MAD-WRF's passively advected
hydrometeors and its tracers would start from zero and nudge toward zero,
and its cloud initialization is a real.exe step reading metgrid's
CLDMASK/BRTEMP/CLDTOPZ/CLDBASEZ, which no MPAS init door carries.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from woof.core import wrfsolar_eps as eps

GROUP = "wrfsolar_eps"
F = np.float32

#: option -> (RunConfig/WRF name, default)
OPTIONS: dict[str, tuple[str, Any]] = {
    "config_multi_perturb": ("multi_perturb", False),
    **{f"config_{name}": (name, False) for name in eps.SWITCHES},
    **{f"config_{name}": (name, 0.0) for switch in eps.SWITCHES for name in eps.FACTORS[switch]},
    "config_spdt": ("spdt", -1.0),
    "config_num_pert_3d": ("num_pert_3d", 15),
    "config_nens": ("nens", 1),
    "config_stochpert_tbl": ("stochpert_tbl", ""),
    "config_multi_perturb_degree": ("degree", 48),
}

#: MAD-WRF's keys, refused by name wherever an MPAS namelist carries them.
MADWRF_KEYS = ("madwrf_opt", "madwrf_cldinit", "madwrf_dt_relax", "madwrf_dt_nudge",
               "config_madwrf_opt", "config_madwrf_cldinit", "config_madwrf_dt_relax",
               "config_madwrf_dt_nudge")
MADWRF_REFUSAL = (
    "MAD-WRF is not ported to the MPAS route: woof's MPAS initialization (the "
    "Rust init engine) writes no qc, qi or qs, so MAD-WRF's passively advected "
    "hydrometeors (mp_physics = 96) and its TR_QC/TR_QI/TR_QS tracers would "
    "start from zero and the nudge would relax toward zero, and its cloud "
    "initialization (madwrf_cldinit) is a real.exe step that reads metgrid's "
    "CLDMASK/BRTEMP/CLDTOPZ/CLDBASEZ, which no MPAS init door carries. Run "
    "MAD-WRF on a WRF domain (woof/core/madwrf.py)")


@dataclass(frozen=True)
class EpsOptions:
    multi_perturb: bool = False
    values: Mapping[str, Any] = field(default_factory=dict)
    degree: int = 48

    def get(self, name):
        return self.values[name]

    @property
    def switches(self):
        return tuple(s for s in eps.SWITCHES if self.values.get(s))

    def factors(self, switch):
        if switch not in self.switches:
            return None
        return tuple(F(self.values[n]) for n in eps.FACTORS[switch])


def _logical(name, value):
    if isinstance(value, str) and value.strip().strip(".").casefold() in ("true", "t", "false", "f"):
        value = value.strip().strip(".").casefold() in ("true", "t")
    if not isinstance(value, bool):
        raise ValueError(f"{name} is logical (.true./.false.), not {value!r}")
    return value


def refuse_madwrf(groups: Mapping[str, Mapping[str, Any]]) -> None:
    for group, values in groups.items():
        for key in values:
            if str(key).casefold() in MADWRF_KEYS:
                raise ValueError(f"&{group} {key}: {MADWRF_REFUSAL}")


def resolve(groups: Mapping[str, Mapping[str, Any]]) -> EpsOptions | None:
    """``&wrfsolar_eps`` (None when absent or multi_perturb is off)."""
    refuse_madwrf(groups)
    supplied = {str(k).casefold(): v for k, v in groups.get(GROUP, {}).items()}
    if not supplied:
        return None
    unknown = sorted(set(supplied) - set(OPTIONS))
    if unknown:
        if any(k.startswith(("config_pert_mynn", "config_pert_noah")) for k in unknown):
            raise ValueError(f"&{GROUP} {unknown[0]}: WRF-Solar EPS's MYNN and Noah hooks "
                             "are not ported (woof ports FARMS, Deng, Thompson and icloud = 3)")
        raise ValueError(f"&{GROUP} has no option {unknown[0]!r}; its options are "
                         + ", ".join(OPTIONS))
    values = {}
    for option, (name, default) in OPTIONS.items():
        value = supplied.get(option, default)
        if isinstance(default, bool):
            value = _logical(option, value)
        elif isinstance(default, float):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{option} is a finite real, not {value!r}")
            value = float(value)
        elif isinstance(default, int):
            if type(value) is not int:
                raise ValueError(f"{option} is an integer, not {value!r}")
        elif not isinstance(value, str):
            raise ValueError(f"{option} names a STOCHPERT.TBL, not {value!r}")
        values[name] = value
    on = values.pop("multi_perturb")
    degree = values.pop("degree")
    if not on:
        switched = [s for s in eps.SWITCHES if values[s]]
        if switched:
            raise ValueError(f"&{GROUP} config_{switched[0]} without config_multi_perturb does "
                             "nothing (WRF's drivers test 'pert_<scheme> .and. multi_perturb == 1')")
        return None
    if not any(values[s] for s in eps.SWITCHES):
        raise ValueError(f"&{GROUP} config_multi_perturb=.true. with every config_pert_<scheme> off "
                         "perturbs nothing")
    if not 1 <= degree <= 512:
        raise ValueError("config_multi_perturb_degree must be 1..512")
    eps.load_stochpert_table(values["stochpert_tbl"], num_pert_3d=values["num_pert_3d"])
    return EpsOptions(True, dict(values), degree)


def check(config: Any) -> None:
    """Refuse a switch whose MPAS seat does not run in this configuration."""
    o = config.wrfsolar_eps
    services = getattr(config, "service_backends", {}) or {}
    sel = config.selectors
    if "pert_farms" in o.switches and str(services.get("farms_sw", "off")) == "off":
        raise ValueError("config_pert_farms perturbs FARMS's inputs: it needs the FARMS service "
                         "(&hex_mpasphys config_farms_sw_backend = 'hex-cuda')")
    if "pert_deng" in o.switches and str(services.get("shcu_deng", "off")) == "off":
        raise ValueError("config_pert_deng perturbs Deng's inputs: it needs the Deng service "
                         "(&hex_mpasphys config_shcu_deng_backend = 'hex-cuda')")
    if "pert_thom" in o.switches:
        if sel.get("microphysics") not in ("mp_thompson", "mp_thompson_aerosols"):
            raise ValueError("config_pert_thom perturbs Thompson's inputs (WRF mp_physics 8 and 28): "
                             "it needs config_microp_scheme = 'mp_thompson' or 'mp_thompson_aerosols'")
    if "pert_cld3" in o.switches and sel.get("cloudiness") != "cld_fraction_thompson":
        raise ValueError("config_pert_cld3 perturbs cal_cldfra3's t and qv: it needs "
                         "config_radt_cld_scheme = 'cld_fraction_thompson'")
    from .driver_v842 import Backend
    for slot, needed in (("microphysics", "pert_thom"), ("cloudiness", "pert_cld3")):
        if needed in o.switches and config.backends.get(slot) != Backend.HEX_CUDA:
            raise ValueError(f"config_{needed} runs inside the {slot} slot's GPU port; put the "
                             f"slot on hex-cuda")


class MpasMultiPerturb:
    """The STOCHPERT.TBL fields of one MPAS run, (nCellsSolve,) each."""

    def __init__(self, options: EpsOptions, mesh, *, dt: float, n_solve: int,
                 resource=None):
        import cupy as cp
        from woof.hex.stochastic import (MeshHarmonics, SphericalPattern,
                                         StochasticConfig)
        self.options = options
        self.n_solve = int(n_solve)
        self.stepsp = eps.stepsp(options.get("spdt"), dt)
        table = eps.load_stochpert_table(options.get("stochpert_tbl"),
                                         num_pert_3d=options.get("num_pert_3d"))
        names = set()
        for switch in options.switches:
            names.update(eps.CONSUMED[switch])
        self.names = tuple(n for n in eps.FIELDS if n in names)
        if resource is None:
            resource = MeshHarmonics.cached(mesh, options.degree)
        self.resource = resource
        self.rows, self.patterns, self.values = {}, {}, {}
        for name in self.names:
            row = table.get(name)
            self.values[name] = cp.zeros(self.n_solve, dtype=cp.float32)
            if row is None or eps.field_index(name) > options.get("num_pert_3d"):
                continue                       # WRF's never-updated zero field
            self.rows[name] = row
            seed = eps.field_seed(options.get("nens"), row.seed, name)
            self.patterns[name] = SphericalPattern(resource, StochasticConfig(
                seed=seed, degree=options.degree, length_m=float(row.lengthscale),
                time_s=float(row.timescale)))

    def due(self, itimestep: int) -> bool:
        return (itimestep % self.stepsp == 0) or itimestep == 1

    def update(self, itimestep: int, time_seconds: float) -> bool:
        if not self.due(int(itimestep)):
            return False
        import cupy as cp
        for name, pattern in self.patterns.items():
            row = self.rows[name]
            pattern.advance(float(time_seconds))
            # the cell rows of basis @ coefficients, first component (formed
            # here rather than read from the process's own buffer, which the
            # host facade's matmul does not write in place)
            unit = cp.matmul(self.resource.basis[:self.n_solve],
                             pattern.coefficients[:, :1]).reshape(-1)
            std, limit = F(row.std), F(F(row.cutoff) * F(row.std))
            self.values[name][...] = cp.clip(unit.astype(cp.float32) * std, -limit, limit)
        return True

    def level_patterns(self, columns: int | None = None) -> dict:
        """{name: Pattern} for [level, cell] operands of ``columns`` cells
        (nCellsSolve by default; halo cells beyond it read zero)."""
        import cupy as cp
        columns = self.n_solve if columns is None else int(columns)
        if columns == self.n_solve:
            return {name: eps.Pattern(v.reshape(1, 1, -1)) for name, v in self.values.items()}
        out = {}
        for name, v in self.values.items():
            padded = cp.zeros(columns, dtype=cp.float32)
            padded[:min(columns, self.n_solve)] = v[:columns]
            out[name] = eps.Pattern(padded.reshape(1, 1, -1))
        return out

    def column_grid(self, name, nz):
        """(nCellsSolve, nz) contiguous, for [cell, level] operands."""
        import cupy as cp
        return cp.ascontiguousarray(cp.broadcast_to(self.values[name][:, None],
                                                    (self.n_solve, nz)))

    def restart_state(self):
        import cupy as cp
        return {"patterns": {n: p.snapshot() for n, p in self.patterns.items()},
                "values": {n: cp.asnumpy(v) for n, v in self.values.items()}}

    def restore_restart_state(self, payload):
        import cupy as cp
        if set(payload.get("patterns", {})) != set(self.patterns) or \
                set(payload.get("values", {})) != set(self.values):
            raise ValueError("WRF-Solar EPS restart state differs")
        for name, p in self.patterns.items():
            p.restore(payload["patterns"][name])
        for name, v in self.values.items():
            a = np.asarray(payload["values"][name], dtype=np.float32)
            if a.shape != tuple(v.shape):
                raise ValueError(f"WRF-Solar EPS restart state layout differs at {name}")
            v[...] = cp.asarray(a)


def provider_of(session):
    return getattr(session, "wrfsolar_eps", None)


# -- seats ----------------------------------------------------------------------

def farms_inputs(session, *, albedo, aod, angexp, aerasy, qv, qc, qs):
    """pert_farms on the FARMS service's (nl, nc) and (nc,) copies; the
    inputs are rewritten in place (Add_; the copies are discarded after)."""
    p = provider_of(session)
    fac = None if p is None else p.options.factors("pert_farms")
    if fac is None:
        return
    pats = p.level_patterns()
    eps.farms_add(pats, fac, albedo=albedo, aod=aod, angexp=angexp, aerasy=aerasy,
                  qv=qv, qc=qc, qs=qs)


def deng_inputs(session, *, qv, qc, w):
    """pert_deng's Add_ on the Deng service's copies of QV, QC and W
    ([level, cell]; w on nl+1 levels).  TH is not a Deng input on MPAS."""
    p = provider_of(session)
    fac = None if p is None else p.options.factors("pert_deng")
    if fac is None:
        return
    import cupy as cp
    pats = p.level_patterns()
    dummy = cp.zeros_like(qv)
    eps.deng_add(pats, fac, th=dummy, qv=qv, qc=qc, w=w)


def thompson_pair(session):
    """(add, remove) callables for the microphysics driver's fields, or None."""
    p = provider_of(session)
    fac = None if p is None else p.options.factors("pert_thom")
    if fac is None:
        return None
    names = ("qv", "qc", "qi", "qs", "ni")

    def add(fields):
        pats = p.level_patterns(fields["qv"].shape[1])
        return eps.thompson_add(pats, fac, **{n: fields[n] for n in names})

    def remove(fields, saved):
        pats = p.level_patterns(fields["qv"].shape[1])
        eps.thompson_remove(pats, fac, saved, **{n: fields[n] for n in names})
    return add, remove


def cld3_perturb(session, profiles):
    """pert_cld3 on the cloudiness slot's [cell, level] t and qv copies."""
    p = provider_of(session)
    fac = None if p is None else p.options.factors("pert_cld3")
    if fac is None:
        return profiles
    nz = profiles["t"].shape[1]
    t, qv = eps.cld3_inputs(profiles["t"], profiles["qv"], p.column_grid("TH", nz),
                            p.column_grid("QVAPOR", nz), fac[0], fac[1])
    return {**profiles, "t": t, "qv": qv}
