"""WRF-Solar diagnostics (``solar_diagnostics = 1``), on the device.

WRF v4.7.1 phys/module_diag_solar.F (T. Juliano and P. Jimenez, NCAR/RAL),
called by module_diagnostics_driver.F:1181-1213 at the end of EVERY time
step.  The routine itself is woof/core/kernels/solar_diag.cu, graded word
for word against the compiled routine (tools/solar_diag_wrf471_oracle).

The 26 outputs are Registry/registry.solar_fields' 2-D ``h`` rows, named and
described as WRF names them (:data:`SOLAR_OUTPUTS`).  None is a restart
field (the Registry gives no ``r``): GHI_ACCUM, the one accumulator, sums
SWDNB*dt from the start of THIS run segment and a WRF restart begins it at
zero; woof does the same (the plane is not serialized).  It is never reset
by a history write.

Inputs, as WRF hands them over (see :func:`update`):

* the end-of-step moisture (qv/qc/qi/qs), geopotential (PH, PHB) and the
  microphysics' effective radii (woof keeps them in microns; WRF's
  RE_CLOUD/RE_ICE/RE_SNOW are metres -- :func:`radius_metres`);
* the physics density and dz8w of the step's own physics preparation (WRF
  computes them in phy_prep at the start of the step);
* the radiation driver's CLDFRA and its qc/qi after the cloud scheme
  (QC_TOT/QI_TOT) and COSZEN -- which WRF's radiation driver recomputes on
  EVERY step when solar_diagnostics = 1 (its Solar_step block,
  module_radiation_driver.F:1176-1597, and :3103-3135), not only on
  radiation steps; woof evaluates them each step for the diagnostics
  (:mod:`woof.core.solar_step`) but keeps the forecast's own CLDFRA and
  COSZEN on the radiation cadence, so switching the diagnostics on changes no
  forecast word (WRF's surface, PBL and cumulus drivers do see the per-step
  values: a documented difference);
* SWDNB/SWDNT, the last radiation call's boundary fluxes.
"""
from __future__ import annotations

import numpy as np

#: (scratch slot, WRF name, units, description) in solar_diag.cu's plane
#: order.  Registry/registry.solar_fields, verbatim.
SOLAR_OUTPUTS = (
    ("solar_clrnidx", "CLRNIDX", " ", "CLEARNESS INDEX"),
    ("solar_sza", "SZA", "deg", "SOLAR ZENITH ANGLE"),
    ("solar_ghi_accum", "GHI_ACCUM", "J m-2", "ACCUMULATED GHI"),
    ("solar_cldfrac2d", "CLDFRAC2D", "%", "2-D MAX CLOUD FRACTION"),
    ("solar_wvp", "WVP", "kg m-2", "WATER VAPOR PATH"),
    ("solar_lwp", "LWP", "kg m-2", "LIQUID CLOUD WATER PATH"),
    ("solar_iwp", "IWP", "kg m-2", "ICE CLOUD WATER PATH"),
    ("solar_swp", "SWP", "kg m-2", "SNOW CLOUD WATER PATH"),
    ("solar_wp_sum", "WP_SUM", "kg m-2", "SUM OF LWP+IWP+SWP"),
    ("solar_lwp_tot", "LWP_TOT", "kg m-2", "LIQUID CLOUD WATER PATH RESOLVED + UNRESOLVED"),
    ("solar_iwp_tot", "IWP_TOT", "kg m-2", "ICE CLOUD WATER PATH RESOLVED + UNRESOLVED"),
    ("solar_wp_tot_sum", "WP_TOT_SUM", "kg m-2", "SUM OF LWP+IWP+SWP RESOLVED + UNRESOLVED"),
    ("solar_re_qc", "RE_QC", "m", "MASS-WEIGHTED LIQUID CLOUD EFFECTIVE RADIUS"),
    ("solar_re_qi", "RE_QI", "m", "MASS-WEIGHTED ICE EFFECTIVE RADIUS"),
    ("solar_re_qs", "RE_QS", "m", "MASS-WEIGHTED SNOW EFFECTIVE RADIUS"),
    ("solar_re_qc_tot", "RE_QC_TOT", "m", "MASS-WEIGHTED LIQUID CLOUD EFFECTIVE RADIUS RESOLVED + UNRESOLVED"),
    ("solar_re_qi_tot", "RE_QI_TOT", "m", "MASS-WEIGHTED ICE EFFECTIVE RADIUS RESOLVED + UNRESOLVED"),
    ("solar_tau_qc", "TAU_QC", " ", "MASS-WEIGHTED LIQUID CLOUD OPTICAL THICKNESS"),
    ("solar_tau_qi", "TAU_QI", " ", "MASS-WEIGHTED ICE OPTICAL THICKNESS"),
    ("solar_tau_qs", "TAU_QS", " ", "MASS-WEIGHTED SNOW OPTICAL THICKNESS"),
    ("solar_tau_qc_tot", "TAU_QC_TOT", " ", "MASS-WEIGHTED LIQUID CLOUD OPTICAL THICKNESS RESOLVED + UNRESOLVED"),
    ("solar_tau_qi_tot", "TAU_QI_TOT", " ", "MASS-WEIGHTED ICE OPTICAL THICKNESS RESOLVED + UNRESOLVED"),
    ("solar_cbaseht", "CBASEHT", "m agl", "CLOUD BASE HEIGHT"),
    ("solar_ctopht", "CTOPHT", "m agl", "CLOUD TOP HEIGHT"),
    ("solar_cbaseht_tot", "CBASEHT_TOT", "m agl", "CLOUD BASE HEIGHT RESOLVED + UNRESOLVED"),
    ("solar_ctopht_tot", "CTOPHT_TOT", "m agl", "CLOUD TOP HEIGHT RESOLVED + UNRESOLVED"),
)
#: slot -> (NAME, units, description), the shape the history builders take.
HISTORY = {slot: (name, units, description)
           for slot, name, units, description in SOLAR_OUTPUTS}
NOUT = len(SOLAR_OUTPUTS)

FLAG_QV, FLAG_QC, FLAG_QI, FLAG_QS = 1, 2, 4, 8
FLAG_REQC, FLAG_REQI, FLAG_REQS = 16, 32, 64


def species_flags(*, qv, qc, qi, qs, has_reqc, has_reqi, has_reqs):
    """solar_diag.cu's ``flags`` word from presence and the has_req* trio."""
    return (FLAG_QV * bool(qv) | FLAG_QC * bool(qc) | FLAG_QI * bool(qi)
            | FLAG_QS * bool(qs) | FLAG_REQC * bool(has_reqc)
            | FLAG_REQI * bool(has_reqi) | FLAG_REQS * bool(has_reqs))


def radius_metres(microns):
    """WRF's RE_* (metres) from woof's micron radii: ``microns * 1e-6``.

    woof's microphysics publishes ``fl(re_m * 1e6)``, the RRTMG wrapper's
    micron word; WRF's grid holds ``re_m``.  The round trip is not always
    the identity (a float32 can sit between two images of the multiply), so
    a radius-weighted output can differ from WRF's by the ULP this conversion
    moves; the routine's arithmetic itself is graded word for word.
    """
    import cupy as cp
    return cp.multiply(microns, np.float32(1.0e-6), dtype=cp.float32)


def run_solar_diag(*, nz, ncol, dt, flags, rho, dz8w, ph, phb, cldfra, coszen,
                   swdnb, swdnt, qv, qc, qi, qs, qc_tot, qi_tot, re_cloud,
                   re_ice, re_snow, out, work):
    """Launch solar_diag.cu over ``ncol`` columns.

    Mass fields are ``(nz, ncol)`` float32 (column fastest), ``ph``/``phb``
    ``(nz+1, ncol)``, the surface fields ``(ncol,)``.  ``out`` is the
    ``(NOUT, ncol)`` output stack in :data:`SOLAR_OUTPUTS` order; its
    GHI_ACCUM plane is read and advanced, and an output WRF does not assign
    keeps its word.  ``work`` is ``(nz, ncol)`` float32 scratch.
    """
    import cupy as cp
    from woof.core.kernels import get_kernel
    if ncol == 0:
        return out
    if nz < 2:
        raise ValueError("solar_diag integrates kts..kte-1 and needs nz >= 2")
    args = []
    for name, a, shape in (
            ("rho", rho, (nz, ncol)), ("dz8w", dz8w, (nz, ncol)),
            ("ph", ph, (nz + 1, ncol)), ("phb", phb, (nz + 1, ncol)),
            ("cldfra", cldfra, (nz, ncol)), ("coszen", coszen, (ncol,)),
            ("swdnb", swdnb, (ncol,)), ("swdnt", swdnt, (ncol,)),
            ("qv", qv, (nz, ncol)), ("qc", qc, (nz, ncol)),
            ("qi", qi, (nz, ncol)), ("qs", qs, (nz, ncol)),
            ("qc_tot", qc_tot, (nz, ncol)), ("qi_tot", qi_tot, (nz, ncol)),
            ("re_cloud", re_cloud, (nz, ncol)), ("re_ice", re_ice, (nz, ncol)),
            ("re_snow", re_snow, (nz, ncol))):
        if (a.dtype != cp.float32 or a.size != int(np.prod(shape))
                or not a.flags.c_contiguous):
            raise ValueError(f"solar_diag: {name} must be contiguous float32 {shape}")
        args.append(a)
    if out.dtype != cp.float32 or out.size != NOUT * ncol or not out.flags.c_contiguous:
        raise ValueError("solar_diag: out must be contiguous float32 (NOUT, ncol)")
    if work.dtype != cp.float32 or work.size < nz * ncol:
        raise ValueError("solar_diag: work must hold nz*ncol float32")
    block = 128
    get_kernel("solar_diag", "solar_diag_columns")(
        ((ncol + block - 1) // block,), (block,),
        (np.int32(ncol), np.int32(nz), np.float32(dt), np.int32(flags),
         *args, work, out))
    return out
