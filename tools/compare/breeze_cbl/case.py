"""The FastEddy dry convective boundary layer, as one shared definition.

This module is the single source of the case parameters the woof driver
(``run_woof.py``) uses and the Breeze driver (``run_breeze.jl``) mirrors.
The Julia script cannot import it, so each constant it duplicates is
marked there with ``# case.py:<NAME>``; ``analyze.py`` cross-checks the
values both drivers record in their receipts (``check_matched``).

Pinned case (Sauer & Munoz-Esparza 2020, JAMES 12, e2020MS002100,
doi:10.1029/2020MS002100, section 4.2; the same parameters as Breeze
v0.12.0 ``benchmarking/src/convective_boundary_layer.jl`` and FastEddy
``tutorials/examples/Example02_CBL.in`` at NCAR/FastEddy-model e0cd2f3e):

* 12 x 12 x 3 km, doubly periodic, flat, rigid lid, no damping layer
  (FastEddy's tutorial has a 400 m Rayleigh layer; Breeze's benchmark
  and this harness have none);
* geostrophic wind (Ug, Vg) = (9, 0) m/s, f-plane at 33.5 N;
* theta = 309 K to 600 m, +4 K/km above; surface pressure 101325 Pa;
* kinematic surface heat flux 0.35 K m/s;
* initial theta perturbation uniform in [-0.25, +0.25] K below 400 m;
* dt = 0.5 s (Breeze's benchmark step).

The one forced difference is the surface momentum flux.  Breeze's
benchmark imposes a constant friction velocity (u* = 0.4 m/s along the
first-level wind); FastEddy runs Monin-Obukhov with z0 = 0.05 m; woof's
LES lower boundary is WRF's ``isfflx = 0`` (a constant bulk drag
coefficient, stress = Cd |U| U).  The harness therefore uses a bulk
drag on BOTH models (``drag = "bulk"``): Cd is the neutral log law for
FastEddy's z0 at the first cell centre, ``Cd = (kappa / ln(z1/z0))^2``,
which gives u* ~ 0.4 m/s for the ~6 m/s near-surface wind of this case.
The Breeze driver keeps the stock constant-u* form selectable
(``--drag ustar``) so its published benchmark case is reproducible.
"""
from __future__ import annotations

import math

import numpy as np

LX = 12_000.0          # m
LY = 12_000.0          # m
LZ = 3_000.0           # m
UG = 9.0               # m/s
VG = 0.0               # m/s
LATITUDE = 33.5        # deg N
OMEGA = 7.2921e-5      # 1/s (the value Breeze's case uses)
THETA_SURFACE = 309.0  # K
Z_INVERSION = 600.0    # m
LAPSE = 0.004          # K/m above Z_INVERSION
P_SURFACE = 101_325.0  # Pa
HEAT_FLUX = 0.35       # K m/s (kinematic)
PERT_AMPLITUDE = 0.25  # K, uniform in [-A, +A]
PERT_DEPTH = 400.0     # m
DT = 0.5               # s
Z0 = 0.05              # m (FastEddy surflayer_z0)
KARMAN = 0.4
USTAR_BREEZE = 0.4     # m/s, Breeze benchmark's constant u*
#: Smagorinsky constant: FastEddy c_s = 0.18, also em_les's value.
C_S = 0.18
#: Prognostic-TKE constant: FastEddy / em_les c_k = 0.10.
C_K = 0.10
#: Turbulent Prandtl number of WRF's km_opt 2/3 (module_diffusion_em.F
#: ``prandtl = 1./3.``); the Breeze Smagorinsky variant uses the same.
PRANDTL = 1.0 / 3.0


def coriolis_f(latitude: float = LATITUDE) -> float:
    """f = 2 Omega sin(latitude), ~8.05e-5 1/s at 33.5 N."""
    return 2.0 * OMEGA * math.sin(math.radians(latitude))


def theta_profile(z):
    """Initial / reference potential temperature (K) at height ``z`` (m)."""
    z = np.asarray(z, dtype=float)
    return THETA_SURFACE + LAPSE * np.maximum(z - Z_INVERSION, 0.0)


def drag_coefficient(nz: int, lz: float = LZ, z0: float = Z0) -> float:
    """Neutral log-law bulk drag at the first cell centre z1 = dz/2."""
    z1 = 0.5 * lz / nz
    return (KARMAN / math.log(z1 / z0)) ** 2


def grid_spacing(nx: int, ny: int, nz: int) -> tuple[float, float, float]:
    return LX / nx, LY / ny, LZ / nz


def bulk_model_zi(t_seconds, entrainment_a: float = 0.2,
                  h0: float = Z_INVERSION, gamma: float = LAPSE,
                  q: float = HEAT_FLUX):
    """Zero-order-jump mixed-layer depth with flux-ratio entrainment.

    Tennekes (1973) / Tennekes & Driedonks: with entrainment flux
    -A*Q at the inversion and a lapse gamma above it, the equilibrium
    jump solution grows as ``h^2 = h0^2 + 2 (1 + 2A) Q t / gamma``.
    ``entrainment_a = 0`` is pure encroachment (the lower bound);
    A = 0.2 is the canonical shear-free value.  This is THEORY, not a
    model run: it is the reference the evolution plot is drawn against.
    """
    t = np.asarray(t_seconds, dtype=float)
    return np.sqrt(h0 * h0 + 2.0 * (1.0 + 2.0 * entrainment_a) * q * t / gamma)


#: Values READ OFF the FastEddy tutorial figures (NCAR/FastEddy-model
#: e0cd2f3e, docs/Tutorials/images/{MEAN,TURB}-PROF-convective.png: 20 m
#: grid, 4 h, turbulence statistics over 3-4 h).  Eyeball readings of a
#: plotted curve, accurate to roughly the figure's line width; they are
#: an ESTIMATE of the reference, not data.  The FastEddy paper's own
#: WRF-LES comparison figure could not be retrieved (publisher 403), so
#: no WRF numbers are transcribed.
FASTEDDY_FIGURE_4H = {
    "source": ("NCAR/FastEddy-model@e0cd2f3e docs/Tutorials/images/"
               "MEAN-PROF-convective.png, TURB-PROF-convective.png"),
    "time_hours": 4.0,
    "zi_heat_flux_min_m": 1800.0,
    "theta_mixed_layer_K": 312.3,
    "ww_max_m2s2": 2.85,
    "ww_max_height_m": 800.0,
    "u_mixed_layer_ms": 8.0,
}
