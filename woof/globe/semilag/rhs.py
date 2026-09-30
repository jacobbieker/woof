"""The advective-form right-hand side, and the linear operator on the grid.

The shipped Eulerian core evaluates its tendencies in FLUX form: the
momentum in the vector-invariant form whose divergence and curl the
spectral analysis takes directly, and theta and vapour as the divergence
of a mass flux.  A semi-Lagrangian step needs neither.  It needs the
material tendency at a point, what a parcel's wind, temperature and
surface pressure do while it travels, and it needs the semi-implicit
linear operator in the same place, so the two can be subtracted.

The identity that turns one into the other is the vector-invariant one::

    (u/(a cos phi)) du/dlam + (v/a) du/dphi = dK/dx - zeta v + u v tan(phi)/a

so the whole curvature term cancels against the vector-invariant form's
own and what is left is the classical advective momentum equation::

    Du/Dt = (f + u tan(phi)/a) v - dPhi/dx - R Tv factor_k dlnps/dx
    Dv/Dt = -(f + u tan(phi)/a) u - dPhi/dy - R Tv factor_k dlnps/dy

with ``factor_k`` the exact full-level ``grad(ln p_k)`` per unit
``grad(ln ps)`` the shipped core already builds (the DN-3 pairing with
the half-layer hydrostatic integration, which is what makes those two
terms balance pointwise at rest).

The ``tan(phi)`` metric terms do NOT appear below.  They are exactly the
terms that vanish when the momentum is carried as three geocentric
Cartesian components of a tangent field: writing ``V = u e_lam + v
e_phi`` and differentiating the basis along the trajectory,

    D V_c/Dt = [Du/Dt - uv tan(phi)/a] e_lam
             + [Dv/Dt + u^2 tan(phi)/a] e_phi - (|V|^2/a) r

and the two bracketed metric terms cancel against the ones above.  The
remaining radial term is the parallel transport that
:mod:`woof.globe.semilag.vectors` integrates exactly as a
rotation.  So this module returns the Coriolis and force terms alone and
the geometry is carried by the trajectory, rather than by an arithmetic
term that would have to be right at a pole where the basis it is written
in turns through 2*pi in 326 m.

The thermodynamic variable has NO explicit material tendency here.
Adiabatic flow conserves theta along a parcel, so the step gathers the
whole variable and its nonlinear residual is ``-L theta``; the semi-implicit
operator's thermodynamic row (the reference profile's vertical advection
by the linearized mass flux) stays implicit at the arrival point and
explicit at the departure point exactly as it does for every other row,
and the linear stability analysis of the step is unchanged, because the
reference change a trajectory reads between its departure and arrival
levels is the SETTLS trajectory average of that same term to second order.

Until 2026-09-06 the core gathered ``theta' = theta - theta_ref(k)`` and
carried ``-D theta_ref/Dt`` as a grid tendency in the operator's own flux
form (the reference interface values, the run's mass flux).  That form is
consistent with the gather in the interior and inconsistent at the two
boundary levels, where the flux form is one-sided and the trajectory is
CLAMPED: under descent at the lid the parcel arriving at level 0 departs
from level 0 (nothing lies above it), the gather reads no reference
change, and the tendency warmed it anyway by ``-omega_{1/2} (face_1 -
theta_ref_0) / dp_0`` per unit time, where the Eulerian core's upwind
flux carries the layer's own value out of it and warms nothing.  MEASURED
2026-09-06 on the Eulerian arm's own 24 h T255 state (the 16 GB host): one 300 s
step of the retired form changes the level-0 potential temperature by
+0.1127 K in the global mean (+0.206 K over the descending half of the
globe, +0.016 K over the ascending half), which is +32.5 K per forecast
day against the +33.6 K the semi-Lagrangian day of record read above the
Eulerian one at level 0, zonally uniform to 1.7 K and growing linearly
through the day (+19.2 K at 12 h, +26.6 K at 18 h); the same form at level
1 is 29 percent too steep for any displacement because the top layer's
zero-gradient reconstruction pulls the first interface value toward the
lid.  Reading the reference along the trajectory instead gives -0.043 K
per step at level 0 on the same state, the sign and half the size of the
Eulerian upwind flux form's -0.090 K.  That is the whole of the 4.7 K
lid disagreement the observation grade could not see, and it is why the
variable is gathered whole.  :func:`reference_theta_faces` stays because
the gate that rebuilds the operator's gamma matrix reads it.
"""
from __future__ import annotations

import numpy as np


def reference_theta_faces(operator) -> np.ndarray:
    """The interface reference potential temperatures the vertical-mode
    operator built, ``(nlev+1,)``, zero at both boundaries.

    Recomputed from the operator's own ``theta_ref`` and pressures by the
    same limited reconstruction ``VerticalStructureOperator._build`` uses,
    rather than reaching into that method for a local: the operator is
    built once per (stack, reference) pair and cached, so this runs once
    per run, and a reconstruction that drifted from ``_build``'s would be
    caught by the unit test that rebuilds ``gamma`` from this array.
    """
    theta_ref = np.asarray(operator.theta_ref, dtype=np.float64)
    p_full = np.asarray(operator.p_full, dtype=np.float64)
    p_half = np.asarray(operator.p_half, dtype=np.float64)
    nlev = int(theta_ref.shape[0])
    gradient = np.zeros(nlev)
    above = (theta_ref[1:-1] - theta_ref[:-2]) / (p_full[1:-1] - p_full[:-2])
    below = (theta_ref[2:] - theta_ref[1:-1]) / (p_full[2:] - p_full[1:-1])
    product = above * below
    monotone = product > 0.0
    gradient[1:-1] = np.where(
        monotone, 2.0 * product / np.where(monotone, above + below, 1.0), 0.0
    )
    lower = np.minimum(theta_ref[:-1], theta_ref[1:])
    upper = np.maximum(theta_ref[:-1], theta_ref[1:])
    face_above = theta_ref[:-1] + gradient[:-1] * (p_half[1:nlev] - p_full[:-1])
    face_below = theta_ref[1:] + gradient[1:] * (p_half[1:nlev] - p_full[1:])
    face_above = np.minimum(np.maximum(face_above, lower), upper)
    face_below = np.minimum(np.maximum(face_below, lower), upper)
    face = np.zeros(nlev + 1)
    face[1:nlev] = 0.5 * (face_above + face_below)
    return face


class ReferenceProfile:
    """The reference column the semi-implicit operator linearizes about,
    built once per model: the operator itself, and its profile and
    interface values on the host for the gates that read them."""

    def __init__(self, model):
        operator = model.semi_implicit.operator(model.vertical)
        self.operator = operator
        self.theta_ref_host = np.asarray(operator.theta_ref, dtype=np.float64)
        self.face_host = reference_theta_faces(operator)


def linear_grid_tendencies(model, state, reference: ReferenceProfile):
    """The semi-implicit operator ``w L y`` where the trajectory needs it.

    Returns ``(phi_l_spectral, l_theta, l_lnps)``: the linearized
    geopotential-plus-gas-constant combination ``B x`` in the spectral
    basis (the momentum rows are its negated gradient, which the caller
    takes together with the geopotential's so both ride one contraction),
    and the thermodynamic rows ``C D`` on the grid.
    """
    transform = model.transform
    backend = transform.backend
    xp = backend.xp
    operator = reference.operator
    b_matrix, c_matrix, _structure = operator.device_matrices(backend)
    weight = backend.float_dtype(model.semi_implicit.divergence_weight)
    x = xp.concatenate(
        [state.theta, state.log_surface_pressure[None]], axis=0
    )
    phi_l = weight * xp.einsum("ij,jnm->inm", b_matrix, x)
    del x
    x_t = weight * xp.einsum("ij,jnm->inm", c_matrix, state.divergence)
    grid = transform.inverse(transform.project(x_t))
    del x_t
    return transform.project(phi_l), grid[:-1], grid[-1]


def advective_tendencies(
    model, state, g, omega_half, ps_t, reference: ReferenceProfile,
):
    """The material tendencies and the linear operator, both on the grid.

    Returns a dict with ``a_u``, ``a_v``, ``a_lnps`` (the advective-form
    tendencies of the eastward and northward wind and of ``ln ps``; theta
    has none, see the module docstring) and ``l_u``, ``l_v``, ``l_theta``,
    ``l_lnps`` (the rows of ``w L y``).  They are returned apart rather
    than already differenced because the departure-side bundle needs ``L``
    on its own.
    """
    transform = model.transform
    xp = transform.backend.xp
    phi_l_spectral, l_theta, l_lnps = linear_grid_tendencies(
        model, state, reference
    )
    geopotential_spectral = transform.project(
        transform.forward(g["geopotential"])
    )
    east, north = transform.gradient(
        xp.stack([geopotential_spectral, phi_l_spectral])
    )
    del geopotential_spectral, phi_l_spectral
    l_u = -east[1]
    l_v = -north[1]
    grad_phi_east = east[0]
    grad_phi_north = north[0]

    grad_lnps_east, grad_lnps_north = transform.gradient(
        state.log_surface_pressure
    )
    factor = model._pressure_gradient_factor(g["ps"], g["p_half"])
    force = model.gas_constant * g["virtual_temperature"] * factor
    del factor
    a_u = model.coriolis * g["v"] - grad_phi_east - force * grad_lnps_east
    a_v = -model.coriolis * g["u"] - grad_phi_north - force * grad_lnps_north
    del force, grad_phi_east, grad_phi_north, east, north

    # ln ps rides the LOWEST model level's trajectory, so its material
    # tendency is the Eulerian one plus the advection by that level's
    # wind.  The Eulerian part is the continuity closure the shipped core
    # already diagnoses, so both arms of the model see one surface
    # pressure tendency rather than two discretizations of it.
    a_lnps = (
        ps_t / g["ps"]
        + g["u"][-1] * grad_lnps_east
        + g["v"][-1] * grad_lnps_north
    )
    del grad_lnps_east, grad_lnps_north

    # ``Dtheta/Dt`` is zero for adiabatic flow and the whole variable is
    # gathered (module docstring), so there is no ``a_theta`` row: the
    # step's nonlinear residual for theta is ``-l_theta``.
    return {
        "a_u": a_u, "a_v": a_v, "a_lnps": a_lnps,
        "l_u": l_u, "l_v": l_v, "l_theta": l_theta, "l_lnps": l_lnps,
    }


__all__ = [
    "ReferenceProfile",
    "advective_tendencies",
    "linear_grid_tendencies",
    "reference_theta_faces",
]
