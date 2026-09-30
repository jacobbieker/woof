"""Vector spherical-harmonic analysis through vorticity/divergence."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .transform import FourierWaist, SphericalHarmonicTransform


@dataclass
class VorticityDivergenceOperator:
    transform: SphericalHarmonicTransform

    def __post_init__(self) -> None:
        # State-independent analysis weights, built once at construction:
        # the retired per-order loop rebuilt them for every m of every call.
        # The 1j*m rotational diagonal and the quadrature/metric factors
        # fold into two small per-(latitude, order) tables applied to the
        # Fourier coefficients before the batched Legendre contraction.
        t = self.transform
        b = t.backend
        scale = 2.0 * math.pi / t.grid.radius_m
        m = np.arange(t.grid.truncation + 1, dtype=np.float64)
        self._rotation_weight = b.asarray(
            (1j * m)[None, :]
            * (t.grid.quadrature_weights / t.grid.cos_lat * scale)[:, None],
            dtype=b.complex_dtype,
        )
        self._shear_weight = b.asarray(
            (t.grid.quadrature_weights * scale)[:, None],
            dtype=b.float_dtype,
        )

    def wind_from_vordiv(self, vorticity, divergence):
        """Invert relative vorticity/divergence to eastward/northward wind.

        The scalar streamfunction and velocity potential satisfy
        ``laplacian(psi)=vorticity`` and ``laplacian(chi)=divergence``.
        Their analytic spherical-harmonic derivatives then produce the wind.
        """
        xp = self.transform.backend.xp
        psi = self.transform.inverse_laplacian(vorticity)
        chi = self.transform.inverse_laplacian(divergence)
        # One gradient over the stacked pair: the synthesis contraction
        # inside runs per call, so two separate gradients pay it twice for
        # the same arithmetic.
        d_east, d_north = self.transform.gradient(xp.stack([psi, chi]))
        u = -d_north[0] + d_east[1]
        v = d_east[0] + d_north[1]
        return u, v

    def vordiv_from_wind(self, u, v):
        """Direct vector spherical-harmonic analysis of a grid wind.

        This uses integration by parts rather than differentiating projected
        scalar U/V components.  The latter is tempting but is not the vector
        transform: it loses several percent even for low-degree wind fields.
        For ``Y_nm=P_nm(phi) exp(i m lambda)`` the coefficients are

        ``D_nm = 1/a ∫ [i m u P/cos(phi) - v dP/dphi] e^-imλ dΩ``
        ``Z_nm = 1/a ∫ [i m v P/cos(phi) + u dP/dphi] e^-imλ dΩ``.
        """
        t = self.transform
        xp = t.backend.xp
        u = xp.asarray(u, dtype=t.backend.float_dtype)
        v = xp.asarray(v, dtype=t.backend.float_dtype)
        t._validate_grid(u)
        t._validate_grid(v)
        if u.shape != v.shape:
            raise ValueError(f"u/v shape mismatch: {u.shape} != {v.shape}")
        return self._vordiv_from_fourier(self._wind_fourier(xp.stack([u, v])))

    def vordiv_from_wind_waist(self, waist: FourierWaist):
        """:meth:`vordiv_from_wind` from a waist a band pipeline filled.

        The two weight tables are ``(nlat, T+1)`` and both contractions
        reduce the latitude axis, so the waist is read WHOLE however many
        bands wrote it and the coefficients are the resident call's.
        """
        return self._vordiv_from_fourier(waist)

    def _wind_fourier(self, pair, *, bands: int | None = None) -> FourierWaist:
        """The Fourier waist of a stacked ``(2, ..., nlat, nlon)`` wind pair.

        The first half of :meth:`vordiv_from_wind`, split out so a caller
        that owns the pair can release it before the Legendre contractions
        run (the dycore's scalar advection: a 2.29 GiB pair per chunk at
        T533 float32).  Real wind components: the real FFT's m = 0..nlon/2
        output covers every retained order (nlon >= 2*truncation+2) at
        half the work of the complex transform it replaces.  u and v are
        stacked so each of the two Legendre contractions is one batched
        GEMM for the pair.

        This split predates the design that named it: the seam it cuts is
        exactly the transform's waist, so it is that object now rather
        than a second, private, full-width zonal spectrum.  The retained
        orders are cut here, where :meth:`_vordiv_from_fourier` used to
        cut them, so the waist a banded fill writes is the operand the
        weights read.  Cutting them here makes the ONE-BAND call cheaper
        rather than neutral.  What this method returned before was the
        full-width ``(2, ..., nlat, nlon//2+1)`` quotient, and it stayed
        alive through both contractions because the cut downstream was a
        view of it; what it returns now is the ``(2, ..., nlat, T+1)``
        waist, two thirds of that width (1.643 against 2.467 GB for a
        six-field forty-level chunk at T533 float32), and the full-width
        rfft output dies with the divide instead of outliving it.  At B
        bands the full-width buffer never exists at full size at all.
        """
        return self.transform.fourier_waist(pair, bands=bands)

    def _vordiv_from_fourier(self, fourier):
        """The second half of :meth:`vordiv_from_wind`: the contractions.

        Takes the waist :meth:`_wind_fourier` fills, or a raw zonal
        spectrum wide enough to cut, and reads it WHOLE IN LATITUDE:
        both weight tables are ``(nlat, T+1)`` and both contractions
        reduce the latitude axis, so nothing here bands and nothing here
        moves when the fill did.
        """
        t = self.transform
        raw = fourier.values if isinstance(fourier, FourierWaist) else fourier
        fm = raw[..., : t.grid.truncation + 1]
        with t._contraction_scope():
            rotational = t._batched_contract(
                fm * self._rotation_weight, t._basis, transposed=True
            )
            sheared = t._batched_contract(
                fm * self._shear_weight, t._derivative_basis, transposed=True
            )
        divergence = rotational[0] - sheared[1]
        zeta = rotational[1] + sheared[0]
        return t.project(zeta), t.project(divergence)

    def flux_curl_divergence(self, eastward_flux, northward_flux):
        return self.vordiv_from_wind(eastward_flux, northward_flux)
