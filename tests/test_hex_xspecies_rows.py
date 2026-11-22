"""Extra-species microphysics rows on hex (part 3): the hour-0 reflectivity view
of a regional run's initial state, and CAM-MG's signed precipitation buckets.

NSSL's operator forms rho from WRF's ``alt`` (woof.core.state_reflectivity), so
the view carries ``alt = 1/rho_dry`` when the seam gives it the density; NTU's
field table names ``w``, so the view carries the mass-level w too.  A
two-moment init whose number moments are zero under analysis masses (Morrison's
ns/ng after --augment-init) gives a non-finite operator result; the hour-0
frame then takes the allocated default with a reason naming the moments.
WRF's CAM-MG adds its step precipitation unclipped, so its buckets are signed.
"""
from __future__ import annotations

import numpy as np

from woof.hex.cuda_arwen_physics_v841 import InitialReflectivityViewV841, nonfinite_initial_reflectivity

NZ, NCOL = 3, 4


def _view(**kwargs):
    theta = np.full((NZ, NCOL), 300.0, np.float32)
    pressure = np.full((NZ, NCOL), 9.0e4, np.float32)
    return InitialReflectivityViewV841(theta=theta, pressure=pressure, species={"qr": np.zeros((NZ, NCOL))},
                                       melting=False, cp=np, **kwargs)


def test_the_view_carries_alt_from_the_density_and_w():
    rho = np.linspace(0.6, 1.2, NZ * NCOL, dtype=np.float32).reshape(NZ, NCOL)
    view = _view(density=rho)
    assert view.alt.shape == (NZ, 1, NCOL)
    assert np.array_equal(view.alt.reshape(NZ, NCOL), np.float32(1.0) / rho)
    assert not hasattr(_view(), "alt")
    w = np.arange((NZ + 1) * NCOL, dtype=np.float32).reshape(NZ + 1, NCOL)
    assert np.array_equal(_view(w=w[:-1]).w.reshape(NZ, NCOL), w[:-1])


def test_a_finite_reflectivity_is_written_and_a_nonfinite_one_names_the_moments():
    qs = np.zeros((NZ, NCOL), np.float32)
    qs[1, :3] = 1.0e-4
    species = {"qs": qs, "ns": np.zeros((NZ, NCOL), np.float32),
               "qr": np.zeros((NZ, NCOL), np.float32), "nr": np.ones((NZ, NCOL), np.float32)}
    finite = np.full((NZ, NCOL), -35.0, np.float32)
    assert nonfinite_initial_reflectivity(finite, species, np) is None
    bad = finite.copy()
    bad[1, :2] = np.nan
    reason = nonfinite_initial_reflectivity(bad, species, np)
    assert "non-finite at 2 points" in reason
    assert "ns=0 under qs>1e-9 at 3 points" in reason
    assert "nr=0" not in reason


def test_cam_mg_buckets_are_signed_as_wrf_adds_them():
    # WRF 4.7.1 module_mp_cammgmp_driver.F:1385-1390 adds RAINNCV/SNOWNCV unclipped.
    from woof.hex.species_row import engine_species_row

    def row(mp):
        return engine_species_row(scheme_name=f"t{mp}", mp_physics=mp, species=("qv", "qc", "qr", "qi", "qs"),
                                  mass_species=("qv", "qc", "qr", "qi", "qs"))
    assert {i.name for i in row(11).surface_accumulators if i.signed_fallout} == {"rainnc", "snownc"}
    assert not any(i.signed_fallout for i in row(6).surface_accumulators)
