"""CCPP TKE-EDMF reaches a hex forecast (hex parity 1.9), host half.

The device half is tests/test_hex_tke_edmf_gpu.py.  Here: the selection
resolves to an engine species row that appends ``tke_edmf`` to the declared
microphysics row, the forecast's init reader starts that scalar from the
engine allocator's value rather than refusing the init as partial, and the
hex column has a vertical-bound row for the scheme.
"""

from __future__ import annotations

import numpy as np
import pytest

from woof.hex import physics_suites
from woof.hex.vertical_spec import hex_column_level_terms


def test_the_selection_appends_tke_to_the_declared_row():
    from woof.hex.cuda_suite_physics_v841 import microphysics_for
    from woof.hex.species_row import species_row_for_scheme

    resolved = physics_suites.resolve_physics(
        "arwen_mpas_column", {"bl_pbl_physics": 901})
    scheme = microphysics_for(resolved.backend_row)
    assert scheme == "mp_wsm6@6"
    row = species_row_for_scheme(scheme)
    declared = species_row_for_scheme("mp_wsm6")
    assert tuple(row.names()) == tuple(declared.names()) + ("tke_edmf",)
    assert row.n_mass_loading == declared.n_mass_loading


def test_init_without_tke_starts_it_from_the_engine_value():
    from woof.hex.drivers import run_cuda_v841_forecast as forecast

    names = ("qv", "qc", "qr", "qi", "qs", "qg", "tke_edmf")
    variables = {name: None for name in names[:-1]}
    assert forecast.select_initial_scalar_names(variables, names) == names[:-1]
    assert forecast._pbl_owned_initial_values()["tke_edmf"] == 0.2
    # A row partial in its microphysics species is still refused.
    with pytest.raises(ValueError, match="partial"):
        forecast.select_initial_scalar_names(
            {name: None for name in names[:4]}, names)


def test_the_hex_column_has_a_vertical_bound_row():
    terms = hex_column_level_terms(microphysics="mp_wsm6", pbl="ccpp-tke-edmf")
    # SATMEDMF declares a floor of eight levels and no ceiling.
    assert not any("TKE-EDMF" in label for label, _ in terms)
    from woof.physics_vertical_contract import SATMEDMF_VERTICAL_LEVEL_BOUNDS
    assert SATMEDMF_VERTICAL_LEVEL_BOUNDS == (8, None)
    assert np.isfinite(min(maximum for _, maximum in terms))
