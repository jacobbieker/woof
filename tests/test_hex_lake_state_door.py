"""The hex door refuses an explicit lake_state_source that no lake mask reads."""
from types import SimpleNamespace

import pytest

from woof.core.mpas_column_suite import MPAS_DEFAULT_PHYSICS
from woof.hex import forecast_door


def _resolve(physics, collected):
    return forecast_door.resolve_surface_statics(
        SimpleNamespace(surface_statics=None, grid=None),
        SimpleNamespace(column_profile={"physics": physics}), "mesh", None, collected)


@pytest.mark.parametrize("source", ["era5", "auto+era5", "glsea"])
def test_an_explicit_source_without_the_lake_model_is_refused(source):
    physics = dict(MPAS_DEFAULT_PHYSICS, lake_state_source=source)
    collected = []
    _resolve(physics, collected)
    assert any("accepted and never read" in p and source in p for p in collected)
    with pytest.raises(forecast_door.ForecastDoorRefusal, match="accepted and never read"):
        _resolve(physics, None)


@pytest.mark.parametrize("source", ["auto", "forcing"])
def test_tolerant_sources_pass_without_the_lake_model(source):
    collected = []
    assert _resolve(dict(MPAS_DEFAULT_PHYSICS, lake_state_source=source), collected) is None
    assert collected == []
