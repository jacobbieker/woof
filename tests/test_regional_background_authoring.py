"""Regional authoring uses exact source windows and preparation owners."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
from dataclasses import replace
import tomllib

import pytest

from conftest import requires_cupy
from woof.local_da import Card, Request, configuration, derive_rung, build_plan, publish


def _request(**values):
    return Request(epoch='2026-09-10T14:00:00Z', point=(40., -100.), card=Card(24.), **values)


def test_nonzero_source_lead_authors_fetch_duration_and_terminal_bracket():
    request = _request()
    rung = derive_rung(request, 1)
    text, _, _, background = configuration(request, rung)
    hints = tomllib.loads(text)['fetch']
    start = hints.get('forecast_start_hour', 0)
    duration = rung['cycles'] * rung['cadence_seconds'] + rung['forecast_seconds']
    assert start == 2
    assert hints['hours'] == 1
    assert start + hints['hours'] == 3
    assert background['selection']['forecast_leads'] == [2, 3]
    assert datetime.fromisoformat(background['selection']['end']) == datetime(2026, 9, 10, 14, tzinfo=timezone.utc) + timedelta(seconds=duration)


def test_authoring_uses_one_publication_selection_at_the_actual_door(monkeypatch):
    from woof import background_contract
    actual = background_contract.plan
    now = datetime(2026, 9, 10, 12, 5, tzinfo=timezone.utc)
    def selected(source, **kwargs):
        kwargs['now'] = now
        return actual(source, **kwargs)
    monkeypatch.setattr(background_contract, 'plan', selected)
    request = replace(_request(), epoch='2026-09-10T12:00:00Z')
    text, _, _, background = configuration(request, derive_rung(request, 1))
    assert tomllib.loads(text)['fetch']['cycle'] == '2026-09-10T06'
    assert background['selection']['cycle'] == '2026-09-10T06:00:00+00:00'


def test_saved_selection_does_not_resolve_a_later_latest_cycle(tmp_path, monkeypatch):
    from test_local_da_plan import availability, price
    from woof import background_contract
    from woof.local_da_runtime import read_plan
    plan = build_plan(_request(), availability=availability, price=price)
    published = publish(plan, tmp_path / 'case')
    monkeypatch.setattr(background_contract, 'plan', lambda *a, **k: pytest.fail('saved selection was resolved again'))
    loaded = read_plan(published['plan_path'])
    assert loaded['background']['selection'] == plan['background']['selection']


@requires_cupy
def test_staged_prepare_only_retains_output_and_never_calls_forecast(tmp_path, monkeypatch):
    from woof import runplan
    from test_runplan import _executed_staged_chain
    original = runplan._staged_chain
    returned = []
    def preparation(*args, **kwargs):
        target = kwargs['run_dir'] / 'chain' / 'run'
        target.mkdir(parents=True, exist_ok=True)
        frame = target / 'preserved.nc'
        frame.write_bytes(b'committed input generation')
        # The dispatch passes the plan's own prepare_only (False here);
        # this double turns it on rather than passing it a second time.
        value = original(*args, **{**kwargs, 'prepare_only': True})
        assert frame.read_bytes() == b'committed input generation'
        returned.append(value)
        return value
    monkeypatch.setattr(runplan, '_staged_chain', preparation)
    staged, _, _, _ = _executed_staged_chain(tmp_path, monkeypatch)
    assert [row[0] for row in staged] == ['fetch', 'prepare']
    assert returned[0]['forecast_started'] is False
    assert Path(returned[0]['experiment_config']).is_file()


def test_native_hourly_prepare_only_returns_published_authorities(tmp_path, monkeypatch):
    from woof import runplan
    from test_runplan_tiles import _drive
    from woof import stage_cli
    # The existing process double writes a digest-bearing empty proof. This
    # control observes its published authority relay, not proof validation.
    monkeypatch.setattr(stage_cli, 'resolve_bundle', lambda root: dict(
        document=root / 'proof.json', source='hrrr', layout='single', domains=1))
    original = runplan._hrrr_chain
    returned = []
    def preparation(*args, **kwargs):
        value = original(*args, **kwargs, prepare_only=True)
        returned.append(value)
        return value
    monkeypatch.setattr(runplan, '_hrrr_chain', preparation)
    captured, _, stages = _drive(tmp_path, monkeypatch)
    assert not captured
    assert [row[0] for row in stages] == ['fetch', 'prepare']
    assert returned[0]['forecast_started'] is False
    assert Path(returned[0]['experiment_config']).parent == Path(returned[0]['prepared_root'])


@pytest.mark.parametrize('cadence', [True, '3', 1.5, float('nan'), float('inf'), 0])
def test_invalid_boundary_cadence_is_refused_before_source_dispatch(cadence):
    request = _request(forcing_cadence_hours=cadence)
    with pytest.raises(ValueError, match='positive integer'):
        request.validate()


def _automatic_sources():
    from woof.background_contract import catalog
    return [row['source'] for row in catalog()['sources'] if 'automatic' in row.get('initialization_modes', [])]


#: The epoch every source is authored at unless its route table dates it out.
_EPOCH = datetime(2026, 9, 10, 12, tzinfo=timezone.utc)


def _epoch_inside_an_era(source):
    """``_EPOCH``, or a fixed epoch inside the source's own publication eras.

    The route table dates each file layout (woof/fetch_routes.py
    publication_era): GEFS v11 is a closed legacy archive and the 0.1 degree
    ECMWF open data begins 2026-10-20, so no single calendar date selects a
    layout for every automatic source.  Read from the table, never from the
    clock: ten days into the newest preparable era, or ``_EPOCH`` when that
    era already serves it.
    """
    from woof.fetch_routes import table_route

    route = table_route(source)
    eras = [era for era in (route.publication_eras if route else ())
            if not era.prep_refusal]
    naive = _EPOCH.replace(tzinfo=None)
    if not eras or any(era.valid_from + timedelta(days=3) <= naive
                       and (era.valid_until is None or naive < era.valid_until)
                       for era in eras):
        return _EPOCH.strftime('%Y-%m-%dT%H:%M:%SZ')
    newest = max(eras, key=lambda era: era.valid_from)
    epoch = newest.valid_from.replace(hour=12, minute=0, second=0) + timedelta(days=10)
    assert newest.valid_until is None or epoch < newest.valid_until - timedelta(days=3), source
    return epoch.strftime('%Y-%m-%dT%H:%M:%SZ')


@pytest.mark.parametrize('source', _automatic_sources())
def test_every_automatic_owner_authors_consumable_exact_fetch_arguments(source):
    from woof.background_contract import capability
    from woof.runplan import _fetch_arguments_from_hints
    from woof.cli import build_parser
    coverage = capability(source)['coverage'] or {}
    point = ((coverage['south'] + coverage['north']) / 2,
             (coverage['west'] + coverage['east']) / 2) if 'south' in coverage else (40., -100.)
    request = replace(_request(source=source), epoch=_epoch_inside_an_era(source), point=point)
    text, _, _, background = configuration(request, derive_rung(request, 1))
    hints = tomllib.loads(text)['fetch']
    args = build_parser().parse_args(['fetch', *_fetch_arguments_from_hints(hints, out=Path('unused'))])
    selection = background['selection']
    assert args.source == selection['source']
    assert args.cycle == selection['cycle'][:13]
    assert args.hours == selection['fetch_leads'][-1] - selection['fetch_leads'][0]
    assert args.member == selection['member']
