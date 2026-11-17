"""A [case_data] table holding only the GFS policy keys stays on woof go's fetch route.

gfs_snow_convention, gfs_lake_skin and gfs_lake_ice are read only by the
fetched-GFS preparation, but any [case_data] table used to send ``woof go``
to the declared-inputs route, where --prepare-only refuses and native
forcing must be GRIB1, so the lake policies could not reach a GFS run.
A table with any other key still routes (and is refused) exactly as before.
"""
import pytest

from woof.case_data import (
    GFS_POLICY_KEYS, declares_case_inputs, gfs_preparation_policy,
    is_gfs_policy_table, optional_case_data_from_config,
    optional_case_policy_from_config)
from woof.cli import main
from woof.experiment import load_experiment
from test_case_data import _CASE_DATA_TOML, _EXPERIMENT_TOML, make_case_toml

_FETCH = '\n[fetch]\nsource = "gfs"\ncycle = "1999-05-03T12"\nhours = 6\narea = "36.0,-105.0,44.0,-91.0"\n'
_AREA = "34.5,-98.5,35.5,-97.5"
_POLICY = '\n[case_data]\ngfs_lake_skin = "air_temperature"\ngfs_lake_ice = "wps"\n'


def _write(tmp_path, text):
    path = tmp_path / "case.toml"
    path.write_text(text, encoding="utf-8")
    return path


@pytest.mark.parametrize("payload,declared", [
    pytest.param({}, False, id="no-table"),
    pytest.param({"case_data": {"gfs_lake_skin": "air_temperature"}}, False, id="policy-only"),
    pytest.param({"case_data": dict.fromkeys(GFS_POLICY_KEYS, "wps")}, False, id="all-policy"),
    pytest.param({"case_data": {}}, True, id="empty-table"),
    pytest.param({"case_data": {"gfs_lake_skin": "wps", "co2_vmr": 4e-4}}, True, id="mixed"),
    pytest.param({"case_data": {"gfs_lake_skn": "wps"}}, True, id="mistyped-key"),
    pytest.param({"case_data": []}, True, id="malformed"),
])
def test_only_a_pure_policy_table_stays_off_the_declared_route(payload, declared):
    assert declares_case_inputs(payload) is declared
    assert is_gfs_policy_table(payload.get("case_data")) is (
        "case_data" in payload and not declared)


def test_policy_only_table_reaches_the_gfs_preparation_policy(tmp_path):
    path = _write(tmp_path, _EXPERIMENT_TOML + _POLICY + _FETCH)
    load_experiment(path)
    data, policy = optional_case_policy_from_config(path)
    assert data is None
    assert optional_case_data_from_config(path) is None
    assert policy == {"gfs_snow_convention": "wps",
                      "gfs_lake_skin": "air_temperature", "gfs_lake_ice": "wps"}


def test_policy_only_table_values_are_still_validated(tmp_path):
    path = _write(tmp_path, _EXPERIMENT_TOML
                  + '\n[case_data]\ngfs_lake_skin = "nearest"\n' + _FETCH)
    with pytest.raises(ValueError, match="gfs_lake_skin"):
        load_experiment(path)
    with pytest.raises(ValueError, match="gfs_lake_skin"):
        optional_case_policy_from_config(path)


def test_defaults_are_unchanged_without_a_table_and_with_a_full_table(tmp_path):
    bare = _write(tmp_path, _EXPERIMENT_TOML + _FETCH)
    data, policy = optional_case_policy_from_config(bare)
    assert data is None
    assert policy == gfs_preparation_policy(None) == {
        "gfs_snow_convention": "wps", "gfs_lake_skin": "wps", "gfs_lake_ice": "observed"}
    full_dir = tmp_path / "full"
    full_dir.mkdir()
    full = make_case_toml(full_dir, case_data=_CASE_DATA_TOML + 'gfs_lake_ice = "wps"\n')
    data, policy = optional_case_policy_from_config(full)
    assert data is not None
    assert policy == gfs_preparation_policy(data)
    assert policy["gfs_lake_ice"] == "wps"


def _gfs_config(tmp_path, case_data):
    """The shared GFS fetch fixture with ``case_data`` added as its [case_data]."""
    from test_companion_setups import gfs_case
    from woof.toml_document import emit_experiment_toml
    source, raw = gfs_case(tmp_path)
    raw["fetch"]["area"] = _AREA
    if case_data is not None:
        raw["case_data"] = case_data
    source.write_text(emit_experiment_toml(raw), encoding="utf-8")
    return source


def _prepare_only_dry_run(tmp_path, monkeypatch, capsys, case_data):
    from _staged_geog import staged_geog_tree
    from woof import go_cli
    asked = []
    # The fetch route's dated-archive probe is that route's own step (the
    # declared-inputs route skips it); answer it offline.
    monkeypatch.setattr(go_cli, "_archive_dry_readiness",
                        lambda args, payload, cycle: asked.append(cycle))
    config = _gfs_config(tmp_path, case_data)
    geog = staged_geog_tree(tmp_path / "geog")
    code = main(["go", str(config), "--prepare-only", "--dry-run",
                 "--geog-root", str(geog), "--outdir", str(tmp_path / "out")])
    captured = capsys.readouterr()
    return code, captured.out + captured.err, asked


def test_prepare_only_accepts_a_policy_only_table_on_the_fetch_route(
        tmp_path, monkeypatch, capsys):
    code, text, asked = _prepare_only_dry_run(
        tmp_path, monkeypatch, capsys,
        {"gfs_lake_skin": "air_temperature", "gfs_lake_ice": "wps"})
    assert code == 0, text
    assert "fetch -> prepare" in text
    assert asked


def test_prepare_only_still_refuses_a_mixed_table(tmp_path, monkeypatch, capsys):
    code, text, _asked = _prepare_only_dry_run(
        tmp_path, monkeypatch, capsys,
        {"gfs_lake_skin": "air_temperature", "co2_vmr": 4.0e-4})
    assert code == 2
    assert "fetch -> prepare" not in text
    assert "names its input files in [case_data]" in text


def test_prepare_only_without_a_table_is_unchanged(tmp_path, monkeypatch, capsys):
    code, text, asked = _prepare_only_dry_run(tmp_path, monkeypatch, capsys, None)
    assert code == 0, text
    assert "fetch -> prepare" in text
    assert asked
