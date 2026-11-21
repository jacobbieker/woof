"""The umbrella over the opt-in quirk corrections (woof.quirk_corrections).

``woof_quirk_corrections = 'all'`` turns on every opt-in correction of a
WRF 4.7.1 / MPAS v8.4.2 quirk where its scheme runs; ``'none'`` (the
default) changes nothing. This file pins:

* the ledger: every opt-in correction the code carries is in
  :data:`woof.quirk_corrections.CORRECTIONS` (a new one cannot be left out
  of the umbrella), and README.md and CONFIGURATION.md list each one;
* the default: nothing changes on any route without the umbrella;
* 'all': each route's corrections take their corrected values, a
  conflicting key is refused by name, and a refused combination stays
  refused by name.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
import warnings
from pathlib import Path

import pytest

from woof import quirk_corrections as qc
from woof.config import ConfigurationRefused, RunConfig, validate_run_config
from woof.hex.errors import ConfigurationRefusal

ROOT = Path(__file__).resolve().parents[1]
MPAS_PORTS = ("mpas_ocean", "mpas_ice", "mpas_seaice")
#: woof_* keys in the MPAS ports that are not corrections themselves
NOT_CORRECTIONS = {"woof_v842_corrections", qc.UMBRELLA}


# --- the ledger ---------------------------------------------------------

def test_the_ledger_is_well_formed():
    assert len(qc.BY_KEY) == len(qc.CORRECTIONS)
    for c in qc.CORRECTIONS:
        assert c.route in qc.ROUTES, c.key
        assert c.default != c.corrected, c.key
        assert c.summary and c.scheme, c.key


def test_every_runconfig_quirk_correction_is_in_the_ledger():
    fields = RunConfig.__dataclass_fields__
    assert set(qc.keys("wrf")) <= set(fields)
    for key in qc.keys("wrf"):
        assert fields[key].default is qc.BY_KEY[key].default
        assert fields[key].kw_only
    # a field whose comment calls it a correction of a WRF quirk
    lines = (ROOT / "woof" / "config.py").read_text(encoding="utf-8").splitlines()
    marked = []
    for i, line in enumerate(lines):
        m = re.match(r"    ([a-z_0-9]+): .*=", line)
        if not m:
            continue
        j, block = i - 1, []
        while j >= 0 and lines[j].strip().startswith("#"):
            block.append(lines[j])
            j -= 1
        if re.search(r"correction of a WRF [0-9.]+ quirk", " ".join(block)):
            marked.append(m.group(1))
    assert "ysu_surface_heat_exner" in marked
    assert set(marked) <= set(qc.keys("wrf")), set(marked) - set(qc.keys("wrf"))
    with pytest.raises(KeyError):
        qc.run_scheme_active("not_a_correction", RunConfig(nx=4, ny=4, nz=4, dx=1.0, dy=1.0, ztop=1.0, dt=1.0,
                                                           run_seconds=1.0))
    for key in qc.keys("wrf"):
        qc.run_scheme_active(key, {}, RunConfig)


def test_every_mpas_opt_in_key_is_in_the_ledger():
    found = set()
    for port in MPAS_PORTS:
        for path in (ROOT / "woof" / "hex" / port).glob("*.py"):
            found |= set(re.findall(r"[\"'](woof_[a-z0-9_]+)[\"']", path.read_text(encoding="utf-8")))
    found -= NOT_CORRECTIONS
    mpas = set(qc.keys("mpas_ocean")) | set(qc.keys("mali"))
    assert found == mpas, (found - mpas, mpas - found)
    from woof.hex.mpas_ocean import forcing_v842
    assert set(forcing_v842.WOOF_OPT_INS) <= set(qc.keys("mpas_ocean"))


def test_every_seaice_correction_table_is_in_the_ledger():
    tables = {}
    for port in MPAS_PORTS:
        for path in (ROOT / "woof" / "hex" / port).glob("*.py"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in tree.body:
                targets = node.targets if isinstance(node, ast.Assign) else (
                    [node.target] if isinstance(node, ast.AnnAssign) else [])
                for t in targets:
                    if isinstance(t, ast.Name) and t.id.endswith("CORRECTIONS"):
                        tables[f"{path.name}:{t.id}"] = node.value
    assert "column_v842.py:CORRECTIONS" in tables
    from woof.hex.mpas_seaice import column_v842
    assert set(column_v842.CORRECTIONS) == set(qc.keys("mpas_seaice"))
    assert set(tables) == {"column_v842.py:CORRECTIONS"}, sorted(tables)


def test_the_hex_keys_are_door_flags():
    from woof.hex import forecast_door
    source = Path(forecast_door.__file__).read_text(encoding="utf-8")
    for key in qc.keys("hex"):
        flag = "--" + key.replace("_", "-")
        assert f'"{flag}"' in source, flag
        assert repr(qc.BY_KEY[key].corrected).strip("'") in source
    assert '"--quirk-corrections"' in source


def test_the_docs_list_every_correction():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    config = (ROOT / "docs" / "public" / "CONFIGURATION.md").read_text(encoding="utf-8")
    start = readme.index("woof_quirk_corrections")
    umbrella_readme = readme[start:start + 3000]
    start = config.index("`woof_quirk_corrections` (default")
    umbrella_config = config[start:start + 3000]
    for c in qc.CORRECTIONS:
        name = "--undriven-species-boundary" if c.key == "undriven_species_boundary" else c.key
        for text in (umbrella_readme, umbrella_config):
            assert re.sub(r"\s+", " ", name) in re.sub(r"\s+", " ", text), name


# --- the default --------------------------------------------------------

def _base(**kw):
    return RunConfig(nx=8, ny=8, nz=12, dx=3000.0, dy=3000.0, ztop=12000.0, dt=10.0, run_seconds=60.0, **kw)


def test_the_default_changes_nothing():
    assert RunConfig.__dataclass_fields__[qc.UMBRELLA].default == "none"
    kw = dict(bl_pbl_physics=1, cu_physics=3)
    assert qc.apply_run_corrections(kw) is kw
    validate_run_config(_base(bl_pbl_physics=1, sf_sfclay_physics=1))
    for key in qc.keys("mpas_ocean") + qc.keys("mali"):
        assert qc.option({}, key) == qc.BY_KEY[key].default
        assert qc.option({qc.UMBRELLA: "none"}, key) == qc.BY_KEY[key].default
        assert qc.option({key: qc.BY_KEY[key].corrected}, key) == qc.BY_KEY[key].corrected
    from woof.hex.mpas_seaice import column_v842
    assert column_v842.corrections({}) == frozenset()
    assert column_v842.corrections({"woof_v842_corrections": "zsal_solve"}) == {"zsal_solve"}
    assert qc.hex_undriven_species_boundary(argparse.Namespace(lbc_dir="x")) is None
    assert qc.hex_undriven_species_boundary(
        argparse.Namespace(lbc_dir="x", undriven_species_boundary="wrf-flow-dependent")) == "wrf-flow-dependent"


def test_an_unknown_umbrella_value_is_refused():
    with pytest.raises(ConfigurationRefused, match="woof_quirk_corrections"):
        validate_run_config(_base(woof_quirk_corrections="some"))
    with pytest.raises(ConfigurationRefused, match="woof_quirk_corrections"):
        qc.apply_run_corrections({qc.UMBRELLA: "yes"})
    with pytest.raises(ConfigurationRefusal):
        qc.option({qc.UMBRELLA: True}, "woof_statistics_extrema")


# --- 'all' ---------------------------------------------------------------

def test_all_turns_on_the_run_corrections_where_their_scheme_runs():
    kw = qc.apply_run_corrections({qc.UMBRELLA: "all", "bl_pbl_physics": 1, "cu_physics": 3})
    assert kw["ysu_surface_heat_exner"] is True and kw["gf_deep_rain_evaporation_booking"] is True
    kw = qc.apply_run_corrections({qc.UMBRELLA: "all", "bl_pbl_physics": 5, "cu_physics": 0})
    assert "ysu_surface_heat_exner" not in kw and "gf_deep_rain_evaporation_booking" not in kw
    run = validate_run_config(_base(**qc.apply_run_corrections(
        {qc.UMBRELLA: "all", "bl_pbl_physics": 1, "sf_sfclay_physics": 1})))
    assert run.ysu_surface_heat_exner is True


def test_all_refuses_a_conflicting_key_by_name():
    with pytest.raises(ConfigurationRefused, match="ysu_surface_heat_exner"):
        qc.apply_run_corrections({qc.UMBRELLA: "all", "bl_pbl_physics": 1, "ysu_surface_heat_exner": False})
    # a RunConfig built without apply_run_corrections is refused, not half-applied
    with pytest.raises(ConfigurationRefused, match="ysu_surface_heat_exner"):
        validate_run_config(_base(bl_pbl_physics=1, sf_sfclay_physics=1, woof_quirk_corrections="all"))
    for key in qc.keys("mpas_ocean") + qc.keys("mali"):
        assert qc.option({qc.UMBRELLA: "all"}, key) == qc.BY_KEY[key].corrected
        assert qc.option({qc.UMBRELLA: "all", key: qc.BY_KEY[key].corrected}, key) == qc.BY_KEY[key].corrected
        with pytest.raises(ConfigurationRefusal, match=key):
            qc.option({qc.UMBRELLA: "all", key: qc.BY_KEY[key].default}, key)
    with pytest.raises(ValueError, match="undriven_species_boundary"):
        qc.hex_undriven_species_boundary(
            argparse.Namespace(lbc_dir="x", quirk_corrections="all", undriven_species_boundary="free"))


def test_all_keeps_a_refused_combination_refused_by_name():
    # YSU's correction is refused with BEP/BEM; the umbrella does not bypass it
    with pytest.raises(ConfigurationRefused, match="ysu_surface_heat_exner"):
        validate_run_config(_base(**qc.apply_run_corrections(dict(
            woof_quirk_corrections="all", bl_pbl_physics=1, sf_sfclay_physics=1, sf_surface_physics=2,
            sf_urban_physics=2))))


def test_all_on_the_hex_door_is_regional_only():
    ns = argparse.Namespace(quirk_corrections="all", lbc_dir=None)
    assert qc.hex_undriven_species_boundary(ns) is None
    ns = argparse.Namespace(quirk_corrections="all", lbc_dir="x")
    assert qc.hex_undriven_species_boundary(ns) == "wrf-flow-dependent"
    from woof.hex import forecast_door
    assert forecast_door._resolve_undriven_species_boundary(ns) == "wrf-flow-dependent"


def test_all_turns_on_every_seaice_column_correction():
    from woof.hex.mpas_seaice import column_v842
    assert column_v842.corrections({qc.UMBRELLA: "all"}) == set(column_v842.CORRECTIONS)
    assert column_v842.named_corrections({qc.UMBRELLA: "all"}) == frozenset()


def test_the_ocean_forcing_configs_resolve_the_umbrella():
    from woof.hex.mpas_ocean import forcing_v842
    merged = forcing_v842._configs({qc.UMBRELLA: "all"})
    assert qc.UMBRELLA not in merged
    assert merged["woof_bottom_drag_dry_edges"] == "skip"
    assert merged["woof_rayleigh_bottom_friction"] == "corrected"
    default = forcing_v842._configs({})
    assert default["woof_bottom_drag_dry_edges"] == "refuse"
    assert default["woof_rayleigh_bottom_friction"] == "v842"
    with pytest.raises(ConfigurationRefusal, match="woof_rayleigh_bottom_friction"):
        forcing_v842._configs({qc.UMBRELLA: "all", "woof_rayleigh_bottom_friction": "v842"})


def test_the_experiment_reader_applies_the_umbrella_per_domain(tmp_path):
    sys.path.insert(0, str(ROOT / "tests"))
    from test_gf_closure_choice import _grell_tree_toml
    from woof.experiment import load_experiment

    def tree(**tables):
        path = tmp_path / "tree.toml"
        path.write_text(_grell_tree_toml(**tables), encoding="utf-8")
        return [(d.run.cu_physics, d.run.gf_deep_rain_evaporation_booking, d.run.ysu_surface_heat_exner)
                for d in load_experiment(path).domains]

    umbrella = 'woof_quirk_corrections = "all"\n'
    assert tree() == [(0, False, False), (0, False, False)]
    assert tree(shared=umbrella, root="cu_physics = 3\n") == [(3, True, True), (0, False, True)]
    with pytest.raises(ValueError, match="ysu_surface_heat_exner"):
        tree(shared=umbrella, root="ysu_surface_heat_exner = false\n")
