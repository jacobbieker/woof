"""WRF v4.8.1 as the newest reference: the source pin and the selector.

Breakage prevented: a re-pin that cites a 4.8.1 file the pin does not hold,
a pin whose commit disagrees with woof.wrf_reference, a 4.8.0 -> 4.8.1
change that no plan row owns, a patch release (4.8.1) read as its minor
release (4.8) by the release-generation table, and a run receipt that
claims 4.8.1 arithmetic for a scheme whose 4.8.1 port has not landed.
CPU only.
"""
from __future__ import annotations

from tools.check_registry_citations import _in_wrf481_pin
from tools.wrf481_source import manifest
from woof import namelist_import
from woof.wrf_namelist_registry import wrf_namelist_table
from woof.wrf_reference import (WRF481_PENDING, WRF_REFERENCE_DEFAULT,
                                WRF_REFERENCE_SOURCES, WRF_REFERENCES,
                                wrf_reference_receipt)


def test_the_pin_is_the_tag_commit_and_its_gitlinks():
    pin = manifest.load()
    assert pin["commit"] == WRF_REFERENCE_SOURCES["4.8.1"]["commit"]
    assert pin["tag"] == WRF_REFERENCE_SOURCES["4.8.1"]["tag"] == "v4.8.1"
    assert {k: pin[k] for k in manifest.PIN} == manifest.PIN
    # 4.8.0's submodules that 4.8.1 did not move.
    for unchanged in ("phys/GFL", "phys/MYNN-EDMF", "phys/MYNN-SFC", "phys/TEMPO"):
        assert pin["submodules"][unchanged]
    assert WRF_REFERENCES[0] == WRF_REFERENCE_DEFAULT == "4.8.1"


def test_every_changed_cited_file_has_a_plan_row():
    files = manifest.load()["files"]
    changed = {path: row["changes"] for path, row in files.items()
               if row["sha256"] != row["sha256_v4.8.0"]}
    assert changed and all("unclassified" not in ids for ids in changed.values())
    assert set(changed) <= set(manifest.CHANGES)
    assert not any(path.startswith(manifest.EXCLUDED) for path in files)


def test_the_481_vocabulary_is_the_480_vocabulary():
    new, old = wrf_namelist_table("4.8.1"), wrf_namelist_table("4.8.0")
    assert new["wrf"]["commit"] == WRF_REFERENCE_SOURCES["4.8.1"]["commit"]
    assert new["keys"] == old["keys"]


def test_a_patch_release_row_splits_480_from_481(monkeypatch):
    monkeypatch.setattr(namelist_import, "WRF_RELEASE_GENERATIONS", (
        ("ntu_version", (4, 8, 1), "wrf_480"),
        ("ysu_version", (4, 8), "wrf_461")))
    generations = namelist_import.wrf_release_generations
    assert generations("4.8.1") == {}
    assert generations("4.8.0") == generations("4.8") == {"ntu_version": "wrf_480"}
    assert generations("4.7.1") == {"ntu_version": "wrf_480",
                                    "ysu_version": "wrf_461"}


def test_a_481_receipt_names_the_ports_still_pending():
    from dataclasses import MISSING, fields
    from types import SimpleNamespace

    from woof.config import RunConfig
    defaults = SimpleNamespace(**{f.name: f.default for f in fields(RunConfig)
                                  if f.default is not MISSING})
    receipt = wrf_reference_receipt(defaults)
    assert receipt["wrf_reference"] == "4.8.1"
    if WRF481_PENDING:
        assert receipt["wrf_reference_pending"] == sorted(WRF481_PENDING)
    else:
        assert "wrf_reference_pending" not in receipt


def test_a_481_citation_must_name_a_pinned_file():
    assert _in_wrf481_pin("phys/module_bl_boulac.F")
    assert _in_wrf481_pin("bl_shinhong.F90")
    assert not _in_wrf481_pin("module_bl_no_such_scheme.F")


def test_the_checker_fails_a_481_citation_the_pin_lacks(monkeypatch):
    from tools import check_registry_citations as crc
    monkeypatch.setitem(crc.EXTERNAL, "module_bl_no_such_scheme.F", crc._WRF481)
    monkeypatch.setitem(crc.EXTERNAL, "module_bl_boulac.F", crc._WRF481)
    monkeypatch.setattr(crc, "named_routine_failures", lambda: [])
    monkeypatch.setattr(crc, "DRIFTED", {})
    failures = crc.check({"a": "module_bl_no_such_scheme.F:10",
                          "b": "module_bl_boulac.F:10"})
    assert [f for f in failures if "v4.8.1" in f] == [
        "module_bl_no_such_scheme.F:10 (in $.a) is declared WRF v4.8.1 but "
        "tools/wrf481_source/source-v4.8.1.json pins no such file"]
