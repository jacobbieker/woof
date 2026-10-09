"""``woof doctor``'s row for ``woof energy``.

The row is local evidence only: whether :mod:`woof.energy` imports, a
census of the Overpass response cache, whether the Rust site sampler's
library is on disk (never loaded), and subprocess import probes of the
optional array-store packages.  Every case forbids sockets, and every
case forces the import probe so no subprocess runs.
"""

from __future__ import annotations

import importlib
import re
import socket
import sys

import pytest

from woof import bridges, doctor

#: First tokens a remedy line may start with (the contract
#: ``tests/test_doctor.py`` holds every remedy to); a ``#`` comment is the
#: other permitted line kind.
_COMMANDS = {"pip", "python", "git", "cd", "cargo", "woof", "export"}


@pytest.fixture(autouse=True)
def _offline(monkeypatch, tmp_path):
    def refuse(*args, **kwargs):
        raise AssertionError("the doctor row tried to use the network")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setenv("WOOF_ENERGY_OSM_CACHE", str(tmp_path / "osm-cache"))
    monkeypatch.delenv("WOOF_SITESAMPLE_BRIDGE", raising=False)


@pytest.fixture(autouse=True)
def _every_stage_implemented(request, monkeypatch):
    """The sampler and package cases assume a build with every stage
    written; the ``test_stub_*`` cases below ask the real reader."""

    if not request.node.name.startswith("test_stub_"):
        monkeypatch.setattr(doctor, "_energy_unimplemented_stages",
                            lambda: [])


def _probe(monkeypatch, answers=None):
    """Force the optional-package probe; returns the names it was asked."""

    asked: list[str] = []
    answers = answers or {}

    def probe(name, distribution=None):
        asked.append(name)
        return answers.get(name, (True, "1.0"))

    monkeypatch.setattr(doctor, "_import_probe", probe)
    return asked


def _ladder(monkeypatch, *paths):
    monkeypatch.setattr(doctor, "_energy_sitesample_candidates",
                        lambda: tuple(paths))


def _assert_pasteable(remedy: str) -> None:
    for line in remedy.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        assert "<" not in stripped and ">" not in stripped, line
        if stripped.startswith("#"):
            continue
        assert stripped.split()[0] in _COMMANDS, line
        for token in stripped.split():
            assert not re.fullmatch(r"[A-Z][A-Z0-9_]{2,}", token.rstrip(",.")), \
                f"placeholder {token!r} in a command line: {line!r}"


def test_the_row_is_in_the_assembled_report():
    assembler = getattr(doctor, "_collect_checks", doctor.collect_checks)
    assert "_energy_check" in assembler.__code__.co_names


def test_the_row_follows_the_other_optional_offline_row():
    import inspect

    source = inspect.getsource(doctor._collect_checks)
    assert (source.index("_dynamical_asos_check()")
            < source.index("_energy_check()"))


def test_everything_present_is_present_and_offline(monkeypatch, tmp_path):
    library = tmp_path / "librw_sitesample.so"
    library.write_bytes(b"\x7fELF")
    _ladder(monkeypatch, tmp_path / "absent.so", library)
    asked = _probe(monkeypatch)
    check = doctor._energy_check()
    assert check.name == doctor.ENERGY_NAME
    assert check.status == "present"
    assert check.blocking is False
    assert check.remedy is None
    assert asked == ["xarray", "zarr", "icechunk"]
    assert str(library) in check.detail
    assert "does not load it" in check.detail
    assert "package woof.energy imports" in check.detail
    assert "no Overpass server or other host is contacted" in check.detail
    assert "ODbL 1.0" in check.detail
    assert "not created yet" in check.detail
    assert not doctor.blocking_gaps([check])


def test_an_absent_sampler_is_info_and_names_the_build(monkeypatch, tmp_path):
    _ladder(monkeypatch, tmp_path / "librw_sitesample.so")
    _probe(monkeypatch)
    monkeypatch.setattr(bridges, "sources_present", lambda *a, **k: True)
    monkeypatch.setattr(doctor, "_energy_sitesample_crate_present",
                        lambda: True)
    check = doctor._energy_check()
    assert check.status == "info"
    assert check.blocking is False
    assert "librw_sitesample.so not found" in check.detail
    assert str(tmp_path / "librw_sitesample.so") in check.detail
    assert "`woof energy extract` refuses" in check.detail
    assert "cargo build --release --locked --offline" in check.remedy
    assert check.brief == "site sampler not built; extract refuses"
    assert check not in doctor.blocking_gaps([check])


@pytest.mark.parametrize("sources,crate", [(True, True), (True, False),
                                           (False, False)])
def test_every_sampler_remedy_is_commands_or_comments(monkeypatch, tmp_path,
                                                      sources, crate):
    _ladder(monkeypatch, tmp_path / "librw_sitesample.so")
    _probe(monkeypatch)
    monkeypatch.setattr(bridges, "sources_present", lambda *a, **k: sources)
    monkeypatch.setattr(doctor, "_energy_sitesample_crate_present",
                        lambda: crate)
    check = doctor._energy_check()
    assert check.status == "info"
    _assert_pasteable(check.remedy)
    commands = [line.strip() for line in check.remedy.splitlines()
                if line.strip() and not line.strip().startswith("#")]
    if sources and crate:
        assert any("cargo build" in line for line in commands)
        assert check.action.startswith("cd ")
    else:
        # No crate to build, or no sources at all: no published bundle
        # carries the sampler, so no command is offered that cannot
        # supply it -- the block is comments only.
        assert commands == []
        assert "woof fetch-bridges" not in check.remedy
    if not sources:
        # On a wheel with a published bundle, tests/test_doctor.py holds
        # every block that names the cargo build to lead with
        # `woof fetch-bridges`, which cannot supply the sampler -- so the
        # wheel block names the build without that literal, and stays
        # composable beside an optional-package pip line.
        assert "cargo build" not in check.remedy


def test_a_wheel_block_with_a_pip_line_names_no_cargo_build(monkeypatch,
                                                          tmp_path):
    _ladder(monkeypatch, tmp_path / "librw_sitesample.so")
    _probe(monkeypatch, {"icechunk": (False, "not installed")})
    monkeypatch.setattr(bridges, "sources_present", lambda *a, **k: False)
    check = doctor._energy_check()
    commands = [line.strip() for line in check.remedy.splitlines()
                if line.strip() and not line.strip().startswith("#")]
    assert commands == ["pip install icechunk"]
    assert "cargo build" not in check.remedy


def test_the_env_override_is_honoured(monkeypatch, tmp_path):
    library = tmp_path / "custom" / "librw_sitesample.so"
    library.parent.mkdir()
    library.write_bytes(b"\x7fELF")
    monkeypatch.setenv("WOOF_SITESAMPLE_BRIDGE", str(library))
    _probe(monkeypatch)
    candidates = doctor._energy_sitesample_candidates()
    assert candidates[0] == library
    check = doctor._energy_check()
    assert check.status == "present"
    assert str(library) in check.detail


def test_an_override_naming_no_file_is_a_degraded_gap(monkeypatch, tmp_path):
    monkeypatch.setenv("WOOF_SITESAMPLE_BRIDGE", str(tmp_path / "nope.so"))
    asked = _probe(monkeypatch)
    check = doctor._energy_check()
    assert check.status == "missing"
    assert check.blocking is False
    assert check.severity == doctor.SEVERITY_DEGRADED
    assert "WOOF_SITESAMPLE_BRIDGE names" in check.detail
    assert "WOOF_SITESAMPLE_BRIDGE" in check.action
    assert asked == []
    _assert_pasteable(check.remedy)
    assert not doctor.blocking_gaps([check])


def test_absent_optional_packages_are_info_with_the_pip_line(monkeypatch,
                                                             tmp_path):
    library = tmp_path / "librw_sitesample.so"
    library.write_bytes(b"\x7fELF")
    _ladder(monkeypatch, library)
    _probe(monkeypatch, {"zarr": (False, "not installed"),
                         "icechunk": (False, "not installed")})
    check = doctor._energy_check()
    assert check.status == "info"
    assert check.blocking is False
    assert check.action == "pip install zarr icechunk"
    assert "pip install zarr icechunk" in check.remedy.splitlines()
    assert "--format icechunk" in check.detail
    _assert_pasteable(check.remedy)


def test_a_broken_optional_package_gets_the_reinstall(monkeypatch, tmp_path):
    library = tmp_path / "librw_sitesample.so"
    library.write_bytes(b"\x7fELF")
    _ladder(monkeypatch, library)
    _probe(monkeypatch, {"xarray": (False, "installed but failed to import: "
                                           "ImportError: boom")})
    check = doctor._energy_check()
    assert check.status == "info"
    assert "pip install --force-reinstall xarray" in check.remedy.splitlines()
    _assert_pasteable(check.remedy)


def test_the_overpass_cache_is_counted(monkeypatch, tmp_path):
    cache = tmp_path / "osm-cache" / "tiles"
    cache.mkdir(parents=True)
    (cache / "51_-4.json").write_bytes(b"x" * 2_000_000)
    _ladder(monkeypatch, tmp_path / "librw_sitesample.so")
    _probe(monkeypatch)
    check = doctor._energy_check()
    assert "1 file(s), 2.0 MB" in check.detail
    assert str(tmp_path / "osm-cache") in check.detail


def test_the_default_cache_root(monkeypatch, tmp_path):
    monkeypatch.delenv("WOOF_ENERGY_OSM_CACHE")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    assert doctor._energy_osm_cache_dir() == (
        tmp_path / ".woof" / "cache" / "energy-osm")


def test_a_build_without_the_package_is_info(monkeypatch):
    monkeypatch.setitem(sys.modules, "woof.energy", None)
    asked = _probe(monkeypatch)
    check = doctor._energy_check()
    assert check.status == "info"
    assert check.blocking is False
    assert asked == [], "no optional probe is worth running without woof.energy"
    assert "does not include the energy package" in check.detail
    _assert_pasteable(check.remedy)


def _failing_import(monkeypatch, error):
    real = importlib.import_module

    def fake(name, package=None):
        if name == "woof.energy":
            raise error
        return real(name, package)

    monkeypatch.setattr(importlib, "import_module", fake)


def test_a_package_missing_a_dependency_is_degraded(monkeypatch):
    error = ModuleNotFoundError("No module named 'woof.energy.contracts'",
                                name="woof.energy.contracts")
    _failing_import(monkeypatch, error)
    _probe(monkeypatch)
    check = doctor._energy_check()
    assert check.status == "missing"
    assert check.severity == doctor.SEVERITY_DEGRADED
    assert check.blocking is False
    assert "does not import" in check.detail
    _assert_pasteable(check.remedy)


def test_a_package_that_raises_is_degraded(monkeypatch):
    _failing_import(monkeypatch, RuntimeError("boom"))
    _probe(monkeypatch)
    check = doctor._energy_check()
    assert check.status == "missing"
    assert check.severity == doctor.SEVERITY_DEGRADED
    assert "RuntimeError: boom" in check.detail
    assert not doctor.blocking_gaps([check])


@pytest.mark.parametrize("platform,name", [
    ("linux", "librw_sitesample.so"),
    ("darwin", "librw_sitesample.dylib"),
    ("win32", "rw_sitesample.dll"),
])
def test_the_library_name_follows_the_platform(monkeypatch, platform, name):
    monkeypatch.setattr(doctor.sys, "platform", platform)
    assert doctor._energy_sitesample_filename() == name


def _stage_module(name, main):
    import types

    module = types.ModuleType(f"woof.energy.{name}")
    module.main = main
    return module


def test_stub_stages_are_read_from_the_code_not_called(monkeypatch):
    from woof.energy import contracts

    # A stub exactly as the scaffold writes one: a module-level main whose
    # only global name is EnergyNotImplemented.
    namespace = {"EnergyNotImplemented": contracts.EnergyNotImplemented}
    exec("def main(args):\n"
         "    raise EnergyNotImplemented('woof energy fetch')\n", namespace)
    stub = namespace["main"]
    calls = []

    def written(args):
        calls.append(args)
        print({"ok": True})
        return 0

    for _stage, module in doctor._ENERGY_STAGES:
        monkeypatch.setitem(sys.modules, f"woof.energy.{module}",
                            _stage_module(module, written))
    monkeypatch.setitem(sys.modules, "woof.energy.osm",
                        _stage_module("osm", stub))
    monkeypatch.setitem(sys.modules, "woof.energy.extract",
                        _stage_module("extract", stub))
    assert doctor._energy_unimplemented_stages() == ["fetch", "extract"]
    assert calls == [], "the reader must never run a stage"


def test_stub_stages_make_the_row_info_and_name_them(monkeypatch, tmp_path):
    library = tmp_path / "librw_sitesample.so"
    library.write_bytes(b"\x7fELF")
    _ladder(monkeypatch, library)
    _probe(monkeypatch)
    monkeypatch.setattr(doctor, "_energy_unimplemented_stages",
                        lambda: ["fetch", "rating"])
    check = doctor._energy_check()
    assert check.status == "info"
    assert check.blocking is False
    assert "2 of 9 stage(s) are not implemented" in check.detail
    assert "fetch, rating" in check.detail
    assert check.brief == "2 of 9 stages not implemented in this build"
    _assert_pasteable(check.remedy)


def test_stub_reader_on_this_tree_answers_without_raising():
    stubs = doctor._energy_unimplemented_stages()
    assert set(stubs) <= {stage for stage, _ in doctor._ENERGY_STAGES}


def test_the_fetchers_own_cache_root_wins(monkeypatch, tmp_path):
    module = _stage_module("osm", lambda args: 0)
    module.cache_root = lambda: tmp_path / "from-osm"
    monkeypatch.setitem(sys.modules, "woof.energy.osm", module)
    assert doctor._energy_osm_cache_dir() == tmp_path / "from-osm"


def test_an_override_the_bridge_ladder_ignores_is_not_judged(monkeypatch,
                                                            tmp_path):
    library = tmp_path / "librw_sitesample.so"
    library.write_bytes(b"\x7fELF")
    monkeypatch.setenv("WOOF_SITESAMPLE_BRIDGE", str(tmp_path / "stale.so"))
    _ladder(monkeypatch, library)
    _probe(monkeypatch)
    check = doctor._energy_check()
    assert check.status == "present"
