"""Opt-in code (``optional/``, :mod:`woof.optin`) is in neither wheel.

The fork allows code under licences the default distributions cannot carry
only as explicit opt-ins (the glibc 2.39 libm transcription, #145, set the
pattern): the source sits under ``optional/<name>`` outside every package,
with its licence beside it, and loads only through :func:`woof.optin.load`.
This file holds both published wheels to that: ``recast-woof`` from the
repository's pyproject.toml and ``metal-woof`` from the same file rewritten
by ``tools/metal_woof_project.py``, as .github/workflows/publish-metal.yml
does.  It asks setuptools which files each declaration selects (the
discipline of tests/test_package_data_coverage.py), so what it measures is
what ``python -m build`` would put in the wheel, and asserts that no selected
file is under ``optional/``, none carries an opt-in marker line
(``WOOF-OPTIN:``), and none carries an opt-in's licence text.
"""

from __future__ import annotations

import glob
import sys
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"
OPTIONAL = REPO_ROOT / "optional"

#: A sentence of each opt-in's licence that no shipped file may carry.
LICENCE_SENTENCES = {
    "wrf_hydro": "revise and copy the WRF-Hydro® software, in both object and source code",
    "nemo": "The Licensee can include the Modified or unmodified Software in a code",
    "wrfchem_pnnl": "The MOSAIC source code is intended for research and educational",
}
MARKER = "WOOF-OPTIN:"


def _require_source_tree():
    if not PYPROJECT.is_file() or not OPTIONAL.is_dir():
        pytest.skip("the opt-in wheel gate needs the source tree with optional/")


def _projects() -> dict[str, dict]:
    sys.path.insert(0, str(REPO_ROOT / "tools"))
    try:
        import metal_woof_project
    finally:
        sys.path.pop(0)
    text = PYPROJECT.read_text(encoding="utf-8")
    return {"recast-woof": tomllib.loads(text),
            "metal-woof": tomllib.loads(metal_woof_project.rewrite(text, "9.9.9"))}


def _shipped(project: dict, monkeypatch) -> set[Path]:
    import setuptools
    from setuptools.command.build_py import build_py
    from setuptools.config import expand

    config = project["tool"]["setuptools"]
    find = config["packages"]["find"]
    packages = expand.find_packages(include=find.get("include", ["*"]),
                                    exclude=find.get("exclude", []),
                                    root_dir=str(REPO_ROOT))
    distribution = setuptools.dist.Distribution({
        "name": project["project"]["name"], "packages": packages,
        "package_data": config.get("package-data", {}),
        "exclude_package_data": config.get("exclude-package-data", {})})
    command = build_py(distribution)
    command.finalize_options()
    monkeypatch.chdir(REPO_ROOT)
    files: set[Path] = set()
    for _package, src_dir, _build, names in command.get_data_files_without_manifest():
        files.update((REPO_ROOT / src_dir / name).resolve() for name in names)
    for package in packages:
        files.update(p.resolve() for p in (REPO_ROOT / package.replace(".", "/")).glob("*.py"))
    for pattern in project["project"].get("license-files", []):
        files.update(Path(p).resolve() for p in glob.glob(str(REPO_ROOT / pattern)))
    return files


@pytest.mark.parametrize("wheel", ["recast-woof", "metal-woof"])
def test_no_opt_in_file_or_licence_reaches_the_wheel(wheel, monkeypatch):
    _require_source_tree()
    files = _shipped(_projects()[wheel], monkeypatch)
    assert files, "setuptools selected nothing; the measurement is broken"
    optional = OPTIONAL.resolve()
    under = sorted(str(f) for f in files if optional in f.parents)
    assert not under, f"{wheel} would ship opt-in files: {under[:5]}"
    carrying = []
    for f in files:
        if f.suffix in (".npz", ".bin", ".xz", ".gz", ".png", ".so", ".dylib", ".zip"):
            continue
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if any(line.lstrip("/#! ").startswith(MARKER) for line in text.splitlines()[:3]):
            carrying.append(str(f))
        elif any(s in text for s in LICENCE_SENTENCES.values()):
            carrying.append(str(f))
    assert not carrying, f"{wheel} would ship opt-in code or licence text: {carrying[:5]}"


def test_every_opt_in_is_marked_and_carries_its_licence():
    _require_source_tree()
    from woof.optin import OPT_INS

    for name in OPT_INS:
        root = OPTIONAL / name
        assert (root / "LICENSE").is_file() and (root / "NOTICE.md").is_file(), name
        licence = (root / "LICENSE").read_bytes().decode("utf-8", errors="replace")
        assert LICENCE_SENTENCES[name] in " ".join(licence.split()), name
        for code in [*root.rglob("*.cu"), *root.rglob("*.py")]:
            if "oracle" in code.parts:
                continue
            head = code.read_text(encoding="utf-8").splitlines()[:3]
            assert any(MARKER in line for line in head), code


def test_the_loader_refuses_by_name_without_the_switch(monkeypatch):
    from woof import optin

    for name, (variable, _package, _licence) in optin.OPT_INS.items():
        monkeypatch.delenv(variable, raising=False)
        with pytest.raises(optin.OptInUnavailable, match=variable):
            optin.load(name)
