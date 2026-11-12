"""The gpuwm-hex helper crates vendored in rust/ (rust/PROVENANCE.md).

Host-only: the copy is whole and licensed, the doors look where the build
script puts the outputs, and no wheel picks the crates up.
"""
from __future__ import annotations

import fnmatch
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUST = REPO / "rust"

#: Every file of the two crates in gpuwm_hex-0.4.14.tar.gz (rust/PROVENANCE.md).
VENDORED = {
    "host-prep": ("Cargo.lock", "Cargo.toml", "README.md", "src/main.rs"),
    "physics-tables": (".gitignore", "Cargo.lock", "Cargo.toml", "src/bin/ruc_table_decode.rs",
                       "src/bin/rw_mpas_rrtmg_lw_tables.rs", "src/bin/rw_mpas_rrtmg_sw_tables.rs",
                       "src/bin/table_decode.rs", "src/bin/table_schema.rs", "src/bin/tempo_table_compute.rs",
                       "src/bin/tempo_table_init.rs", "src/lookup_decode.rs", "src/own_gamma.rs",
                       "src/rrtmg_sw_tables.rs"),
}


def _files(crate: Path) -> set[str]:
    return {str(path.relative_to(crate)) for path in crate.rglob("*")
            if path.is_file() and "target" not in path.relative_to(crate).parts}


def test_the_vendored_crates_are_whole_and_carry_their_licence():
    for crate, files in VENDORED.items():
        assert _files(RUST / crate) == set(files), crate
    provenance = (RUST / "PROVENANCE.md").read_text()
    assert "c47368e24d63c4b8fc69845b530938164957c4ec12853ac83bbc3c9a8c4f2db5" in provenance
    assert (RUST / "LICENSE-gpuwm-hex").read_text().lstrip().startswith("Apache License")
    notice = (RUST / "NOTICE-gpuwm-hex").read_text()
    assert "hex_contract_hostprep (rust/host-prep): project code, Apache-2.0." in notice


def test_the_crates_use_the_standard_library_only():
    for crate in VENDORED:
        manifest = tomllib.loads((RUST / crate / "Cargo.toml").read_text())
        assert manifest["package"]["license"] == "Apache-2.0"
        assert not manifest.get("dependencies"), crate
        lock = tomllib.loads((RUST / crate / "Cargo.lock").read_text())
        assert [row["name"] for row in lock["package"]] == [manifest["package"]["name"]]


def test_the_doors_look_where_the_build_script_writes():
    from woof.hex.mpasphys import resources_build

    assert resources_build._source_crate() == RUST / "physics-tables" / "Cargo.toml"
    source = (REPO / "woof" / "hex" / "drivers" / "contract_hostprep.py").read_text()
    assert 'root / "rust/host-prep/target/release/hex_contract_hostprep"' in source
    script = (REPO / "tools" / "build_hex_helpers.sh").read_text()
    for needle in ('rust/$crate/Cargo.toml', "host-prep physics-tables", "--locked",
                   "--bin rw_mpas_physics_pool"):
        assert needle in script
    built = resources_build._engine_pool_reader()
    assert built is None or built == REPO / "tools" / "rustwx" / "target" / "release" / "rw_mpas_physics_pool"


def test_no_wheel_carries_the_crates_or_their_builds():
    pyproject = tomllib.loads((REPO / "pyproject.toml").read_text())
    setuptools = pyproject["tool"]["setuptools"]
    include = setuptools["packages"]["find"]["include"]
    assert not any(fnmatch.fnmatch("rust", pattern) for pattern in include)
    assert not any(key == "rust" or key.startswith("rust.") for key in setuptools.get("package-data", {}))
    assert "*/target/" in (RUST / ".gitignore").read_text().splitlines()
