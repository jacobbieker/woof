"""``woof hex ice verify``: grade every ported MALI piece against its oracle.

Each piece runs on the active backend (CUDA, or Metal through the facade) over
the committed oracle fixture, and every output word must be bit-identical.
The core's registry status stays ``stub`` until every piece is ported;
docs/mpas-coupled-roadmap.md lists what remains.
"""
from __future__ import annotations


def verify() -> dict:
    from woof import backend
    from woof.hex.coupled import oracle, registry
    from woof.verify import mali_geometry_oracle as geometry_oracle
    from woof.verify import mali_sia_oracle as sia_oracle
    from woof.verify import mali_thermal_oracle as thermal_oracle

    from . import geometry_v842, sia_v842, thermal_init_v842

    pieces = []
    for name, module, oracle_module in (("sia_velocity_solver", sia_v842, sia_oracle),
                                        ("geometry_diagnostics", geometry_v842, geometry_oracle),
                                        ("thermal_init_and_conversions", thermal_init_v842, thermal_oracle)):
        if not (oracle_module.FIXTURE_DIR / "oracle-r4.npz").is_file():
            raise FileNotFoundError(
                f"the MALI oracle fixture is not here ({oracle_module.FIXTURE_DIR}); it ships with the source "
                "tree (tests/data/oracles), not the wheel: run verify from a checkout")
        report = oracle_module.measure()
        mismatched = sum(n for fields in report.values() for n in fields.values())
        pieces.append({
            "name": name,
            "source": list(module.SOURCE_FILES),
            "fixture": str(oracle_module.FIXTURE_DIR.relative_to(oracle_module.FIXTURE_DIR.parents[3])),
            "cases": sorted(report),
            "mismatched_words": report,
            "passed": mismatched == 0,
        })
    return {
        "model": "ice",
        "status": registry.component("landice").status,
        "backend": backend.active(),
        "oracle": {"repo": oracle.ORACLE_REPO, "tag": oracle.ORACLE_TAG, "pin": oracle.ORACLE_PIN},
        "pieces": pieces,
        "refused": {"config_velocity_solver": list(sia_v842.EXTERNAL_DYCORES) + ["simple", "none"]},
        "remaining": "docs/mpas-coupled-roadmap.md",
        "passed": all(p["passed"] for p in pieces),
    }
