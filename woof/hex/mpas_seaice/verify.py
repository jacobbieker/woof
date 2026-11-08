"""``woof hex seaice verify``: grade every ported MPAS-Seaice piece against its oracle.

Each piece runs on the active backend (CUDA, or Metal through the facade) over
the committed oracle fixture, and every output word must be bit-identical.
The core's registry status stays ``stub`` until every piece is ported;
docs/mpas-coupled-roadmap.md lists what remains.
"""
from __future__ import annotations


def verify() -> dict:
    from woof import backend
    from woof.hex.coupled import oracle, registry
    from woof.verify import mpas_seaice_evp_oracle as evp_oracle

    from . import evp_v842

    if not (evp_oracle.FIXTURE_DIR / "oracle-r4.npz").is_file():
        raise FileNotFoundError(
            f"the MPAS-Seaice oracle fixture is not here ({evp_oracle.FIXTURE_DIR}); it ships with the "
            "source tree (tests/data/oracles), not the wheel: run verify from a checkout")
    report = evp_oracle.measure()
    mismatched = sum(n for fields in report.values() for n in fields.values())
    piece = {
        "name": "evp_variational_internal_stress",
        "source": list(evp_v842.SOURCE_FILES),
        "fixture": str(evp_oracle.FIXTURE_DIR.relative_to(evp_oracle.FIXTURE_DIR.parents[3])),
        "cases": sorted(report),
        "mismatched_words": report,
        "passed": mismatched == 0,
    }
    return {
        "model": "seaice",
        "status": registry.component("seaice").status,
        "backend": backend.active(),
        "oracle": {"repo": oracle.ORACLE_REPO, "tag": oracle.ORACLE_TAG, "pin": oracle.ORACLE_PIN},
        "pieces": [piece],
        "remaining": "docs/mpas-coupled-roadmap.md",
        "passed": piece["passed"],
    }
