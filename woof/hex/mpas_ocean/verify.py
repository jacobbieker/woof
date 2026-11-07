"""``woof hex ocean verify``: grade every ported MPAS-Ocean piece against its oracle.

Each piece runs on the active backend (CUDA, or Metal through the facade) over
the committed oracle fixture, and every output word must be bit-identical.
The core's registry status stays ``stub`` until every piece is ported;
docs/mpas-coupled-roadmap.md lists what remains.
"""
from __future__ import annotations


def verify() -> dict:
    from woof import backend
    from woof.hex.coupled import oracle, registry
    from woof.verify import mpas_ocean_tend_oracle as tend_oracle

    from . import tend_v842

    pieces = []
    for precision in tend_oracle.PRECISIONS:
        if not (tend_oracle.FIXTURE_DIR / f"oracle-{precision}.npz").is_file():
            raise FileNotFoundError(
                f"the MPAS-Ocean oracle fixture is not here ({tend_oracle.FIXTURE_DIR}); it ships with the "
                "source tree (tests/data/oracles), not the wheel: run verify from a checkout")
        report = tend_oracle.measure(precision)
        mismatched = sum(n for fields in report.values() for n in fields.values())
        pieces.append({
            "name": f"tendency_terms_{precision}",
            "source": list(tend_v842.SOURCE_FILES),
            "fixture": str(tend_oracle.FIXTURE_DIR.relative_to(tend_oracle.FIXTURE_DIR.parents[3])),
            "cases": sorted(report),
            "mismatched_words": report,
            "passed": mismatched == 0,
        })
    return {
        "model": "ocean",
        "status": registry.component("ocean").status,
        "backend": backend.active(),
        "oracle": {"repo": oracle.ORACLE_REPO, "tag": oracle.ORACLE_TAG, "pin": oracle.ORACLE_PIN},
        "pieces": pieces,
        "remaining": "docs/mpas-coupled-roadmap.md",
        "passed": all(p["passed"] for p in pieces),
    }
