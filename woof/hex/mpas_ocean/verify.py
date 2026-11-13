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
    from woof.verify import (mpas_ocean_diagnostics_oracle, mpas_ocean_eos_oracle, mpas_ocean_pgrad_hmix_oracle,
                             mpas_ocean_tend_oracle, mpas_ocean_tendency_oracle, mpas_ocean_tracer_advection_oracle)

    from . import diagnostics_v842, eos_v842, forcing_v842, pgrad_hmix_v842, tend_v842, tracer_advection_v842

    pieces = []
    for label, oracle_module, port in (("tendency_terms", mpas_ocean_tend_oracle, tend_v842),
                                       ("pressure_gradient_hmix", mpas_ocean_pgrad_hmix_oracle, pgrad_hmix_v842),
                                       ("full_tendencies", mpas_ocean_tendency_oracle, forcing_v842),
                                       ("equation_of_state", mpas_ocean_eos_oracle, eos_v842),
                                       ("diagnostic_solve", mpas_ocean_diagnostics_oracle, diagnostics_v842),
                                       ("tracer_advection", mpas_ocean_tracer_advection_oracle,
                                        tracer_advection_v842)):
        fixture_dir = oracle_module.FIXTURE_DIR
        for precision in oracle_module.PRECISIONS:
            if not (fixture_dir / f"oracle-{precision}.npz").is_file():
                raise FileNotFoundError(
                    f"the MPAS-Ocean oracle fixture is not here ({fixture_dir}); it ships with the "
                    "source tree (tests/data/oracles), not the wheel: run verify from a checkout")
            report = oracle_module.measure(precision)
            mismatched = sum(n for fields in report.values() for n in fields.values())
            pieces.append({
                "name": f"{label}_{precision}",
                "source": list(port.SOURCE_FILES),
                "fixture": str(fixture_dir.relative_to(fixture_dir.parents[3])),
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
