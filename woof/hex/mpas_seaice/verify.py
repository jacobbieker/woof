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
    from woof.verify import mpas_seaice_presubcycle_oracle as presubcycle_oracle
    from woof.verify import mpas_seaice_remap_checks_oracle as remap_checks_oracle
    from woof.verify import mpas_seaice_remap_oracle as remap_oracle
    from woof.verify import mpas_seaice_remap_sphere_oracle as remap_sphere_oracle
    from woof.verify import mpas_seaice_step_oracle as step_oracle
    from woof.verify import mpas_seaice_step_remap_oracle as step_remap_oracle
    from woof.verify import mpas_seaice_step_remap_sphere_oracle as step_remap_sphere_oracle
    from woof.verify import mpas_seaice_velocity_oracle as velocity_oracle

    from . import evp_v842, presubcycle_v842, remap_v842, timestep_v842, velocity_v842

    for fixture in (evp_oracle.FIXTURE_DIR / "oracle-r4.npz", velocity_oracle.FIXTURE_DIR / "oracle-r4.npz",
                    presubcycle_oracle.FIXTURE_DIR / "oracle-r4.npz", step_oracle.FIXTURE_DIR / "oracle-r4.npz",
                    remap_oracle.FIXTURE_DIR / "oracle-r4.npz", step_remap_oracle.FIXTURE_DIR / "oracle-r4.npz",
                    remap_sphere_oracle.FIXTURE_DIR / "oracle-r4.npz",
                    step_remap_sphere_oracle.FIXTURE_DIR / "oracle-r4.npz",
                    remap_checks_oracle.FIXTURE_DIR / "oracle-r4.npz"):
        if not fixture.is_file():
            raise FileNotFoundError(
                f"the MPAS-Seaice oracle fixture is not here ({fixture.parent}); it ships with the "
                "source tree (tests/data/oracles), not the wheel: run verify from a checkout")

    def piece(name, module, oracle_module, report):
        source = getattr(module, "SOURCE_FILES", ())
        mismatched = sum(n for fields in report.values() for n in fields.values())
        return {
            "name": name,
            "source": list(source),
            "fixture": str(oracle_module.FIXTURE_DIR.relative_to(oracle_module.FIXTURE_DIR.parents[3])),
            "cases": sorted(report),
            "mismatched_words": report,
            "passed": mismatched == 0,
        }

    pieces = [piece("evp_variational_internal_stress", evp_v842, evp_oracle, evp_oracle.measure())]
    for precision in velocity_oracle.PRECISIONS:
        pieces.append(piece(f"evp_velocity_solve_{precision}", velocity_v842, velocity_oracle,
                            velocity_oracle.measure(precision)))
    for precision in presubcycle_oracle.PRECISIONS:
        pieces.append(piece(f"velocity_pre_subcycle_{precision}", presubcycle_v842, presubcycle_oracle,
                            presubcycle_oracle.measure(precision)))
    for precision in step_oracle.PRECISIONS:
        pieces.append(piece(f"time_step_dynamics_{precision}", timestep_v842, step_oracle,
                            step_oracle.measure(precision)))
    for precision in remap_oracle.PRECISIONS:
        pieces.append(piece(f"incremental_remap_{precision}", remap_v842, remap_oracle,
                            remap_oracle.measure(precision)))
    for precision in step_remap_oracle.PRECISIONS:
        pieces.append(piece(f"time_step_incremental_remap_{precision}", timestep_v842, step_remap_oracle,
                            step_remap_oracle.measure(precision)))
    for precision in remap_sphere_oracle.PRECISIONS:
        pieces.append(piece(f"incremental_remap_sphere_{precision}", remap_v842, remap_sphere_oracle,
                            remap_sphere_oracle.measure(precision)))
    for precision in step_remap_sphere_oracle.PRECISIONS:
        pieces.append(piece(f"time_step_incremental_remap_sphere_{precision}", timestep_v842,
                            step_remap_sphere_oracle, step_remap_sphere_oracle.measure(precision)))
    for precision in remap_checks_oracle.PRECISIONS:
        pieces.append(piece(f"incremental_remap_checks_{precision}", remap_v842, remap_checks_oracle,
                            remap_checks_oracle.measure(precision)))
    return {
        "model": "seaice",
        "status": registry.component("seaice").status,
        "backend": backend.active(),
        "oracle": {"repo": oracle.ORACLE_REPO, "tag": oracle.ORACLE_TAG, "pin": oracle.ORACLE_PIN},
        "pieces": pieces,
        "remaining": "docs/mpas-coupled-roadmap.md",
        "passed": all(p["passed"] for p in pieces),
    }
