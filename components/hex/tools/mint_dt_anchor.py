#!/usr/bin/env python3
"""Mint and verify a model-timestep anchor for the frozen v8.4.1 lane.

``woof.hex.dt_admission`` refuses any ``config_dt`` holding no anchor.  This
is the runnable procedure that mints one, and the verifier that decides
whether a registered row is telling the truth.

An anchor has two halves and they cost very different things.

**The schedule receipt (host only, seconds).**  Everything a timestep can
get wrong before it ever reaches a card: the physics cadence step counts,
the Grell-Freitas ``cudt == dt`` law, the RK schedule's shape against the
proven 120 s schedule, the WSM6 minor-loop split, and clock closure over the
run length.  ``--dt`` mints it; ``--verify`` re-derives every registered
row's receipt and compares it to the JSON the row names, so a row cannot
point at a file that says something else.

**The integration anchor (one card, hours).**  Two forecasts of the same
case at this timestep whose history is byte-identical under masked digests,
finite at every step.  ``--integration-plan`` prints the exact commands and
the measured cost; this tool does not run them, because a forecast is the
forecast door's job and duplicating it here would create a second execution
path for the same thing.

**The mutation control.**  ``--mutation-control`` proves the verifier has
teeth in BOTH directions: it must pass the real 120 s row and it must fail
every fabricated variant of it -- a receipt file that does not exist, a
receipt whose dt does not match the row, a receipt whose cadence counts were
edited, and a row whose declared cadences contradict its own receipt.  A
verifier that only ever passes is not evidence of anything.

**The regional plan.**  ``--regional-plan --dt 0.5 --mesh ROW --grid G``
prints every part of a mint on a limited-area cull -- the schedule receipt,
the integration pair, the physics-health control and the regional forecast
mint pair -- with the exact command lines, the inputs each reads and the
outputs each writes, after checking every host-checkable rule against the
cull's own files (Courant from its ``min(dcEdge)``, radiation divisibility,
binary64 clock closure, an exact millisecond class key, a real boundary zone,
the level count).  It runs nothing on a card.  Sub-second timesteps are the
case it was written for: a 100 m corridor's Courant limit is under a second,
so no anchored timestep fits it and the physics-health control is a finer
unanchored step.  Every plan shows the candidate-mint authorization by NAME
(``<CANDIDATE_MINT_AUTHORIZATION>``); the sentence is typed by the person
who rules, and this tool never prints, stores or echoes it.  The procedure is
written up in ``components/hex/docs/sub-second-dt-mint.md``.

Nothing here registers anything.  Registering a second anchor moves the
frozen lane off its proven timestep and that is a ruling, not a tool run.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from woof.hex import dt_admission  # noqa: E402
from woof.hex.dt_admission import DtAdmissionError, DtAnchor  # noqa: E402
from woof.hex.vertical_spec import hex_column_level_ceiling  # noqa: E402

SCHEMA = "gpuwm-hex.dt-anchor-mint/v1"

#: The archived 120 s constants this harness must reproduce exactly, read
#: from where the frozen proofs already record them.  This is the harness's
#: KNOWN-ANSWER arm: an instrument that cannot reproduce the one timestep
#: this project has actually proven is not an instrument.
ARCHIVED_120S = {
    "scalar_stage_timesteps": (40.0, 60.0, 120.0),
    "dynamics_stage_timesteps": (40.0 / 3.0, 20.0, 40.0),
    "dynamics_stage_acoustic_steps": (1, 3, 6),
    "stepra": 5,
    "stepbl": 1,
    "stepcu": 1,
    "minor_loops": 1,
    "source": (
        "tools/run_cuda_v841_real_x4.py::EXPECTED_SCALAR_STAGE_TIMESTEPS / "
        "EXPECTED_DYNAMICS_STAGE_TIMESTEPS / EXPECTED_STAGE_ACOUSTIC_STEPS, "
        "and tools/compare_v841_compiled_endpoint.py's dynamics/scalar stage "
        "tables"
    ),
}


class MintError(RuntimeError):
    """The harness refuses to mint or to certify."""


#: What every plan this tool prints shows where the candidate-mint
#: authorization goes.  The sentence itself is never printed, stored or
#: echoed by this tool: the person who rules on the anchor types it.
AUTHORIZATION_PLACEHOLDER = "<CANDIDATE_MINT_AUTHORIZATION>"

#: The environment variable a caller may hand the authorization through
#: (``--authorization-env``) instead of argv, so it stays out of the process
#: list and out of every log that records a command line.  The campaign
#: hands it to its candidate arms this way.
AUTHORIZATION_ENVIRONMENT = "WOOF_HEX_CANDIDATE_MINT_AUTHORIZATION"


def redact_authorization(value: object) -> str:
    """``str(value)`` with the authorization sentence replaced by its name.

    The timestep gate's own refusal quotes the sentence it wanted, and a
    candidate row's ``basis`` names it.  This tool prints both as JSON that
    ends up in logs and receipts, so it never copies the sentence through.
    """

    text = str(value)
    sentence = dt_admission.CANDIDATE_MINT_AUTHORIZATION
    return text.replace(repr(sentence), AUTHORIZATION_PLACEHOLDER).replace(
        sentence, AUTHORIZATION_PLACEHOLDER
    )


#: Top-level directories an anchor may cite a path inside.  A citation that
#: does not start with one of these is prose (a native-run description, a
#: ``module::SYMBOL`` reference) and is deliberately NOT checked for
#: existence -- checking it would turn a sentence into a missing file.
_EVIDENCE_ROOTS = ("evidence/", "tools/", "src/", "docs/", "verification/", "oracle/")


def _repository_paths(citation: str) -> list[str]:
    """The repository paths inside one comma-separated evidence citation."""

    paths: list[str] = []
    for part in citation.split(","):
        entry = part.strip().rstrip(".").split(" ")[0].split("::")[0]
        if entry.startswith(_EVIDENCE_ROOTS):
            paths.append(entry)
    return paths


def mint_schedule_receipt(
    dt_seconds: float,
    *,
    radiation_seconds: float = dt_admission.RADIATION_CADENCE_SECONDS,
    surface_pbl_seconds: float | None = None,
    cumulus_seconds: float | None = None,
    cumulus_scheme: str | None = "gf",
    run_steps: int = 720,
) -> dict[str, Any]:
    """Mint one timestep's schedule receipt, with the known-answer arm run."""

    receipt = dt_admission.schedule_receipt(
        dt_seconds,
        radiation_seconds=radiation_seconds,
        surface_pbl_seconds=surface_pbl_seconds,
        cumulus_seconds=cumulus_seconds,
        cumulus_scheme=cumulus_scheme,
        run_steps=run_steps,
    )
    receipt["schema_mint"] = SCHEMA
    receipt["known_answer_control"] = known_answer_control()
    return receipt


def known_answer_control() -> dict[str, Any]:
    """Re-derive the archived 120 s constants and compare them exactly.

    Runs on every mint, not only on ``--dt 120``: the instrument that
    measures an unproven timestep is the same instrument, so it declares its
    calibration in the receipt of every timestep it mints.
    """

    receipt = dt_admission.schedule_receipt(dt_admission.PROVEN_DT_SECONDS)
    checks = {
        "scalar_stage_timesteps": (
            tuple(receipt["rk_schedule"]["scalar_stage_timesteps"])
            == ARCHIVED_120S["scalar_stage_timesteps"]
        ),
        "dynamics_stage_timesteps": (
            tuple(receipt["rk_schedule"]["dynamics_stage_timesteps"])
            == ARCHIVED_120S["dynamics_stage_timesteps"]
        ),
        "dynamics_stage_acoustic_steps": (
            tuple(receipt["rk_schedule"]["dynamics_stage_acoustic_steps"])
            == ARCHIVED_120S["dynamics_stage_acoustic_steps"]
        ),
        "stepra": receipt["cadences"]["stepra"] == ARCHIVED_120S["stepra"],
        "stepbl": receipt["cadences"]["stepbl"] == ARCHIVED_120S["stepbl"],
        "stepcu": receipt["cadences"]["stepcu"] == ARCHIVED_120S["stepcu"],
        "minor_loops": receipt["wsm6"]["minor_loops"] == ARCHIVED_120S["minor_loops"],
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise MintError(
            "the schedule instrument does not reproduce the archived 120 s "
            f"constants: {failed} disagree with {ARCHIVED_120S['source']}.  "
            "Nothing it says about an unproven timestep can be trusted"
        )
    return {
        "reproduces_archived_120s": True,
        "checks": checks,
        "source": ARCHIVED_120S["source"],
    }


def verify_anchor(anchor: DtAnchor, *, root: Path = ROOT) -> dict[str, Any]:
    """Certify one registered anchor against the evidence it names.

    Refuses, by name, when: a named evidence path is absent; the schedule
    receipt on disk is not valid JSON; its timestep is not the row's; its
    cadences are not the row's; or re-deriving the receipt from the row's own
    declarations does not reproduce the stored numbers.
    """

    findings: list[str] = []
    if anchor.timestep_evidence is not None:
        return {
            "dt_seconds": anchor.dt_seconds,
            "certified": False,
            "findings": [
                f"dt={anchor.dt_seconds:g} s is an {anchor.timestep_evidence} "
                f"record from the opt-in experimental lane, not a table row: "
                f"it names no schedule receipt on disk, no integration pair "
                f"and no physics band, and is never an anchor"
            ],
        }
    if anchor.admitted_on.startswith("CANDIDATE"):
        return {
            "dt_seconds": anchor.dt_seconds,
            "certified": False,
            "findings": [
                f"dt={anchor.dt_seconds:g} s is a CANDIDATE row admitted only "
                f"for the duration of a mint run ({anchor.basis}).  A "
                f"candidate is never an anchor: it names no schedule receipt "
                f"on disk and no integration pair, and certifying one would "
                f"be certifying the run that is still trying to earn it"
            ],
        }
    receipt_path = root / anchor.schedule_receipt
    if not receipt_path.exists():
        findings.append(
            f"dt={anchor.dt_seconds:g} s names a schedule receipt that is not "
            f"in the tree: {anchor.schedule_receipt}"
        )
        stored: dict[str, Any] | None = None
    elif receipt_path.is_dir():
        findings.append(
            f"dt={anchor.dt_seconds:g} s names a DIRECTORY as its schedule "
            f"receipt: {anchor.schedule_receipt}; an anchor names the file "
            f"whose numbers can be compared, not the folder around it"
        )
        stored = None
    else:
        try:
            stored = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            findings.append(
                f"dt={anchor.dt_seconds:g} s names a schedule receipt that "
                f"does not read as JSON ({error})"
            )
            stored = None

    if stored is not None:
        if float(stored.get("dt_seconds", float("nan"))) != float(anchor.dt_seconds):
            findings.append(
                f"dt={anchor.dt_seconds:g} s names a schedule receipt minted "
                f"at dt={stored.get('dt_seconds')!r}: the row and its evidence "
                f"are about different timesteps"
            )
        cadences = stored.get("cadences") or {}
        for field, declared in (
            ("radiation_seconds", anchor.radiation_seconds),
            ("surface_pbl_seconds", anchor.surface_pbl_seconds),
            ("cumulus_seconds", anchor.cumulus_seconds),
        ):
            recorded = cadences.get(field)
            same = (recorded is None and declared is None) or (
                recorded is not None
                and declared is not None
                and float(recorded) == float(declared)
            )
            if not same:
                findings.append(
                    f"dt={anchor.dt_seconds:g} s declares {field}={declared!r} "
                    f"but its schedule receipt records {recorded!r}"
                )
        try:
            rederived = dt_admission.schedule_receipt(
                anchor.dt_seconds,
                radiation_seconds=anchor.radiation_seconds,
                surface_pbl_seconds=anchor.surface_pbl_seconds,
                cumulus_seconds=anchor.cumulus_seconds,
                cumulus_scheme=anchor.cumulus_scheme,
                run_steps=int(
                    (stored.get("clock_closure") or {}).get("run_steps", 720)
                ),
            )
        except DtAdmissionError as error:
            findings.append(
                f"dt={anchor.dt_seconds:g} s cannot be re-derived at its own "
                f"declared cadences: {error}"
            )
        else:
            for section in ("cadences", "rk_schedule", "wsm6", "clock_closure"):
                if stored.get(section) != rederived[section]:
                    findings.append(
                        f"dt={anchor.dt_seconds:g} s: the stored {section!r} "
                        f"block does not reproduce -- the receipt on disk is "
                        f"not what this timestep actually derives"
                    )

    for label, value in (
        ("integration anchor", anchor.integration_anchor),
        ("native reference", anchor.native_reference),
    ):
        if value is None:
            continue
        for entry in _repository_paths(value):
            if not (root / entry).exists():
                findings.append(
                    f"dt={anchor.dt_seconds:g} s names a {label} path that is "
                    f"not in the tree: {entry}"
                )

    if not anchor.card or not anchor.admitted_on or not anchor.basis:
        findings.append(
            f"dt={anchor.dt_seconds:g} s is registered without a card, a date "
            f"or a basis; that is a switch, not an anchor"
        )

    return {
        "dt_seconds": anchor.dt_seconds,
        "certified": not findings,
        "findings": findings,
    }


def verify_registry(*, root: Path = ROOT) -> dict[str, Any]:
    """Certify every registered anchor."""

    results = [
        verify_anchor(anchor, root=root)
        for anchor in sorted(
            dt_admission.ADMITTED_TIMESTEPS.values(), key=lambda a: a.dt_seconds
        )
    ]
    return {
        "schema": SCHEMA,
        "registered": len(results),
        "certified": all(result["certified"] for result in results),
        "anchors": results,
    }


def _fabricate(anchor: DtAnchor, **changes: Any) -> DtAnchor:
    import dataclasses

    return dataclasses.replace(anchor, **changes)


def unanchored_probe_dt() -> float:
    """A timestep that holds no anchor and whose host half mints clean.

    DERIVED, never hardcoded.  The mutation control needs an unanchored
    timestep for its candidate arm, and any literal it named would go stale
    the moment that value was earned -- which is exactly what happened on
    2026-08-26: the control named 20.0, four anchors were minted, and the
    control's own candidate arm started refusing with "already holds an
    anchor".  A control that breaks when the thing it guards succeeds is a
    control nobody can keep.

    So it is searched: exact divisors of the radiation cadence, smallest
    first, taking the first that is unanchored and can mint a schedule
    receipt.  If every divisor in range is anchored the harness says so
    rather than fabricating one.
    """

    cadence = dt_admission.RADIATION_CADENCE_SECONDS
    for count in range(2, 4000):
        candidate = cadence / count
        if dt_admission.admitted_timestep(candidate) is not None:
            continue
        try:
            dt_admission.schedule_receipt(candidate, run_steps=1)
        except DtAdmissionError:
            continue
        return candidate
    raise MintError(
        "no unanchored timestep dividing the "
        f"{cadence:g} s radiation cadence could be found, so the mutation "
        "control has no candidate arm to run.  Either every divisor is "
        "anchored -- which would be a finding in itself -- or the schedule "
        "instrument is refusing everything"
    )


def mutation_control(*, root: Path = ROOT) -> dict[str, Any]:
    """Prove the verifier fails on fabricated evidence, and passes on real.

    Six arms.  The first must certify; the other five must not.  A verifier
    that answers "certified" to all six is measuring nothing, which is the
    failure mode this control exists to catch.
    """

    truth = dt_admission.admitted_timestep(dt_admission.PROVEN_DT_SECONDS)
    if truth is None:  # pragma: no cover - the registry lost its own anchor
        raise MintError(
            "the registry holds no anchor at the proven timestep, so the "
            "mutation control has no true arm to run"
        )

    arms: list[dict[str, Any]] = []

    def _arm(name: str, anchor: DtAnchor, expect_certified: bool) -> None:
        result = verify_anchor(anchor, root=root)
        arms.append(
            {
                "arm": name,
                "expected_certified": expect_certified,
                "certified": result["certified"],
                "agrees": result["certified"] is expect_certified,
                "findings": result["findings"],
            }
        )

    _arm("registered-truth", truth, True)
    _arm(
        "receipt-absent",
        _fabricate(truth, schedule_receipt="evidence/does-not-exist.json"),
        False,
    )
    _arm(
        "receipt-is-a-directory",
        _fabricate(truth, schedule_receipt="evidence/dt-admission-20260826"),
        False,
    )
    _arm(
        "row-claims-a-timestep-its-receipt-does-not",
        _fabricate(truth, dt_seconds=60.0, surface_pbl_seconds=60.0, cumulus_seconds=60.0),
        False,
    )
    _arm(
        "row-claims-a-cadence-its-receipt-does-not",
        _fabricate(truth, surface_pbl_seconds=600.0),
        False,
    )
    _arm(
        "integration-anchor-path-absent",
        _fabricate(truth, integration_anchor="evidence/no-such-campaign"),
        False,
    )
    probe = unanchored_probe_dt()
    with dt_admission.candidate_mint(
        probe,
        authorization=dt_admission.CANDIDATE_MINT_AUTHORIZATION,
        card="mutation control, no card",
    ) as candidate:
        _arm("candidate-row-is-never-an-anchor", candidate, False)

    disagreements = [arm["arm"] for arm in arms if not arm["agrees"]]
    return {
        "schema": SCHEMA,
        "unanchored_probe_dt_seconds": probe,
        "arms": arms,
        "has_teeth_both_directions": not disagreements,
        "disagreements": disagreements,
    }


def run_candidate_forecast(
    dt_seconds: float,
    *,
    card: str,
    authorization: str,
    forecast_argv: list[str],
) -> int:
    """Run ONE forecast arm at an unanchored timestep, to mint its evidence.

    The gate and the mint are a chicken and egg -- the config refuses to
    build a forecast at a timestep with no anchor, and an anchor's
    integration half IS two forecasts at that timestep.  This is the only
    path through, it is the same shape ``regional_admission`` uses for its
    contract half, and every part of it is loud: the caller repeats
    :data:`woof.hex.dt_admission.CANDIDATE_MINT_AUTHORIZATION` verbatim, the
    schedule receipt must mint clean before a card is touched, the admitted
    row is stamped ``CANDIDATE-UNANCHORED`` and lands stamped that way in the
    run's own receipt, and the admission is withdrawn when the arm exits.

    Two arms of this, byte-identical under masked digests, are the evidence.
    Registering the anchor they earn is a ruling.
    """

    from woof.hex import convection_admission, mesh_row_candidate
    from woof.hex import pbl_cadence as pbl_cadence_module
    from woof.hex.cli import build_parser

    parser = build_parser()
    arguments = parser.parse_args(["forecast", *forecast_argv])
    mesh = getattr(arguments, "mesh", None)
    if not mesh:
        raise MintError(
            "a candidate forecast must name --mesh: the arm's timestep is "
            "declared by a registry row, and without one the door has no "
            "schedule to compute and the anchor has no hardware/mesh pair to "
            "record"
        )

    # An anchor certifies a CONFIGURATION at a timestep, so the candidate
    # admission has to be opened for the configuration the arm will actually
    # run.  Derived from the SAME two inputs the door derives it from -- the
    # row's declared spacing and this run's --convection request -- so the
    # mint and the door cannot open and check different doors.
    # Read through the door's OWN loader, so a row registered at runtime
    # ($WOOF_HEX_MESH_ROWS -- every corridor cull) is found exactly as the
    # door will find it, rather than through whichever copy of the binding
    # module happened to be imported first.
    from woof.hex import forecast_door

    registry = forecast_door.load_registry()
    row = registry.get(mesh)
    if row is None:
        raise MintError(
            f"a candidate forecast must name a REGISTERED mesh; {mesh!r} is "
            f"not one of {sorted(registry)}.  A regional cull is registered "
            f"with `woof hex register` and named to the door through "
            f"$WOOF_HEX_MESH_ROWS"
        )
    decision = convection_admission.convection_decision(
        nominal_dx_m=float(row.nominal_dx_m),
        requested=getattr(arguments, "convection", "auto"),
    )
    cumulus_scheme = decision["constructor_scheme"]

    # The surface/PBL cadence is the other half of the same configuration,
    # and it is derived from the SAME two inputs the door derives it from --
    # the row's declared timestep and this run's --pbl-cadence request -- so
    # the mint and the door cannot open and check different doors.  Without
    # it a held-cadence arm would open a candidate admission for the WELDED
    # configuration and then be refused by the door it just opened.
    pbl_decision = pbl_cadence_module.pbl_cadence_decision(
        dt_seconds=float(dt_seconds),
        requested=getattr(arguments, "pbl_cadence", "auto"),
    )
    surface_pbl_seconds = float(pbl_decision["surface_pbl_seconds"])

    with dt_admission.candidate_mint(
        dt_seconds,
        authorization=authorization,
        card=card,
        cumulus_scheme=cumulus_scheme,
        cumulus_seconds=None if cumulus_scheme is None else float(dt_seconds),
        surface_pbl_seconds=surface_pbl_seconds,
    ) as candidate:
        # The row half of the same chicken and egg: the door takes its
        # timestep from the registry row, so admitting the timestep is not
        # enough on its own.  Opened INSIDE the timestep admission and
        # refused outside it, so a row can never declare a timestep the
        # timestep gate has not been opened for by the same caller.
        with mesh_row_candidate.candidate_mesh_dt(
            mesh,
            dt_seconds,
            authorization=authorization,
            cumulus_scheme=cumulus_scheme,
            surface_pbl_seconds=surface_pbl_seconds,
        ):
            # Fail closed if the override did not reach the registry the
            # door will load.  Before the override moved to the END of the
            # binding module, a runtime row was added after it and silently
            # kept its registered timestep -- the arm would have run, and
            # been recorded, at a timestep that was not the candidate's.
            seen = forecast_door.load_registry().get(mesh)
            if seen is None or float(seen.dt_seconds) != float(dt_seconds):
                raise MintError(
                    f"the candidate override for {mesh!r} did not reach the "
                    f"registry the forecast door loads: it reads dt="
                    f"{None if seen is None else seen.dt_seconds!r} s, not the "
                    f"candidate {float(dt_seconds):g} s.  Running now would "
                    f"record evidence for a timestep that was not integrated"
                )
            print(
                redact_authorization(json.dumps(
                    {
                        "schema": SCHEMA,
                        "timestep_evidence": "experimental-unanchored",
                        "candidate_mint": candidate.as_dict(),
                        "candidate_mesh_row": {
                            "mesh": mesh,
                            "declared_dt_seconds": float(dt_seconds),
                            "restored_on_exit": True,
                        },
                        "convection": decision,
                        "pbl_cadence": pbl_decision,
                        "warning": (
                            "this arm runs at a timestep holding NO anchor; its "
                            "output is EVIDENCE FOR A RULING, not a forecast "
                            "anybody may quote as anchored"
                        ),
                    },
                    indent=2,
                    sort_keys=True,
                ))
            )
            return int(arguments.handler(arguments))


def integration_plan(dt_seconds: float, mesh: str, hours: float = 6.0) -> dict[str, Any]:
    """The runnable procedure for the half that needs a card.

    Printed rather than run: the forecast door is the execution path, and a
    second one here would be a second thing to keep true.
    """

    receipt = dt_admission.schedule_receipt(
        dt_seconds, run_steps=max(1, int(round(hours * 3600.0 / float(dt_seconds))))
    )
    steps = int(round(hours * 3600.0 / float(dt_seconds)))
    stride_minutes = receipt["cadences"]["stepra"] * float(dt_seconds) / 60.0
    forecast_arguments = (
        "--mesh {mesh} --grid <grid.nc> --static <static.nc> --init <init.nc> "
        "--init-source '<what produced the init>' --hours {hours:g} "
        "--history-every-minutes {stride:g} --out <arm>/ --scratch <arm>-scratch/"
    ).format(mesh=mesh, hours=hours, stride=max(1.0, stride_minutes))
    command = (
        "python tools/mint_dt_anchor.py --candidate-forecast --dt {dt:g} "
        "--card '<card>' --authorization '{auth}' -- {arguments}"
    ).format(
        dt=float(dt_seconds),
        # By NAME only: a printed plan is pasted into logs, chat and PRs, and
        # the sentence is something the person ruling types, not something a
        # tool hands out.
        auth=AUTHORIZATION_PLACEHOLDER,
        arguments=forecast_arguments,
    )
    return {
        "schema": SCHEMA,
        "dt_seconds": float(dt_seconds),
        "mesh": mesh,
        "hours": hours,
        "steps": steps,
        "arms": 2,
        "procedure": [
            "1. register the row's dt in woof.hex.drivers.mpas_mesh_binding.MESH_BINDINGS "
            "(table work; the row must already pass Courant, dual-edge and "
            "radiation-cadence admission)",
            "2. run arm A THROUGH THE CANDIDATE MINT, because the config "
            "refuses an unanchored timestep and that refusal is correct: "
            + command.replace("<arm>", "arm-a"),
            "3. run arm B, same inputs, separate process, same wrapper: "
            + command.replace("<arm>", "arm-b"),
            "4. compare every history file by masked SHA-256 (the netCDF "
            "file_id nonce NUL'd, as the native authority pins do); every "
            "file must match",
            "5. confirm the driver reported finite state at every one of the "
            f"{steps} steps and rc 0 in both arms",
            "6. mint this dt's schedule receipt into the evidence folder: "
            f"python tools/mint_dt_anchor.py --dt {float(dt_seconds):g} "
            "--out evidence/<campaign>/schedule-receipt-dt<dt>.json",
            "7. bring the pair and the receipt to a RULING.  Registering the "
            "anchor moves the frozen lane off its proven timestep and is not "
            "an agent's decision",
        ],
        "what_this_does_not_prove": [
            "byte-identity against native MPAS-A: the only native v8.4.1 "
            "reference this program holds was integrated at "
            f"{dt_admission.PROVEN_DT_SECONDS:g} s, so no other timestep can "
            "have that half without a fresh native run",
            "that the physics is RIGHT at this timestep: Grell-Freitas is "
            "called every step by WRF's own cudt=0 law, so a smaller dt calls "
            "it proportionally more often, and only obs-skill can referee that",
        ],
    }


#: The deepest column the full-physics forecast lane binds.  The driver's
#: ``N_LEVELS = 55`` is the native x4 column and NOT a limit:
#: ``mpas_mesh_binding.bind_mesh`` rebinds it to the registered row's own
#: count and refuses only past the lane's physics ceiling,
#: ``woof.hex.vertical_spec.hex_column_level_ceiling()`` for the frozen
#: WSM6 + YSU lane (80 while WSM6 compiles its 80-level tier; measured: an
#: 80-level 100 m Greenland cull bound and integrated on this lane).  Read
#: from the same function the bind calls, so the plan cannot quote a stale
#: copy; it is host-only and imports no CUDA.  Only 55-level classes are
#: MEASURED (``regional_admission.STANDARD_CLASS_LEVELS``): a deeper column
#: binds, but its class key holds no anchor until one is minted at it.
FULL_PHYSICS_MAX_LEVELS = hex_column_level_ceiling()

#: Earth radius MPAS scales a unit-sphere grid by (``mpas_constants``).
_MPAS_EARTH_RADIUS_M = 6_371_229.0

#: The number of steps every recorded regional forecast mint pair ran
#: (1,080 per arm; ``woof.hex.cuda_backend.regional_admission``).
REGIONAL_MINT_STEPS = 1080

#: Frames a regional mint arm aims to write.  Enough to see a drift grow,
#: few enough that the comparison stays a short read.
_REGIONAL_MINT_FRAMES = 12


def measure_regional_mesh(grid: Path, init: Path | None = None) -> dict[str, Any]:
    """The facts a regional mint plan needs, read off the cull's own files.

    Host-only and cheap: dimension sizes, ``min(dcEdge)`` (scaled to metres
    when the grid is published on the unit sphere), the boundary-zone width
    (the largest ``bdyMaskCell`` ring) and the vertical level count (from the
    init when given, since a bare grid carries none).
    """

    from netCDF4 import Dataset  # noqa: PLC0415 - host-only optional reader

    import numpy as np  # noqa: PLC0415

    with Dataset(str(grid)) as dataset:
        dataset.set_auto_maskandscale(False)
        dims = dataset.dimensions
        n_cells = int(len(dims["nCells"]))
        n_edges = int(len(dims["nEdges"])) if "nEdges" in dims else None
        if "dcEdge" not in dataset.variables:
            raise MintError(f"{grid} has no dcEdge; it is not an MPAS grid file")
        dc_edge = np.asarray(dataset.variables["dcEdge"][:], dtype=np.float64)
        radius = getattr(dataset, "sphere_radius", None)
        # A grid that omits sphere_radius is judged by magnitude: no real
        # MPAS edge on the Earth is under a metre, and every unit-sphere
        # edge is (the coarsest, 2 x pi / 3, is about 2.1).
        scaled = (
            float(radius) < 2.0
            if radius is not None
            else float(dc_edge.max()) < 3.0
        )
        finest = float(dc_edge.min()) * (_MPAS_EARTH_RADIUS_M if scaled else 1.0)
        zone_width = 0
        if "bdyMaskCell" in dataset.variables:
            zone_width = int(np.asarray(dataset.variables["bdyMaskCell"][:]).max())
        levels = int(len(dims["nVertLevels"])) if "nVertLevels" in dims else None
    if init is not None:
        with Dataset(str(init)) as dataset:
            if "nVertLevels" in dataset.dimensions:
                levels = int(len(dataset.dimensions["nVertLevels"]))
    return {
        "grid": str(grid),
        "init": None if init is None else str(init),
        "n_cells": n_cells,
        "n_edges": n_edges,
        "finest_edge_m": finest,
        "dc_edge_scaled_from_unit_sphere": bool(scaled),
        "boundary_zone_width": zone_width,
        "regional": zone_width > 0,
        "n_vert_levels": levels,
    }


def _whole_second_stride(dt_seconds: float, steps: int, frames: int) -> int:
    """The smallest step stride near ``steps / frames`` that names its frames.

    The stride must divide the run (the last frame lands on the last step)
    and land on a whole second (frames are named and stamped to the second).
    Falls back to the whole run -- two frames, start and end -- which always
    lands on a whole second when the run itself does.
    """

    target = max(1, steps // max(1, frames))
    for stride in range(target, steps + 1):
        if steps % stride:
            continue
        seconds = stride * float(dt_seconds)
        if seconds == round(seconds):
            return stride
    return steps


def _fmt(dt_seconds: float) -> str:
    return f"{float(dt_seconds):g}"


def _label(dt_seconds: float) -> str:
    return "dt" + _fmt(dt_seconds).replace(".", "p")


def regional_mint_plan(
    dt_seconds: float,
    *,
    mesh_row: str,
    facts: dict[str, Any],
    grid: str = "<grid.nc>",
    init: str = "<init.nc>",
    static: str = "<static.nc>",
    lbc_dir: str = "<lbc-dir>",
    start_time: str = "<YYYY-MM-DD_HH:MM:SS>",
    init_source: str = "<what produced the init>",
    gpuwm_checkout: str = "<arwen-checkout>",
    card: str = "<card>",
    work: str = "<work>",
    hours: float = 1.0,
    history_every_minutes: int = 10,
    convection: str = "off",
    mint_steps: int = REGIONAL_MINT_STEPS,
    control_dt_seconds: float | None = None,
) -> dict[str, Any]:
    """Everything a sub-second REGIONAL anchor mint would run, without running it.

    The four parts of a mint, each with the exact command line, the inputs it
    reads and the outputs it writes:

    1. the **schedule receipt** (host only -- minted here, in full);
    2. the **integration anchor**: two candidate arms through the forecast
       door on the cull, byte-identical under masked digests, finite at every
       step -- the only part that needs the candidate-mint authorization,
       shown here by name only;
    3. the **physics_health verdict**: the candidate's band against a
       control on the same card, mesh and init.  On a mesh no anchored
       timestep fits (every 100 m corridor: the smallest anchor is 5 s and
       the Courant limit is under a second) the control is a finer
       unanchored step, labelled ``step-halving-unanchored``;
    4. the **regional forecast mint pair**: the dry regional dycore twice
       through ``tools/run_cuda_regional_forecast.py --mint-anchor``, plus
       the cull's geometry contract deck, and the class key the pair would
       earn.

    Every host-checkable rule is checked HERE and a failure is a blocker by
    name: Courant (the same versioned 125 m/s, 0.9 policy every gate uses),
    radiation-cadence divisibility, binary64 clock closure, an exact
    millisecond class key, a limited-area mesh, and a configuration that is
    not already anchored.  ``runnable`` is true only when nothing blocks.

    Nothing is minted beyond the host receipt, nothing is registered, and
    ``ADMITTED_TIMESTEPS`` is not read for anything but "is this already
    anchored".
    """

    from woof.hex.timestep_admission import CourantPolicy  # noqa: PLC0415

    dt = float(dt_seconds)
    if not math.isfinite(dt) or dt <= 0.0:
        raise MintError(f"dt={dt_seconds!r} must be finite and positive")
    if convection not in ("off", "gf"):
        raise MintError(
            f"--convection {convection!r}: a mint plan names the cumulus "
            "selection it earns, 'off' or 'gf'; 'auto' would let the row decide "
            "after the plan was printed"
        )
    cumulus_scheme = None if convection == "off" else "gf"
    blockers: list[str] = []
    notes: list[str] = []

    # ---- the mesh ---------------------------------------------------------
    finest = float(facts["finest_edge_m"])
    zone_width = int(facts.get("boundary_zone_width") or 0)
    levels = facts.get("n_vert_levels")
    if zone_width <= 0:
        blockers.append(
            "the mesh carries no boundary zone (bdyMaskCell is absent or all "
            "zero), so it is a global mesh, not a regional cull.  A global "
            "mint is --integration-plan; this plan is for limited-area culls"
        )

    # ---- Courant ----------------------------------------------------------
    policy = CourantPolicy()
    limit = policy.safety_factor * finest / policy.max_characteristic_speed_m_s
    courant = {
        "policy_speed_m_s": policy.max_characteristic_speed_m_s,
        "policy_safety_factor": policy.safety_factor,
        "finest_edge_m": finest,
        "limit_seconds": limit,
        "dt_seconds": dt,
        "margin": None if dt <= 0 else limit / dt,
        "admitted": dt <= limit,
    }
    if dt > limit:
        blockers.append(
            f"dt={_fmt(dt)} s exceeds this mesh's outer-step Courant limit of "
            f"{limit:.4g} s (0.9 x {finest:.4g} m / 125 m/s).  The mint would "
            f"integrate an unstable step; choose dt <= {limit:.4g} s"
        )

    # ---- the host half ----------------------------------------------------
    integration_steps_raw = float(hours) * 3600.0 / dt
    integration_steps = int(round(integration_steps_raw))
    if integration_steps < 1 or abs(integration_steps_raw - integration_steps) > 1e-9:
        blockers.append(
            f"--hours {hours:g} is not a whole number of {_fmt(dt)} s steps "
            f"({integration_steps_raw:.4f})"
        )
        integration_steps = max(1, integration_steps)
    history_stride_raw = int(history_every_minutes) * 60 / dt
    if (
        history_stride_raw < 1.0
        or abs(history_stride_raw - round(history_stride_raw)) > 1e-9 * history_stride_raw
        or integration_steps % int(round(history_stride_raw))
    ):
        blockers.append(
            f"--history-every-minutes {history_every_minutes} is not a whole "
            f"number of {_fmt(dt)} s steps dividing the {hours:g} h arm; the "
            f"forecast door would refuse it"
        )
    receipt: dict[str, Any] | None
    try:
        receipt = mint_schedule_receipt(
            dt, cumulus_scheme=cumulus_scheme, run_steps=integration_steps
        )
    except (DtAdmissionError, MintError) as error:
        receipt = None
        blockers.append(f"the schedule receipt does not mint: {error}")

    if dt_admission.registered_anchor(dt, cumulus_scheme, None) is not None:
        blockers.append(
            f"dt={_fmt(dt)} s with convection {convection} already holds a "
            f"registered anchor; there is nothing for a mint to earn"
        )

    # ---- the regional class key ------------------------------------------
    dt_ms = dt * 1000.0
    if dt_ms != round(dt_ms):
        blockers.append(
            f"dt={_fmt(dt)} s is not a whole number of milliseconds; the "
            f"regional class key quantises dt to integer milliseconds "
            f"(RegionalClassKey.dt_ms), so two different timesteps would share "
            f"one class"
        )
    try:
        from woof.hex.cuda_backend.regional_admission import (  # noqa: PLC0415
            kernel_set_sha256,
        )

        kernel_set: str | None = kernel_set_sha256()
    except Exception as error:  # noqa: BLE001 - recorded, not fatal to a plan
        kernel_set = None
        notes.append(f"the regional kernel set could not be digested here: {error}")
    class_key = {
        "boundary_zone_width": zone_width,
        "n_vert_levels": levels,
        "finest_edge_m": round(finest, 3),
        "dt_seconds": dt,
        "dt_ms": int(round(dt_ms)),
        "kernel_set_sha256": kernel_set,
    }

    # ---- the level pin ---------------------------------------------------
    integration_blockers: list[str] = []
    if levels is None:
        notes.append(
            "no level count was measured (pass --init); the class key and the "
            "full-physics bind both need it"
        )
    elif int(levels) > FULL_PHYSICS_MAX_LEVELS:
        integration_blockers.append(
            f"the cull has {levels} levels and the full-physics forecast lane "
            f"binds at most {FULL_PHYSICS_MAX_LEVELS}, so the integration "
            f"arms (part 2) and their control (part 3) would be refused at "
            f"bind.  The dry regional mint pair (part 4) reads its levels off "
            f"the init and is not affected"
        )

    # ---- the regional mint arms' own cadence -----------------------------
    mint_steps = int(mint_steps)
    if mint_steps < 1:
        blockers.append(f"--mint-steps {mint_steps} must be at least one step")
        mint_steps = 1
    mint_seconds = mint_steps * dt
    if mint_seconds != round(mint_seconds):
        blockers.append(
            f"--mint-steps {mint_steps} at dt={_fmt(dt)} s ends at "
            f"{mint_seconds:g} s, not a whole second.  The regional instrument "
            f"names and stamps frames to the second, and this plan keeps every "
            f"frame on one so no two can share a name"
        )
    else:
        try:
            dt_admission.schedule_receipt(dt, cumulus_scheme=None, run_steps=mint_steps)
        except DtAdmissionError as error:
            blockers.append(f"the regional mint arm does not close its clock: {error}")

    # ---- the control for physics_health ----------------------------------
    anchored = sorted(
        anchor.dt_seconds
        for anchor in dt_admission.ADMITTED_TIMESTEPS.values()
        if anchor.cumulus_scheme == cumulus_scheme and not anchor.derived
    )
    fitting = [value for value in anchored if value <= limit]
    if fitting and control_dt_seconds is not None:
        notes.append(
            f"--control-dt {_fmt(control_dt_seconds)} s was asked for although "
            f"an anchored timestep ({max(fitting):g} s) fits this mesh; the "
            f"plan uses the requested step-halving control"
        )
    if fitting and control_dt_seconds is None:
        control = {
            "kind": "anchored-row-timestep",
            "control_dt_seconds": max(fitting),
            "why": (
                f"an anchored timestep ({max(fitting):g} s) fits this mesh's "
                f"{limit:.4g} s Courant limit; the row must DECLARE it for the "
                f"campaign's control arm, which runs at the row's own timestep"
            ),
        }
    else:
        control_dt = (
            dt / 2.0 if control_dt_seconds is None else float(control_dt_seconds)
        )
        control = {
            "kind": "step-halving-unanchored",
            "control_dt_seconds": control_dt,
            "why": (
                f"no anchored timestep for convection {convection} fits: the "
                f"smallest is {min(anchored) if anchored else float('nan'):g} s "
                f"against a {limit:.4g} s Courant limit.  The candidate's band "
                f"is read against a finer UNANCHORED step on the same mesh, "
                f"card and init; agreement says the solution is converged in "
                f"dt, not that it matches an anchored run"
            ),
        }
        if not 0.0 < control_dt < dt:
            blockers.append(
                f"the step-halving control ({_fmt(control_dt)} s) must be "
                f"positive and smaller than dt={_fmt(dt)} s"
            )
            control_dt = dt / 2.0
        try:
            dt_admission.schedule_receipt(
                control_dt,
                cumulus_scheme=cumulus_scheme,
                run_steps=max(1, int(round(float(hours) * 3600.0 / control_dt))),
            )
        except DtAdmissionError as error:
            blockers.append(f"the control timestep {_fmt(control_dt)} s does not mint: {error}")

    # ---- the commands ----------------------------------------------------
    label = _label(dt)
    tools = "components/hex/tools"
    schedule_out = f"{work}/schedule-receipt-{label}.json"
    schedule_command = [
        "python", f"{tools}/mint_dt_anchor.py",
        "--dt", _fmt(dt),
        "--cumulus-scheme", convection,
        "--run-steps", str(integration_steps),
        "--out", schedule_out,
    ]
    integration_work = f"{work}/integration-{label}"
    campaign = [
        "python", f"{tools}/run_dt_anchor_campaign.py",
        "--dt", _fmt(dt),
        "--card", card,
        "--mesh", mesh_row,
        "--grid", grid,
        "--static", static,
        "--init", init,
        "--init-source", init_source,
        "--lbc-dir", lbc_dir,
        "--start-time", start_time,
        "--hours", f"{float(hours):g}",
        "--history-every-minutes", str(int(history_every_minutes)),
        "--convection", convection,
        "--gpuwm-checkout", gpuwm_checkout,
        "--work", integration_work,
    ]
    if control["kind"] == "step-halving-unanchored":
        campaign += ["--control-dt", _fmt(control["control_dt_seconds"])]
    campaign_dry = [*campaign, "--dry-run"]
    campaign_real = [*campaign, "--authorization", AUTHORIZATION_PLACEHOLDER]

    stride = _whole_second_stride(dt, int(mint_steps), _REGIONAL_MINT_FRAMES)
    regional_work = f"{work}/regional-mint-{label}"
    contract = [
        "python", "-m", "woof.hex.drivers.run_cuda_regional_contract",
        "--grid", grid,
        "--init", init,
        "--lbc-dir", lbc_dir,
        "--start-time", start_time,
        "--mesh-row", mesh_row,
        "--out", f"{work}/contract",
    ]
    mint_arm = [
        "python", f"{tools}/run_cuda_regional_forecast.py",
        "--grid", grid,
        "--init", init,
        "--lbc-dir", lbc_dir,
        "--start-time", start_time,
        "--dt", _fmt(dt),
        "--len-disp", f"{finest:.3f}",
        "--steps", str(int(mint_steps)),
        "--history-every", str(stride),
        "--mint-anchor",
    ]
    mint_a = [*mint_arm, "--label", f"{label}-a", "--out", f"{regional_work}/run1"]
    mint_b = [*mint_arm, "--label", f"{label}-b", "--out", f"{regional_work}/run2"]
    compare = [
        "python", f"{tools}/compare_regional_forecast_frames.py",
        "--left", f"{regional_work}/run1",
        "--right", f"{regional_work}/run2",
        "--grid", grid,
        "--left-label", "run1",
        "--right-label", "run2",
        "--out", f"{regional_work}/pair-compare.json",
    ]
    mint_frames = len(range(0, int(mint_steps) + 1, stride))

    # A mint is all four parts; a plan whose integration half would be
    # refused at bind is not a runnable mint even when the dry pair is.
    runnable = not blockers and not integration_blockers
    return {
        "schema": "gpuwm-hex.dt-anchor-regional-mint-plan/v1",
        "timestep_evidence": "experimental-unanchored",
        "dt_seconds": dt,
        "mesh_row": mesh_row,
        "convection": convection,
        "card": card,
        "mesh": facts,
        "runnable": runnable,
        "blockers": blockers,
        "integration_blockers": integration_blockers,
        "notes": notes,
        "checks": {
            "courant": courant,
            "radiation_steps": None if receipt is None else receipt["cadences"]["stepra"],
            "clock_closes_in_binary64": receipt is not None,
            "dt_ms_exact": dt_ms == round(dt_ms),
            "regional": zone_width > 0,
            "already_anchored": (
                dt_admission.registered_anchor(dt, cumulus_scheme, None) is not None
            ),
        },
        "parts": {
            "schedule_receipt": {
                "needs_authorization": False,
                "needs_card": False,
                "command": schedule_command,
                "outputs": [schedule_out],
                "receipt": receipt,
            },
            "integration_anchor": {
                "needs_authorization": True,
                "needs_card": True,
                "inputs": {
                    "mesh_row": f"{mesh_row} (registered: `woof hex register`, "
                                f"named through $WOOF_HEX_MESH_ROWS)",
                    "grid": grid, "static": static, "init": init,
                    "lbc_dir": lbc_dir, "start_time": start_time,
                    "gpuwm_checkout": gpuwm_checkout,
                },
                "steps_per_arm": integration_steps,
                "arms": 2,
                "dry_run_command": campaign_dry,
                "command": campaign_real,
                "outputs": [
                    f"{integration_work}/anchor-{label}.json",
                    f"{integration_work}/arm-a/out/forecast-receipt.json",
                    f"{integration_work}/arm-b/out/forecast-receipt.json",
                    f"{integration_work}/DONE (or FAILED)",
                ],
                "passes_when": (
                    "anchor JSON integration_anchor_earned is true: every "
                    "history frame identical under masked digests, finite "
                    f"at every one of the {integration_steps} steps in both "
                    "arms, every requested step executed"
                ),
                "blocked_by": integration_blockers,
            },
            "physics_health": {
                "needs_authorization": control["kind"] == "step-halving-unanchored",
                "needs_card": True,
                "control": control,
                "command": "run by the integration campaign above (its control leg)",
                "outputs": [f"{integration_work}/control-band.json",
                            f"{integration_work}/anchor-{label}.json "
                            "(health_against_control)"],
                "verdict": (
                    "NOT MEASURED -- the verdict word (TRACKS / DIVERGES ...) "
                    "is written by a person reading health_against_control, "
                    "and leads the row's physics_health field"
                ),
                "blocked_by": integration_blockers,
            },
            "regional_forecast_mint_pair": {
                "needs_authorization": False,
                "needs_card": True,
                "contract_deck": {
                    "command": contract,
                    "outputs": [f"{work}/contract/"],
                    "note": "geometry-keyed and timestep-independent; one per cull",
                },
                "arm_commands": [mint_a, mint_b],
                "compare_command": compare,
                "steps_per_arm": int(mint_steps),
                "history_every_steps": stride,
                "history_stride_seconds": stride * dt,
                "frames_per_arm": mint_frames,
                "model_seconds_per_arm": int(mint_steps) * dt,
                "outputs": [
                    f"{regional_work}/run1/forecast.json",
                    f"{regional_work}/run2/forecast.json",
                    f"{regional_work}/pair-compare.json",
                ],
                "passes_when": (
                    "every frame's masked_sha256 matches between run1 and run2 "
                    "and every published field is finite in every frame"
                ),
                "class_key": class_key,
            },
        },
        "registration": (
            "nothing here registers anything.  ADMITTED_TIMESTEPS and the "
            "regional class table are not edited by a mint; a passing mint is "
            "brought to the user, who rules on it"
        ),
        "authorization": (
            f"shown as {AUTHORIZATION_PLACEHOLDER}.  The sentence is supplied "
            "verbatim by the user at run time; no plan, script or receipt "
            "carries it"
        ),
        "what_this_does_not_prove": [
            "byte-identity against native MPAS-A: the only native v8.4.1 "
            f"reference ran at {dt_admission.PROVEN_DT_SECONDS:g} s",
            "the dry regional mint pair certifies the 22 boundary kernels at "
            "this dt and says nothing about the physics seam; the integration "
            "arms are what exercise it",
            "skill: a converged, deterministic run at 0.5 s is not a verified "
            "one; only obs-skill referees that",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="mint or verify a model-timestep anchor for the frozen lane"
    )
    parser.add_argument("--dt", type=float, help="mint this timestep's schedule receipt")
    parser.add_argument(
        "--radiation-seconds",
        type=float,
        default=dt_admission.RADIATION_CADENCE_SECONDS,
    )
    parser.add_argument("--surface-pbl-seconds", type=float, default=None)
    parser.add_argument("--cumulus-seconds", type=float, default=None)
    parser.add_argument(
        "--cumulus-scheme",
        choices=("gf", "off"),
        default="gf",
        help=(
            "the cumulus selection this receipt is minted for.  'off' is "
            "the 2026-08-26 ruling below 3 km, and it is a DIFFERENT "
            "configuration: it holds its own anchor row and reports no "
            "cumulus cadence at all"
        ),
    )
    parser.add_argument("--run-steps", type=int, default=720)
    parser.add_argument("--out", type=Path, help="write the receipt here")
    parser.add_argument(
        "--verify", action="store_true", help="certify every registered anchor"
    )
    parser.add_argument(
        "--mutation-control",
        action="store_true",
        help="prove the verifier fails on fabricated evidence",
    )
    parser.add_argument(
        "--integration-plan",
        action="store_true",
        help="print the card half's runnable procedure (needs --dt and --mesh)",
    )
    parser.add_argument("--mesh", default=None)
    parser.add_argument(
        "--hours",
        type=float,
        default=None,
        help="arm length (default 6 for --integration-plan, 1 for --regional-plan)",
    )
    parser.add_argument(
        "--regional-plan",
        action="store_true",
        help=(
            "print every part of a REGIONAL anchor mint at --dt (schedule "
            "receipt, integration pair, physics-health control, regional "
            "forecast mint pair) with exact commands, checked against the "
            "cull's own files; runs nothing on a card and needs no "
            "authorization.  Needs --dt, --mesh and --grid"
        ),
    )
    parser.add_argument("--grid", type=Path, default=None, help="the regional cull's grid")
    parser.add_argument("--init", type=Path, default=None, help="the cull's init (levels)")
    parser.add_argument("--static", type=Path, default=None)
    parser.add_argument("--lbc-dir", type=Path, default=None)
    parser.add_argument("--start-time", default=None, metavar="YYYY-MM-DD_HH:MM:SS")
    parser.add_argument("--init-source", default=None)
    parser.add_argument("--gpuwm-checkout", type=Path, default=None)
    parser.add_argument("--work", type=Path, default=None, help="where the mint writes")
    parser.add_argument(
        "--convection",
        choices=("off", "gf"),
        default="off",
        help="the cumulus selection a --regional-plan earns (default off)",
    )
    parser.add_argument("--history-every-minutes", type=int, default=10)
    parser.add_argument("--mint-steps", type=int, default=REGIONAL_MINT_STEPS)
    parser.add_argument(
        "--control-dt",
        type=float,
        default=None,
        help="the step-halving control timestep (default dt/2 when no anchor fits)",
    )
    parser.add_argument(
        "--candidate-forecast",
        action="store_true",
        help=(
            "run ONE forecast arm at an unanchored --dt to mint its "
            "integration evidence; needs --card and --authorization, and "
            "every forecast argument after a bare --"
        ),
    )
    parser.add_argument("--card", default=None)
    parser.add_argument("--authorization", default=None)
    parser.add_argument(
        "--authorization-env",
        default=None,
        metavar="VAR",
        help=(
            "read the candidate-mint authorization from this environment "
            f"variable instead of argv (the campaign uses {AUTHORIZATION_ENVIRONMENT})"
        ),
    )
    parser.add_argument(
        "forecast_argv",
        nargs=argparse.REMAINDER,
        help="forecast arguments, after a bare --",
    )
    arguments = parser.parse_args(argv)

    if arguments.candidate_forecast:
        if arguments.authorization_env and not arguments.authorization:
            arguments.authorization = os.environ.get(arguments.authorization_env)
        if arguments.dt is None or not arguments.card or not arguments.authorization:
            parser.error(
                "--candidate-forecast needs --dt, --card and --authorization "
                "(or --authorization-env naming a variable that is set)"
            )
        forward = list(arguments.forecast_argv)
        if forward and forward[0] == "--":
            forward = forward[1:]
        if not forward:
            parser.error(
                "--candidate-forecast needs the forecast arguments after a bare --"
            )
        try:
            return run_candidate_forecast(
                arguments.dt,
                card=arguments.card,
                authorization=arguments.authorization,
                forecast_argv=forward,
            )
        except (DtAdmissionError, MintError) as error:
            # Printed redacted and without the traceback: the gate's own
            # refusal quotes the sentence, and this lands in an arm's log.
            print(
                json.dumps(
                    {
                        "schema": SCHEMA,
                        "candidate_forecast": "refused",
                        "dt_seconds": arguments.dt,
                        "refusal": redact_authorization(error),
                    },
                    indent=2,
                    sort_keys=True,
                ),
                file=sys.stderr,
            )
            return 2

    if arguments.regional_plan:
        if arguments.dt is None or not arguments.mesh or arguments.grid is None:
            parser.error("--regional-plan needs --dt, --mesh and --grid")
        try:
            facts = measure_regional_mesh(arguments.grid, arguments.init)
            plan = regional_mint_plan(
                arguments.dt,
                mesh_row=arguments.mesh,
                facts=facts,
                grid=str(arguments.grid),
                init=str(arguments.init) if arguments.init else "<init.nc>",
                static=str(arguments.static) if arguments.static else "<static.nc>",
                lbc_dir=str(arguments.lbc_dir) if arguments.lbc_dir else "<lbc-dir>",
                start_time=arguments.start_time or "<YYYY-MM-DD_HH:MM:SS>",
                init_source=arguments.init_source or "<what produced the init>",
                gpuwm_checkout=(
                    str(arguments.gpuwm_checkout)
                    if arguments.gpuwm_checkout
                    else "<arwen-checkout>"
                ),
                card=arguments.card or "<card>",
                work=str(arguments.work) if arguments.work else "<work>",
                hours=1.0 if arguments.hours is None else arguments.hours,
                history_every_minutes=arguments.history_every_minutes,
                convection=arguments.convection,
                mint_steps=arguments.mint_steps,
                control_dt_seconds=arguments.control_dt,
            )
        except (DtAdmissionError, MintError, OSError) as error:
            print(
                json.dumps(
                    {"schema": SCHEMA, "regional_plan": "refused",
                     "refusal": redact_authorization(error)},
                    indent=2,
                    sort_keys=True,
                )
            )
            return 1
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0 if plan["runnable"] else 1

    if not any(
        (
            arguments.dt is not None,
            arguments.verify,
            arguments.mutation_control,
        )
    ):
        parser.error("choose --dt, --verify or --mutation-control")

    payload: dict[str, Any] = {"schema": SCHEMA}
    status = 0

    if arguments.verify:
        payload["registry"] = verify_registry()
        if not payload["registry"]["certified"]:
            status = 1
    if arguments.mutation_control:
        payload["mutation_control"] = mutation_control()
        if not payload["mutation_control"]["has_teeth_both_directions"]:
            status = 1
    if arguments.dt is not None:
        if arguments.integration_plan:
            if not arguments.mesh:
                parser.error("--integration-plan needs --mesh")
            payload["integration_plan"] = integration_plan(
                arguments.dt,
                arguments.mesh,
                6.0 if arguments.hours is None else arguments.hours,
            )
        try:
            scheme = None if arguments.cumulus_scheme == "off" else "gf"
            receipt = mint_schedule_receipt(
                arguments.dt,
                radiation_seconds=arguments.radiation_seconds,
                surface_pbl_seconds=arguments.surface_pbl_seconds,
                cumulus_seconds=arguments.cumulus_seconds,
                cumulus_scheme=scheme,
                run_steps=arguments.run_steps,
            )
        except (DtAdmissionError, MintError) as error:
            payload["schedule_receipt"] = {
                "dt_seconds": arguments.dt,
                "minted": False,
                "refusal": str(error),
            }
            status = 1
        else:
            payload["schedule_receipt"] = receipt
            # Ask about the CONFIGURATION this receipt was minted for, not
            # the timestep alone.  Reading the welded row here would report
            # "anchored" for a receipt minted at a HELD surface/PBL cadence,
            # which is the wrong-row trap the registry key exists to close.
            # The EXACT lookup: a held cadence that is admitted through a
            # row derived from the welded anchor has not been EARNED, and
            # "anchored today" answers whether there is a registered row.
            minted_bldt = float(receipt["cadences"]["surface_pbl_seconds"])
            anchored = (
                dt_admission.registered_anchor(arguments.dt, scheme, minted_bldt)
                is not None
            )
            payload["anchored_today"] = anchored
            if not anchored:
                payload["registration_note"] = dt_admission.unanchored_refusal(
                    arguments.dt, scheme, minted_bldt
                )
            if arguments.out is not None:
                arguments.out.parent.mkdir(parents=True, exist_ok=True)
                arguments.out.write_text(
                    json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                    newline="\n",
                )
                payload["written"] = str(arguments.out)

    print(json.dumps(payload, indent=2, sort_keys=True))
    return status


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
