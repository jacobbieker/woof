#!/usr/bin/env python3
"""Earn one model-timestep anchor's INTEGRATION half on one card.

``tools/mint_dt_anchor.py`` mints the host-derivable half of an anchor and
prints the procedure for the half that needs a card.  This is that procedure,
run rather than printed, as one self-contained job per timestep per card:

1. a **control arm** at the anchored timestep, through the ordinary forecast
   door, on this card, this mesh, this init.  Its per-step health band is what
   the candidate's band is read against -- a band compared across cards or
   across cases would be measuring the card or the case;
2. **two candidate arms** at the timestep being earned, each a separate
   process, each through ``mint_dt_anchor.py --candidate-forecast`` so the
   run is stamped ``CANDIDATE-UNANCHORED`` in its own receipt;
3. the **digest comparison** of every history frame the two candidate arms
   wrote, which is what "byte-identical" means here.  Each digest is labelled
   with the convention that produced it: the house ``file_id``-masked SHA-256
   on a classic-CDF file, and the plain whole-file SHA-256 on the port's own
   ``NETCDF4_CLASSIC``-on-HDF5 frames, which carry no ``file_id`` nonce to
   mask.  See :func:`frame_digest`, which refuses to fall back onto a
   non-classic file that does carry one; and
4. the **health band** of all three arms, side by side, so a closure that
   misbehaves at a higher call rate is visible as a number rather than
   inferred from the run not crashing.

Point 4 is the reason this tool exists instead of a shell loop.  Two
byte-identical forecasts prove the run is deterministic and finite.  They do
not prove the physics is well behaved: WRF pins ``cudt = 0`` for
Grell-Freitas, so the cumulus closure is called every step and a smaller
timestep calls it proportionally more often -- 30 times an hour at 120 s, 720
at 5 s.  Whether that closure still behaves at 24x its proven call rate is a
question only measurement answers, and only obs-skill settles.  This tool
records the measurement; it does not settle the question, and it says so in
its own output.

The result is written as one JSON per timestep plus an rc-gated marker, so a
long leg can run detached and be adjudicated by its marker rather than by
watching it.

The candidate arms run only under the candidate-mint sentence, supplied
verbatim by the person who rules on the anchor (``--authorization``, or
``$WOOF_HEX_CANDIDATE_MINT_AUTHORIZATION`` to keep it off the command line).
This tool used to fill that argument in from the constant itself, which meant
anyone who could run it had opened the gate without ever saying the sentence.
It now takes it from the caller, checks it before the first arm starts, hands
it to each arm through that environment variable rather than argv, and
redacts it from the logs, the anchor JSON and the markers it writes.
``--dry-run`` runs every host check and prints every leg without it.

Sub-second regional culls (a 100 m corridor's Courant limit is under a
second, and the smallest anchored timestep is 5 s) cannot run the control at
an anchored timestep at all.  ``--control-dt`` runs it at a finer unanchored
step through the same candidate path instead, and the anchor JSON labels the
control ``step-halving-unanchored``.

Nothing here registers anything.  Registering an anchor moves the frozen lane
off its proven timestep, which is a ruling.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for _entry in (str(SRC), str(ROOT / "tools")):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from woof.hex import dt_admission  # noqa: E402
from woof.hex import pbl_cadence  # noqa: E402
# ONE redaction rule and one placeholder, owned by the mint harness this
# campaign drives, so the two tools cannot drift on what they keep out of logs.
from mint_dt_anchor import (  # noqa: E402
    AUTHORIZATION_ENVIRONMENT,
    AUTHORIZATION_PLACEHOLDER,
    redact_authorization,
)

SCHEMA = "gpuwm-hex.dt-anchor-campaign/v1"

#: The band fields the driver publishes per step.  Quoted rather than
#: discovered so a driver that stops publishing one is a visible failure
#: instead of a silently shorter table.
BAND_FIELDS = (
    "vertical_velocity_abs_max",
    "theta_m_min",
    "theta_m_max",
    "qv_max",
    "qv_min",
    "exner_min",
    "rho_min",
    "hydrometeor_min",
)


class CampaignError(RuntimeError):
    """The campaign refuses to continue, by name."""


def _log(message: str) -> None:
    print(f"[dt-anchor] {message}", flush=True)


def frame_digest(path: Path) -> dict[str, Any]:
    """One history frame's content digest, and WHICH convention produced it.

    The house masked digest NULs the netCDF ``file_id`` attribute, because a
    native MPAS-A history file carries a random one in its classic header and
    two identical runs would otherwise differ.  It is defined only against the
    classic CDF-1/2/5 header layout and refuses anything else by name.

    The port's own writer does not produce that layout: measured on a real
    frame (2026-08-26, the proving RTX 5090), ``cuda-history.*.nc`` is ``NETCDF4_CLASSIC``
    on HDF5 and its global attributes are ``schema``, ``source``,
    ``arwen_commit``, the four GF/q2 provenance strings and
    ``refl10cm_provenance`` -- there is **no** ``file_id``.  The nonce is a
    native artefact, not a port one, so on these files there is nothing to
    mask and the whole-file SHA-256 IS the content digest.  That is also the
    digest the driver itself records per frame in ``snapshot_files`` and the
    one every previous determinism claim in this tree was made with.

    Falling back silently would be the defect, so this refuses to fall back
    onto a file that DOES carry a nonce, and it labels every digest with the
    convention that produced it.
    """

    from run_cuda_v841_full_physics_x4 import netcdf_masked_digests

    try:
        entry = netcdf_masked_digests(path)
    except Exception as error:  # noqa: BLE001 - the helper refuses by name
        if "classic CDF" not in str(error):
            raise
        import hashlib

        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        try:
            from netCDF4 import Dataset

            with Dataset(str(path)) as dataset:
                attributes = list(dataset.ncattrs())
        except Exception:  # noqa: BLE001 - absence of the reader is not proof
            attributes = None
        if attributes is not None and "file_id" in attributes:
            raise CampaignError(
                f"{path} is not a classic netCDF file, so the house file_id "
                f"mask does not apply to it -- but it DOES carry a file_id "
                f"attribute, so its whole-file digest would differ between two "
                f"identical runs for a reason that is not the physics.  "
                f"Comparing these arms needs a masking convention defined "
                f"against this layout, and this campaign will not invent one"
            ) from error
        return {
            "convention": "whole-file-sha256",
            "digest": digest.hexdigest(),
            "file_id": None,
            "why": (
                "the port's history writer emits NETCDF4_CLASSIC on HDF5 and "
                "carries no file_id nonce, so there is nothing to mask; this "
                "is the same digest the driver records per frame"
            ),
        }
    return {
        "convention": "masked-file_id",
        "digest": entry["masked_sha256"],
        "file_id": entry.get("file_id"),
        "why": "classic CDF header; the file_id nonce is NUL'd before hashing",
    }


def history_frames(out: Path) -> list[Path]:
    """Every history frame in an arm's output directory, in name order."""

    return sorted(out.glob("*history*.nc"))


def band(step_health: list[dict[str, Any]]) -> dict[str, Any]:
    """The health band over every published step, plus the finiteness roll-up."""

    if not step_health:
        raise CampaignError(
            "the driver published no per-step health records, so this arm "
            "cannot say the run was finite at every step -- which is half of "
            "what an integration anchor is"
        )
    summary: dict[str, Any] = {
        "steps": len(step_health),
        "finite_every_step": all(bool(row.get("finite")) for row in step_health),
        "first_nonfinite_step": next(
            (int(row["step"]) for row in step_health if not bool(row.get("finite"))),
            None,
        ),
    }
    for field in BAND_FIELDS:
        values = [row[field] for row in step_health if row.get(field) is not None]
        if not values:
            summary[field] = None
            continue
        summary[field] = {"min": float(min(values)), "max": float(max(values))}
    summary["trend"] = trend(step_health)
    return summary


#: How many equal slices of the run the trend is reported over.  Four, so a
#: 2 h arm reports half-hours and the slices line up across timesteps whose
#: step counts differ by a factor of 24.
TREND_SLICES = 4


def trend(step_health: list[dict[str, Any]]) -> dict[str, Any]:
    """Each band field's mean and max over equal slices of the run.

    A min/max band cannot tell a spin-up transient from a divergence that
    grows with lead time, and those have opposite meanings: the first is the
    model settling, the second is the solution going somewhere.  MEASURED
    (2026-08-26, the proving RTX 5070 Ti, x1.40962): at 20 s the vertical-velocity mean per
    half-hour runs 1.99, 3.48, 4.11, 5.53 m/s against the 120 s control's
    1.15, 1.17, 1.20, 1.48 -- monotone, still climbing at the end of the arm,
    and not a spike.  Reading only the band's 7.51 maximum would have made
    that look like one excursion.
    """

    count = len(step_health)
    slices: list[dict[str, Any]] = []
    for index in range(TREND_SLICES):
        low = count * index // TREND_SLICES
        high = count * (index + 1) // TREND_SLICES
        window = step_health[low:high]
        if not window:
            continue
        entry: dict[str, Any] = {
            "slice": index,
            "steps": [int(window[0]["step"]), int(window[-1]["step"])],
        }
        for field in BAND_FIELDS:
            values = [row[field] for row in window if row.get(field) is not None]
            if not values:
                entry[field] = None
                continue
            entry[field] = {
                "mean": float(sum(values) / len(values)),
                "max": float(max(values)),
            }
        slices.append(entry)
    return {"slices": TREND_SLICES, "windows": slices}


def prepare_arm_directory(arm: Path) -> None:
    """Create an arm's parent, and NOT its ``--out``.

    The door refuses to build a deep path for an expensive run -- a mistyped
    one would otherwise look like a successful new directory -- and it also
    refuses an ``--out`` that already exists.  So the campaign owns exactly
    one level: the arm folder that ``out/`` and ``scratch/`` sit inside.
    """

    arm.mkdir(parents=True, exist_ok=True)
    for child in ("out", "scratch"):
        if (arm / child).exists():
            raise CampaignError(
                f"{arm / child} already exists; the forecast door requires a "
                f"fresh output directory, so this arm would be refused.  Point "
                f"--work at a directory this campaign owns"
            )


def _run(
    argv: list[str],
    log_path: Path,
    cwd: Path,
    environment: dict[str, str] | None = None,
) -> int:
    """Run one leg, tee its output to a log, return its rc.

    ``environment`` is ADDED to this process's environment for the leg.  It
    is how a candidate arm receives the authorization: through the variable
    the mint harness reads (``--authorization-env``), never through argv,
    which the log below records and the process list shows.  The argv line
    is redacted as well, in case a caller passed the sentence there anyway.
    """

    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    env = None if not environment else {**os.environ, **environment}
    with log_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(
            f"argv: {redact_authorization(json.dumps(argv))}\ncwd: {cwd}\n\n"
        )
        handle.flush()
        completed = subprocess.run(
            argv,
            cwd=str(cwd),
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
            env=env,
        )
        handle.write(f"\nrc: {completed.returncode}\n")
        handle.write(f"seconds: {time.time() - started:.1f}\n")
    _log(f"{log_path.name}: rc={completed.returncode} in {time.time() - started:.1f}s")
    return int(completed.returncode)


def door_argv(
    *,
    python: str,
    repo: Path,
    convection: str,
    pbl_cadence: str,
    mesh: str,
    grid: Path,
    static: Path,
    init: Path,
    init_source: str,
    start_time: str | None,
    hours: float,
    history_every_minutes: int,
    out: Path,
    scratch: Path,
    gpuwm_checkout: Path,
    case_label: str,
    device_fixed_mib: float | None,
    device_bytes_per_cell: float | None,
    lbc_dir: Path | None = None,
) -> list[str]:
    """The forecast door's argument vector for one arm.

    ``lbc_dir`` is the limited-area boundary series.  A regional corridor
    cull is integrated with ``config_apply_lbcs`` on, and the door refuses
    a cull with no boundary series by name, so a regional mint names it.
    """

    argv = [
        python, "-m", "woof.hex.cli", "forecast",
        "--mesh", mesh,
        "--grid", str(grid),
        "--static", str(static),
        "--init", str(init),
        "--init-source", init_source,
        "--hours", f"{hours:g}",
        "--history-every-minutes", str(history_every_minutes),
        "--repo", str(repo),
        "--gpuwm-checkout", str(gpuwm_checkout),
        "--out", str(out),
        "--scratch", str(scratch),
        "--case-label", case_label,
        "--convection", convection,
        "--pbl-cadence", pbl_cadence,
    ]
    if start_time:
        argv += ["--start-time", start_time]
    if lbc_dir is not None:
        argv += ["--lbc-dir", str(lbc_dir)]
    if device_fixed_mib is not None and device_bytes_per_cell is not None:
        argv += [
            "--device-fixed-mib", f"{device_fixed_mib:g}",
            "--device-bytes-per-cell", f"{device_bytes_per_cell:g}",
        ]
    return argv


def read_receipt(out: Path) -> dict[str, Any]:
    """One arm's forecast receipt, or a refusal naming what is missing."""

    path = out / "forecast-receipt.json"
    if not path.exists():
        raise CampaignError(
            f"the arm at {out} wrote no forecast receipt ({path.name}); the "
            f"door writes one on every terminated run, so this arm did not "
            f"reach the door's own exit path.  Its log is the authority"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def arm_summary(out: Path) -> dict[str, Any]:
    """One arm reduced to what an anchor records."""

    receipt = read_receipt(out)
    driver = receipt.get("driver_receipt") or {}
    forecast = driver.get("forecast") or {}
    frames = history_frames(out)
    digests = []
    for frame in frames:
        entry = frame_digest(frame)
        digests.append(
            {
                "file": frame.name,
                "bytes": frame.stat().st_size,
                "digest": entry["digest"],
                "digest_convention": entry["convention"],
                "file_id": entry.get("file_id"),
            }
        )
    schedule = forecast.get("schedule") or receipt.get("schedule") or {}
    walls = forecast.get("walls") or {}
    capability = forecast.get("capability") or {}
    return {
        "out": str(out),
        "status": receipt.get("status"),
        "driver_status": driver.get("status"),
        "refusal": forecast.get("refusal"),
        "steps_requested": forecast.get("steps_requested"),
        "steps_executed": forecast.get("steps_executed"),
        "dt_seconds": schedule.get("dt_seconds"),
        "history_stride_steps": schedule.get("history_stride_steps"),
        "execution_seconds": driver.get("execution_seconds"),
        "seconds_per_step_after_first": walls.get("seconds_per_step_after_first"),
        "device": {
            "name": capability.get("name"),
            "sm": capability.get("sm"),
            "compute_capability": capability.get("compute_capability"),
            "multiprocessor_count": capability.get("multiprocessor_count"),
        },
        "memory_admission": forecast.get("memory_admission"),
        # The candidate stamp lives under mesh_binding.observed, where
        # bind_mesh records what it admitted.  Lifted into the anchor's own
        # JSON so the row that admitted the run travels with the evidence
        # rather than only inside the door receipt.
        "dt_admission": (
            ((receipt.get("mesh_binding") or {}).get("observed") or {}).get(
                "dt_admission"
            )
        ),
        "mesh_row_notes": (
            ((receipt.get("mesh_binding") or {}).get("notes"))
        ),
        "band": band(list(forecast.get("step_health") or [])),
        "history": digests,
    }


def compare_arms(arm_a: dict[str, Any], arm_b: dict[str, Any]) -> dict[str, Any]:
    """Frame-by-frame masked-digest comparison of the two candidate arms."""

    by_name_a = {row["file"]: row for row in arm_a["history"]}
    by_name_b = {row["file"]: row for row in arm_b["history"]}
    names = sorted(set(by_name_a) | set(by_name_b))
    rows = []
    for name in names:
        left = by_name_a.get(name)
        right = by_name_b.get(name)
        rows.append(
            {
                "file": name,
                "in_arm_a": left is not None,
                "in_arm_b": right is not None,
                "digest_a": None if left is None else left["digest"],
                "digest_b": None if right is None else right["digest"],
                "identical": (
                    left is not None
                    and right is not None
                    and left["digest"] == right["digest"]
                ),
            }
        )
    return {
        "frames": len(rows),
        "frames_identical": sum(1 for row in rows if row["identical"]),
        "all_identical": bool(rows) and all(row["identical"] for row in rows),
        "rows": rows,
    }


def compare_bands(control: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """The candidate's band against the control's, field by field.

    Reported as the candidate's interval, the control's interval, and whether
    the candidate stays inside the control's.  ``within_control`` is a
    DESCRIPTION and not a gate: a smaller timestep resolving more vertical
    velocity is expected physics, not a defect, and a gate here would refuse
    the very thing the campaign is measuring.  What it makes impossible is
    reporting "nothing drifts" without the numbers that would show it.
    """

    rows: dict[str, Any] = {}
    for field in BAND_FIELDS:
        left = control.get(field)
        right = candidate.get(field)
        if not isinstance(left, dict) or not isinstance(right, dict):
            rows[field] = {"control": left, "candidate": right, "comparable": False}
            continue
        span = left["max"] - left["min"]
        rows[field] = {
            "control": left,
            "candidate": right,
            "comparable": True,
            "within_control": (
                right["min"] >= left["min"] and right["max"] <= left["max"]
            ),
            "min_delta": right["min"] - left["min"],
            "max_delta": right["max"] - left["max"],
            "max_delta_relative_to_control_span": (
                None if span == 0 else (right["max"] - left["max"]) / span
            ),
        }
    return rows


def resolve_cumulus_scheme(mesh: str, convection: str) -> str | None:
    """The cumulus selection the ARMS will run, decided as the harness decides it.

    An explicit ``off`` or ``gf`` is the selection whatever the mesh.  ``auto``
    applies the 3 km ruling to the registered row's spacing -- the same
    ``convection_decision`` call ``mint_dt_anchor.run_candidate_forecast``
    makes -- so the host half and the pre-check are about the configuration
    the arms actually run, not about Grell-Freitas by default.
    """

    if convection == "off":
        return None
    if convection == "gf":
        return "gf"
    from woof.hex import convection_admission, forecast_door  # noqa: PLC0415

    row = forecast_door.load_registry().get(mesh)
    if row is None:
        raise CampaignError(
            f"--convection auto needs the registered row to decide on, and "
            f"{mesh!r} is not registered here (name the row file with "
            f"$WOOF_HEX_MESH_ROWS), or pass --convection off|gf explicitly"
        )
    return convection_admission.convection_decision(
        nominal_dx_m=float(row.nominal_dx_m), requested="auto"
    )["constructor_scheme"]


def preflight_configuration(
    dt_seconds: float,
    *,
    cumulus_scheme: str | None,
    pbl_request: str,
    hours: float,
) -> dict[str, Any]:
    """The host checks one candidate configuration must pass, sentence aside.

    The SAME checks the candidate admission makes before a card is touched --
    its schedule receipt mints at the arm's cumulus selection and surface/PBL
    cadence over the arm's length, and the configuration holds no REGISTERED
    anchor -- so a dry run that prints ``runnable: true`` is a campaign whose
    real run would get as far as the authorization.
    """

    pbl = pbl_cadence.pbl_cadence_decision(dt_seconds=dt_seconds, requested=pbl_request)
    surface_pbl_seconds = float(pbl["surface_pbl_seconds"])
    receipt = dt_admission.schedule_receipt(
        dt_seconds,
        cumulus_scheme=cumulus_scheme,
        surface_pbl_seconds=surface_pbl_seconds,
        run_steps=max(1, int(round(hours * 3600.0 / dt_seconds))),
    )
    if (
        dt_admission.registered_anchor(dt_seconds, cumulus_scheme, surface_pbl_seconds)
        is not None
    ):
        raise CampaignError(
            f"dt={float(dt_seconds):g} s with "
            f"{'convection off' if cumulus_scheme is None else cumulus_scheme} "
            f"already holds a registered anchor; a candidate arm there would be "
            f"refused, and there is nothing for it to earn"
        )
    return {"pbl_cadence": pbl, "surface_pbl_seconds": surface_pbl_seconds,
            "schedule_receipt": receipt}


def check_authorization(
    dt_seconds: float,
    *,
    card: str,
    authorization: str,
    cumulus_scheme: str | None,
    surface_pbl_seconds: float,
) -> None:
    """Refuse a wrong authorization BEFORE the control arm spends card time.

    Constructs the candidate admission the arm will open and does not enter
    it, so nothing is admitted here: the constructor checks the sentence, the
    card, that the configuration holds no real anchor, and that its schedule
    receipt mints clean -- which is everything the arm would check first,
    hours later, after the control arm had already run.
    """

    try:
        dt_admission.candidate_mint(
            dt_seconds,
            authorization=authorization,
            card=card,
            cumulus_scheme=cumulus_scheme,
            cumulus_seconds=None if cumulus_scheme is None else float(dt_seconds),
            surface_pbl_seconds=surface_pbl_seconds,
        )
    except dt_admission.DtAdmissionError as error:
        raise CampaignError(
            f"the candidate admission for dt={float(dt_seconds):g} s would be "
            f"refused, so no arm is started: {redact_authorization(error)}"
        ) from None


def candidate_argv(
    *,
    python: str,
    repo: Path,
    dt_seconds: float,
    card: str,
    forwarded: list[str],
) -> list[str]:
    """One candidate arm: the mint harness wrapping the door's arguments.

    The authorization travels in the environment (see :func:`_run`), named by
    ``--authorization-env``; it is never an argument.
    """

    return [
        python,
        str(repo / "tools" / "mint_dt_anchor.py"),
        "--candidate-forecast",
        "--dt", f"{float(dt_seconds):g}",
        "--card", card,
        "--authorization-env", AUTHORIZATION_ENVIRONMENT,
        "--",
        *forwarded,
    ]


def _forwarded(argv: list[str]) -> list[str]:
    """Everything after the door subcommand, which the mint harness forwards."""

    return argv[argv.index("forecast") + 1:]


def control_kind(arguments: argparse.Namespace) -> str:
    """Which control the physics-health verdict is read against."""

    if arguments.control_band is not None:
        return "reused-band"
    if arguments.skip_control:
        return "none"
    if arguments.control_dt is not None:
        return "step-halving-unanchored"
    return "anchored-row-timestep"


def leg_argvs(arguments: argparse.Namespace, *, label: str) -> dict[str, list[str]]:
    """Every card leg this campaign would run, as argument vectors.

    One function for the dry run and the real run, so the plan a dry run
    prints is the plan the real run executes.
    """

    work = arguments.work
    common = dict(
        python=arguments.python,
        repo=arguments.repo,
        convection=arguments.convection,
        pbl_cadence=arguments.pbl_cadence,
        mesh=arguments.mesh,
        grid=arguments.grid,
        static=arguments.static,
        init=arguments.init,
        init_source=arguments.init_source,
        start_time=arguments.start_time,
        hours=arguments.hours,
        history_every_minutes=arguments.history_every_minutes,
        gpuwm_checkout=arguments.gpuwm_checkout,
        device_fixed_mib=arguments.device_fixed_mib,
        device_bytes_per_cell=arguments.device_bytes_per_cell,
        lbc_dir=arguments.lbc_dir,
    )
    legs: dict[str, list[str]] = {}
    kind = control_kind(arguments)
    if kind in ("anchored-row-timestep", "step-halving-unanchored"):
        control = door_argv(
            out=work / "control" / "out",
            scratch=work / "control" / "scratch",
            case_label=f"dt-anchor-control-{label}",
            **common,
        )
        if kind == "step-halving-unanchored":
            control = candidate_argv(
                python=arguments.python,
                repo=arguments.repo,
                dt_seconds=float(arguments.control_dt),
                card=arguments.card,
                forwarded=_forwarded(control),
            )
        legs["control"] = control
    for name in ("arm-a", "arm-b"):
        legs[name] = candidate_argv(
            python=arguments.python,
            repo=arguments.repo,
            dt_seconds=float(arguments.dt),
            card=arguments.card,
            forwarded=_forwarded(
                door_argv(
                    out=work / name / "out",
                    scratch=work / name / "scratch",
                    case_label=f"dt-anchor-{label}-{name}",
                    **common,
                )
            ),
        )
    return legs


def check_control_dt(dt_seconds: float, control_dt: float) -> None:
    """A step-halving control must be a smaller, positive timestep."""

    if not 0.0 < control_dt < dt_seconds:
        raise CampaignError(
            f"--control-dt {control_dt:g} s is not smaller than --dt "
            f"{dt_seconds:g} s (and positive).  The step-halving control reads "
            "the candidate's band against a FINER step on the same mesh, card "
            "and init; a coarser control would be the candidate's own question "
            "again"
        )


def host_preflight(
    arguments: argparse.Namespace,
) -> tuple[str | None, dict[str, Any], dict[str, Any] | None]:
    """Every host check of the campaign, for the dry run and the real run alike.

    Returns the cumulus selection, the candidate's preflight and the control's
    (``None`` unless a step-halving control will run).
    """

    dt = float(arguments.dt)
    scheme = resolve_cumulus_scheme(arguments.mesh, arguments.convection)
    candidate = preflight_configuration(
        dt, cumulus_scheme=scheme, pbl_request=arguments.pbl_cadence,
        hours=arguments.hours,
    )
    control = None
    if control_kind(arguments) == "step-halving-unanchored":
        check_control_dt(dt, float(arguments.control_dt))
        control = preflight_configuration(
            float(arguments.control_dt), cumulus_scheme=scheme,
            pbl_request=arguments.pbl_cadence, hours=arguments.hours,
        )
    return scheme, candidate, control


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="earn one timestep anchor's integration half on one card"
    )
    parser.add_argument("--dt", type=float, required=True)
    parser.add_argument("--card", required=True, help="the card's name, recorded in the anchor")
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--static", type=Path, required=True)
    parser.add_argument("--init", type=Path, required=True)
    parser.add_argument("--init-source", required=True)
    parser.add_argument("--start-time", default=None)
    parser.add_argument(
        "--lbc-dir",
        type=Path,
        default=None,
        help="lateral-boundary files (lbc.*.nc) for a regional cull; forwarded "
             "to every arm's door",
    )
    parser.add_argument("--hours", type=float, default=2.0)
    parser.add_argument("--history-every-minutes", type=int, default=30)
    parser.add_argument("--gpuwm-checkout", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True, help="fresh working directory")
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument(
        "--convection",
        choices=("auto", "off", "gf"),
        default="auto",
        help=(
            "the cumulus selection every arm of this campaign runs.  An "
            "anchor certifies a CONFIGURATION at a timestep, so 'off' earns "
            "its own row and never borrows the Grell-Freitas one"
        ),
    )
    parser.add_argument(
        "--pbl-cadence",
        default="auto",
        metavar="{auto,SECONDS}",
        help=(
            "the surface/PBL cadence every arm of this campaign runs.  "
            "'auto' welds it to dt, which is the proven configuration.  An "
            "explicit number of seconds HOLDS it there while dt shrinks -- "
            "an anchor certifies a CONFIGURATION at a timestep, so a held "
            "cadence earns its own row and never borrows the welded one"
        ),
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device-fixed-mib", type=float, default=None)
    parser.add_argument("--device-bytes-per-cell", type=float, default=None)
    controls = parser.add_mutually_exclusive_group()
    controls.add_argument(
        "--skip-control",
        action="store_true",
        help="reuse a control band already measured on this card; the anchor "
             "then names the run it came from instead of one of its own",
    )
    controls.add_argument("--control-band", type=Path, default=None)
    controls.add_argument(
        "--control-dt",
        type=float,
        default=None,
        metavar="SECONDS",
        help=(
            "run the control arm at this SMALLER unanchored timestep, through "
            "the same candidate path, instead of at the row's anchored "
            "timestep.  For a mesh no anchored timestep fits: a 100 m cull's "
            "Courant limit is under a second and the smallest anchor is 5 s, "
            "so its health band can only be read against a finer step on the "
            "same mesh.  Recorded as control kind 'step-halving-unanchored'"
        ),
    )
    parser.add_argument(
        "--authorization",
        default=None,
        help=(
            "the candidate-mint authorization sentence, verbatim, supplied by "
            "the person who rules on the anchor.  When omitted it is read from "
            f"${AUTHORIZATION_ENVIRONMENT}, which keeps it off the command "
            "line.  Required to run any candidate arm; checked before the "
            "first arm starts and handed to each arm through that variable, "
            "never through argv"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "run every host check, print every card leg this campaign would "
            "run and exit; writes nothing, touches no card and needs no "
            "authorization"
        ),
    )
    parser.add_argument(
        "--reextract",
        action="store_true",
        help="rebuild the anchor JSON from an existing --work directory "
             "without touching a card; the arms' own receipts are the source, "
             "so an analysis added after a mint ran is applied to it rather "
             "than leaving one anchor's evidence shaped differently",
    )
    arguments = parser.parse_args(argv)

    dt = float(arguments.dt)
    label = f"dt{dt:g}".replace(".", "p")

    if arguments.dry_run:
        try:
            scheme, candidate, control = host_preflight(arguments)
        except (dt_admission.DtAdmissionError, CampaignError,
                pbl_cadence.PblCadenceError) as error:
            print(
                json.dumps(
                    {"schema": SCHEMA, "dry_run": True, "runnable": False,
                     "refusal": redact_authorization(error)},
                    indent=2, sort_keys=True,
                )
            )
            return 1
        print(
            json.dumps(
                {
                    "schema": SCHEMA,
                    "dry_run": True,
                    "runnable": True,
                    "dt_seconds": dt,
                    "timestep_evidence": "experimental-unanchored",
                    "cumulus_scheme": scheme,
                    "control_kind": control_kind(arguments),
                    "control_dt_seconds": arguments.control_dt,
                    "expected_steps": int(round(arguments.hours * 3600.0 / dt)),
                    "schedule_receipt": candidate["schedule_receipt"],
                    "control_schedule_receipt": (
                        None if control is None else control["schedule_receipt"]
                    ),
                    "legs": legs_in_order(leg_argvs(arguments, label=label)),
                    "outputs": {
                        "schedule_receipt": str(
                            arguments.work / f"schedule-receipt-{label}.json"
                        ),
                        "anchor": str(arguments.work / f"anchor-{label}.json"),
                        "logs": str(arguments.work / "logs"),
                        "markers": [
                            str(arguments.work / "DONE"),
                            str(arguments.work / "FAILED"),
                        ],
                    },
                    "authorization": (
                        f"{AUTHORIZATION_PLACEHOLDER}: a real run takes the "
                        "sentence verbatim from the person ruling on the "
                        f"anchor (--authorization or ${AUTHORIZATION_ENVIRONMENT}), "
                        "refuses anything else before the first arm starts, "
                        "and hands it to each candidate arm in "
                        f"${AUTHORIZATION_ENVIRONMENT}, never in argv"
                    ),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    authorization = arguments.authorization or os.environ.get(
        AUTHORIZATION_ENVIRONMENT
    )
    if not arguments.reextract and not authorization:
        parser.error(
            f"--authorization (or ${AUTHORIZATION_ENVIRONMENT}) is required to "
            "run candidate arms; use --dry-run to print the plan, or "
            "--reextract to rebuild from an existing --work"
        )

    work = arguments.work
    work.mkdir(parents=True, exist_ok=True)
    logs = work / "logs"
    logs.mkdir(exist_ok=True)
    marker_done = work / "DONE"
    marker_failed = work / "FAILED"
    for marker in (marker_done, marker_failed):
        if marker.exists():
            marker.unlink()

    result: dict[str, Any] = {
        "schema": SCHEMA,
        "dt_seconds": dt,
        "timestep_evidence": "experimental-unanchored",
        "card": arguments.card,
        "mesh": arguments.mesh,
        "hours": arguments.hours,
        "host": __import__("woof.hex.host_identity", fromlist=["host_record"]).host_record(),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "init": str(arguments.init),
        "init_source": arguments.init_source,
        "lbc_dir": None if arguments.lbc_dir is None else str(arguments.lbc_dir),
        "gpuwm_checkout": str(arguments.gpuwm_checkout),
        "control_kind": control_kind(arguments),
        "control_dt_seconds": arguments.control_dt,
    }
    arm_environment = (
        None if not authorization else {AUTHORIZATION_ENVIRONMENT: authorization}
    )

    try:
        # ---- the host half, refused before a card is touched --------------
        expected_steps = int(round(arguments.hours * 3600.0 / dt))
        cumulus_scheme, candidate, control_preflight = host_preflight(arguments)
        result["convection"] = arguments.convection
        result["cumulus_scheme"] = cumulus_scheme
        result["pbl_cadence"] = candidate["pbl_cadence"]
        surface_pbl_seconds = candidate["surface_pbl_seconds"]
        result["schedule_receipt"] = candidate["schedule_receipt"]
        (work / f"schedule-receipt-{label}.json").write_text(
            json.dumps(result["schedule_receipt"], indent=2, sort_keys=True) + "\n",
            encoding="utf-8", newline="\n",
        )
        _log(f"schedule receipt minted for dt={dt:g} s ({expected_steps} steps)")

        # ---- the gate, checked before ANY arm spends card time -------------
        if not arguments.reextract:
            check_authorization(
                dt,
                card=arguments.card,
                authorization=authorization,
                cumulus_scheme=cumulus_scheme,
                surface_pbl_seconds=surface_pbl_seconds,
            )
            if control_preflight is not None:
                check_authorization(
                    float(arguments.control_dt),
                    card=arguments.card,
                    authorization=authorization,
                    cumulus_scheme=cumulus_scheme,
                    surface_pbl_seconds=control_preflight["surface_pbl_seconds"],
                )
        legs = leg_argvs(arguments, label=label)

        # ---- the control arm ------------------------------------------------
        control_band: dict[str, Any] | None = None
        if arguments.control_band is not None:
            control_band = json.loads(
                arguments.control_band.read_text(encoding="utf-8")
            )
            result["control"] = {
                "reused_from": str(arguments.control_band),
                "band": control_band,
            }
            _log(f"control band reused from {arguments.control_band}")
        elif "control" in legs:
            control_out = work / "control" / "out"
            if arguments.reextract:
                rc = 0
            else:
                prepare_arm_directory(work / "control")
                rc = _run(
                    legs["control"], logs / "control.log", arguments.repo,
                    arm_environment if control_preflight is not None else None,
                )
            if rc != 0:
                where = (
                    f"the step-halving control at {float(arguments.control_dt):g} s"
                    if control_preflight is not None
                    else "the control arm at the row's anchored timestep"
                )
                raise CampaignError(
                    f"{where} failed (rc={rc}) on this card.  The candidate's "
                    f"band has nothing to be read against, so this campaign "
                    f"stops here rather than reporting a band with no reference"
                )
            control = arm_summary(control_out)
            control_band = control["band"]
            result["control"] = control
            (work / "control-band.json").write_text(
                json.dumps(control_band, indent=2, sort_keys=True) + "\n",
                encoding="utf-8", newline="\n",
            )

        # ---- the two candidate arms ---------------------------------------
        arms: list[dict[str, Any]] = []
        for name in ("arm-a", "arm-b"):
            out = work / name / "out"
            if arguments.reextract:
                arms.append(arm_summary(out))
                continue
            prepare_arm_directory(work / name)
            rc = _run(legs[name], logs / f"{name}.log", arguments.repo, arm_environment)
            if rc != 0:
                raise CampaignError(
                    f"candidate {name} at dt={dt:g} s failed with rc={rc}; "
                    f"see {logs / (name + '.log')}.  An integration anchor is "
                    f"two arms that BOTH ran, so there is nothing partial to "
                    f"record here"
                )
            arms.append(arm_summary(out))

        result["arms"] = arms
        result["determinism"] = compare_arms(arms[0], arms[1])

        finite = all(arm["band"]["finite_every_step"] for arm in arms)
        complete = all(
            arm["steps_executed"] == arm["steps_requested"] == expected_steps
            for arm in arms
        )
        result["integration_anchor_earned"] = bool(
            finite and complete and result["determinism"]["all_identical"]
        )
        result["why"] = {
            "finite_every_step_both_arms": finite,
            "every_requested_step_executed": complete,
            "expected_steps": expected_steps,
            "history_byte_identical_masked": result["determinism"]["all_identical"],
        }

        if control_band is not None:
            result["health_against_control"] = compare_bands(
                control_band, arms[0]["band"]
            )
        result["what_this_does_not_prove"] = [
            "byte-identity against native MPAS-A: the only native v8.4.1 "
            f"reference this program holds ran at "
            f"{dt_admission.PROVEN_DT_SECONDS:g} s, so no other timestep can "
            "have that half without a fresh native run",
        ]
        if control_preflight is not None:
            result["what_this_does_not_prove"].append(
                "that the band is right in absolute terms: the control is a "
                f"finer UNANCHORED step ({float(arguments.control_dt):g} s) on "
                "the same mesh, so agreement says the solution is converged "
                "in dt, not that either step matches an anchored run"
            )
        calls = result["schedule_receipt"]["cadences"]["cumulus_calls_per_hour"]
        if calls is None:
            result["what_this_does_not_prove"].append(
                "that switching the closure off produces BETTER weather: this "
                "arm removes a forcing, and whether the result has more skill "
                "is a question only obs-skill referees.  What it can settle "
                "is attribution -- whether a band this configuration does not "
                "produce was being produced by the closure"
            )
        else:
            result["what_this_does_not_prove"].append(
                "that Grell-Freitas is RIGHT at this call rate: WRF pins "
                "cudt=0 for cu_physics=3, so the closure is called every step "
                f"and this timestep calls it {calls:g} times an hour against "
                "the proven "
                f"{result['schedule_receipt']['cadences']['cumulus_calls_per_hour_at_proven_dt']:g}. "
                "Only obs-skill referees that"
            )
        result["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    except Exception as error:  # noqa: BLE001 - the marker must record anything
        result["error"] = f"{type(error).__name__}: {redact_authorization(error)}"
        result["integration_anchor_earned"] = False
        (work / f"anchor-{label}.json").write_text(
            redact_authorization(
                json.dumps(result, indent=2, sort_keys=True, default=str)
            ) + "\n",
            encoding="utf-8", newline="\n",
        )
        marker_failed.write_text(result["error"] + "\n", encoding="utf-8", newline="\n")
        _log(f"FAILED: {result['error']}")
        return 1

    (work / f"anchor-{label}.json").write_text(
        # Redacted as a whole: an arm summary lifts the door receipt's
        # candidate row, whose basis names the authorization sentence.
        redact_authorization(
            json.dumps(result, indent=2, sort_keys=True, default=str)
        ) + "\n",
        encoding="utf-8", newline="\n",
    )
    if not result["integration_anchor_earned"]:
        marker_failed.write_text(
            json.dumps(result["why"], sort_keys=True) + "\n",
            encoding="utf-8", newline="\n",
        )
        _log(f"FAILED: integration anchor not earned: {result['why']}")
        return 1
    marker_done.write_text(
        json.dumps(
            {
                "dt_seconds": dt,
                "card": arguments.card,
                "mesh": arguments.mesh,
                "steps_per_arm": expected_steps,
                "frames_identical": result["determinism"]["frames_identical"],
                "timestep_evidence": "experimental-unanchored",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8", newline="\n",
    )
    _log(f"DONE: dt={dt:g} s integration anchor earned on {arguments.card}")
    return 0


def legs_in_order(legs: dict[str, list[str]]) -> list[dict[str, Any]]:
    """The legs as an ordered list, in the order the real run executes them."""

    return [{"leg": name, "argv": argv} for name, argv in legs.items()]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
