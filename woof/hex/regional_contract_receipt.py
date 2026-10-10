"""Local contract-deck receipts and the regional-admission preflight.

A regional (limited-area) CUDA run is admitted on two halves
(:mod:`woof.hex.cuda_backend.regional_admission`): the cull's own CONTRACT
DECK (the geometry half, one receipt per ``bdyMask`` digest) and a FORECAST
MINT of its configuration class (the class half).  A freshly generated cull --
a 100 m corridor placed this cycle -- has neither a ``SHIPPED_CONTRACTS`` row
nor, at those resolutions, a minted class.

This module closes the first gap locally and labels the second:

* :func:`generate_local_contract_receipt` runs the real deck instrument
  (``python -m woof.hex.drivers.run_cuda_regional_contract``) on the cull, or
  takes a deck receipt that instrument already wrote (``--deck-receipt``),
  cross-checks it against the geometry measured here, and STAMPS it with the
  class the run will measure.  The deck's own verdicts are never edited: a
  failing deck stays failing and the gate refuses it by name.
* When that class holds no mint, the stamp is only written with the
  experimental lane open, and it reads ``"class_evidence":
  "experimental-unminted"`` with ``"mint_pair": null``.
* :func:`regional_preflight` measures the class key off the grid/static/init
  files exactly as the regional forecast opener does at run time and runs
  :func:`~woof.hex.cuda_backend.regional_admission.require_regional_anchor`,
  so ``woof hex forecast --preflight`` reports the regional verdict before
  the dycore starts.

Everything here is orchestration: the only arithmetic is a min, a max and a
digest over arrays the file already holds.

Usage (writes the receipt into a ledger directory the gate reads through
``$WOOF_HEX_REGIONAL_CONTRACT_DIR``)::

    python -m woof.hex.regional_contract_receipt \\
        --grid corridor.grid.nc --static corridor.static.nc \\
        --init corridor.init.nc --lbc-dir lbc/ --dt-seconds 0.5 \\
        --experimental-dt --out ledger/corridor.json
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .cuda_backend import regional_admission as ra

SCHEMA = "woof-hex.regional-contract-stamp.v1"
PREFLIGHT_SCHEMA = "woof-hex.regional-preflight.v1"

#: The Earth radius MPAS publishes unit-sphere grids against.  Used only when
#: no static file is supplied and the grid stores unit-sphere ``dcEdge``.
_MPAS_EARTH_RADIUS_M = 6_371_229.0

#: A deck runner takes (grid, init, lbc_dir, out, class_id, mesh_row,
#: start_time) and returns the process exit code after writing ``out``.
DeckRunner = Callable[..., int]


class RegionalReceiptRefusal(RuntimeError):
    """A local contract receipt cannot be produced, and the message says why."""


@dataclass(frozen=True)
class CullGeometry:
    """What the gate reads off a cull, measured from its files."""

    grid: str
    bdy_mask_sha256: str
    n_cells: int
    boundary_zone_width: int
    finest_edge_m: float
    finest_edge_source: str
    n_vert_levels: int
    n_vert_levels_source: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_variable(dataset: Any, name: str) -> np.ndarray | None:
    variable = dataset.variables.get(name)
    if variable is None:
        return None
    variable.set_auto_mask(False)
    return np.asarray(variable[...])


def grid_is_regional(grid: Path | str) -> bool:
    """Does the grid carry any of the bdyMask triple (a regional cull)?"""

    import netCDF4

    from .mesh import REGIONAL_BOUNDARY_MASK_NAMES

    with netCDF4.Dataset(str(grid)) as dataset:
        return any(name in dataset.variables for name in REGIONAL_BOUNDARY_MASK_NAMES)


def measure_cull_geometry(
    grid: Path | str,
    *,
    static: Path | str | None = None,
    init: Path | str | None = None,
    n_vert_levels: int | None = None,
) -> CullGeometry:
    """Measure the regional class inputs off a cull's files.

    The finest edge is ``min(dcEdge)`` from the STATIC file when one is
    given -- the dynamics mesh the forecast builds overlays the static file's
    Earth-scaled metrics onto the grid, so that is the length the run-time
    gate measures.  Without a static file the grid's own ``dcEdge`` is scaled
    by its ``sphere_radius`` to the MPAS Earth radius.  The column count is
    the init file's ``nVertLevels`` when given, else ``n_vert_levels``, else
    the program's standard 55.
    """

    import netCDF4

    from .mesh import REGIONAL_BOUNDARY_MASK_NAMES, regional_boundary_mask_digest

    grid_path = Path(grid)
    if not grid_path.is_file():
        raise RegionalReceiptRefusal(f"the grid {grid_path} does not exist")
    with netCDF4.Dataset(str(grid_path)) as dataset:
        masks = {
            name: _read_variable(dataset, name)
            for name in REGIONAL_BOUNDARY_MASK_NAMES
        }
        missing = [name for name, value in masks.items() if value is None]
        if missing:
            raise RegionalReceiptRefusal(
                f"{grid_path} carries no {', '.join(missing)}: it is not a "
                f"regional cull, so it has no boundary rings for a contract "
                f"deck to measure"
            )
        n_cells = int(len(dataset.dimensions["nCells"]))
        grid_dc_edge = _read_variable(dataset, "dcEdge")
        grid_radius = float(getattr(dataset, "sphere_radius", 0.0) or 0.0)
    digest = regional_boundary_mask_digest(masks)  # type: ignore[arg-type]
    zone_width = int(np.max(np.asarray(masks["bdyMaskCell"])))

    if static is not None:
        with netCDF4.Dataset(str(static)) as dataset:
            dc_edge = _read_variable(dataset, "dcEdge")
        if dc_edge is None:
            raise RegionalReceiptRefusal(f"{static} carries no dcEdge")
        edge_source = f"min(dcEdge) of the static file {Path(static).name}"
        finest = float(np.min(np.asarray(dc_edge, dtype=np.float64)))
    else:
        if grid_dc_edge is None:
            raise RegionalReceiptRefusal(f"{grid_path} carries no dcEdge")
        finest = float(np.min(np.asarray(grid_dc_edge, dtype=np.float64)))
        if grid_radius <= 0.0 or not math.isfinite(grid_radius):
            raise RegionalReceiptRefusal(
                f"{grid_path} carries no positive sphere_radius, so its dcEdge "
                f"cannot be read as metres; pass --static"
            )
        if grid_radius < 1.0e3:
            finest *= _MPAS_EARTH_RADIUS_M / grid_radius
            edge_source = (
                f"min(dcEdge) of the grid scaled from sphere_radius "
                f"{grid_radius:g} to {_MPAS_EARTH_RADIUS_M:g} m"
            )
        else:
            edge_source = "min(dcEdge) of the grid (Earth-scaled)"
    if not math.isfinite(finest) or finest <= 0.0:
        raise RegionalReceiptRefusal(
            f"the cull's finest edge measured {finest!r}, which is not a "
            f"positive length"
        )

    if init is not None:
        with netCDF4.Dataset(str(init)) as dataset:
            if "nVertLevels" not in dataset.dimensions:
                raise RegionalReceiptRefusal(f"{init} carries no nVertLevels")
            levels = int(len(dataset.dimensions["nVertLevels"]))
        levels_source = f"nVertLevels of the init file {Path(init).name}"
    elif n_vert_levels is not None:
        levels = int(n_vert_levels)
        levels_source = "declared"
    else:
        levels = ra.STANDARD_CLASS_LEVELS
        levels_source = "the program's standard column"
    return CullGeometry(
        grid=str(grid_path),
        bdy_mask_sha256=digest,
        n_cells=n_cells,
        boundary_zone_width=zone_width,
        finest_edge_m=finest,
        finest_edge_source=edge_source,
        n_vert_levels=levels,
        n_vert_levels_source=levels_source,
    )


def class_key_for(
    geometry: CullGeometry, dt_seconds: float, *, kernel_set: str | None = None
) -> ra.RegionalClassKey:
    return ra.RegionalClassKey.build(
        boundary_zone_width=geometry.boundary_zone_width,
        n_vert_levels=geometry.n_vert_levels,
        finest_edge_m=geometry.finest_edge_m,
        dt_seconds=float(dt_seconds),
        kernel_set=kernel_set or ra.kernel_set_sha256(),
    )


def resolve_class(
    key: ra.RegionalClassKey, *, experimental: bool | None = None
) -> ra.RegionalClass:
    """The minted class this key is, or (lane open) its experimental class.

    Refuses by name otherwise; never adds to ``ADMITTED_CLASSES``.
    """

    minted = ra.admitted_class_for_key(key)
    if minted is not None:
        return minted
    if not ra.experimental_lane_requested(experimental):
        raise RegionalReceiptRefusal(
            ra.unanchored_class_refusal(key, None)
            + f".  The class this cull measures is "
            f"{ra.class_id_for_key(key)!r}; pass --experimental-dt (or set "
            f"{ra.EXPERIMENTAL_DT_ENVIRONMENT}=1) to stamp its deck receipt "
            f"{ra.CLASS_EVIDENCE_EXPERIMENTAL!r}"
        )
    try:
        return ra.experimental_class_for_key(key)
    except ra.RegionalAdmissionRefusal as error:
        raise RegionalReceiptRefusal(str(error)) from None


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def stamp_contract_receipt(
    deck: Mapping[str, Any],
    *,
    geometry: CullGeometry,
    klass: ra.RegionalClass,
    deck_receipt_sha256: str | None = None,
    mesh_row: str | None = None,
) -> dict[str, Any]:
    """The deck receipt, cross-checked against the geometry and class-stamped.

    The deck's verdict fields are copied, never edited.  A deck run on other
    rings or at another kernel set is refused here rather than stamped,
    because the gate would refuse it anyway and a stamp on it would only
    look like evidence.
    """

    if deck.get("instrument") != "run_cuda_regional_contract":
        raise RegionalReceiptRefusal(
            "the deck receipt was not written by "
            "woof.hex.drivers.run_cuda_regional_contract, so it is not a "
            "contract deck"
        )
    measured = str(deck.get("bdy_mask_sha256") or "")
    if measured != geometry.bdy_mask_sha256:
        raise RegionalReceiptRefusal(
            f"the deck receipt was run on bdyMask digest "
            f"{measured[:16] or '(absent)'}... and this cull's is "
            f"{geometry.bdy_mask_sha256[:16]}...: it measured other rings"
        )
    if deck.get("n_cells") is not None and int(deck["n_cells"]) != geometry.n_cells:
        raise RegionalReceiptRefusal(
            f"the deck receipt was run on {int(deck['n_cells'])} cells and "
            f"this cull carries {geometry.n_cells}"
        )
    deck_mesh = deck.get("mesh") if isinstance(deck.get("mesh"), dict) else {}
    deck_levels = deck_mesh.get("n_vert_levels")
    if deck_levels is not None and int(deck_levels) != geometry.n_vert_levels:
        raise RegionalReceiptRefusal(
            f"the deck receipt was run on {int(deck_levels)} levels and the "
            f"class is being measured at {geometry.n_vert_levels}; every "
            f"regional kernel launches over (levels, elements), so the stamp "
            f"would name a column the deck never ran"
        )
    if str(deck.get("kernel_set_sha256") or "") != klass.key.kernel_set_sha256:
        raise RegionalReceiptRefusal(
            "the deck receipt was produced at another regional kernel set, so "
            "its decks measured bytes this tree would not launch; re-run it"
        )
    stamped = dict(deck)
    experimental = klass.class_evidence == ra.CLASS_EVIDENCE_EXPERIMENTAL
    stamped.update(
        {
            "class_id": klass.class_id,
            "class_evidence": klass.class_evidence,
            "class_key": klass.key.as_dict(),
            "mint_pair": (
                None if experimental else list(klass.mint_receipts)
            ),
            "boundary_zone_width": geometry.boundary_zone_width,
            "mesh_row": mesh_row if mesh_row is not None else deck.get("mesh_row"),
            "stamp": {
                "schema": SCHEMA,
                "tool": "woof.hex.regional_contract_receipt",
                "stamped_utc": datetime.now(timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                ),
                "deck_receipt_sha256": deck_receipt_sha256,
                "geometry": geometry.as_dict(),
                "courant_limit_seconds": ra.regional_courant_limit_seconds(
                    geometry.finest_edge_m
                ),
                "note": (
                    "EXPERIMENTAL: the class this deck is stamped with holds "
                    "no forecast mint pair; only a run with the experimental "
                    "lane open is admitted on it, and its output is "
                    "unanchored."
                    if experimental
                    else "the class this deck is stamped with is a minted row "
                    "of ADMITTED_CLASSES."
                ),
            },
        }
    )
    return stamped


def default_deck_runner(
    *,
    grid: Path,
    init: Path,
    lbc_dir: Path,
    out: Path,
    class_id: str,
    mesh_row: str | None,
    start_time: str | None,
) -> int:
    """Run the shipped deck instrument in its own process (needs a card)."""

    argv = [
        sys.executable, "-m", "woof.hex.drivers.run_cuda_regional_contract",
        "--grid", str(grid), "--init", str(init), "--lbc-dir", str(lbc_dir),
        "--class-id", class_id, "--out", str(out),
    ]
    if mesh_row:
        argv += ["--mesh-row", mesh_row]
    if start_time:
        argv += ["--start-time", start_time]
    return subprocess.run(argv, check=False).returncode


def _write_json(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(document, indent=1, default=str), encoding="utf-8")
    os.replace(temporary, path)


def generate_local_contract_receipt(
    grid: Path | str,
    *,
    out: Path | str,
    dt_seconds: float,
    static: Path | str | None = None,
    init: Path | str | None = None,
    lbc_dir: Path | str | None = None,
    deck_receipt: Path | str | None = None,
    n_vert_levels: int | None = None,
    experimental: bool | None = None,
    mesh_row: str | None = None,
    start_time: str | None = None,
    deck_runner: DeckRunner | None = None,
) -> dict[str, Any]:
    """Produce a cull's class-stamped contract-deck receipt at ``out``.

    Exactly one deck source: ``deck_receipt`` (a receipt the deck instrument
    already wrote for this cull) or ``init`` + ``lbc_dir`` (the deck is run
    now, on a card).  The class is resolved BEFORE any card time is spent, so
    a class that cannot be admitted refuses without running the deck.

    Returns a summary: the class id, its evidence label, whether the deck
    passed, and the receipt path.  A failing deck is still written (the gate
    refuses it by name); ``deck_passed`` says so.
    """

    if (deck_receipt is None) == (init is None or lbc_dir is None):
        raise RegionalReceiptRefusal(
            "pass exactly one deck source: --deck-receipt, or --init with "
            "--lbc-dir to run the contract deck now"
        )
    existing: bytes | None = None
    if deck_receipt is not None:
        existing = Path(deck_receipt).read_bytes()
        if init is None and n_vert_levels is None:
            # The deck read the init file it ran on; its column count is the
            # one the run will measure, so a stamp never assumes 55.
            try:
                recorded = json.loads(existing.decode("utf-8")).get("mesh", {})
                if recorded.get("n_vert_levels") is not None:
                    n_vert_levels = int(recorded["n_vert_levels"])
            except (UnicodeError, json.JSONDecodeError, AttributeError):
                pass  # refused by name below, when the receipt is parsed
    geometry = measure_cull_geometry(
        grid, static=static, init=init, n_vert_levels=n_vert_levels
    )
    key = class_key_for(geometry, dt_seconds)
    klass = resolve_class(key, experimental=experimental)
    out_path = Path(out)

    if existing is not None:
        payload = existing
    else:
        runner = deck_runner or default_deck_runner
        with tempfile.TemporaryDirectory(prefix="regional-deck-") as scratch:
            raw = Path(scratch) / "deck.json"
            code = runner(
                grid=Path(grid), init=Path(init),  # type: ignore[arg-type]
                lbc_dir=Path(lbc_dir), out=raw,  # type: ignore[arg-type]
                class_id=klass.class_id, mesh_row=mesh_row,
                start_time=start_time,
            )
            if not raw.is_file():
                raise RegionalReceiptRefusal(
                    f"the contract deck wrote no receipt (exit {code}); "
                    f"without a deck on THESE rings nothing has checked this "
                    f"cull's specified and relaxation zones"
                )
            payload = raw.read_bytes()
    try:
        deck = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise RegionalReceiptRefusal(
            f"the deck receipt is not readable JSON: {error}"
        ) from None
    if not isinstance(deck, dict):
        raise RegionalReceiptRefusal("the deck receipt is not a JSON object")
    stamped = stamp_contract_receipt(
        deck,
        geometry=geometry,
        klass=klass,
        deck_receipt_sha256=_sha256_bytes(payload),
        mesh_row=mesh_row,
    )
    _write_json(out_path, stamped)
    defects = ra.contract_receipt_defects(
        stamped,
        bdy_mask_sha256=geometry.bdy_mask_sha256,
        n_cells=geometry.n_cells,
        kernel_set=key.kernel_set_sha256,
    )
    return {
        "receipt": str(out_path),
        "class_id": klass.class_id,
        "class_evidence": klass.class_evidence,
        "class_key": key.as_dict(),
        "geometry": geometry.as_dict(),
        "deck_passed": not defects,
        "deck_defects": defects,
    }


def regional_preflight(
    grid: Path | str,
    *,
    dt_seconds: float,
    static: Path | str | None = None,
    init: Path | str | None = None,
    n_vert_levels: int | None = None,
    mesh_row: str | None = None,
    experimental: bool | None = None,
    contract_directories: Sequence[Path | str] = (),
) -> dict[str, Any]:
    """Run the regional anchor gate on files, before any device work.

    The same measurement the regional forecast opener makes at run time
    (zone width, column count, finest edge, timestep, kernel set, bdyMask
    digest and cell count), handed to the same gate.  ``mesh_row`` is passed
    through exactly as the run passes it (the forecast passes ``None``,
    because nothing writes a registry row onto the mesh).

    Returns a receipt block: ``admitted``, the class and its
    ``class_evidence``, the anchor, or the refusal sentence.
    """

    lane = ra.experimental_lane_requested(experimental)
    block: dict[str, Any] = {
        "schema": PREFLIGHT_SCHEMA,
        "experimental_lane": lane,
        "dt_seconds": float(dt_seconds),
    }
    try:
        geometry = measure_cull_geometry(
            grid, static=static, init=init, n_vert_levels=n_vert_levels
        )
    except RegionalReceiptRefusal as error:
        block.update({"admitted": False, "refusal": str(error)})
        return block
    key = class_key_for(geometry, dt_seconds)
    block["geometry"] = geometry.as_dict()
    block["class_key"] = key.as_dict()
    block["measured_class_id"] = ra.class_id_for_key(key)
    try:
        anchor = ra.require_regional_anchor(
            mesh_row,
            bdy_mask_sha256=geometry.bdy_mask_sha256,
            n_cells=geometry.n_cells,
            boundary_zone_width=geometry.boundary_zone_width,
            n_vert_levels=geometry.n_vert_levels,
            finest_edge_m=geometry.finest_edge_m,
            dt_seconds=float(dt_seconds),
            contract_directories=contract_directories,
            kernel_set=key.kernel_set_sha256,
            experimental=lane,
        )
    except ra.RegionalAdmissionRefusal as error:
        block.update({"admitted": False, "refusal": str(error)})
        return block
    block.update(
        {
            "admitted": True,
            "class_id": anchor.class_id,
            "class_evidence": anchor.class_evidence,
            "contract_route": anchor.contract_route,
            "contract_receipt": anchor.contract_receipt,
            "anchor": anchor.as_dict(),
        }
    )
    return block


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m woof.hex.regional_contract_receipt",
        description=(
            "Produce a regional cull's class-stamped contract-deck receipt for "
            "the $WOOF_HEX_REGIONAL_CONTRACT_DIR ledger, or (--preflight-only) "
            "run the regional anchor gate on the files."
        ),
    )
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--static", type=Path, default=None,
                        help="static file whose Earth-scaled dcEdge the run measures")
    parser.add_argument("--init", type=Path, default=None,
                        help="init file (column count; deck input)")
    parser.add_argument("--lbc-dir", type=Path, default=None,
                        help="boundary series; with --init the deck is run now")
    parser.add_argument("--deck-receipt", type=Path, default=None,
                        help="stamp a receipt the deck instrument already wrote")
    parser.add_argument("--dt-seconds", type=float, required=True)
    parser.add_argument("--n-vert-levels", type=int, default=None)
    parser.add_argument("--mesh-row", default=None)
    parser.add_argument("--start-time", default=None, metavar="YYYY-MM-DD_HH:MM:SS")
    parser.add_argument(
        "--experimental-dt", action="store_true", default=None,
        help="open the EXPERIMENTAL lane: an unminted fine class is stamped "
             "'experimental-unminted' (also WOOF_HEX_EXPERIMENTAL_DT=1)")
    parser.add_argument("--out", type=Path, default=None,
                        help="receipt path (required unless --preflight-only)")
    parser.add_argument("--preflight-only", action="store_true",
                        help="run the regional anchor gate and print its verdict")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.preflight_only:
        block = regional_preflight(
            args.grid, dt_seconds=args.dt_seconds, static=args.static,
            init=args.init, n_vert_levels=args.n_vert_levels,
            mesh_row=args.mesh_row, experimental=args.experimental_dt,
        )
        print(json.dumps(block, indent=1, default=str))
        return 0 if block.get("admitted") else 1
    if args.out is None:
        print("--out is required unless --preflight-only", file=sys.stderr)
        return 2
    try:
        summary = generate_local_contract_receipt(
            args.grid, out=args.out, dt_seconds=args.dt_seconds,
            static=args.static, init=args.init, lbc_dir=args.lbc_dir,
            deck_receipt=args.deck_receipt, n_vert_levels=args.n_vert_levels,
            experimental=args.experimental_dt, mesh_row=args.mesh_row,
            start_time=args.start_time,
        )
    except (RegionalReceiptRefusal, ra.RegionalAdmissionRefusal) as error:
        print(f"REFUSED {error}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=1, default=str))
    if summary["class_evidence"] == ra.CLASS_EVIDENCE_EXPERIMENTAL:
        print(
            f"EXPERIMENTAL class {summary['class_id']} is "
            f"{ra.CLASS_EVIDENCE_EXPERIMENTAL}: no forecast mint pair",
            file=sys.stderr,
        )
    return 0 if summary["deck_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
