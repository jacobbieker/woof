"""``woof hex vertical`` -- mint the native-free vertical artifact on a GLOBAL grid.

WHY THIS DOOR EXISTS.  The vertical artifact (``woof.hex.vertical_spec``)
is what lets a limited-area cull be initialised without a native init: it
is minted on the GLOBAL parent from grid + static + a declarative vertical
spec (no meteorology), culled with the same region as the grid and static
(``woof hex cull --parent-vertical``), and handed to ``woof hex init
--capsule/--reference``.  Until now the only way to mint one was inside
``woof hex mesh-plan --point --generate``, so a mesh generated any other way
(a raster density, a polygon, a regional window) had no door to its own
vertical.  This door is that step on its own, for any global grid.

WHAT IT REFUSES.  A regional grid, with the closed-sphere authority's own
words: its cells have neighbours outside the file, and a vertical built by a
routine that assumes every edge has two cells would invent the atmosphere
on the other side.  The refusal is raised BEFORE the build, on the mesh's
own boundary mask and connectivity, so a user is not told after minutes of
geometry what the first edge already said.  The supported route for a
regional case is to mint here on its global parent and cull.

The level count is the SPEC's (``n_vert_levels``), never a constant here;
terrain smoothing is the spec's too (``terrain_smoothing_passes``,
``smooth_surfaces``, ``surface_smoothing_passes``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

from .errors import MpasPortError

#: The closed-sphere authority's refusal, verbatim from ``woof.hex.vertical``
#: (``regional_boundary_metrics``).  Stated once so this door and the build
#: it fronts say the same thing.
REGIONAL_REFUSAL = (
    "the closed-sphere vertical authority does not invent exterior state"
)


class VerticalDoorRefusal(MpasPortError):
    """The vertical artifact cannot be minted here, and the message says why."""


def _refuse(message: str) -> VerticalDoorRefusal:
    print(f"woof hex: {message}", file=sys.stderr)
    return VerticalDoorRefusal(message)


def _require_file(path: Path | None, flag: str) -> Path:
    if path is None:
        raise _refuse(f"{flag} is required")
    resolved = Path(path).expanduser().absolute()
    if not resolved.is_file():
        raise _refuse(f"{flag} {resolved} is not a file")
    return resolved


def regional_reason(grid: Path) -> str | None:
    """Why ``grid`` is not a closed sphere, or ``None`` when it is one.

    Two measurements, either sufficient: a nonzero ``bdyMaskCell`` (the
    cull's own ring labels) and a ``cellsOnEdge`` slot that names no cell
    (``0`` in the 1-based file convention, the cull's sentinel on the
    outermost ring).
    """

    from netCDF4 import Dataset
    import numpy as np

    with Dataset(str(grid)) as dataset:
        dataset.set_auto_maskandscale(False)
        if "bdyMaskCell" in dataset.variables:
            mask = np.asarray(dataset.variables["bdyMaskCell"][:])
            if mask.size and int(mask.max()) > 0:
                return (
                    f"{grid.name} carries a {int(mask.max())}-ring boundary "
                    f"zone (bdyMaskCell)"
                )
        if "cellsOnEdge" in dataset.variables:
            cells_on_edge = np.asarray(dataset.variables["cellsOnEdge"][:])
            if cells_on_edge.size and int(cells_on_edge.min()) <= 0:
                count = int(np.count_nonzero(np.any(cells_on_edge <= 0, axis=1)))
                return (
                    f"{grid.name} has {count} edge(s) with only one cell "
                    f"(cellsOnEdge holds no-cell sentinels)"
                )
    return None


def mint_vertical(
    *,
    grid: Path,
    static: Path,
    vertical_spec: Path,
    output: Path,
    receipt: Path | None = None,
) -> dict[str, Any]:
    """Mint the artifact on a global grid, refusing a regional one by name."""

    from .vertical_spec import VerticalSpec, materialize_vertical_artifact

    grid = _require_file(grid, "--grid")
    static = _require_file(static, "--static")
    vertical_spec = _require_file(vertical_spec, "--vertical-spec")
    output = Path(output).expanduser().absolute()
    if output in (grid, static):
        raise _refuse(
            f"-o {output} would overwrite an input; the artifact is a new file"
        )
    for candidate in (grid, static):
        reason = regional_reason(candidate)
        if reason is not None:
            raise _refuse(
                f"{reason}: {REGIONAL_REFUSAL}.  A limited-area grid has cells "
                f"whose neighbours are outside it.  Mint the vertical on the "
                f"GLOBAL parent with this door and cut it with `woof hex cull "
                f"--parent-vertical`, which carries the parent's vertical onto "
                f"the child cell for cell"
            )
    spec = VerticalSpec.from_file(vertical_spec)
    started = time.perf_counter()
    payload = materialize_vertical_artifact(
        grid=grid,
        static=static,
        spec_path=vertical_spec,
        output=output,
        receipt_path=receipt,
    )
    seconds = time.perf_counter() - started
    receipt_path = (
        Path(receipt).expanduser().absolute()
        if receipt is not None
        else output.with_name(output.name + ".receipt.json")
    )
    return {
        "schema": "gpuwm-hex.vertical-door/v1",
        "grid": str(grid),
        "static": str(static),
        "vertical_spec": str(vertical_spec),
        "vertical_spec_sha256": spec.sha256(),
        "n_vert_levels": int(spec.n_vert_levels),
        "ztop_m": float(spec.ztop_m),
        "terrain_smoothing_passes": int(spec.terrain_smoothing_passes),
        "output": str(output),
        "output_sha256": payload["output"]["sha256"],
        "receipt": str(receipt_path),
        "invariants": payload.get("invariants"),
        "seconds": round(seconds, 3),
    }


def run_vertical(arguments: argparse.Namespace) -> int:
    summary = mint_vertical(
        grid=arguments.grid,
        static=arguments.static,
        vertical_spec=arguments.vertical_spec,
        output=arguments.output,
        receipt=arguments.receipt,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(
        f"NEXT woof hex cull --parent-grid {summary['grid']} --parent-static "
        f"{summary['static']} --parent-vertical {summary['output']} "
        f"--region <REGION.json> --out-dir <DIR>",
        file=sys.stderr,
    )
    return 0


def add_vertical_parser(commands: Any) -> None:
    parser = commands.add_parser(
        "vertical",
        help="mint the native-free vertical artifact on a global grid",
        description=(
            "Build the vertical artifact (zgrid, zz, zxu, zb, zb3 and the "
            "derived geometry) on a GLOBAL grid from grid + static + a "
            "gpuwm-hex.vertical-spec/v1 declaration, with no meteorology. "
            "Cut it for a limited-area case with `woof hex cull "
            "--parent-vertical`. A regional grid is refused: the "
            "closed-sphere vertical authority does not invent exterior state."
        ),
    )
    parser.add_argument(
        "--grid", type=Path, required=True, metavar="FILE",
        help="the global MPAS grid")
    parser.add_argument(
        "--static", type=Path, required=True, metavar="FILE",
        help="the static generated with that grid (carries ter)")
    parser.add_argument(
        "--vertical-spec", type=Path, required=True, metavar="JSON",
        help="gpuwm-hex.vertical-spec/v1 declaration; its n_vert_levels sets "
             "the level count and its smoothing keys the terrain smoothing")
    parser.add_argument(
        "-o", "--output", type=Path, required=True, metavar="FILE",
        help="vertical artifact to write")
    parser.add_argument(
        "--receipt", type=Path, default=None, metavar="FILE",
        help="provenance receipt (default: <output>.receipt.json)")
    parser.set_defaults(handler=run_vertical)


__all__ = [
    "REGIONAL_REFUSAL",
    "VerticalDoorRefusal",
    "add_vertical_parser",
    "mint_vertical",
    "regional_reason",
    "run_vertical",
]
