"""``woof hex cull`` -- cut a limited-area case out of a global one.

WHY THIS DOOR EXISTS, and it is not convenience.

``woof hex init`` cannot build a limited-area initial condition, and the
refusal it raises is correct: ``vertical.py``'s closed-sphere authority meets
a ring-7 one-cell edge and says *"the closed-sphere vertical authority does
not invent exterior state"*.  A limited-area domain has cells whose
neighbours are outside it, and a vertical grid built by a routine that
assumes every edge has two cells would be inventing the atmosphere on the
other side.

The native answer is to run ``init_atmosphere_model`` with
``config_init_case=7`` and ``config_blend_bdy_terrain=YES``, which puts
Fortran back in the chain -- and the 2.5.0 Python/Rust boundary does not
admit that.

THE ANSWER THIS DOOR SHIPS, measured by the swath-as-lam lane on 2026-08-26:
**cull the parent's own init.**  ``rw_mpas_mesh --cull-parent`` subsets any
classic-netCDF file that carries the mesh dimensions, record variables
included, so the same region row that cuts the grid cuts the static and the
init as well.  Culling took **one second**; building the PARENT's own
init with ``woof hex init`` took **775 s** on the 121,182-cell parent
``v4.75.121182`` (2026-08-26, evidence/swath-real-cascade-20260826).
No native regional init has been timed, so this is OUR cost against OUR
cost, not a comparison against native.  It was previously written as a
native regional init, and it is better than the native route rather than
merely faster:

* the child's terrain IS the parent's terrain, cell for cell, so
  ``blend_bdy_terrain`` has nothing to blend and there is no terrain seam to
  make -- the seam is removed by construction, not smoothed;
* the child's vertical grid IS the parent's, so the two runs start
  bit-identical on every cell they share and a limited-area forecast can be
  compared with its parent EXACTLY rather than approximately;
* no vertical authority is asked to invent exterior state, so the refusal
  above never has to be relaxed.

The one thing a cull does not carry is lineage: ``rw_mpas_mesh
--cull-parent`` byte-matches MPAS-Limited-Area v2.2, and that tool writes
exactly ``on_a_sphere`` and ``sphere_radius`` into the regional file.
``rw_mpas_lbc`` reads twelve global attributes off its ``--grid`` and refuses
rather than invent one, so this door carries the parent's own attributes onto
the child before it hands anything on.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from .errors import MpasPortError

#: What a cull produces, and the order it produces it in.  The grid comes
#: first because everything else is checked against it; the init comes last
#: because it is the biggest and a failure earlier should not have paid for
#: it.  The vertical artifact (``woof hex vertical``, minted on the global
#: parent) is cut beside the static so a cull with no global init still gets
#: the parent's vertical cell for cell -- the road ``woof hex init
#: --capsule/--reference`` takes with a regional meteorological source.
CULL_ROLES: tuple[tuple[str, str], ...] = (
    ("grid", "grid"),
    ("static", "static"),
    ("vertical", "vertical"),
    ("init", "init"),
)

#: The global attribute ``woof.hex.vertical_spec`` stamps on every vertical
#: artifact; a ``--parent-vertical`` without it is not one.
VERTICAL_ARTIFACT_ATTRIBUTE = "gpuwm_hex_vertical_artifact_schema"

#: The global attributes ``rw_mpas_lbc`` reads off its ``--grid`` and refuses
#: rather than invent, transcribed from ``rw-mpas/src/lbc/emit.rs:67-136``.
#: A culled file that lacks one of these cannot drive its own boundary, and
#: this door says so at the cull rather than at the boundary build.
LBC_REQUIRED_ATTRIBUTES: tuple[str, ...] = (
    "model_name",
    "core_name",
    "version",
    "source",
    "Conventions",
    "git_version",
    "on_a_sphere",
    "sphere_radius",
    "is_periodic",
    "x_period",
    "y_period",
    "mesh_spec",
    "file_id",
)


class CullRefusal(MpasPortError):
    """The cull cannot be performed, and the message says what would break."""


def _refuse(message: str) -> CullRefusal:
    print(f"woof hex: {message}", file=sys.stderr)
    return CullRefusal(message)


def _require_file(path: Path | None, flag: str, why: str) -> Path:
    if path is None:
        raise _refuse(f"{flag} was not given: {why}")
    resolved = Path(path).expanduser().absolute()
    if not resolved.is_file():
        raise _refuse(f"{flag} {resolved} is not a file")
    return resolved


def resolve_mesh_engine(explicit: Path | None) -> Path:
    """``rw_mpas_mesh``, through the one ladder every door resolves by."""

    from .engines import MESH, EngineRefusal, resolve

    try:
        return resolve(MESH, explicit)
    except EngineRefusal as error:
        raise _refuse(str(error)) from error


def cull_one(
    engine: Path,
    parent: Path,
    region: Path,
    out: Path,
    *,
    graph: Path | None = None,
    clobber: bool = False,
) -> dict[str, Any]:
    """Cut one file, and return what the engine said it did."""

    receipt = out.with_suffix(out.suffix + ".cull-receipt.json")
    argv = [
        str(engine),
        "--cull-parent",
        str(parent),
        "--region",
        str(region),
        "--out",
        str(out),
        "--receipt",
        str(receipt),
    ]
    if graph is not None:
        argv += ["--graph", str(graph)]
    if clobber:
        argv.append("--clobber")
    started = time.perf_counter()
    completed = subprocess.run(argv, capture_output=True, text=True)
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        sys.stderr.write(completed.stdout)
        sys.stderr.write(completed.stderr)
        raise _refuse(
            f"rw_mpas_mesh --cull-parent refused {parent.name} "
            f"(exit {completed.returncode}); the message above is the "
            f"engine's own and names what it could not do"
        )
    row: dict[str, Any] = {
        "parent": str(parent),
        "out": str(out),
        "seconds": elapsed,
        "argv": argv,
    }
    if receipt.is_file():
        row["receipt"] = json.loads(receipt.read_text(encoding="utf-8"))
    return row


def port_identity() -> dict[str, str]:
    """This program's own identity, for a parent that carries none.

    ``model_name`` names THIS program.  Writing ``mpas`` would be the false
    stamp the boundary producer's refusal exists to prevent: these files
    carry WOOF's physics and this port's numerics, and a stream stamped
    ``mpas`` would claim a provenance no file in the chain has.
    """

    from . import __version__

    return {
        "model_name": "woof hex",
        "core_name": "atmosphere",
        "version": str(__version__),
        "git_version": f"gpuwm-hex-{__version__}",
    }


def carry_lineage(
    parent: Path, child: Path, *, drives_boundaries: bool = False
) -> dict[str, Any]:
    """Put the parent's global attributes onto the child.

    ``rw_mpas_mesh --cull-parent`` reproduces MPAS-Limited-Area v2.2 byte for
    byte, and that tool writes ``on_a_sphere`` and ``sphere_radius`` and
    nothing else.  ``rw_mpas_lbc`` reads twelve attributes off its ``--grid``
    and refuses rather than invent one, so a cull that carried only what the
    culler writes could never drive its own boundary.  The child's lineage IS
    the parent's: a cull moves no cell centre, invents no field and changes
    no configuration.
    """

    from netCDF4 import Dataset

    def plain(value: Any) -> Any:
        try:
            return value.item()
        except AttributeError:
            return value

    carried: dict[str, Any] = {}
    with Dataset(str(parent), "r") as source:
        available = {name: source.getncattr(name) for name in source.ncattrs()}
    with Dataset(str(child), "a") as target:
        present = set(target.ncattrs())
        for name, value in available.items():
            if name in present:
                continue
            target.setncattr(name, value)
            carried[name] = plain(value)
        final = sorted(target.ncattrs())
    missing = [name for name in LBC_REQUIRED_ATTRIBUTES if name not in final]
    # The requirement lands on the file that will be handed to
    # ``rw_mpas_lbc --grid``, which is the INIT.  A grid file carries the
    # MESH's lineage and never the model's -- no mesh generator writes
    # ``model_name`` -- so demanding it of one refuses a perfectly good cull
    # for not being an initial condition.
    minted: dict[str, str] = {}
    if missing and drives_boundaries:
        # A parent minted BEFORE 2026-08-26 carries no model lineage, because
        # rw_mpas_init did not write one.  The engine writes it now, so this
        # branch is for archived parents only -- and every attribute it
        # supplies is recorded as MINTED rather than carried, so a reader can
        # tell an old parent from a new one by reading the receipt.
        identity = port_identity()
        supplied = {name: identity[name] for name in missing if name in identity}
        if supplied:
            with Dataset(str(child), "a") as target:
                for name, value in supplied.items():
                    target.setncattr(name, value)
                final = sorted(target.ncattrs())
            minted = supplied
            print(
                f"woof hex: advisory: {parent.name} predates the engine "
                f"lineage block (2026-08-26) and carries no "
                f"{sorted(supplied)}; this cull stamps them with THIS "
                f"program's identity so the child can drive its own "
                f"boundary.  A parent re-minted with a current "
                f"`woof hex init` carries them and nothing is stamped.",
                file=sys.stderr,
            )
        missing = [
            name for name in LBC_REQUIRED_ATTRIBUTES if name not in final
        ]
        if missing:
            raise _refuse(
                f"{child.name} still lacks {missing} after carrying "
                f"everything {parent.name} had and stamping this program's "
                f"own identity.  rw_mpas_lbc reads those off its --grid and "
                f"refuses rather than invent one, so this cull could not "
                f"drive its own boundary"
            )
    return {
        "carried_from_parent": carried,
        "minted_from_port_identity": minted,
        "parent_predates_engine_lineage": sorted(minted),
        "child_attributes_after": final,
        "boundary_producer_requirements_checked": bool(drives_boundaries),
        "boundary_producer_requirements_missing": missing,
    }


def check_parent_vertical(grid: Path, vertical: Path) -> dict[str, Any]:
    """Refuse a ``--parent-vertical`` that is not the parent grid's artifact.

    The culler subsets any file that carries the mesh dimensions, so a
    vertical minted on a DIFFERENT mesh of the same size would be cut without
    complaint and hand the child another mesh's terrain-following levels.
    Measured here: the artifact must carry the vertical-artifact stamp, the
    parent's nCells and nEdges, and -- where both carry cell centres -- the
    parent's cell centres bit for bit.
    """

    from netCDF4 import Dataset
    import numpy as np

    with Dataset(str(vertical)) as artifact, Dataset(str(grid)) as parent:
        artifact.set_auto_maskandscale(False)
        parent.set_auto_maskandscale(False)
        if VERTICAL_ARTIFACT_ATTRIBUTE not in artifact.ncattrs():
            raise _refuse(
                f"--parent-vertical {vertical.name} carries no "
                f"{VERTICAL_ARTIFACT_ATTRIBUTE}; it is not a vertical artifact. "
                f"Mint one on the global parent with `woof hex vertical`"
            )
        sizes: dict[str, tuple[int | None, int | None]] = {}
        for name in ("nCells", "nEdges"):
            have = artifact.dimensions.get(name)
            want = parent.dimensions.get(name)
            sizes[name] = (
                None if have is None else len(have),
                None if want is None else len(want),
            )
        if any(have != want for have, want in sizes.values()):
            raise _refuse(
                f"--parent-vertical {vertical.name} has "
                f"nCells/nEdges {[v[0] for v in sizes.values()]} and the "
                f"parent grid {grid.name} has {[v[1] for v in sizes.values()]}; "
                f"the artifact was minted on another mesh"
            )
        checked = False
        if "latCell" in artifact.variables and "latCell" in parent.variables:
            for name in ("latCell", "lonCell"):
                if name not in artifact.variables or name not in parent.variables:
                    continue
                have = np.asarray(artifact.variables[name][:], dtype=np.float64)
                want = np.asarray(parent.variables[name][:], dtype=np.float64)
                if have.shape != want.shape or not np.allclose(
                    have, want, rtol=0.0, atol=1.0e-6
                ):
                    raise _refuse(
                        f"--parent-vertical {vertical.name} and the parent "
                        f"grid {grid.name} disagree on {name}; the artifact "
                        f"was minted on another mesh of the same size"
                    )
            checked = True
        n_levels = (
            len(artifact.dimensions["nVertLevels"])
            if "nVertLevels" in artifact.dimensions else None
        )
        schema = str(artifact.getncattr(VERTICAL_ARTIFACT_ATTRIBUTE))
        spec_sha = (
            str(artifact.getncattr("gpuwm_hex_vertical_spec_sha256"))
            if "gpuwm_hex_vertical_spec_sha256" in artifact.ncattrs() else None
        )
    return {
        "artifact_schema": schema,
        "vertical_spec_sha256": spec_sha,
        "n_vert_levels": n_levels,
        "cell_centres_checked": checked,
    }


def run_cull(arguments: argparse.Namespace) -> int:
    started = time.monotonic()
    engine = resolve_mesh_engine(getattr(arguments, "engine", None))
    region = _require_file(
        getattr(arguments, "region", None),
        "--region",
        "a cull is a SHAPE applied to a parent, and there is no default "
        "shape. Pass the cull-region document the swath placement layer "
        "emitted, or a row of your own: "
        '{"kind": "polygon"|"cap"|"lat_lon_box", ...}',
    )
    out_dir = Path(
        getattr(arguments, "out_dir", None) or "."
    ).expanduser().absolute()
    if not out_dir.is_dir():
        raise _refuse(
            f"--out-dir {out_dir} does not exist; create it before the cull "
            "rather than discovering a typo as a new directory full of files"
        )
    name = str(getattr(arguments, "name", None) or "regional")

    parents: dict[str, Path] = {}
    for role, _ in CULL_ROLES:
        given = getattr(arguments, f"parent_{role}", None)
        if given is None:
            continue
        parents[role] = _require_file(
            given, f"--parent-{role}", f"the parent {role} file"
        )
    if "grid" not in parents:
        raise _refuse(
            "--parent-grid is required: the grid carries the topology every "
            "other file is subset against, and a static or init cut without "
            "it would have no mesh to be a mesh of"
        )
    vertical_check: dict[str, Any] | None = None
    if "vertical" in parents:
        vertical_check = check_parent_vertical(parents["grid"], parents["vertical"])
    if "init" not in parents:
        print(
            "woof hex: advisory: no --parent-init was given, so this cull "
            "produces a mesh and no initial condition.  `woof hex init` "
            "REFUSES a limited-area grid by name -- its closed-sphere "
            "vertical authority does not invent exterior state -- so the "
            "supported way to get one is to cull the parent's init here"
            + (
                ", or to initialise the cut from the culled vertical artifact "
                "with `woof hex init --capsule/--reference`."
                if "vertical" in parents
                else ", or to pass --parent-vertical (minted on this parent "
                "with `woof hex vertical`) and initialise from it."
            ),
            file=sys.stderr,
        )

    from .mesh_rows import sha256_file

    rows: dict[str, Any] = {}
    for role, _ in CULL_ROLES:
        if role not in parents:
            continue
        out = out_dir / f"{name}.{role}.nc"
        graph = out_dir / f"{name}.graph.info" if role == "grid" else None
        rows[role] = cull_one(
            engine,
            parents[role],
            region,
            out,
            graph=graph,
            clobber=bool(getattr(arguments, "clobber", False)),
        )
        cells = rows[role].get("receipt", {}).get("region_cells")
        parent_cells = rows[role].get("receipt", {}).get("parent_cells")
        print(
            f"CULL {role} {parents[role].name} -> {out.name} "
            f"{parent_cells} -> {cells} cells {rows[role]['seconds']:.2f}s",
            flush=True,
        )
        rows[role]["lineage"] = carry_lineage(
            parents[role], out, drives_boundaries=(role == "init")
        )
        # Digests AFTER the lineage carry, because that is the file a row
        # pins: `woof hex register --cull-receipt` checks the grid and static
        # it is handed against these, and the parent's against its row.
        rows[role]["parent_sha256"] = sha256_file(parents[role])
        rows[role]["out_sha256"] = sha256_file(out)
        if role == "vertical" and vertical_check is not None:
            rows[role]["parent_vertical"] = vertical_check

    receipt_path = Path(
        getattr(arguments, "receipt", None) or out_dir / f"{name}.cull.json"
    ).expanduser().absolute()
    try:
        region_document = json.loads(region.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        region_document = None
    receipt = {
        "schema": "gpuwm-hex.cull-door/v1",
        "name": name,
        "region": str(region),
        "region_sha256": sha256_file(region),
        "region_document": region_document,
        "engine": str(engine),
        "out_dir": str(out_dir),
        "files": rows,
        "seconds": time.monotonic() - started,
        "why_this_route": (
            "woof hex init refuses a limited-area grid by name -- the "
            "closed-sphere vertical authority does not invent exterior state "
            "-- and the native alternative puts init_atmosphere_model back in "
            "the chain.  Culling the parent's own init is faster (1 s against "
            "775 s to build the parent's own init on v4.75.121182), needs no "
            "Fortran, and removes the "
            "terrain-blend seam by construction: the child's terrain IS the "
            "parent's, so blend_bdy_terrain has nothing to blend"
        ),
    }
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"RECEIPT {receipt_path}", flush=True)
    grid = rows.get("grid", {}).get("out")
    if grid:
        print(
            f"NEXT woof hex register --grid {grid} --static "
            f"{rows.get('static', {}).get('out', '<CULLED-STATIC.nc>')} "
            f"--parent-row <PARENT-ROW> --cull-receipt {receipt_path} "
            f"--name <ROW> --rows <mesh-rows.json>",
            flush=True,
        )
        vertical_out = rows.get("vertical", {}).get("out")
        if vertical_out and "init" not in rows:
            print(
                f"NEXT woof hex init --met <MET/FILE:YYYY-MM-DD_HH> --static "
                f"{rows.get('static', {}).get('out', '<CULLED-STATIC.nc>')} "
                f"--capsule {vertical_out} --reference {vertical_out} "
                f"--out <CULLED-INIT.nc> --start-time ...",
                flush=True,
            )
        print(
            "NEXT rw_mpas_lbc --source unstructured-port-stream "
            f"--grid {rows.get('init', {}).get('out', '<CULLED-INIT.nc>')} "
            "--parent-grid <PARENT-INIT.nc> --out-dir <LBC-DIR> "
            "--start-time ... --stop-time ... --interval <TIME>=<PARENT-FRAME>",
            flush=True,
        )
        print(
            f"NEXT woof hex forecast --mesh <ROW> --grid {grid} "
            f"--static {rows.get('static', {}).get('out', '<CULLED-STATIC.nc>')} "
            f"--init {rows.get('init', {}).get('out', '<CULLED-INIT.nc>')} "
            "--lbc-dir <LBC-DIR> --hours H --history-every-minutes M "
            "--out <DIR> --gpuwm-checkout <GPUWM>",
            flush=True,
        )
    return 0


def add_cull_parser(commands: Any) -> None:
    parser = commands.add_parser(
        "cull",
        help="cut a limited-area grid, static, vertical artifact and initial condition out of a global case",
        description=(
            "Cut a limited-area case out of a global one with "
            "rw_mpas_mesh --cull-parent. This is the SUPPORTED way to get a "
            "limited-area initial condition: `woof hex init` refuses a "
            "regional grid by name, because its closed-sphere vertical "
            "authority does not invent exterior state, and culling the "
            "parent's own init is both faster and free of the terrain-blend "
            "seam a native regional init has to smooth."
        ),
    )
    parser.add_argument(
        "--parent-grid", type=Path, default=None, metavar="FILE",
        help="the global grid to cut from; required")
    parser.add_argument(
        "--parent-static", type=Path, default=None, metavar="FILE",
        help="the global static file generated with that grid")
    parser.add_argument(
        "--parent-vertical", type=Path, default=None, metavar="FILE",
        help="the global vertical artifact minted on that grid with `woof hex "
             "vertical`; cut beside the static so the child's vertical IS the "
             "parent's, cell for cell, and the cut can be initialised from it")
    parser.add_argument(
        "--parent-init", type=Path, default=None, metavar="FILE",
        help="the global initial condition. CUTTING THIS IS THE POINT: it is "
             "how a limited-area run gets an init at all")
    parser.add_argument(
        "--region", type=Path, default=None, metavar="FILE",
        help="Shape row naming the piece to keep -- polygon, cap or "
             "lat_lon_box. The swath placement layer emits one per swath")
    parser.add_argument(
        "--out-dir", type=Path, default=None, metavar="DIR",
        help="existing directory for the culled files (default: .)")
    parser.add_argument(
        "--name", default=None, metavar="TEXT",
        help="file-name stem for the cut files (default: regional)")
    parser.add_argument(
        "--clobber", action="store_true",
        help="replace culled files that already exist")
    parser.add_argument(
        "--engine", type=Path, default=None, metavar="FILE",
        help="rw_mpas_mesh executable (default: the shared engine ladder)")
    parser.add_argument(
        "--receipt", type=Path, default=None, metavar="FILE",
        help="provenance receipt path (default: <out-dir>/<name>.cull.json)")
    parser.set_defaults(handler=run_cull)
