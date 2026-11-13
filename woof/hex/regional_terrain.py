"""High-resolution terrain in the mesh stage of ``woof hex regional`` (and so of hex ``woof go``).

WHY THIS EXISTS.  ``woof hex static-highres`` (:mod:`woof.hex.static_highres`) replaces a hex
static's 30-arc-second terrain with Copernicus GLO-30 or GEM-15 cell averages, but it must run
after the static is built and BEFORE the vertical grid (which smooths ``ter`` into its height
coordinate) and before the mesh rows pin the static's SHA-256.  A regional forecast that wanted it
had to chain the doors by hand.  ``--terrain-source`` makes that chain one step of the mesh stage.

THE CHAIN (:func:`apply_terrain`), run on a mesh generated on the ``--parent-static`` route, where
the whole parent's static and vertical grid exist:

1. ``static-highres`` on the parent static;
2. the parent vertical grid, re-minted on that static;
3. the cull's static and vertical grid, re-cut from them by the same cull region and engine that cut
   the originals, so the cull grid is untouched;
4. the mesh rows' static bytes and SHA-256 refreshed, and the point-generate receipt stamped.

The originals move to ``<mesh>/orig30s``.  The result is recorded in ``<mesh>/terrain-receipt.json``
and in the regional receipt: the requested source, the dataset that served (GLO-30 where the
experimental GEM-15 does not reach the footprint, with the reason) and the cell counts.

With no ``--terrain-source`` none of this runs and nothing about the plan, the argv or the outputs
changes.
"""

from __future__ import annotations

import dataclasses
import json
import shutil
from pathlib import Path
from typing import Any, Callable, Mapping

from .errors import MpasPortError

SCHEMA = "woof-hex.regional-terrain.v1"
RECEIPT_NAME = "terrain-receipt.json"
#: ``--terrain-max-spacing-km`` defaults to this multiple of the fine spacing, so the refined core is
#: processed and the coarser ramp keeps its 30-arc-second terrain.
DEFAULT_SPACING_FACTOR = 1.25


class RegionalTerrainRefusal(MpasPortError):
    """A terrain request the regional door cannot honour, and the message says why."""


def default_max_spacing_km(fine_dx_m: float) -> float:
    return round(DEFAULT_SPACING_FACTOR * float(fine_dx_m) / 1000.0, 6)


def request_from_arguments(args, *, fine_dx_m: float, has_parent: bool) -> dict[str, Any] | None:
    """The terrain request ``woof hex regional`` was given, or None; every unsupported combination refused."""

    source = getattr(args, "terrain_source", None)
    spacing = getattr(args, "terrain_max_spacing_km", None)
    cache = getattr(args, "terrain_cache_root", None)
    if source is None:
        stray = [flag for flag, value in (("--terrain-max-spacing-km", spacing),
                                          ("--terrain-cache-root", cache)) if value is not None]
        if stray:
            raise RegionalTerrainRefusal(f"{', '.join(stray)} qualify --terrain-source, which was not given")
        return None
    from .static_highres import TERRAIN_CHOICES

    if source not in TERRAIN_CHOICES:
        raise RegionalTerrainRefusal(f"--terrain-source {source!r} is not one of {sorted(TERRAIN_CHOICES)}")
    if has_parent:
        raise RegionalTerrainRefusal(
            "--terrain-source replaces the terrain of a mesh this door generates; a region cut from a kept "
            "hex run (--parent-*) takes the terrain its parent was built with")
    if "mesh" in getattr(args, "skip", ()):
        raise RegionalTerrainRefusal(
            "--terrain-source runs inside the mesh stage, and --skip mesh was given; build the mesh with "
            "the terrain, or drop --terrain-source")
    if getattr(args, "stage", "all") not in ("all", "mesh"):
        mesh_dir = Path(args.work_dir).expanduser().absolute() / "mesh"
        if not (mesh_dir / RECEIPT_NAME).is_file():
            raise RegionalTerrainRefusal(
                f"--stage {args.stage} with --terrain-source needs a mesh already built with that terrain "
                f"({mesh_dir / RECEIPT_NAME} is absent)")
    if spacing is None:
        spacing = default_max_spacing_km(fine_dx_m)
    spacing = float(spacing)
    if not spacing > 0.0 or spacing != spacing or spacing == float("inf"):
        raise RegionalTerrainRefusal(f"--terrain-max-spacing-km {spacing!r} must be a positive spacing")
    return {"source": str(source), "max_spacing_km": spacing,
            "cache_root": None if cache is None else str(Path(cache).expanduser().absolute())}


def require_applied(mesh_dir: Path, request: Mapping[str, Any]) -> dict[str, Any]:
    """The terrain receipt of a mesh already built, refused when it is not the terrain asked for."""

    path = Path(mesh_dir) / RECEIPT_NAME
    if not path.is_file():
        raise RegionalTerrainRefusal(
            f"{mesh_dir} holds a mesh built without --terrain-source {request['source']}; its terrain is "
            "the 30-arc-second static, and replacing it after the vertical grid and rows are built would "
            "leave them on the old terrain.  Use a fresh --work-dir")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if (receipt.get("requested") != request["source"]
            or float(receipt.get("max_spacing_km", -1.0)) != float(request["max_spacing_km"])):
        raise RegionalTerrainRefusal(
            f"{mesh_dir} was built with --terrain-source {receipt.get('requested')} "
            f"(max spacing {receipt.get('max_spacing_km')} km), not {request['source']} "
            f"(max spacing {request['max_spacing_km']} km).  Use a fresh --work-dir")
    return receipt


def _sha256(path: Path) -> str:
    from .mesh_rows import sha256_file

    return sha256_file(path)


def apply_terrain(*, mesh_dir: Path, vertical_spec: Path | str, request: Mapping[str, Any],
                  note: Callable[[str], None] = print, offline: bool = False) -> dict[str, Any]:
    """Run the chain on the mesh generated in ``mesh_dir``; return (and write) the terrain receipt."""

    from . import mesh_rows
    from .cull_door import carry_lineage, cull_one
    from .engines import MESH, resolve
    from .static_highres import run_static_highres
    from .vertical_door import mint_vertical

    mesh_dir = Path(mesh_dir)
    receipts = sorted(mesh_dir.glob("*.point-generate.json"))
    if len(receipts) != 1:
        raise RegionalTerrainRefusal(f"{mesh_dir} holds {len(receipts)} point-generate receipts, not one")
    generated_path = receipts[0]
    generated = json.loads(generated_path.read_text(encoding="utf-8"))
    if generated.get("route") != "parent-static" or not generated["parent"].get("static"):
        raise RegionalTerrainRefusal(
            f"the mesh in {mesh_dir} was generated on the {generated.get('route')!r} route; terrain "
            "replacement needs the parent's own static (--parent-static)")
    vertical = generated.get("vertical")
    if not vertical or not vertical.get("parent_artifact"):
        raise RegionalTerrainRefusal(f"the mesh in {mesh_dir} has no parent vertical grid to re-mint")
    parent_grid = Path(generated["parent"]["grid"])
    parent_static = Path(generated["parent"]["static"])
    parent_vertical = Path(vertical["parent_artifact"])
    cull = generated["cull"]
    cull_grid, cull_static = Path(cull["grid"]), Path(cull["static"])
    cull_vertical = Path(vertical["cull_artifact"])
    region = Path(cull["region"])
    rows_path = Path(generated["rows_file"])

    originals = mesh_dir / "orig30s"
    originals.mkdir(exist_ok=True)
    moved = [parent_static, parent_vertical, cull_static, cull_vertical]
    for path in moved:
        for part in (path, path.with_name(path.name + ".cull-receipt.json"),
                     path.with_name(path.name + ".receipt.json")):
            if part.is_file():
                if (originals / part.name).exists():
                    raise RegionalTerrainRefusal(f"{originals / part.name} exists; terrain was applied here already")
                shutil.move(str(part), originals / part.name)

    try:
        note(f"terrain: static-highres {request['source']} on {parent_static.name}")
        highres = run_static_highres(
            originals / parent_static.name, parent_static, terrain=request["source"], landuse=None,
            cache_root=None if request.get("cache_root") is None else Path(request["cache_root"]),
            offline=offline, max_spacing_km=float(request["max_spacing_km"]),
            receipt_path=mesh_dir / "terrain-static-highres.json")
        note(f"terrain: re-minting the parent vertical grid on {parent_static.name}")
        mint_vertical(grid=parent_grid, static=parent_static, vertical_spec=vertical_spec,
                      output=parent_vertical,
                      receipt=parent_vertical.with_name(parent_vertical.name + ".receipt.json"))
        note("terrain: re-cutting the cull's static and vertical grid")
        engine = resolve(MESH, None)
        for parent_file, out in ((parent_static, cull_static), (parent_vertical, cull_vertical)):
            row = cull_one(engine, parent_file, region, out, clobber=False, provenance_grid=cull_grid)
            row["lineage"] = carry_lineage(parent_file, out, drives_boundaries=False)
    except BaseException:
        # Put the 30-arc-second files back: a refused request leaves the mesh as it was.
        for path in moved:
            for part in (path, path.with_name(path.name + ".cull-receipt.json"),
                         path.with_name(path.name + ".receipt.json")):
                part.unlink(missing_ok=True)
                if (originals / part.name).is_file():
                    shutil.move(str(originals / part.name), part)
        raise

    rows = mesh_rows.read_rows(rows_path)
    names = {generated["parent"]["row"]: parent_static, cull["row"]: cull_static}
    refreshed = []
    for row in rows:
        path = names.get(row.name)
        if path is not None:
            row = dataclasses.replace(row, static_bytes=path.stat().st_size, static_sha256=_sha256(path))
        refreshed.append(row)
    mesh_rows.write_rows(rows_path, refreshed)

    terrain = highres.get("terrain", {})
    sources = highres.get("sources", {})
    selection = sources.get("terrain_source_selection")
    receipt = {
        "schema": SCHEMA,
        "requested": request["source"],
        "max_spacing_km": float(request["max_spacing_km"]),
        # The dataset that served: GLO-30 where an experimental choice fell back (the reason is in
        # ``selection``).
        "used": terrain.get("source", sources.get("terrain_source")),
        "fallback": bool(selection and selection.get("used") != selection.get("requested")),
        "selection": selection,
        "datum": sources.get("terrain_vertical_datum"),
        "attribution": sources.get("terrain_attribution"),
        "license": sources.get("terrain_license"),
        "cells_selected": highres.get("cells_selected"),
        "delta_mean_abs_m": terrain.get("delta_mean_abs_m"),
        "delta_max_abs_m": terrain.get("delta_max_abs_m"),
        "static_highres_receipt": str(mesh_dir / "terrain-static-highres.json"),
        "originals": str(originals),
        "static_sha256": {"parent": _sha256(parent_static), "cull": _sha256(cull_static)},
        "rows_file": str(rows_path),
        "gwd_statistics": "kept on the 30-arc-second static (box statistics by definition)",
    }
    generated["terrain"] = receipt
    generated_path.write_text(json.dumps(generated, indent=2, sort_keys=True, default=str) + "\n",
                              encoding="utf-8", newline="\n")
    (mesh_dir / RECEIPT_NAME).write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


__all__ = ["RECEIPT_NAME", "RegionalTerrainRefusal", "SCHEMA", "apply_terrain", "default_max_spacing_km",
           "request_from_arguments", "require_applied"]
