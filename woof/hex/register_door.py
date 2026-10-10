"""``woof hex register`` -- register ANY generated global mesh, or a cull of one.

THE GAP THIS CLOSES.  A runtime mesh row (:mod:`woof.hex.mesh_rows`) was
written in exactly one place: ``woof hex mesh-plan --point --generate``,
right after it built a nested-cap pair at a point.  A mesh generated any
other way -- ``woof mesh`` from a raster density, a polygon, a regional
window, a uniform background -- had no door to a row, so ``woof hex
forecast --mesh`` could never resolve it without a checkout edit, which is
the hand-written-row defect the runtime rows exist to retire.

WHAT THIS DOOR DOES.  It takes a grid and its static, runs the SAME
admission pass the point door runs on a pair it just built
(:func:`woof.hex.mesh_point.admit_pair`: dual edges, cell coordination,
Courant and the timestep anchor), and appends a row to a row file:

* a ``generated-global`` row, with the spec it was generated from named by
  digest (``--spec``, or the ``rw_mesh_spec_json`` attribute the generator
  stamps on every grid it writes) and its nominal dx read off the static's
  own ``nominalMinDc`` -- the value the bind compares FP32-exactly;
* a ``generated-cull`` row (``--parent-row`` and ``--cull-receipt``), whose
  parent must already be a ``generated-global`` row IN THE SAME FILE and
  whose grid, static and parent bytes must match the digests the
  ``woof hex cull`` receipt recorded.

WHAT IT REFUSES, by name: a cull with no parent row, a parent that is not in
the file, a parent whose bytes moved, a cull receipt whose digests do not
match the files handed in, a global grid with a boundary zone, a regional
grid registered as global, a level count that disagrees with the parent's,
and any pair the admission pass refuses.

THE TIMESTEP.  With no ``--dt-seconds`` the largest anchored timestep the
mesh's Courant limit admits is chosen (a cull defaults to its parent's).
An explicit one must pass Courant AND hold an ``ADMITTED_TIMESTEPS`` row,
refused with that table's own message otherwise.  The Python API's
``experimental=True`` skips the anchor lookup ONLY and labels the row
``experimental-unanchored``; this door adds no command-line switch for it
(the experimental lane owns that switch).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

from .errors import MpasPortError


class RegisterRefusal(MpasPortError):
    """A mesh cannot be registered, and the message says why."""


def _refuse(message: str) -> RegisterRefusal:
    return RegisterRefusal(message)


def _require_file(path: Path | str | None, flag: str) -> Path:
    if path is None:
        raise _refuse(f"{flag} is required")
    resolved = Path(path).expanduser().absolute()
    if not resolved.is_file():
        raise _refuse(f"{flag} {resolved} is not a file")
    return resolved


def resolve_rows_path(explicit: Path | str | None) -> Path:
    """``--rows``, else the ONE file ``$WOOF_HEX_MESH_ROWS`` names."""

    from .mesh_rows import MESH_ROWS_ENVIRONMENT, row_files

    if explicit is not None:
        return Path(explicit).expanduser().absolute()
    named = row_files(None)
    if not named:
        raise _refuse(
            f"no row file: pass --rows FILE or set ${MESH_ROWS_ENVIRONMENT}. "
            f"A row written nowhere the forecast door reads is a row nobody "
            f"can bind"
        )
    if len(named) > 1:
        raise _refuse(
            f"${MESH_ROWS_ENVIRONMENT} names {len(named)} files "
            f"({', '.join(str(p) for p in named)}); this door appends to one "
            f"and will not guess which.  Pass --rows FILE"
        )
    return named[0].expanduser().absolute()


def shipped_mesh_names() -> set[str]:
    """The names the shipped registry holds before any runtime row is applied.

    Read out of ``drivers/mpas_mesh_binding.py`` the way the forecast door
    reads it (by path), with the runtime-row and cascade-row environment
    cleared for the load, so the answer is the checkout's own rows and
    never a row file's.  A runtime row that reuses one of these names makes
    ``apply_rows`` refuse the WHOLE file at import, so it is refused here,
    before it is written.
    """

    import importlib.util

    from .cascade_row import CASCADE_ROWS_ENVIRONMENT
    from .mesh_rows import MESH_ROWS_ENVIRONMENT

    path = Path(__file__).resolve().parent / "drivers" / "mpas_mesh_binding.py"
    saved = {
        key: os.environ.pop(key)
        for key in (MESH_ROWS_ENVIRONMENT, CASCADE_ROWS_ENVIRONMENT)
        if key in os.environ
    }
    name = "_woof_hex_register_shipped_registry"
    try:
        specification = importlib.util.spec_from_file_location(name, path)
        if specification is None or specification.loader is None:  # pragma: no cover
            raise _refuse(f"{path} could not be loaded to read the shipped registry")
        module = importlib.util.module_from_spec(specification)
        sys.modules[name] = module
        specification.loader.exec_module(module)
        return set(module.MESH_BINDINGS)
    finally:
        sys.modules.pop(name, None)
        os.environ.update(saved)


def _refuse_name_collisions(name: str, rows_path: Path) -> None:
    from .mesh_rows import read_rows, row_files

    if name in shipped_mesh_names():
        raise _refuse(
            f"--name {name!r} is a shipped registry row.  A runtime row may "
            f"never shadow one, and the bind refuses the whole row file when "
            f"one tries, so every row beside it would stop binding too"
        )
    for other in row_files(None):
        other = other.expanduser().absolute()
        if other == rows_path or not other.is_file():
            continue
        if any(row.name == name for row in read_rows(other)):
            raise _refuse(
                f"--name {name!r} is already registered in {other}, which "
                f"$WOOF_HEX_MESH_ROWS also names; two files registering one "
                f"name refuse at bind"
            )


def static_nominal_dx_m(static: Path) -> float:
    """The static's ``nominalMinDc``, as the FP32 value the bind compares."""

    from netCDF4 import Dataset
    import numpy as np

    with Dataset(str(static)) as dataset:
        dataset.set_auto_maskandscale(False)
        variable = dataset.variables.get("nominalMinDc")
        if variable is None:
            raise _refuse(
                f"{static} carries no nominalMinDc; the bind compares the "
                f"row's nominal dx FP32-exactly against it, so a static "
                f"without one cannot be registered (rw_mpas_static writes it)"
            )
        radius = float(getattr(dataset, "sphere_radius", 0.0) or 0.0)
        value = float(np.float32(np.asarray(variable[:]).ravel()[0]))
    if not value > 0.0 or (radius and radius <= 1.0):
        raise _refuse(
            f"{static} declares nominalMinDc={value!r} on a sphere of radius "
            f"{radius!r}; that is a unit-sphere GRID file, not an "
            f"Earth-scaled static.  Pass the static rw_mpas_static built"
        )
    return value


def vertical_levels(
    vertical_spec: Path | None, vertical: Path | None
) -> int | None:
    """``n_vert_levels`` from a vertical spec and/or artifact, agreeing."""

    found: dict[str, int] = {}
    if vertical_spec is not None:
        from .vertical_spec import VerticalSpec

        found["--vertical-spec"] = int(VerticalSpec.from_file(vertical_spec).n_vert_levels)
    if vertical is not None:
        from netCDF4 import Dataset

        with Dataset(str(vertical)) as dataset:
            if "nVertLevels" not in dataset.dimensions:
                raise _refuse(f"--vertical {vertical} declares no nVertLevels")
            found["--vertical"] = len(dataset.dimensions["nVertLevels"])
    if len(set(found.values())) > 1:
        raise _refuse(
            f"the level counts disagree: {found}; a row pins one column"
        )
    return next(iter(found.values()), None)


def default_levels() -> int:
    """The vertical spec's own default column, when nothing names one."""

    from .vertical_spec import VerticalSpec

    return int(VerticalSpec().n_vert_levels)


def grid_spec(grid: Path) -> tuple[dict[str, Any] | None, str | None]:
    """The spec ``rw_mpas_mesh`` stamped on the grid, and its digest."""

    from netCDF4 import Dataset

    with Dataset(str(grid)) as dataset:
        if "rw_mesh_spec_json" not in dataset.ncattrs():
            return None, None
        text = str(dataset.getncattr("rw_mesh_spec_json"))
        has_receipt = "rw_mesh_receipt_json" in dataset.ncattrs()
    try:
        spec = json.loads(text)
    except json.JSONDecodeError as error:
        raise _refuse(f"{grid} carries an unreadable rw_mesh_spec_json: {error}") from error
    spec = dict(spec) if isinstance(spec, Mapping) else {"spec": spec}
    spec["_has_generator_receipt"] = has_receipt
    return spec, hashlib.sha256(text.encode("utf-8")).hexdigest()


def _numbers_as_float(document: Any) -> Any:
    if isinstance(document, bool) or document is None or isinstance(document, str):
        return document
    if isinstance(document, (int, float)):
        return float(document)
    if isinstance(document, Mapping):
        return {str(k): _numbers_as_float(v) for k, v in document.items()}
    if isinstance(document, (list, tuple)):
        return [_numbers_as_float(v) for v in document]
    return document


def _canonical_spec(document: Mapping[str, Any]) -> str:
    """A spec as the generator READ it: numbers as floats, defaults filled.

    ``rw_mpas_mesh`` stamps the spec it parsed (serde), so ``1000`` reads
    back as ``1000.0`` and an omitted ``regions`` or ``name`` comes back as
    ``[]`` or ``null``; nulls are dropped on both sides so an optional field
    the generator writes as null compares equal to one the file omits.
    """

    def strip_nulls(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {k: strip_nulls(v) for k, v in value.items() if v is not None}
        if isinstance(value, list):
            return [strip_nulls(v) for v in value]
        return value

    filled = {"regions": [], **{k: v for k, v in document.items()
                                if not str(k).startswith("_")}}
    return json.dumps(
        strip_nulls(_numbers_as_float(filled)), sort_keys=True, separators=(",", ":")
    )


def _region_summary(spec: Mapping[str, Any], source: str) -> dict[str, Any]:
    regions = spec.get("regions") or []
    kinds: list[str] = []
    for item in regions:
        shape = item.get("shape") if isinstance(item, Mapping) else None
        if isinstance(shape, Mapping):
            kinds.append(str(shape.get("kind", "?")))
        elif isinstance(shape, str):
            kinds.append(shape)
    return {
        "source": source,
        "name": spec.get("name"),
        "background_km": spec.get("background_km"),
        "region_kinds": kinds,
        "spec": {k: v for k, v in spec.items() if not str(k).startswith("_")},
    }


def _admit(grid: Path, static: Path, *, fine_dx_m: float, dt_seconds: float | None,
           experimental: bool, log) -> dict[str, Any]:
    from .mesh_point import PointPlanRefusal, admit_pair

    try:
        return admit_pair(
            grid, static, fine_dx_m=fine_dx_m, log=log,
            dt_seconds=dt_seconds, experimental=experimental,
        )
    except PointPlanRefusal as error:
        raise _refuse(f"{grid.name} is NOT registered: {error}") from error


def _read_cull_receipt(path: Path) -> dict[str, Any]:
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise _refuse(f"--cull-receipt {path} is not readable JSON: {error}") from error
    if not isinstance(receipt, dict) or receipt.get("schema") != "gpuwm-hex.cull-door/v1":
        raise _refuse(
            f"--cull-receipt {path} is not a gpuwm-hex.cull-door/v1 receipt "
            f"(schema {receipt.get('schema') if isinstance(receipt, dict) else None!r}); "
            f"pass the <name>.cull.json `woof hex cull` writes"
        )
    return receipt


def _cull_region(receipt: Mapping[str, Any], receipt_path: Path) -> dict[str, Any]:
    document = receipt.get("region_document")
    if isinstance(document, Mapping):
        return dict(document)
    region = receipt.get("region")
    if region:
        try:
            loaded = json.loads(Path(region).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise _refuse(
                f"the cull receipt {receipt_path} names region {region}, which "
                f"cannot be read ({error}); a cull row records the shape it cut"
            ) from error
        if isinstance(loaded, Mapping):
            return dict(loaded)
    raise _refuse(f"the cull receipt {receipt_path} records no region")


def _check_cull_digests(
    receipt: Mapping[str, Any], receipt_path: Path, *, grid: Path, static: Path,
    parent: Any,
) -> None:
    from .mesh_rows import sha256_file

    files = receipt.get("files") or {}
    for role, handed, parent_sha in (
        ("grid", grid, parent.grid_sha256),
        ("static", static, parent.static_sha256),
    ):
        entry = files.get(role)
        if not isinstance(entry, Mapping):
            raise _refuse(
                f"the cull receipt {receipt_path} records no {role}; a cull "
                f"row pins a grid AND a static cut by the same cull"
            )
        out_sha = entry.get("out_sha256")
        cut_from = entry.get("parent_sha256")
        if not out_sha or not cut_from:
            raise _refuse(
                f"the cull receipt {receipt_path} records no digests for its "
                f"{role} (written before `woof hex cull` recorded them); re-run "
                f"the cull so the row can prove which bytes it cut"
            )
        measured = sha256_file(handed)
        if measured != out_sha:
            raise _refuse(
                f"--{role} {handed} digests to {measured[:16]}... and the cull "
                f"receipt recorded {str(out_sha)[:16]}... for the {role} it "
                f"wrote; the file moved after the cull, or is another cull's"
            )
        if cut_from != parent_sha:
            raise _refuse(
                f"the cull receipt says its {role} was cut from bytes "
                f"{str(cut_from)[:16]}... and the parent row {parent.name!r} "
                f"pins {parent_sha[:16]}...; this cull was not cut from that "
                f"parent"
            )


def register_mesh(
    *,
    grid: Path | str,
    static: Path | str,
    name: str,
    rows: Path | str | None = None,
    dt_seconds: float | None = None,
    parent_row: str | None = None,
    cull_receipt: Path | str | None = None,
    spec: Path | str | None = None,
    generator_receipt: Path | str | None = None,
    static_receipt: Path | str | None = None,
    vertical_spec: Path | str | None = None,
    vertical: Path | str | None = None,
    lbc_source: str | None = None,
    cull_pad_scale: float | None = None,
    experimental: bool = False,
    log=print,
) -> tuple[Any, Path]:
    """Admit and register one pair; return ``(row, rows_path)``.

    ``experimental=True`` labels the row ``experimental-unanchored`` and skips
    ONLY the anchor-table lookup -- the Courant check stays -- and needs an
    explicit ``dt_seconds`` (a cull may inherit an experimental parent's).
    """

    from . import mesh_rows

    if not name or any(ch.isspace() for ch in str(name)):
        raise _refuse(f"--name {name!r} must be a non-empty name with no whitespace")
    if (parent_row is None) != (cull_receipt is None):
        raise _refuse(
            "--parent-row and --cull-receipt go together: a cull row names "
            "the generated-global row it was cut from AND the receipt of the "
            "cut; a global row names neither"
        )
    grid = _require_file(grid, "--grid")
    static = _require_file(static, "--static")
    rows_path = resolve_rows_path(rows)
    existing = mesh_rows.read_rows(rows_path) if rows_path.is_file() else []
    if any(item.name == name for item in existing):
        raise _refuse(
            f"{rows_path} already holds a row named {name!r}; a pair is "
            f"registered once"
        )
    _refuse_name_collisions(str(name), rows_path)
    vertical_spec_path = (
        None if vertical_spec is None else _require_file(vertical_spec, "--vertical-spec")
    )
    vertical_path = None if vertical is None else _require_file(vertical, "--vertical")
    declared_levels = vertical_levels(vertical_spec_path, vertical_path)
    nominal = static_nominal_dx_m(static)

    if parent_row is None:
        admission = _admit(grid, static, fine_dx_m=nominal, dt_seconds=dt_seconds,
                           experimental=experimental, log=log)
        if admission["regional"]:
            raise _refuse(
                f"{grid} carries a boundary zone, so it is a limited-area "
                f"cull, not a generated global mesh; register it with "
                f"--parent-row and --cull-receipt"
            )
        stamped, stamped_sha = grid_spec(grid)
        spec_path = None if spec is None else _require_file(spec, "--spec")
        region: dict[str, Any] | None = None
        spec_sha: str | None = None
        if spec_path is not None:
            try:
                declared = json.loads(spec_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise _refuse(f"--spec {spec_path} is not readable JSON: {error}") from error
            if not isinstance(declared, Mapping):
                raise _refuse(f"--spec {spec_path} is not a JSON object")
            if stamped is not None:
                # rw_mpas_mesh stamps the spec it parsed, so compare what the
                # spec SAYS, not how it was spelled.
                if _canonical_spec(stamped) != _canonical_spec(declared):
                    raise _refuse(
                        f"--spec {spec_path} is not the spec {grid.name} was "
                        f"generated from (its rw_mesh_spec_json digests to "
                        f"{stamped_sha[:16]}...); a row naming the wrong spec "
                        f"would carry false lineage"
                    )
            region = _region_summary(declared, str(spec_path))
        elif stamped is not None:
            spec_sha = stamped_sha
            region = _region_summary(stamped, f"{grid.name} attribute rw_mesh_spec_json")
        generator = (
            None if generator_receipt is None
            else _require_file(generator_receipt, "--generator-receipt")
        )
        if generator is None and stamped is not None and stamped.get("_has_generator_receipt"):
            generator = Path(f"{grid}#rw_mesh_receipt_json")
        static_rcpt = (
            None if static_receipt is None
            else _require_file(static_receipt, "--static-receipt")
        )
        background = None
        if region is not None and region.get("background_km") is not None:
            background = float(region["background_km"])
        row = mesh_rows.describe_generated(
            name=name, grid=grid, static=static,
            dt_seconds=admission["timestep_choice"]["dt_seconds"],
            n_levels=declared_levels if declared_levels is not None else default_levels(),
            admission=admission,
            spec_path=spec_path, spec_sha256=spec_sha,
            generator_receipt=generator, static_receipt=static_rcpt,
            background_km=background, nominal_dx_m=nominal, region=region,
            timestep_evidence=admission["timestep_choice"]["timestep_evidence"],
        )
    else:
        parents = [item for item in existing if item.name == parent_row]
        if not parents:
            raise _refuse(
                f"--parent-row {parent_row!r} is not in {rows_path}"
                + (f" (it holds {[item.name for item in existing]})" if existing else
                   " (the file does not exist yet)")
                + ".  A cull row's parent must be a generated-global row in the "
                "same file; register the parent first"
            )
        parent = parents[0]
        if parent.kind != "generated-global":
            raise _refuse(
                f"--parent-row {parent_row!r} is a {parent.kind} row; a cull "
                f"is cut from a generated-global row"
            )
        try:
            mesh_rows._verify_bytes(parent)
        except mesh_rows.MeshRowRefusal as error:
            raise _refuse(str(error)) from error
        if declared_levels is not None and declared_levels != parent.n_levels:
            raise _refuse(
                f"the cull declares {declared_levels} levels and its parent "
                f"{parent.name!r} pins {parent.n_levels}; a cull moves no level"
            )
        receipt_path = _require_file(cull_receipt, "--cull-receipt")
        receipt = _read_cull_receipt(receipt_path)
        _check_cull_digests(receipt, receipt_path, grid=grid, static=static, parent=parent)
        if nominal != float(parent.nominal_dx_m):
            raise _refuse(
                f"the cull's static declares nominalMinDc={nominal!r} and the "
                f"parent row pins {parent.nominal_dx_m!r}; the bind compares "
                f"them FP32-exactly"
            )
        inherit_experimental = parent.experimental and dt_seconds is None
        if inherit_experimental and not experimental:
            raise _refuse(
                f"the parent row {parent.name!r} declares an "
                f"{mesh_rows.EXPERIMENTAL_TIMESTEP_EVIDENCE} dt of "
                f"{parent.dt_seconds:g} s; its cull inherits that timestep only "
                f"on the same explicit opt-in"
            )
        # Inheriting an ANCHORED parent's dt is anchored whatever the caller
        # asked: labelling it experimental would claim no anchor stands
        # behind a timestep one does.
        effective_experimental = experimental and (
            dt_seconds is not None or parent.experimental
        )
        admission = _admit(
            grid, static, fine_dx_m=parent.nominal_dx_m,
            dt_seconds=parent.dt_seconds if dt_seconds is None else dt_seconds,
            experimental=effective_experimental, log=log,
        )
        if not admission["regional"]:
            raise _refuse(
                f"{grid} carries no boundary zone; it is not a limited-area "
                f"cull.  Register it as a global row (no --parent-row)"
            )
        region_doc = _cull_region(receipt, receipt_path)
        pad = cull_pad_scale
        centre = region_doc.get("center_deg")
        if (
            pad is None and region_doc.get("kind") == "cap"
            and parent.core_radius_km and region_doc.get("radius_km")
            and parent.point_deg is not None
            and isinstance(centre, (list, tuple)) and len(centre) == 2
            and abs(float(centre[0]) - parent.point_deg[0]) < 1.0e-6
            and abs(((float(centre[1]) - parent.point_deg[1] + 180.0) % 360.0) - 180.0) < 1.0e-6
        ):
            # Only a cap on the parent's own core is a dilation of it.
            pad = float(region_doc["radius_km"]) / float(parent.core_radius_km)
        lbc = lbc_source or (
            f"{grid.parent / (name + '.lbc')} (rw_mpas_lbc on the "
            f"wps-intermediate route, or --source unstructured-port-stream from "
            f"the parent's history; the forecast door refuses an absent or "
            f"empty --lbc-dir)"
        )
        try:
            row = mesh_rows.describe_cull(
                name=name, parent=parent, grid=grid, static=static,
                cull_receipt=receipt_path, cull_region=region_doc,
                cull_pad_scale=pad, lbc_source=lbc, admission=admission,
                dt_seconds=admission["timestep_choice"]["dt_seconds"],
                timestep_evidence=admission["timestep_choice"]["timestep_evidence"],
            )
        except mesh_rows.MeshRowRefusal as error:
            raise _refuse(str(error)) from error
    mesh_rows.append_row(rows_path, row)
    if row.experimental:
        log(
            f"ADVISORY {row.name}: dt {row.dt_seconds:g} s is "
            f"{mesh_rows.EXPERIMENTAL_TIMESTEP_EVIDENCE}.  The forecast bind "
            f"still requires an anchor unless the run opts into the "
            f"experimental timestep lane; this row records the opt-in, it "
            f"does not grant it"
        )
    log(
        f"ROW {row.name} ({row.kind}, {row.n_cells:,} cells, dt "
        f"{row.dt_seconds:g} s {row.timestep_evidence}, {row.n_levels} levels) "
        f"-> {rows_path}"
    )
    return row, rows_path


def run_register(arguments: argparse.Namespace) -> int:
    def log(message: str) -> None:
        print(message, file=sys.stderr, flush=True)

    row, rows_path = register_mesh(
        grid=arguments.grid, static=arguments.static, name=arguments.name,
        rows=arguments.rows, dt_seconds=arguments.dt_seconds,
        parent_row=arguments.parent_row, cull_receipt=arguments.cull_receipt,
        spec=arguments.spec, generator_receipt=arguments.generator_receipt,
        static_receipt=arguments.static_receipt,
        vertical_spec=arguments.vertical_spec, vertical=arguments.vertical,
        lbc_source=arguments.lbc_source, cull_pad_scale=arguments.cull_pad_scale,
        log=log,
    )
    from .mesh_rows import MESH_ROWS_ENVIRONMENT

    print(json.dumps(
        {"rows_file": str(rows_path), "row": row.as_dict(), "notes": row.notes()},
        indent=2, sort_keys=True,
    ))
    print(
        f"NEXT {MESH_ROWS_ENVIRONMENT}={rows_path} woof hex forecast --mesh "
        f"{row.name} --grid {row.grid} --static {row.static} --init <INIT.nc> "
        + ("--lbc-dir <LBC-DIR> " if row.kind == "generated-cull" else "")
        + "--hours H --out <RUN> --preflight",
        file=sys.stderr,
    )
    return 0


def add_register_parser(commands: Any) -> None:
    parser = commands.add_parser(
        "register",
        help="register a generated global mesh, or a cull of one, as a runtime row",
        description=(
            "Admit a grid/static pair (dual edges, cell coordination, Courant "
            "and the timestep anchor) and append it to a runtime mesh-row file "
            "so `woof hex forecast --mesh NAME` can resolve it. A global row "
            "names its spec by digest; a cull row (--parent-row with "
            "--cull-receipt) names a generated-global row in the same file and "
            "the `woof hex cull` receipt whose digests its files must match."
        ),
    )
    parser.add_argument("--grid", type=Path, required=True, metavar="FILE",
                        help="the grid to register")
    parser.add_argument("--static", type=Path, required=True, metavar="FILE",
                        help="its static (its nominalMinDc becomes the row's nominal dx)")
    parser.add_argument("--name", required=True, metavar="NAME",
                        help="the row name `woof hex forecast --mesh` resolves")
    parser.add_argument("--dt-seconds", type=float, default=None, metavar="DT",
                        help="the row's timestep; must pass Courant and hold an anchor "
                             "(default: the largest anchored dt the mesh admits; a cull "
                             "defaults to its parent's)")
    parser.add_argument("--rows", type=Path, default=None, metavar="FILE",
                        help="row file to append to (default: the one file "
                             "$WOOF_HEX_MESH_ROWS names)")
    parser.add_argument("--parent-row", default=None, metavar="NAME",
                        help="register a cull of this generated-global row (same file)")
    parser.add_argument("--cull-receipt", type=Path, default=None, metavar="FILE",
                        help="the <name>.cull.json `woof hex cull` wrote; required "
                             "with --parent-row")
    parser.add_argument("--spec", type=Path, default=None, metavar="JSON",
                        help="the resolution spec the grid was generated from "
                             "(default: the spec rw_mpas_mesh stamped on the grid)")
    parser.add_argument("--generator-receipt", type=Path, default=None, metavar="FILE",
                        help="rw_mpas_mesh receipt (default: the receipt stamped on the grid)")
    parser.add_argument("--static-receipt", type=Path, default=None, metavar="FILE",
                        help="rw_mpas_static receipt")
    parser.add_argument("--vertical-spec", type=Path, default=None, metavar="JSON",
                        help="vertical spec whose n_vert_levels the row pins "
                             "(default: the vertical spec's own default; a cull: its parent's)")
    parser.add_argument("--vertical", type=Path, default=None, metavar="FILE",
                        help="vertical artifact whose nVertLevels the row pins")
    parser.add_argument("--lbc-source", default=None, metavar="TEXT",
                        help="cull rows: what forces the boundary zone "
                             "(default: <grid-dir>/<name>.lbc from rw_mpas_lbc)")
    parser.add_argument("--cull-pad-scale", type=float, default=None, metavar="X",
                        help="cull rows: the cut as a multiple of the parent's core "
                             "radius (default: derived for a cap cut of a point row)")
    parser.set_defaults(handler=run_register)


__all__ = [
    "RegisterRefusal",
    "add_register_parser",
    "register_mesh",
    "resolve_rows_path",
    "run_register",
    "static_nominal_dx_m",
]
