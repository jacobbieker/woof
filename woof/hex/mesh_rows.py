"""Runtime mesh rows: a generated mesh registers itself from its own bytes.

THE BLOCKER THIS REMOVES.  ``woof hex forecast`` resolves ``--mesh``
against ``woof.hex.drivers.mpas_mesh_binding.MESH_BINDINGS`` and refuses a name it
does not hold.  Every row in that file was typed by a person after the mesh
existed.  ``woof hex mesh-plan --point`` generates a mesh at a point nobody
typed a row for -- the whole point of the door is that the point is
arbitrary -- so the registry could never hold its row in advance, and a
checkout edit per point would be the hand-written-row-per-swath defect
``woof.hex.cascade_row`` retired for the cascade, wearing a different hat.

WHAT THIS MODULE IS, AND HOW IT DIFFERS FROM ``cascade_row``.  A cascade row
describes a CULL of a row somebody registered, and it insists on that:
lineage stops at a person.  A runtime row describes a mesh the GENERATOR
made from a spec the door wrote, so its lineage stops at the spec and at the
generator's own receipt:

* a ``generated-global`` row names the spec by SHA-256, the generator
  receipt, the static receipt, and the pair's bytes;
* a ``generated-cull`` row names the ``generated-global`` row it was cut
  from (which must be in the same file, or already registered), the cull
  receipt, the region, the boundary-mask digest and the boundary source.

Everything a registry row BUYS is kept, and it is worth restating which
half is a review and which is a measurement (``cascade_row`` says the same):
the bytes are pinned by count and SHA-256 and re-hashed at bind; the
Courant, dual-edge, cell-coordination and regional admissions are
measurements of the files at bind; the timestep is refused unless an anchor
covers the configuration; the regional forecast opener re-measures the class
key and requires a contract deck on THESE rings.  None of those become
weaker because the row was written by the door that generated the mesh.
What is genuinely lost is a person reading the row; what stands in for that
is the door's own admission pass on the pair it just built (dual edge, cell
coordination, Courant at the declared timestep), recorded in the row file
and refused at write time when it fails.

WHAT THIS MODULE REFUSES, by name:

* a name that collides with a shipped registry row (a runtime row may never
  shadow one);
* a cull row whose parent is neither in the same file nor registered;
* a row whose declared bytes do not match the files on disk now;
* a cull row carrying a boundary zone and no ``lbc_source``;
* a file that is not a ``gpuwm-hex.mesh-rows/v1`` document.

Rows are supplied through a FILE, like cascade rows, because the forecast
door runs the driver in another process.  ``woof.hex.drivers.mpas_mesh_binding``
applies them where it builds ``MESH_BINDINGS`` -- before the cascade rows,
so a cascade may cull a generated parent -- and the door's by-path
re-execution of that module picks them up as the first copy did.  The file
is named by ``$WOOF_HEX_MESH_ROWS`` (``os.pathsep``-separated when there
are several) and, by convention, lives beside the mesh it describes as
``mesh-rows.json``.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import json
import os
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .errors import MpasPortError

#: The document schema this module writes and reads.
ROWS_SCHEMA = "gpuwm-hex.mesh-rows/v1"

#: Where the forecast door is told to look for runtime rows.
MESH_ROWS_ENVIRONMENT = "WOOF_HEX_MESH_ROWS"

#: The file name a generating door leaves beside the mesh.
MESH_ROWS_FILENAME = "mesh-rows.json"

#: Stamped into a row's ``notes`` so the bind log and the run receipt say
#: what the row is, in the row's own words.
GENERATED_ROW_MARKER = "GENERATED-AT-POINT"
CULL_ROW_MARKER = "GENERATED-CULL"
#: A generated global row with no point cap (`woof hex register`).
GENERATED_GLOBAL_ROW_MARKER = "GENERATED-GLOBAL"

ROW_KINDS = ("generated-global", "generated-cull")

#: ``timestep_evidence`` values.  The second is the experimental sub-anchor
#: lane's label: a row carrying it passed the Courant check and skipped
#: ONLY the anchor-table lookup, on an explicit opt-in.
ANCHORED_TIMESTEP_EVIDENCE = "anchored"
EXPERIMENTAL_TIMESTEP_EVIDENCE = "experimental-unanchored"
TIMESTEP_EVIDENCE = (ANCHORED_TIMESTEP_EVIDENCE, EXPERIMENTAL_TIMESTEP_EVIDENCE)


class MeshRowRefusal(MpasPortError):
    """A runtime mesh row is refused, and the message says why."""


@dataclass(frozen=True, slots=True)
class MeshRow:
    """One generated mesh, described by the receipts that made it."""

    kind: str
    name: str
    n_cells: int
    n_edges: int
    n_levels: int
    n_interfaces: int
    n_soil_levels: int
    nominal_dx_m: float
    dt_seconds: float
    grid: str
    grid_bytes: int
    grid_sha256: str
    static: str
    static_bytes: int
    static_sha256: str
    #: The resolution spec the generator was handed, by digest, and the
    #: receipts it and the static builder wrote.  Lineage stops here.  A
    #: pair registered with ``woof hex register`` and no receipt on disk
    #: carries ``None`` for the receipt it does not have, never a guess.
    spec_sha256: str | None
    generator_receipt: str | None
    static_receipt: str | None
    #: The door's own admission pass over the pair, recorded so the row
    #: says what was measured before it was written.
    admission: Mapping[str, Any]
    #: The point, fine spacing and core radius the spec was built from --
    #: a ``mesh-plan --point`` cap only.  A mesh generated from any other
    #: spec (raster density, polygon, a uniform background) carries
    #: ``None`` here and names its spec in ``region`` instead.
    point_deg: tuple[float, float] | None = None
    fine_dx_m: float | None = None
    core_radius_km: float | None = None
    background_km: float | None = None
    #: The resolution spec itself (or the part of it a reader needs), for
    #: rows registered by ``woof hex register``; ``spec_sha256`` digests it.
    region: Mapping[str, Any] | None = None
    #: What stands behind ``dt_seconds``: ``"anchored"`` (an
    #: ``ADMITTED_TIMESTEPS`` row admits this configuration) or
    #: ``"experimental-unanchored"`` (the Courant check passed and the
    #: anchor lookup was skipped on an explicit opt-in).  ``None`` on rows
    #: written before the field existed, which were all anchored.
    timestep_evidence: str | None = None
    # -- cull rows only ---------------------------------------------------
    parent_row: str | None = None
    boundary_zone_width: int | None = None
    bdy_mask_sha256: str | None = None
    lbc_source: str | None = None
    cull_receipt: str | None = None
    cull_region: Mapping[str, Any] | None = None
    cull_pad_scale: float | None = None

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, Mapping):
                value = json.loads(json.dumps(dict(value)))
            elif isinstance(value, tuple):
                value = list(value)
            out[item.name] = value
        return out

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "MeshRow":
        names = {item.name for item in fields(cls)}
        missing = sorted(name for name in names if name not in raw
                         and name in _REQUIRED_KEYS)
        if missing:
            raise MeshRowRefusal(
                f"a mesh row is missing {missing}; a row that does not say "
                f"what it pins cannot be bound.  Rows are written by "
                f"`woof hex mesh-plan --point --generate` or `woof hex "
                f"register`, never by hand"
            )
        unknown = sorted(name for name in raw if name not in names)
        if unknown:
            raise MeshRowRefusal(
                f"a mesh row carries fields this build does not know "
                f"({unknown}); a row from a newer door is refused rather than "
                f"read with its unknown fields ignored"
            )
        values = dict(raw)
        if values.get("point_deg") is not None:
            values["point_deg"] = tuple(float(v) for v in values["point_deg"])
        values["admission"] = MappingProxyType(dict(values.get("admission") or {}))
        for key in ("cull_region", "region"):
            if values.get(key) is not None:
                values[key] = MappingProxyType(dict(values[key]))
        evidence = values.get("timestep_evidence")
        if evidence is not None and evidence not in TIMESTEP_EVIDENCE:
            raise MeshRowRefusal(
                f"a mesh row declares timestep_evidence {evidence!r}; this "
                f"build knows {list(TIMESTEP_EVIDENCE)}"
            )
        return cls(**{name: values.get(name) for name in names})

    @property
    def experimental(self) -> bool:
        """True when the row's timestep holds no anchor (explicit opt-in)."""

        return self.timestep_evidence == EXPERIMENTAL_TIMESTEP_EVIDENCE

    def _origin(self) -> str:
        """What the mesh was generated from, in words, cap or not."""

        if self.point_deg is not None:
            lat, lon = self.point_deg
            core = (
                f"a {self.fine_dx_m:g} m core"
                if self.fine_dx_m is not None else "a core"
            )
            radius = (
                f" of {self.core_radius_km:g} km radius"
                if self.core_radius_km is not None else ""
            )
            background = (
                f" on a {self.background_km:g} km background"
                if self.background_km is not None else ""
            )
            return f"{core}{radius} at {lat:.4f} N {lon:.4f} E{background}"
        region = dict(self.region or {})
        pieces = [f"a generated mesh of nominal dx {self.nominal_dx_m:g} m"]
        label = region.get("name")
        if label:
            pieces.append(f"named {label!s}")
        kinds = region.get("region_kinds")
        if kinds:
            pieces.append(f"refined over {', '.join(str(k) for k in kinds)} regions")
        if self.background_km is not None:
            pieces.append(f"on a {self.background_km:g} km background")
        return " ".join(pieces)

    def _spec_words(self) -> str:
        if self.spec_sha256:
            return f"the spec digest {self.spec_sha256[:16]}..."
        return "no recorded spec digest"

    def _timestep_words(self) -> str:
        if not self.experimental:
            return ""
        return (
            f"  TIMESTEP {self.dt_seconds:g} s IS "
            f"{EXPERIMENTAL_TIMESTEP_EVIDENCE.upper()}: the Courant check "
            f"passed and no ADMITTED_TIMESTEPS row stands behind it."
        )

    def notes(self) -> str:
        if self.kind == "generated-cull":
            pad = (
                f"cut at pad {float(self.cull_pad_scale):g}"
                if self.cull_pad_scale is not None else "cut"
            )
            return (
                f"{CULL_ROW_MARKER}: a limited-area cull of the runtime row "
                f"{self.parent_row!r} ({self._origin()}), {pad} with "
                f"rw_mpas_mesh --cull-parent.  Nobody hand-wrote this row: it "
                f"is written from the cull receipt at {self.cull_receipt}, the "
                f"parent's generator receipt at {self.generator_receipt}, and "
                f"{self._spec_words()}; every admission behind it is a "
                f"measurement of these files at bind, and the regional opener "
                f"requires a contract deck on these rings and re-measures the "
                f"class key.  Boundary series: {self.lbc_source}."
                + self._timestep_words()
            )
        static = (
            f"static by rw_mpas_static (receipt {self.static_receipt})"
            if self.static_receipt else "static registered without a receipt"
        )
        return (
            f"{GENERATED_ROW_MARKER if self.point_deg is not None else GENERATED_GLOBAL_ROW_MARKER}: "
            f"{self._origin()}, generated by "
            f"rw_mpas_mesh from {self._spec_words()} (receipt "
            f"{self.generator_receipt}); {static}.  Nobody hand-wrote this "
            f"row: the door that registered the pair measured dual edges, "
            f"cell coordination and Courant at dt {self.dt_seconds:g} s "
            f"before writing it, and the bind measures them again."
            + self._timestep_words()
        )


#: Keys a row must SPELL (a value may still be ``null`` where the field
#: allows it).  The cap fields left this set when ``woof hex register``
#: began registering meshes that have no cap; a row written before then
#: still reads, because every field it carries is still known.
_REQUIRED_KEYS = {
    "kind", "name", "n_cells", "n_edges", "n_levels", "n_interfaces",
    "n_soil_levels", "nominal_dx_m", "dt_seconds", "grid", "grid_bytes",
    "grid_sha256", "static", "static_bytes", "static_sha256", "spec_sha256",
    "generator_receipt", "static_receipt", "admission",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# the file
# ---------------------------------------------------------------------------
def write_rows(path: Path, rows: Sequence[MeshRow]) -> Path:
    """Write (replace) the runtime row document at ``path``."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = [row.name for row in rows]
    if len(set(names)) != len(names):
        raise MeshRowRefusal(
            f"two rows in one document share a name ({names}); a name must "
            f"resolve to exactly one pair of files"
        )
    path.write_text(
        json.dumps(
            {"schema": ROWS_SCHEMA, "rows": [row.as_dict() for row in rows]},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


def read_rows(path: Path) -> list[MeshRow]:
    """The rows one document holds, refused by name when it is not one."""

    location = Path(path)
    if not location.is_file():
        raise MeshRowRefusal(
            f"{location} is not a file.  A door told to read runtime mesh "
            f"rows from a file that is not there would fall back to the "
            f"shipped registry and run the wrong mesh"
        )
    try:
        document = json.loads(location.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MeshRowRefusal(f"{location} is not readable JSON: {error}") from error
    if not isinstance(document, dict) or document.get("schema") != ROWS_SCHEMA:
        raise MeshRowRefusal(
            f"{location} is not a {ROWS_SCHEMA} document (schema "
            f"{document.get('schema') if isinstance(document, dict) else None!r})"
        )
    rows = document.get("rows")
    if not isinstance(rows, list):
        raise MeshRowRefusal(f"{location} carries no 'rows' list")
    return [MeshRow.from_dict(raw) for raw in rows]


def append_row(path: Path, row: MeshRow) -> Path:
    """Add one row to a document, creating it when absent."""

    location = Path(path)
    existing = read_rows(location) if location.is_file() else []
    if any(item.name == row.name for item in existing):
        raise MeshRowRefusal(
            f"{location} already holds a row named {row.name!r}; a generated "
            f"pair is registered once, and re-registering it would let two "
            f"sets of bytes claim one name"
        )
    return write_rows(location, [*existing, row])


def row_files(path: Path | str | None = None) -> list[Path]:
    """The documents to read: an explicit path, or ``$WOOF_HEX_MESH_ROWS``."""

    if path is not None:
        return [Path(path)]
    raw = os.environ.get(MESH_ROWS_ENVIRONMENT, "")
    return [Path(piece.strip()) for piece in raw.split(os.pathsep) if piece.strip()]


def load_rows(path: Path | str | None = None) -> list[MeshRow]:
    rows: list[MeshRow] = []
    for location in row_files(path):
        rows.extend(read_rows(location))
    return rows


# ---------------------------------------------------------------------------
# describing a pair
# ---------------------------------------------------------------------------
def _dimensions(path: Path) -> dict[str, int]:
    from netCDF4 import Dataset

    with Dataset(str(path)) as dataset:
        return {key: int(value.size) for key, value in dataset.dimensions.items()}


def describe_generated(
    *,
    name: str,
    grid: Path,
    static: Path,
    dt_seconds: float,
    n_levels: int,
    admission: Mapping[str, Any],
    spec_path: Path | None = None,
    spec_sha256: str | None = None,
    generator_receipt: Path | None = None,
    static_receipt: Path | None = None,
    point_deg: tuple[float, float] | None = None,
    fine_dx_m: float | None = None,
    core_radius_km: float | None = None,
    background_km: float | None = None,
    nominal_dx_m: float | None = None,
    region: Mapping[str, Any] | None = None,
    timestep_evidence: str | None = None,
    n_soil_levels: int = 4,
) -> MeshRow:
    """Measure a generated pair and write the row that describes it.

    A ``mesh-plan --point`` pair passes its cap (``point_deg``,
    ``fine_dx_m``, ``core_radius_km``, ``background_km``) and its spec file;
    any other generated mesh passes ``nominal_dx_m`` (the static's own
    ``nominalMinDc``, which the bind compares FP32-exactly) and a ``region``
    naming the spec instead, with ``spec_sha256`` when the spec is known only
    by digest.  At least one of ``fine_dx_m`` and ``nominal_dx_m`` is
    required: a row with no nominal spacing cannot be bound.
    """

    grid = Path(grid)
    static = Path(static)
    if nominal_dx_m is None and fine_dx_m is None:
        raise MeshRowRefusal(
            f"the row {name!r} declares neither a fine spacing nor a nominal "
            f"dx; the bind compares the registry's nominal dx FP32-exactly "
            f"against the static's nominalMinDc and has nothing to compare"
        )
    if spec_path is not None and spec_sha256 is not None:
        measured_spec = sha256_file(Path(spec_path))
        if measured_spec != spec_sha256:
            raise MeshRowRefusal(
                f"the spec {spec_path} digests to {measured_spec[:16]}... and "
                f"the row was told {spec_sha256[:16]}...; one of the two names "
                f"the wrong spec"
            )
    if timestep_evidence is not None and timestep_evidence not in TIMESTEP_EVIDENCE:
        raise MeshRowRefusal(
            f"timestep_evidence {timestep_evidence!r} is not one of "
            f"{list(TIMESTEP_EVIDENCE)}"
        )
    dims = _dimensions(grid)
    for key in ("nCells", "nEdges"):
        if key not in dims:
            raise MeshRowRefusal(
                f"{grid} declares no {key}; it is not an MPAS grid and cannot "
                f"be registered"
            )
    static_dims = _dimensions(static)
    if static_dims.get("nCells") != dims["nCells"]:
        raise MeshRowRefusal(
            f"{grid} has nCells={dims['nCells']} and {static} has "
            f"nCells={static_dims.get('nCells')}; the pair was not generated "
            f"together and a row pinning both would bind a static to the "
            f"wrong mesh"
        )
    return MeshRow(
        kind="generated-global",
        name=name,
        n_cells=int(dims["nCells"]),
        n_edges=int(dims["nEdges"]),
        n_levels=int(n_levels),
        n_interfaces=int(n_levels) + 1,
        n_soil_levels=int(n_soil_levels),
        nominal_dx_m=float(nominal_dx_m if nominal_dx_m is not None else fine_dx_m),
        dt_seconds=float(dt_seconds),
        grid=str(grid),
        grid_bytes=grid.stat().st_size,
        grid_sha256=sha256_file(grid),
        static=str(static),
        static_bytes=static.stat().st_size,
        static_sha256=sha256_file(static),
        spec_sha256=(
            sha256_file(Path(spec_path)) if spec_path is not None else spec_sha256
        ),
        generator_receipt=None if generator_receipt is None else str(generator_receipt),
        static_receipt=None if static_receipt is None else str(static_receipt),
        point_deg=(
            None if point_deg is None
            else (float(point_deg[0]), float(point_deg[1]))
        ),
        fine_dx_m=None if fine_dx_m is None else float(fine_dx_m),
        core_radius_km=None if core_radius_km is None else float(core_radius_km),
        background_km=None if background_km is None else float(background_km),
        admission=MappingProxyType(dict(admission)),
        region=(
            None if region is None
            else MappingProxyType(json.loads(json.dumps(dict(region))))
        ),
        timestep_evidence=timestep_evidence,
    )


def describe_cull(
    *,
    name: str,
    parent: MeshRow,
    grid: Path,
    static: Path,
    cull_receipt: Path,
    cull_region: Mapping[str, Any],
    cull_pad_scale: float | None,
    lbc_source: str,
    admission: Mapping[str, Any],
    dt_seconds: float | None = None,
    timestep_evidence: str | None = None,
) -> MeshRow:
    """Measure a freshly-cut pair and write the cull row that describes it.

    ``cull_pad_scale`` is ``None`` when the cut was not made as a multiple
    of a cap's core radius (a polygon, a box, a cap that is not a point
    door's).  ``timestep_evidence`` defaults to the parent's.
    """

    from netCDF4 import Dataset
    import numpy as np

    from .mesh import REGIONAL_BOUNDARY_MASK_NAMES, regional_boundary_mask_digest

    grid = Path(grid)
    static = Path(static)
    with Dataset(str(grid)) as dataset:
        dataset.set_auto_maskandscale(False)
        dims = {key: int(value.size) for key, value in dataset.dimensions.items()}
        masks = {
            item: dataset.variables[item][:]
            for item in REGIONAL_BOUNDARY_MASK_NAMES
            if item in dataset.variables
        }
        if len(masks) != len(REGIONAL_BOUNDARY_MASK_NAMES):
            raise MeshRowRefusal(
                f"{grid} carries {sorted(masks)} of the boundary-mask triple "
                f"{list(REGIONAL_BOUNDARY_MASK_NAMES)}; a cull with an "
                f"incomplete triple cannot identify its own rings"
            )
        zone_width = int(np.max(np.asarray(masks["bdyMaskCell"])))
        digest = regional_boundary_mask_digest(masks)
    if zone_width <= 0:
        raise MeshRowRefusal(
            f"{grid} carries an all-zero boundary mask, so it is not a "
            f"limited-area cull; register it as a global row instead"
        )
    if not lbc_source:
        raise MeshRowRefusal(
            f"a cull row for {name!r} declares a {zone_width}-ring boundary "
            f"zone and no lbc_source, so nothing is registered to force it "
            f"and a run would integrate an unforced boundary inward"
        )
    return MeshRow(
        kind="generated-cull",
        name=name,
        n_cells=int(dims["nCells"]),
        n_edges=int(dims["nEdges"]),
        n_levels=parent.n_levels,
        n_interfaces=parent.n_interfaces,
        n_soil_levels=parent.n_soil_levels,
        nominal_dx_m=parent.nominal_dx_m,
        dt_seconds=float(parent.dt_seconds if dt_seconds is None else dt_seconds),
        grid=str(grid),
        grid_bytes=grid.stat().st_size,
        grid_sha256=sha256_file(grid),
        static=str(static),
        static_bytes=static.stat().st_size,
        static_sha256=sha256_file(static),
        spec_sha256=parent.spec_sha256,
        generator_receipt=parent.generator_receipt,
        static_receipt=parent.static_receipt,
        point_deg=parent.point_deg,
        fine_dx_m=parent.fine_dx_m,
        core_radius_km=parent.core_radius_km,
        background_km=parent.background_km,
        admission=MappingProxyType(dict(admission)),
        region=parent.region,
        timestep_evidence=(
            parent.timestep_evidence if timestep_evidence is None else timestep_evidence
        ),
        parent_row=parent.name,
        boundary_zone_width=zone_width,
        bdy_mask_sha256=digest,
        lbc_source=str(lbc_source),
        cull_receipt=str(cull_receipt),
        cull_region=MappingProxyType(json.loads(json.dumps(dict(cull_region)))),
        cull_pad_scale=None if cull_pad_scale is None else float(cull_pad_scale),
    )


# ---------------------------------------------------------------------------
# applying rows to the registry
# ---------------------------------------------------------------------------
def _verify_bytes(row: MeshRow) -> None:
    for role, file_path, want_bytes, want_sha in (
        ("grid", Path(row.grid), row.grid_bytes, row.grid_sha256),
        ("static", Path(row.static), row.static_bytes, row.static_sha256),
    ):
        if not file_path.is_file():
            raise MeshRowRefusal(
                f"the mesh row {row.name!r} names a {role} at {file_path} "
                f"that is not a file"
            )
        size = file_path.stat().st_size
        if size != want_bytes:
            raise MeshRowRefusal(
                f"the mesh row {row.name!r} declares a {role} of {want_bytes} "
                f"bytes and {file_path} is {size}.  The row was written from "
                f"the generator's receipt; a mismatch means the file moved "
                f"under it"
            )
        measured = sha256_file(file_path)
        if measured != want_sha:
            raise MeshRowRefusal(
                f"the mesh row {row.name!r} declares a {role} SHA-256 "
                f"{want_sha[:16]}... and {file_path} digests to "
                f"{measured[:16]}..."
            )


def apply_rows(
    registry: Mapping[str, Any], binding_type: Any, path: Path | str | None = None
) -> Mapping[str, Any]:
    """Return ``registry`` with the runtime rows added.

    Returns the mapping UNCHANGED when no runtime row file is named, which
    is every ordinary run.
    """

    rows = load_rows(path)
    if not rows:
        return registry
    patched = dict(registry)
    for row in rows:
        if row.kind not in ROW_KINDS:
            raise MeshRowRefusal(
                f"the mesh row {row.name!r} is of kind {row.kind!r}; this "
                f"build knows {list(ROW_KINDS)}"
            )
        if row.name in registry:
            raise MeshRowRefusal(
                f"the runtime row {row.name!r} collides with a shipped "
                f"registry row.  A runtime row may never shadow one: the "
                f"shipped row pins bytes a person reviewed and this one "
                f"would replace them silently"
            )
        if row.name in patched:
            raise MeshRowRefusal(
                f"two runtime row files both register {row.name!r}; a name "
                f"must resolve to exactly one pair of files"
            )
        _verify_bytes(row)
        regional: dict[str, Any] = {}
        if row.kind == "generated-cull":
            if not row.parent_row or row.parent_row not in patched:
                raise MeshRowRefusal(
                    f"the cull row {row.name!r} names parent "
                    f"{row.parent_row!r}, which is neither in its own row "
                    f"file nor a registered mesh.  Lineage stops at a "
                    f"generated-global row: a cull of a mesh nobody described "
                    f"is how a shape nobody looked at becomes a forecast"
                )
            if not row.lbc_source:
                raise MeshRowRefusal(
                    f"the cull row {row.name!r} declares a "
                    f"{row.boundary_zone_width}-ring boundary zone and no "
                    f"lbc_source, so nothing is registered to force it"
                )
            regional = {
                "boundary_zone_width": int(row.boundary_zone_width or 0),
                "bdy_mask_sha256": row.bdy_mask_sha256,
                "lbc_source": row.lbc_source,
            }
        patched[row.name] = binding_type(
            name=row.name,
            n_cells=row.n_cells,
            n_edges=row.n_edges,
            n_levels=row.n_levels,
            n_interfaces=row.n_interfaces,
            n_soil_levels=row.n_soil_levels,
            nominal_dx_m=row.nominal_dx_m,
            dt_seconds=row.dt_seconds,
            grid_bytes=row.grid_bytes,
            grid_sha256=row.grid_sha256,
            static_bytes=row.static_bytes,
            static_sha256=row.static_sha256,
            drop_carried_deformation=True,
            notes=row.notes(),
            **regional,
        )
    return MappingProxyType(patched)


__all__ = [
    "ANCHORED_TIMESTEP_EVIDENCE",
    "CULL_ROW_MARKER",
    "EXPERIMENTAL_TIMESTEP_EVIDENCE",
    "GENERATED_GLOBAL_ROW_MARKER",
    "GENERATED_ROW_MARKER",
    "MESH_ROWS_ENVIRONMENT",
    "MESH_ROWS_FILENAME",
    "ROWS_SCHEMA",
    "ROW_KINDS",
    "TIMESTEP_EVIDENCE",
    "MeshRow",
    "MeshRowRefusal",
    "append_row",
    "apply_rows",
    "describe_cull",
    "describe_generated",
    "load_rows",
    "read_rows",
    "row_files",
    "sha256_file",
    "write_rows",
]
