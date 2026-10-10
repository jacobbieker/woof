"""Runtime mesh rows: written from receipts, verified at bind, refused by name."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from woof.hex import mesh_rows  # noqa: E402
from woof.hex.mesh_rows import MeshRowRefusal  # noqa: E402
from _layout import PACKAGE_DIR

netCDF4 = pytest.importorskip("netCDF4")


def _grid(path: Path, cells: int, edges: int, *, zone: int = 0) -> Path:
    with netCDF4.Dataset(str(path), "w", format="NETCDF3_CLASSIC") as ds:
        ds.createDimension("nCells", cells)
        ds.createDimension("nEdges", edges)
        ds.createDimension("nVertices", cells * 2)
        ds.createVariable("latCell", "f8", ("nCells",))[:] = np.linspace(0.6, 0.7, cells)
        ds.createVariable("lonCell", "f8", ("nCells",))[:] = np.linspace(4.6, 4.7, cells)
        mask = np.zeros(cells, dtype=np.int32)
        if zone:
            mask[: min(zone, cells)] = np.arange(1, min(zone, cells) + 1)
        ds.createVariable("bdyMaskCell", "i4", ("nCells",))[:] = mask
        ds.createVariable("bdyMaskEdge", "i4", ("nEdges",))[:] = np.zeros(edges, dtype=np.int32)
        ds.createVariable("bdyMaskVertex", "i4", ("nVertices",))[:] = np.zeros(cells * 2, dtype=np.int32)
    return path


class Binding:
    """A stand-in for ``woof.hex.drivers.mpas_mesh_binding.MeshBinding``: records its kwargs."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.name = kwargs["name"]


@pytest.fixture
def pair(tmp_path: Path) -> dict[str, Path]:
    grid = _grid(tmp_path / "p.grid.nc", 12, 30)
    static = _grid(tmp_path / "p.static.nc", 12, 30)
    spec = tmp_path / "p.spec.json"
    spec.write_text('{"background_km": 120}', encoding="utf-8")
    for name in ("p.grid.receipt.json", "p.static.receipt.json"):
        (tmp_path / name).write_text("{}", encoding="utf-8")
    return {"grid": grid, "static": static, "spec": spec, "dir": tmp_path}


def _parent(pair: dict[str, Path]) -> mesh_rows.MeshRow:
    return mesh_rows.describe_generated(
        name="p0.9375.120.12.n39.10w94.58", grid=pair["grid"], static=pair["static"],
        spec_path=pair["spec"], generator_receipt=pair["dir"] / "p.grid.receipt.json",
        static_receipt=pair["dir"] / "p.static.receipt.json", point_deg=(39.1, -94.58),
        fine_dx_m=937.5, core_radius_km=100.0, background_km=120.0, dt_seconds=5.0,
        n_levels=55, admission={"dual_edge_admission": {"ok": True}},
    )


def test_a_generated_row_is_measured_off_the_files_and_round_trips(pair) -> None:
    row = _parent(pair)
    assert row.kind == "generated-global" and row.n_cells == 12 and row.n_edges == 30
    assert row.n_interfaces == 56 and row.dt_seconds == 5.0
    assert row.grid_sha256 == mesh_rows.sha256_file(pair["grid"])
    path = mesh_rows.write_rows(pair["dir"] / mesh_rows.MESH_ROWS_FILENAME, [row])
    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["schema"] == mesh_rows.ROWS_SCHEMA
    back = mesh_rows.read_rows(path)
    assert back == [row]
    assert mesh_rows.GENERATED_ROW_MARKER in row.notes()


def test_apply_rows_adds_the_row_with_the_bytes_verified_and_leaves_the_registry_alone_otherwise(pair, monkeypatch) -> None:
    monkeypatch.delenv(mesh_rows.MESH_ROWS_ENVIRONMENT, raising=False)
    registry = {"x4.163842": object()}
    assert mesh_rows.apply_rows(registry, Binding) is registry
    row = _parent(pair)
    path = mesh_rows.write_rows(pair["dir"] / "mesh-rows.json", [row])
    monkeypatch.setenv(mesh_rows.MESH_ROWS_ENVIRONMENT, str(path))
    patched = mesh_rows.apply_rows(registry, Binding)
    assert set(patched) == {"x4.163842", row.name}
    kwargs = patched[row.name].kwargs
    assert kwargs["n_cells"] == 12 and kwargs["dt_seconds"] == 5.0
    assert kwargs["drop_carried_deformation"] is True
    assert "boundary_zone_width" not in kwargs


def test_a_row_whose_bytes_moved_is_refused_by_name(pair, monkeypatch) -> None:
    row = _parent(pair)
    path = mesh_rows.write_rows(pair["dir"] / "mesh-rows.json", [row])
    with open(pair["grid"], "ab") as handle:
        handle.write(b"\0")
    with pytest.raises(MeshRowRefusal) as refusal:
        mesh_rows.apply_rows({}, Binding, path)
    assert "bytes" in str(refusal.value) and "moved under it" in str(refusal.value)


def test_a_row_may_never_shadow_a_shipped_row(pair) -> None:
    row = _parent(pair)
    path = mesh_rows.write_rows(pair["dir"] / "mesh-rows.json", [row])
    with pytest.raises(MeshRowRefusal) as refusal:
        mesh_rows.apply_rows({row.name: object()}, Binding, path)
    assert "collides with a shipped registry row" in str(refusal.value)


def test_a_cull_row_needs_its_parent_a_zone_and_a_boundary_source(pair, monkeypatch) -> None:
    parent = _parent(pair)
    cull_grid = _grid(pair["dir"] / "c.grid.nc", 8, 20, zone=7)
    cull_static = _grid(pair["dir"] / "c.static.nc", 8, 20, zone=7)
    receipt = pair["dir"] / "c.grid.nc.cull-receipt.json"
    receipt.write_text("{}", encoding="utf-8")
    with pytest.raises(MeshRowRefusal) as refusal:
        mesh_rows.describe_cull(
            name="q1", parent=parent, grid=cull_grid, static=cull_static, cull_receipt=receipt,
            cull_region={"kind": "cap"}, cull_pad_scale=1.35, lbc_source="", admission={},
        )
    assert "no lbc_source" in str(refusal.value)
    cull = mesh_rows.describe_cull(
        name="q1", parent=parent, grid=cull_grid, static=cull_static, cull_receipt=receipt,
        cull_region={"kind": "cap", "radius_km": 135.0}, cull_pad_scale=1.35,
        lbc_source="lbc-dir", admission={},
    )
    assert cull.kind == "generated-cull" and cull.boundary_zone_width == 7
    assert cull.parent_row == parent.name and len(cull.bdy_mask_sha256) == 64
    assert cull.n_levels == parent.n_levels and cull.dt_seconds == parent.dt_seconds
    # A cull whose parent is in neither the file nor the registry is refused.
    path = mesh_rows.write_rows(pair["dir"] / "cull-only.json", [cull])
    with pytest.raises(MeshRowRefusal) as refusal:
        mesh_rows.apply_rows({}, Binding, path)
    assert "Lineage stops" in str(refusal.value)
    # With the parent beside it, both bind and the cull carries its regional slots.
    path = mesh_rows.write_rows(pair["dir"] / "mesh-rows.json", [parent, cull])
    patched = mesh_rows.apply_rows({}, Binding, path)
    kwargs = patched["q1"].kwargs
    assert kwargs["boundary_zone_width"] == 7 and kwargs["lbc_source"] == "lbc-dir"
    assert kwargs["bdy_mask_sha256"] == cull.bdy_mask_sha256
    assert mesh_rows.CULL_ROW_MARKER in kwargs["notes"]


def test_a_global_grid_cannot_be_described_as_a_cull(pair) -> None:
    parent = _parent(pair)
    receipt = pair["dir"] / "r.json"
    receipt.write_text("{}", encoding="utf-8")
    with pytest.raises(MeshRowRefusal) as refusal:
        mesh_rows.describe_cull(
            name="q1", parent=parent, grid=pair["grid"], static=pair["static"], cull_receipt=receipt,
            cull_region={}, cull_pad_scale=1.0, lbc_source="x", admission={},
        )
    assert "all-zero boundary mask" in str(refusal.value)


def test_append_refuses_a_second_row_of_the_same_name(pair) -> None:
    row = _parent(pair)
    path = mesh_rows.append_row(pair["dir"] / "mesh-rows.json", row)
    with pytest.raises(MeshRowRefusal):
        mesh_rows.append_row(path, row)


def test_a_document_of_another_schema_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "x.json"
    path.write_text('{"schema": "gpuwm-hex.cascade-rows/v1", "rows": []}', encoding="utf-8")
    with pytest.raises(MeshRowRefusal) as refusal:
        mesh_rows.read_rows(path)
    assert mesh_rows.ROWS_SCHEMA in str(refusal.value)
    with pytest.raises(MeshRowRefusal):
        mesh_rows.read_rows(tmp_path / "missing.json")


def _generic(pair: dict[str, Path], **extra) -> mesh_rows.MeshRow:
    return mesh_rows.describe_generated(
        name="g.raster.12", grid=pair["grid"], static=pair["static"],
        dt_seconds=20.0, n_levels=7, admission={"dual_edge_admission": {"ok": True}},
        spec_sha256="ab" * 32, nominal_dx_m=2500.0,
        region={"source": "grid attribute", "name": "raster density",
                "region_kinds": ["raster"], "background_km": 60.0,
                "spec": {"background_km": 60.0,
                         "regions": [{"shape": "raster", "path": "d.nc"}]}},
        background_km=60.0, **extra,
    )


def test_a_row_with_no_point_cap_round_trips_and_says_what_it_is(pair) -> None:
    row = _generic(pair, timestep_evidence=mesh_rows.ANCHORED_TIMESTEP_EVIDENCE)
    assert row.point_deg is None and row.core_radius_km is None and row.fine_dx_m is None
    assert row.nominal_dx_m == 2500.0 and row.n_interfaces == 8
    assert row.generator_receipt is None and row.static_receipt is None
    path = mesh_rows.write_rows(pair["dir"] / "generic.json", [row])
    back = mesh_rows.read_rows(path)
    assert back == [row]
    notes = row.notes()
    assert notes.startswith(mesh_rows.GENERATED_GLOBAL_ROW_MARKER)
    assert "raster" in notes and "abababababababab" in notes
    assert "EXPERIMENTAL" not in notes
    # A cull of it carries the region and needs no pad scale.
    cull_grid = _grid(pair["dir"] / "gc.grid.nc", 8, 20, zone=7)
    cull_static = _grid(pair["dir"] / "gc.static.nc", 8, 20, zone=7)
    receipt = pair["dir"] / "gc.cull.json"
    receipt.write_text("{}", encoding="utf-8")
    cull = mesh_rows.describe_cull(
        name="gc", parent=row, grid=cull_grid, static=cull_static, cull_receipt=receipt,
        cull_region={"kind": "polygon"}, cull_pad_scale=None, lbc_source="lbc",
        admission={},
    )
    assert cull.cull_pad_scale is None and cull.region == row.region
    assert cull.timestep_evidence == mesh_rows.ANCHORED_TIMESTEP_EVIDENCE
    assert "cut with rw_mpas_mesh" in cull.notes()
    path = mesh_rows.write_rows(pair["dir"] / "generic.json", [row, cull])
    assert mesh_rows.read_rows(path) == [row, cull]


def test_an_experimental_row_says_so_in_its_notes(pair) -> None:
    row = _generic(pair, timestep_evidence=mesh_rows.EXPERIMENTAL_TIMESTEP_EVIDENCE)
    assert row.experimental
    assert "EXPERIMENTAL-UNANCHORED" in row.notes()


def test_a_row_needs_a_nominal_spacing_and_a_known_evidence_label(pair) -> None:
    with pytest.raises(MeshRowRefusal) as refusal:
        mesh_rows.describe_generated(
            name="g", grid=pair["grid"], static=pair["static"], dt_seconds=20.0,
            n_levels=7, admission={},
        )
    assert "nominal dx" in str(refusal.value)
    with pytest.raises(MeshRowRefusal):
        _generic(pair, timestep_evidence="trust-me")
    raw = _generic(pair).as_dict()
    raw["timestep_evidence"] = "trust-me"
    with pytest.raises(MeshRowRefusal):
        mesh_rows.MeshRow.from_dict(raw)


def test_a_row_written_before_the_cap_fields_became_optional_still_reads(pair) -> None:
    raw = _parent(pair).as_dict()
    for key in ("region", "timestep_evidence"):
        raw.pop(key)
    back = mesh_rows.MeshRow.from_dict(raw)
    assert back.point_deg == (39.1, -94.58) and back.timestep_evidence is None
    assert back.notes().startswith(mesh_rows.GENERATED_ROW_MARKER)


def test_the_registry_module_applies_runtime_rows_where_it_builds_the_table() -> None:
    text = (PACKAGE_DIR / "drivers" / "mpas_mesh_binding.py").read_text(encoding="utf-8")
    assert "mesh_rows.apply_rows(MESH_BINDINGS, MeshBinding)" in text
    # Before the cascade rows, so a cascade may cull a generated parent.
    assert text.index("mesh_rows.apply_rows") < text.index("cascade_row.apply_rows")
