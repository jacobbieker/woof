"""``woof hex regional --terrain-source``: the static-highres hook in the mesh stage (offline, deterministic)."""
import json
from pathlib import Path

import pytest

from woof.hex import regional_door as door
from woof.hex import regional_terrain as rt
from woof.hex.cli import build_parser


def parse(tmp_path, *extra):
    return build_parser().parse_args([
        "regional", "--point", "36.5,-97.5", "--cycle", "2024-05-06T18", "--hours", "3",
        "--work-dir", str(tmp_path / "run"), *extra])


def test_no_option_adds_nothing(tmp_path):
    args = parse(tmp_path)
    assert (args.terrain_source, args.terrain_max_spacing_km, args.terrain_cache_root) == (None, None, None)
    assert rt.request_from_arguments(args, fine_dx_m=1000.0, has_parent=False) is None


def test_request_defaults_the_spacing_to_the_fine_core(tmp_path):
    args = parse(tmp_path, "--terrain-source", "glo30")
    assert rt.request_from_arguments(args, fine_dx_m=3000.0, has_parent=False) == {
        "source": "glo30", "max_spacing_km": 3.75, "cache_root": None}
    args = parse(tmp_path, "--terrain-source", "gem15-dtm", "--terrain-max-spacing-km", "4",
                 "--terrain-cache-root", str(tmp_path / "c"))
    assert rt.request_from_arguments(args, fine_dx_m=3000.0, has_parent=False) == {
        "source": "gem15-dtm", "max_spacing_km": 4.0, "cache_root": str(tmp_path / "c")}


@pytest.mark.parametrize("extra,parent,words", [
    (["--terrain-max-spacing-km", "2"], False, "qualify --terrain-source"),
    (["--terrain-cache-root", "x"], False, "qualify --terrain-source"),
    (["--terrain-source", "glo30", "--skip", "mesh"], False, "--skip mesh"),
    (["--terrain-source", "glo30", "--stage", "init"], False, "built without --terrain-source"),
    (["--terrain-source", "glo30", "--terrain-max-spacing-km", "0"], False, "positive spacing"),
    (["--terrain-source", "glo30"], True, "kept hex run"),
])
def test_unsupported_combinations_are_refused_by_name(tmp_path, extra, parent, words):
    with pytest.raises(rt.RegionalTerrainRefusal, match=words):
        rt.request_from_arguments(parse(tmp_path, *extra), fine_dx_m=1000.0, has_parent=parent)


def test_the_choices_are_the_static_highres_choices():
    from woof.hex import static_highres as sh

    assert set(door._terrain_choices()) == set(sh.TERRAIN_CHOICES)


def test_a_reused_mesh_must_carry_the_terrain_asked_for(tmp_path):
    request = {"source": "glo30", "max_spacing_km": 1.25, "cache_root": None}
    with pytest.raises(rt.RegionalTerrainRefusal, match="built without"):
        rt.require_applied(tmp_path, request)
    (tmp_path / rt.RECEIPT_NAME).write_text(json.dumps(
        {"requested": "glo30", "max_spacing_km": 1.25, "used": "copernicus-dem-glo30"}))
    assert rt.require_applied(tmp_path, request)["used"] == "copernicus-dem-glo30"
    with pytest.raises(rt.RegionalTerrainRefusal, match="not gem15-dtm"):
        rt.require_applied(tmp_path, {**request, "source": "gem15-dtm"})


def test_mesh_stage_adds_parent_static_only_with_terrain(tmp_path, monkeypatch):
    seen = []

    class Status:
        def note(self, text):
            pass

    monkeypatch.setattr(door, "_start_source_prefetch", lambda *a: None)
    monkeypatch.setattr(door, "_cli", lambda argv, status: seen.append(list(argv)) or 0)
    applied = []
    monkeypatch.setattr(rt, "apply_terrain", lambda **kw: applied.append(kw) or {"used": "x"})
    (tmp_path / "mesh").mkdir()
    base = {"paths": {"mesh": str(tmp_path / "mesh"), "work": str(tmp_path), "vertical_spec": "spec.json"},
            "point_deg": [36.5, -97.5], "radius_km": 50.0, "fine_dx_m": 3000.0, "background_km": 96.0,
            "card": None, "drawn_shape": None}
    args = parse(tmp_path)
    assert door._stage_mesh(args, dict(base), Status()) == {"returncode": 0}
    assert "--parent-static" not in seen[0] and not applied
    terrain = {"source": "glo30", "max_spacing_km": 3.75, "cache_root": None}
    result = door._stage_mesh(args, {**base, "terrain": terrain}, Status())
    assert "--parent-static" in seen[1] and result == {"returncode": 0, "terrain": {"used": "x"}}
    assert applied[0]["request"] == terrain and applied[0]["vertical_spec"] == "spec.json"


def _mesh(tmp_path):
    """A generated mesh directory (``--parent-static`` route) of stand-in files."""
    from woof.hex import mesh_rows

    mesh = tmp_path / "mesh"
    mesh.mkdir()
    names = {key: mesh / f"m.{key}" for key in
             ("grid", "static", "vertical", "cull-grid", "cull-static", "cull-vertical", "region")}
    for key, path in names.items():
        path.write_bytes(f"30s {key}".encode())
    rows = mesh / mesh_rows.MESH_ROWS_FILENAME

    def row(kind, name, static, **extra):
        return mesh_rows.MeshRow(
            kind=kind, name=name, n_cells=1, n_edges=1, n_levels=1, n_interfaces=2, n_soil_levels=4,
            nominal_dx_m=1.0, dt_seconds=1.0, grid=str(names["grid"]), grid_bytes=1, grid_sha256="g",
            static=str(static), static_bytes=1, static_sha256="old", spec_sha256=None,
            generator_receipt=None, static_receipt=None, admission={}, **extra)

    mesh_rows.write_rows(rows, [row("generated-global", "parent", names["static"]),
                                row("generated-cull", "cull", names["cull-static"], parent_row="parent")])
    generated = mesh / "m.point-generate.json"
    generated.write_text(json.dumps({
        "route": "parent-static", "rows_file": str(rows),
        "parent": {"row": "parent", "grid": str(names["grid"]), "static": str(names["static"])},
        "cull": {"row": "cull", "grid": str(names["cull-grid"]), "static": str(names["cull-static"]),
                 "region": str(names["region"])},
        "vertical": {"parent_artifact": str(names["vertical"]), "cull_artifact": str(names["cull-vertical"])}}))
    return mesh, names, rows, generated


def _fake_doors(monkeypatch, selection=None):
    from woof.hex import cull_door, engines, static_highres, vertical_door

    calls = []

    def highres(static, out, **kw):
        calls.append(("static-highres", Path(static).name, Path(out).name, kw["terrain"], kw["max_spacing_km"]))
        Path(out).write_bytes(b"hr static")
        return {"cells_selected": 7, "terrain": {"source": "copernicus-dem-glo30", "delta_mean_abs_m": 1.5,
                                                 "delta_max_abs_m": 9.0},
                "sources": {"terrain_source": "copernicus-dem-glo30", "terrain_vertical_datum": "EGM2008",
                            "terrain_attribution": "Copernicus", "terrain_license": "Copernicus",
                            **({"terrain_source_selection": selection} if selection else {})}}

    def vertical(**kw):
        calls.append(("vertical", Path(kw["static"]).name, Path(kw["output"]).name))
        Path(kw["output"]).write_bytes(b"hr vertical")
        return {}

    def cull(engine, parent, region, out, **kw):
        calls.append(("cull", Path(parent).name, Path(out).name, Path(kw["provenance_grid"]).name,
                      Path(kw["mark_parent"]).name))
        Path(out).write_bytes(b"cut " + Path(parent).read_bytes())
        return {}

    monkeypatch.setattr(static_highres, "run_static_highres", highres)
    monkeypatch.setattr(vertical_door, "mint_vertical", vertical)
    monkeypatch.setattr(cull_door, "cull_one", cull)
    monkeypatch.setattr(cull_door, "carry_lineage", lambda *a, **k: {})
    monkeypatch.setattr(engines, "resolve", lambda *a, **k: Path("rw_mpas_mesh"))
    return calls


def test_apply_terrain_runs_the_chain_and_refreshes_the_rows(tmp_path, monkeypatch):
    from woof.hex import mesh_rows

    mesh, names, rows, generated = _mesh(tmp_path)
    calls = _fake_doors(monkeypatch)
    request = {"source": "glo30", "max_spacing_km": 3.75, "cache_root": None}
    receipt = rt.apply_terrain(mesh_dir=mesh, vertical_spec="spec.json", request=request, note=lambda t: None)
    assert [c[0] for c in calls] == ["static-highres", "vertical", "cull", "cull"]
    assert calls[0][1:] == ("m.static", "m.static", "glo30", 3.75)  # the original moved to orig30s first
    assert (mesh / "orig30s" / "m.static").read_bytes() == b"30s static"
    assert names["static"].read_bytes() == b"hr static"
    assert names["cull-static"].read_bytes() == b"cut hr static"
    assert names["cull-vertical"].read_bytes() == b"cut hr vertical"
    assert names["cull-grid"].read_bytes() == b"30s cull-grid"  # the cull grid is untouched
    by_name = {row.name: row for row in mesh_rows.read_rows(rows)}
    assert by_name["parent"].static_sha256 == mesh_rows.sha256_file(names["static"])
    assert by_name["cull"].static_sha256 == mesh_rows.sha256_file(names["cull-static"])
    assert by_name["cull"].static_bytes == names["cull-static"].stat().st_size
    assert receipt["requested"] == "glo30" and receipt["used"] == "copernicus-dem-glo30"
    assert receipt["fallback"] is False and receipt["datum"] == "EGM2008"
    assert json.loads((mesh / rt.RECEIPT_NAME).read_text()) == receipt
    assert json.loads(generated.read_text())["terrain"] == receipt
    # a second application is refused before anything moves
    with pytest.raises(rt.RegionalTerrainRefusal, match="already"):
        rt.apply_terrain(mesh_dir=mesh, vertical_spec="spec.json", request=request, note=lambda t: None)


def test_apply_terrain_records_the_gem15_fallback(tmp_path, monkeypatch):
    mesh, *_ = _mesh(tmp_path)
    selection = {"requested": "gem15-dtm", "used": "copernicus-dem-glo30", "fallback_reason": "outside coverage"}
    _fake_doors(monkeypatch, selection)
    receipt = rt.apply_terrain(mesh_dir=mesh, vertical_spec="spec.json", note=lambda t: None,
                               request={"source": "gem15-dtm", "max_spacing_km": 1.0, "cache_root": None})
    assert receipt["fallback"] is True and receipt["selection"] == selection


def test_a_refused_replacement_restores_the_30s_mesh(tmp_path, monkeypatch):
    from woof.hex import static_highres

    mesh, names, rows, _ = _mesh(tmp_path)
    before = {key: path.read_bytes() for key, path in names.items()}
    rows_before = rows.read_bytes()
    _fake_doors(monkeypatch)

    def refuse(*a, **k):
        raise static_highres.StaticHighresRefusal("no terrain coverage")

    monkeypatch.setattr(static_highres, "run_static_highres", refuse)
    with pytest.raises(static_highres.StaticHighresRefusal):
        rt.apply_terrain(mesh_dir=mesh, vertical_spec="spec.json", note=lambda t: None,
                         request={"source": "glo30", "max_spacing_km": 1.0, "cache_root": None})
    assert {key: path.read_bytes() for key, path in names.items()} == before
    assert rows.read_bytes() == rows_before and not (mesh / rt.RECEIPT_NAME).exists()


def test_the_halo_route_is_refused(tmp_path):
    mesh, *_, generated = _mesh(tmp_path)
    value = json.loads(generated.read_text())
    value["route"] = "halo"
    generated.write_text(json.dumps(value))
    with pytest.raises(rt.RegionalTerrainRefusal, match="--parent-static"):
        rt.apply_terrain(mesh_dir=mesh, vertical_spec="spec.json", note=lambda t: None,
                         request={"source": "glo30", "max_spacing_km": 1.0, "cache_root": None})


def test_plan_needs_no_mesh(tmp_path):
    """``--stage plan`` on a fresh work directory plans; the mesh it would build does not exist yet."""
    args = parse(tmp_path, "--terrain-source", "glo30", "--stage", "plan")
    assert rt.request_from_arguments(args, fine_dx_m=3000.0, has_parent=False)["source"] == "glo30"


def test_a_separate_stage_must_match_the_meshs_terrain(tmp_path):
    """A stage run on its own consumes the mesh built earlier and runs no mesh stage to check it: a GLO-30 mesh is
    refused for a GEM-15 request, and for another spacing limit, not just for a missing receipt."""
    mesh = tmp_path / "run" / "mesh"
    mesh.mkdir(parents=True)
    (mesh / rt.RECEIPT_NAME).write_text(json.dumps({"requested": "glo30", "max_spacing_km": 3.75}))
    ok = parse(tmp_path, "--terrain-source", "glo30", "--stage", "forecast")
    assert rt.request_from_arguments(ok, fine_dx_m=3000.0, has_parent=False)["source"] == "glo30"
    for extra in (["--terrain-source", "gem15-dtm"], ["--terrain-source", "glo30", "--terrain-max-spacing-km", "9"]):
        with pytest.raises(rt.RegionalTerrainRefusal, match="was built with"):
            rt.request_from_arguments(parse(tmp_path, *extra, "--stage", "forecast"), fine_dx_m=3000.0,
                                      has_parent=False)


def test_the_replacement_cut_is_marked_on_the_parent_grid(tmp_path, monkeypatch):
    mesh, names, *_ = _mesh(tmp_path)
    calls = _fake_doors(monkeypatch)
    rt.apply_terrain(mesh_dir=mesh, vertical_spec="spec.json", note=lambda t: None,
                     request={"source": "glo30", "max_spacing_km": 1.0, "cache_root": None})
    assert [c[4] for c in calls if c[0] == "cull"] == ["m.grid", "m.grid"]


def test_a_rolled_back_replacement_is_retried_by_the_mesh_stage(tmp_path, monkeypatch):
    """THE BREAKAGE: after a rollback the point-generate receipt remained and no terrain receipt existed, so the
    next mesh stage refused the mesh as one built without terrain instead of retrying the replacement."""
    from woof.hex import static_highres

    mesh, names, *_ = _mesh(tmp_path)
    calls = _fake_doors(monkeypatch)
    good = static_highres.run_static_highres

    def refuse(*a, **k):
        raise static_highres.StaticHighresRefusal("source acquisition failed")

    monkeypatch.setattr(static_highres, "run_static_highres", refuse)
    request = {"source": "glo30", "max_spacing_km": 1.0, "cache_root": None}
    with pytest.raises(static_highres.StaticHighresRefusal):
        rt.apply_terrain(mesh_dir=mesh, vertical_spec="spec.json", note=lambda t: None, request=request)
    assert rt.pending(mesh)
    monkeypatch.setattr(static_highres, "run_static_highres", good)

    class Status:
        def note(self, text):
            pass

    monkeypatch.setattr(door, "_start_source_prefetch", lambda *a: None)
    value = {"paths": {"mesh": str(mesh), "work": str(tmp_path), "vertical_spec": "spec.json"}, "terrain": request}
    result = door._stage_mesh(parse(tmp_path), value, Status())
    assert result["reused"] and result["terrain"]["requested"] == "glo30"
    assert not rt.pending(mesh) and (mesh / rt.RECEIPT_NAME).is_file()
    assert names["static"].read_bytes() == b"hr static"
    # once applied, a further mesh stage reuses it
    assert door._stage_mesh(parse(tmp_path), value, Status())["terrain"]["requested"] == "glo30"


def test_a_failure_after_the_cut_restores_the_rows_and_receipt_and_retries(tmp_path, monkeypatch):
    """THE BREAKAGE: a failure after the replacement cut (the rows refresh, its write, the receipts) was outside
    the rollback, so the replacements stayed beside their orig30s copies and every retry was refused as "applied
    here already"."""
    from woof.hex import mesh_rows

    mesh, names, rows, generated = _mesh(tmp_path)
    before = {key: path.read_bytes() for key, path in names.items()}
    rows_before, generated_before = rows.read_bytes(), generated.read_bytes()
    _fake_doors(monkeypatch)
    write_rows = mesh_rows.write_rows

    def broken(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(mesh_rows, "write_rows", broken)
    request = {"source": "glo30", "max_spacing_km": 1.0, "cache_root": None}
    with pytest.raises(OSError, match="disk full"):
        rt.apply_terrain(mesh_dir=mesh, vertical_spec="spec.json", note=lambda t: None, request=request)
    assert {key: path.read_bytes() for key, path in names.items()} == before
    assert rows.read_bytes() == rows_before and generated.read_bytes() == generated_before
    assert rt.pending(mesh) and not (mesh / rt.RECEIPT_NAME).exists()
    monkeypatch.setattr(mesh_rows, "write_rows", write_rows)
    receipt = rt.apply_terrain(mesh_dir=mesh, vertical_spec="spec.json", note=lambda t: None, request=request)
    assert receipt["requested"] == "glo30" and not rt.pending(mesh)
    assert names["static"].read_bytes() == b"hr static"
    assert not list((mesh / "orig30s").glob("pre-terrain.*"))


def test_a_replacement_killed_part_way_is_rolled_back_by_the_retry(tmp_path, monkeypatch):
    """A killed process runs no exception handler: the retry puts the manifest's files back before it starts."""
    mesh, names, rows, generated = _mesh(tmp_path)
    before = {key: path.read_bytes() for key, path in names.items()}
    rows_before = rows.read_bytes()
    _fake_doors(monkeypatch)
    monkeypatch.setattr(rt, "_roll_back", lambda mesh_dir: None)        # the process dies: nothing runs
    from woof.hex import vertical_door

    def killed(**kw):
        raise KeyboardInterrupt

    monkeypatch.setattr(vertical_door, "mint_vertical", killed)
    request = {"source": "glo30", "max_spacing_km": 1.0, "cache_root": None}
    with pytest.raises(KeyboardInterrupt):
        rt.apply_terrain(mesh_dir=mesh, vertical_spec="spec.json", note=lambda t: None, request=request)
    assert names["static"].read_bytes() == b"hr static"               # left half replaced
    monkeypatch.undo()
    calls = _fake_doors(monkeypatch)
    assert rt.pending(mesh)
    receipt = rt.apply_terrain(mesh_dir=mesh, vertical_spec="spec.json", note=lambda t: None, request=request)
    assert receipt["requested"] == "glo30"
    assert calls[0][1:] == ("m.static", "m.static", "glo30", 1.0)
    assert (mesh / "orig30s" / "m.static").read_bytes() == before["static"]
    assert names["cull-static"].read_bytes() == b"cut hr static"
    assert rows.read_bytes() != rows_before and not rt.pending(mesh)


def _rolled_back(tmp_path, monkeypatch):
    from woof.hex import static_highres

    mesh, names, *_ = _mesh(tmp_path)
    _fake_doors(monkeypatch)

    def refuse(*a, **k):
        raise static_highres.StaticHighresRefusal("source acquisition failed")

    good = static_highres.run_static_highres
    monkeypatch.setattr(static_highres, "run_static_highres", refuse)
    request = {"source": "glo30", "max_spacing_km": 1.0, "cache_root": None}
    with pytest.raises(static_highres.StaticHighresRefusal):
        rt.apply_terrain(mesh_dir=mesh, vertical_spec="spec.json", note=lambda t: None, request=request)
    monkeypatch.setattr(static_highres, "run_static_highres", good)
    monkeypatch.setattr(door, "_start_source_prefetch", lambda *a: None)

    class Status:
        def note(self, text):
            pass

    return mesh, request, Status()


def test_an_unfinished_replacement_is_not_reused_without_its_terrain(tmp_path, monkeypatch):
    mesh, request, status = _rolled_back(tmp_path, monkeypatch)
    value = {"paths": {"mesh": str(mesh), "work": str(tmp_path), "vertical_spec": "spec.json"}, "terrain": None}
    with pytest.raises(door.RegionalRefusal, match="--terrain-source glo30 to finish it"):
        door._stage_mesh(parse(tmp_path), value, status)


def test_an_unfinished_replacement_is_not_retried_under_built_artifacts(tmp_path, monkeypatch):
    """THE BREAKAGE: an init (or boundary, contract) stage run between the rollback and the retry built on the
    30 s terrain; the retry then replaced the mesh's terrain under it and the reuse guards kept it."""
    mesh, request, status = _rolled_back(tmp_path, monkeypatch)
    init = tmp_path / "init" / "init.nc"
    init.parent.mkdir()
    init.write_bytes(b"init on 30s")
    value = {"paths": {"mesh": str(mesh), "work": str(tmp_path), "vertical_spec": "spec.json",
                       "init": str(init), "contract": str(tmp_path / "contract")}, "terrain": request}
    with pytest.raises(door.RegionalRefusal, match="init.nc was built on the mesh"):
        door._stage_mesh(parse(tmp_path), value, status)
    assert rt.pending(mesh)
    init.unlink()
    assert door._stage_mesh(parse(tmp_path), value, status)["terrain"]["requested"] == "glo30"


@pytest.mark.parametrize("killed_at", [rt.PENDING_NAME, "m.point-generate.json", rt.RECEIPT_NAME])
def test_a_replacement_killed_while_writing_a_record_is_retried(tmp_path, monkeypatch, killed_at):
    """The marker, the point-generate receipt and the terrain receipt are written whole or not at all.  THE
    BREAKAGE: a process stopped mid-write left half a JSON file that the retry (or require_applied) could not
    parse, or a half terrain receipt that made pending() skip the retry."""
    import os

    mesh, names, rows, generated = _mesh(tmp_path)
    generated_before = generated.read_bytes()
    _fake_doors(monkeypatch)
    monkeypatch.setattr(rt, "_roll_back", lambda mesh_dir: None)        # the process dies: nothing runs
    replace = os.replace

    def dies(src, dst):
        if Path(dst).name == killed_at:
            raise KeyboardInterrupt
        return replace(src, dst)

    monkeypatch.setattr(rt.os, "replace", dies)
    request = {"source": "glo30", "max_spacing_km": 1.0, "cache_root": None}
    with pytest.raises(KeyboardInterrupt):
        rt.apply_terrain(mesh_dir=mesh, vertical_spec="spec.json", note=lambda t: None, request=request)
    assert not (mesh / rt.RECEIPT_NAME).exists()
    assert rt.pending(mesh)
    json.loads(generated.read_text())                                    # never half written
    monkeypatch.undo()
    _fake_doors(monkeypatch)
    monkeypatch.setattr(door, "_start_source_prefetch", lambda *a: None)

    class Status:
        def note(self, text):
            pass

    value = {"paths": {"mesh": str(mesh), "work": str(tmp_path), "vertical_spec": "spec.json"}, "terrain": request}
    result = door._stage_mesh(parse(tmp_path), value, Status())
    assert result["terrain"]["requested"] == "glo30" and not rt.pending(mesh)
    assert names["static"].read_bytes() == b"hr static"
    assert json.loads(generated.read_text())["terrain"]["requested"] == "glo30"
    assert not list(mesh.glob("*.partial")) and generated_before != generated.read_bytes()


def test_a_rollback_killed_while_restoring_the_receipt_is_retried(tmp_path, monkeypatch):
    """THE BREAKAGE: the rollback unlinked the point-generate receipt before moving its backup back; a process
    killed between the two left no receipt, so the next mesh stage skipped recovery and refused the grid."""
    import shutil
    from woof.hex import static_highres

    mesh, names, rows, generated = _mesh(tmp_path)
    _fake_doors(monkeypatch)

    def refuse(*a, **k):
        raise static_highres.StaticHighresRefusal("source acquisition failed")

    def dies_on_receipt(real):
        def wrapped(src, dst, *a, **k):
            if Path(dst).name == generated.name and "orig30s" in str(src):
                raise KeyboardInterrupt
            return real(src, dst, *a, **k)
        return wrapped

    monkeypatch.setattr(static_highres, "run_static_highres", refuse)
    monkeypatch.setattr(rt.os, "replace", dies_on_receipt(rt.os.replace))
    monkeypatch.setattr(rt.shutil, "move", dies_on_receipt(shutil.move))
    request = {"source": "glo30", "max_spacing_km": 1.0, "cache_root": None}
    with pytest.raises(KeyboardInterrupt):
        rt.apply_terrain(mesh_dir=mesh, vertical_spec="spec.json", note=lambda t: None, request=request)
    assert generated.is_file()
    monkeypatch.undo()
    _fake_doors(monkeypatch)
    monkeypatch.setattr(door, "_start_source_prefetch", lambda *a: None)

    class Status:
        def note(self, text):
            pass

    value = {"paths": {"mesh": str(mesh), "work": str(tmp_path), "vertical_spec": "spec.json"}, "terrain": request}
    result = door._stage_mesh(parse(tmp_path), value, Status())
    assert result["terrain"]["requested"] == "glo30" and not rt.pending(mesh)
