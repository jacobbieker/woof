"""Finite-difference sensitivities of ``J_o`` through perturbed restart states.

A direction ``p`` is a set of increments to restart-checkpoint fields
(``thp``, ``u``, ``v``, ``qv``, ... on their own staggered grids).  The
perturbed initial states ``x0 +/- eps p`` are written through
:func:`woof.ensemble.increments.apply_increments_to_checkpoint`, the same
writer an analysis uses (vapour capped at saturation, microphysics moments
guarded), and the forecast is continued with ``woof go --restart``.  The
central difference ``(J(x0 + eps p) - J(x0 - eps p)) / (2 eps)`` is then the
directional derivative of ``J_o`` along ``p``.

On hex the same is done with an MPAS-layout state file: the JEDI background
a run writes at its checkpoint time (``--checkpoint-hours`` with
``--jedi-background``) is copied with ``scale * p`` added
(:func:`perturb_mpas_file`) and given to ``--restart-from CKPT --da-analysis
FILE``, which recouples the dycore state from it.

Nothing here runs a model; a direction file plus two scored forecasts is all
:func:`fd_summary` reads.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

DIRECTION_SCHEMA = "woof.grad.direction/v1"
FD_SCHEMA = "woof.grad.fd/v1"


class SensitivityError(ValueError):
    """A direction, checkpoint or finite-difference input that cannot be used."""


def checkpoint_fields(path, names: Sequence[str] | None = None) -> dict[str, np.ndarray]:
    """The ``state/<name>`` arrays of a restart checkpoint (all of them when ``names`` is None)."""
    with np.load(path, allow_pickle=False) as data:
        available = [key[len("state/"):] for key in data.files if key.startswith("state/")]
        wanted = available if names is None else list(names)
        missing = sorted(set(wanted) - set(available))
        if missing:
            raise SensitivityError(f"checkpoint {path} carries no state field(s) {missing}")
        return {name: np.array(data[f"state/{name}"]) for name in wanted}


def gaussian_blob(shape: Sequence[int], *, k: float, j: float, i: float,
                  radius_cells: float, depth_levels: float, amplitude: float = 1.0) -> np.ndarray:
    """``amplitude * exp(-r^2 / 2 radius^2 - dk^2 / 2 depth^2)`` on a ``(nz, ny, nx)`` field's own index grid.

    The centre is in the field's own indices, so for a staggered field a
    half-cell shift is the caller's choice.
    """
    shape = tuple(int(n) for n in shape)
    if len(shape) != 3 or min(shape) < 1:
        raise SensitivityError(f"a blob needs a (nz, ny, nx) shape, got {shape}")
    if not (radius_cells > 0 and depth_levels > 0 and np.isfinite(amplitude)):
        raise SensitivityError("a blob needs a positive radius and depth and a finite amplitude")
    kk, jj, ii = (np.arange(n, dtype=np.float64) for n in shape)
    vertical = np.exp(-0.5 * ((kk - k) / depth_levels) ** 2)
    horizontal = np.exp(-0.5 * (((jj[:, None] - j) ** 2 + (ii[None, :] - i) ** 2) / radius_cells ** 2))
    return amplitude * vertical[:, None, None] * horizontal[None, :, :]


def direction_digest(fields: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(fields):
        array = np.ascontiguousarray(fields[name], dtype=np.float64)
        digest.update(name.encode() + b"\0" + repr(array.shape).encode() + b"\0")
        digest.update(array.tobytes())
    return digest.hexdigest()


def save_direction(path, fields: Mapping[str, np.ndarray], meta: Mapping | None = None) -> str:
    """Write a direction (``field -> array``) and return its digest."""
    if not fields:
        raise SensitivityError("a direction needs at least one field")
    sha = direction_digest(fields)
    np.savez(path, schema=np.asarray(DIRECTION_SCHEMA), sha256=np.asarray(sha),
             meta=np.asarray(json.dumps(dict(meta or {}), sort_keys=True)),
             names=np.asarray(sorted(fields)),
             **{f"field/{name}": np.asarray(fields[name], dtype=np.float64) for name in fields})
    return sha


def load_direction(path) -> tuple[dict[str, np.ndarray], dict]:
    with np.load(path, allow_pickle=False) as data:
        if "schema" not in data.files or str(data["schema"]) != DIRECTION_SCHEMA:
            raise SensitivityError(f"{path} is not a {DIRECTION_SCHEMA} file")
        fields = {str(name): np.array(data[f"field/{name}"]) for name in data["names"]}
        meta = json.loads(str(data["meta"]))
        meta["sha256"] = str(data["sha256"])
    if direction_digest(fields) != meta["sha256"]:
        raise SensitivityError(f"{path}: the stored fields do not match the stored digest")
    return fields, meta


def perturb_checkpoint(source, direction: Mapping[str, np.ndarray], scale: float, destination, *,
                       mp_physics: int | None = None) -> dict:
    """Write ``source + scale * direction`` to ``destination`` through the analysis writer; return its receipt."""
    from woof.ensemble.increments import apply_increments_to_checkpoint

    if not np.isfinite(scale):
        raise SensitivityError(f"scale must be finite, got {scale!r}")
    increments = {name: float(scale) * np.asarray(values, dtype=np.float64) for name, values in direction.items()}
    receipt = apply_increments_to_checkpoint(source, increments, destination, publish=True, mp_physics=mp_physics)
    receipt["direction_sha256"] = direction_digest(direction)
    receipt["scale"] = float(scale)
    return receipt


def _cost(entry) -> tuple[float, dict]:
    """``(J, summary)`` from a ``woof grad cost`` summary file, a summary dict or a bare number."""
    if isinstance(entry, (int, float)):
        return float(entry), {}
    summary = entry if isinstance(entry, Mapping) else json.loads(Path(entry).read_text())
    return float(summary["J"]), summary


def _same_definition(reference: Mapping | None, frozen: Sequence[Mapping]) -> None:
    """Every scored run must use one observation set and error table, and the perturbed runs one frozen pair set."""
    seen = {}
    for summary in ([reference] if reference else []) + [s for s in frozen if s]:
        for key in ("obs_sha256", "error_table_sha256"):
            if key in seen and seen[key] != summary.get(key):
                raise SensitivityError(f"the scored runs differ in {key} ({seen[key]} != {summary.get(key)}); "
                                       "a difference of J is only meaningful for one definition of J")
            seen[key] = summary.get(key)
    # The pairs file's digest when the summary has one (paths differ between working directories).
    accepted = {s.get("accepted_sha256") or s.get("accepted_from") for s in frozen if s}
    if None in accepted or len(accepted) > 1:
        raise SensitivityError("score every perturbed run with --accept and the same reference pairs file "
                               f"(frozen observation set); got {sorted(map(str, accepted))}")
    for summary in frozen:
        lost = (summary or {}).get("counts", {}).get("lost_from_reference")
        if lost:
            raise SensitivityError(f"a scored run lost {lost} of the reference's observations, so J counts "
                                   "different observations; pick a smaller scale")


def fd_summary(legs: Sequence[tuple[float, object, object]], *, reference=None, zero=None) -> dict:
    """Central differences for each ``(scale, plus, minus)`` leg.

    ``plus``/``minus`` are ``woof grad cost`` summaries (paths or dicts) or bare
    J values.  ``reference`` is the run written through the same writer
    with scale 0 (the writer's own moment repairs and vapour cap, no
    perturbation), for the one-sided differences and the curvature
    ``(J+ + J- - 2 J0) / scale^2``.  Summaries
    that carry ``by_type`` also give the central difference per observation
    type, which shows which observations carry the response (and the noise).  ``zero`` is
    a restart from the untouched checkpoint: its distance from the
    reference is what the writer's repairs alone change.  With two or more scales
    the relative change of the derivative between consecutive scales says
    whether the smallest one is in the linear range.
    """
    if not legs:
        raise SensitivityError("fd_summary needs at least one (scale, plus, minus) leg")
    summaries = []
    j0, s0 = None, None
    if reference is not None:
        j0, s0 = _cost(reference)
    floor = None
    if zero is not None:
        jz, sz = _cost(zero)
        summaries.append(sz)
        floor = None if j0 is None else abs(jz - j0)
    rows = []
    for scale, plus, minus in sorted(legs, key=lambda leg: -abs(float(leg[0]))):
        scale = float(scale)
        if not (np.isfinite(scale) and scale > 0):
            raise SensitivityError(f"scales must be positive, got {scale!r}")
        jp, sp = _cost(plus)
        jm, sm = _cost(minus)
        summaries += [sp, sm]
        row = {"scale": scale, "J_plus": jp, "J_minus": jm, "dJ_central": (jp - jm) / (2.0 * scale)}
        types = sorted(set(sp.get("by_type", {})) & set(sm.get("by_type", {})))
        if types:
            row["dJ_central_by_type"] = {t: (sp["by_type"][t]["J"] - sm["by_type"][t]["J"]) / (2.0 * scale)
                                         for t in types}
        if j0 is not None:
            row["dJ_forward"] = (jp - j0) / scale
            row["dJ_backward"] = (j0 - jm) / scale
            row["curvature"] = (jp + jm - 2.0 * j0) / scale ** 2
        if floor is not None:
            row["difference_over_repair_offset"] = (abs(jp - jm) / floor) if floor > 0 else float("inf")
        rows.append(row)
    if any(summaries):
        _same_definition(s0, summaries)
    for coarse, fine in zip(rows, rows[1:]):
        if fine["dJ_central"] != 0:
            fine["relative_change_from_larger_scale"] = abs(coarse["dJ_central"] - fine["dJ_central"]) / abs(fine["dJ_central"])
    best = rows[-1]
    return {"schema": FD_SCHEMA, "J_reference": j0, "repair_offset": floor, "legs": rows,
            "dJ": best["dJ_central"], "dJ_scale": best["scale"]}


# ---------------------------------------------------------------------------
# hex (MPAS layout): directions on cells, perturbed analysis files
# ---------------------------------------------------------------------------

EARTH_RADIUS_KM = 6371.0


def mesh_cells(path) -> dict:
    """Cell latitude/longitude (degrees) and ``bdyMaskCell`` (None on a global mesh) of an MPAS-layout file."""
    import netCDF4

    with netCDF4.Dataset(path) as d:
        v = d.variables
        if "latCell" not in v or "lonCell" not in v:
            raise SensitivityError(f"{path} carries no latCell/lonCell; pass a grid, static or init file")
        lat = np.degrees(np.asarray(v["latCell"][:], np.float64))
        lon = np.degrees(np.asarray(v["lonCell"][:], np.float64))
        bdy = np.asarray(v["bdyMaskCell"][:], np.int64) if "bdyMaskCell" in v else None
        nz = len(d.dimensions["nVertLevels"]) if "nVertLevels" in d.dimensions else None
    return {"lat": lat, "lon": lon, "bdy": bdy, "nz": nz}


def great_circle_km(lat, lon, lat0, lon0) -> np.ndarray:
    lat, lon = np.radians(lat), np.radians(lon)
    lat0, lon0 = np.radians(lat0), np.radians(lon0)
    h = np.sin((lat - lat0) / 2) ** 2 + np.cos(lat) * np.cos(lat0) * np.sin((lon - lon0) / 2) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.minimum(h, 1.0)))


def cell_blob(lat, lon, nz: int, *, lat0: float, lon0: float, level: float, radius_km: float,
              depth_levels: float, amplitude: float = 1.0, interior=None) -> np.ndarray:
    """A Gaussian blob ``(nCells, nz)`` in great-circle distance and level, zero outside ``interior``."""
    if not (radius_km > 0 and depth_levels > 0 and np.isfinite(amplitude)):
        raise SensitivityError("a blob needs a positive radius and depth and a finite amplitude")
    r = great_circle_km(np.asarray(lat), np.asarray(lon), lat0, lon0)
    k = np.arange(int(nz), dtype=np.float64)
    out = amplitude * np.exp(-0.5 * (r / radius_km) ** 2)[:, None] * np.exp(-0.5 * ((k - level) / depth_levels) ** 2)[None, :]
    if interior is not None:
        out = out * np.asarray(interior, bool)[:, None]
    return out


def perturb_mpas_file(source, direction: Mapping[str, np.ndarray], scale: float, destination) -> dict:
    """Copy an MPAS-layout state file (e.g. the hex JEDI background) adding ``scale * direction``.

    Each direction field is ``(nCells, nVertLevels)`` and is added to the
    variable's first record.  The copy is then a ``--da-analysis`` for a hex
    ``--restart-from`` run.
    """
    import os
    import shutil
    import uuid

    import netCDF4

    src, dst = Path(source), Path(destination)
    if dst.exists() and dst.resolve() == src.resolve():
        raise SensitivityError(f"refusing to write {src} over itself")
    if not np.isfinite(scale):
        raise SensitivityError(f"scale must be finite, got {scale!r}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f"{dst.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    shutil.copyfile(src, tmp)
    fields = []
    try:
        with netCDF4.Dataset(tmp, "r+") as d:
            for name, values in sorted(direction.items()):
                if name not in d.variables:
                    raise SensitivityError(f"{src} has no variable {name!r}")
                var = d.variables[name]
                if var.dimensions[:2] != ("Time", "nCells") or var.ndim != 3:
                    raise SensitivityError(f"{name} is {var.dimensions}; a direction adds to (Time, nCells, level) fields")
                old = np.asarray(var[0], np.float64)
                values = np.asarray(values, np.float64)
                if values.shape != old.shape:
                    raise SensitivityError(f"direction {name} is {values.shape}, {src} holds {old.shape}")
                new = old + float(scale) * values
                if not np.all(np.isfinite(new)):
                    raise SensitivityError(f"{name} + scale * direction is not finite")
                var[0] = new.astype(var.dtype)
                fields.append({"field": name, "max_abs_increment": float(np.max(np.abs(float(scale) * values)))})
        tmp.replace(dst)
    finally:
        if tmp.exists():
            tmp.unlink()
    return {"source": str(src), "destination": str(dst), "scale": float(scale), "fields": fields,
            "direction_sha256": direction_digest(direction)}
