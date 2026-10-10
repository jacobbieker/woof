"""Precision probes for fine-mesh, small-timestep hex runs.

Two measurements, both CPU, both float32 against float64:

1. :func:`deformation_geometry_error` -- how far the v8.4.1 deformation
   weights land from a binary64 analytic reference on a synthetic patch of
   plane-like regular hexagons placed on the sphere, for each geometry
   evaluation mode of :func:`woof.hex.mixing_v841.initialize_deformation_weights_v841`.
2. :func:`large_step_increment_loss` -- how much of the theta and momentum
   increments a run of small timesteps loses to binary32 rounding of the
   carried large-step state, using the CPU recovery authority
   (:func:`woof.hex.integration.recover_large_step_variables`) and the
   acoustic perturbation accumulation in the same dtype.

Neither probe changes model arithmetic.  The synthetic patch builder
(:func:`synthetic_hex_patch`) is shared with the tests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

EARTH_RADIUS_M = 6_371_229.0


@dataclass
class SyntheticPatchMesh:
    """A minimal MPAS-shaped mesh: arrays by MPAS name plus global attrs."""

    arrays: dict[str, NDArray[Any]]
    attrs: dict[str, Any] = field(default_factory=dict)
    #: Index of the cells whose full stencil is inside the patch.
    interior_cells: NDArray[np.int64] = field(
        default_factory=lambda: np.zeros(0, dtype=np.int64)
    )
    #: Plane geometry the patch was built from.
    spacing_m: float = 0.0
    side_m: float = 0.0
    rotation_rad: float = 0.0


def _local_basis(lat_deg: float, lon_deg: float) -> tuple[NDArray[np.float64], ...]:
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    up = np.array([math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat)])
    east = np.array([-math.sin(lon), math.cos(lon), 0.0])
    north = np.cross(up, east)
    return up, east, north


def synthetic_hex_patch(
    spacing_m: float,
    *,
    rings: int = 4,
    lat_deg: float = 0.0,
    lon_deg: float = 45.0,
    rotation_deg: float = 7.0,
    radius_m: float = EARTH_RADIUS_M,
) -> SyntheticPatchMesh:
    """Regular hexagons of centre spacing ``spacing_m`` on a tangent plane,
    mapped onto the sphere by the inverse gnomonic projection.

    Every cell is a regular hexagon of side ``spacing_m / sqrt(3)`` in the
    plane; the projection distorts it by ``O((patch radius / R)**2)``, which
    is below 1e-9 for the patches the tests use.  Vertices are listed
    counter-clockwise about the outward normal; edge ``k`` of a cell joins
    its vertices ``k`` and ``k+1``.  Cells on the outer ring carry ``-1``
    neighbour slots, exactly like a regional cull's outermost rows.
    Coordinates are binary64 (``binary64_earth_centred``).
    """

    if spacing_m <= 0.0 or rings < 1:
        raise ValueError("spacing_m must be positive and rings >= 1")
    side = spacing_m / math.sqrt(3.0)
    phi = math.radians(rotation_deg)
    rot = np.array([[math.cos(phi), -math.sin(phi)], [math.sin(phi), math.cos(phi)]])
    axial: list[tuple[int, int]] = [
        (q, r)
        for q in range(-rings, rings + 1)
        for r in range(-rings, rings + 1)
        if abs(q) <= rings and abs(r) <= rings and abs(q + r) <= rings
    ]
    centres = np.array(
        [[spacing_m * (q + 0.5 * r), spacing_m * (math.sqrt(3.0) / 2.0) * r] for q, r in axial]
    )
    centres = centres @ rot.T
    corner_angles = phi + np.radians(30.0 + 60.0 * np.arange(6))
    corners = side * np.stack([np.cos(corner_angles), np.sin(corner_angles)], axis=-1)

    vertex_key: dict[tuple[int, int], int] = {}
    vertex_xy: list[NDArray[np.float64]] = []
    quantum = side * 1.0e-6
    verts_on_cell = np.zeros((len(axial), 6), dtype=np.int64)
    for cell, centre in enumerate(centres):
        for k in range(6):
            point = centre + corners[k]
            key = (int(round(point[0] / quantum)), int(round(point[1] / quantum)))
            index = vertex_key.get(key)
            if index is None:
                index = len(vertex_xy)
                vertex_key[key] = index
                vertex_xy.append(point)
            verts_on_cell[cell, k] = index

    edge_key: dict[tuple[int, int], int] = {}
    cells_on_edge: list[list[int]] = []
    vertices_on_edge: list[list[int]] = []
    edges_on_cell = np.zeros((len(axial), 6), dtype=np.int64)
    for cell in range(len(axial)):
        for k in range(6):
            a = int(verts_on_cell[cell, k])
            b = int(verts_on_cell[cell, (k + 1) % 6])
            key = (min(a, b), max(a, b))
            index = edge_key.get(key)
            if index is None:
                index = len(cells_on_edge)
                edge_key[key] = index
                cells_on_edge.append([cell, -1])
                vertices_on_edge.append([a, b])
            else:
                cells_on_edge[index][1] = cell
            edges_on_cell[cell, k] = index
    coe = np.array(cells_on_edge, dtype=np.int64)
    cells_on_cell = np.full((len(axial), 6), -1, dtype=np.int64)
    for cell in range(len(axial)):
        for k in range(6):
            pair = coe[edges_on_cell[cell, k]]
            cells_on_cell[cell, k] = pair[1] if pair[0] == cell else pair[0]

    up, east, north = _local_basis(lat_deg, lon_deg)

    def to_sphere(xy: NDArray[np.float64]) -> NDArray[np.float64]:
        p = radius_m * up[None, :] + xy[:, :1] * east[None, :] + xy[:, 1:2] * north[None, :]
        return radius_m * p / np.linalg.norm(p, axis=-1, keepdims=True)

    cell_xyz = to_sphere(centres)
    vertex_xyz = to_sphere(np.array(vertex_xy))
    interior = np.flatnonzero(np.all(cells_on_cell >= 0, axis=1))
    arrays: dict[str, NDArray[Any]] = {
        "nEdgesOnCell": np.full(len(axial), 6, dtype=np.int64),
        "edgesOnCell": edges_on_cell,
        "cellsOnCell": cells_on_cell,
        "verticesOnCell": verts_on_cell,
        "cellsOnEdge": coe,
        "verticesOnEdge": np.array(vertices_on_edge, dtype=np.int64),
        "xCell": cell_xyz[:, 0].copy(),
        "yCell": cell_xyz[:, 1].copy(),
        "zCell": cell_xyz[:, 2].copy(),
        "xVertex": vertex_xyz[:, 0].copy(),
        "yVertex": vertex_xyz[:, 1].copy(),
        "zVertex": vertex_xyz[:, 2].copy(),
        "dcEdge": np.full(coe.shape[0], spacing_m),
        "dvEdge": np.full(coe.shape[0], side),
        "areaCell": np.full(len(axial), 1.5 * math.sqrt(3.0) * side * side),
    }
    attrs = {
        "on_a_sphere": "YES",
        "sphere_radius": float(radius_m),
        "rw_coordinate_representation": "binary64_earth_centred",
    }
    return SyntheticPatchMesh(
        arrays=arrays,
        attrs=attrs,
        interior_cells=interior,
        spacing_m=float(spacing_m),
        side_m=float(side),
        rotation_rad=float(phi),
    )


def analytic_deformation_weights(
    patch: SyntheticPatchMesh,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Binary64 closed-form weights of a regular plane hexagon.

    Edge ``k`` of every cell joins corners at ``phi + 30 + 60k`` and
    ``phi + 90 + 60k`` degrees, so its outward normal points at
    ``phi + 60(k+1)`` degrees from local east; every edge is one side ``s``
    long and the area is ``3 sqrt(3)/2 s**2``.  ``coef_c2 = s cos**2 / A``
    etc., negated where the cell is not ``cellsOnEdge[edge, 0]``.  Valid for
    cells whose local north is the patch's north to the patch's distortion
    order (the default equatorial patch), and zero for non-interior cells,
    as the halo guard leaves them.
    """

    arrays = patch.arrays
    n_cells = arrays["nEdgesOnCell"].size
    s = patch.side_m
    area = 1.5 * math.sqrt(3.0) * s * s
    theta = patch.rotation_rad + np.radians(60.0 * (np.arange(6) + 1))
    c2 = s * np.cos(theta) ** 2 / area
    s2 = s * np.sin(theta) ** 2 / area
    cs = s * np.sin(theta) * np.cos(theta) / area
    sign = np.where(
        arrays["cellsOnEdge"][arrays["edgesOnCell"], 0] == np.arange(n_cells)[:, None],
        1.0,
        -1.0,
    )
    out = []
    for base in (c2, s2, cs):
        full = np.zeros((n_cells, 6))
        full[patch.interior_cells] = (sign * base[None, :])[patch.interior_cells]
        out.append(full)
    return out[0], out[1], out[2]


def _relative_error(actual: NDArray[Any], reference: NDArray[Any], rows: NDArray[np.int64]) -> float:
    a = np.asarray(actual, dtype=np.float64)[rows]
    r = np.asarray(reference, dtype=np.float64)[rows]
    scale = float(np.max(np.abs(r)))
    return float(np.max(np.abs(a - r)) / scale) if scale > 0.0 else 0.0


def deformation_geometry_error(
    spacing_m: float,
    *,
    rings: int = 3,
    lat_deg: float = 0.0,
    lon_deg: float = 45.0,
) -> dict[str, Any]:
    """Max relative weight error per evaluation mode against the analytic
    regular-hexagon reference, on interior cells of a synthetic patch.

    The error is normalised by the largest reference weight (about
    ``2/(3 s)``), so components that are analytically zero do not divide by
    zero.  ``native32`` is the binary32 reference-build mirror,
    ``local64_32`` the binary64 local-plane evaluation cast to binary32,
    ``native64`` the binary64 Earth-centred mirror.
    """

    from .mixing_v841 import initialize_deformation_weights_v841

    patch = synthetic_hex_patch(spacing_m, rings=rings, lat_deg=lat_deg, lon_deg=lon_deg)
    reference = analytic_deformation_weights(patch)
    rows = patch.interior_cells
    result: dict[str, Any] = {"spacing_m": float(spacing_m), "interior_cells": int(rows.size)}
    runs = {
        "native32": dict(dtype=np.float32, geometry="native"),
        "local64_32": dict(dtype=np.float32, geometry="local64"),
        "native64": dict(dtype=np.float64, geometry="native"),
        "auto32": dict(dtype=np.float32, geometry="auto"),
    }
    names = ("coef_c2", "coef_s2", "coef_cs")
    computed = {}
    for label, kwargs in runs.items():
        weights = initialize_deformation_weights_v841(patch, **kwargs)
        computed[label] = weights
        result[label] = max(
            _relative_error(getattr(weights, name), ref, rows)
            for name, ref in zip(names, reference)
        )
        if label == "auto32":
            result["auto32_geometry"] = weights.geometry
    # Against the binary64 Earth-centred mirror: the reference that holds
    # off the equator too, where meridian convergence rotates each cell's
    # local frame away from the analytic patch frame.
    for label in ("native32", "local64_32"):
        result[label + "_vs_native64"] = max(
            _relative_error(
                getattr(computed[label], name), getattr(computed["native64"], name), rows
            )
            for name in names
        )
    return result


# ---------------------------------------------------------------------------
# small-timestep increment loss through the large-step recovery
# ---------------------------------------------------------------------------


def _standard_column(nlev: int, ztop: float = 20_000.0) -> dict[str, NDArray[np.float64]]:
    """Dry hydrostatic base state (isothermal 250 K reference) on flat zz=1."""

    z_w = np.linspace(0.0, ztop, nlev + 1)
    z = 0.5 * (z_w[:-1] + z_w[1:])
    rgas, cp, g, p0 = 287.0, 1004.5, 9.80616, 100_000.0
    t_ref = 250.0
    p_base = p0 * np.exp(-g * z / (rgas * t_ref))
    rho_base = p_base / (rgas * t_ref)
    exner_base = (p_base / p0) ** (rgas / cp)
    theta_base = t_ref / exner_base
    return {
        "z": z,
        "rho_base": rho_base,
        "theta_base": theta_base,
        "exner_base": exner_base,
        "pressure_base": p_base,
    }


def large_step_increment_loss(
    dt: float,
    *,
    simulated_seconds: float = 60.0,
    acoustic_steps: int = 2,
    nlev: int = 12,
    spacing_m: float = 100.0,
    theta_tendency: float = 2.0e-5,
    u_tendency: float = 2.0e-5,
    compensated: bool = False,
) -> dict[str, Any]:
    """Fraction of the intended theta/u increment lost to binary32 rounding.

    A synthetic regional patch (``synthetic_hex_patch``) carries a realistic
    state -- theta 2-40 K off a 250 K isothermal base, density 1-3 % off its
    base, normal wind 3-25 m/s of either sign -- and is advanced
    ``simulated_seconds / dt`` timesteps.  Each timestep runs the final-stage
    shape of the split-explicit step: the perturbations ``rtheta_pp`` and
    ``ru_p`` start at zero and accumulate ``acoustic_steps`` sub-steps of
    ``dts * rho * tendency`` in the working dtype; the CPU authority
    :func:`~woof.hex.integration.recover_large_step_variables` (stage 3,
    zero diabatic term) recovers ``rtheta_p = rtheta_p_save + rtheta_pp``,
    ``theta_m``, ``ru`` and ``u`` in the same dtype; and the recovered
    ``rtheta_p``/``ru`` are carried as the next step's saved state, exactly
    as ``atm_rk_integration_setup`` copies them.

    The tendencies are constant, so the intended increment after ``T``
    seconds is ``T * tendency``; the float64 run reproduces it to ~1e-10 and
    is reported alongside.  Metrics are of the state the MODEL READS
    (``theta_m`` and ``u`` as recovered), never of a residual it does not:

    * ``*_lost_fraction`` -- ``1 - mean(realised)/intended``, signed
      (positive: increment lost);
    * ``*_rms_relative_error`` -- RMS of the per-point error over the
      intended increment.

    ``compensated=True`` runs the recovery through
    :func:`~woof.hex.integration.recover_large_step_variables_compensated`,
    the candidate remedy this probe measures.
    """

    from .integration import (
        RecoveryBackground,
        RecoveryResidual,
        RecoveryState,
        recover_large_step_variables_compensated,
    )

    if dt <= 0.0 or simulated_seconds <= 0.0:
        raise ValueError("dt and simulated_seconds must be positive")
    n_steps = int(round(simulated_seconds / dt))
    patch = synthetic_hex_patch(spacing_m, rings=2)
    arrays = patch.arrays
    n_cells = arrays["nEdgesOnCell"].size
    n_edges = arrays["cellsOnEdge"].shape[0]
    # Recovery needs density on both sides of every edge; close the patch's
    # open edges onto their own cell (an inert self-pair for this probe).
    coe = arrays["cellsOnEdge"].copy()
    coe[:, 1] = np.where(coe[:, 1] < 0, coe[:, 0], coe[:, 1])
    mesh = SyntheticPatchMesh(arrays=dict(arrays, cellsOnEdge=coe), attrs=dict(patch.attrs))

    col = _standard_column(nlev)
    rng = np.random.default_rng(20261010)
    theta_off = rng.uniform(2.0, 40.0, size=(nlev, n_cells))
    rho_off = rng.uniform(0.01, 0.03, size=(nlev, n_cells))
    u0 = rng.uniform(3.0, 25.0, size=(nlev, n_edges)) * rng.choice(
        [-1.0, 1.0], size=(nlev, n_edges)
    )
    rho_base = np.repeat(col["rho_base"][:, None], n_cells, axis=1)
    theta_base = np.repeat(col["theta_base"][:, None], n_cells, axis=1)
    exner_base = np.repeat(col["exner_base"][:, None], n_cells, axis=1)
    rho_zz0 = rho_base * (1.0 + rho_off)
    theta0 = theta_base + theta_off
    rtheta_base = rho_base * theta_base

    report: dict[str, Any] = {
        "dt": float(dt),
        "steps": n_steps,
        "simulated_seconds": float(n_steps * dt),
        "acoustic_steps": int(acoustic_steps),
        "theta_tendency_K_per_s": float(theta_tendency),
        "u_tendency_m_per_s2": float(u_tendency),
        "compensated": bool(compensated),
    }
    for dtype in (np.float64, np.float32):
        f = np.dtype(dtype).type
        rho_zz = rho_zz0.astype(dtype)
        rho_p = (rho_zz0 - rho_base).astype(dtype)
        rtheta_p = (rho_zz0 * theta0 - rtheta_base).astype(dtype)
        rho_edge64 = 0.5 * (rho_zz0[:, coe[:, 0]] + rho_zz0[:, coe[:, 1]])
        ru = (rho_edge64 * u0).astype(dtype)
        background = RecoveryBackground(
            rho_base=rho_base.astype(dtype),
            rtheta_base=rtheta_base.astype(dtype),
            exner_base=exner_base.astype(dtype),
            zz=np.ones((nlev, n_cells), dtype=dtype),
        )
        dts = f(dt / acoustic_steps)
        tend_rt = rho_zz * f(theta_tendency)
        tend_ru = rho_edge64.astype(dtype) * f(u_tendency)
        zeros_c = np.zeros((nlev, n_cells), dtype=dtype)
        zeros_w = np.zeros((nlev + 1, n_cells), dtype=dtype)
        zeros_e = np.zeros((nlev, n_edges), dtype=dtype)
        zb = np.zeros((nlev + 1, n_cells, 6), dtype=dtype)
        fz = np.full(nlev + 1, 0.5, dtype=dtype)

        def _state(rtheta_pp: Any, ru_p: Any) -> RecoveryState:
            return RecoveryState(
                ww_avg=zeros_w.copy(), rw_save=zeros_w.copy(), w=zeros_w.copy(),
                rw=zeros_w.copy(), rw_p=zeros_w.copy(),
                rtheta_p=rtheta_p.copy(), rtheta_pp=rtheta_pp,
                rtheta_p_save=rtheta_p.copy(),
                rho_p=rho_p.copy(), rho_p_save=rho_p.copy(), rho_pp=zeros_c.copy(),
                rho_zz=rho_zz.copy(), ru_avg=zeros_e.copy(), ru_save=ru.copy(),
                ru_p=ru_p, u=zeros_e.copy(), ru=ru.copy(),
                exner=np.ones((nlev, n_cells), dtype=dtype),
                pressure_p=zeros_c.copy(), theta_m=zeros_c.copy(),
            )

        residual = RecoveryResidual.zeros_like(_state(zeros_c, zeros_e))
        theta_start: Any = None
        u_start: Any = None
        out: Any = None
        for _ in range(n_steps):
            rtheta_pp = zeros_c.copy()
            ru_p = zeros_e.copy()
            for _sub in range(acoustic_steps):
                rtheta_pp = np.add(rtheta_pp, dts * tend_rt)
                ru_p = np.add(ru_p, dts * tend_ru)
            kwargs = dict(
                dt=float(dt), acoustic_steps=acoustic_steps, rk_step=3,
                rt_diabatic_tendency=zeros_c, fzm=fz, fzp=fz,
                cf1=f(1.0), cf2=f(0.0), cf3=f(0.0), zb_cell=zb, zb3_cell=zb,
                boundary_mask_cell=np.full(n_cells, 99, dtype=np.int32),
            )
            if theta_start is None:
                # The state the model reads before the first step.
                theta_start = ((rtheta_p + background.rtheta_base) / rho_zz).astype(np.float64)
                u_start = (
                    (f(2.0) * ru) / (rho_zz[:, coe[:, 0]] + rho_zz[:, coe[:, 1]])
                ).astype(np.float64)
            if compensated:
                out, residual = recover_large_step_variables_compensated(
                    mesh, _state(rtheta_pp, ru_p), background, residual,
                    final_stage=True, **kwargs,
                )
            else:
                from .integration import recover_large_step_variables

                out = recover_large_step_variables(
                    mesh, _state(rtheta_pp, ru_p), background, **kwargs
                )
            rtheta_p = out.rtheta_p
            ru = out.ru
        assert out is not None
        d_theta = np.asarray(out.theta_m, dtype=np.float64) - theta_start
        d_u = np.asarray(out.u, dtype=np.float64) - u_start
        intended_theta = theta_tendency * n_steps * dt
        intended_u = u_tendency * n_steps * dt
        label = "float64" if dtype == np.float64 else "float32"
        report[label] = {
            "theta_lost_fraction": float(1.0 - np.mean(d_theta) / intended_theta),
            "theta_rms_relative_error": float(
                np.sqrt(np.mean((d_theta - intended_theta) ** 2)) / intended_theta
            ),
            "u_lost_fraction": float(1.0 - np.mean(d_u) / intended_u),
            "u_rms_relative_error": float(
                np.sqrt(np.mean((d_u - intended_u) ** 2)) / intended_u
            ),
        }
    return report


def main(argv: list[str] | None = None) -> int:
    """``python -m woof.hex.precision_probe``: print both probes as JSON."""

    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--spacings", type=float, nargs="+", default=[50.0, 100.0, 250.0, 711.0, 3000.0])
    parser.add_argument("--dts", type=float, nargs="+", default=[5.0, 0.5, 0.25])
    parser.add_argument("--tendency", type=float, default=2.0e-5,
                        help="theta (K/s) and momentum (m/s2) tendency held constant")
    parser.add_argument("--seconds", type=float, default=60.0)
    args = parser.parse_args(argv)
    report = {
        "geometry": [deformation_geometry_error(s) for s in args.spacings],
        "increment_loss": [
            large_step_increment_loss(
                dt, simulated_seconds=args.seconds, theta_tendency=args.tendency,
                u_tendency=args.tendency, compensated=comp,
            )
            for dt in args.dts
            for comp in (False, True)
        ],
    }
    print(json.dumps(report, indent=2))
    return 0


__all__ = [
    "EARTH_RADIUS_M",
    "SyntheticPatchMesh",
    "analytic_deformation_weights",
    "deformation_geometry_error",
    "large_step_increment_loss",
    "synthetic_hex_patch",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
