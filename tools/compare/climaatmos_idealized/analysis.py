"""Analysis for the woof vs ClimaAtmos.jl idealized benchmark (CPU only).

* The Schaer mountain-wave linear steady-state solution, ported from
  ClimaAtmos.jl ``src/topography/steady_state_solutions.jl``
  (``∂FΔU_∂Fh_approximation`` + ``steady_state_velocity(::ScharTopography)``)
  so both models are scored against the same analytic yardstick on their
  own grids.  The port is cross-checked against ClimaAtmos's own
  ``wapredicted`` diagnostic in the run summary.
* Density-current metrics (theta'_min, surface -1 K front position) and the
  Straka et al. (1993) reference values.
* Throughput table assembly from ``result.json`` files written by
  ``woof_driver.py`` and ``ca_driver.jl``.

Command line::

    python -m tools.compare.climaatmos_idealized.analysis summarize RUNS_DIR

reads ``RUNS_DIR/<run>/result.json`` (+ fields) for every run directory,
scores it, and prints/writes ``summary.json`` and ``summary.md``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

# ClimaParams defaults used by ClimaAtmos (grav, R_d = R/M_d, kappa_d = 2/7,
# MSLP, p_ref_theta).
GRAV = 9.81
R_D = 8.3144598 / 0.02897
CP_D = R_D / (2.0 / 7.0)
MSLP = 101325.0
P_REF_THETA = 1.0e5

# ClimaAtmos background for ConstantBuoyancyFrequencyProfile.
SCHAR_U = 10.0
SCHAR_N = 0.01
SCHAR_T_SFC = 288.0
SCHAR_T_MIN, SCHAR_T_MAX = 100.0, 500.0
# ScharTopography defaults (h_max, lambda, a) and the toml sponge.
SCHAR_H_MAX = 25.0
SCHAR_LAMBDA = 4000.0
SCHAR_A = 5000.0
SCHAR_ZD_RAYLEIGH = 13000.0
SCHAR_ALPHA_RAYLEIGH_W = 0.1

#: Straka et al. (1993), Table/Fig. for the converged (dx = 25 m) solution at
#: t = 900 s, as tabulated by later intercomparisons (e.g. Giraldo &
#: Restelli 2008, J. Comput. Phys. 227, Table 2): theta'_min and the surface
#: -1 K front position.
STRAKA_REF = {"theta_min": -9.77, "front_km": 15.54}

#: Density-current geometry shared by both models (x relative to the bubble
#: centre; ClimaAtmos places it at x = 25.6 km of a [0, 51.2] km domain).
DC_X_CENTER = 25600.0
DC_THETA_B = 300.0


# --------------------------------------------------------------------------
# Grids and base states
# --------------------------------------------------------------------------

def cell_centers(nx: int, dx: float) -> np.ndarray:
    """Domain-centred cell-centre x (woof's idealized convention)."""
    return (np.arange(nx) + 0.5) * dx - 0.5 * nx * dx


def schar_height(x, x_center: float = 0.0, h_max: float = SCHAR_H_MAX,
                 lam: float = SCHAR_LAMBDA, a: float = SCHAR_A):
    """ClimaAtmos ``ScharTopography``: h exp(-(x/a)^2) cos^2(pi x / lambda)."""
    xr = np.asarray(x, dtype=np.float64) - x_center
    return h_max * np.exp(-(xr / a) ** 2) * np.cos(np.pi * xr / lam) ** 2


def background_p_and_T(z):
    """ClimaAtmos ``background_p_and_T`` (constant N, isothermal cap)."""
    z = np.asarray(z, dtype=np.float64)
    beta = SCHAR_N ** 2 / GRAV
    a = GRAV / (CP_D * SCHAR_T_SFC * beta)
    p = MSLP * (1.0 - a + a * np.exp(-beta * z)) ** (CP_D / R_D)
    T = SCHAR_T_SFC * ((1.0 - a) * np.exp(beta * z) + a)
    T_iso = SCHAR_T_MIN if a > 1 else SCHAR_T_MAX
    z_iso = np.log((T_iso / SCHAR_T_SFC - a) / (1.0 - a)) / beta
    p_iso = MSLP * ((1.0 - a) / (1.0 - a * SCHAR_T_SFC / T_iso)) ** (
        CP_D / R_D)
    above = z > z_iso
    p = np.where(above, p_iso * np.exp(-GRAV / (R_D * T_iso) * (z - z_iso)),
                 p)
    T = np.where(above, T_iso, T)
    return p, T


def constant_n_theta(z):
    """theta(z) of the ClimaAtmos constant-N background (woof sounding)."""
    p, T = background_p_and_T(z)
    return T * (P_REF_THETA / p) ** (R_D / CP_D)


def isentropic_theta(z):
    """theta = 300 K (density-current base state)."""
    return np.full_like(np.asarray(z, dtype=np.float64), DC_THETA_B)


def eta_levels_for_heights(z_faces, theta_func, p_surf: float,
                           ztop: float) -> np.ndarray:
    """woof dry-hydrostatic eta levels whose flat-terrain heights are
    ``z_faces``.

    Integrates the Exner function of ``theta_func`` on a 1 m grid with
    woof's constants (as ``woof.core.grid.make_base_state`` does; woof then
    builds its own discrete hydrostatic heights from these eta levels, so
    they match the targets to within that integration) and maps each target height to
    eta = (p(z) - p_top)/(p_surf - p_top).
    """
    z_faces = np.asarray(z_faces, dtype=np.float64)
    if z_faces[0] != 0.0 or not np.all(np.diff(z_faces) > 0.0):
        raise ValueError("z_faces must start at 0 and increase strictly")
    from woof.core import constants as wc

    zf = np.append(np.arange(0.0, ztop, 1.0), ztop)
    zm = 0.5 * (zf[:-1] + zf[1:])
    pi0 = (p_surf / wc.P0) ** wc.RCP
    pi = np.concatenate(
        [[pi0], pi0 - np.cumsum(wc.G * np.diff(zf) / (wc.CP * theta_func(zm)))])
    p = wc.P0 * pi ** (1.0 / wc.RCP)
    p_at = np.interp(z_faces, zf, p)
    eta = (p_at - p[-1]) / (p_surf - p[-1])
    eta[0], eta[-1] = 1.0, 0.0
    return eta


# --------------------------------------------------------------------------
# Schaer linear steady state (port of ClimaAtmos steady_state_solutions.jl)
# --------------------------------------------------------------------------

def _dFdU_dFh(eta, k, z_top):
    """``∂FΔU_∂Fh_approximation`` for a 2-D (k_y = 0) mountain.

    ``eta`` (P,) terrain-following height, ``k`` (K,) positive wavenumbers;
    returns complex (dFu, dFw) of shape (P, K).
    """
    g, R, cp = GRAV, R_D, CP_D
    N, T_sfc, u = SCHAR_N, SCHAR_T_SFC, SCHAR_U
    beta = N ** 2 / g
    a = g / (cp * T_sfc * beta)
    gamma = cp / (cp - R)
    alpha = g / (gamma * R * T_sfc * beta)
    eta = np.asarray(eta, dtype=np.float64)[:, None]
    k = np.asarray(k, dtype=np.float64)[None, :]

    p, T = background_p_and_T(eta)
    rho_sfc = MSLP / (R * T_sfc)
    m_sfc = u ** 2 / (gamma * R * T_sfc)
    mu_sfc = 1.0 - m_sfc
    d_sfc = rho_sfc / mu_sfc

    rho = p / (R * T)
    m = u ** 2 / (gamma * R * T)
    mu = 1.0 - m
    nu = m / mu
    d = rho / mu
    dEta_x = -1j * k * (1.0 - eta / z_top)

    k_eta2 = (k ** 2 * (-mu + (N / (k * u)) ** 2 * (1.0 + nu * (1.0 - a)))
              - beta ** 2 / 4.0 * ((1.0 + alpha) ** 2
                                   + 2.0 / mu * alpha * (1.0 - a)
                                   + nu * (1.0 + 3.0 / mu) * (1.0 - a) ** 2))
    # In 2-D the forcing terms ∂Δf_∂Fh vanish identically (mu = 1 - m).
    pm = np.where(k_eta2 > 0.0, np.sign(k * u), 1.0)
    root = np.sqrt(k_eta2.astype(np.complex128))
    dFw = np.sqrt(d_sfc / d) * (1j * k * u) * np.exp(pm * 1j * root * eta)
    d2Fw = pm * 1j * root * dFw
    dFp = -(1j * k / k ** 2 * g * d) * ((1.0 - m) * dEta_x - m / u * dFw
                                        + u / g * d2Fw)
    dFu = -(1j / k * g * rho * dEta_x + dFp) / (rho * u)
    return dFu, dFw


def schar_steady_state(x, z, z_top: float, x_center: float = 0.0,
                       n_k: int = 2000, chunk: int = 1024):
    """Linear steady (u', w) at points (x, z) for the Schaer mountain.

    Same construction as ClimaAtmos: eta = (z - h)/(1 - h/z_top), inverse
    transform 2 * int_0^kmax Re(FΔU Fh e^{ik(x - xc)}) dk with the analytic
    transform of the centred Schaer profile and kmax where Fh < eps(Float32).
    Returns (du, w) arrays shaped like ``x``.
    """
    x = np.asarray(x, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    shape = np.broadcast(x, z).shape
    xf = np.broadcast_to(x, shape).ravel()
    zf = np.broadcast_to(z, shape).ravel()
    h = schar_height(xf, x_center)
    eta = (zf - h) / (1.0 - h / z_top)

    a, lam, hmax = SCHAR_A, SCHAR_LAMBDA, SCHAR_H_MAX
    k_peak = 2.0 * np.pi / lam
    k_max = k_peak + 2.0 * math.sqrt(-math.log(float(np.finfo(np.float32).eps))) / a
    # Midpoint rule on (0, k_max].
    dk = k_max / n_k
    k = (np.arange(n_k) + 0.5) * dk
    Fh = hmax * a / (8.0 * math.sqrt(math.pi)) * (
        np.exp(-a ** 2 / 4.0 * (k + k_peak) ** 2)
        + 2.0 * np.exp(-a ** 2 / 4.0 * k ** 2)
        + np.exp(-a ** 2 / 4.0 * (k - k_peak) ** 2))
    du = np.empty(xf.size)
    w = np.empty(xf.size)
    for s in range(0, xf.size, chunk):
        sl = slice(s, s + chunk)
        dFu, dFw = _dFdU_dFh(eta[sl], k, z_top)
        phase = np.exp(1j * k[None, :] * (xf[sl, None] - x_center)) * Fh
        du[sl] = 2.0 * dk * np.real(dFu * phase).sum(axis=1)
        w[sl] = 2.0 * dk * np.real(dFw * phase).sum(axis=1)
    return du.reshape(shape), w.reshape(shape)


def schar_error_metrics(w_model, w_ref, z, z_max: float = SCHAR_ZD_RAYLEIGH,
                        z_min: float = 0.0) -> dict:
    """Error of model w against the analytic w on points z_min <= z < z_max.

    ``rel_l2``: ||w - w_ref||_2 / ||w_ref||_2; ``corr``: Pearson correlation;
    ``max_abs_err``/``ref_max`` in m/s.
    """
    shape = np.shape(w_model)
    w_model = np.asarray(w_model, dtype=np.float64).ravel()
    w_ref = np.asarray(w_ref, dtype=np.float64).ravel()
    z = np.broadcast_to(np.asarray(z, dtype=np.float64), shape).ravel()
    m = (z >= z_min) & (z < z_max) & np.isfinite(w_model)
    if not m.any():
        raise ValueError("no points inside the scoring window")
    wm, wr = w_model[m], w_ref[m]
    err = wm - wr
    return {
        "rel_l2": float(np.linalg.norm(err) / np.linalg.norm(wr)),
        "corr": float(np.corrcoef(wm, wr)[0, 1]),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "max_abs_err": float(np.abs(err).max()),
        "ref_max": float(np.abs(wr).max()),
        "model_max": float(np.abs(wm).max()),
        "n_points": int(m.sum()),
    }


# --------------------------------------------------------------------------
# Density current
# --------------------------------------------------------------------------

def density_current_theta_prime(x, z, exner_b, ic: str = "straka"):
    """Initial theta' for the density current (x relative to the centre).

    ``straka``: T' = -15 K (cos(pi L) + 1)/2, L over radii 4 km x 2 km about
    z = 3 km, theta' = T'/exner_b (Straka et al. 1993).
    ``clima``: ClimaAtmos ``DryDensityCurrentProfile``: theta' = -15 K
    (cos(pi L) + 1)/2 about z = 2 km.
    """
    if ic == "straka":
        zc, scale = 3000.0, 1.0 / np.asarray(exner_b, dtype=np.float64)
    elif ic == "clima":
        zc, scale = 2000.0, 1.0
    else:
        raise ValueError(f"unknown ic {ic!r}")
    L = np.sqrt((np.asarray(x) / 4000.0) ** 2
                + ((np.asarray(z) - zc) / 2000.0) ** 2)
    return np.where(L <= 1.0, -15.0 * (np.cos(np.pi * L) + 1.0) / 2.0,
                    0.0) * scale


def front_position_km(x, theta_prime_sfc, threshold: float = -1.0) -> float:
    """x (km from the centre) of the rightmost surface ``threshold`` crossing.

    ``x`` need not be sorted or unique (spectral-element nodes duplicate
    element edges); duplicates are averaged.  Linear interpolation between
    neighbouring nodes; NaN if the surface never reaches the threshold.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    t = np.asarray(theta_prime_sfc, dtype=np.float64).ravel()
    xr = np.round(x, 3)
    xu, inv = np.unique(xr, return_inverse=True)
    tu = np.bincount(inv, weights=t) / np.bincount(inv)
    below = tu <= threshold
    idx = np.nonzero(below[:-1] & ~below[1:])[0]
    if idx.size == 0:
        return float("nan")
    i = int(idx[-1])
    t0, t1 = tu[i], tu[i + 1]
    frac = (threshold - t0) / (t1 - t0)
    return float(xu[i] + frac * (xu[i + 1] - xu[i])) / 1000.0


def density_current_metrics(x, z, theta, theta_b: float = DC_THETA_B,
                            sfc=None) -> dict:
    """theta'_min, front position (km, right half), and mirror symmetry
    from point-cloud fields (x relative to the bubble centre).

    ``sfc`` is a boolean mask (same shape as ``x``) selecting the lowest
    model level.  Without it the lowest level is taken as the points at the
    minimum ``z``, which is only right for a fixed, flat mesh (ClimaAtmos);
    woof's level heights move with the cold pool, so pass the level-0 mask.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    z = np.asarray(z, dtype=np.float64).ravel()
    thp = np.asarray(theta, dtype=np.float64).ravel() - theta_b
    if sfc is None:
        z_lowest = z.min()
        sfc = np.isclose(z, z_lowest, atol=1e-3 * max(1.0, abs(z_lowest)))
    else:
        sfc = np.asarray(sfc, dtype=bool).ravel()
        z_lowest = float(z[sfc].mean())
    front_r = front_position_km(x[sfc], thp[sfc])
    front_l = front_position_km(-x[sfc], thp[sfc])
    return {
        "theta_min": float(thp.min()),
        "front_km": front_r,
        "front_left_km": front_l,
        "front_asym_km": float(abs(front_r - front_l)),
        "z_lowest_m": float(z_lowest),
    }


def straka_errors(metrics: dict) -> dict:
    """Signed differences from the Straka et al. (1993) reference."""
    return {
        "theta_min_err": metrics["theta_min"] - STRAKA_REF["theta_min"],
        "front_err_km": metrics["front_km"] - STRAKA_REF["front_km"],
    }


# --------------------------------------------------------------------------
# Loading runs
# --------------------------------------------------------------------------

def load_ca_fields(run_dir: Path) -> dict:
    """Raw float64 arrays written by ca_driver.jl (``fields/*.f64``)."""
    fdir = Path(run_dir) / "fields"
    return {p.stem: np.fromfile(p, dtype="<f8") for p in fdir.glob("*.f64")}


def score_run(run_dir: Path) -> dict:
    """Score one run directory; returns the result.json plus metrics."""
    run_dir = Path(run_dir)
    res = json.loads((run_dir / "result.json").read_text())
    res["run"] = run_dir.name
    case = classify_case(res)
    res["case_kind"] = case
    if res.get("mode") != "accuracy":
        return res
    if res["model"] == "woof":
        f = np.load(run_dir / "fields.npz")
        if case == "schar":
            x2 = np.broadcast_to(f["x"][None, :], f["zw"].shape)
            _, w_ana = schar_steady_state(x2, f["zw"], z_top=res["ztop"])
            res["metrics"] = schar_error_metrics(f["w"][1:-1], w_ana[1:-1],
                                                 f["zw"][1:-1])
        else:
            zw = f["zw"]
            zc = 0.5 * (zw[:-1] + zw[1:])
            x2 = np.broadcast_to(f["x"][None, :], zc.shape)
            level0 = np.zeros(zc.shape, dtype=bool)
            level0[0] = True
            res["metrics"] = density_current_metrics(x2, zc, f["theta"],
                                                     sfc=level0)
    else:
        f = load_ca_fields(run_dir)
        if case == "schar":
            ztop = float(f["f_z"].max())
            inner = ca_interior_faces(f)
            # ScharTopography's x_center is a fixed 50 km (ClimaAtmos
            # src/topography/topography.jl), not the domain centre.
            _, w_ana = schar_steady_state(f["f_x"][inner], f["f_z"][inner],
                                          z_top=ztop, x_center=50e3)
            res["metrics"] = schar_error_metrics(f["f_wa"][inner], w_ana,
                                                 f["f_z"][inner])
            if "f_wapredicted" in f:
                # Cross-check the Python port against ClimaAtmos's own
                # predicted w and report ClimaAtmos's error vs its own.
                wp = f["f_wapredicted"][inner]
                res["port_vs_climaatmos_rel_l2"] = float(
                    np.linalg.norm(w_ana - wp) / np.linalg.norm(wp))
                res["metrics_vs_own_prediction"] = schar_error_metrics(
                    f["f_wa"][inner], wp, f["f_z"][inner])
        else:
            res["metrics"] = density_current_metrics(
                f["c_x"] - DC_X_CENTER, f["c_z"], f["c_thetaa"])
    if case.startswith("dc_straka"):
        res["metrics"].update(straka_errors(res["metrics"]))
    return res


def ca_interior_faces(f: dict) -> np.ndarray:
    """Mask of ClimaAtmos face nodes excluding the surface and top faces.

    ca_driver.jl flattens ``parent(field)`` column-major, so the level index
    runs fastest: each horizontal node contributes ``nlev + 1`` consecutive
    faces (``f_z.size / sfc_z.size`` of them), bottom to top.
    """
    ncol = f["sfc_z"].size
    nf = f["f_z"].size // ncol
    if nf * ncol != f["f_z"].size:
        raise ValueError("face count is not a multiple of the column count")
    zc = f["f_z"].reshape(ncol, nf)
    if not np.all(np.diff(zc, axis=1) > 0.0):
        raise ValueError("unexpected ClimaAtmos field layout (levels are "
                         "not the fastest-varying index)")
    mask = np.ones((ncol, nf), dtype=bool)
    mask[:, 0] = mask[:, -1] = False
    return mask.ravel()


def classify_case(res: dict) -> str:
    """``schar``, ``dc_straka`` or ``dc_clima`` for a result.json."""
    if res["model"] == "woof":
        return "schar" if res["case"] == "schar" else f"dc_{res['ic']}"
    cfg = Path(res["config_file"]).name
    if "schar" in cfg:
        return "schar"
    return "dc_straka" if res.get("straka_ic") else "dc_clima"


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------

def throughput_rows(results: list[dict]) -> list[dict]:
    """One row per run with the throughput columns."""
    rows = []
    for r in results:
        rows.append({
            "run": r["run"],
            "model": r["model"],
            "case": r["case_kind"],
            "mode": r["mode"],
            "points": r["unique_points"],
            "dt": r["dt"],
            "sec_per_step": r["sec_per_step"],
            "Mpoint_steps_per_s": r["point_steps_per_sec"] / 1e6,
            "sim_per_wall": r["sim_seconds_per_wall_second"],
            "nan": r.get("nan", False),
        })
    return rows


def _fmt(v, nd=3):
    if isinstance(v, float):
        if not math.isfinite(v):
            return "nan"
        return f"{v:.{nd}g}" if abs(v) < 1e4 else f"{v:.4g}"
    return str(v)


def markdown_tables(results: list[dict]) -> str:
    """Throughput and accuracy tables as Markdown."""
    out = ["| run | model | case | mode | points | dt (s) | s/step | "
           "Mpt-steps/s | sim s / wall s | NaN |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for row in throughput_rows(results):
        out.append("| " + " | ".join(_fmt(row[k]) for k in (
            "run", "model", "case", "mode", "points", "dt", "sec_per_step",
            "Mpoint_steps_per_s", "sim_per_wall", "nan")) + " |")
    acc = [r for r in results if "metrics" in r]
    if acc:
        out += ["", "| run | model | case | t (s) | metrics |",
                "|---|---|---|---|---|"]
        for r in acc:
            m = ", ".join(f"{k}={_fmt(v)}" for k, v in r["metrics"].items())
            if "port_vs_climaatmos_rel_l2" in r:
                m += (f", port_vs_CA_prediction="
                      f"{_fmt(r['port_vs_climaatmos_rel_l2'])}")
            out.append(f"| {r['run']} | {r['model']} | {r['case_kind']} | "
                       f"{_fmt(r.get('t_final', float('nan')))} | {m} |")
    return "\n".join(out) + "\n"


def summarize(runs_dir: Path) -> list[dict]:
    runs_dir = Path(runs_dir)
    results = []
    for p in sorted(runs_dir.glob("*/result.json")):
        try:
            results.append(score_run(p.parent))
        except Exception as exc:  # one bad run must not hide the others
            print(f"warning: could not score {p.parent.name}: {exc!r}")
    results = [r for r in results if "unique_points" in r]
    (runs_dir / "summary.json").write_text(
        json.dumps(results, indent=1, sort_keys=True, default=float) + "\n")
    md = markdown_tables(results)
    (runs_dir / "summary.md").write_text(md)
    print(md)
    return results


def ca_flat_column_faces(run_dir: Path) -> np.ndarray:
    """Face heights of the ClimaAtmos column with the lowest surface (the
    flat far field of the Schaer channel) from a ca_driver.jl field dump."""
    f = load_ca_fields(run_dir)
    x0 = f["sfc_x"][int(np.argmin(f["sfc_z"]))]
    col = np.isclose(f["f_x"], x0, rtol=0.0, atol=1e-3)
    z = np.unique(np.round(f["f_z"][col], 4))
    return z - z[0]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Score benchmark runs.")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("summarize")
    s.add_argument("runs_dir")
    zf = sub.add_parser("z-faces", help="write the ClimaAtmos face heights "
                        "of a Schaer run as JSON for woof_driver --z-faces")
    zf.add_argument("run_dir")
    zf.add_argument("out_json")
    args = p.parse_args(argv)
    if args.cmd == "summarize":
        summarize(Path(args.runs_dir))
    elif args.cmd == "z-faces":
        z = ca_flat_column_faces(Path(args.run_dir))
        Path(args.out_json).write_text(json.dumps({
            "source": "ClimaAtmos.jl plane_schar_mountain_float32_test.yml "
                      "face heights (flat column), via analysis z-faces",
            "z_faces": [round(float(v), 4) for v in z]}, indent=1) + "\n")
        print(f"{z.size} faces, dz_bottom={z[1]:.3f} m, top={z[-1]:.1f} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
