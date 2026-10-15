"""Analysis for the woof/Breeze dry-CBL comparison.

Reads what ``run_woof.py`` and ``run_breeze.jl`` write (``STEM.json`` plus
``STEM.npz`` or ``STEM_profiles.json``) and produces:

* the throughput table: median seconds/step, Mpts*steps/s (grid points
  times steps per wall second / 1e6) and simulated seconds per wall second,
  per model variant and size, the median taken over every timed rep of every
  interleaved repeat (``timing`` JSONs);
* the profile comparison: mean theta, resolved <w'w'>, total <w'theta'>/Q and
  wind speed averaged over a trailing window, and the time evolution of the
  boundary-layer depth z_i, mixed-layer theta and peak <w'w'>, against the
  zero-order-jump bulk model (theory) and the FastEddy tutorial figure values
  (an eyeball reading; see ``case.FASTEDDY_FIGURE_4H``);
* ``summary.json``, ``summary.md`` and PNG figures in ``--out``.

Diagnostics are computed identically for both models from slab means:

* ``zi_grad``: the face height of the largest mean d(theta)/dz above the
  surface layer (closure-independent: works for Breeze's implicit-LES runs,
  which carry no explicit SGS flux);
* ``zi_flux``: the centre height of the minimum TOTAL heat flux (resolved +
  explicit SGS), the em_les / FastEddy definition;
* ``theta_ml``: mean theta over 0.2 z_i .. 0.8 z_i (z_i = zi_grad);
* ``ww_max`` / ``ww_max_z``: peak resolved vertical-velocity variance.

Nothing is gated: differences are reported, not judged.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tools.compare.breeze_cbl import case  # noqa: E402

#: Matched fields every profile run must share (``check_matched``).
MATCHED_FIELDS = ("dt", "heat_flux", "f", "ug", "vg", "p_surf", "float_type")

#: Categorical slots (dataviz reference palette, light mode, fixed order).
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
          "#008300", "#6250d6", "#e34948")
LINESTYLES = ("-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1)),
              (0, (1, 1)), (0, (5, 3)))
REFERENCE_INK = "#52514e"
#: Canonical series order: a variant keeps its colour and line style in
#: every figure, whatever subset of variants a report contains.
CANONICAL_LABELS = ("woof_km3", "woof_km2", "breeze_anelastic_weno5_ci",
                    "breeze_se_weno5_ci", "breeze_se_matched",
                    "breeze_se_weno5_smag", "breeze_anelastic_matched")


def series_style(label: str, labels: list[str]) -> dict:
    """Colour/line style fixed by the label's canonical slot."""
    order = list(CANONICAL_LABELS) + sorted(
        lab for lab in set(labels) if lab not in CANONICAL_LABELS)
    i = order.index(label) if label in order else len(order)
    return {"color": SERIES[i % len(SERIES)],
            "linestyle": LINESTYLES[i % len(LINESTYLES)]}


# ------------------------------------------------------------------ loading

def load_run(stem: str | Path) -> dict:
    """One simulate run: ``{"meta": dict, <profile arrays>...}``."""
    stem = str(stem)
    for suffix in (".json",):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    meta = json.loads(Path(stem + ".json").read_text(encoding="utf-8"))
    if Path(stem + ".npz").exists():
        with np.load(stem + ".npz") as data:
            arrays = {k: np.asarray(data[k], dtype=float) for k in data.files}
    else:
        raw = json.loads(Path(stem + "_profiles.json").read_text(
            encoding="utf-8"))
        arrays = {k: np.asarray(v, dtype=float) for k, v in raw.items()}
    nt = arrays["theta"].shape[0]
    if arrays["z_f"].ndim == 1:
        arrays["z_f"] = np.broadcast_to(arrays["z_f"], (nt,) + arrays[
            "z_f"].shape).copy()
    return {"meta": meta, **arrays}


def check_matched(metas: list[dict]) -> list[str]:
    """Return human-readable mismatches between runs meant to be matched.

    Size-dependent fields (grid, drag coefficient) are compared only among
    runs of the same size; bulk-drag runs must carry the same Cd.
    """
    problems = []
    if not metas:
        return problems
    ref = metas[0]
    for m in metas[1:]:
        for key in MATCHED_FIELDS:
            a, b = ref.get(key), m.get(key)
            if isinstance(a, float) or isinstance(b, float):
                same = (a is not None and b is not None
                        and math.isclose(float(a), float(b), rel_tol=1e-6))
            else:
                same = a == b
            if not same:
                problems.append(f"{m.get('label')}: {key}={b!r} vs "
                                f"{ref.get('label')}: {a!r}")
    by_size = defaultdict(list)
    for m in metas:
        by_size[(m["nx"], m["ny"], m["nz"])].append(m)
    for size, group in by_size.items():
        expected = case.drag_coefficient(size[2])
        for m in group:
            if m.get("drag") == "bulk" and not math.isclose(
                    float(m["drag_coefficient"]), expected, rel_tol=1e-6):
                problems.append(f"{m.get('label')}: Cd={m['drag_coefficient']}"
                                f" != case.drag_coefficient={expected}")
    return problems


# -------------------------------------------------------------- diagnostics

def centres(z_f: np.ndarray) -> np.ndarray:
    return 0.5 * (z_f[..., 1:] + z_f[..., :-1])


def zi_gradient(theta: np.ndarray, z_c: np.ndarray,
                z_min: float = 100.0) -> float:
    """Face height of max d(theta)/dz above ``z_min``."""
    dth = np.diff(theta) / np.diff(z_c)
    z_face = 0.5 * (z_c[1:] + z_c[:-1])
    ok = z_face > z_min
    if not ok.any():
        return float("nan")
    k = int(np.argmax(np.where(ok, dth, -np.inf)))
    return float(z_face[k])


def total_heat_flux(wth_res: np.ndarray, wth_sgs: np.ndarray) -> np.ndarray:
    """Resolved (centres) + SGS (faces, averaged to centres)."""
    return wth_res + 0.5 * (wth_sgs[..., 1:] + wth_sgs[..., :-1])


def zi_flux(wth_total: np.ndarray, z_c: np.ndarray) -> float:
    """Centre height of the minimum total heat flux."""
    return float(z_c[int(np.argmin(wth_total))])


def theta_mixed_layer(theta: np.ndarray, z_c: np.ndarray, zi: float) -> float:
    sel = (z_c >= 0.2 * zi) & (z_c <= 0.8 * zi)
    if not sel.any():
        return float("nan")
    return float(theta[sel].mean())


def ww_peak(ww: np.ndarray, z_f: np.ndarray) -> tuple[float, float]:
    k = int(np.argmax(ww))
    return float(ww[k]), float(z_f[k])


def time_series(run: dict) -> dict:
    """Per-sample z_i, theta_ml, ww_max, ww_max_z."""
    out = defaultdict(list)
    for n in range(run["theta"].shape[0]):
        z_f = run["z_f"][n]
        z_c = centres(z_f)
        zg = zi_gradient(run["theta"][n], z_c)
        tot = total_heat_flux(run["wth_res"][n], run["wth_sgs"][n])
        wmax, wz = ww_peak(run["ww"][n], z_f)
        out["t"].append(float(run["t_seconds"][n]))
        out["zi_grad"].append(zg)
        out["zi_flux"].append(zi_flux(tot, z_c))
        out["theta_ml"].append(theta_mixed_layer(run["theta"][n], z_c, zg))
        out["ww_max"].append(wmax)
        out["ww_max_z"].append(wz)
    return {k: np.asarray(v) for k, v in out.items()}


def window_mean(run: dict, window_s: float) -> dict:
    """Trailing-window mean profiles (samples with t >= t_end - window)."""
    t = run["t_seconds"]
    sel = t >= t[-1] - window_s - 1e-6
    sel[0] = sel[0] and len(t) == 1        # never average in t = 0
    if not sel.any():
        sel[-1] = True
    z_f = run["z_f"][sel].mean(axis=0)
    tot = total_heat_flux(run["wth_res"][sel], run["wth_sgs"][sel])
    return {
        "z_f": z_f, "z_c": centres(z_f),
        "theta": run["theta"][sel].mean(axis=0),
        "ww": run["ww"][sel].mean(axis=0),
        "wth_total": tot.mean(axis=0),
        "speed": np.hypot(run["u"][sel], run["v"][sel]).mean(axis=0),
        "n_samples": int(sel.sum()),
        "t_start": float(t[sel][0]), "t_end": float(t[-1]),
    }


def summarise_profile_run(run: dict, window_s: float) -> dict:
    ts = time_series(run)
    tail = ts["t"] >= ts["t"][-1] - window_s - 1e-6
    tail &= ts["t"] > 0
    if not tail.any():
        tail = ts["t"] == ts["t"][-1]
    t_end = float(ts["t"][-1])
    wm = window_mean(run, window_s)
    m = run["meta"]
    return {
        "label": m["label"], "model": m["model"],
        "size": f"{m['nx']}x{m['ny']}x{m['nz']}",
        "t_end_hours": t_end / 3600.0,
        "window_minutes": window_s / 60.0,
        "zi_grad_m": float(np.nanmean(ts["zi_grad"][tail])),
        "zi_flux_m": float(np.nanmean(ts["zi_flux"][tail])),
        "theta_ml_K": float(np.nanmean(ts["theta_ml"][tail])),
        "ww_max_m2s2": float(np.nanmean(ts["ww_max"][tail])),
        "ww_max_z_m": float(np.nanmean(ts["ww_max_z"][tail])),
        "wth_min_over_q": float(wm["wth_total"].min() / case.HEAT_FLUX),
        "speed_ml_ms": theta_mixed_layer(wm["speed"], wm["z_c"],
                                         float(np.nanmean(
                                             ts["zi_grad"][tail]))),
        "bulk_model_zi_A0_m": float(case.bulk_model_zi(t_end, 0.0)),
        "bulk_model_zi_A02_m": float(case.bulk_model_zi(t_end, 0.2)),
        "finite": bool(m.get("finite", True)),
        "w_max": m.get("w_max"),
    }


# ----------------------------------------------------------------- timing

def load_timings(paths) -> list[dict]:
    out = []
    for p in paths:
        d = json.loads(Path(p).read_text(encoding="utf-8"))
        if d.get("mode") == "timing":
            out.append(d)
    return out


def throughput_table(timings: list[dict]) -> list[dict]:
    """One row per (label, size): medians over every timed rep."""
    groups = defaultdict(list)
    for d in timings:
        groups[(d["label"], d["nx"], d["ny"], d["nz"])].append(d)
    rows = []
    for (label, nx, ny, nz), runs in groups.items():
        per_step = [s for d in runs for s in d["seconds_per_step"]]
        med = statistics.median(per_step)
        npts = nx * ny * nz
        first = runs[0]
        rows.append({
            "label": label, "model": first["model"],
            "dynamics": first.get("dynamics"),
            "advection": first.get("advection"),
            "closure": first.get("closure"),
            "size": f"{nx}x{ny}x{nz}", "points": npts,
            "dt": first["dt"], "substeps": first.get("substeps"),
            "acoustic_steps_per_dt": first.get("acoustic_steps_per_dt"),
            "repeats": len(runs), "timed_reps": len(per_step),
            "median_seconds_per_step": med,
            "min_seconds_per_step": min(per_step),
            "max_seconds_per_step": max(per_step),
            "mpts_steps_per_second": npts / med / 1e6,
            "simulated_seconds_per_wall_second": first["dt"] / med,
            "all_finite": all(d.get("finite", False) for d in runs),
            # live device bytes: CuPy pool used_bytes (woof) or
            # CUDA.used_memory (Breeze); see each run's memory_counter
            "memory_gib_max": max(float(d.get("memory_gib")
                                        or d.get("pool_gib") or 0.0)
                                  for d in runs),
            "memory_counter": first.get("memory_counter"),
            "gpu": first.get("gpu"),
        })
    rows.sort(key=lambda r: (r["points"], r["model"] != "woof", r["label"]))
    # ratio to the woof km_opt=3 row of the same size, when present
    for r in rows:
        ref = next((x for x in rows if x["size"] == r["size"]
                    and x["model"] == "woof" and "km3" in x["label"]), None)
        r["speed_vs_woof_km3"] = (ref["median_seconds_per_step"]
                                  / r["median_seconds_per_step"]
                                  if ref else None)
    return rows


# ---------------------------------------------------------------- reports

def _fmt(x, nd=3):
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def markdown(rows: list[dict], prof: list[dict], problems: list[str]) -> str:
    lines = []
    if rows:
        lines += ["## Throughput", "",
                  "| run | size | s/step | Mpts*steps/s | sim s / wall s | "
                  "vs woof km3 | acoustic steps/dt | timed reps | finite |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            lines.append(
                f"| {r['label']} | {r['size']} | "
                f"{_fmt(r['median_seconds_per_step'], 4)} | "
                f"{_fmt(r['mpts_steps_per_second'], 1)} | "
                f"{_fmt(r['simulated_seconds_per_wall_second'], 2)} | "
                f"{_fmt(r['speed_vs_woof_km3'], 2)} | "
                f"{_fmt(r['acoustic_steps_per_dt'])} | {r['timed_reps']} | "
                f"{r['all_finite']} |")
        lines.append("")
    if prof:
        lines += ["## Boundary-layer statistics (trailing-window means)", "",
                  "| run | size | t (h) | z_i grad (m) | z_i flux (m) | "
                  "theta_ml (K) | max <w'w'> (m2/s2) | at z (m) | "
                  "min <w'th'>/Q | U_ml (m/s) | bulk z_i A=0 / 0.2 (m) |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for p in prof:
            lines.append(
                f"| {p['label']} | {p['size']} | {_fmt(p['t_end_hours'], 2)}"
                f" | {_fmt(p['zi_grad_m'], 0)} | {_fmt(p['zi_flux_m'], 0)} | "
                f"{_fmt(p['theta_ml_K'], 2)} | {_fmt(p['ww_max_m2s2'], 2)} | "
                f"{_fmt(p['ww_max_z_m'], 0)} | "
                f"{_fmt(p['wth_min_over_q'], 3)} | "
                f"{_fmt(p['speed_ml_ms'], 2)} | "
                f"{_fmt(p['bulk_model_zi_A0_m'], 0)} / "
                f"{_fmt(p['bulk_model_zi_A02_m'], 0)} |")
        fe = case.FASTEDDY_FIGURE_4H
        lines += ["", f"FastEddy tutorial figure at {fe['time_hours']:.0f} h "
                  f"(eyeball reading, 20 m grid): z_i(flux) ~ "
                  f"{fe['zi_heat_flux_min_m']:.0f} m, theta_ml ~ "
                  f"{fe['theta_mixed_layer_K']} K, max <w'w'> ~ "
                  f"{fe['ww_max_m2s2']} m2/s2 at ~{fe['ww_max_height_m']:.0f}"
                  f" m.", ""]
    if problems:
        lines += ["## Matching problems", ""] + [f"* {p}" for p in problems]
    return "\n".join(lines) + "\n"


def plot_profiles(runs: list[dict], window_s: float, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(15, 5.2), sharey=True)
    labels = [r["meta"]["label"] for r in runs]
    for run in runs:
        wm = window_mean(run, window_s)
        kw = dict(**series_style(run["meta"]["label"], labels), linewidth=2,
                  label=run["meta"]["label"])
        axes[0].plot(wm["theta"], wm["z_c"], **kw)
        axes[1].plot(wm["ww"], wm["z_f"], **kw)
        axes[2].plot(wm["wth_total"] / case.HEAT_FLUX, wm["z_c"], **kw)
        axes[3].plot(wm["speed"], wm["z_c"], **kw)
    t_end = max(float(r["t_seconds"][-1]) for r in runs)
    z = np.linspace(0, case.LZ, 301)
    axes[0].plot(case.theta_profile(z), z, color=REFERENCE_INK,
                 linewidth=1, linestyle=":", label="initial sounding")
    for ax, xl in zip(axes, ("mean theta (K)", "resolved <w'w'> (m2/s2)",
                             "total <w'theta'> / Q", "mean wind speed (m/s)")):
        ax.set_xlabel(xl)
        ax.grid(color="#e6e5e0", linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("z (m)")
    axes[0].set_ylim(0, case.LZ)
    axes[2].axvline(0.0, color=REFERENCE_INK, linewidth=0.8)
    axes[0].legend(fontsize=8, frameon=False, loc="upper left")
    fig.suptitle(f"FastEddy dry CBL: profiles averaged over the last "
                 f"{window_s / 60:.0f} min ending t = {t_end / 3600:.2f} h")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_evolution(runs: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    t_end = max(float(r["t_seconds"][-1]) for r in runs)
    labels = [r["meta"]["label"] for r in runs]
    for run in runs:
        ts = time_series(run)
        th = ts["t"] / 3600.0
        kw = dict(**series_style(run["meta"]["label"], labels), linewidth=2,
                  label=run["meta"]["label"])
        sel = ts["t"] > 0
        axes[0].plot(th[sel], ts["zi_flux"][sel], **kw)
        axes[1].plot(th[sel], ts["theta_ml"][sel], **kw)
        axes[2].plot(th[sel], ts["ww_max"][sel], **kw)
    tt = np.linspace(0, t_end, 200)
    axes[0].plot(tt / 3600, case.bulk_model_zi(tt, 0.2), color=REFERENCE_INK,
                 linewidth=1, linestyle="--", label="bulk model, A = 0.2")
    axes[0].plot(tt / 3600, case.bulk_model_zi(tt, 0.0), color=REFERENCE_INK,
                 linewidth=1, linestyle=":", label="encroachment, A = 0")
    for ax, yl in zip(axes, ("z_i (min total heat flux) (m)",
                             "mixed-layer theta (K)",
                             "peak resolved <w'w'> (m2/s2)")):
        ax.set_xlabel("time (h)")
        ax.set_ylabel(yl)
        ax.grid(color="#e6e5e0", linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_throughput(rows: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sizes = sorted({r["size"] for r in rows},
                   key=lambda s: int(np.prod([int(v) for v in s.split("x")])))
    labels = []
    for r in rows:
        if r["label"] not in labels:
            labels.append(r["label"])
    fig, axes = plt.subplots(1, len(sizes), figsize=(6 * len(sizes), 0.5
                                                       * len(labels) + 1.8),
                             squeeze=False)
    for ax, size in zip(axes[0], sizes):
        vals = {r["label"]: r["mpts_steps_per_second"] for r in rows
                if r["size"] == size}
        ys = [lab for lab in labels if lab in vals]
        ax.barh(range(len(ys)), [vals[y] for y in ys], height=0.6,
                color=[series_style(y, labels)["color"] for y in ys])
        for j, y in enumerate(ys):
            ax.text(vals[y], j, f" {vals[y]:.1f}", va="center", fontsize=8,
                    color="#0b0b0b")
        ax.set_yticks(range(len(ys)), ys, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Mpts * steps / s (higher is faster)")
        ax.set_title(size)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--timing", type=Path, nargs="*", default=[],
                    help="timing JSON files or directories of them")
    ap.add_argument("--profiles", nargs="*", default=[],
                    help="simulate-run stems (STEM.json + STEM.npz | "
                         "STEM_profiles.json)")
    ap.add_argument("--window-minutes", type=float, default=30.0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 when the runs are not matched (the report "
                         "is written and lists the problems either way)")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    timing_files = []
    for p in args.timing:
        timing_files += sorted(p.glob("*.json")) if p.is_dir() else [p]
    timings = load_timings(timing_files)
    rows = throughput_table(timings)
    runs = [load_run(s) for s in args.profiles]
    window_s = args.window_minutes * 60.0
    prof = [summarise_profile_run(r, window_s) for r in runs]
    problems = check_matched([r["meta"] for r in runs])
    problems += check_matched([{**d, "drag": "timing"} for d in timings])

    (args.out / "summary.json").write_text(json.dumps(
        {"throughput": rows, "profiles": prof, "matching_problems": problems},
        indent=1) + "\n", encoding="utf-8")
    (args.out / "summary.md").write_text(markdown(rows, prof, problems),
                                         encoding="utf-8")
    if rows:
        plot_throughput(rows, args.out / "throughput.png")
    if runs:
        plot_profiles(runs, window_s, args.out / "profiles.png")
        plot_evolution(runs, args.out / "evolution.png")
    print((args.out / "summary.md").read_text(encoding="utf-8"))
    if problems:
        print(f"WARNING: {len(problems)} matching problem(s); see "
              f"{args.out / 'summary.md'}", file=sys.stderr)
    return 1 if (problems and args.strict) else 0


if __name__ == "__main__":
    sys.exit(main())
