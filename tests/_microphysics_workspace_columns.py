"""Shared fixtures for the WSM6 / WDM6 column-workspace gates.

``tests/test_wsm6_workspace.py`` and ``tests/test_wdm6_workspace.py`` both
need (1) physically plausible, column-varying soundings with every
hydrometeor present somewhere, (2) the PRE-CUT kernels -- the per-thread
local-array sources the workspace replaced -- compiled the way the old
launchers compiled them, and (3) a raw launch of the shipped kernels against
a caller-owned workspace.  numpy only at import; cupy is imported inside the
functions that need a device.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

#: The last commit whose ``wsm6.cu`` / ``wdm6.cu`` kept their column arrays
#: per thread (``levels-256`` = jacobbieker/woof main on 2026-10-12).  The
#: bit-identity gates compile those sources beside the shipped ones.
PRE_CUT_COMMIT = "7950e4171af94587f25ec3da2e279acfea0bf51e"

SURFACE = ("rainnc", "rainncv", "snownc", "snowncv", "graupelnc",
           "graupelncv", "sr")


def pre_cut_source(module: str) -> str | None:
    """``kernels/<module>.cu`` at :data:`PRE_CUT_COMMIT`, or None if this
    checkout cannot reach that commit (a shallow clone, an export)."""
    try:
        done = subprocess.run(
            ["git", "-C", str(ROOT), "show",
             f"{PRE_CUT_COMMIT}:woof/core/kernels/{module}.cu"],
            capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout if done.returncode == 0 and done.stdout else None


def pre_cut_kernel(module: str, nz: int, text: str):
    """The pre-cut kernel exactly as its launcher compiled it for ``nz``:
    the loader's preamble, ``#define <M>_KMAX 80`` above 64 levels (the 64
    tier was the source's own default), then the unit with the active
    parameter set's literal edits -- the same assembly
    :func:`woof.core.kernels.module_source_int_defines` performs."""
    import cupy as cp

    from woof.core.kernels import _unit_text, module_source
    from woof.physics_params import edit_kernel_source

    full = module_source(module)
    unit = _unit_text(module)
    assert full.endswith(unit)
    preamble = full[: len(full) - len(unit)]
    tier = 64 if nz <= 64 else 80
    define = "" if tier == 64 else f"#define {module.upper()}_KMAX {tier}\n"
    src = preamble + define + edit_kernel_source(module, text)
    return cp.RawModule(code=src, options=("-std=c++17",)).get_function(
        f"{module}_column")


def _qsat(t, p):
    ttp = 273.16
    xa = -(1846.4 - 4190.0) / 461.6
    xb = xa + 2.5e6 / (461.6 * ttp)
    tr = ttp / t
    es = np.minimum(610.78 * tr ** xa * np.exp(xb * (1.0 - tr)), 0.99 * p)
    return (287.0 / 461.6) * es / (p - es)


def soundings(nz: int, ncol: int, *, seed: int = 0, ztop: float | None = None,
              number_moments: bool = False) -> dict[str, np.ndarray]:
    """``ncol`` different (nz, 1, ncol) soundings, float32, kernel layout.

    Stretched levels (thin at the ground), a tropospheric lapse rate to a
    column-varying tropopause and a weakly warming stratosphere, relative
    humidity 40-106 % below the tropopause and dry above, and cloud water,
    ice, rain, snow and graupel each present in about 60 % of the levels
    where their temperature range allows -- so neighbouring lanes take
    different branches.  ``ztop`` defaults to 20 km at nz <= 80 and to
    51 km (about 1 hPa with the 7.4 km scale height) above that.
    """
    rng = np.random.default_rng(seed)
    if ztop is None:
        ztop = 20000.0 if nz <= 80 else 51000.0
    s = np.linspace(0.0, 1.0, nz + 1)
    zw = (ztop * (0.4 * s + 0.6 * s * s))[:, None] * (
        1.0 + 0.02 * rng.standard_normal(ncol))[None, :]
    zw[0] = 0.0
    zw = np.maximum.accumulate(zw, axis=0)
    zw[1:] = np.maximum(zw[1:], zw[:-1] + 5.0)
    dz = np.diff(zw, axis=0)
    z = 0.5 * (zw[1:] + zw[:-1])
    t0 = rng.uniform(268.0, 305.0, ncol)
    trop = rng.uniform(10000.0, 16000.0, ncol)
    t = np.where(z < trop, t0 - 0.0065 * z,
                 t0 - 0.0065 * trop + 0.001 * (z - trop))
    t = np.maximum(t, 190.0) + 0.5 * rng.standard_normal(t.shape)
    p = 100000.0 * np.exp(-z / 7400.0)
    rho = p / (287.0 * t)
    pii = (p / 1.0e5) ** (287.0 / 1004.5)
    qv = rng.uniform(0.4, 1.06, t.shape) * _qsat(t, p)
    qv = np.where(z > trop, np.minimum(qv, 5.0e-6), qv)

    def layer(scale, where):
        on = (rng.random(t.shape) < 0.6) & where
        return np.where(on, scale * rng.random(t.shape) ** 2, 0.0)

    fields = {
        "theta": t / pii, "qv": qv,
        "qc": layer(1.5e-3, (t > 238.0) & (z < trop)),
        "qi": layer(2.0e-4, (t < 268.0) & (z < trop + 2000.0)),
        "qr": layer(3.0e-3, (t > 255.0) & (z < 8000.0)),
        "qs": layer(1.5e-3, (t < 276.0) & (z < trop)),
        "qg": layer(3.0e-3, (t < 280.0) & (z < trop)),
        "rho": rho, "pii": pii, "pressure": p, "dz": dz,
        "effc": np.zeros_like(t), "effi": np.zeros_like(t),
        "effs": np.zeros_like(t),
    }
    if number_moments:
        fields["nn"] = 10.0 ** rng.uniform(8.0, 10.3, t.shape)
        fields["nc"] = np.where(fields["qc"] > 0.0,
                                10.0 ** rng.uniform(7.0, 9.0, t.shape), 0.0)
        nr = np.where(fields["qr"] > 0.0,
                      10.0 ** rng.uniform(2.0, 5.0, t.shape), 0.0)
        # A sprinkle of numberless rain, the regime WDM6's A144 guard owns.
        nr[rng.random(t.shape) < 0.03] = 0.0
        fields["nr"] = nr
    out = {name: np.ascontiguousarray(
        np.asarray(value, np.float64).reshape(nz, 1, ncol), dtype=np.float32)
        for name, value in fields.items()}
    if number_moments:
        out["xland"] = np.where(rng.random((1, ncol)) < 0.4, 2.0,
                                1.0).astype(np.float32)
    return out


def column_water(fields) -> np.ndarray:
    """Column-integrated water substance per unit area (kg m-2), float64."""
    total = sum(np.asarray(fields[name], np.float64)
                for name in ("qv", "qc", "qr", "qi", "qs", "qg"))
    return (total * np.asarray(fields["rho"], np.float64)
            * np.asarray(fields["dz"], np.float64)).sum(axis=0)


def as_bits(arrays) -> dict[str, np.ndarray]:
    return {name: np.ascontiguousarray(value).view(np.uint32)
            for name, value in arrays.items()}


def differing_words(a, b) -> dict[str, int]:
    """Per-field count of 32-bit words that differ, compared as bit
    patterns (never as tolerances)."""
    ba, bb = as_bits(a), as_bits(b)
    assert set(ba) == set(bb)
    return {name: int((ba[name] != bb[name]).sum()) for name in ba
            if int((ba[name] != bb[name]).sum())}
