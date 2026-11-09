"""The varying-dt Kain-Fritsch driver case, both sides.

One frozen three-column state (a moist, conditionally unstable tropical
sounding with a 10 m/s wind, so KF's convective time scale is its 1800 s
floor and holds expire inside the run), stepped through an adaptive-style
``dt`` sequence: growth by 5% a step to a 96 s ceiling, one CFL cut, regrowth.
No step lands on a multiple of the one before it, which is the adaptive
clock's normal state.

* :func:`write_case` writes the WRF input for ``run_kf_adaptive_oracle``
  from the very float32 words woof's driver hands KF (:func:`case_words`:
  ``_prepare_atmosphere`` of the state and ``state.w``).
* :func:`run_woof` steps woof's real ``PhysicsDriver`` (production
  ``KainFritsch`` adapter, ``_run_cumulus``, ``_advance_cumulus_clock``) on
  the same sequence and returns what :func:`read_oracle` returns.

The scheme is called on WRF's own run_param steps (the oracle reports them),
handed to the driver as the adaptive clock hands it ``cumulus_due_override``.

    python -m tools.kf_adaptive_wrf471_oracle.case CASE.bin
"""

from __future__ import annotations

import dataclasses
import struct
import sys

import numpy as np

NZ = 30
CUDT_MINUTES = 5.0


def dt_sequence() -> tuple[float, ...]:
    """Quarter-second steps, so every partial sum is exact in float32."""
    out, dt = [], 30.0
    for n in range(110):
        if n == 45:
            dt = 41.0                       # a CFL cut, then regrowth
        out.append(dt)
        dt = min(96.0, np.floor(dt * 1.05 * 4.0 + 0.5) / 4.0)
    return tuple(out)


def build_case():
    """The RunConfig and the frozen device state (needs CuPy)."""
    from woof.config import RunConfig
    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.core.moist import init_moist_balanced

    cfg = RunConfig(
        nx=3, ny=1, nz=NZ, dx=12000.0, dy=12000.0, ztop=16000.0,
        dt=30.0, run_seconds=0.0, moist=True, mp_physics=10, cu_physics=1,
        cudt_minutes=CUDT_MINUTES, use_adaptive_time_step=True)
    coord = make_vertical_coord(cfg.nz)

    def theta(z):
        z = np.asarray(z)
        return np.where(z < 1000.0, 301.0, np.where(
            z < 13000.0, 301.0 + 0.0028 * (z - 1000.0),
            334.6 + 0.02 * (z - 13000.0)))

    def qv(z):
        z = np.asarray(z)
        return np.where(z < 1000.0, 0.019,
                        0.019 * np.exp(-(z - 1000.0) / 2200.0))

    base = make_base_state(coord, theta, p_surf=cfg.p_surf, ztop=cfg.ztop)
    state = init_moist_balanced(cfg, coord, base, qv)
    state.u[...] = 10.0
    state.w[1:12, 0, 0] = 4.0
    state.w[1:12, 0, 2] = 2.0
    return cfg, state


#: The KF_eta_CPS inputs the case carries, as woof's atmosphere keys.
FIELDS = ("u", "v", "temperature", "qv", "pressure", "dz", "rho", "exner")


def case_words(state) -> dict:
    """The float32 words woof's driver hands KF: ``_prepare_atmosphere`` of
    the state and ``state.w``, each (levels, columns)."""
    import cupy as cp

    from woof.core.physics import _prepare_atmosphere

    atmosphere = _prepare_atmosphere(state)
    words = {name: cp.asnumpy(atmosphere[name]).astype("<f4").reshape(
        atmosphere[name].shape[0], -1) for name in FIELDS}
    words["w"] = cp.asnumpy(state.w).astype("<f4").reshape(
        state.w.shape[0], -1)
    return words


def write_case(path, cfg, words, dts) -> bytes:
    """The ``run_kf_adaptive_oracle`` input for ``words`` and ``dts``."""
    from woof.core.kf import (kf_has_rain_for_microphysics,
                              kf_phase_mode_for_microphysics, KFPhaseMode)

    mode = kf_phase_mode_for_microphysics(cfg.mp_physics)
    if mode != KFPhaseMode.SEPARATE_ICE_SNOW:
        raise ValueError("the case is written for the separate ice/snow branch")
    nz, ncol = words["temperature"].shape
    stepcu = max(int(np.floor(cfg.cudt_minutes * 60.0 / dts[0] + 0.5)), 1)
    head = struct.pack("<8s9i", b"KFADT1  ", ncol, nz, len(dts), 0,
                       int(kf_has_rain_for_microphysics(cfg.mp_physics)),
                       1, 1, int(cfg.kfeta_trigger), stepcu)
    head += struct.pack("<2f", cfg.dx, cfg.cudt_minutes)
    head += np.asarray(dts, "<f4").tobytes()
    body = b"".join(words[name][:, c].tobytes()
                    for c in range(ncol) for name in FIELDS + ("w",))
    data = head + body
    with open(path, "wb") as handle:
        handle.write(data)
    return data


def read_case(path):
    """(dts, words) back from a case file."""
    raw = memoryview(open(path, "rb").read())
    magic, ncol, nz, nsteps = struct.unpack_from("<8s3i", raw, 0)
    if magic != b"KFADT1  ":
        raise ValueError(f"{path}: not a KFADT1 file")
    at = 8 + 9 * 4 + 2 * 4
    dts = tuple(float(x) for x in np.frombuffer(raw, "<f4", nsteps, at))
    at += 4 * nsteps
    words = {name: np.zeros((nz + (name == "w"), ncol), "<f4")
             for name in FIELDS + ("w",)}
    for c in range(ncol):
        for name in FIELDS + ("w",):
            count = words[name].shape[0]
            words[name][:, c] = np.frombuffer(raw, "<f4", count, at)
            at += 4 * count
    if at != len(raw):
        raise ValueError(f"{path}: {len(raw) - at} trailing bytes")
    return dts, words


def read_oracle(path) -> dict:
    raw = memoryview(open(path, "rb").read())
    magic, ncol, nz, nsteps = struct.unpack_from("<8s3i", raw, 0)
    if magic != b"KFADO1  ":
        raise ValueError(f"{path}: not a KFADO1 file")
    at = 20
    out = {"run_param": np.zeros(nsteps, bool), "fatal": np.zeros(nsteps, bool),
           "rth_rk": np.zeros((nsteps, ncol, nz), np.float32)}
    for name in ("nca", "rainc", "pratec", "raincv"):
        out[name] = np.zeros((nsteps, ncol), np.float32)
    for name in ("rth", "rqv"):
        out[name] = np.zeros((nsteps, ncol, nz), np.float32)

    def take(count):
        nonlocal at
        values = np.frombuffer(raw, "<f4", count, at).copy()
        at += 4 * count
        return values

    for n in range(nsteps):
        run, fatal = struct.unpack_from("<2i", raw, at)
        at += 8
        out["run_param"][n], out["fatal"][n] = bool(run), bool(fatal)
        for c in range(ncol):
            out["rth_rk"][n, c] = take(nz)
        for c in range(ncol):
            (out["nca"][n, c], out["rainc"][n, c], out["pratec"][n, c],
             out["raincv"][n, c]) = take(4)
            out["rth"][n, c] = take(nz)
            out["rqv"][n, c] = take(nz)
    if at != len(raw):
        raise ValueError(f"{path}: {len(raw) - at} trailing bytes")
    return out


def run_woof(cfg, state, words, dts, run_param) -> dict:
    """woof's driver on the same steps; the same keys as :func:`read_oracle`.

    The state is pinned to the case's words through the driver's own
    ``prepared_physics_atmosphere`` seam (and ``state.w``), so both sides
    read the same inputs on any backend.
    """
    import cupy as cp

    from woof.core.physics import _prepare_atmosphere, initialize_physics

    atmosphere = dict(_prepare_atmosphere(state))
    shape = tuple(int(n) for n in atmosphere["temperature"].shape)
    for name in FIELDS:
        atmosphere[name] = cp.ascontiguousarray(
            cp.asarray(words[name]).reshape(shape), dtype=cp.float32)
    state.w[...] = cp.asarray(words["w"]).reshape(state.w.shape)
    state.prepared_physics_atmosphere = lambda: dict(atmosphere)
    driver = initialize_physics(state, cfg)
    nsteps, ncol, nz = len(dts), int(cfg.nx * cfg.ny), int(cfg.nz)
    out = {"run_param": np.zeros(nsteps, bool),
           "rth_rk": np.zeros((nsteps, ncol, nz), np.float32)}
    for name in ("nca", "rainc", "pratec", "raincv"):
        out[name] = np.zeros((nsteps, ncol), np.float32)
    for name in ("rth", "rqv"):
        out[name] = np.zeros((nsteps, ncol, nz), np.float32)
    now = 0.0
    for n, dt in enumerate(dts):
        step_cfg = dataclasses.replace(cfg, dt=float(dt))
        state.elapsed_seconds = now
        driver.cumulus_due_override = bool(run_param[n])
        before = driver.call_counts["cumulus"]
        held = driver.compute(state, step_cfg)
        out["run_param"][n] = driver.call_counts["cumulus"] != before
        weight = (state.c1h[:, None, None] * state.total_mu()[None]
                  + state.c2h[:, None, None])
        out["rth_rk"][n] = cp.asnumpy(held.rtheta / weight).reshape(nz, ncol).T
        driver.finish_history_mass(state, step_cfg)
        out["nca"][n] = cp.asnumpy(driver.cu_nca).ravel()
        out["rainc"][n] = cp.asnumpy(driver.rainc).ravel()
        out["pratec"][n] = cp.asnumpy(driver.cu_pratec).ravel()
        out["raincv"][n] = cp.asnumpy(driver.cu_raincv).ravel()
        out["rth"][n] = cp.asnumpy(driver.cu_rates["rthcuten"]).reshape(nz, ncol).T
        out["rqv"][n] = cp.asnumpy(driver.cu_rates["rqvcuten"]).reshape(nz, ncol).T
        now += float(dt)
    return out


if __name__ == "__main__":
    cfg, state = build_case()
    write_case(sys.argv[1], cfg, case_words(state), dt_sequence())
