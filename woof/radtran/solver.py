"""Launch woof's discrete-ordinate solver (radtran_disort.cu) on the device.

One thread per problem (a column at one representative wavelength).  The
per-problem workspace (layer eigen-solutions and the banded boundary-value
matrix) lives in one device buffer, launched in chunks so it stays under
``budget_bytes``.  ``backend="host"`` runs the same source compiled as host
C++ (:mod:`woof.radtran._host`).
"""

from __future__ import annotations

import numpy as np

DELTAM = 1
SPHERICAL = 2
TWOSTREAM = 4
TMS = 8

MAX_STREAMS = 32
MAX_UMU = 32
_TPB = 64
#: Mode-parallel radiances up to this many threads.  Measured on Apple M-series
#: Metal (soft-float binary64), 16 streams, 30 layers, TMS: one thread per
#: problem takes ~7 s for 64-1024 problems (latency bound); modes over
#: threads take 1.4 s (64), 2.4 s (128), 3.7 s (256), 5.5 s (512), 11.6 s (1024).
MODE_THREADS = 8192


def work_size(nlyr: int, nstr: int) -> int:
    """Doubles of workspace per problem (rt_work_size in the kernel)."""
    N = nstr // 2
    n = 2 * N * nlyr
    ldab = 9 * N - 2
    return (6 * nlyr + 2 * (nlyr + 1) + nlyr * nstr + nlyr * 2 * N * N + nlyr * N
            + nlyr * 2 * N + ldab * n + 2 * n + 2 * N)


def _check(nstr, flags, umu, phi):
    if not (flags & TWOSTREAM):
        if nstr % 2 or nstr < 2 or nstr > MAX_STREAMS:
            raise ValueError(f"number of streams must be even and 2..{MAX_STREAMS}; got {nstr}")
    if len(umu) > MAX_UMU or len(phi) > MAX_UMU:
        raise ValueError(f"at most {MAX_UMU} user polar angles and azimuths")
    if any(u == 0.0 for u in umu):
        raise ValueError("umu must not be zero")


def solve(*, nstr, flags, dtau, ssa, pmom, umu0, fbeam, albedo, zlev=None, radius=6370.0,
          outlev, umu=(), phi=(), ptrue=None, backend="gpu", budget_bytes=1 << 29):
    """Fluxes (nprob, nout, 6) and radiances (nprob, nout, numu, nphi).

    Flux columns: edir, edn, eup, uavgdir, uavgdn, uavgup (uvspec's default
    output order).  ``edir`` is the unscaled direct beam and ``edn`` the
    diffuse field with the delta-M forward peak returned to it.
    """
    umu = [float(u) for u in umu]
    phi = [float(p) for p in phi]
    _check(nstr, flags, umu, phi)
    if backend == "host":
        from woof.radtran import _host
        return _host.solve(nstr=nstr, flags=flags, dtau=dtau, ssa=ssa, pmom=pmom, umu0=umu0,
                           fbeam=fbeam, albedo=albedo, zlev=zlev, radius=radius, outlev=outlev,
                           umu=umu, phi=phi, ptrue=ptrue)
    import cupy as cp
    from woof.core.kernels import get_kernel
    dtau = np.ascontiguousarray(dtau, dtype=np.float64)
    nprob, nlyr = dtau.shape
    pmom = np.ascontiguousarray(pmom, dtype=np.float64)
    nmom = pmom.shape[-1] - 1
    bc = lambda a: np.ascontiguousarray(np.broadcast_to(np.asarray(a, dtype=np.float64), (nprob,)))
    zl = (np.zeros((nprob, nlyr + 1)) if zlev is None
          else np.ascontiguousarray(np.broadcast_to(zlev, (nprob, nlyr + 1)), dtype=np.float64))
    outlev = np.ascontiguousarray(outlev, dtype=np.int32)
    nout, numu, nphi = outlev.size, len(umu), len(phi)
    n_str = 2 if flags & TWOSTREAM else nstr
    wstride = work_size(nlyr, n_str)
    chunk = max(1, min(nprob, budget_bytes // (8 * wstride)))
    d = {k: cp.asarray(v) for k, v in dict(
        dtau=dtau, ssa=np.ascontiguousarray(ssa, dtype=np.float64), pmom=pmom,
        umu0=bc(umu0), fbeam=bc(fbeam), albedo=bc(albedo), zlev=zl).items()}
    d_out = cp.asarray(outlev)
    d_umu = cp.asarray(np.asarray(umu if numu else [1.0], dtype=np.float64))
    d_phi = cp.asarray(np.asarray(phi if nphi else [0.0], dtype=np.float64))
    d_pt = cp.asarray(np.zeros(1) if ptrue is None else np.ascontiguousarray(ptrue, dtype=np.float64))
    flux = cp.zeros((nprob, nout, 6), dtype=cp.float64)
    rad = cp.zeros((nprob, nout, max(numu, 1), max(nphi, 1)), dtype=cp.float64)
    common = lambda p0, n: (np.int32(p0), np.int32(p0 + n), np.int32(nlyr), np.int32(nmom), np.int32(nstr),
                            np.int32(flags), d["dtau"], d["ssa"], d["pmom"], d["umu0"], d["fbeam"],
                            d["albedo"], d["zlev"], np.float64(radius), np.int32(nout), d_out, np.int32(numu),
                            d_umu, np.int32(nphi), d_phi, d_pt)
    if numu > 0 and not (flags & TWOSTREAM) and nprob * n_str <= MODE_THREADS:
        # radiances on few problems: the Fourier modes on separate threads
        # (radtran_disort_modes), then the partials summed in mode order and TMS
        # (radtran_disort_finish); the same words as one thread per problem,
        # with nstr times the parallelism
        nmodes = n_str
        chunk = max(1, min(nprob, budget_bytes // (8 * wstride * nmodes)))
        work = cp.zeros(wstride * chunk * nmodes, dtype=cp.float64)
        radm = cp.zeros(chunk * nmodes * nout * numu * nphi, dtype=cp.float64)
        kmodes = get_kernel("radtran_disort", "radtran_disort_modes")
        kfin = get_kernel("radtran_disort", "radtran_disort_finish")
        for p0 in range(0, nprob, chunk):
            n = min(chunk, nprob - p0)
            kmodes(((n * nmodes + _TPB - 1) // _TPB,), (_TPB,),
                   common(p0, n) + (work, np.int64(wstride), flux, np.int32(nmodes), radm))
            kfin(((n + _TPB - 1) // _TPB,), (_TPB,),
                 common(p0, n) + (work, np.int64(wstride), rad, np.int32(nmodes), radm))
    else:
        work = cp.zeros(wstride * chunk, dtype=cp.float64)
        kern = get_kernel("radtran_disort", "radtran_disort")
        for p0 in range(0, nprob, chunk):
            n = min(chunk, nprob - p0)
            kern(((n + _TPB - 1) // _TPB,), (_TPB,), common(p0, n) + (work, np.int64(wstride), flux, rad))
    flux_h = cp.asnumpy(flux)
    rad_h = cp.asnumpy(rad)[:, :, :numu, :nphi]
    return flux_h, rad_h
