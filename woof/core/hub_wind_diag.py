"""Hub-height wind: in-model 2-D fields and per-history-interval statistics.

What a wind-farm reanalysis reads at every turbine is the wind at HUB
HEIGHT, and the events it is graded on -- ramps, gust fronts, cut-out
excursions -- live INSIDE a history interval: a 15-minute file that holds
only the instantaneous state at its own valid time cannot say that the
hub wind went from 9 to 21 m/s and back in the eight minutes before it.
So this diagnostic folds EVERY completed model step into a running window
and publishes, per configured height above ground:

``WSPD_HUB`` / ``WDIR_HUB``
    Instantaneous speed and earth-relative direction (meteorological,
    the bearing the wind blows FROM, degrees clockwise from north) at the
    frame's valid time.
``WSPD_HUB_MAX`` / ``WSPD_HUB_MIN`` / ``WSPD_HUB_MEAN``
    Over the window ``(T_prev_frame, T_frame]``: the maximum and minimum
    of the end-of-step speeds, and their TIME-WEIGHTED mean
    ``sum(speed_n * dt_n) / sum(dt_n)`` (an adaptive or substepped run
    weights every step by its own ``dt``).
``HUB_HEIGHT`` (the ``hub_height`` coordinate) and ``HUB_WINDOW_S``
    The heights, and how many model seconds the window covers.

The gridded members share one ``hub_height`` axis
(``Time, hub_height, south_north, west_east``) -- WRF's own precedent for
height-level diagnostics is the ``z_lev_diags`` family on
``num_z_levels_stag`` -- rather than one variable per height, so the
names are fixed (they join the [output] vocabulary like any other row)
whatever heights a run configures.

THE ARITHMETIC, PINNED (and mirrored operation for operation by the NumPy
reference below; the device kernel compiles with ``--fmad=false`` so the
speed is bit-identical to the mirror and the direction differs only by
``atan2f``'s ulps):

1. heights of the MASS levels above ground, per column::

       z_w[k] = (PH[k] + PHB[k]) / g          (w levels, k = 0..nz)
       z[k]   = 0.5 * (z_w[k] + z_w[k+1]) - HGT

2. C-grid destaggering to the mass point, at each mass level::

       u[k] = 0.5 * (U[k, j, i] + U[k, j, i+1])
       v[k] = 0.5 * (V[k, j, i] + V[k, j+1, i])

3. linear interpolation in z-AGL between the bracketing mass levels
   ``z[kb] <= h < z[kb+1]``: ``w = (h - z[kb]) / (z[kb+1] - z[kb])``,
   ``u_h = u[kb] + w * (u[kb+1] - u[kb])`` (and ``v_h`` alike).  A height
   BELOW the lowest mass level holds the lowest level's wind; one above
   the top mass level holds the top's.  No log-law extrapolation is
   invented: configure model levels whose first mass level sits below the
   lowest hub height (the reanalysis profiles do), and the hold never
   fires.
4. speed ``sqrt(u_h*u_h + v_h*v_h)`` from the GRID-relative components
   (the rotation cannot change it, so it is not allowed to round it);
5. earth-relative components with WRF's rotation
   ``ue = u_h*COSALPHA - v_h*SINALPHA``, ``ve = v_h*COSALPHA + u_h*SINALPHA``
   and direction ``180 + atan2(ue, ve) * 180/pi`` wrapped to ``[0, 360)``;
   a calm (speed exactly 0) reports direction 0.

WHEN IT FOLDS, AND WHEN IT RESETS.  Folded in the dycore epilogue of every
completed step (:func:`woof.core.dycore.step`, through
:mod:`woof.core.interval_diagnostics`), the same instant ``UP_HELI_MAX``
is folded; restarted immediately after each history frame is snapshotted
(:meth:`woof.io.wrfout.PerDomainWrfoutWriters.submit`,
:func:`woof.runtime.write_case_output`).  A frame whose window holds no
step (the cold-start frame) reports MAX = MIN = MEAN = the instantaneous
speed and ``HUB_WINDOW_S = 0``.

RESTART SEMANTICS (documented, not checkpointed).  The window is not
written into checkpoints: it is output, not model state, and like the
rest of ``[output]`` it binds no restart identity.  A resumed run starts a
fresh window at the restart instant, so the first frame after a resume
covers ``(T_restart, T_frame]`` and SAYS SO through ``HUB_WINDOW_S``.
With the restart cadence a multiple of the history cadence -- the normal
configuration -- the restart instant is a frame instant and every window
is exactly what an uninterrupted run would have written.  Per-site samples
taken between the last frame and a crash are lost with the window they
belonged to; the resumed run's next sidecar starts after the restart.

OPTIONAL SITE SAMPLING.  With ``[output] hub_sites`` the diagnostic also
samples every site's column every ``hub_sample_interval_s`` (the step
whose end is nearest each multiple of the interval, counted from the
window start) into device memory, and flushes one sidecar per history
write: ``hubwind_d0N_<valid time>.nc``, dimensions
``time x site x height``.  A site maps to its NEAREST mass cell
(great-circle distance), which at the 100-300 m spacings this is built
for is within ~0.7 dx of the turbine; sites farther than ``max(dx, dy)``
from every cell, or nearest an outermost row or column, are outside the
domain and are listed in the sidecar's ``sites_outside_domain`` rather
than sampled.

NOT SUPPORTED, AND REFUSED BY NAME (:class:`HubWindUnsupportedRoad`):
``[tiles]``-streamed domains (the step runs as a tile sweep that never
reaches the dycore epilogue, and the frame is built from the host store),
ensemble member capture, and nest relocation.

DEVICE MEMORY.  ``6 * n_hub`` float32 planes per domain (instantaneous
speed and direction, max, min, time-weighted sum and mean) plus the site
buffer; these are not scratch slots, so the preflight VRAM plan does not
price them: 6 heights on a 1500 x 1500 domain is 324 MB.
"""
from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import math
import os
from pathlib import Path

import numpy as np

from woof.core.constants import G
from woof.io.wrf_output_schema import HUB_WIND_OUTPUT_FIELDS

DTYPE = np.float32

#: The registry name (:mod:`woof.core.interval_diagnostics`).
DIAGNOSTIC_NAME = "hub_wind"

_G = DTYPE(G)
_F05 = DTYPE(0.5)
_F0 = DTYPE(0.0)
_F180 = DTYPE(180.0)
_F360 = DTYPE(360.0)
#: 180/pi rounded once to float32, the constant both the kernel and the
#: mirror multiply by.
_RAD2DEG = DTYPE(180.0 / math.pi)

_THREADS = 256

#: Mode argument of the kernel: instantaneous only, start a window, fold.
MODE_INSTANT = 0
MODE_START = 1
MODE_FOLD = 2


class HubWindError(RuntimeError):
    """The hub-height wind diagnostic cannot run as configured."""


class HubWindUnsupportedRoad(HubWindError):
    """A run road that would bypass the window (streamed, captured, moved)."""


# ---------------------------------------------------------------------------
# The CUDA kernel.  Inline (the cumulus_clock / tendency_coupling precedent)
# rather than a woof/core/kernels/*.cu module: a new translation unit there
# must carry a measured local-frame row for every complete per-architecture
# recording in woof/core/kernel_frame_recordings.py, and this unit allocates
# no local arrays.  Its bytes are pinned by SHA-256 in
# tests/test_hub_wind_diag.py exactly as the module freeze pins a .cu file.
# ---------------------------------------------------------------------------

KERNEL_SOURCE = r"""
extern "C" {

__device__ __forceinline__ float hw_zw(const float* ph, const float* phb,
                                       int phb3d, long long plane, long long c,
                                       int k, float grav)
{
    const float base = phb3d ? phb[(long long)k * plane + c] : phb[k];
    return (ph[(long long)k * plane + c] + base) / grav;
}

__device__ __forceinline__ float hw_um(const float* u, int ny, int nx,
                                       int k, int j, int i)
{
    const long long row = ((long long)k * ny + j) * (long long)(nx + 1);
    return 0.5f * (u[row + i] + u[row + i + 1]);
}

__device__ __forceinline__ float hw_vm(const float* v, int ny, int nx,
                                       int k, int j, int i)
{
    const long long lo = ((long long)k * (ny + 1) + j) * (long long)nx;
    return 0.5f * (v[lo + i] + v[lo + nx + i]);
}

__device__ __forceinline__ void hw_emit(float uh, float vh, float sa, float ca,
                                        long long o, float* spd, float* wdir,
                                        float* smax, float* smin, float* ssum,
                                        float dt, int mode)
{
    const float s = sqrtf(uh * uh + vh * vh);
    float d = 0.0f;
    if (s > 0.0f) {
        const float ue = uh * ca - vh * sa;
        const float ve = vh * ca + uh * sa;
        d = 180.0f + atan2f(ue, ve) * HW_RAD2DEG;
        if (d >= 360.0f) d = d - 360.0f;
        if (d < 0.0f) d = d + 360.0f;
    }
    spd[o] = s;
    wdir[o] = d;
    if (mode == 1) {
        smax[o] = s;
        smin[o] = s;
        ssum[o] = s * dt;
    } else if (mode == 2) {
        smax[o] = fmaxf(smax[o], s);
        smin[o] = fminf(smin[o], s);
        ssum[o] = ssum[o] + s * dt;
    }
}

__global__ void hub_wind_columns(
    const float* __restrict__ u, const float* __restrict__ v,
    const float* __restrict__ ph, const float* __restrict__ phb,
    const int phb3d, const float* __restrict__ ht,
    const float* __restrict__ sina, const float* __restrict__ cosa,
    const float* __restrict__ heights, const int nh,
    const int nz, const int ny, const int nx, const float grav,
    float* spd, float* wdir, float* smax, float* smin, float* ssum,
    const float dt, const int mode)
{
    const long long plane = (long long)ny * nx;
    const long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= plane) return;
    const int j = (int)(c / nx);
    const int i = (int)(c - (long long)j * nx);
    const float hgt = ht[c];
    const float sa = sina[c];
    const float ca = cosa[c];

    float zw_hi = hw_zw(ph, phb, phb3d, plane, c, 1, grav);
    float z_lo = 0.5f * (hw_zw(ph, phb, phb3d, plane, c, 0, grav) + zw_hi) - hgt;
    float u_lo = hw_um(u, ny, nx, 0, j, i);
    float v_lo = hw_vm(v, ny, nx, 0, j, i);
    int h = 0;
    // Below the lowest mass level: hold the lowest level's wind.
    while (h < nh && heights[h] < z_lo) {
        hw_emit(u_lo, v_lo, sa, ca, (long long)h * plane + c,
                spd, wdir, smax, smin, ssum, dt, mode);
        ++h;
    }
    for (int k = 0; k + 1 < nz && h < nh; ++k) {
        const float zw_next = hw_zw(ph, phb, phb3d, plane, c, k + 2, grav);
        const float z_hi = 0.5f * (zw_hi + zw_next) - hgt;
        const float u_hi = hw_um(u, ny, nx, k + 1, j, i);
        const float v_hi = hw_vm(v, ny, nx, k + 1, j, i);
        while (h < nh && heights[h] < z_hi) {
            const float w = (heights[h] - z_lo) / (z_hi - z_lo);
            hw_emit(u_lo + w * (u_hi - u_lo), v_lo + w * (v_hi - v_lo),
                    sa, ca, (long long)h * plane + c,
                    spd, wdir, smax, smin, ssum, dt, mode);
            ++h;
        }
        zw_hi = zw_next;
        z_lo = z_hi;
        u_lo = u_hi;
        v_lo = v_hi;
    }
    // Above the top mass level: hold the top level's wind.
    while (h < nh) {
        hw_emit(u_lo, v_lo, sa, ca, (long long)h * plane + c,
                spd, wdir, smax, smin, ssum, dt, mode);
        ++h;
    }
}

}
""".replace("HW_RAD2DEG", f"{float(_RAD2DEG):.9e}f")

#: NVRTC options: no FMA contraction, so every multiply-add rounds twice
#: exactly as the NumPy mirror's separate operators do.
KERNEL_OPTIONS = ("--fmad=false",)

#: SHA-256 of the assembled source, pinned by tests/test_hub_wind_diag.py.
KERNEL_SOURCE_SHA256 = hashlib.sha256(KERNEL_SOURCE.encode("utf-8")).hexdigest()


@lru_cache(maxsize=None)
def _kernel(device_id: int):
    """The compiled kernel for one device (CuPy modules are per device)."""
    import cupy as cp

    with cp.cuda.Device(device_id):
        module = cp.RawModule(code=KERNEL_SOURCE, options=KERNEL_OPTIONS)
        return module.get_function("hub_wind_columns")


# ---------------------------------------------------------------------------
# NumPy reference (the mirror the kernel is graded against)
# ---------------------------------------------------------------------------

def mass_heights_agl_np(ph, phb, ht) -> np.ndarray:
    """``z[k] = 0.5*(z_w[k] + z_w[k+1]) - HGT`` with ``z_w = (PH+PHB)/g``.

    ``phb`` may be the flat base state's ``(nz+1,)`` column or the full
    ``(nz+1, ny, nx)`` volume, as on :class:`woof.core.state.DomainState`.
    """
    ph = np.asarray(ph, dtype=DTYPE)
    phb = np.asarray(phb, dtype=DTYPE)
    if phb.ndim == 1:
        phb = phb[:, None, None]
    z_w = (ph + phb) / _G
    return _F05 * (z_w[:-1] + z_w[1:]) - np.asarray(ht, dtype=DTYPE)


def destagger_np(u, v) -> tuple[np.ndarray, np.ndarray]:
    """C-grid U ``(nz, ny, nx+1)`` and V ``(nz, ny+1, nx)`` to mass points."""
    u = np.asarray(u, dtype=DTYPE)
    v = np.asarray(v, dtype=DTYPE)
    return (_F05 * (u[:, :, :-1] + u[:, :, 1:]),
            _F05 * (v[:, :-1, :] + v[:, 1:, :]))


def interp_to_heights_np(fields, z, heights) -> list[np.ndarray]:
    """Linear-in-z interpolation of mass-level ``fields`` to ``heights``.

    ``z`` is ``(nz, ny, nx)`` metres above ground, increasing with ``k``.
    Returns one ``(n_hub, ny, nx)`` array per input field.  A height below
    ``z[0]`` holds level 0; one at or above ``z[nz-1]`` holds the top.
    """
    z = np.asarray(z, dtype=DTYPE)
    nz = z.shape[0]
    outs = [np.empty((len(heights),) + z.shape[1:], dtype=DTYPE)
            for _ in fields]
    for h_index, height in enumerate(np.asarray(heights, dtype=DTYPE)):
        kb = np.sum(z <= height, axis=0) - 1                 # (ny, nx)
        below = kb < 0
        above = kb >= nz - 1
        lo = np.clip(kb, 0, nz - 2)[None]
        hi = lo + 1
        z_lo = np.take_along_axis(z, lo, axis=0)[0]
        z_hi = np.take_along_axis(z, hi, axis=0)[0]
        with np.errstate(all="ignore"):
            w = (height - z_lo) / (z_hi - z_lo)
        for field, out in zip(fields, outs):
            f_lo = np.take_along_axis(field, lo, axis=0)[0]
            f_hi = np.take_along_axis(field, hi, axis=0)[0]
            value = f_lo + w * (f_hi - f_lo)
            value = np.where(below, field[0], value)
            value = np.where(above, field[nz - 1], value)
            out[h_index] = value
    return outs


def speed_direction_np(uh, vh, sina, cosa) -> tuple[np.ndarray, np.ndarray]:
    """Speed (grid-relative components) and earth-relative FROM-direction."""
    uh = np.asarray(uh, dtype=DTYPE)
    vh = np.asarray(vh, dtype=DTYPE)
    sina = np.asarray(sina, dtype=DTYPE)
    cosa = np.asarray(cosa, dtype=DTYPE)
    speed = np.sqrt(uh * uh + vh * vh)
    ue = uh * cosa - vh * sina
    ve = vh * cosa + uh * sina
    direction = _F180 + np.arctan2(ue, ve) * _RAD2DEG
    direction = np.where(direction >= _F360, direction - _F360, direction)
    direction = np.where(direction < _F0, direction + _F360, direction)
    direction = np.where(speed > _F0, direction, _F0).astype(DTYPE)
    return speed.astype(DTYPE), direction


def hub_wind_np(u, v, ph, phb, ht, sina, cosa, heights
                ) -> tuple[np.ndarray, np.ndarray]:
    """``(speed, direction)``, each ``(n_hub, ny, nx)``: steps 1-5 above."""
    z = mass_heights_agl_np(ph, phb, ht)
    um, vm = destagger_np(u, v)
    uh, vh = interp_to_heights_np((um, vm), z, heights)
    return speed_direction_np(uh, vh, sina, cosa)


def fold_window_np(speed, smax, smin, ssum, dt, mode) -> None:
    """The kernel's window update on host arrays, in place."""
    dt = DTYPE(dt)
    if mode == MODE_START:
        smax[...] = speed
        smin[...] = speed
        ssum[...] = speed * dt
    elif mode == MODE_FOLD:
        np.fmax(smax, speed, out=smax)
        np.fmin(smin, speed, out=smin)
        ssum[...] = ssum + speed * dt


# ---------------------------------------------------------------------------
# Site location
# ---------------------------------------------------------------------------

_EARTH_RADIUS_M = 6370000.0  # WRF's EARTH_RADIUS_M (share/module_llxy.F)


def _haversine_m(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dlam = np.radians(((np.asarray(lon2) - lon1 + 180.0) % 360.0) - 180.0)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlam / 2) ** 2
    return 2.0 * _EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def locate_sites(sites, lat, lon, *, dx: float, dy: float) -> dict:
    """Nearest mass cell of each ``(site_id, lat, lon)``; split in/out.

    Returns ``{"inside": [(site_id, lat, lon, j, i, cell_lat, cell_lon,
    distance_m), ...], "outside": [site_id, ...]}``.  A site is OUTSIDE
    when its nearest cell is on an outermost row/column or farther than
    ``max(dx, dy)`` from it (a site inside a cell is at most ~0.71 of a
    cell from its centre).
    """
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    ny, nx = lat.shape
    limit = float(max(dx, dy))
    inside, outside = [], []
    for site_id, site_lat, site_lon in sites:
        # Local equirectangular metric for the search, great circle for
        # the recorded distance.
        dlat = lat - site_lat
        dlon = (((lon - site_lon) + 180.0) % 360.0 - 180.0) * math.cos(
            math.radians(site_lat))
        flat = int(np.argmin(dlat * dlat + dlon * dlon))
        j, i = divmod(flat, nx)
        distance = float(_haversine_m(site_lat, site_lon, lat[j, i], lon[j, i]))
        on_edge = j in (0, ny - 1) or i in (0, nx - 1)
        if on_edge or distance > limit:
            outside.append(site_id)
            continue
        inside.append((site_id, float(site_lat), float(site_lon), j, i,
                       float(lat[j, i]), float(lon[j, i]), distance))
    return {"inside": inside, "outside": outside}


# ---------------------------------------------------------------------------
# The diagnostic
# ---------------------------------------------------------------------------

def _array_module(state):
    if isinstance(state.u, np.ndarray):
        return np
    import cupy as cp
    return cp


def _epoch_seconds(moment: datetime) -> float:
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.timestamp()


class _SiteSampler:
    """Per-site hub-wind time series of one window, held in device memory."""

    def __init__(self, xp, located, heights, interval_s: float,
                 outside, sites_source):
        self.xp = xp
        self.interval_s = float(interval_s)
        self.rows = located
        self.outside = tuple(outside)
        self.sites_source = sites_source
        self.heights = np.asarray(heights, dtype=DTYPE)
        self.sj = xp.asarray(np.array([r[3] for r in located], dtype=np.int64))
        self.si = xp.asarray(np.array([r[4] for r in located], dtype=np.int64))
        self.n_sites = len(located)
        self._capacity = 0
        self._speed = None
        self._direction = None
        self.reset()

    def reset(self) -> None:
        self.offsets: list[float] = []
        self._next = self.interval_s

    def _grow(self) -> None:
        capacity = max(16, 2 * self._capacity)
        shape = (capacity, self.n_sites, len(self.heights))
        speed = self.xp.empty(shape, dtype=DTYPE)
        direction = self.xp.empty(shape, dtype=DTYPE)
        n = len(self.offsets)
        if n:
            speed[:n] = self._speed[:n]
            direction[:n] = self._direction[:n]
        self._speed, self._direction, self._capacity = speed, direction, capacity

    def record(self, offset_s: float, speed, direction) -> None:
        n = len(self.offsets)
        if n >= self._capacity:
            self._grow()
        self._speed[n] = speed[:, self.sj, self.si].T
        self._direction[n] = direction[:, self.sj, self.si].T
        self.offsets.append(float(offset_s))

    def after_step(self, window_s: float, dt: float, speed, direction) -> None:
        # The step whose END is nearest each multiple of the interval.
        if window_s >= self._next - 0.5 * dt:
            self.record(window_s, speed, direction)
            while self._next <= window_s + 0.5 * dt:
                self._next += self.interval_s

    def host_series(self):
        n = len(self.offsets)
        if self.xp is np:
            return self._speed[:n].copy(), self._direction[:n].copy()
        return self._speed[:n].get(), self._direction[:n].get()


class HubWindDiagnostic:
    """One domain's hub-height wind window (see the module docstring).

    Construct through :func:`attach_hub_wind`.  The rotation is read from
    the state's own ``sina``/``cosa`` (``SINALPHA``/``COSALPHA``) at every
    evaluation, so it is always the grid the state is on.
    """

    name = DIAGNOSTIC_NAME

    def __init__(self, request, *, ny: int, nx: int, xp=np,
                 device_id: int | None = None, site_rows=None,
                 sites_outside=(), grid_id: int = 1):
        self.request = request
        self.heights_m = np.asarray(request.heights_m, dtype=DTYPE)
        self.ny, self.nx = int(ny), int(nx)
        self.xp = xp
        self.device_id = device_id
        self.grid_id = int(grid_id)
        n_hub = len(self.heights_m)
        shape = (n_hub, self.ny, self.nx)
        with self._device():
            self._heights = xp.asarray(self.heights_m)
            self.speed = xp.zeros(shape, dtype=DTYPE)
            self.direction = xp.zeros(shape, dtype=DTYPE)
            self.smax = xp.zeros(shape, dtype=DTYPE)
            self.smin = xp.zeros(shape, dtype=DTYPE)
            self.ssum = xp.zeros(shape, dtype=DTYPE)
            #: WSPD_HUB_MEAN, divided into a standing buffer per frame.
            self.smean = xp.zeros(shape, dtype=DTYPE)
            self.sampler = None
            if request.sites and site_rows:
                self.sampler = _SiteSampler(
                    xp, site_rows, self.heights_m, request.sample_interval_s,
                    sites_outside, request.sites_source)
        self.sites_outside = tuple(sites_outside)
        #: Model seconds folded into the open window, and its step count.
        self.window_s = 0.0
        self.window_steps = 0

    # -- plumbing ----------------------------------------------------------

    def _device(self):
        if self.xp is np or self.device_id is None:
            from contextlib import nullcontext
            return nullcontext()
        import cupy as cp
        return cp.cuda.Device(self.device_id)

    def _compute(self, state, mode: int, dt: float) -> None:
        """One evaluation of steps 1-5 over the domain, folding by ``mode``."""
        if self.xp is np:
            speed, direction = hub_wind_np(
                state.u, state.v, state.php, state.phb, state.ht,
                state.sina, state.cosa, self.heights_m)
            self.speed[...] = speed
            self.direction[...] = direction
            fold_window_np(speed, self.smax, self.smin, self.ssum, dt, mode)
            return
        nz = int(state.u.shape[0])
        for label, array in (("u", state.u), ("v", state.v),
                             ("ph", state.php), ("phb", state.phb),
                             ("ht", state.ht), ("sina", state.sina),
                             ("cosa", state.cosa)):
            if array.dtype != DTYPE or not array.flags.c_contiguous:
                raise HubWindError(
                    f"hub-height wind needs contiguous float32 {label}, got "
                    f"{array.dtype} (contiguous={array.flags.c_contiguous})")
        if state.u.shape != (nz, self.ny, self.nx + 1):
            raise HubWindError(
                f"U has shape {state.u.shape}; this diagnostic was built for "
                f"a {(self.ny, self.nx)} mass grid")
        plane = self.ny * self.nx
        blocks = (plane + _THREADS - 1) // _THREADS
        _kernel(int(state.u.device.id))(
            (blocks,), (_THREADS,),
            (state.u, state.v, state.php, state.phb,
             np.int32(0 if state.phb.ndim == 1 else 1), state.ht,
             state.sina, state.cosa, self._heights,
             np.int32(len(self.heights_m)), np.int32(nz),
             np.int32(self.ny), np.int32(self.nx), _G,
             self.speed, self.direction, self.smax, self.smin, self.ssum,
             DTYPE(dt), np.int32(mode)))

    # -- the registry protocol ----------------------------------------------

    def on_step(self, state, dt: float) -> None:
        """Fold one completed step of length ``dt`` into the window."""
        with self._device():
            mode = MODE_START if self.window_steps == 0 else MODE_FOLD
            self._compute(state, mode, dt)
            self.window_s += float(dt)
            self.window_steps += 1
            if self.sampler is not None:
                self.sampler.after_step(self.window_s, float(dt),
                                        self.speed, self.direction)

    def history_fields(self, state) -> dict:
        """The frame's hub fields: instantaneous NOW, plus the window."""
        with self._device():
            if self.window_steps == 0:
                # No step behind this frame (the cold start, the first frame
                # after a resume, an activating nest): evaluate the instant
                # it describes.  Otherwise the last step's evaluation IS the
                # frame's instant -- nothing moves u/v/ph after the epilogue.
                self._compute(state, MODE_INSTANT, 0.0)
                smax = smin = smean = self.speed
            else:
                smax, smin, smean = self.smax, self.smin, self.smean
                self.xp.divide(self.ssum, DTYPE(self.window_s), out=smean)
        fields = {
            "HUB_HEIGHT": self.heights_m.copy(),
            "HUB_WINDOW_S": np.asarray(self.window_s, dtype=DTYPE),
            "WSPD_HUB": self.speed,
            "WDIR_HUB": self.direction,
            "WSPD_HUB_MAX": smax,
            "WSPD_HUB_MIN": smin,
            "WSPD_HUB_MEAN": smean,
        }
        assert tuple(fields) == HUB_WIND_OUTPUT_FIELDS
        return fields

    def close_interval(self, state, *, valid_time, output_dir,
                       grid_id: int) -> Path | None:
        """Flush the window's site series (if any) and restart the window."""
        path = None
        with self._device():
            if self.sampler is not None:
                if not self.sampler.offsets and self.window_steps == 0:
                    # A frame with no step behind it (cold start): its one
                    # sample is the frame's own instant.
                    self._compute(state, MODE_INSTANT, 0.0)
                    self.sampler.record(0.0, self.speed, self.direction)
                if self.sampler.offsets:
                    path = write_site_sidecar(
                        self.sampler, valid_time=valid_time,
                        window_s=self.window_s, output_dir=output_dir,
                        grid_id=int(grid_id))
                self.sampler.reset()
        self.window_s = 0.0
        self.window_steps = 0
        return path

    def refuse_publisher(self, reason: str) -> None:
        raise HubWindUnsupportedRoad(
            f"[output] hub-height wind is configured on d{self.grid_id:02d}, "
            f"but {reason}.  The diagnostic folds every step in the dycore "
            "epilogue and restarts at every history write; this road would "
            "publish frames without it.  Remove the hub_* keys from this "
            "domain's [output], or run it on the resident road.")


# ---------------------------------------------------------------------------
# Sidecar
# ---------------------------------------------------------------------------

def sidecar_filename(valid_time: datetime, grid_id: int) -> str:
    """``hubwind_d0N_YYYY-MM-DD_HH:MM:SS.nc``, the wrfout time spelling."""
    return f"hubwind_d{int(grid_id):02d}_{valid_time:%Y-%m-%d_%H:%M:%S}.nc"


def write_site_sidecar(sampler: _SiteSampler, *, valid_time: datetime,
                       window_s: float, output_dir, grid_id: int) -> Path:
    """One window's ``time x site x height`` series, published atomically."""
    import netCDF4

    from woof.io.netcdf_serialization import NETCDF4_IO_LOCK

    speed, direction = sampler.host_series()
    end = _epoch_seconds(valid_time)
    times = np.array([round(end - (window_s - offset), 3)
                      for offset in sampler.offsets], dtype=np.float64)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / sidecar_filename(valid_time, grid_id)
    temp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    rows = sampler.rows
    with NETCDF4_IO_LOCK:
        with netCDF4.Dataset(temp, "w", format="NETCDF4") as ds:
            ds.createDimension("time", len(times))
            ds.createDimension("site", len(rows))
            ds.createDimension("height", len(sampler.heights))
            var = ds.createVariable("time", "f8", ("time",))
            var.units = "seconds since 1970-01-01 00:00:00"
            var.calendar = "standard"
            var.standard_name = "time"
            var[:] = times
            var = ds.createVariable("height", "f4", ("height",))
            var.units = "m"
            var.long_name = "height above ground level"
            var.positive = "up"
            var[:] = sampler.heights
            var = ds.createVariable("site_id", str, ("site",))
            for index, row in enumerate(rows):
                var[index] = row[0]
            for name, column, dtype, units, long_name in (
                    ("site_lat", 1, "f8", "degree_north", "requested site latitude"),
                    ("site_lon", 2, "f8", "degree_east", "requested site longitude"),
                    ("grid_j", 3, "i4", "1", "0-based south_north index of the sampled mass cell"),
                    ("grid_i", 4, "i4", "1", "0-based west_east index of the sampled mass cell"),
                    ("grid_lat", 5, "f8", "degree_north", "sampled mass cell latitude"),
                    ("grid_lon", 6, "f8", "degree_east", "sampled mass cell longitude"),
                    ("site_distance_m", 7, "f8", "m", "great-circle distance site to cell centre")):
                var = ds.createVariable(name, dtype, ("site",))
                var.units = units
                var.long_name = long_name
                var[:] = np.array([row[column] for row in rows])
            var = ds.createVariable("WSPD", "f4", ("time", "site", "height"))
            var.units = "m s-1"
            var.long_name = "wind speed at hub height AGL"
            var[:] = speed
            var = ds.createVariable("WDIR", "f4", ("time", "site", "height"))
            var.units = "degree"
            var.long_name = ("earth-relative wind direction (from), "
                             "clockwise from north, at hub height AGL")
            var[:] = direction
            ds.setncatts({
                "title": "woof hub-height wind site samples",
                "grid_id": np.int32(grid_id),
                "valid_time": f"{valid_time:%Y-%m-%d_%H:%M:%S}",
                "window_seconds": float(window_s),
                "sample_interval_s": float(sampler.interval_s),
                "sites_source": str(sampler.sites_source or ""),
                "sites_outside_domain": ",".join(sampler.outside),
                "method": ("nearest mass cell; linear in z-AGL between mass "
                           "levels from (PH+PHB)/g-HGT; destaggered C-grid "
                           "winds rotated with SINALPHA/COSALPHA; samples at "
                           "the step end nearest each multiple of "
                           "sample_interval_s from the window start"),
            })
    os.replace(temp, path)
    return path


# ---------------------------------------------------------------------------
# Attachment
# ---------------------------------------------------------------------------

def attach_hub_wind(state, selection, *, nx: int, ny: int, dx: float,
                    dy: float, grid_id: int, latlon=None,
                    streamed: bool = False):
    """Attach the diagnostic ``selection`` asks for to ``state`` (or nothing).

    ``selection`` is the domain's resolved
    :class:`woof.io.history_selection.HistorySelection`; ``latlon`` is a
    zero-argument callable returning the mass-point ``(XLAT, XLONG)``
    planes, needed only when sites are configured.  Refuses a streamed
    domain by name.  Returns the attached diagnostic, or ``None`` when hub
    wind is off.
    """
    from woof.core import interval_diagnostics

    request = getattr(selection, "hub_wind", None)
    if request is None:
        return None
    if streamed or getattr(state, "_streamed_domain", None) is not None:
        raise HubWindUnsupportedRoad(
            f"[output] hub-height wind is configured on d{int(grid_id):02d}, "
            "which runs [tiles]-streamed: the step is a tile sweep that never "
            "reaches the dycore epilogue the window folds in, and the frame "
            "is built from the host store.  Run the domain resident (its "
            "own tiles = { mode = \"off\" }) or remove the hub_* keys.")
    existing = interval_diagnostics.get(state, DIAGNOSTIC_NAME)
    if existing is not None:
        if existing.request == request:
            return existing
        raise HubWindError(
            f"d{int(grid_id):02d} already carries a hub-height wind window "
            "built from a different [output] request")
    xp = _array_module(state)
    device_id = None if xp is np else int(state.u.device.id)
    site_rows, outside = None, ()
    if request.sites:
        if latlon is None:
            raise HubWindError(
                f"d{int(grid_id):02d}: [output] hub_sites needs the domain's "
                "mass-point latitude/longitude to place the sites, and this "
                "road supplied none")
        lat, lon = (_to_host(plane) for plane in latlon())
        located = locate_sites(request.sites, lat, lon, dx=dx, dy=dy)
        site_rows, outside = located["inside"], tuple(located["outside"])
        if not site_rows:
            from woof.explain import warn
            warn(f"d{int(grid_id):02d}: none of the {len(request.sites)} "
                 "[output] hub_sites lies inside this domain, so it writes "
                 "no hubwind sidecar",
                 why="a site is sampled by every domain that contains it "
                     "and by no other")
    diagnostic = HubWindDiagnostic(
        request, ny=ny, nx=nx,
        xp=xp, device_id=device_id, site_rows=site_rows,
        sites_outside=outside, grid_id=grid_id)
    interval_diagnostics.attach(state, diagnostic)
    return diagnostic


def _to_host(array):
    return array if isinstance(array, np.ndarray) else np.asarray(
        array.get() if hasattr(array, "get") else array)


def hub_history_shapes(selection, ny: int, nx: int) -> dict[str, tuple]:
    """On-disk shapes the diagnostic adds to a frame (disk pricing)."""
    request = getattr(selection, "hub_wind", None)
    if request is None:
        return {}
    n_hub = len(request.heights_m)
    shapes = {"HUB_HEIGHT": (n_hub,), "HUB_WINDOW_S": ()}
    for name in HUB_WIND_OUTPUT_FIELDS[2:]:
        shapes[name] = (n_hub, int(ny), int(nx))
    return shapes


__all__ = [
    "DIAGNOSTIC_NAME",
    "HubWindDiagnostic",
    "HubWindError",
    "HubWindUnsupportedRoad",
    "KERNEL_OPTIONS",
    "KERNEL_SOURCE",
    "KERNEL_SOURCE_SHA256",
    "attach_hub_wind",
    "destagger_np",
    "fold_window_np",
    "hub_history_shapes",
    "hub_wind_np",
    "interp_to_heights_np",
    "locate_sites",
    "mass_heights_agl_np",
    "sidecar_filename",
    "speed_direction_np",
    "write_site_sidecar",
]
