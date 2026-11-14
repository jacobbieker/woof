"""E1: hex grid nudging tendencies, column by column, against woof's WRF FDDA path.

Each hex cell column is replayed as a horizontally uniform WRF domain (3 x 3
columns, C-grid winds) through the WRF route's own operator
(woof.spectral_nudging.forecast.attach -> SpectralNudger) on CUDA, or through
the same WRF kernels on the WRF layout (staggered wind weights, the WRF
boundary row) on a facade backend, where cuFFT graphs are not available.  The
held tendency of the centre column must equal the hex tendency word for word.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from woof import backend
from woof.hex import nudging
from woof.spectral_nudging.config import NudgingConfig

cp = None


@pytest.fixture(autouse=True)
def _device():
    """Each test needs a device array module (CUDA, or the Metal facade); a CPU-only box skips."""
    global cp
    cp = pytest.importorskip("cupy")

NZ, NC, DT, INTERVAL_M = 8, 5, 30.0, 60.0
ELAPSED = DT  # step 2: a calculation step for every fgdt cadence used here


def _columns(seed=3):
    rng = np.random.default_rng(seed)
    f = np.float32
    z = np.cumsum(np.full((NZ + 1, NC), 400.0) + rng.uniform(0, 80, (NZ + 1, NC)), axis=0)
    z = (z - z[:1] + rng.uniform(0, 900, (1, NC))).astype(f)
    base = dict(
        theta_m=(300.0 + 4.0 * np.arange(NZ)[:, None] + rng.normal(0, 1, (NZ, NC))).astype(f),
        qv=(0.012 * np.exp(-np.arange(NZ) / 3.0)[:, None] * rng.uniform(.6, 1.2, (NZ, NC))).astype(f),
        u=rng.normal(5, 4, (NZ, NC)).astype(f), v=rng.normal(-2, 4, (NZ, NC)).astype(f))
    analysis = {}
    for tag, scale in (("a", 1.0), ("b", 1.7)):
        analysis[tag] = dict(
            temperature=(base["theta_m"] / (f(1) + (f(461.6) / f(287)) * base["qv"])
                         + f(scale) * rng.normal(0, 1.5, (NZ, NC)).astype(f)).astype(f),
            qv=(base["qv"] * f(1 + .1 * scale)).astype(f),
            u=(base["u"] + rng.normal(0, 3, (NZ, NC))).astype(f),
            v=(base["v"] + rng.normal(0, 3, (NZ, NC))).astype(f))
    pblh = rng.uniform(300, 2500, NC).astype(f)
    return z, base, analysis, pblh


CONFIGS = {
    "grid-pbl-zfac": dict(mode="grid", guv=3e-4, gt=3e-4, gq=1e-4, if_no_pbl_nudging_uv=1,
                          if_no_pbl_nudging_t=0, if_no_pbl_nudging_q=1, if_zfac_t=1, k_zfac_t=2,
                          if_zfac_uv=0, if_zfac_q=0),
    "grid-plain": dict(mode="grid", guv=2e-4, gt=1e-4, gq=0.0, if_no_pbl_nudging_uv=0,
                       if_no_pbl_nudging_t=0, if_no_pbl_nudging_q=0),
    "grid-cadence-ramp": dict(mode="grid", guv=3e-4, gt=3e-4, gq=1e-4, fgdt=1.0, if_ramping=1,
                              dtramp_min=30.0, gfdda_end_h=0.5, if_no_pbl_nudging_uv=1,
                              if_no_pbl_nudging_t=1, if_no_pbl_nudging_q=1),
}


def _config(name):
    row = dict(enabled=True, convention="wrf", gfdda_interval_m=INTERVAL_M, boundary_cells=1,
               cutoff_x=0, cutoff_y=0)
    row.update(CONFIGS[name])
    return NudgingConfig(**row)


def _hex(config, z, base, analysis, pblh, monkeypatch):
    mesh = SimpleNamespace(n_cells=NC, n_edges=7, attrs={},
                           arrays={"latCell": np.zeros(NC), "lonCell": np.zeros(NC)})
    frames = [nudging.AnalysisFrame(None, "t", i * INTERVAL_M * 60.0, "") for i in range(2)]
    forcing = nudging.HexNudging(
        config, frames=frames, mesh=mesh, prep_geometry=SimpleNamespace(n_cells=NC, n_edges=7),
        physics_geometry=None, zgrid=z, scalar_names=("qv", "qc"), dt=DT, kernel_cache=None,
        run_seconds=3600.0, pbl_height=(lambda: cp.asarray(pblh)))
    device = {tag: {k: cp.asarray(v) for k, v in fields.items()} for tag, fields in analysis.items()}
    monkeypatch.setattr(forcing, "_frame_device", lambda index: device["ab"[index]])
    winds = {"u": cp.asarray(base["u"]), "v": cp.asarray(base["v"])}
    monkeypatch.setattr(forcing, "_reconstruct", lambda normal: winds)
    scalars = cp.asarray(np.stack([base["qv"], np.zeros_like(base["qv"])]))
    tend = forcing.tendencies(time_seconds=ELAPSED, theta_m=cp.asarray(base["theta_m"]), scalars=scalars,
                              normal_velocity=None, pblh=cp.asarray(pblh))
    return {name: cp.asnumpy(value) for name, value in tend.items()}


def _wrf_state(column, z, base, pblh):
    """A 3 x 3 WRF domain whose every column (and every C-grid wind point) is hex column ``column``."""
    ny = nx = 3

    def mass(values):
        return cp.asarray(np.ascontiguousarray(np.broadcast_to(values[:, column, None, None], (values.shape[0], ny, nx))))

    phb = (z * nudging.GRAVITY_KERNEL).astype(np.float32)
    state = SimpleNamespace(
        u=cp.asarray(np.ascontiguousarray(np.broadcast_to(base["u"][:, column, None, None], (NZ, ny, nx + 1)))),
        v=cp.asarray(np.ascontiguousarray(np.broadcast_to(base["v"][:, column, None, None], (NZ, ny + 1, nx)))),
        thp=mass(base["theta_m"]), thb=cp.zeros((NZ, ny, nx), dtype=cp.float32),
        p=cp.full((NZ, ny, nx), 80000.0, dtype=cp.float32), php=cp.zeros((NZ + 1, ny, nx), dtype=cp.float32),
        phb=mass(phb), ht=cp.full((ny, nx), z[0, column], dtype=cp.float32), qv=mass(base["qv"]),
        elapsed_seconds=ELAPSED, use_theta_m=1)
    for name, array in (("u", state.u), ("v", state.v), ("temperature", state.thp), ("qv", state.qv)):
        setattr(state, "nudging_tendency_" + name, cp.zeros_like(array))
    return state, cp.full((ny, nx), pblh[column], dtype=cp.float32)


def _wrf_analysis(column, analysis, shapes):
    out = {}
    for name, shape in shapes.items():
        pair = []
        for tag in "ab":
            values = analysis[tag][name][:, column]
            pair.append(cp.asarray(np.ascontiguousarray(np.broadcast_to(values[:, None, None], shape))))
        out[name] = tuple(pair)
    return out


def _wrf_held_cuda(config, column, z, base, analysis, pblh):
    from woof.spectral_nudging.forecast import attach

    state, pbl = _wrf_state(column, z, base, pblh)
    shapes = {"u": state.u.shape, "v": state.v.shape, "temperature": state.thp.shape,
              "geopotential": state.php.shape, "qv": state.qv.shape}
    run = SimpleNamespace(use_adaptive_time_step=False, specified=True, nested=False, spec_zone=1,
                          relax_zone=0, dt=DT)
    names = [name for name in ("u", "v", "temperature", "qv") if config.coefficient(name) > 0]
    plan = attach(state, run, config, _wrf_analysis(column, analysis, {n: shapes[n] for n in names}),
                  interval_start=0.0, interval_end=INTERVAL_M * 60.0, pbl_height=pbl)
    # The WRF path's clock starts at state.elapsed_seconds; one enqueue is this step.
    plan.enqueue()
    cp.cuda.get_current_stream().synchronize()
    plan.check_at_seam()
    return {name: cp.asnumpy(plan.held[name])[:, 1, 1] for name in names}


def _wrf_held_kernels(config, column, z, base, analysis, pblh):
    """The same WRF kernels on the WRF layout, as SpectralNudger records them (facade backends)."""
    kernels = nudging.fdda_kernels()
    state, pbl = _wrf_state(column, z, base, pblh)
    ny = nx = 3
    clock = cp.asarray(np.asarray([ELAPSED, DT, 0.0, INTERVAL_M * 60.0]))
    error = cp.zeros(1, dtype=cp.uint32)
    end = np.float64(-1 if config.gfdda_end_h is None else config.gfdda_end_h * 3600.0)
    ramp = np.float64(config.dtramp_min * 60.0 if config.if_ramping else 0.0)
    cadence = np.int32(config.cadence_steps(DT))
    theta = cp.empty_like(state.thp)
    nudging._theta_kernel()(state.thp, state.qv, theta)  # forecast.cu pack_fields' expression
    model = {"u": state.u, "v": state.v, "temperature": theta, "qv": state.qv}
    held = {}
    for name in ("u", "v", "temperature", "qv"):
        if config.coefficient(name) <= 0:
            continue
        suffix = {"u": "uv", "v": "uv", "temperature": "t", "qv": "q"}[name]
        field = ("u", "v", "temperature", "geopotential", "qv").index(name)
        value = cp.ascontiguousarray(model[name])
        shape = value.shape
        weight = cp.empty(shape, dtype=cp.float32)
        sx, sy = shape[2], shape[1]
        kernels["wrf_weights"](((sx * sy + 255) // 256,), (256,), (
            state.php, state.phb, state.ht, pbl, weight, np.int32(NZ), np.int32(ny), np.int32(nx), np.int32(1),
            *(np.int32(v) for v in (field, config.mode == "grid", getattr(config, "if_no_pbl_nudging_" + suffix),
                                    getattr(config, "if_zfac_" + suffix), getattr(config, "k_zfac_" + suffix),
                                    getattr(config, "dk_zfac_" + suffix)))))
        a, b = _wrf_analysis(column, analysis, {name: shape})[name]
        n = value.size
        work = cp.zeros(shape + (2,), dtype=cp.float32)
        out = cp.zeros(shape, dtype=cp.float32)
        grid = ((n + 255) // 256,)
        kernels["difference"](grid, (256,), (value, a, b, work, clock, error, np.int32(n), np.int32(1),
                                             cadence, end, ramp, np.int32(config.mode == "grid")))
        kernels["relax"](grid, (256,), (value.copy(), work, weight, pbl, clock, np.int32(n),
                                        np.int32(shape[2]), np.int32(shape[1]), np.int32(config.boundary_cells),
                                        np.float32(config.minimum_height_m), np.float32(config.ramp_depth_m),
                                        np.float32(1.0 / config.coefficient(name)), weight, out, np.int32(1),
                                        cadence, np.int32(config.fgdtzero), end, ramp))
        held[name] = cp.asnumpy(out)[:, 1, 1]
    assert int(error.get()[0]) == 0
    return held


@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_grid_nudging_tendency_equals_wrf_per_column(name, monkeypatch):
    config = _config(name)
    z, base, analysis, pblh = _columns()
    hexed = _hex(config, z, base, analysis, pblh, monkeypatch)
    reference = _wrf_held_cuda if backend.active() == "cuda" else _wrf_held_kernels
    nonzero = 0
    for column in range(NC):
        wrf = reference(config, column, z, base, analysis, pblh)
        assert set(wrf) == set(hexed)
        for field, values in wrf.items():
            np.testing.assert_array_equal(hexed[field][:, column].view(np.uint32), values.view(np.uint32),
                                          err_msg=f"{name} {field} column {column}")
            nonzero += int(np.count_nonzero(values))
    assert nonzero > 0


def test_spectral_mean_mode_keeps_a_uniform_offset(monkeypatch):
    """Spectral FDDA keeps the domain mean (k = 0 in WRF, l = 0 here): a uniform offset passes unfiltered."""
    pytest.importorskip("netCDF4")
    config = NudgingConfig(enabled=True, mode="spectral", convention="wrf", gt=3e-4, guv=0.0, gq=0.0,
                           gfdda_interval_m=INTERVAL_M, cutoff_x=0, cutoff_y=0)
    rng = np.random.default_rng(1)
    n = 40
    lat = rng.uniform(-1.2, 1.2, n)
    lon = rng.uniform(-np.pi, np.pi, n)
    mesh = SimpleNamespace(n_cells=n, n_edges=1, attrs={},
                           arrays={"latCell": lat, "lonCell": lon, "areaCell": rng.uniform(1, 2, n)})
    filt = nudging.HarmonicFilter(mesh, 0)
    field = cp.asarray(np.full((3, n), 2.5, dtype=np.float32))
    np.testing.assert_allclose(cp.asnumpy(filt.apply(field)), 2.5, rtol=2e-6)
    assert config.coefficient("temperature") == pytest.approx(3e-4)


def _sphere(n, seed=5):
    rng = np.random.default_rng(seed)
    xyz = rng.normal(size=(n, 3))
    xyz /= np.linalg.norm(xyz, axis=1, keepdims=True)
    lat, lon = np.arcsin(xyz[:, 2]), np.arctan2(xyz[:, 1], xyz[:, 0])
    return SimpleNamespace(n_cells=n, attrs={"sphere_radius": 1.0},
                           arrays={"latCell": lat, "lonCell": lon, "areaCell": np.full(n, 4 * np.pi / n)}), xyz


def test_harmonic_filter_keeps_low_degrees_and_removes_high_ones():
    mesh, xyz = _sphere(3000)
    x, y, z = xyz.T
    low = 1.0 + 2.0 * z + (3 * z * z - 1) + x * y            # degrees 0..2
    high = np.cos(12 * np.arctan2(y, x)) * (1 - z * z) ** 6    # degree 12, order 12
    filt = nudging.HarmonicFilter(mesh, 3)
    assert filt.kept == filt.modes == 16
    field = cp.asarray(np.stack([low, low + high]).astype(np.float32))
    out = cp.asnumpy(filt.apply(field))
    np.testing.assert_allclose(out[0], low, atol=2e-4)
    # Least squares on scattered points aliases a little of the degree-12 wave.
    assert np.sqrt(np.mean((out[1] - low) ** 2)) < 0.1 * np.sqrt(np.mean(high ** 2))


def test_harmonic_filter_on_a_cap_drops_unresolved_modes():
    mesh, xyz = _sphere(20000)
    cap = xyz[:, 2] > np.cos(0.5)
    mesh.n_cells = int(cap.sum())
    for name in ("latCell", "lonCell", "areaCell"):
        mesh.arrays[name] = mesh.arrays[name][cap]
    mesh.arrays["bdyMaskCell"] = np.zeros(mesh.n_cells, dtype=np.int32)
    mesh.arrays["bdyMaskCell"][np.flatnonzero(xyz[cap, 2] < np.cos(0.47))] = 7
    filt = nudging.HarmonicFilter(mesh, 8)
    assert filt.kept < filt.modes
    x, y, z = xyz[cap].T
    smooth = (x + 0.5 * y).astype(np.float32)
    out = cp.asnumpy(filt.apply(cp.asarray(smooth[None, :])))[0]
    interior = mesh.arrays["bdyMaskCell"] == 0
    np.testing.assert_allclose(out[interior], smooth[interior], atol=2e-3)
