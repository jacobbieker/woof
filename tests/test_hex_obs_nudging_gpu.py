"""E1: the hex observation-nudging kernels against WRF's (woof/fddaobs/kernels.cu) where the geometries agree.

The hex kernels (woof/hex/obs_nudging.cu) are WRF v4.7.1 ERROB and NUDOB with the mesh in place of the
grid.  On a WRF grid whose cells are hex cells, with each report at a mass point (WRF's bilinear weights
are then (1, 0), hex's barycentric (1, 0, 0)) and the hex squared distance equal to WRF's planar
(i - ri)^2 + (j - rj)^2 (dx = the hex distance unit), the hex innovations and weights must be WRF's words.
WRF's fddaobs_nudob_rows does not translate to MSL (it assigns through a conditional thread/device
lvalue), so the word-for-word comparison runs on CUDA; every backend runs the analytic single-report
check on a mesh.  All observations are synthetic.
"""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from woof import backend
from woof.hex import obs_nudging as obs

cp = None
NX, NY, NZ = 11, 9, 8
NC = NX * NY
FDDAOBS = Path(obs.__file__).resolve().parents[1] / "fddaobs"


@pytest.fixture(autouse=True)
def _device():
    """Each test needs a device array module (CUDA, or the Metal facade); a CPU-only box skips."""
    global cp
    cp = pytest.importorskip("cupy")


def _wrf_module():
    if backend.active() in backend.FACADE_BACKENDS:
        pytest.skip("WRF's fddaobs_nudob_rows does not translate to MSL (a conditional thread/device lvalue); "
                    "the word-for-word comparison runs on CUDA")
    source = (FDDAOBS / "kernels.cu").read_text() + "\n" + (FDDAOBS / "forecast.cu").read_text()
    return cp.RawModule(code=source, options=obs.OPTIONS)


def _cfg(**over):
    row = dict(obs_nudge_opt=1, max_obs=100, fdda_start=0.0, fdda_end=120.0, obs_nudge_wind=1, obs_coef_wind=6e-4,
               obs_nudge_temp=1, obs_coef_temp=5e-4, obs_nudge_mois=1, obs_coef_mois=4e-4, obs_nudge_pstr=0,
               obs_coef_pstr=0.0, obs_no_pbl_nudge_uv=0, obs_no_pbl_nudge_t=0, obs_no_pbl_nudge_q=0,
               obs_sfc_scheme_horiz=0, obs_sfc_scheme_vert=0, obs_max_sndng_gap=20.0, obs_sfcfact=1.0,
               obs_sfcfacr=1.0, obs_dpsmx=7.5, obs_rinxy=3.0, obs_rinsig=0.0, obs_twindo=0.5, obs_ionf=1,
               obs_idynin=0, obs_dtramp=0.0, obs_scl_neg_qv_innov=0)
    row.update(over)
    return SimpleNamespace(**row)


def _frame(count, xtime=6.0, dtmin=0.5):
    head = np.array([0, count, 1, 0, 1], dtype=np.int32).tobytes()
    tail = np.array([xtime, 1.0, dtmin], dtype=np.float32).tobytes()
    return cp.asarray(np.frombuffer(head + tail, dtype=np.uint8).copy())


def _wrf_grid():
    return obs._record([1, NX + 1, 1, NY + 1, 1, NZ + 1, 1, NX + 1, 1, NY + 1, 1, NZ, 1, NX, 1, NY, 1, NZ],
                       ["i4"] * 18)


def _to_wrf(hex_volume, levels=NZ):
    """hex [k][cell] (cell = i-1 + NX (j-1)) -> WRF memory (i, k, j) on (NX+1) x (NY+1), edge-copied."""
    h = np.asarray(hex_volume).reshape(levels, NY, NX)
    w = np.zeros((NY + 1, levels, NX + 1), dtype=h.dtype)
    w[:NY, :, :NX] = h.transpose(1, 0, 2)
    w[NY] = w[NY - 1]
    w[:, :, NX] = w[:, :, NX - 1]
    return np.ascontiguousarray(w.reshape(-1))


def _plane_to_wrf(hex_plane):
    return _to_wrf(np.asarray(hex_plane).reshape(1, NC), levels=1)


def _columns(seed, uniform=False):
    rng = np.random.default_rng(seed)
    f = np.float32
    ht = rng.uniform(0, 600, NC).astype(f)
    dz = rng.uniform(250, 450, (NZ, 1)).astype(f) * np.ones((1, NC), f)
    zw = np.concatenate([ht[None], ht[None] + np.cumsum(dz, axis=0)]).astype(f)
    z = (f(.5) * (zw[:-1] + zw[1:])).astype(f)
    pb = (f(100000.) * np.exp(-z / f(8000.))).astype(f)
    pp = rng.normal(0, 150, (NZ, NC)).astype(f)
    fields = dict(
        tb=(f(5.) + f(3.) * np.arange(NZ, dtype=f)[:, None] + rng.normal(0, 1, (NZ, NC))).astype(f),
        qv=(f(.01) * np.exp(-np.arange(NZ, dtype=f) / f(3.))[:, None] * rng.uniform(.6, 1.2, (NZ, NC))).astype(f),
        u=rng.normal(5, 3, (NZ, NC)).astype(f), v=rng.normal(-2, 3, (NZ, NC)).astype(f))
    planes = dict(ur=rng.uniform(.8, 1.3, NC).astype(f), vr=rng.uniform(.8, 1.3, NC).astype(f),
                  tr=rng.uniform(.98, 1.02, NC).astype(f), kpbl=rng.integers(2, 5, NC).astype(np.int32),
                  pblh=rng.uniform(300, 1500, NC).astype(f), regime=rng.choice([1., 2., 4.], NC).astype(f))
    if uniform:
        for name in ("u", "v", "tb", "qv"):
            fields[name] = np.repeat(fields[name][:, :1], NC, axis=1)
        pb = np.repeat(pb[:, :1], NC, axis=1)
        pp = np.repeat(pp[:, :1], NC, axis=1)
        for name in ("ur", "vr", "tr"):
            planes[name] = np.repeat(planes[name][:1], NC)
    return dict(ht=ht, z=z, pb=pb, pp=pp, **fields, **planes)


#: Reports at WRF mass points (i, j), one-based: two surface, one single-level upper-air, one 3-level sounding.
REPORTS = [(5, 4, "surface", 0.05), (7, 6, "surface", 0.15), (4, 6, "upper", 0.08), (6, 5, "sounding", 0.1)]


def _reports(col, seed=7):
    rng = np.random.default_rng(seed)
    rows = []
    for (i, j, kind, time) in REPORTS:
        c = (i - 1) + NX * (j - 1)
        if kind == "surface":
            rows.append(dict(cell=c, i=i, j=j, rko=1.0, nlev=1.0, lev=1.0, time=time,
                             p=-888888.0, height=-888888.0))
        elif kind == "upper":
            rows.append(dict(cell=c, i=i, j=j, rko=-99.0, nlev=1.0, lev=1.0, time=time,
                             p=float(col["pb"][3, c] + col["pp"][3, c]) * .001 + .4, height=-888888.0))
        else:
            for level, k in enumerate((1, 3, 5)):
                rows.append(dict(cell=c, i=i, j=j, rko=-99.0, nlev=3.0, lev=float(level + 1), time=time,
                                 p=float(col["pb"][k, c] + col["pp"][k, c]) * .001 - .3, height=-888888.0))
    count = len(rows)
    varobs = np.zeros((count, obs.NVAR), dtype=np.float32)
    for n, row in enumerate(rows):
        varobs[n] = [rng.normal(4, 2), rng.normal(-1, 2), rng.normal(290, 3), rng.uniform(.002, .009),
                     row["p"], row["height"]]
        if row["rko"] == 1.0:
            varobs[n, 4] = float(col["pb"][0, row["cell"]] + col["pp"][0, row["cell"]]) * .001
    return rows, varobs


def _errob_pair(uniform, wind):
    module = _wrf_module()
    col = _columns(11, uniform=uniform)
    rows, varobs = _reports(col)
    count = len(rows)
    g_hex = obs.grid_record(NC, NZ)
    args_common = (np.int32(count), np.int32(obs.NVAR), np.int32(obs.NERR), np.int32(wind), np.int32(1),
                   np.int32(1), np.float32(300.), np.float32(287. / 1004.5))
    rko = np.array([r["rko"] for r in rows], dtype=np.float32)
    # hex
    k = obs.kernels()
    d = {name: cp.asarray(np.ascontiguousarray(col[name].reshape(-1))) for name in
         ("u", "v", "tb", "qv", "pb", "pp", "z", "ur", "vr", "tr", "kpbl")}
    corner = np.array([[r["cell"] + 1] * 3 for r in rows], dtype=np.int32).reshape(-1)
    bary = np.array([[1.0, 0.0, 0.0]] * count, dtype=np.float32).reshape(-1)
    near = np.array([r["cell"] + 1 for r in rows], dtype=np.int32)
    hex_out = dict(rko=cp.asarray(rko), obs=cp.asarray(varobs.reshape(-1)), err=cp.zeros(count * obs.NERR, cp.float32))
    k["hex_fddaobs_errob"](((count + 127) // 128,), (128,), (
        d["u"], d["v"], d["tb"], d["qv"], d["pb"], d["pp"], d["z"], d["ur"], d["vr"], d["tr"], d["kpbl"],
        cp.asarray(corner), cp.asarray(bary), cp.asarray(near), hex_out["rko"], hex_out["obs"], hex_out["err"],
        *args_common, g_hex, _frame(count)))
    # WRF, the same columns on the C grid (u, v staggered copies of the cell columns)
    w = {name: cp.asarray(_to_wrf(col[name])) for name in ("u", "v", "tb", "qv", "pb", "pp", "z")}
    wp = {name: cp.asarray(_plane_to_wrf(col[name])) for name in ("ur", "vr", "tr", "kpbl")}
    rio = np.array([r["i"] + .5 for r in rows], dtype=np.float32)
    rjo = np.array([r["j"] + .5 for r in rows], dtype=np.float32)
    wrf_out = dict(rko=cp.asarray(rko), obs=cp.asarray(varobs.reshape(-1)), err=cp.zeros(count * obs.NERR, cp.float32))
    module.get_function("fddaobs_errob")(((count + 127) // 128,), (128,), (
        w["u"], w["v"], w["tb"], w["qv"], w["pb"], w["pp"], w["z"], wp["ur"], wp["vr"], wp["tr"], wp["kpbl"],
        cp.asarray(rio), cp.asarray(rjo), wrf_out["rko"], wrf_out["obs"], wrf_out["err"],
        *args_common, _wrf_grid(), _frame(count)))
    return ({name: cp.asnumpy(v) for name, v in hex_out.items()},
            {name: cp.asnumpy(v) for name, v in wrf_out.items()}, count)


def test_errob_mass_innovations_equal_wrf_at_mass_points():
    hex_out, wrf_out, count = _errob_pair(uniform=False, wind=0)
    assert np.array_equal(hex_out["rko"], wrf_out["rko"]) and (hex_out["rko"] >= 1).all()
    assert np.array_equal(hex_out["obs"], wrf_out["obs"])
    he, we = hex_out["err"].reshape(count, obs.NERR), wrf_out["err"].reshape(count, obs.NERR)
    for column in (2, 3, 4, 5, 8, 9):  # theta, qv, PBL level, surface pressure, rko, model qv
        assert np.array_equal(he[:, column], we[:, column]), column


def test_errob_wind_innovations_equal_wrf_on_uniform_columns():
    """On horizontally uniform columns WRF's staggered u/v interpolation is the cell value: every word agrees."""
    hex_out, wrf_out, count = _errob_pair(uniform=True, wind=1)
    assert np.array_equal(hex_out["rko"], wrf_out["rko"])
    assert np.array_equal(hex_out["err"], wrf_out["err"])
    assert np.array_equal(hex_out["obs"], wrf_out["obs"])


@pytest.mark.parametrize("over", [
    {},
    {"obs_sfc_scheme_horiz": 1, "obs_sfc_scheme_vert": 1, "obs_no_pbl_nudge_t": 1, "obs_no_pbl_nudge_q": 1},
    {"obs_sfcfacr": 1.6, "obs_sfcfact": 1.5, "obs_twindo": 0.25, "obs_no_pbl_nudge_uv": 1},
])
def test_nudob_weights_and_tendencies_equal_wrf(over):
    module = _wrf_module()
    cfg = _cfg(**over)
    col = _columns(5)
    rows, varobs = _reports(col)
    count = len(rows)
    rng = np.random.default_rng(3)
    err = np.zeros((count, obs.NERR), dtype=np.float32)
    err[:, 0:4] = rng.normal(0, 1.5, (count, 4))
    err[:, 3] *= 1e-3
    err[:, 4] = [col["kpbl"][r["cell"]] for r in rows]
    for n, r in enumerate(rows):
        err[n, 5] = err[n, 6] = err[n, 7] = col["pb"][0, r["cell"]] * np.float32(.001)
    err[:, 9] = rng.uniform(.002, .009, count)
    rko = np.array([1.0 if r["rko"] == 1.0 else 2.0 + 1.3 * n / count for n, r in enumerate(rows)], dtype=np.float32)
    timeob = np.array([r["time"] for r in rows], dtype=np.float32)
    nlev = np.array([r["nlev"] for r in rows], dtype=np.float32)
    lev = np.array([r["lev"] for r in rows], dtype=np.float32)
    k = obs.kernels()
    g_hex = obs.grid_record(NC, NZ)
    plane = {name: cp.asarray(np.ascontiguousarray(col[name])) for name in ("pblh", "regime", "ht", "kpbl")}
    z = cp.asarray(np.ascontiguousarray(col["z"].reshape(-1)))
    pb = cp.asarray(np.ascontiguousarray(col["pb"].reshape(-1)))
    pp = cp.asarray(np.ascontiguousarray(col["pp"].reshape(-1)))
    error = cp.zeros(1, dtype=cp.uint32)
    # Reach lists: every cell, every report, WRF's planar squared distance in grid lengths (dx = the hex unit).
    starts = [n for n, r in enumerate(rows) if r["lev"] == 1.0]
    start = np.arange(NC + 1, dtype=np.int32) * len(starts)
    reach = np.tile(np.array(starts, dtype=np.int32), NC)
    rsq = np.empty(NC * len(starts), dtype=np.float32)
    for c in range(NC):
        i, j = c % NX + 1, c // NX + 1
        for m, n in enumerate(starts):
            rx = np.float32(j) - np.float32(rows[n]["j"])
            ry = np.float32(i) - np.float32(rows[n]["i"])
            rsq[c * len(starts) + m] = rx * rx + ry * ry
    frame = _frame(count, xtime=6.0, dtmin=0.5)
    wrf = module.get_function("fddaobs_nudob_rows")
    gw = _wrf_grid()
    common = dict(rko=cp.asarray(rko), timeob=cp.asarray(timeob), obs=cp.asarray(varobs.reshape(-1)),
                  err=cp.asarray(err.reshape(-1)), nlev=cp.asarray(nlev), lev=cp.asarray(lev))
    wz, wpb, wpp = (cp.asarray(_to_wrf(col[name])) for name in ("z", "pb", "pp"))
    wplane = {name: cp.asarray(_plane_to_wrf(col[name])) for name in ("pblh", "regime", "ht", "kpbl")}
    for suffix, ivars, coef in (("uv", (1, 2), cfg.obs_coef_wind), ("t", (3,), cfg.obs_coef_temp),
                                ("q", (4,), cfg.obs_coef_mois)):
        vif = cp.asarray([50., 50., 50., 50., -5000., 50.], dtype=cp.float32)
        h1, h2, top = cp.zeros(NC, cp.float32), cp.zeros(NC, cp.float32), cp.zeros(NC, cp.int32)
        k["fddaobs_vif"](((NC + 127) // 128, 1), (128, 1), (vif, plane["regime"], plane["pblh"], h1, h2,
                         np.float32(3000.), np.float32(50.), np.float32(50.), g_hex, error, z, plane["ht"], top))
        wh1, wh2, wtop = (cp.zeros((NX + 1) * (NY + 1), d) for d in (cp.float32, cp.float32, cp.int32))
        module.get_function("fddaobs_vif")(((NX + 15) // 16, (NY + 15) // 16), (16, 16), (
            vif, wplane["regime"], wplane["pblh"], wh1, wh2, np.float32(3000.), np.float32(50.), np.float32(50.),
            gw, error, wz, wplane["ht"], wtop))
        for ivar in ivars:
            controls = obs.controls_record(cfg, ivar, 30.0)
            hex_t = cp.zeros(NZ * NC, cp.float32)
            hex_s = cp.zeros(NZ * NC * obs.NVAR, cp.float32)
            k["hex_fddaobs_nudob"](((NC + 31) // 32, (NZ + 3) // 4), (32, 4), (
                hex_t, hex_s, cp.ones(NC, cp.float32), cp.asarray(start), cp.asarray(reach), cp.asarray(rsq),
                common["rko"], common["timeob"], common["obs"], common["err"], common["nlev"], common["lev"],
                pb, pp, plane["kpbl"], h1, h2, plane["ht"], z, np.int32(obs.NVAR), np.int32(obs.NERR),
                np.int32(ivar), controls, g_hex, frame, top))
            volume = (NX + 1) * (NY + 1) * NZ
            wrf_t = cp.zeros(volume, cp.float32)
            wrf_s = cp.zeros(volume * obs.NVAR, cp.float32)
            one = cp.ones((NX + 1) * (NY + 1), cp.float32)
            # NUDOB measures from the field's own points (ra = rio - gx, rb = rjo - gy): placing the report
            # at (i0 + gx, j0 + gy) makes WRF's u/v/mass distances the hex cell distances of (i0, j0).
            gx, gy = (0.0 if ivar == 1 else .5), (0.0 if ivar == 2 else .5)
            rio = cp.asarray(np.array([r["i"] + gx for r in rows], dtype=np.float32))
            rjo = cp.asarray(np.array([r["j"] + gy for r in rows], dtype=np.float32))
            wrf(((NX + 31) // 32, NY, (NZ + 3) // 4), (32, 1, 4), (
                wrf_t, wrf_s, cp.zeros(1, cp.float32), cp.zeros(1, cp.float32), one,
                cp.zeros(NZ, cp.float32), cp.ones(NZ, cp.float32), one, one, rio, rjo,
                common["rko"], common["timeob"], common["obs"], common["err"], common["nlev"], common["lev"],
                wpb, wpp, wplane["kpbl"], wh1, wh2, wplane["ht"], wz, cp.zeros(volume, cp.float32),
                np.int32(count), np.int32(obs.NVAR), np.int32(obs.NERR), np.int32(ivar), controls, gw, frame, wtop))
            hs = cp.asnumpy(hex_s).reshape(NZ, NY, NX, obs.NVAR)[..., ivar - 1]
            ws = cp.asnumpy(wrf_s).reshape(NY + 1, NZ, NX + 1, obs.NVAR)[..., ivar - 1].transpose(1, 0, 2)[:, :NY, :NX]
            ht_ = cp.asnumpy(hex_t).reshape(NZ, NY, NX)
            wt_ = cp.asnumpy(wrf_t).reshape(NY + 1, NZ, NX + 1).transpose(1, 0, 2)[:, :NY, :NX]
            # WRF's own boundary rows/columns (inest == 1): it skips rows j <= 2 and j >= jde - 1 (winds) or
            # jde - 2 (mass), and the column i = 1; the interior is the comparison.
            j_hi = NY - 2 if ivar >= 3 else NY - 1
            interior = (slice(None), slice(2, j_hi), slice(1, NX))
            assert np.count_nonzero(hs[interior]) > 20, (suffix, ivar)
            assert np.array_equal(hs[interior], ws[interior]), (suffix, ivar)
            assert np.array_equal(ht_[interior], wt_[interior]), (suffix, ivar)
            assert coef > 0


def test_single_surface_report_weight_is_wrf_cressman_on_the_sphere():
    """One surface report on a mesh: savwt at the lowest level is tw * wh * innovation, wh = (R^2 - d^2)/(R^2 + d^2)."""
    from test_hex_obs_nudging import _mesh

    mesh = _mesh()
    locator = obs.MeshLocator(mesh)
    lat, lon = np.array([40.05]), np.array([-100.1])
    cfg = _cfg(obs_rinxy=60.0)
    start, report, rsq = locator.reach(lat, lon, cfg.obs_rinxy * 1000.0 * 2.0)
    nc, nz = mesh.n_cells, 4
    f = np.float32
    pb = np.repeat((f(100000.) * np.exp(-np.arange(nz, dtype=f)[:, None] * f(0.1))), nc, axis=1).astype(f)
    z = np.repeat(np.arange(nz, dtype=f)[:, None] * f(30.) + f(15.), nc, axis=1)
    err = np.zeros(obs.NERR, dtype=f)
    err[2] = f(1.5)
    err[4] = f(2.)
    err[5] = err[6] = err[7] = pb[0, 0] * f(.001)
    varobs = np.zeros(obs.NVAR, dtype=f)
    varobs[4] = pb[0, 0] * f(.001)
    k = obs.kernels()
    g = obs.grid_record(nc, nz)
    h1, h2, top = cp.full(nc, 50., cp.float32), cp.full(nc, 100., cp.float32), cp.full(nc, nz, cp.int32)
    tend = cp.zeros(nz * nc, cp.float32)
    savwt = cp.zeros(nz * nc * obs.NVAR, cp.float32)
    k["hex_fddaobs_nudob"](((nc + 31) // 32, (nz + 3) // 4), (32, 4), (
        tend, savwt, cp.ones(nc, cp.float32), cp.asarray(start.astype(np.int32)),
        cp.asarray(report.astype(np.int32)), cp.asarray(rsq), cp.ones(1, cp.float32), cp.asarray([f(0.1)]),
        cp.asarray(varobs), cp.asarray(err), cp.ones(1, cp.float32), cp.ones(1, cp.float32),
        cp.asarray(pb.reshape(-1)), cp.zeros(nz * nc, cp.float32), cp.full(nc, 2, cp.int32), h1, h2,
        cp.zeros(nc, cp.float32), cp.asarray(z.reshape(-1)), np.int32(obs.NVAR), np.int32(obs.NERR), np.int32(3),
        obs.controls_record(cfg, 3, 30.0), g, _frame(1, xtime=6.0, dtmin=0.0), top))
    lowest = cp.asnumpy(savwt).reshape(nz, nc, obs.NVAR)[0, :, 2]
    d = obs.great_circle_m(locator.radius, np.radians(lat[0]), np.radians(lon[0]),
                           mesh.arrays["latCell"], mesh.arrays["lonCell"]) / 1000.0
    radius = cfg.obs_rinxy * cfg.obs_sfcfacr
    expected = np.maximum(0.0, (radius ** 2 - d ** 2) / (radius ** 2 + d ** 2)) * 1.5
    assert np.allclose(lowest, expected, rtol=2e-5, atol=1e-6)
    assert (lowest > 0).sum() > 10
    np.testing.assert_allclose(cp.asnumpy(tend).reshape(nz, nc)[0], cfg.obs_coef_temp * lowest, rtol=1e-6)
