"""woof.da.var.radiance: the clear-sky radiance obs term (plan V4.3, section 13; gate GR-V1).

What is asserted, and why it is enough:

- **Adjoint (GR-V1):** ``<H' x, y> = <x, H'^T y>`` to 1e-13 in binary64 for
  the K-matrix term alone and through ``VarCost``'s whole ``H' U`` chain,
  with random corners, weights and Jacobians.
- **The TL is WRFDA's:** one location checked against a dense sum written
  from ``da_transform_xtoy_crtm`` (top layer dropped, humidity scaled by
  ``1000 / (1 - q)^2`` and zero above 75 hPa).
- **Gradient:** ``da_check_gradient``'s Taylor ratio within 1e-6 of 1 with
  the radiance term in J.
- **Thinning is GSI's:** one observation per box, the one nearest the box
  centre, the first on a tie; independent of input order only through that
  tie rule.
- **Surface type is ``da_detsurtyp``'s;** QC is ``da_qc_goesabi``'s and
  ``da_qc_seviri``'s on hand-made cases.
- **The door:** :func:`radiance_term` with a linear stand-in for CRTM gives
  ``d = O - H(x_b)`` and the stand-in's Jacobians; MW footprints shaped like
  WIS2's plug in without the reader.
CRTM itself is not needed here (its own tests are tests/test_rtm_crtm*.py).
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pytest

from woof.da.var import radiance as rad
from woof.da.var.cost import VarCost, check_gradient
from woof.da.var.obs_wrfda import LambertWRFDA

from test_var_cost import case, synthetic_obs

WHEN = datetime(2026, 10, 28, 12, tzinfo=timezone.utc)


def _serial(a, b):
    acc = 0.0
    for x, y in zip(np.ravel(a), np.ravel(b)):
        acc = acc + float(x) * float(y)
    return acc


def _batch(rng, nx, ny, nz, n=40, nchan=3, sensor="abi_g16"):
    i0 = rng.integers(0, nx - 1, n)
    j0 = rng.integers(0, ny - 1, n)
    dx, dy = rng.uniform(0, 1, n), rng.uniform(0, 1, n)
    h = (i0, j0, dx, dy, 1.0 - dx, 1.0 - dy)
    p = np.sort(rng.uniform(10.0, 1000.0, (n, nz)), axis=1)
    return rad.RadianceBatch(
        sensor=sensor, channels=np.arange(8, 8 + nchan), h=h, kt=rng.uniform(-0.2, 0.5, (n, nchan, nz)),
        kr=rng.uniform(-3.0, 0.5, (n, nchan, nz)), q=rng.uniform(1e-5, 1.5e-2, (n, nz)), pressure_hpa=p,
        obs=rng.uniform(220, 260, (n, nchan)), hx=rng.uniform(220, 260, (n, nchan)),
        inv=rng.standard_normal((n, nchan)), qc=np.where(rng.uniform(size=(n, nchan)) < 0.1, -1, 1),
        error=rng.uniform(1.0, 3.0, (n, nchan)))


def _xa(rng, nx, ny, nz):
    return {"t": rng.standard_normal((nx, ny, nz)), "q": rng.standard_normal((nx, ny, nz)) * 1e-3}


# ------------------------------------------------------------------ the term


def test_term_adjoint_to_1e13():
    rng = np.random.default_rng(3)
    nx, ny, nz = 9, 8, 7
    term = rad.RadianceTerm([_batch(rng, nx, ny, nz), _batch(rng, nx, ny, nz, n=17, nchan=2, sensor="ahi_himawari9")],
                            (nx, ny, nz))
    x = _xa(rng, nx, ny, nz)
    y = rng.standard_normal(term.m)
    lhs = _serial(term.tl(x), y)
    ad = term.ad(y)
    rhs = _serial(x["t"], ad["t"]) + _serial(x["q"], ad["q"])
    rel = abs(lhs - rhs) / abs(lhs)
    print(f"radiance term adjoint: rel={rel:.3e}")
    assert rel <= 1e-13
    assert [g[0] for g in term.jo_split()[0][1]] == ["abi_g16", "ahi_himawari9"]
    assert term.jo_split()[0][0] == "radiance"


def test_tl_is_wrfdas_k_matrix_sum():
    rng = np.random.default_rng(4)
    nx, ny, nz = 6, 5, 8
    b = _batch(rng, nx, ny, nz, n=1, nchan=2)
    b.pressure_hpa[0] = np.r_[20.0, 50.0, np.linspace(100.0, 1000.0, nz - 2)]  # the top two layers are above 75 hPa
    term = rad.RadianceTerm([b], (nx, ny, nz))
    x = _xa(rng, nx, ny, nz)
    i, j, dx, dy, dxm, dym = (a[0] for a in b.h)
    want = np.zeros(2)
    for c in range(2):
        for layer in range(1, nz):  # kts+1 .. kte, top first
            lev = nz - 1 - layer
            col = lambda f: (dym * (dxm * f[i, j, lev] + dx * f[i + 1, j, lev])
                             + dy * (dxm * f[i, j + 1, lev] + dx * f[i + 1, j + 1, lev]))
            p = b.pressure_hpa[0, layer]
            fr = 0.0 if p < 75.0 else 1000.0 / (1.0 - b.q[0, layer]) ** 2
            want[c] += b.kt[0, c, layer] * col(x["t"]) + b.kr[0, c, layer] * fr * col(x["q"])
    np.testing.assert_allclose(term.tl(x), want, rtol=1e-13)
    # the top layer and the q above 75 hPa never reach the analysis
    ad = term.ad(np.ones(2))
    assert np.all(ad["t"][..., nz - 1] == 0.0) and np.all(ad["q"][..., nz - 2:] == 0.0)


def test_chain_adjoint_and_gradient_with_radiances():
    b, xb, conv = case()
    rng = np.random.default_rng(6)
    batch = _batch(rng, *xb["t"].shape, n=30)
    term = rad.RadianceTerm([batch], xb["t"].shape)
    cost = VarCost.reference(b, xb, [conv, term])
    # errors scaled to the increments U makes, as synthetic_obs scales its own
    scale = float(np.std(term.tl(cost.u.apply(rng.standard_normal(cost.cv_size))["xa"])))
    batch.error = batch.error * scale
    term = rad.RadianceTerm([batch], xb["t"].shape)
    cost = VarCost.reference(b, xb, [conv, term])
    v = rng.standard_normal(cost.cv_size)
    y = [rng.standard_normal(conv.d.size), rng.standard_normal(term.m)]
    hv = cost.vtoy(v)
    lhs = _serial(hv[0], y[0]) + _serial(hv[1], y[1])
    rhs = _serial(v, cost.vtoy_adj(y))
    assert abs(lhs - rhs) / abs(lhs) <= 1e-13
    j = cost.calculate_j(v, None, 1)
    assert set(j.jo_types) == {"sound", "synop", "radiance"}
    assert list(j.jo_types)[-1] == "radiance"
    chk = check_gradient(cost, rng.uniform(-0.5, 0.5, cost.cv_size))
    print(f"Taylor: best={chk['best']!r} digits={chk['digits']}")
    assert abs(chk["best"] - 1.0) < 1e-6


# ------------------------------------------------------------------ thinning


def test_gsi_thin_keeps_the_box_centre_and_first_on_ties():
    dlat, dlon = np.meshgrid(np.linspace(30.0, 40.0, 11), np.linspace(-100.0, -88.0, 13), indexing="ij")
    lat = np.array([35.0, 35.01, 35.0, 31.0])
    lon = np.array([-95.0, -95.0, -95.0, -99.0])
    kept = rad.gsi_thin(lat, lon, 60.0, dlat, dlon)
    # 0, 1 and 2 share a box; 0 and 2 tie and 0 is first; 3 is alone
    assert 3 in kept and 2 not in kept and kept.size == 2
    # a much finer mesh keeps everyone distinct except the exact duplicate
    fine = rad.gsi_thin(lat, lon, 0.5, dlat, dlon)
    assert fine.tolist() == [0, 1, 3]


def test_gsi_thin_one_per_box_over_a_dense_swath():
    rng = np.random.default_rng(8)
    dlat, dlon = np.meshgrid(np.linspace(30.0, 40.0, 11), np.linspace(-100.0, -88.0, 13), indexing="ij")
    lat, lon = rng.uniform(30, 40, 5000), rng.uniform(-100, -88, 5000)
    kept = rad.gsi_thin(lat, lon, 60.0, dlat, dlon)
    # ~ (1100 km x 1100 km) / 60 km boxes; never more than one per box
    assert 150 < kept.size < 450
    assert np.all(np.diff(kept) > 0)
    np.testing.assert_array_equal(kept, rad.gsi_thin(lat, lon, 60.0, dlat, dlon))


# ------------------------------------------------------------------ surface and columns


def _h(n=1, dx=0.25, dy=0.5):
    return (np.zeros(n, np.int64), np.zeros(n, np.int64), np.full(n, dx), np.full(n, dy), np.full(n, 1 - dx),
            np.full(n, 1 - dy))


@pytest.mark.parametrize("land, ice, snow, want", [
    ([[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]], 0),
    ([[1, 1], [1, 1]], [[0, 0], [0, 0]], [[0, 0], [0, 0]], 2),
    ([[1, 1], [1, 1]], [[0, 0], [0, 0]], [[1, 1], [1, 1]], 3),
    ([[0, 0], [0, 0]], [[1, 1], [1, 1]], [[0, 0], [0, 0]], 1),
    ([[1, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]], 4),   # mostly sea
    ([[1, 1], [0, 1]], [[0, 0], [0, 0]], [[0, 0], [0, 0]], 6),   # mostly land
])
def test_detsurtyp(land, ice, snow, want):
    isflg, fr = rad._detsurtyp(_h(), np.array(land, float), np.array(ice, float), np.array(snow, float))
    assert int(isflg[0]) == want
    assert fr["sea"][0] + fr["ice"][0] + fr["land"][0] + fr["snow"][0] == pytest.approx(1.0)


def _xb_columns(nx=4, ny=4, nz=10):
    znw = np.linspace(1.0, 0.0, nz + 1)
    znu = 0.5 * (znw[:-1] + znw[1:])
    ptop, ps = 5000.0, 100000.0
    p = np.broadcast_to(ptop + znu * (ps - ptop), (nx, ny, nz)).copy()
    t = np.broadcast_to(np.linspace(295.0, 210.0, nz), (nx, ny, nz)).copy()
    q = np.broadcast_to(np.geomspace(1.2e-2, 3e-6, nz), (nx, ny, nz)).copy()
    return {"p": p, "t": t, "q": q, "psfc": np.full((nx, ny), ps), "ptop": ptop, "sigmah": znu, "sigmaf": znw,
            "terr": np.zeros((nx, ny)), "lat": np.zeros((nx, ny)), "lon": np.zeros((nx, ny))}


def _surface(nx=4, ny=4, nz=10, qcloud=None):
    s = {"TSK": np.full((nx, ny), 290.0), "U10": np.full((nx, ny), 3.0), "V10": np.full((nx, ny), -4.0),
         "LANDMASK": np.zeros((nx, ny)), "XICE": np.zeros((nx, ny)), "SNOWC": np.zeros((nx, ny))}
    if qcloud is not None:
        s["QCLOUD"] = np.full((nx, ny, nz), qcloud)
    return s


def test_background_columns_are_wrfdas_crtm_atmosphere():
    xb = _xb_columns()
    cols = rad.background_columns(xb, _surface(qcloud=1e-4), _h(2))
    assert cols.ok.all()
    np.testing.assert_allclose(cols.level_pressure_hpa[:, 0], 50.0)
    np.testing.assert_allclose(cols.level_pressure_hpa[:, -1], 1000.0)
    # linear p in sigma: the interfaces are exactly the full levels
    np.testing.assert_allclose(cols.level_pressure_hpa[0], (5000.0 + xb["sigmaf"][::-1] * 95000.0) / 100.0,
                               rtol=1e-12)
    np.testing.assert_allclose(cols.temperature_k[0], xb["t"][0, 0, ::-1])
    q = xb["q"][0, 0, ::-1]
    np.testing.assert_allclose(cols.mixing_ratio_gkg[0], 1000.0 * q / (1.0 - q))
    assert cols.wind_speed_ms[0] == pytest.approx(5.0)
    assert cols.isflg.tolist() == [0, 0]
    # clwp over p >= 100 hPa: 1e-4 kg/kg * (1000 - 100) hPa / g, to within one layer
    assert 0.85 < cols.clwp_kgm2[0] / (1e-4 * 90000.0 / 9.81) < 1.0


# ------------------------------------------------------------------ QC


def _cols(n, isflg=0, clwp=0.0):
    return SimpleNamespace(ok=np.ones(n, bool), isflg=np.full(n, isflg), clwp_kgm2=np.full(n, clwp))


def test_qc_goesabi_clear_sky():
    chans = np.arange(7, 17)
    n = 5
    obs = np.full((n, 10), 250.0)
    hx = np.full((n, 10), 249.0)
    inv = obs - hx
    inv[1, 1] = 16.0       # gross 15 K
    inv[2, 2] = 3.1 * 1.92  # 3 sigma on channel 9
    qc, err, counts = rad.qc_wrfda("abi_g16", chans, obs, hx, inv, _cols(n), rad.RadianceQC())
    good = qc == rad.QC_GOOD
    # only the WV channels 8, 9, 10 are used
    assert good[0].tolist() == [c in (8, 9, 10) for c in chans]
    assert not good[1, 1] and not good[2, 2] and good[3, 1]
    assert err[0, 1] == 1.79 and err[0, 0] == rad.REJECTED_ERROR_K
    assert counts["gross_15k"] == 1 and counts["gross_3sigma"] == 1
    qc, _, counts = rad.qc_wrfda("abi_g16", chans, obs, hx, inv, _cols(n, isflg=5), rad.RadianceQC())
    assert (qc == rad.QC_BAD).all() and counts["mixed_surface"] == 3 * n
    qc, _, _ = rad.qc_wrfda("abi_g16", chans, obs, hx, inv, _cols(n, clwp=0.3), rad.RadianceQC())
    assert (qc == rad.QC_BAD).all()


def test_qc_seviri_has_no_surface_test_and_window_screen_is_opt_in():
    chans = np.arange(4, 12)
    obs = np.full((2, 8), 250.0)
    inv = np.zeros((2, 8))
    inv[1, chans.tolist().index(9)] = -6.0
    qc, _, _ = rad.qc_wrfda("seviri_m10", chans, obs, obs, inv, _cols(2, isflg=6), rad.RadianceQC())
    assert ((qc[0] == 1) == np.isin(chans, (5, 6, 7, 10))).all() and (qc[1] == qc[0]).all()
    qc, _, counts = rad.qc_wrfda("seviri_m10", chans, obs, obs, inv, _cols(2), rad.RadianceQC(window_screen_k=4.0))
    assert (qc[1] == -1).all() and (qc[0] == qc[0]).all() and counts["window_screen"] == 4


def test_reader_screens_and_errors_pass_through():
    chans = np.array([1, 2, 3])
    obs = np.full((2, 3), 230.0)
    rqc = np.array([[0, -1, 0], [0, 0, 0]])
    rerr = np.full((2, 3), 0.5)
    qc, err, _ = rad.qc_wrfda("amsua_metop-b", chans, obs, obs, np.zeros((2, 3)), _cols(2),
                              rad.RadianceQC(), rqc, rerr)
    assert qc.tolist() == [[1, -1, 1], [1, 1, 1]]
    assert err[0, 0] == 0.5 and err[0, 1] == rad.REJECTED_ERROR_K


# ------------------------------------------------------------------ the door


class _LinearCRTM:
    """A stand-in with CRTM's ``simulate``: fixed K, and BT near 240 K that moves with the profile."""

    def simulate(self, sensor, profiles, geometry, when, channels=None, jacobian=True):
        n, nz = profiles.temperature_k.shape
        nchan = len(channels)
        kt = np.broadcast_to(np.linspace(0.0, 0.1, nz), (n, nchan, nz)).copy()
        kr = np.broadcast_to(-np.linspace(0.0, 0.5, nz), (n, nchan, nz)).copy()
        bt = 240.0 + 1e-3 * (np.einsum("ncl,nl->nc", kt, profiles.temperature_k - 250.0)
                             + np.einsum("ncl,nl->nc", kr, profiles.water_vapour_gkg))
        return SimpleNamespace(brightness_temperature=bt, k_temperature=kt, k_water_vapour=kr)


def _proj(nx, ny):
    return LambertWRFDA(cen_lat=35.0, cen_lon=-95.0, stand_lon=-95.0, truelat1=30.0, truelat2=60.0,
                        moad_cen_lat=35.0, dx=30000.0, ide=nx, jde=ny)


def test_radiance_term_door_with_a_linear_operator():
    nx, ny, nz = 20, 18, 10
    proj = _proj(nx, ny)
    xb = _xb_columns(nx, ny, nz)
    # observations at known grid positions: invert llxy by Newton steps
    pts = [(3.25, 4.5), (10.5, 9.75), (15.5, 2.25)]
    lats, lons = [], []
    for x, y in pts:
        # Newton on llxy to find (lat, lon) at grid point (x, y)
        la, lo = 35.0, -95.0
        for _ in range(30):
            fx, fy = proj.llxy(la, lo)
            ex, ey = proj.llxy(la + 1e-4, lo), proj.llxy(la, lo + 1e-4)
            jm = np.array([[(ex[0] - fx) / 1e-4, (ey[0] - fx) / 1e-4], [(ex[1] - fy) / 1e-4, (ey[1] - fy) / 1e-4]])
            step = np.linalg.solve(jm, [x - fx, y - fy])
            la, lo = la + step[0], lo + step[1]
        lats.append(la)
        lons.append(lo)
    xb["lat"], xb["lon"] = np.linspace(28, 42, nx * ny).reshape(nx, ny), np.linspace(-105, -85, nx * ny).reshape(nx, ny)
    n = len(pts) + 1
    rec = rad.RadianceObs(sensor="abi_g16", channels=[8, 9, 10], bt=np.full((n, 3), 240.0),
                          lat=lats + [80.0], lon=lons + [0.0], sensor_zenith=[40.0, 40.0, 70.0, 40.0],
                          sensor_azimuth=0.0, sensor_scan=0.0, solar_zenith=100.0, solar_azimuth=0.0, time=WHEN)
    term, report = rad.radiance_term([rec], xb, _surface(nx, ny, nz), proj, _LinearCRTM(),
                                     settings=rad.RadianceQC(thinning=False))
    counts = report["sensors"]["abi_g16"]
    assert counts["zenith"] == 1 and counts["outside_domain"] == 1 and counts["locations"] == 2
    b = term.batches[0]
    np.testing.assert_allclose(b.h[2], [0.25, 0.5], atol=1e-6)
    np.testing.assert_allclose(b.h[3], [0.5, 0.75], atol=1e-6)
    assert term.d.tolist() == (b.obs - b.hx).ravel().tolist()
    assert term.m == 6 and term.mask.all()


def test_from_footprints_duck_types_wis2_mw():
    fp = SimpleNamespace(
        sensor=np.array(["amsua_metop-b", "atms_n20", "amsua_metop-b"]), channels=np.array([5, 6, 7]),
        bt=np.full((3, 3), 230.0), lat=[10.0, 11.0, 12.0], lon=[0.0, 1.0, 2.0], fov=[1, 48, 30],
        scan_angle=[48.0, 0.0, -1.0], sensor_zenith=[57.0, 0.0, 1.0], sensor_azimuth=[0.0] * 3,
        solar_zenith=[30.0] * 3, solar_azimuth=[0.0] * 3,
        time=np.array(["2026-10-28T11:50", "2026-10-28T11:40", "2026-10-28T12:10"], dtype="datetime64[s]"))
    recs = rad.from_footprints(fp)
    assert [r.sensor for r in recs] == ["amsua_metop-b", "atms_n20"]
    assert recs[0].n == 2 and recs[0].scanpos.tolist() == [1.0, 30.0]
    assert recs[1].time == datetime(2026, 10, 28, 11, 40, tzinfo=timezone.utc)


def test_imager_scene_split_by_sensor():
    from woof.obs.imager_scene import ImagerScene, Platform

    ny, nx = 3, 4
    lat, lon = np.meshgrid(np.linspace(30, 32, ny), np.linspace(-100, -97, nx), indexing="ij")
    pidx = np.array([[0, 0, 1, 1]] * ny)
    vals = {nm: np.full((ny, nx), 240.0) for nm in (6200, 6900, 7330, 11200)}
    plats = [Platform(name="GOES-19", sector="full", family="abi", crtm_sensor="abi_g19",
                      sub_longitude_deg=-75.2),
             Platform(name="Meteosat-10", sector="full", family="seviri", crtm_sensor="seviri_m10",
                      sub_longitude_deg=0.0)]
    z = np.zeros((ny, nx))
    scene = ImagerScene(time=WHEN, lat=lat, lon=lon, values=vals, platforms=plats, platform_index=pidx,
                        view_zenith=z + 30, view_azimuth=z, scan_angle=z + 5, solar_zenith=z + 100,
                        solar_azimuth=z, flags=np.zeros((ny, nx), np.int64))
    recs = rad.from_imager_scene(scene)
    by = {r.sensor: r for r in recs}
    assert by["abi_g19"].channels.tolist() == [8, 9, 10, 14] and by["abi_g19"].n == 6
    assert by["seviri_m10"].channels.tolist() == [5, 6, 9]
