"""The L3a climatological ionosphere (woof.spacewx.ionosphere, igrf, indices).

Synthetic coefficient files stand in for the CCIR/URSI maps and IGRF so that
every test runs without a download: the readers, the host contraction, the
NumPy reference and the device kernel (CUDA and Metal) are exercised on them.
The device kernel is graded against the float64 reference to float32
rounding.  Real-coefficient checks run only where the pinned files already
sit in the space-weather cache (``$WOOF_SPACEWX_CACHE``); the graded
comparison with GNSS TEC maps is ``tools/spacewx_gim_validate.py`` (numbers
in docs/spacewx.md).
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
import gzip
import hashlib
import io
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from woof.spacewx import assets, igrf, indices
from woof.spacewx import ionosphere as ion

from conftest import requires_gpu  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- fixtures

def _asc(values) -> str:
    """Write values as Fortran ``(1X,4E15.8)``."""
    lines = []
    for i in range(0, len(values), 4):
        lines.append(" " + "".join(f"{v:15.8E}" for v in values[i:i + 4]))
    return "\n".join(lines) + "\n"


def _synthetic_maps(month, seed=0):
    rng = np.random.default_rng(seed + month)
    f2 = rng.normal(0.0, 0.03, (ion.NJ_F2, ion.NK_F2, 2))
    m3 = rng.normal(0.0, 0.01, (ion.NJ_M3, ion.NK_M3, 2))
    f2[0, 0] = (5.0 + 0.1 * month, 9.0 + 0.1 * month)  # constant term per activity level
    m3[0, 0] = (3.1, 2.9)
    f2[1, 0] = (-1.0, -1.5)    # sin(h): a diurnal cycle
    f2[0, 1] = (0.8, 1.2)      # sin(modip): a latitude structure
    return f2, m3


def _loader(seed=0, ursi_offset=0.0):
    def load(asset):
        month = int(asset.name[4:6]) - 10
        f2, m3 = _synthetic_maps(month, seed)
        if asset.name.startswith("ursi"):
            f2 = f2.copy()
            f2[0, 0] += ursi_offset
            return _asc(f2.ravel(order="F"))
        return _asc(np.concatenate([f2.ravel(order="F"), m3.ravel(order="F")]))
    return load


_DIPOLE = """# synthetic: an axial dipole and its secular variation
c/s  main main SV
g/h n m 2020.0 2025.0 2025-30
g  1  0  -30000.0 -29000.0 100.0
g  1  1  0.0 0.0 0.0
h  1  1  0.0 0.0 0.0
"""


def _dipole():
    return igrf.parse(_DIPOLE, "synthetic")


WHEN = datetime(2024, 5, 10, 18, 0)


def _iono(lat, lon, **kw):
    kw.setdefault("igrf_model", _dipole())
    kw.setdefault("map_loader", _loader())
    return ion.Ionosphere(lat, lon, when=WHEN, **kw)


# ---------------------------------------------------------------- IGRF

def test_igrf_axial_dipole_gives_the_dipole_inclination():
    model = _dipole()
    lat = np.array([-80.0, -45.0, -10.0, 0.0, 10.0, 45.0, 80.0])
    lon = np.linspace(-170, 170, lat.size)
    inc = igrf.inclination(model, 2022.0, lat, lon, 300.0)
    r, theta, _ = igrf._geodetic_to_geocentric(lat, 300.0)
    lat_gc = 90.0 - np.rad2deg(theta)
    # Geocentric tan I = 2 tan(lat); the geodetic frame tilts I by the
    # latitude difference (< 0.2 deg).
    expected = np.rad2deg(np.arctan(2.0 * np.tan(np.deg2rad(lat_gc))))
    assert np.all(np.abs(inc - expected) < 0.25)
    assert np.all(np.sign(inc[lat != 0]) == np.sign(lat[lat != 0]))
    x, y, z = igrf.field(model, 2022.0, 0.0, 0.0, 0.0)
    assert abs(float(y)) < 1e-9 and float(x) > 0.0          # northward on the equator
    # |g10(2022.0)| = 29600 nT at the reference radius, falling as r^-3.
    assert float(x) == pytest.approx(29600.0 * (6371.2 / igrf.WGS84_A_KM) ** 3, rel=1e-9)


def test_igrf_interpolates_epochs_extrapolates_sv_and_refuses_outside():
    model = _dipole()
    g, _ = model.coefficients(2022.5)
    assert g[1, 0] == pytest.approx(-29500.0)
    g, _ = model.coefficients(2027.0)
    assert g[1, 0] == pytest.approx(-28800.0)
    for year in (2019.9, 2030.1):
        with pytest.raises(igrf.IgrfError):
            model.coefficients(year)


def test_modip_and_dip_latitude_formulas():
    assert igrf.modip(0.0, 30.0) == pytest.approx(0.0)
    assert float(igrf.dip_latitude(45.0)) == pytest.approx(np.rad2deg(np.arctan(0.5)))
    inc, lat = 60.0, 50.0
    expected = np.rad2deg(np.arctan(np.deg2rad(inc) / np.sqrt(np.cos(np.deg2rad(lat)))))
    assert float(igrf.modip(inc, lat)) == pytest.approx(expected)


def test_igrf_parser_refuses_a_table_without_header():
    with pytest.raises(igrf.IgrfError):
        igrf.parse("g 1 0 -30000 1\n")


# ---------------------------------------------------------------- maps

def test_asc_reader_and_fortran_order_layout():
    values = np.arange(1, 7, dtype=float) * 0.5
    assert np.array_equal(ion.parse_asc(_asc(values)), values)
    f2, m3 = ion.read_maps(5, loader=_loader())
    ref_f2, ref_m3 = _synthetic_maps(5)
    assert f2.shape == (13, 76, 2) and m3.shape == (9, 49, 2)
    assert np.allclose(f2, ref_f2, rtol=1e-7, atol=1e-12)
    assert np.allclose(m3, ref_m3, rtol=1e-7, atol=1e-12)
    ursi, m3u = ion.read_maps(5, "ursi", loader=_loader(ursi_offset=0.5))
    assert ursi[0, 0, 0] == pytest.approx(ref_f2[0, 0, 0] + 0.5)
    assert np.array_equal(m3u, m3)                       # M(3000)F2 is always CCIR's


def test_map_assets_are_pinned_per_month_and_family():
    names = {ion.map_asset(f, m).name for f in ion.FAMILIES for m in range(1, 13)}
    assert len(names) == 24
    a = ion.map_asset("ccir", 1)
    assert a.name == "ccir11.asc" and ion.PYIRI_COMMIT in a.url and len(a.sha256) == 64
    for bad in (("iri", 1), ("ccir", 13)):
        with pytest.raises(ion.IonosphereError):
            ion.map_asset(*bad)


def test_expansion_sizes_and_diurnal_series():
    lat = np.array([10.0, -30.0])
    assert ion.geographic(ion.Q_F2, lat, lat, lat).shape == (76, 2)
    assert ion.geographic(ion.Q_M3, lat, lat, lat).shape == (49, 2)
    d = ion.diurnal(5, 12.0)   # h = 0
    assert np.allclose(d, [1.0, 0.0, 1.0, 0.0, 1.0])
    d = ion.diurnal(5, 18.0)   # h = 90 deg
    assert np.allclose(d, [1.0, 1.0, 0.0, 0.0, -1.0], atol=1e-12)


def test_solar_index_conversions_and_month_weights():
    for r in (0.0, 50.0, 150.0):
        f = 63.75 + 0.728 * r + 8.9e-4 * r * r
        assert ion.f107_to_r12(f) == pytest.approx(r, abs=1e-9)
    assert ion.f107_to_r12(50.0) == pytest.approx(0.0)
    (m1, w1), (m2, w2) = ion.month_weights(datetime(2024, 5, 15))
    assert (m1, w1, m2, w2) == (5, 1.0, 6, 0.0)
    (m1, w1), (m2, w2) = ion.month_weights(datetime(2024, 1, 1))
    assert (m1, m2) == (12, 1) and w1 + w2 == pytest.approx(1.0) and 0.5 < w2 < 0.6


def test_ig12_does_not_turn_over_at_high_flux():
    peak = ion.r12_to_ig12(ion.R12_FIT_PEAK)
    values = [ion.r12_to_ig12(r) for r in (0.0, 100.0, 200.0, 240.0, 300.0, 400.0)]
    assert all(b >= a for a, b in zip(values, values[1:]))
    assert values[-1] == pytest.approx(peak) and values[-2] == pytest.approx(peak)
    # Only IG12 is held: the drivers keep the R12 the flux gives (M(3000)F2, hmF2, topside).
    d = _iono([10.0], [0.0]).drivers(WHEN, 350.0)
    assert d["r12"] == pytest.approx(ion.f107_to_r12(350.0)) and d["r12"] > ion.R12_FIT_PEAK
    assert d["ig12"] == pytest.approx(peak)


def test_aware_times_are_converted_to_utc_not_relabelled():
    from datetime import timezone
    est = timezone(timedelta(hours=-5))
    aware = datetime(2024, 6, 20, 15, 51, tzinfo=est)
    naive = datetime(2024, 6, 20, 20, 51)
    assert ion.subsolar_point(aware) == ion.subsolar_point(naive)
    assert ion.month_weights(datetime(2024, 6, 30, 22, tzinfo=est)) == \
        ion.month_weights(datetime(2024, 7, 1, 3))
    assert igrf.decimal_year(datetime(2024, 12, 31, 22, tzinfo=est)) == \
        igrf.decimal_year(datetime(2025, 1, 1, 3))


def test_subsolar_point():
    lat, _ = ion.subsolar_point(datetime(2024, 6, 20, 20, 51))
    assert lat == pytest.approx(23.44, abs=0.02)
    lat, _ = ion.subsolar_point(datetime(2024, 3, 20, 3, 6))
    assert abs(lat) < 0.02
    # At 12 UT the Sun is over the Greenwich meridian to the equation of time.
    _, lon = ion.subsolar_point(datetime(2024, 11, 3, 12, 0))
    assert lon == pytest.approx(-4.1, abs=0.2)   # EoT +16.4 min


# ---------------------------------------------------------------- the profile

def _columns():
    lat = np.array([89.9, 75.0, 52.0, 30.0, 12.0, 0.0, -8.0, -20.0, -45.0, -70.0, -89.9])
    lon = np.linspace(-180.0, 175.0, 36)
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    return la.ravel(), lo.ravel()


def test_reference_profile_is_physical():
    lat, lon = _columns()
    io_ = _iono(lat, lon)
    ref = io_.reference(WHEN, 150.0)
    assert np.all(ref["fof2"] > 0.0) and np.all(np.isfinite(ref["tec"]))
    assert np.all(ref["nmf2"] == pytest.approx(np.maximum(1.24e10 * ref["fof2"] ** 2, 1e6)))
    assert np.all((ref["hmf2"] > 150.0) & (ref["hmf2"] < 600.0))
    assert np.all(ref["foe"] >= 0.7 - 1e-9) and ref["foe"].max() > 3.0      # night floor, day E
    f1 = ref["fof1"] > 0.0
    assert f1.any() and (~f1).any()
    assert np.all(ref["hmf1"][f1] < ref["hmf2"][f1])
    # More solar activity, more electrons.
    assert np.all(io_.reference(WHEN, 220.0)["tec"] > ref["tec"])
    # TEC ~ NmF2 x slab thickness of a few hundred km.
    slab = ref["tec"] * 1e16 / ref["nmf2"] / 1e3
    assert np.all((slab > 50.0) & (slab < 1000.0))


def test_drivers_refuse_a_bad_flux_and_bad_columns():
    lat, lon = _columns()
    io_ = _iono(lat, lon)
    for bad in (0.0, -5.0, float("nan")):
        with pytest.raises(ion.IonosphereError):
            io_.drivers(WHEN, bad)
    with pytest.raises(ion.IonosphereError):
        _iono([95.0], [0.0])
    with pytest.raises(ion.IonosphereError):
        _iono([float("nan")], [0.0])
    with pytest.raises(ion.IonosphereError):
        _iono([0.0], [0.0], family="iri")
    with pytest.raises(ion.IonosphereError):
        _iono([0.0], [0.0], alt_lo=500.0, alt_hi=100.0)


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("f107,when", [(150.0, WHEN), (95.0, datetime(2024, 12, 21, 3, 30)),
                                       (230.0, datetime(2024, 6, 1, 0, 0))])
def test_device_kernel_matches_the_float64_reference(f107, when):
    import cupy as cp
    lat, lon = _columns()
    io_ = _iono(lat, lon)
    dev = {k: cp.asnumpy(v).astype(np.float64) for k, v in io_.evaluate(when, f107).items()}
    ref = io_.reference(when, f107)
    for key in ion.OUTPUTS:
        tol = 3e-3 if key in ("foe", "fof1", "hmf1") else 1e-3
        scale = np.maximum(np.abs(ref[key]), 1e-3 * np.abs(ref[key]).max())
        err = np.abs(dev[key] - ref[key]) / scale
        # The F1 switch (hmF1 > 110 km, cos chi > 1/3) may flip on a column
        # sitting on its threshold in float32; at most one such column.
        bad = np.flatnonzero(err > tol)
        assert bad.size <= (1 if key in ("fof1", "hmf1") else 0), (key, bad, dev[key][bad], ref[key][bad])


# ---------------------------------------------------------------- indices

_GFZ_ROW = ("{y:04d} {m:02d} {d:02d} 33731 33731.5 2601 18  1.000  1.667  1.667  1.667  1.667  2.000  0.667  "
            "{kp8}    4    6    6    6    6    7    3    4     5 171    {f:5.1f}    231.4 2")


def _gfz(days):
    lines = ["# header"] * 3
    for (y, m, d), f, kp8 in days:
        lines.append(_GFZ_ROW.format(y=y, m=m, d=d, f=f, kp8=kp8))
    return "\n".join(lines) + "\n"


def test_index_parser_drivers_and_missing_values():
    start = date(2024, 3, 1)
    days = [((start + timedelta(days=i)).timetuple()[:3], 100.0 + i, "1.000") for i in range(90)]
    rec = indices.parse(_gfz(days), "t")
    assert rec.days.size == 90 and rec.kp.shape == (90, 8)
    when = datetime.combine(start + timedelta(days=89), datetime.min.time()).replace(hour=12)
    assert rec.f107_driver(when, "daily") == pytest.approx(100.0 + 88)      # the previous day's
    assert rec.f107_driver(when, "81day") == pytest.approx(np.mean(100.0 + np.arange(8, 89)))
    # An aware time is read at its UTC day: 22 EST on day 88 is 03 UTC on day 89.
    from datetime import timezone
    est = when.replace(hour=22, tzinfo=timezone(timedelta(hours=-5))) - timedelta(days=1)
    assert rec.f107_driver(est, "daily") == rec.f107_driver(when, "daily")
    assert rec.kp_at(datetime(2024, 3, 1, 22)) == pytest.approx(1.0)
    missing = [((2024, 3, 1), 100.0, "-1.000"), ((2024, 3, 2), -1.0, "1.000"), ((2024, 3, 3), 120.0, "1.000")]
    rec = indices.parse(_gfz(missing), "t")
    with pytest.raises(indices.IndicesError):
        rec.kp_at(datetime(2024, 3, 1, 23))
    with pytest.raises(indices.IndicesError):
        rec.f107_driver(datetime(2024, 3, 3, 1), "daily")
    with pytest.raises(indices.IndicesError):
        rec.f107_driver(datetime(2024, 3, 3, 1), "81day")
    with pytest.raises(indices.IndicesError):
        rec.f107_driver(datetime(2024, 3, 3, 1), "weekly")
    with pytest.raises(indices.IndicesError):
        indices.parse(_gfz([((2024, 3, 1), 1.0, "1.000"), ((2024, 3, 3), 1.0, "1.000")]), "gap")


def test_index_file_sunspot_column_is_never_read():
    source = (ROOT / "woof" / "spacewx" / "indices.py").read_text()
    assert "words[24]" not in source.replace("# words[24]", "")


# ---------------------------------------------------------------- assets

def test_pinned_asset_refuses_wrong_bytes_and_leaves_nothing(tmp_path):
    good = b"coefficients\n"
    asset = assets.PinnedAsset("x.asc", "https://example.invalid/x.asc", len(good),
                               hashlib.sha256(good).hexdigest(), "t")
    with pytest.raises(assets.AssetError):
        assets.fetch_pinned(asset, cache=tmp_path, opener=lambda url: io.BytesIO(b"tampered\n"))
    assert not any((tmp_path / "t").iterdir())
    path = assets.fetch_pinned(asset, cache=tmp_path, opener=lambda url: io.BytesIO(good))
    assert path.read_bytes() == good
    assert assets.fetch_pinned(asset, cache=tmp_path, opener=None) == path   # cached, no network
    path.write_bytes(b"changed\n")
    with pytest.raises(assets.AssetError):
        assets.fetch_pinned(asset, cache=tmp_path)


# ---------------------------------------------------------------- GIM grading

_IONEX = """     1.0            IONOSPHERE MAPS     GNSS                IONEX VERSION / TYPE
    -1                                                      EXPONENT
    10.0 -10.0 -10.0                                        LAT1 / LAT2 / DLAT
  -180.0 180.0 180.0                                        LON1 / LON2 / DLON
                                                            END OF HEADER
     1                                                      START OF TEC MAP
  2024     5    10     0     0     0                        EPOCH OF CURRENT MAP
    10.0-180.0 180.0 180.0 450.0                            LAT/LON1/LON2/DLON/H
  100  200 9999
     0.0-180.0 180.0 180.0 450.0                            LAT/LON1/LON2/DLON/H
  300  400  500
   -10.0-180.0 180.0 180.0 450.0                            LAT/LON1/LON2/DLON/H
   10   20   30
     1                                                      END OF TEC MAP
     1                                                      START OF RMS MAP
"""


def test_ionex_reader_and_statistics(tmp_path):
    sys.path.insert(0, str(ROOT))
    from tools.spacewx_gim_validate import read_ionex, statistics
    path = tmp_path / "t.inx.gz"
    path.write_bytes(gzip.compress(_IONEX.encode()))
    gim = read_ionex(path)
    assert gim["epochs"] == [datetime(2024, 5, 10)]
    assert np.allclose(gim["lat"], [10.0, 0.0, -10.0]) and np.allclose(gim["lon"], [-180.0, 0.0, 180.0])
    tec = gim["tec"][0]
    assert np.isnan(tec[0, 2]) and tec[1, 1] == pytest.approx(40.0) and tec[2, 0] == pytest.approx(1.0)
    stats = statistics(gim["tec"] + 2.0, gim["tec"], gim["lat"])
    assert stats["all"]["bias"] == pytest.approx(2.0) and stats["all"]["rmse"] == pytest.approx(2.0)
    assert stats["all"]["corr"] == pytest.approx(1.0) and stats["all"]["nodes"] == 8


# ---------------------------------------------------------------- the door and the default path

def test_spacewx_door_is_registered_lazily():
    out = subprocess.run([sys.executable, "-m", "woof.cli", "spacewx", "ionosphere", "--help"],
                         capture_output=True, text=True, cwd=ROOT, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    assert "--f107-mode" in out.stdout and "--maps" in out.stdout
    cli = (ROOT / "woof" / "cli.py").read_text()
    assert '_lazy_register("woof.spacewx.cli")(sub)' in cli
    assert "import woof.spacewx" not in cli and "from woof.spacewx" not in cli


def test_forecast_paths_do_not_reach_the_ionosphere():
    """The forecast reaches the model only through the opt-in history door
    (woof.spacewx.history, tests/test_spacewx_ionosphere_history.py); the hex
    seam not at all."""
    for rel in ("woof/core/physics.py", "woof/config.py"):
        text = (ROOT / rel).read_text()
        assert "spacewx.ionosphere" not in text, rel
        assert "import woof.spacewx" not in text, rel
    seam = ROOT / "woof/hex/physics_seam.py"
    if seam.is_file():
        assert "spacewx" not in seam.read_text()


# ---------------------------------------------------------------- real coefficients (cached only)

def _cached(asset):
    path = assets.cache_root() / asset.subdir / asset.name
    if not path.is_file():
        pytest.skip(f"{asset.name} not in the space-weather cache ({assets.cache_root()})")
    return path


def test_real_igrf14_at_known_sites():
    model = igrf.load(_cached(igrf.IGRF14))
    year = igrf.decimal_year(datetime(2024, 5, 10))
    inc = igrf.inclination(model, year, [40.015, 51.5], [-105.27, -0.1], 0.0)
    assert inc[0] == pytest.approx(66.1, abs=0.3)      # Boulder
    assert inc[1] == pytest.approx(66.5, abs=0.3)      # London
    inc_eq = igrf.inclination(model, year, -11.95, -76.87, 0.0)
    assert abs(float(inc_eq)) < 3.0                     # Jicamarca, on the dip equator


def test_real_ccir_maps_give_a_physical_ionosphere():
    for month in (4, 5, 6):
        _cached(ion.map_asset("ccir", month))
    model = igrf.load(_cached(igrf.IGRF14))
    lat, lon = ion.grid(5.0, 10.0)
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    io_ = ion.Ionosphere(la, lo, when=WHEN, igrf_model=model)
    ref = io_.reference(WHEN, 160.0)
    assert 8.0 < ref["fof2"].max() < 20.0 and ref["fof2"].min() > 0.5
    assert 10.0 < np.average(ref["tec"], weights=np.cos(np.deg2rad(la.ravel()))) < 50.0
