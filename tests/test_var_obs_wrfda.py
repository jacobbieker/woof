"""woof.da.var.obs_wrfda: WRFDA's obs_gts ASCII reader and preprocessing (plan V1.1).

The synthetic tests run everywhere. The tutorial gate reads the WRFDA
tutorial's ``ob.ascii`` (``WOOF_WRFDA471_DIR``; see ``test_var_be_dat.py``):
the counts and committed digests always, and the oracle build's
``woof_iv.bin`` when that is in the same directory.

Against the oracle the default (correctly rounded) transcendentals agree word
for word, except where glibc's own ``sin``/``cos``/``tan``/``pow``/``exp`` is
not correctly rounded: an ulp in a few dozen positions and in one ``q``
error. ``libm="host"`` on a glibc host is the oracle's libm and agrees word
for word. The exception is the GNSS-RO heights (see the module docstring).
"""

from __future__ import annotations

import json
import math
import os
import platform
from pathlib import Path

import numpy as np
import pytest

from woof.da.var import crmath
from woof.da.var import obs_wrfda as ow
from woof.da.var.oracle import array_digest, read_dump

MANIFEST = Path(__file__).parent / "data" / "wrfda471" / "manifest.json"

PROJ = dict(cen_lat=float(np.float32(40.000008)), cen_lon=-95.0, stand_lon=-95.0, truelat1=30.0,
            truelat2=60.0, moad_cen_lat=float(np.float32(40.000008)), dx=60000.0, ide=89, jde=59)
WINDOW = dict(analysis_date="2008-02-05_12:00:00", time_window_min="2008-02-05_11:00:00",
              time_window_max="2008-02-05_13:00:00")

HEADER = """TOTAL =      8, MISS. =-888888.,
INFO  = PLATFORM, DATE, NAME, LEVELS, LATITUDE, LONGITUDE, ELEVATION, ID.
SRFC  = SLP, PW (DATA,QC,ERROR).
EACH  = PRES, SPEED, DIR, HEIGHT, TEMP, DEW PT, HUMID (DATA,QC,ERROR)*LEVELS.
INFO_FMT = (A12,1X,A19,1X,A40,1X,I6,3(F12.3,11X),6X,A40)
SRFC_FMT = (F12.3,I4,F7.2,F12.3,I4,F7.3)
EACH_FMT = (3(F12.3,I4,F7.2),11X,3(F12.3,I4,F7.2),11X,3(F12.3,I4,F7.2))
#------------------------------------------------------------------------------#
"""
M = -888888.0


def _info(platform_, date, name, levels, lat, lon, elv, sid):
    s = f"{platform_:<12} {date:<19} {name:<40} {levels:6d}"
    for v in (lat, lon, elv):
        s += f"{v:12.3f}" + " " * 11
    return s + " " * 6 + f"{sid:<40}"


def _trip(v, qc, err):
    return f"{v:12.3f}{qc:4d}{err:7.2f}"


def _srfc(slp=M, pw=M, pw_err=0.2, pw_qc=-88):
    return _trip(slp, 0, 200.0) + f"{pw:12.3f}{pw_qc:4d}{pw_err:7.3f}"


def _each(p, spd, dirn, h, t, td, rh, err=(100.0, 1.1, 5.0, 6.0, 2.0, 2.0, 10.0), wind_qc=0):
    return (_trip(p, 0, err[0]) + _trip(spd, wind_qc, err[1]) + _trip(dirn, wind_qc, err[2]) + " " * 11
            + _trip(h, 0, err[3]) + _trip(t, 0, err[4]) + _trip(td, 0, err[5]) + " " * 11
            + _trip(rh, 0, err[6]))


def _write(tmp_path, reports):
    lines = HEADER.splitlines()
    for info, srfc, each in reports:
        lines += [info, srfc, *each]
    path = tmp_path / "ob.ascii"
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.fixture
def synthetic(tmp_path):
    d = "2008-02-05_11:30:00"
    reports = [
        # synop at the domain centre: 10 m/s from the west
        (_info("FM-12 SYNOP", d, "S1", 1, 40.0, -95.0, 300.0, "72001"), _srfc(),
         [_each(97000.0, 10.0, 270.0, 300.0, 280.0, 275.0, 70.0)]),
        # at the window's end: dropped (obs_time >= time_slots(nt))
        (_info("FM-12 SYNOP", "2008-02-05_13:00:00", "S2", 1, 40.0, -95.0, 0.0, "72002"), _srfc(),
         [_each(97000.0, 1.0, 90.0, 0.0, 280.0, 275.0, 70.0)]),
        # outside every domain: not even counted
        (_info("FM-12 SYNOP", d, "S3", 1, 0.0, 0.0, 0.0, "72003"), _srfc(),
         [_each(97000.0, 1.0, 90.0, 0.0, 280.0, 275.0, 70.0)]),
        # sounding whose first level is the surface (height == elevation)
        (_info("FM-35 TEMP", d, "R1", 3, 41.0, -96.0, 250.0, "72004"), _srfc(),
         [_each(98000.0, 5.0, 180.0, 250.0, 285.0, 280.0, 80.0),
          _each(85000.0, 15.0, 225.0, 1500.0, 275.0, 270.0, 60.0),
          _each(50000.0, 30.0, 250.0, 5600.0, 255.0, 240.0, 30.0)]),
        # satem: two layers, cloud-free (pw <= 10)
        (_info("FM-86 SATEM", d, "T1", 2, 39.0, -94.0, 0.0, "72005"), _srfc(slp=100000.0, pw=5.0, pw_err=3.0, pw_qc=0),
         [_each(85000.0, M, M, M, 1400.0, M, M, err=(100.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0), wind_qc=-88),
          _each(50000.0, M, M, M, 5600.0, M, M, err=(100.0, 0.0, 0.0, 0.0, 15.0, 0.0, 0.0), wind_qc=-88)]),
        # a MODIS AMV is a polar AMV
        (_info("FM-88 SATOB", d, "MODIS AMV", 1, 40.5, -95.5, 0.0, "72006"), _srfc(),
         [_each(40000.0, 20.0, 300.0, M, M, M, M)]),
        # scatterometer: error floor 1 m/s
        (_info("FM-281 QSCAT", d, "Q1", 1, 38.0, -97.0, 0.0, "72007"), _srfc(),
         [_each(M, 8.0, 45.0, 10.0, M, M, M, err=(0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0))]),
        # metar with a temperature past the gross check (t > 350 K)
        (_info("FM-15 METAR", d, "K1", 1, 40.2, -95.2, 200.0, "72008"), _srfc(),
         [_each(98000.0, 3.0, 10.0, 200.0, 400.0, 275.0, 70.0)]),
    ]
    return ow.read_obs_ascii(_write(tmp_path, reports), ow.LambertWRFDA(**PROJ), **WINDOW)


def test_format_expansion():
    f = ow.fortran_format_fields("(A12,1X,A19,1X,A40,1X,I6,3(F12.3,11X),6X,A40)")
    assert f[:8] == [("A", 12, 0), ("X", 1, 0), ("A", 19, 0), ("X", 1, 0), ("A", 40, 0), ("X", 1, 0),
                     ("I", 6, 0), ("F", 12, 3)]
    assert f.count(("X", 11, 0)) == 3 and f[-2:] == [("X", 6, 0), ("A", 40, 0)]
    each = ow.fortran_format_fields("(3(F12.3,I4,F7.2),11X,3(F12.3,I4,F7.2),11X,3(F12.3,I4,F7.2))")
    assert len([k for k in each if k[0] != "X"]) == 27
    with pytest.raises(ValueError):
        ow.fortran_format_fields("(F12.3,(I4)")


def test_fortran_numeric_reads():
    fields = [("F", 8, 3), ("I", 4, 0), ("F", 6, 2)]
    # blanks are zero; an F field without a decimal point carries d implied decimals
    assert ow._read_record("  12345 -88    ", fields) == [12.345, -88, 0.0]
    assert ow._read_record(" -1.5e2", fields[:1]) == [-150.0]


def test_julian_minutes():
    # da_get_julian_time counts minutes from 1978-01-01 00:00
    assert ow.julian_minutes(1978, 1, 1, 0, 0) == 0.0
    assert ow.julian_minutes(2008, 2, 5, 12, 0) - ow.julian_minutes(2008, 2, 4, 12, 0) == 1440.0
    assert ow.julian_minutes(2008, 3, 1, 0, 0) - ow.julian_minutes(2008, 2, 28, 0, 0) == 2 * 1440.0


def test_crmath_against_the_host_within_one_ulp():
    xs = np.linspace(-3.0, 3.0, 61)
    for f, g in ((crmath.sin, math.sin), (crmath.cos, math.cos), (crmath.tan, math.tan),
                 (crmath.exp, math.exp)):
        for x in xs:
            a, b = f(float(x)), g(float(x))
            # Apple's libm is 2 ulps off at tan(-2.1) (below)
            assert abs(a - b) <= 2 * math.ulp(b), (f.__name__, x)
    for x in (0.25, 1.5, 1000.0, 3.7e-5):
        assert abs(crmath.log10(x) - math.log10(x)) <= math.ulp(math.log10(x))
        assert abs(crmath.pow_(x, 0.716) - math.pow(x, 0.716)) <= math.ulp(math.pow(x, 0.716))
    assert crmath.log10(1000.0) == 3.0 and crmath.exp(0.0) == 1.0 and crmath.cos(0.0) == 1.0


def test_crmath_tan_is_correctly_rounded():
    """tan(-2.1) against an independent continued fraction at 100 digits."""
    import decimal

    ctx = decimal.Context(prec=100)
    x = decimal.Decimal(-2.1)
    x2 = ctx.multiply(x, x)
    t = decimal.Decimal(0)
    for k in range(400, 0, -1):
        t = ctx.divide(x2, ctx.subtract(2 * k + 1, t))
    assert crmath.tan(-2.1) == float(ctx.divide(x, ctx.subtract(1, t))) == 1.7098465429045073


def test_lambert_centre_maps_to_the_known_point():
    proj = ow.LambertWRFDA(**PROJ)
    x, y = proj.llxy(proj.cen_lat, proj.cen_lon)
    assert abs(x - 45.0) < 1e-9 and abs(y - 30.0) < 1e-9
    # 10 m/s from the west at the standard longitude is +u (to WRFDA's pi = 3.1415926)
    u, v = proj.ffdduv(10.0, 270.0, proj.stand_lon)
    assert u == pytest.approx(10.0, rel=1e-12) and abs(v) < 1e-6


def test_synthetic_reports(synthetic):
    obs = synthetic
    assert obs.counts()["synop"] == 1 and obs.ntotal["synop"] == 1  # S2 out of window, S3 outside all
    s = obs["synop"]
    u, v = s.fields["u"], s.fields["v"]
    assert math.hypot(u.inv[0], v.inv[0]) == pytest.approx(10.0)
    assert v.error[0] == u.error[0] == 1.1  # wind_sd = false: v takes u's (the speed's) error
    # 40.0 N is just south of CEN_LAT (40.000008 as float32), so j is 29
    assert s.info["i"][0] == 45 and s.info["j"][0] == 29 and s.info["proc_domain"][0]
    assert s.info["dx"][0] + s.info["dxm"][0] == 1.0 and 0.0 <= s.info["dy"][0] < 1.0
    # q from rh, and its error from the rh error
    _, qs = ow.tp_to_qs(280.0, 97000.0)
    assert s.fields["q"].inv[0] == 70.0 * qs / 100.0
    assert s.fields["q"].error[0] > 0.0

    snd, sfc = obs["sound"], obs["sonde_sfc"]
    assert snd.info["levels"].tolist() == [2] and sfc.nlocal == 1
    assert snd.coords["p"].tolist() == [85000.0, 50000.0]
    assert sfc.fields["p"].inv[0] == 98000.0 and sfc.coords["h"][0] == 250.0
    assert sfc.info["obs_global_index"][0] == snd.info["obs_global_index"][0] == 1

    sat = obs["satem"]
    th = sat.fields["thickness"]
    assert th.inv.tolist() == [1400.0, 4200.0]
    assert th.error[1] == math.sqrt(15.0 * 15.0 + 10.0 * 10.0)
    assert th.error[0] == math.sqrt(10.0 * 10.0 + 3.0 * 3.0)
    assert sat.extra["ref_p"].tolist() == [100000.0]

    assert obs["polaramv"].nlocal == 1 and obs["geoamv"].nlocal == 0
    assert obs.ntotal["polaramv"] == 2  # WRFDA counts an inside polar AMV twice

    q = obs["qscat"]
    assert q.fields["u"].error[0] == q.fields["v"].error[0] == 1.0

    m = obs["metar"]
    assert m.fields["t"].inv[0] == ow.MISSING_R and m.fields["t"].qc[0] == ow.MISSING
    assert m.fields["q"].inv[0] == ow.MISSING_R  # no q without t
    assert m.fields["q"].qc[0] == ow.MISSING_DATA  # da_get_q_error found no error


def test_to_obsset(synthetic):
    o = synthetic.to_obsset()
    kinds = set(o.kind.tolist())
    assert {"surface_wind_u", "surface_temperature", "upper_wind_u", "upper_temperature"} <= kinds
    assert np.all(np.isfinite(o.value)) and np.all(o.error > 0)


def test_unknown_libm_is_refused():
    with pytest.raises(ValueError):
        ow.LambertWRFDA(**PROJ, libm="fast")


# ---------------------------------------------------------------- tutorial (V1.1 gate)

def _tutorial_dir():
    d = os.environ.get("WOOF_WRFDA471_DIR")
    if not d or not (Path(d) / "ob.ascii").is_file():
        pytest.skip("WOOF_WRFDA471_DIR with the WRFDA tutorial ob.ascii is not set")
    return Path(d)


def _digests(obs) -> dict:
    out = {}
    for name, t in obs.types.items():
        parts = [array_digest(np.asarray(t.info[k], dtype=np.float64))
                 for k in ("lat", "lon", "elv", "x", "y", "i", "j", "dx", "dy", "dxm", "dym",
                           "proc_domain", "obs_global_index", "levels")]
        parts += [array_digest(v) for _, v in sorted(t.coords.items())]
        for _, f in sorted(t.fields.items()):
            parts += [array_digest(f.inv), array_digest(np.asarray(f.qc, dtype=np.float64)),
                      array_digest(f.error)]
        out[name] = array_digest(np.frombuffer("".join(parts).encode(), dtype=np.uint8))
    return out


def _read_tutorial(libm="correctly-rounded"):
    d = _tutorial_dir()
    man = json.loads(MANIFEST.read_text())["obs"]
    proj = ow.LambertWRFDA.from_attributes(man["first_guess_attributes"], libm=libm)
    return d, man, ow.read_obs_ascii(d / "ob.ascii", proj, **man["window"])


def test_tutorial_counts_and_digests():
    """Every type's count (rsl.out's local counts) and committed digests."""
    d, man, obs = _read_tutorial()
    import hashlib
    assert hashlib.sha256((d / "ob.ascii").read_bytes()).hexdigest() == man["ob_ascii_sha256"]
    assert {k: v for k, v in obs.counts().items() if v} == man["counts"]
    assert _digests(obs) == man["digests"]


def _ulps(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return np.max(np.abs(a - b) / np.spacing(np.maximum(np.abs(a), np.abs(b))), initial=0.0)


def _compare(obs, dump, exact: bool):
    """Info, coordinates and errors against woof_iv.bin. ``exact`` False allows
    the ulps WRFDA's non-correctly-rounded libm puts in x, y and one q error."""
    for name, t in obs.types.items():
        tag = "gpsrf" if name == "gpsref" else name
        if f"{tag}%levels" not in dump:
            assert t.nlocal == 0, name
            continue
        assert t.nlocal == len(dump[f"{tag}%levels"]), name
        for k in ("levels", "obs_global_index"):
            assert np.array_equal(t.info[k], dump[f"{tag}%{k}"]), (name, k)
        assert np.array_equal(t.info["elv"], dump[f"{tag}%elv"]), name
        for k in ("lat", "lon"):
            assert np.array_equal(t.info[k], dump[f"{tag}%{k}"][0]), (name, k)
        for k in ("i", "j"):
            assert np.array_equal(t.info[k], dump[f"{tag}%{k}"][0]), (name, k)
        assert np.array_equal(t.info["proc_domain"].astype(np.int32), dump[f"{tag}%proc_domain"][0]), name
        for k in ("x", "y", "dx", "dy", "dxm", "dym"):
            a, b = t.info[k], dump[f"{tag}%{k}"][0]
            if exact:
                assert np.array_equal(a, b), (name, k)
            else:
                assert np.max(np.abs(a - b), initial=0.0) <= 6e-14, (name, k)
        for k, v in t.coords.items():
            ref = dump[f"{tag}%{k}"]
            if name == "gpsref" and k == "h":
                assert _ulps(v, ref) <= 4, name
            else:
                assert np.array_equal(v, ref), (name, k)
        for k, f in t.fields.items():
            ref = dump[f"{name}%{k}%error"]
            if exact or k != "q":
                assert np.array_equal(f.error, ref), (name, k)
            else:
                assert _ulps(f.error, ref) <= 1, (name, k)


def test_tutorial_against_the_oracle_dump():
    d, _, obs = _read_tutorial()
    if not (d / "woof_iv.bin").is_file():
        pytest.skip("woof_iv.bin (oracle dump) not in WOOF_WRFDA471_DIR")
    _compare(obs, read_dump(d / "woof_iv.bin"), exact=False)


@pytest.mark.skipif(platform.libc_ver()[0] != "glibc", reason="the oracle's words need glibc's libm")
def test_tutorial_host_glibc_is_the_oracle_word_for_word():
    d, _, obs = _read_tutorial(libm="host")
    if not (d / "woof_iv.bin").is_file():
        pytest.skip("woof_iv.bin (oracle dump) not in WOOF_WRFDA471_DIR")
    _compare(obs, read_dump(d / "woof_iv.bin"), exact=True)
