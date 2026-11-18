"""L2 solar-proton ionisation (woof.spacewx.particles).

The physics is checked against itself (energy conservation, the Bethe
formula's known values) and the readers against synthetic records; the
comparison with the CMIP7 ion-pair rates is in docs/spacewx.md.
"""
from __future__ import annotations

from datetime import datetime
import json
import math

import numpy as np
import pytest

from woof.spacewx import particles as P
from woof.spacewx import solarwind

JAN2012 = [math.nan, math.nan, math.nan, 2693.35, 265.57, 8.39]


def test_bethe_stopping_power_and_range_of_protons_in_air():
    # PSTAR (NIST) for dry air: 10 MeV 40.9 MeV cm2/g, CSDA range 0.145 g/cm2;
    # 100 MeV 6.44 MeV cm2/g, range 9.0 g/cm2.  Bethe without shell
    # corrections is within a few per cent there.
    assert P.stopping_power(10.0) == pytest.approx(40.9, rel=0.03)
    assert P.stopping_power(100.0) == pytest.approx(6.44, rel=0.03)
    t = P.range_table()
    r = lambda e: float(np.exp(np.interp(np.log(e), np.log(t.energy), np.log(t.range))))
    assert r(10.0) == pytest.approx(0.145, rel=0.06)
    assert r(100.0) == pytest.approx(9.0, rel=0.05)
    # Residual energy: none after the full range, all at zero depth.
    assert t.residual(np.array([50.0]), np.array([r(50.0) * 1.001]))[0] == 0.0
    assert t.residual(np.array([50.0]), np.array([0.0]))[0] == pytest.approx(50.0, rel=1e-6)


def test_spectrum_is_a_power_law_through_the_channels():
    edges, counts = P.spectrum(JAN2012)
    assert edges[0] == 10.0 and edges[-1] == pytest.approx(P.E_MAX)
    big_j = counts[::-1].cumsum()[::-1]                       # J(>edge)
    for e, j in ((10.0, 2693.35), (30.0, 265.57), (60.0, 8.39)):
        k = np.argmin(np.abs(edges[:-1] - e))
        assert big_j[k] == pytest.approx(j, rel=0.05)
    # A higher channel above a lower one is held; one channel is refused.
    _, c = P.spectrum([np.nan, np.nan, np.nan, 10.0, 20.0, 1.0])
    assert np.all(c >= 0)
    with pytest.raises(P.ParticleError):
        P.spectrum([np.nan] * 5 + [3.0])
    with pytest.raises(P.ParticleError):
        P.spectrum([1.0, 2.0], channels=(10.0, 30.0, 60.0))


def test_energy_is_conserved_and_layers_add_up():
    t = P.range_table()
    f = P.absorbed(JAN2012, P.DEPTH_GRID, table=t)
    assert np.all(np.diff(f) >= -1e-9 * f[-1])
    assert f[-1] == pytest.approx(P.energy_flux(JAN2012), rel=1e-3)
    p = np.array([1.0e-3, 0.1, 1.0, 10.0, 100.0, 1000.0]) * 100.0
    q = P.ion_pairs_per_gram(f, p)
    direct = P.ion_pair_rate(JAN2012, p, np.full(p.size - 1, 1.0e3), table=t)    # 1 g cm-3: per gram
    np.testing.assert_allclose(q, direct, rtol=0.02)
    # Layer energies sum to the absorption over the whole column, either way up.
    x = P.depth_from_pressure(p)
    total = np.sum(q * np.diff(x)) * P.W_ION_MEV
    assert total == pytest.approx(np.interp(np.log(x[-1]), np.log(P.DEPTH_GRID), f)
                                  - np.interp(np.log(x[0]), np.log(P.DEPTH_GRID), f), rel=1e-6)
    np.testing.assert_allclose(P.ion_pairs_per_gram(f, p[::-1]), q[::-1])
    # The mesosphere takes most of a soft event's energy.
    assert q[0] > 100.0 * q[3]


def test_mixing_ratio_tendency_needs_no_temperature():
    # 1e6 ion pairs per gram per second, 1.25 NO each: 1.25e6 * 28.9647 / N_A.
    assert P.vmr_tendency(1.0e6, P.NOX_PER_ION_PAIR) == pytest.approx(1.25e6 * 28.9647 / 6.02214076e23)
    np.testing.assert_array_equal(P.polar_cap([-75.0, -59.9, 0.0, 60.0, 89.0]), [1, 0, 0, 1, 1])


def _omni_row(year, doy, hour, protons):
    words = ["0"] * 55
    words[0], words[1], words[2] = str(year), str(doy), str(hour)
    fills = {16: "999.9", 17: "999.9", 24: "999.9", 25: "9999.", 39: "99", 41: "99999", 52: "999.9",
             53: "99999"}
    for k, v in fills.items():
        words[k - 1] = v
    for k, v in zip(range(43, 49), protons):
        words[k - 1] = v
    return " ".join(words)


def test_omni2_proton_words_and_swpc_hourly_means(tmp_path):
    rows = [_omni_row(2012, 24, 0, ["999999.99", "99999.99", "99999.99", "2693.35", "265.57", "8.39"]),
            _omni_row(2012, 24, 1, ["999999.99", "99999.99", "99999.99", "99999.99", "99999.99", "99999.99"])]
    sw = solarwind.parse_omni2("\n".join(rows) + "\n", "t")
    rec = P.from_solar_wind(sw)
    np.testing.assert_allclose(rec.at(datetime(2012, 1, 24, 0, 30)), JAN2012)
    assert rec.active(rec.at(datetime(2012, 1, 24, 0)))
    assert not rec.active(rec.at(datetime(2012, 1, 24, 1)))
    with pytest.raises(P.ParticleError):
        rec.at(datetime(2012, 1, 24, 2))
    recs = []
    for minute in range(0, 60, 5):
        for e, v in ((">=10 MeV", 20.0 + minute), (">=100 MeV", 0.5), (">=500 MeV", -1.0)):
            recs.append({"time_tag": f"2026-11-17T07:{minute:02d}:00Z", "satellite": 18, "flux": v, "energy": e})
    recs.append({"time_tag": "2026-11-17T08:00:00Z", "satellite": 18, "flux": 9.0, "energy": ">=10 MeV"})
    sw2 = P.parse_swpc(json.dumps(recs))
    assert sw2.channels == (10.0, 100.0)
    np.testing.assert_allclose(sw2.at(datetime(2026, 11, 17, 7)), [20.0 + 27.5, 0.5])
    assert np.all(np.isnan(sw2.at(datetime(2026, 11, 17, 8))))      # 1 of 12 values: not an hour
    with pytest.raises(P.ParticleError):
        P.parse_swpc(json.dumps([{"time_tag": "2026-11-17T07:00:00Z", "flux": 1.0, "energy": ">=100 MeV"}]))


def test_cli_prints_profiles_and_a_receipt(tmp_path, capsys):
    from woof.spacewx import cli
    rows = [_omni_row(2012, 24, h, ["999999.99", "99999.99", "99999.99", v, "265.57", "8.39"])
            for h, v in ((0, "2693.35"), (1, "5.00"))]
    path = tmp_path / "omni2_2012.dat"
    path.write_text("\n".join(rows) + "\n")
    out = tmp_path / "p.json"
    assert cli.main(["spacewx", "particles", "--time", "2012-01-24T00:00Z", "--count", "2",
                     "--source", str(path), "--levels", "0.1", "1", "10", "--out", str(out)]) == 0
    doc = json.loads(out.read_text())
    first, second = doc["times"]
    assert first["spe"] and not second["spe"]
    assert first["levels_hpa"] == [0.1, 1.0, 10.0]
    q = np.asarray(first["ion_pairs_per_gram_s"])
    assert q[0] > q[1] > q[2] > 0 and second["ion_pairs_per_gram_s"] == [0.0, 0.0, 0.0]
    np.testing.assert_allclose(first["nox_ppbv_per_day"], P.vmr_tendency(q, 1.25) * 86400e9)
    assert first["flux"][">1 MeV"] is None and doc["spacewx_proton_channels_mev"] == list(P.CHANNELS_MEV)
    assert "SPE" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="positive"):
        cli.main(["spacewx", "particles", "--time", "2012-01-24T00:00Z", "--source", str(path), "--levels", "-1"])
