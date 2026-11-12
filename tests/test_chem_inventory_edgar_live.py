"""The real EDGAR v8.1 archives through woof.chem_inventory_frames (opt-in).

Fetches the nine pinned TOTALS archives (about 145 MB, once, into the
emission cache) and checks them against their pins, their global totals and
a city-over-countryside contrast.  Network and disk heavy, so opt-in:
``WOOF_EDGAR_LIVE=1``.  A leg that has the archives cached (or network) runs
it; CI does not.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

if os.environ.get("WOOF_EDGAR_LIVE") != "1":
    pytest.skip("set WOOF_EDGAR_LIVE=1 to fetch and check the real EDGAR v8.1 "
                "archives", allow_module_level=True)

from woof import chem_inventory_frames as inv  # noqa: E402
from woof import chem_table  # noqa: E402
from woof.core.chem_radm2_emiss import EMISSIONS  # noqa: E402

#: Each file's fluxes:global_total attribute, Mt yr-1 (2022).
GLOBAL_MT = {"co": 450.16, "nox": 109.60, "nh3": 57.32, "nmvoc": 138.12,
             "pm25": 37.03, "pm10": 57.61}
RADIUS = 6371007.2


def test_every_archive_matches_its_pin_and_its_global_total():
    row = chem_table.catalog().sources["edgar-v81"]
    lat = row.grid["lat_first"] + row.grid["spacing_deg"] * np.arange(row.grid["nlat"])
    half = np.radians(0.5 * row.grid["spacing_deg"])
    area = (RADIUS ** 2 * np.radians(row.grid["spacing_deg"])
            * (np.sin(np.radians(lat) + half) - np.sin(np.radians(lat) - half)))
    cache = inv.InventoryFrames(np.zeros(1), np.zeros(1))._cache("edgar-v81")
    for field in row.variables:
        path = inv.fetch_archive(row, field, cache)
        values = inv.decode_archive(row, field, path)
        total = float(np.sum(values * area[:, None])) * 365 * 86400 / 1e9
        if field in GLOBAL_MT:
            assert total == pytest.approx(GLOBAL_MT[field], abs=0.01), field
        assert total > 0.5, field


def test_a_city_emits_far_above_its_countryside():
    lat = np.array([55.95, 55.86, 57.6])     # Edinburgh, Glasgow, Highlands
    lon = np.array([-3.19, -4.25, -4.6])
    provider = inv.InventoryFrames(lat, lon)
    frames = inv.radm2_emission_frames(provider, "edgar-v81")
    assert tuple(frames) == EMISSIONS
    for name in ("e_co", "e_no", "e_no2", "e_hc5", "e_pm25j", "e_ecj"):
        assert frames[name][0] > 20 * frames[name][2], name
        assert frames[name][1] > 20 * frames[name][2], name
