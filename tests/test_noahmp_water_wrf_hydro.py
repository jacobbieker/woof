"""Noah-MP WATER as WRF builds it with -DWRF_HYDRO, bitwise against WRF 4.6.1.

Under WRF_HYDRO, WATER (module_sf_noahmplsm.F :5954-6261) adds the ponded
head WRF-Hydro returns to the LSM to its surface input,
QINSUR = QINSUR+sfcheadrt/DT*0.001 (:6173-6175).  The fixture
``tests/data/oracles/noahmp/noahmp-water-hydro.csv`` is
``tools/noahmp_wrf461_oracle/build_water_hydro.sh``: the pinned module
compiled at -O0 with -DWRF_HYDRO, and run_water.F90's 36 cases extended by
``patch_run_water_hydro.py`` with a ponded head per case.  The CPU
transcription (``water(..., sfcheadrt=)``) and the device slab
(``evaluate_water_slab(..., sfcheadrt=)``, on the selected-options kernel)
must reproduce it word for word.
"""

from __future__ import annotations

import csv
import hashlib
import struct
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

import test_noahmp_water as base
from conftest import requires_gpu
from woof.core.noahmp_water import water

FIX = Path(__file__).resolve().parent / "data" / "oracles" / "noahmp" / "noahmp-water-hydro.csv"


def _load():
    table = defaultdict(dict)
    for r in csv.DictReader(FIX.open(newline="")):
        key = (r["field"], int(r["index"]))
        v = (int(r["value"]) if r["dtype"] == "int"
             else np.float32(struct.unpack(">f", bytes.fromhex(r["bits"]))[0]))
        table[(r["case"], r["stage"])][key] = v
    return table


HYDRO = _load()
CASES = sorted({c for c, stage in HYDRO if stage == "input"})


@pytest.fixture
def hydro_table(monkeypatch):
    import test_noahmp_water_slab as slab

    monkeypatch.setattr(base, "TABLE", HYDRO)
    monkeypatch.setattr(slab, "TABLE", HYDRO)
    return slab


def _head(case):
    return HYDRO[(case, "input")][("sfcheadrt", 0)]


def test_fixture_is_the_one_the_build_hashed():
    pins = (FIX.parent / "noahmp-water-hydro-sha256sums.txt").read_text().split("\n")
    pinned = {line.split()[1]: line.split()[0] for line in pins if line.strip()}
    assert pinned["noahmp-water-hydro.csv"] == hashlib.sha256(FIX.read_bytes()).hexdigest()


def test_the_cases_carry_ponded_heads():
    heads = [float(_head(c)) for c in CASES]
    assert len(CASES) == 36 and max(heads) > 0 and len(set(heads)) > 5


@pytest.mark.parametrize("case", CASES)
def test_water_with_a_ponded_head_matches_wrf_hydro(case, hydro_table):
    args = base._call(case)
    out = water(*args, sfcheadrt=_head(case))
    got = base._produced(args[1], out)
    want = HYDRO[(case, "output")]
    bad = [f"{k[0]}[{k[1]}] port={base._bits(v)} oracle={base._bits(want[k])}"
           for k, v in got.items() if base._bits(v) != base._bits(want[k])]
    assert not bad, f"{case}: " + "; ".join(bad)


def test_the_head_changes_the_answer(hydro_table):
    moved = 0
    for case in CASES:
        if float(_head(case)) == 0.0:
            continue
        a, b = base._call(case), base._call(case)
        with_head = water(*a, sfcheadrt=_head(case)).acc_qinsur
        moved += base._bits(with_head) != base._bits(water(*b).acc_qinsur)
    assert moved > 20


@pytest.mark.gpu
@requires_gpu
def test_the_device_slab_with_a_ponded_head_matches(hydro_table):
    import cupy as cp

    from woof.core.noahmp_water_slab import evaluate_water_slab

    slab = hydro_table
    heads = cp.asarray(np.array([_head(c) for c in CASES], np.float32))
    got = evaluate_water_slab(slab._slab_fields(cp), len(CASES), sfcheadrt=heads)
    calls = [(base._call(c), {}) for c in CASES]
    fluxes = [water(*args, sfcheadrt=_head(c)) for (args, _), c in zip(calls, CASES)]
    want = slab._host_outputs(calls, fluxes)
    for name, value in want.items():
        slab._same_bits(cp.asnumpy(got[name]), value, name)
