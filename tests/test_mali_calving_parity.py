"""MALI calving and the calving-front restoration, word for word against v8.4.2.

The reference is the forward-Euler step's oracle
(``tools/mali_oracle/run_time_integration_fe.F90``; MALI v8.4.2 compiled
verbatim with gfortran -O0 -ffp-contract=off at RKIND 4 and 8), whose fixture
carries a case for each ``config_calving`` option and one restoring the
calving front. tests/test_mali_time_integration_fe_parity.py compares every
output word of every case; this file holds what is particular to calving.

THE BREAKAGES THESE PREVENT
---------------------------
1. A calving option the fixture never exercises: each one must calve ice, the
   restoration must both restore and calve, and eigencalving must find front
   cells with a calving rate to distribute.
2. A gate that cannot fail. Mutating one evaluation order in the calving
   kernels, or the order of the serial flux distribution, must produce
   mismatches.
3. An option the model rejects that the port runs anyway.
"""
from __future__ import annotations

import numpy as np
import pytest

from conftest import requires_gpu
from woof.verify import mali_time_integration_fe_oracle as fo

CALVING_CASES = ("calving_thickness_threshold", "calving_floating_data", "calving_topographic",
                 "calving_eigen_scalar", "calving_eigen_data", "calving_restore_front")


@pytest.mark.parametrize("precision", ["r4", "r8"])
def test_fixture_covers_every_calving_option(precision):
    fixture = fo.load_fixture(precision)
    cases = {case["name"]: case for case in fo.cases(precision)}
    options = {cases[name]["config"]["config_calving"] for name in CALVING_CASES
               if not cases[name]["config"]["config_restore_calving_front"]}
    assert options == {"thickness_threshold", "floating", "topographic_threshold", "eigencalving"}
    assert {cases[n]["config"]["config_calving_eigencalving_parameter_source"]
            for n in ("calving_eigen_scalar", "calving_eigen_data")} == {"scalar", "data"}
    for name in CALVING_CASES:
        out = fixture[name]
        assert not any(out[k][0] for k in ("errInit", "errStep", "logErrors"))
        assert (out["geometry/calvingThickness"] > 0).any(), name
    for name in ("calving_eigen_scalar", "calving_eigen_data"):
        assert (fixture[name]["geometry/requiredCalvingVolumeRate"] > 0).any(), name
    restore = fixture["calving_restore_front"]
    assert (restore["geometry/restoreThickness"] > 0).any()
    # a timescale calves a fraction of the column: thickness_threshold's step is 30 days of 8e6 s
    assert cases["calving_thickness_threshold"]["config"]["config_calving_timescale"] > 0


_MUTATIONS = [
    # the calved share of the column as what the uncalved share leaves
    ("if (ocean == 1 && inactiveMargin == 0 && thickness[i] > LI_C(0.0)) calved = M(thickness[i], fraction);",
     "if (ocean == 1 && inactiveMargin == 0 && thickness[i] > LI_C(0.0))\n"
     "        calved = S(thickness[i], M(thickness[i], S(LI_C(1.0), fraction)));",
     "calving_thickness_threshold"),
    # the calving velocity's factors in another order
    ("const rk v = M(M(M(eigencalvingParameter[i], li_max(LI_C(0.0), eMax[i])), li_max(LI_C(0.0), eMin[i])), floating);",
     "const rk v = M(M(eigencalvingParameter[i], M(li_max(LI_C(0.0), eMax[i]), li_max(LI_C(0.0), eMin[i]))), floating);",
     "calving_eigen_data"),
    # the front's required volume rate in another order
    ("rate = M(M(v, length), height);", "rate = M(v, M(length, height));", "calving_eigen"),
    # the flux spread inward: the removed thickness with a reciprocal area
    ("const rk here = li_min(D(left, (rk)inward), cellVolume[jc]);\n"
     "                    calvingThickness[jc] = A(calvingThickness[jc], D(here, areaCell[jc]));",
     "const rk here = li_min(D(left, (rk)inward), cellVolume[jc]);\n"
     "                    calvingThickness[jc] = A(calvingThickness[jc], M(here, D(LI_C(1.0), areaCell[jc])));",
     "calving_eigen"),
    # the removed volume's thickness taken with a reciprocal area
    ("calvingThickness[jc] = A(calvingThickness[jc], D(here, areaCell[jc]));\n"
     "                cellVolume[jc] = S(cellVolume[jc], here);\n                left = S(left, here);\n            }\n"
     "        }\n        if (left > LI_C(0.0)) {",
     "calvingThickness[jc] = A(calvingThickness[jc], M(here, D(LI_C(1.0), areaCell[jc])));\n"
     "                cellVolume[jc] = S(cellVolume[jc], here);\n                left = S(left, here);\n            }\n"
     "        }\n        if (left > LI_C(0.0)) {",
     "calving_eigen_data"),
    # the restored thickness without the column's own ice subtracted
    ("restoreThickness[i] = S(restoreMin, thickness[i]);", "restoreThickness[i] = restoreMin;",
     "calving_restore_front"),
]


@requires_gpu
@pytest.mark.parametrize("precision", ["r4", "r8"])
@pytest.mark.parametrize("original, mutated, case", _MUTATIONS)
def test_a_mutated_calving_order_is_caught(monkeypatch, precision, original, mutated, case):
    from woof.hex.mpas_ice import calving_v842 as cv
    source = cv.module_source(precision)
    assert source.count(original) == 1
    monkeypatch.setattr(cv, "module_source", lambda p="r4": source.replace(original, mutated))
    cv._module_on.cache_clear()
    try:
        fixture = fo.load_fixture(precision)
        total = 0
        for row in (c for c in fo.cases(precision) if c["name"].startswith(case)):
            got = fo.run_gpu(row)
            total += sum(fo.word_mismatches(got[k], fixture[row["name"]][k]) for k in got)
        assert total > 0
    finally:
        cv._module_on.cache_clear()


@requires_gpu
def test_options_the_model_rejects_are_refused():
    import cupy as cp
    from woof.hex.errors import ConfigurationRefusal
    from woof.hex.mpas_ice import calving_v842 as cv
    case = next(c for c in fo.cases() if c["name"] == "calving_eigen_scalar")
    dev = {pool: {k: cp.asarray(np.ascontiguousarray(v).reshape(np.shape(v) or (1,)))
                  for k, v in case["pools"][pool].items()} for pool in fo.POOLS}
    for knob, value in (("config_calving", "von_mises_stress"),
                        ("config_calving_eigencalving_parameter_source", "file")):
        with pytest.raises(ConfigurationRefusal, match=knob):
            cv.calve_ice(mesh=dev["mesh"], geometry=dev["geometry"], velocity=dev["velocity"],
                         config=dict(case["config"], **{knob: value}))
    with pytest.raises(ValueError, match="config_calving_thickness"):
        cv.calve_ice(mesh=dev["mesh"], geometry=dev["geometry"], velocity=dev["velocity"],
                     config=dict(case["config"], config_calving="thickness_threshold",
                                 config_calving_thickness=1.0))
