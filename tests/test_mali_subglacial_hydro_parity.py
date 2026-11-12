"""MALI subglacial hydrology (li_SGH_solve), word for word against v8.4.2.

The reference is the forward-Euler step's oracle
(``tools/mali_oracle/run_time_integration_fe.F90``; MALI v8.4.2 compiled
verbatim with gfortran -O0 -ffp-contract=off at RKIND 4 and 8), whose fixture
carries four cases with ``config_SGH``. Together they cover:

- the sheet alone and with channels;
- 'cavity' and 'overburden' pressure;
- the three basal melt inputs;
- the three tangent-slope calculations;
- the SHMIP C and D forcings (RKIND 4);
- several subcycles each.

tests/test_mali_time_integration_fe_parity.py compares every output word of
every case; this file holds what is particular to the hydrology.

THE BREAKAGES THESE PREVENT
---------------------------
1. A hydrology option the fixture never exercises, or a step that never
   subcycles.
2. A gate that cannot fail. Mutating one evaluation order in the hydrology
   kernels, or the operand order of a Fortran MAX on a signed-zero tie, must
   produce mismatches.
3. Something the model rejects, or that cannot be matched here, running anyway.
"""
from __future__ import annotations

import numpy as np
import pytest

from conftest import requires_gpu
from woof.verify import mali_time_integration_fe_oracle as fo

SGH_CASES = ("sgh_sheet_cavity", "sgh_channels_thermal", "sgh_overburden_shmip", "sgh_shmip_d")


@pytest.mark.parametrize("precision", ["r4", "r8"])
def test_fixture_covers_the_hydrology(precision):
    fixture = fo.load_fixture(precision)
    cases = {case["name"]: case["config"] for case in fo.cases(precision)}
    configs = [cases[name] for name in SGH_CASES]
    assert all(c["config_SGH"] for c in configs)
    assert {c["config_SGH_chnl_active"] for c in configs} == {False, True}
    assert {c["config_SGH_pressure_calc"] for c in configs} == {"cavity", "overburden"}
    assert {c["config_SGH_basal_melt"] for c in configs} == {"file", "thermal", "basal_heat"}
    assert {c["config_SGH_tangent_slope_calculation"] for c in configs} == {
        "from_normal_slope", "from_vertex_barycentric", "from_vertex_barycentric_kiteareas"}
    assert {c["config_SGH_shmip_forcing"] for c in configs} == ({"none"} if precision == "r8"
                                                                else {"none", "C2", "D3"})
    for name in SGH_CASES:
        out, deltat = fixture[name], cases[name]
        assert not any(out[k][0] for k in ("errInit", "errStep", "logErrors"))
        # the last subcycle's step is shorter than the master step: the step subcycled
        assert 0 < out["hydro/deltatSGH"] < out["mesh/deltat"], name
        assert (out["hydro/waterThickness"] > 0).any() and (out["hydro/waterFlux"] != 0).any(), name
    for name in ("sgh_channels_thermal", "sgh_overburden_shmip"):
        assert (fixture[name]["hydro/channelDischarge"] != 0).any(), name


_MUTATIONS = [
    # the effective conductivity's powers multiplied in another order
    ("k = M(M(conducCoeff, LI_POW(wte, S(alpha, LI_C(1.0)))), LI_POW(grad, S(beta, LI_C(2.0))));",
     "k = M(conducCoeff, M(LI_POW(wte, S(alpha, LI_C(1.0))), LI_POW(grad, S(beta, LI_C(2.0)))));",
     "sgh_sheet_cavity"),
    # the upwind sheet thickness's MAX with the tie's operand order flipped
    ("const rk up = li_max_first(M(velSign, waterThickness[c1]), M(M(velSign, LI_C(-1.0)), waterThickness[c2]));",
     "const rk up = li_max(M(velSign, waterThickness[c1]), M(M(velSign, LI_C(-1.0)), waterThickness[c2]));",
     "sgh_shmip_d"),
    # the cavity pressure's rho_water * gravity taken first
    ("p = A(D(M(M(M(S(S(S(z, divergence[i]), divergenceChannel[i]), channelAreaChangeCell[i]), LI_RHO_WATER),\n"
     "                    LI_GRAVITY), deltatSGH), porosity), old);",
     "p = A(D(M(M(S(S(S(z, divergence[i]), divergenceChannel[i]), channelAreaChangeCell[i]),\n"
     "                    M(LI_RHO_WATER, LI_GRAVITY)), deltatSGH), porosity), old);",
     "sgh_sheet_cavity"),
    # N**3 as a real power instead of gfortran's (N*N)*N
    ("__device__ __forceinline__ rk li_cube(rk x) { return M(M(x, x), x); }",
     "__device__ __forceinline__ rk li_cube(rk x) { return LI_POW(x, LI_C(3.0)); }", "sgh_channels_thermal"),
    # the sheet update's till term distributed over the step
    ("                       D(S(tillWaterThickness[i], tillWaterThicknessOld[i]), deltatSGH));\n    const rk g",
     "                       S(D(tillWaterThickness[i], deltatSGH), D(tillWaterThicknessOld[i], deltatSGH)));\n"
     "    const rk g", "sgh_sheet_cavity"),
]


@requires_gpu
@pytest.mark.parametrize("precision", ["r4", "r8"])
@pytest.mark.parametrize("original, mutated, case", _MUTATIONS)
def test_a_mutated_hydrology_order_is_caught(monkeypatch, precision, original, mutated, case):
    from woof.hex.mpas_ice import subglacial_hydro_v842 as sg
    source = sg.module_source(precision)
    assert source.count(original) == 1
    monkeypatch.setattr(sg, "module_source", lambda p="r4": source.replace(original, mutated))
    sg._module_on.cache_clear()
    try:
        fixture = fo.load_fixture(precision)
        total = 0
        for row in (c for c in fo.cases(precision) if c["name"].startswith("sgh")):
            got = fo.run_gpu(row)
            total += sum(fo.word_mismatches(got[k], fixture[row["name"]][k]) for k in got)
        assert total > 0
    finally:
        sg._module_on.cache_clear()


@requires_gpu
def test_options_the_model_rejects_and_r8_shmip_are_refused():
    from woof.hex.errors import ConfigurationRefusal
    from woof.hex.mpas_ice import subglacial_hydro_v842 as sg
    config = next(c for c in fo.cases() if c["name"] == "sgh_sheet_cavity")["config"]
    for knob, value in (("config_SGH_basal_melt", "ismip6"), ("config_SGH_pressure_calc", "flotation"),
                        ("config_SGH_tangent_slope_calculation", "upwind"), ("config_SGH_shmip_forcing", "E1")):
        with pytest.raises(ConfigurationRefusal, match=knob):
            sg.check_config(dict(config, **{knob: value}), "r4")
    with pytest.raises(ConfigurationRefusal, match="RKIND 8"):
        sg.check_config(dict(config, config_SGH_shmip_forcing="C1"), "r8")
    sg.check_config(dict(config, config_SGH_shmip_forcing="C1"), "r4")
