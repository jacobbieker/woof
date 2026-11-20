"""E3SM v3.2.0's ocean import merge: cases, the oracle's stream, the GPU run.

``tools/e3sm_coupler_oracle/run_ocean_merge.F90`` drives E3SM v3.2.0's
``prep_ocn_merge`` (cut verbatim out of ``driver-mct/main/prep_ocn_mod.F90``)
on the pinned MCT's attribute vectors, built with gfortran -O0
-ffp-contract=off. Each case is one process: the field lists, the switches
(``flux_epbalfact``, ``seq_flds_i2o_per_cat``, ``rof2ocn_nutrients``,
``wav_ocn_coup``) and several calls' worth of attribute vectors (the first
call works out the merge plan, the later ones reuse it, as in a run).

The lists are modelled on what ``seq_flds_mod`` registers for an MPAS
ocean + sea-ice case with a data atmosphere and runoff (``GCASE``), plus
synthetic lists for the branches a G case does not reach (the nutrients, the
per-category fields, the waves coupled two ways, an ``Sx_`` state merged from
two senders, a state copied and then merged onto) and for E3SM's aborts.
"""
from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import numpy as np

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "tests" / "data" / "oracles" / "e3sm_ocean_merge"
VECTORS = ("a2x", "i2x", "r2x", "w2x", "g2x", "xao", "fractions", "x2o")

_AEROSOLS = ("Faxa_bcphidry Faxa_bcphodry Faxa_bcphiwet Faxa_ocphidry Faxa_ocphodry Faxa_ocphiwet "
             "Faxa_dstwet1 Faxa_dstwet2 Faxa_dstwet3 Faxa_dstwet4 Faxa_dstdry1 Faxa_dstdry2 Faxa_dstdry3 "
             "Faxa_dstdry4").split()
GCASE = {
    "a2x": ("Sa_z Sa_topo Sa_u Sa_v Sa_tbot Sa_ptem Sa_shum Sa_pbot Sa_dens Sa_uovern Sa_pslv "
            "Faxa_rainc Faxa_rainl Faxa_snowc Faxa_snowl Faxa_lwdn Faxa_swndr Faxa_swvdr Faxa_swndf "
            "Faxa_swvdf Faxa_swnet").split() + _AEROSOLS,
    "i2x": ("Si_avsdr Si_anidr Si_avsdf Si_anidf Si_tref Si_qref Si_t Si_snowh Si_u10 Si_u10withgusts "
            "Si_ifrac Si_bpress Si_ithick Faii_swnet Fioi_swpen Faii_taux Fioi_taux Faii_tauy Fioi_tauy "
            "Faii_lat Faii_sen Faii_lwup Faii_evap Fioi_melth Fioi_meltw PFioi_bergh PFioi_bergw Fioi_salt "
            "Fioi_bcphi Fioi_bcpho Fioi_flxdst").split(),
    "r2x": "Forr_rofl Forr_rofi Firr_rofi Flrr_flood Flrr_volr Flrr_volrmch Flrr_supply Flrr_deficit".split(),
    "w2x": [],
    "g2x": ("Sg_icemask Sg_icemask_coupled_fluxes Sg_ice_covered Sg_topo Sg_blit Sg_blis Sg_lithop "
            "Sg_icemask_grounded Sg_icemask_floating Sg_tbot Sg_dztbot Fogg_rofl Fogg_rofi Figg_rofi "
            "Flgg_hflx Fogx_qicelo Fogx_qiceho").split(),
    "xao": ("So_avsdr So_anidr So_avsdf So_anidf So_tref So_qref So_ssq So_re So_u10 So_u10withgusts "
            "So_duu10n So_ustar So_fswpen Faox_taux Faox_tauy Faox_lat Faox_sen Faox_lwup Faox_evap "
            "Faox_swdn Faox_swup So_tbulk_diurn So_tskin_diurn So_tskin_day_diurn So_tskin_night_diurn "
            "So_cskin_diurn So_cskin_night_diurn So_tod_diurn So_qbot_diurn So_fswpen_diurn So_ustar_diurn "
            "So_cdbot_diurn So_dwarm_diurn So_twarm_diurn So_dcool_diurn So_tcool_diurn So_warmmax_diurn "
            "So_windmax_diurn So_qsun_diurn So_ninc_diurn").split(),
    "fractions": "afrac ifrac ofrac ifrad ofrad".split(),
    "x2o": ("Sa_pbot Sa_pslv So_duu10n Si_ifrac Si_bpress Sg_icemask Sg_blit Sg_blis Sg_lithop "
            "Sg_icemask_grounded Sg_icemask_floating Sg_tbot Sg_dztbot Faxa_rain Faxa_snow Faxa_prec "
            "Faxa_lwdn Foxx_swnet").split() + _AEROSOLS + (
            "Foxx_taux Foxx_tauy Foxx_lat Foxx_sen Foxx_lwup Foxx_evap Fioi_melth Fioi_meltw PFioi_bergh "
            "PFioi_bergw Fioi_salt Fioi_bcphi Fioi_bcpho Fioi_flxdst Foxx_rofl Foxx_rofi Fogx_qicelo "
            "Fogx_qiceho").split(),
}
NUTRIENTS = ("DIN", "DIP", "DON", "DOP", "DOC", "PP", "DSi", "POC", "PN", "DIC", "Fe")


def _with(base: dict, **extra) -> dict:
    out = {k: list(v) for k, v in base.items()}
    for vector, names in extra.items():
        out[vector] = out[vector] + list(names)
    return out


def _without(base: dict, vector: str, name: str) -> dict:
    out = {k: list(v) for k, v in base.items()}
    out[vector].remove(name)
    return out


#: the branches a G case does not reach
BRANCHES = _with(
    GCASE,
    r2x=[f"Forr_rof{t}" for t in NUTRIENTS],
    w2x=["Sw_lamult", "Sw_ustokes", "Sw_vstokes", "Sw_hstokes"],
    i2x=["Sx_shared"],
    x2o=[f"Foxx_rof{t}" for t in NUTRIENTS] + ["Sf_afrac", "Sf_afracr", "Foxx_swnet_afracr", "Sw_lamult",
                                                "Sw_ustokes", "Sx_tref", "Sx_qref", "Sx_shared"],
)

#: (name, lists, switches, lsize, ncalls, aborts)
_SPECS = (
    ("gcase", GCASE, dict(flux_epbalfact=1.0, per_cat=False, nutrients=False, wav="none"), 61, 3, False),
    ("gcase_epbal", GCASE, dict(flux_epbalfact=0.9873046875000123, per_cat=False, nutrients=False, wav="none"),
     40, 2, False),
    ("branches", BRANCHES, dict(flux_epbalfact=1.0371, per_cat=True, nutrients=True, wav="twoway"), 53, 2, False),
    ("branches_wave_oneway", BRANCHES, dict(flux_epbalfact=1.0, per_cat=False, nutrients=False, wav="none"),
     29, 2, False),
    ("abort_multiple_matches", _with(GCASE, a2x=["Fbxa_lwdn"]),
     dict(flux_epbalfact=1.0, per_cat=False, nutrients=False, wav="none"), 5, 1, True),
    ("abort_atm_and_xao", _with(GCASE, a2x=["Faxa_lat"]),
     dict(flux_epbalfact=1.0, per_cat=False, nutrients=False, wav="none"), 5, 1, True),
    ("abort_merge_logic", _with(GCASE, i2x=["Fioi_lwdn"]),
     dict(flux_epbalfact=1.0, per_cat=False, nutrients=False, wav="none"), 5, 1, True),
    ("abort_missing_field", _without(GCASE, "a2x", "Faxa_swvdr"),
     dict(flux_epbalfact=1.0, per_cat=False, nutrients=False, wav="none"), 5, 1, True),
)


def _uniform(rng, low: float, high: float, n) -> np.ndarray:
    """Uniform draws built from ``rng.random`` with separate numpy operations:
    ``Generator.uniform`` and ``normal`` are C code a compiler may contract
    into fused multiply-adds (clang does on arm64), so their words differ
    between machines; ``random`` is exact and numpy's ufuncs round each
    operation."""
    return (rng.random(n) * (high - low)) + low


def _fractions(rng, lsize: int, names: list[str]) -> np.ndarray:
    """afrac, ifrac, ofrac, ifrad, ofrad: open ocean, full ice, partial ice,
    land (both zero), sums other than one, radiative fractions of their own."""
    out = np.zeros((lsize, len(names)))
    col = {n: names.index(n) for n in names}
    kind = np.arange(lsize) % 6
    ifrac = np.where(kind == 0, 0.0, np.where(kind == 1, 1.0, _uniform(rng, 0.0, 1.0, lsize)))
    ofrac = np.where(kind == 1, 0.0, 1.0 - ifrac)
    ofrac = np.where(kind == 3, _uniform(rng, 0.1, 0.6, lsize), ofrac)  # the sum is not one
    ifrac = np.where(kind == 2, 0.0, ifrac)
    ofrac = np.where(kind == 2, 0.0, ofrac)  # land: both zero, left as they are
    ifrad = np.where(kind == 4, np.clip(ifrac + 0.137, 0, 1), ifrac)
    ofrad = np.where(kind == 4, 1.0 - ifrad, ofrac)
    ofrad = np.where(kind == 5, ofrad * 0.75, ofrad)  # radiative sum not one
    for name, values in (("ifrac", ifrac), ("ofrac", ofrac), ("ifrad", ifrad), ("ofrad", ofrad),
                         ("afrac", _uniform(rng, 0, 1, lsize))):
        if name in col:
            out[:, col[name]] = values
    return out


def _vector(rng, names: list[str]) -> callable:
    def make(lsize: int) -> np.ndarray:
        out = _uniform(rng, -300.0, 300.0, (lsize, len(names)))
        for k, name in enumerate(names):
            if name.startswith(("So_a", "Si_a")):  # albedos
                out[:, k] = _uniform(rng, 0.02, 0.9, lsize)
            elif name.startswith("Faxa_sw"):
                out[:, k] = _uniform(rng, 0.0, 400.0, lsize)
            elif name.startswith(("Faxa_rain", "Faxa_snow", "Forr_", "Fogg_", "Flrr_")):
                out[:, k] = _uniform(rng, 0.0, 1.0e-4, lsize)
        return out
    return make


def cases() -> list[dict]:
    out = []
    for seed, (name, lists, switches, lsize, ncalls, aborts) in enumerate(_SPECS):
        rng = np.random.default_rng(20261120 + seed)
        calls = []
        for _ in range(ncalls):
            call = {}
            for vector in VECTORS:
                names = list(lists[vector])
                call[vector] = (_fractions(rng, lsize, names) if vector == "fractions"
                                else _vector(rng, names)(lsize))
            calls.append(call)
        out.append({"name": name, "lists": {k: list(v) for k, v in lists.items()}, "switches": dict(switches),
                    "lsize": lsize, "calls": calls, "aborts": aborts})
    return out


def input_bytes(case: dict) -> bytes:
    """run_ocean_merge.F90's stream."""
    sw = case["switches"]
    parts = [struct.pack("<dii8sii", sw["flux_epbalfact"], int(sw["per_cat"]), int(sw["nutrients"]),
                         sw["wav"].ljust(8).encode(), case["lsize"], len(case["calls"]))]
    for vector in VECTORS:
        rlist = ":".join(case["lists"][vector]).encode()
        parts.append(struct.pack("<i", len(rlist)) + rlist)
    for call in case["calls"]:
        for vector in VECTORS:
            if case["lists"][vector]:
                parts.append(np.ascontiguousarray(call[vector], dtype="<f8").tobytes())
    return b"".join(parts)


def input_digest(case: dict) -> str:
    return hashlib.sha256(input_bytes(case)).hexdigest()


def read_outputs(raw: bytes, case: dict) -> list[np.ndarray]:
    no, lsize = len(case["lists"]["x2o"]), case["lsize"]
    words = np.frombuffer(raw, dtype="<f8")
    assert words.size == no * lsize * len(case["calls"]), (case["name"], words.size)
    return [words[i * no * lsize:(i + 1) * no * lsize].reshape(lsize, no).copy() for i in range(len(case["calls"]))]


def plan_for(case: dict):
    from woof.hex.coupled import prep_ocn_merge_e3sm as pom
    sw = case["switches"]
    return pom.plan(case["lists"], nutrients=sw["nutrients"], per_cat=sw["per_cat"], wav_ocn_coup=sw["wav"])


def run_gpu(case: dict) -> list[np.ndarray]:
    """The port over the case's calls (one plan, as the Fortran's first call)."""
    import cupy as cp
    from woof.hex.coupled import prep_ocn_merge_e3sm as pom
    the_plan = plan_for(case)
    out = []
    for call in case["calls"]:
        vectors = {v: cp.asarray(call[v]) for v in pom.SOURCES if case["lists"][v]}
        x2o = cp.asarray(call["x2o"])
        pom.merge(the_plan, vectors, x2o, flux_epbalfact=case["switches"]["flux_epbalfact"])
        out.append(cp.asnumpy(x2o))
    return out


def word_mismatches(a: np.ndarray, b: np.ndarray) -> int:
    """Words that differ as bit patterns (so -0.0 != 0.0)."""
    a, b = np.ascontiguousarray(a), np.ascontiguousarray(b)
    if a.shape != b.shape or a.dtype.itemsize != b.dtype.itemsize:
        return int(max(a.size, b.size))
    word = np.dtype(f"u{a.dtype.itemsize}")
    return int(np.count_nonzero(a.view(word) != b.view(word)))


def load_fixture() -> tuple[dict[str, list[np.ndarray]], dict]:
    """The oracle's x2o per case and call, and the aborts (case -> the Fortran's stderr line)."""
    data = np.load(FIXTURE_DIR / "oracle-r8.npz")
    outputs: dict[str, list[np.ndarray]] = {}
    for key in sorted(data.files, key=lambda k: (k.split("/")[0], int(k.split("/call")[1]))):
        outputs.setdefault(key.split("/")[0], []).append(data[key])
    aborts = json.loads((FIXTURE_DIR / "aborts.json").read_text(encoding="utf-8"))
    return outputs, aborts


def measure() -> dict[str, dict[str, int]]:
    """case -> x2o field -> words that differ from the oracle, over the calls."""
    outputs, _ = load_fixture()
    report = {}
    for case in cases():
        if case["aborts"]:
            continue
        got = run_gpu(case)
        names = case["lists"]["x2o"]
        report[case["name"]] = {name: sum(word_mismatches(g[:, k], w[:, k]) for g, w in zip(got, outputs[case["name"]]))
                                for k, name in enumerate(names)}
    return report
