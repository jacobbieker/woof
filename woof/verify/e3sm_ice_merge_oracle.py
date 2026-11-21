"""E3SM v3.2.0's sea-ice import merge: cases, the oracle's stream, the GPU run.

``tools/e3sm_coupler_oracle/run_ice_merge.F90`` drives E3SM v3.2.0's
``prep_ice_merge`` (cut verbatim out of ``driver-mct/main/prep_ice_mod.F90``)
on the pinned MCT's attribute vectors, built with gfortran -O0
-ffp-contract=off. Each case is one process: the field lists, the switches
(``flux_epbalfact``, ``wav_ice_coup``) and several calls' worth of attribute
vectors (the first call sets MCT's shared indices, the later ones reuse
them, as in a run). x2i is an input too: the merge does not zero it.

The lists are modelled on what ``seq_flds_mod`` registers for an MPAS
ocean + sea-ice case with a data atmosphere and runoff (``GCASE``), plus
synthetic ones for the branches a G case does not reach (the waves coupled
two ways and one way, a field two senders carry, a field the merge then
overwrites, an x2i field no sender carries) and for MCT's stops.
"""
from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import numpy as np

from woof.verify.e3sm_ocean_merge_oracle import GCASE as _OCN_GCASE
from woof.verify.e3sm_ocean_merge_oracle import _uniform, word_mismatches  # noqa: F401
from woof.verify.e3sm_seaice_coupling_oracle import AEROSOL_X2I, X2I_BASE

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "tests" / "data" / "oracles" / "e3sm_ice_merge"
VECTORS = ("a2x", "o2x", "r2x", "g2x", "w2x", "x2i")

GCASE = {
    "a2x": list(_OCN_GCASE["a2x"]),
    "o2x": ("So_t So_s So_u So_v So_dhdx So_dhdy So_bldepth Fioo_q Fioo_frazil Fioo_meltw Fioo_meltw_corr "
            "So_ssh So_fswpen Faoo_fco2_ocn Faoo_fdms_ocn").split(),
    "r2x": list(_OCN_GCASE["r2x"]),
    "g2x": list(_OCN_GCASE["g2x"]),
    "w2x": [],
    "x2i": X2I_BASE[:9] + AEROSOL_X2I[:7] + X2I_BASE[9:] + AEROSOL_X2I[7:] + ["Fixx_rofi"],
}


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
    w2x=["Sw_ustokes", "Sw_vstokes", "Sw_elevation_spectrum1", "Sw_elevation_spectrum2", "So_u"],
    o2x=["Sx_shared"],  # also a2x's: a2x's copy lands last
    a2x=["Sx_shared", "Faxa_rain"],  # a2x's own Faxa_rain: copied, then overwritten by the sum
    g2x=["Sg_shared_ice"],
    x2i=["Sw_ustokes", "Sw_vstokes", "Sw_elevation_spectrum1", "Sw_elevation_spectrum2", "Sx_shared",
         "Sg_shared_ice", "Sx_nobody"],  # Sx_nobody: no sender, keeps x2i's word
)

#: (name, lists, flux_epbalfact, wav_ice_coup, lsize, ncalls, aborts)
_SPECS = (
    ("gcase", GCASE, 1.0, "none", 61, 3, False),
    ("gcase_epbal", GCASE, 0.9873046875000123, "none", 40, 2, False),
    ("branches_wave_twoway", BRANCHES, 1.0371, "twoway", 53, 2, False),
    ("branches_wave_oneway", BRANCHES, 1.0, "oneway", 29, 2, False),
    ("abort_missing_a2x_snowc", _without(GCASE, "a2x", "Faxa_snowc"), 1.0, "none", 5, 1, True),
    ("abort_missing_r2x_rofi", _without(GCASE, "r2x", "Firr_rofi"), 1.0, "none", 5, 1, True),
    ("abort_missing_x2i_rofi", _without(GCASE, "x2i", "Fixx_rofi"), 1.0, "none", 5, 1, True),
)


def _vector(rng, names: list[str], lsize: int) -> np.ndarray:
    out = _uniform(rng, -300.0, 300.0, (lsize, len(names)))
    for k, name in enumerate(names):
        if name.startswith(("Faxa_rain", "Faxa_snow", "Forr_", "Firr_", "Fogg_", "Figg_", "Flrr_", "Fixx_")):
            out[:, k] = _uniform(rng, 0.0, 1.0e-4, lsize)
    # signed zeros in the sums' terms: -0 + -0 stays -0, -0 + 0 is +0
    for name in ("Faxa_rainc", "Faxa_rainl", "Faxa_snowc", "Faxa_snowl", "Figg_rofi", "Firr_rofi"):
        if name in names and lsize > 4:
            k = names.index(name)
            out[0, k], out[1, k] = -0.0, (0.0 if name in ("Faxa_rainl", "Firr_rofi") else -0.0)
            out[2, k] = 0.0
    return out


def cases() -> list[dict]:
    out = []
    for seed, (name, lists, epbal, wav, lsize, ncalls, aborts) in enumerate(_SPECS):
        rng = np.random.default_rng(20261121 + seed)
        calls = [{v: _vector(rng, list(lists[v]), lsize) for v in VECTORS} for _ in range(ncalls)]
        out.append({"name": name, "lists": {k: list(v) for k, v in lists.items()}, "flux_epbalfact": epbal,
                    "wav": wav, "lsize": lsize, "calls": calls, "aborts": aborts})
    return out


def input_bytes(case: dict) -> bytes:
    """run_ice_merge.F90's stream."""
    parts = [struct.pack("<d8sii", case["flux_epbalfact"], case["wav"].ljust(8).encode(), case["lsize"],
                         len(case["calls"]))]
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
    ni, lsize = len(case["lists"]["x2i"]), case["lsize"]
    words = np.frombuffer(raw, dtype="<f8")
    assert words.size == ni * lsize * len(case["calls"]), (case["name"], words.size)
    return [words[i * ni * lsize:(i + 1) * ni * lsize].reshape(lsize, ni).copy() for i in range(len(case["calls"]))]


def plan_for(case: dict):
    from woof.hex.coupled import prep_ice_merge_e3sm as pim
    return pim.plan(case["lists"], wav_ice_coup=case["wav"])


def run_gpu(case: dict) -> list[np.ndarray]:
    """The port over the case's calls (one plan, as the Fortran's first call)."""
    import cupy as cp
    from woof.hex.coupled import prep_ice_merge_e3sm as pim
    the_plan = plan_for(case)
    out = []
    for call in case["calls"]:
        vectors = {v: cp.asarray(call[v]) for v in pim.SOURCES if case["lists"][v]}
        x2i = cp.asarray(call["x2i"])
        pim.merge(the_plan, vectors, x2i, flux_epbalfact=case["flux_epbalfact"])
        out.append(cp.asnumpy(x2i))
    return out


def load_fixture() -> tuple[dict[str, list[np.ndarray]], dict]:
    """The oracle's x2i per case and call, and the aborts (case -> MCT's stderr line)."""
    data = np.load(FIXTURE_DIR / "oracle-r8.npz")
    outputs: dict[str, list[np.ndarray]] = {}
    for key in sorted(data.files, key=lambda k: (k.split("/")[0], int(k.split("/call")[1]))):
        outputs.setdefault(key.split("/")[0], []).append(data[key])
    aborts = json.loads((FIXTURE_DIR / "aborts.json").read_text(encoding="utf-8"))
    return outputs, aborts


def measure() -> dict[str, dict[str, int]]:
    """case -> x2i field -> words that differ from the oracle, over the calls."""
    outputs, _ = load_fixture()
    report = {}
    for case in cases():
        if case["aborts"]:
            continue
        got = run_gpu(case)
        names = case["lists"]["x2i"]
        report[case["name"]] = {name: sum(word_mismatches(g[:, k], w[:, k]) for g, w in zip(got, outputs[case["name"]]))
                                for k, name in enumerate(names)}
    return report
