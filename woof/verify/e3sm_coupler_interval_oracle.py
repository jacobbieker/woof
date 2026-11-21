"""One E3SM v3.2.0 coupler interval, chained: cases, stream, GPU run.

``tools/e3sm_coupler_oracle/run_coupler_interval.F90`` links E3SM v3.2.0's
own coupler routines, each cut verbatim as its single-routine oracle cuts it
(``prep_ocn_mod.F90``'s ``prep_ocn_calc_i2x_ox``, ``prep_ocn_mrg``,
``prep_ocn_merge``, ``prep_ocn_accum`` and ``prep_ocn_accum_avg``;
``prep_ice_mod.F90``'s ``prep_ice_calc_o2x_ix``, ``prep_ice_calc_a2x_ix``,
``prep_ice_mrg`` and ``prep_ice_merge``; ``seq_flux_mct.F90``'s
``seq_flux_init_mct``, ``seq_flux_atmocn_mct`` and ``seq_flux_ocnalb_mct``
over ``shr_flux_mod``; ``seq_frac_mct.F90``'s ``seq_frac_set`` over
``seq_map_mod``), and drives them over several coupler steps in
``cime_comp_mod``'s RASM_OPTION1 order on the pinned MCT, one task, built
with gfortran -O0 -ffp-contract=off. The atmosphere's export (on the ocean
grid) and the ocean's and sea ice's new exports are scripted. After each
step the oracle writes xao, the accumulator and its count, x2o, x2i and the
three fractions.

What the chain adds over the single-routine oracles is the state carried
between routines and steps: the flux's mask from the starting fractions;
xao written by the flux, then the albedos, read by the next step's merge;
the fractions seq_frac_set leaves read by the next merge and albedos; the
accumulation windows the ocean's alarms cut; i2x_ox, a2x_ix and o2x_ix
taken when the coupler takes them (and left zero when a switch is off); the
exports changing between phases of a step.
"""
from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import numpy as np

from woof.verify.e3sm_flux_atmocn_oracle import XAO_FIELDS as _FLUX_XAO
from woof.verify.e3sm_ice_merge_oracle import GCASE as _ICE_GCASE
from woof.verify.e3sm_ocean_merge_oracle import GCASE as _OCN_GCASE
from woof.verify.e3sm_ocean_merge_oracle import _fractions, _uniform, _vector
from woof.verify.e3sm_ocean_merge_oracle import word_mismatches  # noqa: F401
from woof.verify.e3sm_seq_frac_oracle import FA_FIELDS, FI_FIELDS, FO_FIELDS, KIND_CODES, _mapped, _matrix

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "tests" / "data" / "oracles" / "e3sm_coupler_interval"

DOM_COORDS = ("lat", "lon", "hgt")
DOM_OTHER = ("area", "aream", "mask", "frac")
#: the merge's atmosphere fields and the flux's response, stress estimate and gust
A2X = tuple(_OCN_GCASE["a2x"]) + ("Sa_wsresp", "Sa_tau_est", "Sa_ugust")
O2X = tuple(_ICE_GCASE["o2x"])
I2X = tuple(_OCN_GCASE["i2x"])
XAO = tuple(_OCN_GCASE["xao"]) + tuple(f for f in _FLUX_XAO if f not in _OCN_GCASE["xao"])
LISTS = {
    "dom": DOM_COORDS + DOM_OTHER, "a2x": A2X, "o2x": O2X, "i2x": I2X, "xao": XAO, "fractions_o": FO_FIELDS,
    "fractions_i": FI_FIELDS, "fractions_a": FA_FIELDS, "r2x_o": tuple(_OCN_GCASE["r2x"]),
    "w2x_o": tuple(_OCN_GCASE["w2x"]), "g2x_o": tuple(_OCN_GCASE["g2x"]), "r2x_i": tuple(_ICE_GCASE["r2x"]),
    "g2x_i": tuple(_ICE_GCASE["g2x"]), "w2x_i": tuple(_ICE_GCASE["w2x"]), "x2o": tuple(_OCN_GCASE["x2o"]),
    "x2i": tuple(_ICE_GCASE["x2i"]),
}
#: the stream's order after the domain
STREAM = ("a2x", "o2x", "i2x", "xao", "fractions_o", "fractions_i", "fractions_a", "r2x_o", "w2x_o", "g2x_o",
          "r2x_i", "g2x_i", "w2x_i", "x2o", "x2i")
OUTPUTS = ("xao", "acc", "x2o", "x2i", "fractions_o", "fractions_i", "fractions_a")


def _spec(name, *, seed, steps, ocn_every=3, ice_every=1, n=16, n_a=11, i2a="matrix", implicit=False, gust=False,
          dead=False, albav=False, epbal=1.0, ocn_prognostic=True, ice_prognostic=True, ice_c2_ocn=True,
          ocn_c2_ice=True, atm_c2_ice=True):
    return dict(name=name, seed=seed, steps=steps, ocn_every=ocn_every, ice_every=ice_every, n=n,
                n_a=n if i2a != "matrix" else n_a, i2a=i2a, implicit=implicit, gust=gust, dead=dead, albav=albav,
                epbal=epbal, ocn_prognostic=ocn_prognostic, ice_prognostic=ice_prognostic, ice_c2_ocn=ice_c2_ocn,
                ocn_c2_ice=ocn_c2_ice, atm_c2_ice=atm_c2_ice)


SPECS = (
    # the ocean every third coupler step, the sea ice every step: windows of 1, 3 and 3 samples
    _spec("ocean_every_third", seed=1, steps=7),
    _spec("albav_ice_every_other_epbal", seed=2, steps=4, ocn_every=2, ice_every=2, i2a="copy", albav=True,
          epbal=0.9873046875000123),
    _spec("implicit_gust", seed=3, steps=4, ocn_every=4, implicit=True, gust=True),
    _spec("dead_models", seed=4, steps=3, ocn_every=1, dead=True),
    _spec("ocean_not_prognostic", seed=5, steps=4, ocn_prognostic=False, i2a="rearrange"),
    _spec("no_component_maps", seed=6, steps=4, ocn_every=2, ice_c2_ocn=False, ocn_c2_ice=False, atm_c2_ice=False),
    _spec("ice_not_prognostic", seed=7, steps=3, ice_prognostic=False),
)


def alarms(spec) -> list[tuple[bool, bool, bool]]:
    """Per step: the ocean's run alarm (its interval starts), the sea ice's,
    and the ocean's next alarm (its interval's last coupler step)."""
    k, m = spec["ocn_every"], spec["ice_every"]
    return [(s % k == 0, s % m == 0, s % k == k - 1) for s in range(spec["steps"])]


def _a2x(rng, n) -> np.ndarray:
    a = {f: _uniform(rng, 0.0, 1.0e-6, n) for f in A2X}  # aerosols and the rest: small and positive
    a["Sa_topo"] = _uniform(rng, 0.0, 10.0, n)
    a["Sa_z"] = _uniform(rng, 5.0, 60.0, n)
    a["Sa_u"] = _uniform(rng, -15.0, 15.0, n)
    a["Sa_v"] = _uniform(rng, -15.0, 15.0, n)
    a["Sa_ptem"] = _uniform(rng, 255.0, 310.0, n)
    a["Sa_tbot"] = a["Sa_ptem"] - _uniform(rng, 0.0, 3.0, n)
    a["Sa_shum"] = _uniform(rng, 1.0e-4, 2.0e-2, n)
    a["Sa_pbot"] = _uniform(rng, 9.5e4, 1.03e5, n)
    a["Sa_dens"] = _uniform(rng, 1.0, 1.35, n)
    a["Sa_pslv"] = _uniform(rng, 9.6e4, 1.04e5, n)
    a["Sa_uovern"] = _uniform(rng, -1.0, 1.0, n)
    a["Sa_wsresp"] = _uniform(rng, 0.0, 3.0, n)
    a["Sa_tau_est"] = _uniform(rng, 0.0, 0.6, n)
    a["Sa_ugust"] = _uniform(rng, 0.0, 4.0, n)
    a["Faxa_lwdn"] = _uniform(rng, 150.0, 450.0, n)
    for f in ("Faxa_swndr", "Faxa_swvdr", "Faxa_swndf", "Faxa_swvdf", "Faxa_swnet"):
        a[f] = _uniform(rng, 0.0, 400.0, n)
    for f in ("Faxa_rainc", "Faxa_rainl", "Faxa_snowc", "Faxa_snowl"):
        a[f] = _uniform(rng, 0.0, 1.0e-4, n)
    return np.column_stack([a[f] for f in A2X])


def _o2x(rng, n) -> np.ndarray:
    o = {f: _uniform(rng, -1.0e-3, 1.0e-3, n) for f in O2X}
    o["So_t"] = _uniform(rng, 271.0, 305.0, n)
    o["So_s"] = _uniform(rng, 30.0, 37.0, n)
    o["So_u"] = _uniform(rng, -1.0, 1.0, n)
    o["So_v"] = _uniform(rng, -1.0, 1.0, n)
    o["So_bldepth"] = _uniform(rng, 5.0, 50.0, n)
    o["So_fswpen"] = _uniform(rng, 0.0, 1.0, n)
    o["Fioo_q"] = _uniform(rng, -100.0, 100.0, n)
    o["So_ssh"] = _uniform(rng, -1.0, 1.0, n)
    return np.column_stack([o[f] for f in O2X])


def _i2x(rng, n) -> np.ndarray:
    i2x = _vector(rng, list(I2X))(n)
    sif = _uniform(rng, 0.0, 1.0, n)
    sif[:4] = (0.0, -0.0, 1.0, 1.0e-300)
    i2x[:, I2X.index("Si_ifrac")] = sif
    return i2x


def _build(spec) -> dict:
    rng = np.random.default_rng(7000 + spec["seed"])
    n, na = spec["n"], spec["n_a"]
    dom = _uniform(rng, -1.0, 1.0, (n, len(LISTS["dom"])))
    col = LISTS["dom"].index
    dom[:, col("mask")] = 1.0
    dom[[9, 15], col("mask")] = 0.0
    frac = np.ones(n)
    frac[[11, 12]] = 0.0                                           # land
    frac[[13, 14]] = _uniform(rng, 0.05, 0.95, 2)                  # coast
    dom[:, col("frac")] = frac
    matrix = _matrix(rng, na, n, empty=2) if spec["i2a"] == "matrix" else None
    lfrac = np.clip((1.0 - _mapped(spec["i2a"], matrix, frac, na)) + _uniform(rng, -0.004, 0.004, na), 0.0, 1.0)
    fo = _fractions(rng, n, list(FO_FIELDS))
    fo[[5, 6], FO_FIELDS.index("ifrac")] = 0.0
    fo[[5, 6], FO_FIELDS.index("ofrac")] = 0.0                     # the flux's mask drops them
    xao = _uniform(rng, -5.0, 5.0, (n, len(XAO)))
    for f in ("So_avsdr", "So_anidr", "So_avsdf", "So_anidf"):
        xao[:, XAO.index(f)] = _uniform(rng, 0.02, 0.9, n)
    xao[[7, 8], XAO.index("So_anidr")] = 1.0                       # the dark side: no shortwave
    values = {
        "dom": dom, "o2x": _o2x(rng, n), "i2x": _i2x(rng, n), "xao": xao, "fractions_o": fo,
        "fractions_i": np.column_stack([_uniform(rng, 0.0, 1.0, n) for _ in FI_FIELDS]),
        "fractions_a": np.column_stack([_uniform(rng, 0.0, 1.0, na), np.full(na, 7.5), np.full(na, -7.5), lfrac,
                                        _uniform(rng, 0.0, 1.0, na)]),
    }
    for k in ("r2x_o", "w2x_o", "g2x_o", "r2x_i", "g2x_i", "w2x_i", "x2o", "x2i"):
        values[k] = _vector(rng, list(LISTS[k]))(n)
    steps = []
    for ocn_run, ice_run, ocn_next in alarms(spec):
        steps.append(dict(a2x=_a2x(rng, n), i2x=_i2x(rng, n) if ice_run else None,
                          o2x=_o2x(rng, n) if ocn_next else None))
    return dict(name=spec["name"], spec=spec, aborts=False, lists=LISTS, values=values, steps=steps,
                matrix=matrix)


def cases() -> list[dict]:
    return [_build(s) for s in SPECS]


def _block(names, array=None) -> bytes:
    text = ":".join(names).encode()
    out = struct.pack("<i", len(text)) + text
    if array is not None and len(names):
        out += np.ascontiguousarray(array, dtype="<f8").tobytes()
    return out


def input_bytes(case: dict) -> bytes:
    s, v = case["spec"], case["values"]
    out = [struct.pack("<8i", int(s["dead"]), int(s["implicit"]), int(s["gust"]), 0, -1, 0, 0, 0),
           struct.pack("<8i", int(s["ocn_prognostic"]), int(s["ice_prognostic"]), int(s["ice_c2_ocn"]),
                       int(s["ocn_c2_ice"]), int(s["atm_c2_ice"]), int(s["albav"]), 0, KIND_CODES[s["i2a"]]),
           struct.pack("<3i", s["n"], s["n_a"], s["steps"]),
           struct.pack("<6d", 0.0, 0.5, 0.06, 0.07, 120.25 if s["albav"] else -1.0, s["epbal"]),
           _block(DOM_OTHER, v["dom"])]
    for k in STREAM:
        out.append(_block(case["lists"][k], None if k == "a2x" else v[k]))
    if case["matrix"] is not None:
        rows, cols, w = case["matrix"]
        out.append(struct.pack("<i", rows.size) + rows.astype("<i4").tobytes() + cols.astype("<i4").tobytes()
                   + w.astype("<f8").tobytes())
    for (ocn_run, ice_run, ocn_next), step in zip(alarms(s), case["steps"]):
        out.append(struct.pack("<3i", int(ocn_run), int(ice_run), int(ocn_next)))
        out.append(np.ascontiguousarray(step["a2x"], dtype="<f8").tobytes())
        if ice_run:
            out.append(np.ascontiguousarray(step["i2x"], dtype="<f8").tobytes())
        if ocn_next:
            out.append(np.ascontiguousarray(step["o2x"], dtype="<f8").tobytes())
    return b"".join(out)


def input_digest(case: dict) -> str:
    return hashlib.sha256(input_bytes(case)).hexdigest()


def _shapes(case):
    s, lists = case["spec"], case["lists"]
    return {"xao": (s["n"], len(lists["xao"])), "acc": (s["n"], len(lists["x2o"])),
            "x2o": (s["n"], len(lists["x2o"])), "x2i": (s["n"], len(lists["x2i"])),
            "fractions_o": (s["n"], len(lists["fractions_o"])), "fractions_i": (s["n"], len(lists["fractions_i"])),
            "fractions_a": (s["n_a"], len(lists["fractions_a"]))}


def read_outputs(raw: bytes, case: dict) -> dict[str, np.ndarray]:
    shapes, out, counts, at = _shapes(case), {}, [], 0
    for step in range(case["spec"]["steps"]):
        for key in OUTPUTS:
            if key == "acc":
                counts.append(struct.unpack_from("<i", raw, at)[0])
                at += 4
            shape = shapes[key]
            size = shape[0] * shape[1]
            out[f"step{step}/{key}"] = np.frombuffer(raw, dtype="<f8", count=size, offset=at).reshape(shape).astype(
                np.float64)
            at += 8 * size
    if at != len(raw):
        raise ValueError(f"{case['name']}: {len(raw)} output bytes, expected {at}")
    out["counts"] = np.array(counts, dtype=np.int32)
    return out


def load_fixture() -> tuple[dict[str, dict[str, np.ndarray]], dict]:
    with np.load(FIXTURE_DIR / "oracle-r8.npz") as z:
        arrays = {k: z[k] for k in z.files}
    outputs: dict[str, dict[str, np.ndarray]] = {}
    for key, value in arrays.items():
        case, _, name = key.partition("/")
        outputs.setdefault(case, {})[name] = value
    aborts = json.loads((FIXTURE_DIR / "aborts.json").read_text(encoding="utf-8"))
    return outputs, aborts


def interval_for(case: dict):
    """The port's interval, set up as the oracle's driver sets it up."""
    from woof.hex.coupled import flux_atmocn_e3sm as fx
    from woof.hex.coupled import interval_e3sm as ie
    from woof.hex.coupled import seq_frac_e3sm as sf
    s = case["spec"]
    if s["i2a"] == "matrix":
        rows, cols, w = case["matrix"]
        mapper = sf.Mapper("matrix", rows, cols, w, n_dst=s["n_a"], n_src=s["n"])
    else:
        mapper = sf.Mapper(s["i2a"])
    flux = fx.FluxOptions(dead_comps=s["dead"], atm_flux_method="implicit_stress" if s["implicit"] else "explicit",
                          atm_gustiness=s["gust"])
    switches = ie.Switches(ocn_prognostic=s["ocn_prognostic"], ice_prognostic=s["ice_prognostic"],
                           ice_c2_ocn=s["ice_c2_ocn"], ocn_c2_ice=s["ocn_c2_ice"], atm_c2_ice=s["atm_c2_ice"],
                           flux_albav=s["albav"], nextsw_cday=120.25 if s["albav"] else -1.0,
                           flux_epbalfact=s["epbal"])
    return ie.CouplerInterval(case["lists"], case["values"], n=s["n"], n_a=s["n_a"], mapper_i2a=mapper, flux=flux,
                              switches=switches)


def run_gpu(case: dict) -> dict[str, np.ndarray]:
    """The port, step by step, as the oracle writes it."""
    import cupy as cp
    interval = interval_for(case)
    out, counts = {}, []
    for k, ((ocn_run, ice_run, ocn_next), step) in enumerate(zip(alarms(case["spec"]), case["steps"])):
        interval.step(cp.asarray(step["a2x"]), ocn_run=ocn_run, ice_run=ice_run, ocn_next=ocn_next,
                      i2x=None if step["i2x"] is None else cp.asarray(step["i2x"]),
                      o2x=None if step["o2x"] is None else cp.asarray(step["o2x"]))
        counts.append(interval.accumulator.count)
        for key in OUTPUTS:
            out[f"step{k}/{key}"] = cp.asnumpy(interval.vector(key))
    out["counts"] = np.array(counts, dtype=np.int32)
    return out


def measure() -> dict[str, dict[str, int]]:
    """case -> output -> words that differ from the oracle."""
    outputs, _ = load_fixture()
    report = {}
    for case in cases():
        got, want = run_gpu(case), outputs[case["name"]]
        report[case["name"]] = {k: word_mismatches(got[k], want[k]) if k != "counts"
                                else int((got[k] != want[k]).sum()) for k in want}
    return report
