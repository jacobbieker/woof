"""E3SM v3.2.0's coupler surface fractions update: cases, stream, GPU run.

``tools/e3sm_coupler_oracle/run_seq_frac.F90`` drives E3SM v3.2.0's
``seq_frac_set`` and ``seq_frac_check`` (cut verbatim out of
``driver-mct/main/seq_frac_mct.F90``) and the maps they make,
``seq_map_map`` and ``seq_map_avNormArr`` (cut out of ``seq_map_mod.F90``),
over the pinned MCT on one task, built with gfortran -O0 -ffp-contract=off.
Each case is one process: the three switches and ``seq_frac_dead``, the
two mappers' kinds (and a sparse matrix's elements, in the mapping file's
order), i2x, the ice domain and the three fraction vectors. The oracle
writes the fraction vectors after the call; ``seq_frac_check``'s lines go to
its log.

The cases reach: each mapper kind on each map (copy, rearrangement, sparse
matrix with unsorted and repeated elements, empty rows and a -0 product);
each component absent; ice fractions of 0, -0, 1, tiny, and exactly at the
lower bound (``-eps_fracval``, whose ``ofrac`` is exactly ``1 + eps``);
land and coastal ice domain points; i2x without ``Si_ifrac`` (MCT's copy
copies nothing); each of the three checks stopping the coupler (a bound and
the sum); and the dead models reporting all three without stopping.
"""
from __future__ import annotations

import hashlib
import re
import struct
from pathlib import Path

import numpy as np

from woof.verify.e3sm_ocean_merge_oracle import word_mismatches  # noqa: F401

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "tests" / "data" / "oracles" / "e3sm_seq_frac"

#: seq_frac_init's lists (v3.2.0) and seq_flds_mod's domain fields
FI_FIELDS = ("afrac", "ifrac", "ofrac")
FO_FIELDS = ("afrac", "ifrac", "ofrac", "ifrad", "ofrad")
FA_FIELDS = ("afrac", "ifrac", "ofrac", "lfrac", "lfrin")
DOM_COORDS = ("lat", "lon", "hgt")
DOM_OTHER = ("area", "aream", "mask", "frac")
#: a few of the sea ice's i2x fields, Si_ifrac not first
I2X_FIELDS = ("Si_t", "Si_avsdr", "Si_ifrac", "Si_snowh")
KIND_CODES = {"copy": 0, "rearrange": 1, "matrix": 2}


def _spec(name, *, seed, ni=37, no=None, na=23, i2o="rearrange", i2a="matrix", atm=True, ice=True, ocn=True,
          dead=False, aborts=False, poke=(), dom_other=DOM_OTHER, i2x_fields=I2X_FIELDS):
    no = ni if no is None or i2o != "matrix" else no
    na = ni if i2a != "matrix" else na
    return dict(name=name, seed=seed, ni=ni, no=no, na=na, i2o=i2o, i2a=i2a, atm=atm,
                ice=ice, ocn=ocn, dead=dead, aborts=aborts, poke=tuple(poke), dom_other=tuple(dom_other),
                i2x_fields=tuple(i2x_fields))


_BELOW = float(np.nextafter(-0.01, -1.0))
SPECS = (
    _spec("mpas_default", seed=1),
    _spec("ocean_matrix_atm_copy", seed=2, ni=29, no=33, i2o="matrix", i2a="copy"),
    _spec("ocean_copy_atm_rearrange", seed=3, ni=19, i2o="copy", i2a="rearrange"),
    _spec("atm_absent", seed=4, atm=False),
    _spec("ocn_absent", seed=5, ocn=False),
    _spec("ice_absent", seed=6, ice=False),
    _spec("dead_reports_all", seed=7, dead=True,
          poke=(("Si_ifrac", 6, _BELOW), ("fo_afrac", 2, 1.0101), ("fa_lfrac_add", 4, 0.02))),
    _spec("abort_ice_bound", seed=8, aborts=True, poke=(("Si_ifrac", 6, _BELOW),)),
    _spec("abort_ocn_bound", seed=9, aborts=True, poke=(("fo_afrac", 2, 1.0101),)),
    _spec("abort_atm_sum", seed=10, aborts=True, poke=(("fa_lfrac_add", 4, 0.02),)),
    _spec("abort_atm_lfrin", seed=11, aborts=True, poke=(("fa_lfrin", 3, -0.0101),)),
    # no Si_ifrac: mct_aVect_copy copies nothing, so ifrac is the vector's own, times frac
    _spec("i2x_without_si_ifrac", seed=12, i2x_fields=("Si_t", "Si_avsdr", "Si_snowh")),
)


def _uniform(rng, low: float, high: float, n) -> np.ndarray:
    return (rng.random(n) * (high - low)) + low


def _matrix(rng, n_dst: int, n_src: int, empty: int):
    """Elements in a shuffled file order: rows of 1-4 columns with weights
    summing to about 1, ``empty`` rows with none, one element split in two
    (a repeated row and column)."""
    rows, cols, wts = [], [], []
    skip = set(rng.choice(n_dst, size=empty, replace=False).tolist())
    for r in range(n_dst):
        if r in skip:
            continue
        m = int(rng.integers(1, 5))
        c = rng.choice(n_src, size=m, replace=False)
        w = _uniform(rng, 0.05, 1.0, m)
        w = w / w.sum()
        for cj, wj in zip(c.tolist(), w.tolist()):
            if len(rows) % 7 == 3:
                part = wj * 0.25
                rows += [r + 1, r + 1]
                cols += [cj + 1, cj + 1]
                wts += [part, wj - part]
            else:
                rows.append(r + 1)
                cols.append(cj + 1)
                wts.append(wj)
    order = rng.permutation(len(rows))
    return (np.asarray(rows, dtype=np.int32)[order], np.asarray(cols, dtype=np.int32)[order],
            np.asarray(wts, dtype=np.float64)[order])


def _mapped(kind, matrix, x, n_dst):
    if kind != "matrix":
        return x.copy()
    rows, cols, w = matrix
    y = np.zeros(n_dst)
    np.add.at(y, rows - 1, w * x[cols - 1])
    return y


def _build(spec) -> dict:
    rng = np.random.default_rng(1000 + spec["seed"])
    ni, no, na = spec["ni"], spec["no"], spec["na"]
    dom_fields = DOM_COORDS + spec["dom_other"]
    dom = _uniform(rng, -1.0, 1.0, (ni, len(dom_fields)))
    frac = np.ones(ni)
    frac[rng.choice(np.arange(8, ni), size=4, replace=False)] = 0.0  # land
    coast = rng.choice(np.arange(8, ni), size=5, replace=False)
    frac[coast] = _uniform(rng, 0.05, 0.95, 5)
    if "frac" in spec["dom_other"]:
        dom[:, dom_fields.index("frac")] = frac
    i2x_fields = spec["i2x_fields"]
    i2x = _uniform(rng, -2.0, 300.0, (ni, len(i2x_fields)))
    sif = _uniform(rng, 0.0, 1.0, ni)
    sif[:6] = (0.0, -0.0, 1.0, 1e-300, -0.01, 0.5)  # -0.01: exactly the lower bound, ofrac exactly 1 + eps
    frac[:6] = 1.0
    if "frac" in spec["dom_other"]:
        dom[:6, dom_fields.index("frac")] = 1.0
    if "Si_ifrac" in i2x_fields:
        i2x[:, i2x_fields.index("Si_ifrac")] = sif
    matrices = {}
    for key, nd in (("i2o", no), ("i2a", na)):
        if spec[key] == "matrix":
            matrices[key] = _matrix(rng, nd, ni, empty=2)
    fi = np.column_stack([_uniform(rng, 0.0, 1.0, ni), np.full(ni, 9.5) if "Si_ifrac" in i2x_fields else sif,
                          np.full(ni, -9.5)])
    fo = np.column_stack([_uniform(rng, 0.0, 1.0, no), np.full(no, 8.5), np.full(no, -8.5),
                          _uniform(rng, 0.0, 1.0, no), _uniform(rng, 0.0, 1.0, no)])
    covered = _mapped(spec["i2a"], matrices.get("i2a"), frac, na)
    lfrac = np.clip((1.0 - covered) + _uniform(rng, -0.004, 0.004, na), 0.0, 1.0)
    fa = np.column_stack([_uniform(rng, 0.0, 1.0, na), np.full(na, 7.5), np.full(na, -7.5), lfrac,
                          _uniform(rng, 0.0, 1.0, na)])
    for what, k, v in spec["poke"]:
        if what == "Si_ifrac":
            i2x[k, i2x_fields.index("Si_ifrac")] = v
        elif what == "fo_afrac":
            fo[k, 0] = v
        elif what == "fa_lfrac_add":
            fa[k, 3] = fa[k, 3] + v
        elif what == "fa_lfrin":
            fa[k, 4] = v
    return dict(name=spec["name"], spec=spec, aborts=spec["aborts"], i2x_fields=i2x_fields, dom_fields=dom_fields,
                i2x=i2x, dom=dom, fractions_i=fi, fractions_o=fo, fractions_a=fa, matrices=matrices)


def cases() -> list[dict]:
    return [_build(s) for s in SPECS]


def input_bytes(case: dict) -> bytes:
    s = case["spec"]
    out = [struct.pack("<9i", int(s["atm"]), int(s["ice"]), int(s["ocn"]), int(s["dead"]),
                       KIND_CODES[s["i2o"]], KIND_CODES[s["i2a"]], s["ni"], s["no"], s["na"])]
    lists = (case["i2x_fields"], s["dom_other"], FI_FIELDS, FO_FIELDS, FA_FIELDS)
    arrays = (case["i2x"], case["dom"], case["fractions_i"], case["fractions_o"], case["fractions_a"])
    for names, a in zip(lists, arrays):
        text = ":".join(names).encode()
        out.append(struct.pack("<i", len(text)) + text)
        out.append(np.ascontiguousarray(a, dtype="<f8").tobytes())
    for key in ("i2o", "i2a"):
        if key in case["matrices"]:
            rows, cols, w = case["matrices"][key]
            out.append(struct.pack("<i", rows.size) + rows.astype("<i4").tobytes() + cols.astype("<i4").tobytes()
                       + w.astype("<f8").tobytes())
    return b"".join(out)


def input_digest(case: dict) -> str:
    return hashlib.sha256(input_bytes(case)).hexdigest()


def read_outputs(raw: bytes, case: dict) -> dict[str, np.ndarray]:
    s = case["spec"]
    out, at = {}, 0
    for key, n, nf in (("fractions_i", s["ni"], len(FI_FIELDS)), ("fractions_o", s["no"], len(FO_FIELDS)),
                       ("fractions_a", s["na"], len(FA_FIELDS))):
        out[key] = np.frombuffer(raw, dtype="<f8", count=n * nf, offset=at).reshape(n, nf).astype(np.float64)
        at += 8 * n * nf
    if at != len(raw):
        raise ValueError(f"{case['name']}: {len(raw)} output bytes, expected {at}")
    return out


_STAGE = re.compile(r"\(seq_frac_check\) \[([a-z ]+)\] ERROR (but NOT )?aborting")


def read_log(stderr: str) -> dict[str, np.ndarray]:
    """The stages ``seq_frac_check`` reported without stopping (dead models)."""
    return {"reported": np.array([m.group(1) for m in map(_STAGE.search, stderr.splitlines()) if m and m.group(2)],
                                 dtype="<U8")}


def abort_line(stderr: str) -> str:
    for line in stderr.splitlines():
        if _STAGE.search(line) or "FATAL" in line:
            return line.strip()
    raise RuntimeError(f"no abort line in {stderr!r}")


def load_fixture() -> tuple[dict[str, dict[str, np.ndarray]], dict]:
    import json
    with np.load(FIXTURE_DIR / "oracle-r8.npz") as z:
        arrays = {k: z[k] for k in z.files}
    outputs: dict[str, dict[str, np.ndarray]] = {}
    for key, value in arrays.items():
        case, _, name = key.partition("/")
        outputs.setdefault(case, {})[name] = value
    aborts = json.loads((FIXTURE_DIR / "aborts.json").read_text(encoding="utf-8"))
    return outputs, aborts


def mappers_for(case: dict):
    from woof.hex.coupled import seq_frac_e3sm as sf
    s = case["spec"]
    out = []
    for key, nd in (("i2o", s["no"]), ("i2a", s["na"])):
        if s[key] == "matrix":
            rows, cols, w = case["matrices"][key]
            out.append(sf.Mapper("matrix", tuple(rows.tolist()), tuple(cols.tolist()), tuple(w.tolist()),
                                 n_dst=nd, n_src=s["ni"]))
        else:
            out.append(sf.Mapper(s[key]))
    return out


def run_gpu(case: dict) -> dict[str, np.ndarray]:
    """The port, outputs as the oracle writes them (``reported``: the dead
    models' stages)."""
    import cupy as cp
    from woof.hex.coupled import seq_frac_e3sm as sf
    s = case["spec"]
    i2o, i2a = mappers_for(case)
    dev = {k: cp.asarray(np.ascontiguousarray(case[k], dtype=np.float64))
           for k in ("i2x", "dom", "fractions_i", "fractions_o", "fractions_a")}
    reported = sf.seq_frac_set(i2x=dev["i2x"], i2x_fields=case["i2x_fields"], dom=dev["dom"],
                               dom_fields=case["dom_fields"], fractions_i=dev["fractions_i"], fi_fields=FI_FIELDS,
                               fractions_o=dev["fractions_o"], fo_fields=FO_FIELDS,
                               fractions_a=dev["fractions_a"], fa_fields=FA_FIELDS, mapper_i2o=i2o, mapper_i2a=i2a,
                               atm_present=s["atm"], ice_present=s["ice"], ocn_present=s["ocn"], dead=s["dead"])
    out = {k: cp.asnumpy(dev[k]) for k in ("fractions_i", "fractions_o", "fractions_a")}
    out["reported"] = np.array(reported, dtype="<U8")
    return out


def measure() -> dict[str, dict[str, int]]:
    """case -> vector -> words that differ from the oracle (``reported``: 0 or 1)."""
    outputs, _ = load_fixture()
    report = {}
    for case in cases():
        if case["aborts"]:
            continue
        got, want = run_gpu(case), outputs[case["name"]]
        report[case["name"]] = {k: (int(list(got[k]) != list(want[k])) if k == "reported"
                                    else word_mismatches(got[k], want[k])) for k in want}
    return report
