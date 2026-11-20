"""The SLUCM's NUDAPT gridded-morphology arm against WRF v4.7.1, bit for bit.

``tools/urban_wrf471_oracle/ucm_nudapt_oracle.F90`` calls ``urban`` in the
byte-unmodified ``phys/module_sf_urban.F`` with ``MH_URB > 0`` -- the arm at
:649-788 that replaces read_param's canyon geometry with the gridded one:
building/street width from LP/LB/HGT/FRC (or the table's first widths when
LB <= LP), the frontal-area sector chosen from the wind direction, the
Macdonald (1998) ZDC/Z0C/Z0R/Z0HC and the 99-step sky-view factor.  Every
NLCD and LCZ category, four morphologies, eight forcing scenarios and six
carried steps whose wind turns through every sector; 2,688 column steps.

WRF's sector read ``ATAN2(U10,V10)`` happens before ``urban`` writes U10/V10
(the caller's locals still hold the previous column's 10 m wind).  The port
reads the column's own ``ATAN2(U1,V1)`` and the oracle stores U1/V1 into
U10/V10 before each call (``woof/core/urban_ucm.py`` says why).

The same rows run through the compiled CPU twin (``urban_ucm_host.cpp``,
the kernel's own source) without a GPU, and through the device kernel under
``requires_gpu``.  ``init-*.csv`` grades urban_var_init's morphology
keep/zero rule (:2788-2833) in ``woof.core.urban_state``.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from conftest import requires_gpu
from woof.core.fp32_ulp import fp32_ulp_distance
from woof.verify.urban_ucm_oracle import (
    NUDAPT_MORPH_COLUMNS,
    NUDAPT_VARIANTS,
    OUTPUT_COLUMNS,
    STATE_COLUMNS,
    UCM_NUDAPT_DIR,
    _flatten,
    _inputs,
    _reference,
    _state_from,
    carried_replay,
    load_init,
    load_variant,
    port_params,
    replay,
)

ROOT = Path(__file__).resolve().parents[1]
GRADED = ("nudapt-nlcd-main", "nudapt-lcz-main", "nudapt-nlcd-probe")


def _finite_probe(name):
    """The probe variant's sets 1-2 are finite in WRF and graded; sets 3-4
    are WRF's own NaN (refused at ingest, test below)."""
    if name.endswith("-probe"):
        return lambda rows: rows["morph"] <= 2
    return None


def _report(port, ref) -> str:
    lines = []
    for key, want in ref.items():
        got = np.asarray(port[key], np.float32)
        want = np.asarray(want, np.float32)
        bad = np.nonzero(got.view(np.uint32) != want.view(np.uint32))[0]
        if bad.size:
            ulp = int(np.max(fp32_ulp_distance(got, want)))
            lines.append(f"{key}: {bad.size} rows, max {ulp} ULP, first row "
                         f"{bad[0]}: port {got[bad[0]]!r} wrf {want[bad[0]]!r}")
    return "\n".join(lines)


@pytest.fixture(scope="module")
def twin(tmp_path_factory):
    from woof.verify.urban_ucm_host import HostTwin

    compiler = shutil.which(os.environ.get("CXX", "c++"))
    if compiler is None:
        pytest.skip("compiled SLUCM host twin needs a C++ compiler")
    output = tmp_path_factory.mktemp("slucm-nudapt") / "liburban_ucm_host.so"
    subprocess.run([
        compiler, "-std=c++17", "-O2", "-fno-fast-math", "-ffp-contract=off",
        "-fPIC", "-shared", str(ROOT / "woof/core/kernels/urban_ucm_host.cpp"),
        "-o", str(output),
    ], check=True)
    return HostTwin(output)


def test_fixture_is_the_pinned_one():
    sums = (UCM_NUDAPT_DIR / "oracle-sha256sums.txt").read_text().split("\n")
    pinned = dict(reversed(line.split()) for line in sums if line.strip())
    names = [f"{n}{s}" for n in NUDAPT_VARIANTS
             for s in (".csv.gz", "-table.csv", "-switches.csv")]
    names += [f"init-opt{o}-lcz{l}.csv" for o in (1, 2, 3) for l in (0, 1)]
    for name in names:
        path = UCM_NUDAPT_DIR / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == pinned[name], name


def _sector(u1, v1):
    """module_sf_urban.F:709-733's lambda_f index (1-4), in double."""
    wdr = np.degrees(np.arctan2(u1.astype(np.float64), v1.astype(np.float64)))
    s = np.ones(wdr.shape, int)
    s[((wdr > 22.5) & (wdr <= 67.5)) | ((wdr >= -157.5) & (wdr < -112.5))] = 2
    s[((wdr > 67.5) & (wdr <= 112.5)) | ((wdr >= -112.5) & (wdr < -67.5))] = 3
    s[((wdr > 112.5) & (wdr <= 157.5)) | ((wdr >= -67.5) & (wdr < -22.5))] = 4
    return s


def test_fixture_reaches_the_branches_it_claims():
    for name in ("nudapt-nlcd-main", "nudapt-lcz-main"):
        rows, _, _ = load_variant(name, UCM_NUDAPT_DIR)
        assert np.all(rows["mh_urb"] > 0)           # the arm runs on every row
        sector = _sector(rows["u1"], rows["v1"])
        assert set(sector.tolist()) == {1, 2, 3, 4}, name
        # the sector chain's axis cases: due north, east, south
        assert np.any((rows["u1"] == 0) & (rows["v1"] > 0))
        assert np.any((rows["u1"] > 0) & (rows["v1"] == 0))
        assert np.any((rows["u1"] == 0) & (rows["v1"] < 0))
        # both width arms (:686-699)
        assert np.any(rows["lb_urb"] > rows["lp_urb"])
        assert np.any(rows["lb_urb"] <= rows["lp_urb"])
        # first level above and (with the tall cell) near the canopy
        assert np.all(rows["za"] > rows["mh_urb"])


def test_wrf_goes_non_finite_on_the_refused_morphologies():
    """Probe sets 3 (a zero LF sector) and 4 (LP = 0 below LB) are NaN in WRF
    itself; woof.core.urban_param.check_nudapt_columns refuses both, and
    passes the finite sets 1-2."""
    from woof.core.urban_param import UrbanParamError, check_nudapt_columns

    rows, _, _ = load_variant("nudapt-nlcd-probe", UCM_NUDAPT_DIR)
    for m in (1, 2, 3, 4):
        sel = rows["morph"] == m
        nonfinite = ~np.isfinite(rows["ts"][sel])
        first = {k: rows[k][sel][:1] for k in NUDAPT_MORPH_COLUMNS}
        morph = {"mh": first["mh_urb"], "stdh": first["stdh_urb"],
                 "lp": first["lp_urb"], "lb": first["lb_urb"],
                 "lf": np.stack([first[f"lf_urb{k}"] for k in range(1, 5)])}
        urban = np.ones(1, bool)
        if m <= 2:
            assert not nonfinite.any(), m
            check_nudapt_columns(morph, urban)
        else:
            assert nonfinite.any(), m
            with pytest.raises(UrbanParamError):
                check_nudapt_columns(morph, urban)


@pytest.mark.parametrize("name", GRADED)
def test_twin_every_row_is_bitwise_wrf(twin, name):
    port, ref, codes = replay(name, runner=twin.run_columns,
                              select=_finite_probe(name))
    assert not np.any(codes), np.unique(codes)
    assert set(ref) == set(OUTPUT_COLUMNS) | {f"after:{c}" for c in STATE_COLUMNS}
    report = _report(port, ref)
    assert not report, f"{name} differs from WRF:\n{report}"


@pytest.mark.parametrize("name", ("nudapt-nlcd-main", "nudapt-lcz-main"))
def test_twin_carried_state_stays_on_wrf(twin, name):
    port, ref = carried_replay(name, runner=twin.run_columns)
    report = _report(port, ref)
    assert not report, f"{name} (carried) differs from WRF:\n{report}"


def test_twin_gridded_morphology_moves_the_answer(twin):
    """Without the morphology the same rows take read_param's geometry and
    must come out different: the arm is what the fixture grades."""
    rows, table, switches = load_variant("nudapt-nlcd-main", UCM_NUDAPT_DIR)
    sel = np.ones(rows["ta"].size, bool)
    outputs, _, codes = twin.run_columns(
        port_params(table, switches), _inputs(rows, sel),
        _state_from(rows, sel, "_in"), generation="wrf_471")
    assert not np.any(codes)
    moved = outputs["ts"].view(np.uint32) != rows["ts"].view(np.uint32)
    assert moved.mean() > 0.9


@requires_gpu
@pytest.mark.parametrize("name", GRADED)
def test_every_row_is_bitwise_wrf(name):
    port, ref, codes = replay(name, select=_finite_probe(name))
    assert not np.any(codes), np.unique(codes)
    report = _report(port, ref)
    assert not report, f"{name} differs from WRF:\n{report}"


@requires_gpu
@pytest.mark.parametrize("name", ("nudapt-nlcd-main", "nudapt-lcz-main"))
def test_carried_state_stays_on_wrf(name):
    port, ref = carried_replay(name)
    report = _report(port, ref)
    assert not report, f"{name} (carried) differs from WRF:\n{report}"


# ---------------------------------------------------------------------------
# urban_var_init's keep/zero rule
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("option", (1, 2, 3))
@pytest.mark.parametrize("lcz", (0, 1))
def test_urban_var_init_morphology_is_wrfs(option, lcz):
    from woof.core.urban_state import (option_spec, resolve_dimensions,
                                        urban_var_init_host)
    from woof.core.urban_tables import load_urban_params, urban_category_set

    rows = load_init(option, lcz)
    n = rows["col"].size
    ivgtyp = rows["ivgtyp"].reshape(1, n)
    morphology = {
        "lp": rows["lp_in"].reshape(1, n), "lb": rows["lb_in"].reshape(1, n),
        "hgt": rows["hgt_in"].reshape(1, n),
        "mh": (rows["mh_in"] if option == 1 else 0 * rows["mh_in"]).reshape(1, n),
        "stdh": (rows["stdh_in"] if option == 1 else 0 * rows["stdh_in"]).reshape(1, n),
        "lf": np.stack([rows[f"lf{k}_in"] for k in range(1, 5)])[:, None, :],
        "hi": np.stack([rows[f"hi{k}_in"] for k in range(1, 16)])[:, None, :],
    }
    ones = np.full((1, n), 290.0, np.float32)
    out = urban_var_init_host(
        option=option, use_wudapt_lcz=lcz,
        params=load_urban_params(option, lcz),
        categories=urban_category_set(isurban=13),
        ivgtyp=ivgtyp, tsk=ones, tslb=np.repeat(ones[None], 4, 0), tmn=ones,
        smois=np.full((4, 1, n), 0.3, np.float32),
        frc_urb2d=rows["frc_in"].reshape(1, n), num_urban_hi=15, nz=4,
        dims=resolve_dimensions(option), spec=option_spec(option),
        morphology=morphology)
    bits = lambda a: np.asarray(a, np.float32).reshape(-1).view(np.uint32)
    assert out["utype_urb2d"].reshape(-1).tolist() == rows["utype"].tolist()
    assert np.array_equal(bits(out["frc_urb2d"]), bits(rows["frc"]))
    for name in ("lp", "lb", "hgt", "mh", "stdh"):
        assert np.array_equal(bits(out[f"{name}_urb2d"]), bits(rows[name])), name
    for k in range(4):
        assert np.array_equal(bits(out["lf_urb2d"][k]), bits(rows[f"lf{k + 1}"])), k
    for k in range(15):
        assert np.array_equal(bits(out["hi_urb2d"][k]), bits(rows[f"hi{k + 1}"])), k
    # the fixture exercises keep, zero-on-urban and zero-off-urban
    urban = rows["utype"] > 0
    kept = urban & (rows["hgt_in"] > 0)
    assert kept.any() and (urban & ~kept).any() and (~urban).any()
