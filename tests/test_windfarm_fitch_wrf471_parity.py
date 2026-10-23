"""The Fitch wind-farm drag, word for word against WRF v4.7.1's own Fortran.

``tools/windfarm_fitch_wrf471_oracle`` compiles module_wind_fitch.F's
dragforce, turbine_area and dragcof byte-unmodified at -O0 in real(4) and
runs them on eleven columns (one turbine, several of one type, two mixed
cells, a rotor across many levels and within one, below cut-in, above
cut-out, rated wind, an empty cell, a uniform column) for two cell sizes and
TKE factors, with the PBL tendencies already in DU, DV and QKE.  The device
kernel (``fitch_column``) and the binary32 CPU mirror
(:func:`woof.verify.windfarm_fitch_ref.f32_fitch_column`) must reproduce
every word of DU, DV, QKE and POWER: max ULP 0, against the oracle and
against each other.

HELD OUT (no oracle exists, PROVENANCE.md): the ``overlap`` and ``thrust``
diagnostics, which WRF does not form, are held CPU against GPU at 0 ULP
only; the three deviations the kernel documents are not reached by the
fixture, which this file pins.  The float64 mirror ``np_fitch_column`` is a
SECONDARY reference, held to the oracle at float32 rounding distance.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import requires_gpu

ROOT = Path(__file__).resolve().parents[1]
ORACLE = ROOT / "woof" / "data" / "windfarm_fitch" / "oracle"
TOOL = ROOT / "tools" / "windfarm_fitch_wrf471_oracle"
CASES = ("default", "coarse")
FIELDS = ("du", "dv", "qke", "power")


def _load(case: str) -> dict:
    from woof.verify.urban_oracle import load_case
    return load_case(ORACLE / case)


def _kji(a):
    """Fortran (i, k, j) to woof's (k, j, i)."""
    return np.ascontiguousarray(np.transpose(a, (1, 2, 0)))


def layout_from_turbines(o):
    """WRF's turbine list as groups: a cell's consecutive identical
    turbines are one group, in the list's order."""
    from woof.core.windfarm_fitch import FitchLayout

    nx, _, ny = o["u"].shape
    rows = []
    last = {}
    for n in range(o["ival"].size):
        cell = int(o["ival"][n]) - 1        # jval = 1: one row of cells
        key = (int(o["nkind"][n]) - 1, float(o["hubheight"][n]),
               float(o["diameter"][n]))
        if last.get(cell) is not None and tuple(rows[last[cell]][1:4]) == key:
            rows[last[cell]][4] += 1
        else:
            last[cell] = len(rows)
            rows.append([cell, *key, 1])
    stc = np.zeros(o["nval"].size, np.float32)
    for n in range(o["ival"].size):
        stc[int(o["nkind"][n]) - 1] = o["stc"][n]
    return FitchLayout(
        ny=ny, nx=nx, cell=[r[0] for r in rows], curve=[r[1] for r in rows],
        hub_height_m=[r[2] for r in rows],
        rotor_diameter_m=[r[3] for r in rows], count=[r[4] for r in rows],
        speeds_ms=o["turbws"], thrust_coefficient=o["turbtc"],
        power_kw=o["turbpw"], nval=o["nval"], standing_thrust=stc)


def oracle_case(case: str) -> dict:
    o = _load(case)
    nz = o["u"].shape[1] - 1
    out = {"layout": layout_from_turbines(o),
           "z_w": _kji(o["z_at_w"]),
           "dx": float(o["dx"]), "dt": float(o["dt"]),
           "factor": float(o["windfarm_tke_factor"]),
           "power": np.ascontiguousarray(o["power"].T)}
    for name in ("u", "v", "du", "dv", "qke", "du_in", "dv_in", "qke_in"):
        out[name] = _kji(o[name])[:nz]
    out["rho"] = np.full_like(out["u"], np.float32(1.2))
    return out


def run_gpu(o):
    import cupy as cp
    from woof.core.windfarm_fitch import launch_fitch

    du, dv, qke = (cp.asarray(o[f"{n}_in"]) for n in ("du", "dv", "qke"))
    diagnostics = {}
    power = launch_fitch(cp.asarray(o["u"]), cp.asarray(o["v"]),
                         cp.asarray(o["z_w"]), o["layout"], du, dv, dx=o["dx"],
                         qke=qke, dt=o["dt"], tke_factor=o["factor"],
                         rho=cp.asarray(o["rho"]), diagnostics=diagnostics)
    return {"du": cp.asnumpy(du), "dv": cp.asnumpy(dv), "qke": cp.asnumpy(qke),
            "power": cp.asnumpy(power),
            "thrust": cp.asnumpy(diagnostics["thrust"]),
            "overlap": cp.asnumpy(diagnostics["overlap"])}


def run_cpu(o):
    from woof.verify.windfarm_fitch_ref import f32_fitch_column

    du, dv, qke = (o[f"{n}_in"].copy() for n in ("du", "dv", "qke"))
    power, thrust, overlap = f32_fitch_column(
        o["u"], o["v"], o["z_w"], o["layout"], du, dv, qke, dx=o["dx"],
        dt=o["dt"], tke_factor=o["factor"], rho=o["rho"])
    return {"du": du, "dv": dv, "qke": qke, "power": power, "thrust": thrust,
            "overlap": overlap}


def words(port, ref) -> dict:
    from woof.verify.urban_oracle import ulp_table
    return ulp_table(port, ref)


@pytest.fixture(scope="module", params=CASES)
def case(request):
    return request.param, oracle_case(request.param)


# ---------------------------------------------------------------------------
# Receipts and reach (no device)
# ---------------------------------------------------------------------------

def test_fixture_is_the_pinned_wrf_and_its_own_receipts():
    """Every published word is the one build.sh wrote, from pinned sources."""
    pins = [line.split() for line in
            (TOOL / "SOURCES.sha256").read_text().splitlines()
            if line and not line.startswith("#")]
    recorded = [line.split() for line in
                (ORACLE / "SOURCES.sha256").read_text().splitlines() if line]
    assert recorded == pins
    for line in (ORACLE / "FIXTURES.sha256").read_text().splitlines():
        digest, rel = line.split()
        assert hashlib.sha256((ORACLE / rel).read_bytes()).hexdigest() == digest, rel
    harness = dict(reversed(line.split()) for line in
                   (ORACLE / "HARNESS.sha256").read_text().splitlines())
    for rel, digest in harness.items():
        assert hashlib.sha256((TOOL / rel).read_bytes()).hexdigest() == digest, rel
    report = (ORACLE / "libmvec-report.txt").read_text()
    head = report.split("at WRF's -O2")[0]
    assert "_ZGV" not in head
    assert "U powf" in head and "U asinf" in head
    assert "_ZGVbN4v_expf" in report


def test_fixture_reaches_every_case_and_none_of_the_held_out_paths(case):
    """The columns reach what they claim, and no hub sits below its first
    mass level or above its top one, where WRF reads unassigned values."""
    from woof.verify.windfarm_fitch_ref import np_fitch_column

    _, o = case
    layout = o["layout"]
    groups = np.diff(layout.cell_offsets)
    assert groups.max() >= 3 and (groups == 2).any()      # mixed cells
    assert (groups == 0).any()                            # an empty cell
    assert (layout.count > 1).any()                       # several of a type
    ref = np_fitch_column(o["u"], o["v"], o["z_w"], layout, dx=o["dx"],
                          dt=o["dt"], tke_factor=o["factor"])
    levels = (ref["overlap"] > 0).sum(axis=0)
    assert levels.max() >= 15 and (levels == 1).any()     # many / one level
    hub = [g["speedhub"] for g in ref["groups"].values()]
    cutin = layout.speeds_ms[layout.curve, 0]
    cutout = layout.speeds_ms[layout.curve, layout.nval[layout.curve] - 1]
    assert (np.array(hub) < cutin).any() and (np.array(hub) > cutout).any()
    assert any(12.0 < h < 25.0 for h in hub)              # rated, on the taper
    dz = np.diff(o["z_w"], axis=0)
    first = 0.5 * dz[0]
    top = np.cumsum(dz, axis=0)[-1] - 0.5 * dz[-1]
    for g in range(layout.cell.size):
        j, i = divmod(int(layout.cell[g]), layout.nx)
        assert first[j, i] <= layout.hub_height_m[g] < top[j, i]
    # The PBL's own tendencies were in DU before WRF added the drag.
    assert np.any(o["du_in"] != 0) and np.any(o["du"] != o["du_in"])


def test_the_cpu_mirror_reproduces_wrf_word_for_word(case):
    name, o = case
    cpu = run_cpu(o)
    for field in FIELDS:
        table = words(cpu[field], o[field])
        assert table["max_ulp"] == 0, (name, field, table)


def test_the_float64_mirror_is_a_secondary_reference_at_rounding_distance(case):
    from woof.verify.windfarm_fitch_ref import np_fitch_column

    _, o = case
    ref = np_fitch_column(o["u"], o["v"], o["z_w"], o["layout"], dx=o["dx"],
                          dt=o["dt"], tke_factor=o["factor"])
    pairs = (("du", ref["du"] + o["du_in"]), ("dv", ref["dv"] + o["dv_in"]),
             ("qke", ref["dqke"] + o["qke_in"]), ("power", ref["power"]))
    for field, value in pairs:
        scale = np.abs(o[field]).max()
        assert np.abs(value - o[field]).max() <= 2e-5 * scale, field


# ---------------------------------------------------------------------------
# The device kernel
# ---------------------------------------------------------------------------

@requires_gpu
@pytest.mark.gpu
def test_the_device_kernel_reproduces_wrf_word_for_word(case):
    name, o = case
    gpu = run_gpu(o)
    for field in FIELDS:
        table = words(gpu[field], o[field])
        assert table["max_ulp"] == 0, (name, field, table)


@requires_gpu
@pytest.mark.gpu
def test_cpu_and_gpu_agree_on_every_word_including_the_diagnostics(case):
    name, o = case
    gpu, cpu = run_gpu(o), run_cpu(o)
    for field in FIELDS + ("thrust", "overlap"):
        table = words(gpu[field], cpu[field])
        assert table["max_ulp"] == 0, (name, field, table)


@requires_gpu
@pytest.mark.gpu
def test_a_changed_word_in_the_kernel_is_seen(case, monkeypatch):
    """Mutation control: the comparison can fail.  The kernel with WRF's
    1.23 power density read as 1.225 must move POWER off the oracle."""
    import cupy as cp
    from woof.core import windfarm_fitch as wf

    source = wf.module_source()
    assert source.count("1.23f") >= 4
    mutated = source.replace("0.5f * 1.23f * gfk_pow(speedhub",
                             "0.5f * 1.225f * gfk_pow(speedhub")
    assert mutated != source
    from woof import nvrtc_ptx_cache as compiler
    ptx, _ = compiler.compile_using_nvrtc(mutated, wf.MODULE_OPTIONS, None,
                                          "windfarm_fitch_mutant.cu")
    module = cp.cuda.function.Module()
    module.load(ptx.encode() if isinstance(ptx, str) else ptx)
    monkeypatch.setattr(wf, "_kernel", module.get_function)
    _, o = case
    gpu = run_gpu(o)
    assert words(gpu["power"], o["power"])["max_ulp"] > 0
