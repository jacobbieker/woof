"""The 4D.4 L1 gate's woof side (docs/dev/WRFDA-4DVAR-PLAN.md section 4), run as a child
process by tests/test_var_outer4d.py (``test_l1_two_outer_loops_against_wrfda``) on CUDA.

``python tests/_var4d_l1.py LANE_DIR WORK_DIR MAX_EXT_ITS`` runs
``analyse(var4d=..., max_ext_its=MAX_EXT_ITS)`` on the lane's files with P3a's window
(:mod:`tests._var4d_l1_case`) as both the nonlinear model and, through
:class:`~woof.da.var.window4d.WindowModel`, the TL/AD model, and prints ``RESULT
{json}``: woof's and WRFDA's ``minimize_cg`` lines per loop, and each loop's analysis
increment against WRFDA's (``wrfvar_output_01`` after loop 1 when it ran two,
``wrfvar_output`` at the end).

``LANE_DIR`` is ``tools/wrfda_oracle/run_4dvar.sh l1``'s run directory: ``namelist.input``,
``fg``, ``wrfbdy_d01``, ``be.dat``, ``ob01.ascii`` .. ``ob07.ascii``, ``wrfda.out`` (the
log) and the ``wrfvar_output*`` files.

WRFPLUS's per-step states are not readable from this WRFDA build (``trajectory_io =
.false.`` is broken: tools/wrfda_oracle/run_4dvar.sh), so every trajectory here is woof's:
the window's binary64 run from the lane's binary32 first guess.  The slots
(``fg01`` .. ``fg07``) are that run's states every ``var4d_bin / time_step`` = 10 steps on
the first guess's other fields (PSFC, the base state, the grid); loop 2's rerun starts
from loop 1's analysis (its prognostics; the window's cold first step rediagnoses p, al
and alt) on the boundary tables loop 1 updated (``FirstGuess.lbc_increment``, which
``analyse`` has already added to the window's tables: the rerun and the TL share them).
Agreement is therefore tolerance-level: two models (woof's window, WRF 4.7.1 + WRFPLUS)
from the same start, not WRF's trajectory under woof's DA.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import numpy as np

import woof  # noqa: F401  (installs the backend's cupy)
import cupy as cp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests._var4d_l1_case import build_window, prepare_case  # noqa: E402
from woof.da.var import fourd as f4  # noqa: E402
from woof.da.var import transfer4d as t4  # noqa: E402
from woof.da.var import window4d as w4  # noqa: E402
from woof.da.var.analysis import FirstGuess, WRF_FIELDS, analyse, read_first_guess  # noqa: E402
from woof.da.var.outer4d import Trajectory  # noqa: E402

NSLOTS, SPS, VAR4D_BIN, TIME_STEP = 7, 10, 3600, 360
WINDOW = ("2008-02-05_12:00:00", "2008-02-05_18:00:00")
#: A slot file's prognostics -> the window's state names.
SLOT = {"U": "u", "V": "v", "W": "w", "PH": "php", "T": "thp", "MU": "mup", "P": "p_pert", "QVAPOR": "qv"}
_CG = re.compile(r"^\s*minimize_cg\s+(\d+)\s+(\d+)\s+(\S+)\s+(\S+)")


def _stamp(times: str, n: int) -> str:
    return f"{times[:11]}{12 + n:02d}{times[13:]}"


def slot_states(base: FirstGuess, states: dict) -> list:
    """``fg01`` .. ``fg07``: the window's states every SPS steps on ``base``'s other fields."""
    out = []
    for n in range(NSLOTS):
        st = states[n * SPS]
        fields = dict(base.fields)
        for k, w in SLOT.items():
            fields[k] = w4.from_window(cp.asnumpy(st[w]))
        out.append(FirstGuess(path=base.path, fields=fields, attrs=dict(base.attrs), times=_stamp(base.times, n)))
    return out


def cg_lines(lines) -> list:
    out = []
    for ln in lines:
        m = _CG.match(ln)
        if m:
            out.append([int(m.group(1)), int(m.group(2)), float(m.group(3).replace("D", "e")),
                        float(m.group(4).replace("D", "e"))])
    return out


def increment_agreement(fg: FirstGuess, got: dict, path: Path) -> dict:
    """woof's analysis increment ``got - fg`` against WRFDA's ``path - fg``, per field:
    the max relative difference (over WRFDA's max), the rms ratio and the correlation."""
    ref = read_first_guess(path)
    out = {}
    for k in ("U", "V", "T", "QVAPOR", "MU", "PH", "PSFC"):
        if k not in ref.fields or k not in got:
            continue
        dr = np.asarray(ref.fields[k], dtype=np.float64) - fg.fields[k]
        dw = np.asarray(got[k], dtype=np.float64) - fg.fields[k]
        scale = float(np.max(np.abs(dr)))
        if scale == 0.0:
            out[k] = {"wrfda_max": 0.0, "woof_max": float(np.max(np.abs(dw)))}
            continue
        corr = float(np.corrcoef(dr.ravel(), dw.ravel())[0, 1]) if np.std(dw) > 0 else 0.0
        out[k] = {"max_rel": float(np.max(np.abs(dw - dr))) / scale,
                  "rms_ratio": float(np.sqrt(np.mean(dw ** 2)) / np.sqrt(np.mean(dr ** 2))),
                  "corr": corr, "wrfda_max": scale}
    return out


def dot_check(model, rng) -> dict:
    """The real window's TL/AD with the boundary tables (lbc='wrfda'):
    ``sum_n <TL(g0)_n, f_n>`` against ``<g0, AD(f)>``, and how much the tables move slot 7."""
    win = model.window
    scale = {"u_2": 0.5, "v_2": 0.5, "w_2": 0.01, "t_2": 0.5, "mu_2": 50.0, "ph_2": 0.0, "p": 20.0, "qv": 1e-4}
    names = {"u_2": "u", "v_2": "v", "w_2": "w", "t_2": "thp", "mu_2": "mup", "ph_2": "php", "p": "p_pert",
             "qv": "qv"}

    def rnd(s):
        return {k: s[k] * rng.standard_normal(tuple(win._start[w].shape)[::-1]) for k, w in names.items()}

    g0 = rnd(scale)
    fs = [rnd({k: 1.0 for k in scale}) for _ in range(NSLOTS)]
    gs = model.tl(g0, NSLOTS)
    def dot(a, b):
        return sum(float(np.sum(a[k] * b[k])) for k in t4.TL_KEYS)

    lhs = sum(dot(g, f) for g, f in zip(gs, fs))
    rhs = dot(g0, model.ad(fs))
    fixed = w4.WindowModel(win, SPS, lbc="fixed")
    gf = fixed.tl(g0, NSLOTS)
    moved = {k: float(np.max(np.abs(gs[-1][k] - gf[-1][k]))) / (float(np.max(np.abs(gs[-1][k]))) or 1.0)
             for k in ("u_2", "t_2", "mu_2")}
    return {"lhs": lhs, "rhs": rhs, "rel": abs(lhs - rhs) / abs(lhs), "lbc_moves_slot7": moved}


def main(lane: Path, work: Path, max_ext_its: int) -> dict:
    case = prepare_case(lane, work / "case")
    c = build_window(case, NSLOTS * SPS - SPS, checkpoint_every=1)
    win = c.window
    # halo_start "wrfplus", moisture "advect"; lbc "wrfda" (WOOF_L1_LBC=fixed: the
    # tables unperturbed, to measure what WRFDA's boundary perturbation is worth)
    # WOOF_L1_CHECKPOINT: the window's checkpoint policy (all, the default, or slot; the
    # gradient is the same to the word under either, window4d's module docstring)
    model = w4.WindowModel(win, SPS, lbc=os.environ.get("WOOF_L1_LBC", "wrfda"),
                           checkpoint=os.environ.get("WOOF_L1_CHECKPOINT", "all"))
    assert model.moisture == "advect"
    fg = read_first_guess(lane / "fg")
    states = {k: win._checkpoints[k] for k in range(0, win.nsteps, win.every)}
    states[win.nsteps] = win.forward(win._checkpoints[0])
    slots1 = slot_states(fg, states)
    analyses = {}

    def nl_model(it, state):
        analyses[it - 1] = dict(state.fields)
        x0 = {n: win._start[n].copy() for n in win.inputs}
        for k, w in SLOT.items():
            if w in x0 and k in state.fields:
                x0[w][...] = cp.asarray(w4.to_window(state.fields[k]))
        x0[model.halo][...] = x0["mup"]                      # WRFPLUS's start (halo_start "wrfplus")
        x0["muts"][...] = win._start["mub2d"] + x0["mup"]
        end = win.forward(x0)   # the slot states are checkpoints under either policy
        run = {**{k: win._checkpoints[k] for k in range(0, win.nsteps, win.every)}, win.nsteps: end}
        # the rerun is the window's own model: relinearise reruns it from x0 under the policy
        return Trajectory(slots=slot_states(state, run), start=x0)

    lines = []
    r = analyse(lane / "be.dat", [lane / f"ob{n:02d}.ascii" for n in range(1, NSLOTS + 1)], fg, fgat=slots1,
                time_window=WINDOW, ntmax=[50] + [75] * (max_ext_its - 1), eps=0.01, max_ext_its=max_ext_its,
                check_rh=0, var4d=f4.Var4D(model=model, nl_model=nl_model, var4d_bin=VAR4D_BIN,
                                           time_step=TIME_STEP), write=lines.append)
    log = (lane / "wrfda.out").read_text(errors="replace").splitlines()
    out = {"woof_cg": cg_lines(lines), "wrfda_cg": cg_lines(log),
           "woof_iterations": [x.iterations for x in r.cg], "nobs": r.nobs}
    final = {k: r.wrf[k] for k in WRF_FIELDS if k in r.wrf}
    out["increments"] = {"final": increment_agreement(fg, final, lane / "wrfvar_output")}
    if max_ext_its > 1 and 1 in analyses and (lane / "wrfvar_output_01").exists():
        out["increments"]["loop1"] = increment_agreement(fg, analyses[1], lane / "wrfvar_output_01")
    out["dot"] = dot_check(model, np.random.default_rng(7))
    return out


if __name__ == "__main__":
    res = main(Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]))
    print("RESULT " + json.dumps(res))
