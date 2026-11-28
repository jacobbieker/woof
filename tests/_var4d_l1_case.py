"""The 4D.4 L1 lane's domain as P3a's window (docs/dev/WRFDA-4DVAR-PLAN.md section 4), for
tests/_var4d_l1.py.

The WRFDA 4.7.1 4D-Var tutorial case (2008-02-05 12Z, 90x60x41, ``time_step`` 360 s) run
as lane L1 (``tools/wrfda_oracle/run_4dvar.sh l1``): no physics, ``diff_opt = 0``, and the
dynamics P3a's window covers (``damp_opt 0``, ``emdiv 0``, ``moist_adv_opt 0``).  The
woof state comes through woof's wrfinput door from a prepared copy of the lane's inputs
(:func:`prepare_case`), with two test-side workarounds, each a request in the
coordinator's P3A-REQUESTS note:

- woof's namelist importer refuses ``diff_opt = 0`` (WRF 4.7.1 runs it: no mixing), so
  the case imports ``diff_opt = 1, km_opt = 4`` and :func:`build_window` overrides
  ``diff_opt = 0, km_opt = 1`` (``khdif = kvdif = 0``: no constant-K package either) on
  the RunConfig the window is built from;
- at ``mp_physics = 0`` woof's reader gives a qv-only state (WRF's passiveqv) and P3a's
  moist binding wants qv, qc and qr, so the case imports with ``mp_physics = 1``
  (Kessler's qv, qc, qr) on a copy of the wrfinput whose physics attributes are patched
  to match, :func:`build_window` zeroes qc and qr (WRF's moist array has qv only; zero
  species add nothing to calc_cq's sum and stay zero without microphysics) and
  overrides ``mp_physics = 0``.

The boundary tables are the wrfbdy's first interval only (12Z-18Z): the window's last
step ends on 18Z, which WRF serves from that interval (its next read is at the next
step's start).
"""

from __future__ import annotations

import dataclasses
import re
import shutil
from pathlib import Path

import numpy as np

#: &groups the woof importer does not know (WRFDA's, WRFPLUS's and the SCM/TC ones).
_DROP_GROUPS = re.compile(r"^&(wrfvar\d+|perturbation|scm|tc)\b", re.I)
#: real.exe / WRFDA keys the importer refuses or does not map.
_DROP_KEYS = {"t_extrap_type", "interp_type", "smooth_option", "nproc_x", "mp_physics_ad", "real_data_init_type",
              "io_form_auxhist7", "io_form_auxhist8", "frames_per_auxhist7", "frames_per_auxhist8", "km_opt",
              "diff_opt", "mp_physics", "history_interval"}
#: Physics attributes of the wrfinput copy, patched to the import's selectors (the door
#: checks the file against the namelist).
_ATTRS = {"MP_PHYSICS": 1, "RA_LW_PHYSICS": 0, "RA_SW_PHYSICS": 0, "SF_SFCLAY_PHYSICS": 0,
          "SF_SURFACE_PHYSICS": 0, "BL_PBL_PHYSICS": 0, "CU_PHYSICS": 0}


def woof_namelist(lane_namelist: str) -> str:
    """The lane's ``namelist.input`` as woof's importer takes it (module docstring)."""
    out, skip = [], False
    for line in lane_namelist.splitlines():
        s = line.strip()
        if s.startswith("&"):
            skip = bool(_DROP_GROUPS.match(s))
            if skip:
                continue
            out.append(line)
            group = s[1:].lower()
            if group == "time_control":
                out.append("history_interval=60,")
            elif group == "physics":
                out.append("mp_physics=1,")   # imported as Kessler, run as 0 (module docstring)
            elif group == "dynamics":
                out += ["diff_opt=1,", "km_opt=4,"]   # imported, then overridden to 0 / 1
            continue
        if skip:
            if s == "/":
                skip = False
            continue
        m = re.match(r"\s*(\w+)\s*=", line)
        if m and m.group(1).lower() in _DROP_KEYS:
            continue
        out.append(line)
    return "\n".join(out) + "\n"


def prepare_case(run_dir: Path, out: Path) -> Path:
    """``out``: the lane's first guess, boundaries and namelist as woof's door takes them."""
    import netCDF4

    run_dir, out = Path(run_dir), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "namelist.input").write_text(woof_namelist((run_dir / "namelist.input").read_text()))
    shutil.copyfile(run_dir / "fg", out / "wrfinput_d01")
    with netCDF4.Dataset(out / "wrfinput_d01", "a") as d:
        for k, v in _ATTRS.items():
            if k in d.ncattrs():
                d.setncattr(k, np.int32(v))
    shutil.copyfile(run_dir / "wrfbdy_d01", out / "wrfbdy_d01")
    return out


@dataclasses.dataclass
class Case:
    state: object
    cfg: object          #: the RunConfig the window runs (mp_physics 0, diff_opt 0)
    bounds: object       #: the one boundary interval attached
    window: object
    interval_seconds: float


def build_window(case_dir: Path, nsteps: int = 60, *, checkpoint_every: int = 1) -> Case:
    """The lane's window from :func:`prepare_case`'s directory (binary64, every step kept)."""
    import cupy as cp

    from woof.config import soil_layer_count
    from woof.grad.wrfplus import window as W
    from woof.ingest.lateral_bc import attach_lateral_boundaries
    from woof.ingest.wrfinput import read_wrfbdy, read_wrfinput, restore_domain_state
    from woof.wrfinput_door import resolve_wrfinput_run

    run = resolve_wrfinput_run(Path(case_dir))
    cfg = run.experiment.root.run
    dims = dict(west_east=cfg.nx, west_east_stag=cfg.nx + 1, south_north=cfg.ny, south_north_stag=cfg.ny + 1,
                bottom_top=cfg.nz, bottom_top_stag=cfg.nz + 1, soil_layers_stag=soil_layer_count(cfg))
    restored = read_wrfinput(run.wrfinput_paths[1], expected_dimensions=dims, cfg=cfg)
    bounds = read_wrfbdy(run.wrfbdy_path, run_seconds=nsteps * cfg.dt, restored=restored,
                         forcing_interval_seconds=run.coverage.forcing_interval_seconds,
                         spec_bdy_width=cfg.spec_bdy_width, cfg=cfg)
    bounds = dataclasses.replace(bounds, intervals=bounds.intervals[:1])
    state = restore_domain_state(restored, cfg)
    attach_lateral_boundaries(state, bounds)
    # qv-only moist array (WRF's passiveqv at mp_physics 0): module docstring
    state.qc[...] = 0
    state.qr[...] = 0
    # diff_opt 0 and mp_physics 0, which the importer cannot express: module docstring
    wcfg = dataclasses.replace(cfg, mp_physics=0, diff_opt=0, km_opt=1, khdif=0.0, kvdif=0.0)
    win = W.Window(state, wcfg, nsteps, dtype=np.float64, checkpoint_every=checkpoint_every)
    iv = bounds.intervals[0]
    cp.get_default_memory_pool().free_all_blocks()
    return Case(state=state, cfg=wcfg, bounds=bounds, window=win,
                interval_seconds=float(iv.end_seconds - iv.start_seconds))
