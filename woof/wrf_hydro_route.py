"""The forecast route's door to a coupled WRF/WRF-Hydro run (``--wrf-hydro``).

WRF built with ``WRF_HYDRO`` couples its LSM to WRF-Hydro's routing: after
every surface-driver call that ran the LSM (``HYDRO_dt = DTBL``,
module_surface_driver.F :2647 Noah, :3125 Noah-MP) the first RK step calls
``wrf_drv_HYDRO`` (dyn_em/module_first_rk_step_part1.F :1093-1097), which
hands SMOIS, SH2O, INFXSRT and SOLDRAIN to the routing and takes SMOIS, SH2O
and the ponded head SFCHEADRT back; the next LSM call reads SFCHEADRT
through its ``WRF_HYDRO`` hooks.  This module is that seam on WOOF's side:
the exchange planes, the LSM hooks (``launch_noah(wrf_hydro=...)``,
``noahmp_lsm_step(wrf_hydro=...)``) and the call after the land surface.

The routing itself is the WRF-Hydro opt-in (``optional/wrf_hydro``, UCAR's
licence, ``WOOF_ENABLE_WRF_HYDRO=1``; docs/coupling/WRF-HYDRO-NEMO.md),
reached only through :func:`woof.optin.load` when a run asks for it.  This
door is distinct from upstream's NextGen coupling (``--hydro on``, a
hydrofabric), and the two are not combined.

A run asks with ``--wrf-hydro HYDRO.NAMELIST`` (``woof run`` and ``woof go``)
or a ``[wrf_hydro]`` table in the run TOML::

    [wrf_hydro]
    namelist = "hydro.namelist"      # relative to the TOML
    output_dir = "wrf_hydro_output"  # optional; frxst_pts_out.txt, CHRTOUT, RTOUT

Either sets ``WOOF_WRF_HYDRO_NAMELIST`` (and ``WOOF_WRF_HYDRO_OUTPUT``) for
the launch and every process it starts.  Without them nothing here runs: no
plane is allocated, no driver attribute is set, the LSM calls are the
default ones, so the forecast is byte-identical to a run without this module.

Refused by name: an LSM other than Noah (2) or Noah-MP (4); Noah's mosaic;
an adaptive time step; an ensemble of more than one member; a run with more
than one domain (WRF couples only a domain without nests, and WRF-Hydro
holds one routing domain); checkpoints (``restart_interval_s > 0``: the
routing state is not checkpointed), refused before the first step as #235
does for the external ocean; Noah with urban physics (noah_column_hydro has
no urban hand-over); the NextGen coupling at the same time.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

ENV_NAMELIST = "WOOF_WRF_HYDRO_NAMELIST"
ENV_OUTPUT = "WOOF_WRF_HYDRO_OUTPUT"
TABLE = "wrf_hydro"
TABLE_KEYS = frozenset({"namelist", "output_dir"})
#: LSMs with ported WRF_HYDRO hooks: sf_surface_physics -> exchange planes
LSM_PLANES = {2: ("sfcheadrt", "infxsrt", "soldrain"),
              4: ("sfcheadrt", "infxsrt", "soldrain", "qtiledrain")}


def add_arguments(parser) -> None:
    parser.add_argument(
        "--wrf-hydro", default=None, metavar="HYDRO.NAMELIST",
        help="couple WRF-Hydro's routing (the opt-in under optional/wrf_hydro, "
             "WOOF_ENABLE_WRF_HYDRO=1) to the LSM as WRF built with WRF_HYDRO does, "
             "from this hydro.namelist; distinct from --hydro (NextGen)")
    parser.add_argument(
        "--wrf-hydro-output", default=None, metavar="DIR",
        help="where --wrf-hydro writes frxst_pts_out.txt and its CHRTOUT/RTOUT files "
             "(default: wrf_hydro_output in the working directory)")


def read_table(path) -> dict | None:
    """The ``[wrf_hydro]`` table of a run TOML, paths anchored at the TOML,
    or None when there is none.  Unknown keys are refused."""
    import tomllib

    path = Path(path)
    try:
        raw = tomllib.loads(path.read_text())
    except (OSError, tomllib.TOMLDecodeError):
        return None
    table = raw.get(TABLE)
    if table is None:
        return None
    if not isinstance(table, dict):
        raise ValueError(f"[{TABLE}] in {path} must be a table")
    unknown = sorted(set(table) - TABLE_KEYS)
    if unknown:
        raise ValueError(f"unknown [{TABLE}] key(s) {unknown} in {path}; known: "
                         f"{sorted(TABLE_KEYS)}")
    if not isinstance(table.get("namelist"), str) or not table["namelist"]:
        raise ValueError(f"[{TABLE}] in {path} needs namelist = \"hydro.namelist\"")
    out = {}
    for key in ("namelist", "output_dir"):
        if key in table:
            value = Path(table[key])
            out[key] = str(value if value.is_absolute() else (path.parent / value).resolve())
    return out


def consume_table(raw: dict, source) -> dict:
    """A config loader's check of ``[wrf_hydro]``: the launch (``launch_scope``)
    reads the table; a door that loads the file without that launch refuses
    it rather than run uncoupled under its name.  Returns ``raw`` without it."""
    if TABLE not in raw:
        return raw
    table = raw[TABLE]
    if not isinstance(table, dict) or sorted(set(table) - TABLE_KEYS) or not isinstance(
            table.get("namelist"), str):
        raise ValueError(f"[{TABLE}] in {source} takes namelist = \"hydro.namelist\" and "
                         f"optionally output_dir; got {table!r}")
    if configured() is None:
        raise ValueError(
            f"{source} carries [{TABLE}], which `woof run` and `woof go` read at launch "
            f"(it sets {ENV_NAMELIST}); this door loads the file without that launch and "
            "would run uncoupled under the table's name")
    return {k: v for k, v in raw.items() if k != TABLE}


@contextmanager
def launch_scope(namelist=None, output=None, config=None, members=None):
    """Set the environment a coupled launch reads, for its duration.  The
    command line wins over a ``[wrf_hydro]`` table in ``config``.  An ensemble
    (``members`` > 1) is refused: every member would write the one output
    directory."""
    table = read_table(config) if config else None
    if namelist is None and table is not None:
        namelist, output = table["namelist"], output or table.get("output_dir")
    if namelist is None:
        yield
        return
    if members is not None and int(members) > 1:
        raise NotImplementedError(
            f"--wrf-hydro with {int(members)} ensemble members: the members would share one "
            "WRF-Hydro output directory; run the coupled forecast as a single member")
    saved = {k: os.environ.get(k) for k in (ENV_NAMELIST, ENV_OUTPUT)}
    os.environ[ENV_NAMELIST] = str(Path(namelist).resolve())
    if output is not None:
        os.environ[ENV_OUTPUT] = str(Path(output).resolve())
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def configured() -> str | None:
    """The hydro.namelist this launch couples, or None (the default)."""
    return os.environ.get(ENV_NAMELIST) or None


def refuse_multiple_domains(count: int) -> None:
    if configured() and int(count) > 1:
        raise NotImplementedError(
            f"--wrf-hydro with {count} domains: WRF calls wrf_drv_HYDRO only on a domain "
            "without nests (num_nests < 1) and WRF-Hydro holds one routing domain; run "
            "the coupled forecast on a single domain")


def soil_centres(dzs):
    """WRF's ZS from the layer thicknesses as real.exe forms it
    (share/module_soil_pre.F :1145-1149, binary32)."""
    import numpy as np

    f = np.float32
    dzs = np.asarray(dzs, f)
    zs = np.empty_like(dzs)
    zs[0] = f(f(0.5) * dzs[0])
    for k in range(1, len(dzs)):
        zs[k] = f(f(zs[k - 1] + f(f(0.5) * dzs[k - 1])) + f(f(0.5) * dzs[k]))
    return zs


class WrfHydroSeam:
    """The exchange planes and the call after the land surface."""

    def __init__(self, routing, shape, lsm):
        import cupy as cp

        self.routing = routing
        self.lsm = int(lsm)
        self.planes = {name: cp.zeros(shape, cp.float32) for name in LSM_PLANES[self.lsm]}

    def lsm_kwargs(self) -> dict:
        """The keyword the LSM call takes: its WRF_HYDRO planes."""
        return {"wrf_hydro": self.planes}

    def exchange(self, fields) -> None:
        """wrf_drv_HYDRO after the surface driver: the routing reads SMOIS, SH2O,
        INFXSRT and SOLDRAIN and writes SMOIS, SH2O and SFCHEADRT."""
        p = self.planes
        self.routing.exchange(smois=fields["smois"], sh2o=fields["sh2o"],
                              infxsrt=p["infxsrt"], soldrain=p["soldrain"],
                              sfcheadrt=p["sfcheadrt"])

    def history_fields(self) -> dict:
        """WRF's WRF_HYDRO history rows on the mass grid (Registry: SFCHEADRT,
        INFXSRT, SOLDRAIN)."""
        p = self.planes
        return {"SFCHEADRT": p["sfcheadrt"], "INFXSRT": p["infxsrt"],
                "SOLDRAIN": p["soldrain"]}

    def digest_arrays(self) -> dict:
        out = {f"wrf_hydro/lsm/{k}": v for k, v in self.planes.items()}
        out.update(self.routing.digest_arrays())
        return out


def attach(driver, state, cfg, *, start_time=None, dzs=None) -> WrfHydroSeam | None:
    """Attach the coupled routing to ``driver`` when this launch asked for it.
    Returns None, touching nothing, otherwise."""
    namelist = configured()
    if namelist is None:
        return None
    if getattr(state, "hex_columns", False):
        # Hex columns (woof.core.mpas_column_batch) are not the routing domain's mass grid: the hex host
        # couples them through the conservative cell <-> grid exchange (woof.hex.wrf_hydro_hex.attach_host).
        return None
    lsm = int(getattr(cfg, "sf_surface_physics", 0))
    if lsm not in LSM_PLANES:
        raise NotImplementedError(
            f"--wrf-hydro with sf_surface_physics = {lsm}: only Noah (2) and Noah-MP (4) "
            "carry the ported WRF_HYDRO hooks")
    if int(getattr(cfg, "grid_id", 1)) != 1:
        raise NotImplementedError("--wrf-hydro on a nest: WRF couples one domain without "
                                  "nests (refuse_multiple_domains)")
    if float(getattr(cfg, "restart_interval_s", 0.0) or 0.0) > 0.0:
        raise NotImplementedError(
            "restart checkpoints of a run coupled to WRF-Hydro are not ported (the routing "
            "state -- soil and surface heads on the routing grid, channel, lakes and "
            "buckets -- is not checkpointed): set restart_interval_s = 0 in [experiment]")
    if lsm == 2 and int(getattr(cfg, "sf_surface_mosaic", 0) or 0) == 1:
        raise NotImplementedError("--wrf-hydro with Noah and sf_surface_mosaic = 1: the mosaic "
                                  "path (_run_noah_mosaic) has no WRF_HYDRO exchange")
    if bool(getattr(cfg, "use_adaptive_time_step", False)):
        raise NotImplementedError("--wrf-hydro with use_adaptive_time_step: HYDRO_dt and the "
                                  "snapped DTRT_TER/DTRT_CH are fixed at attachment, and the "
                                  "routing does not follow a changing LSM step")
    if lsm == 2 and int(getattr(cfg, "sf_urban_physics", 0) or 0) > 0:
        raise NotImplementedError("--wrf-hydro with Noah and sf_urban_physics > 0: "
                                  "noah_column_hydro has no urban hand-over")
    if getattr(driver, "hydro", None) is not None:
        raise NotImplementedError("--wrf-hydro together with --hydro (NextGen): two "
                                  "routings of the same runoff are not combined")
    if start_time is None:
        raise ValueError("--wrf-hydro needs the forecast start time for WRF-Hydro's output "
                         "clock")
    if dzs is None:
        raise ValueError("--wrf-hydro needs the LSM's soil layer thicknesses")
    from woof import optin

    module = optin.load("wrf_hydro")
    from importlib import import_module

    coupled = import_module(module.__name__ + ".coupled")
    fields = driver.fields
    shape = tuple(fields["smois"].shape[1:])
    ivgtyp, isltyp = fields.get("ivgtyp"), fields.get("isltyp")
    if ivgtyp is None or isltyp is None:
        raise NotImplementedError("--wrf-hydro needs IVGTYP and ISLTYP on the mass grid")
    routing = coupled.WrfHydroCoupled(
        namelist, ix=shape[1], jx=shape[0], zs=soil_centres(dzs), vegtyp=ivgtyp.get(),
        soltyp=isltyp.get(), hydro_dt=float(driver.bldt_seconds), start=start_time,
        output_dir=os.environ.get(ENV_OUTPUT) or str(Path.cwd() / "wrf_hydro_output"))
    seam = WrfHydroSeam(routing, shape, lsm)
    driver.wrf_hydro = seam
    return seam
