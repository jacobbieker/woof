"""MPAS-Ocean v8.4.2: the split-explicit step's barotropic stage, on the GPU.

:func:`barotropic_stage` is stage 2 of ``ocn_time_integrator_split``
(``mpas_ocn_time_integration_split.F``) for one block, in the Fortran's order
and halo extents:

- ``config_time_integrator = 'unsplit_explicit'``: the barotropic velocity
  zeroed, the new velocity the baroclinic one, the transport velocity
  masked;
- ``'split_explicit'``: the barotropic velocity and SSH subcycled
  ``nBtrSubcycles * config_btr_subcycle_loop_factor`` times (the velocity
  predictor, skipped for ``config_btr_gam1_velWt1 <= 1e-12``; the SSH
  predictor; ``config_n_btr_cor_iter`` velocity correctors; the SSH corrector
  under ``config_btr_solve_SSH2``), the thickness flux and velocity sums
  normalised, then the transport velocity with the velocity correction
  (``config_vel_correction``).

The halo counters are the Fortran's: each subcycle computes over the
``nCellsArray``/``nEdgesArray`` extent its counter names, and a halo exchange
is the identity (one block). The kernels (``btr_subcycle_v842.cu``) are held
word for word to the stage, cut verbatim out of v8.4.2 and compiled with
gfortran ``-O0 -ffp-contract=off`` at RKIND 4 and 8
(``tools/mpas_ocean_oracle/run_btr_subcycle.F90``,
``tests/test_mpas_ocean_btr_subcycle_parity.py``).

v8.4.2's own behaviours kept as written: the predictor-skipped subcycle reads
whatever the new subcycle level holds; the SSH corrector reads the new SSH of
the cells before it in the loop (run in the Fortran's order on one thread).

Refused by name: a configuration whose halo counters index ``nCellsArray``
or ``nEdgesArray`` outside ``1..config_num_halos+1`` (v8.4.2 reads outside
the array: ``config_btr_solve_SSH2`` with ``config_n_btr_cor_iter >= 1``,
for one), and an ``nEdgesArray`` whose corrector copy would leave an
``edgesOnEdge`` neighbour uncopied (the Fortran reads an uninitialised
scratch word).
"""
from __future__ import annotations

from functools import lru_cache
from importlib.resources import files

import numpy as np

from woof.hex import numerics as _hex_numerics
from woof.hex.errors import ConfigurationRefusal

from .tend_v842 import COMPILE_OPTIONS, PRECISIONS, _precision_of

SOURCE_FILES = ("src/core_ocean/mode_forward/mpas_ocn_time_integration_split.F:84-223,227-259,638-1384",)
MODULE_OPTIONS = COMPILE_OPTIONS
GRAVITY = 9.80616
_BLOCK = 128


def module_source(precision: str = "r4") -> str:
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {sorted(PRECISIONS)}")
    body = files("woof.hex.mpas_ocean").joinpath("btr_subcycle_v842.cu").read_text(encoding="utf-8")
    return ("#define MPAS_RKIND8 1\n" if precision == "r8" else "") + body


@lru_cache(maxsize=None)
def _module_on(device: int, precision: str, options: tuple[str, ...]):
    import cupy as cp
    from woof.nvrtc_ptx_cache import compile_using_nvrtc

    image, _ = compile_using_nvrtc(module_source(precision), options, None,
                                   f"ocn_btr_subcycle_v842_{precision}.cu")
    with cp.cuda.Device(device):
        module = cp.cuda.function.Module()
        module.load(image.encode() if isinstance(image, str) else image)
    return module


def _launch(precision: str, name: str, n: int, args: tuple, threads: int | None = None) -> None:
    import cupy as cp
    if n > 0:
        options = _hex_numerics.options(COMPILE_OPTIONS, keep_ftz=True)
        block = _BLOCK if threads is None else threads
        _module_on(cp.cuda.Device().id, precision, options).get_function(name)(
            ((n + block - 1) // block,), (block,), args)


def halo_schedule(configs: dict, n_btr_subcycles: int) -> list[dict]:
    """The extents (as ``nCellsArray``/``nEdgesArray`` indices) each subcycle's
    steps use, as the Fortran's counters walk them; refuses by name an index
    outside ``1..config_num_halos+1``."""
    num_halos = int(configs["config_num_halos"])
    n_cor = int(configs["config_n_btr_cor_iter"])
    ssh2 = bool(configs["config_btr_solve_SSH2"])
    predictor = float(configs["config_btr_gam1_velWt1"]) > 1.0e-12
    needed = 1 + n_cor
    cell = edge = 0
    schedule = []

    def use(index: int) -> int:
        if not 1 <= index <= num_halos + 1:
            knob = "config_btr_solve_SSH2" if ssh2 else "config_n_btr_cor_iter"
            raise ConfigurationRefusal(
                knob=knob, value=configs[knob],
                reason=(f"v8.4.2's barotropic halo counters index nCellsArray/nEdgesArray({index}), outside "
                        f"1..{num_halos + 1} (config_num_halos = {num_halos}, config_n_btr_cor_iter = {n_cor}, "
                        f"config_btr_solve_SSH2 = {ssh2}, {n_btr_subcycles} subcycles)"),
                declaration=("config_btr_solve_SSH2 = .false." if ssh2 else
                             "config_n_btr_cor_iter <= config_num_halos - 1"))
        return index

    for _ in range(n_btr_subcycles * int(configs["config_btr_subcycle_loop_factor"])):
        if cell < needed:
            cell = num_halos - num_halos % needed
            edge = num_halos + 1 - num_halos % needed
        step = {"predictor": use(edge) if predictor else None}
        edge -= 1
        step["ssh_cells"], step["ssh_edges"] = use(cell), use(edge)
        cell -= 1
        step["correctors"] = []
        for _ in range(n_cor):
            step["correctors"].append((use(min(edge + 1, num_halos + 1)), use(edge)))
            edge -= 1
            cell -= 1
        if ssh2:
            step["ssh2_cells"], step["ssh2_edges"] = use(cell), use(edge)
            edge = num_halos + 1
        schedule.append(step)
    return schedule


def check_config(configs: dict, n_btr_subcycles: int) -> None:
    """Refuse, by name, what v8.4.2 leaves undefined."""
    if configs["config_time_integrator"] == "split_explicit":
        halo_schedule(configs, n_btr_subcycles)


def _check_copies(mesh: dict, schedule: list[dict]) -> None:
    """Every edgesOnEdge neighbour a corrector reads lies in the extent it copied."""
    import cupy as cp
    edges = [int(x) for x in mesh["nEdgesArray"]]
    n_eoe = cp.asnumpy(mesh["nEdgesOnEdge"])
    eoe = cp.asnumpy(mesh["edgesOnEdge"])
    for step in schedule:
        for copied, computed in step["correctors"]:
            n = edges[computed - 1]
            limit = edges[copied - 1] + 1
            cols = np.arange(eoe.shape[1])[None, :] < n_eoe[:n, None]
            if (eoe[:n][cols] > limit).any():
                raise ConfigurationRefusal(
                    knob="nEdgesArray", value=edges,
                    reason="a barotropic corrector reads an edgesOnEdge neighbour outside the extent it copied "
                           "(v8.4.2 reads an uninitialised scratch word)",
                    declaration="each halo layer's edgesOnEdge within the next layer")


def barotropic_stage(*, dt: float, n_btr_subcycles: int, mesh: dict, state: dict, diagnostics: dict,
                     tend: dict, configs: dict) -> None:
    """Stage 2 of ``ocn_time_integrator_split`` for one block, in place.

    ``state`` holds the time levels as lists: ``normalBarotropicVelocity``,
    ``sshSubcycle`` and ``normalBarotropicVelocitySubcycle`` ``[level1,
    level2]``; ``ssh`` ``[level1]``; ``normalVelocity`` and
    ``normalBaroclinicVelocity`` ``[level1, level2]`` (level 2 is used).
    ``diagnostics`` holds ``barotropicForcing``, ``barotropicThicknessFlux``,
    ``normalTransportVelocity``, ``normalGMBolusVelocity`` and
    ``layerThicknessEdge``; ``tend`` holds ``ssh``."""
    import cupy as cp
    integrator = configs["config_time_integrator"]
    m, d = mesh, diagnostics
    btr = state["normalBarotropicVelocity"]
    precision = _precision_of(btr[0])
    real = PRECISIONS[precision]
    i32 = np.int32
    nl = int(m["nVertLevels"])
    n_cells, n_edges = int(m["nCells"]), int(m["nEdges"])
    cells = [int(x) for x in m["nCellsArray"]]
    edges = [int(x) for x in m["nEdgesArray"]]
    bcl_new = state["normalBaroclinicVelocity"][1]
    bolus = d["normalGMBolusVelocity"]
    transport = d["normalTransportVelocity"]
    edge_mask = m["edgeMask"]

    if integrator == "unsplit_explicit":
        _launch(precision, "ocn_btr_unsplit_v842", n_edges, (
            btr[1], state["normalVelocity"][1], transport, bcl_new, bolus, edge_mask, i32(n_edges), i32(nl)))
        return
    if integrator != "split_explicit":
        return
    schedule = halo_schedule(configs, n_btr_subcycles)
    _check_copies(m, schedule)

    ssh_sub = state["sshSubcycle"]
    u_sub = state["normalBarotropicVelocitySubcycle"]
    forcing, flux = d["barotropicForcing"], d["barotropicThicknessFlux"]
    ssh_tend = tend["ssh"]
    dtn = real(real(dt) / real(n_btr_subcycles))
    gravity = real(GRAVITY)
    g1, g2, g3 = (real(configs[k]) for k in ("config_btr_gam1_velWt1", "config_btr_gam2_SSHWt1",
                                              "config_btr_gam3_velWt2"))
    omg1, omg2, omg3 = real(real(1.0) - g1), real(real(1) - g2), real(real(1.0) - g3)
    ssh2 = bool(configs["config_btr_solve_SSH2"])
    me = int(m["edgesOnCell"].shape[1])
    me2 = int(m["edgesOnEdge"].shape[1])

    old, new = 0, 1
    _launch(precision, "ocn_btr_init_cells_v842", n_cells, (ssh_sub[old], state["ssh"][0], i32(n_cells)))
    _launch(precision, "ocn_btr_init_edges_v842", n_edges, (
        u_sub[old], btr[1], flux, forcing, btr[0], i32(n_edges), i32(bool(configs["config_filter_btr_mode"]))))
    temp = cp.full(n_edges + 1, np.finfo(real).max, dtype=real)

    def velocity(n: int, u_perp, corrector: int) -> None:
        _launch(precision, "ocn_btr_velocity_v842", n, (
            u_sub[new], u_sub[old], u_perp, ssh_sub[old], ssh_sub[new], forcing, m["fEdge"], m["dcEdge"],
            m["weightsOnEdge"], m["edgesOnEdge"], m["nEdgesOnEdge"], m["cellsOnEdge"], edge_mask, i32(n), i32(nl),
            i32(me2), i32(corrector), dtn, gravity, omg2, g2))

    def ssh_args(n_c: int):
        return (ssh_tend, ssh_sub[new], ssh_sub[old], u_sub[old], u_sub[new], m["bottomDepth"], m["dvEdge"],
                m["areaCell"], m["nEdgesOnCell"], m["edgesOnCell"], m["edgeSignOnCell"], m["cellsOnEdge"],
                i32(n_c), i32(me))

    def flux_sum(n_e: int, weighted: int, omgv, gv) -> None:
        _launch(precision, "ocn_btr_flux_sum_v842", n_e, (
            flux, u_sub[old], u_sub[new], ssh_sub[old], ssh_sub[new], m["bottomDepth"], m["cellsOnEdge"],
            i32(n_e), i32(weighted), omg2, g2, omgv, gv))

    for step in schedule:
        if step["predictor"] is not None:
            velocity(edges[step["predictor"] - 1], u_sub[old], 0)
        n_c = cells[step["ssh_cells"] - 1]
        _launch(precision, "ocn_btr_ssh_predict_v842", n_c, ssh_args(n_c) + (g1, omg1, dtn))
        if not ssh2:
            flux_sum(edges[step["ssh_edges"] - 1], 0, omg1, g1)
        for copied, computed in step["correctors"]:
            n_copy = edges[copied - 1] + 1
            _launch(precision, "ocn_btr_copy_v842", n_copy, (temp, u_sub[new], i32(n_copy)))
            velocity(edges[computed - 1], temp, 1)
        if ssh2:
            n_c = cells[step["ssh2_cells"] - 1]
            _launch(precision, "ocn_btr_ssh_correct_v842", 1, ssh_args(n_c) + (omg2, g2, omg3, g3, dtn), threads=1)
            flux_sum(edges[step["ssh2_edges"] - 1], 1, omg3, g3)
        _launch(precision, "ocn_btr_accumulate_v842", n_edges, (btr[1], u_sub[new], i32(n_edges)))
        old, new = new, old

    n_sum = n_btr_subcycles * int(configs["config_btr_subcycle_loop_factor"])
    _launch(precision, "ocn_btr_normalize_v842", edges[0], (flux, btr[1], i32(edges[0]), i32(n_sum)))
    n_h = edges[int(configs["config_num_halos"]) - 1]
    _launch(precision, "ocn_btr_transport_v842", n_h, (
        transport, btr[1], bcl_new, bolus, d["layerThicknessEdge"], flux, m["maxLevelEdgeTop"], edge_mask, i32(n_h),
        i32(nl), i32(1 if configs["config_vel_correction"] else 0)))
